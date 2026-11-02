#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build LEAKAGE-FREE user profiles (+ static item profiles) for one split protocol.

Legacy behaviour (goodreads_preprocess.py -> apply_k_core.py) built each user's
profile from the first 20 interactions in raw-file order, regardless of time or
split, truncated to 2,000 characters (= in practice 1-2 books). Those books could
be validation/test items, so user intents -- and every KG/retrieval channel that
uses them -- could see the future. Under TO_GLOBAL, users with ZERO training
interactions still got a profile (built from their test books).

This script fixes that. For a given split (TO or TO_GLOBAL) it
  1. reads the exact RecBole atomic .inter file used for evaluation,
  2. reproduces RecBole's temporal split (ikgr_core/temporal_split.py),
  3. builds each user's profile ONLY from profile-safe training rows
     (train rows strictly earlier than the user's/global first non-train row),
  4. writes the usual cover CSV (user_id,user_profile,item_id,item_profile)
     consumed by step1/step2, plus a provenance file listing every
     (user_id, item_id) that contributed text. eval_slices.py verifies the
     provenance against RecBole's real train split and aborts on any violation.

Users without safe history get an EMPTY profile -> no LLM call, no intents.

Item profiles are static catalog metadata (title | authors | description |
shelves, first 2,000 chars), identical to the legacy item profiles, so item
intents are split-independent and their LLM cache can be shared.

Profile styles
  structured (default): recent N titles (newest first, with author names when
      --authors_gz is given) + frequent authors + frequent content shelves,
      aggregated over the user's most recent --agg_n safe books.
  legacy: the old rule (first --max_items safe books in file order, full item
      profiles concatenated, truncated) -- only the time restriction changes.

Usage
  python build_trainonly_profiles.py --split TO \
      --inter data/kc_k30/ikgr-custom/ikgr-custom.inter \
      --books_gz goodreads_books_children.json.gz \
      --out_profiles data/kc_k30/profiles_k30_TO.csv \
      --out_provenance data/kc_k30/profile_provenance_k30_TO.csv.gz
"""
import argparse
import gzip
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

from goodreads_preprocess import build_item_profile, _clean_text
from ikgr_core.temporal_split import SPLITS, assign_phases, read_atomic_inter

# reading-status / non-content shelves (same list as build_meta_kg.py)
SHELF_STOP = {
    "to-read", "currently-reading", "read", "owned", "owned-books", "books-i-own",
    "favorites", "favourites", "default", "to-buy", "wish-list", "wishlist",
    "my-books", "my-library", "library", "re-read", "reread", "dnf", "abandoned",
    "have-read", "finished", "unfinished", "tbr", "to-read-owned", "kindle",
    "ebook", "ebooks", "audiobook", "audiobooks", "audio", "audible", "owned-tbr",
    "i-own", "want-to-read", "books", "book", "all-books", "general", "unread",
    "to-read-fiction", "maybe", "to-read-maybe", "not-read", "did-not-finish",
}


def load_books(books_gz, item_set, item_max_text_len, top_shelves=10):
    """Return {item: (title, [author_ids], [content shelves], item_profile)}."""
    out = {}
    with gzip.open(books_gz, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            b = json.loads(line)
            bid = str(b.get("book_id") or b.get("id") or "")
            if bid not in item_set:
                continue
            title = _clean_text(b.get("title", "")) or _clean_text(b.get("title_without_series", ""))
            authors = [str(a.get("author_id")) for a in (b.get("authors") or []) if a.get("author_id")]
            shelves = []
            for s in sorted(b.get("popular_shelves") or [],
                            key=lambda s: -int(s.get("count", 0) or 0)):
                name = (s.get("name") or "").strip().lower()
                if name and name not in SHELF_STOP:
                    shelves.append(name)
                if len(shelves) >= top_shelves:
                    break
            prof = build_item_profile(b)
            if item_max_text_len > 0:
                prof = prof[:item_max_text_len]
            out[bid] = (title, authors, shelves, prof)
    return out


def load_author_names(authors_gz, needed):
    names = {}
    if not authors_gz or not os.path.exists(authors_gz):
        return names
    with gzip.open(authors_gz, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            a = json.loads(line)
            aid = str(a.get("author_id", ""))
            if aid in needed and a.get("name"):
                names[aid] = _clean_text(a["name"])
    return names


def structured_profile(items_newest_first, books, author_names, ubiquitous,
                       recent_n, agg_n, max_len):
    """Return (text, contributing_item_ids)."""
    agg = items_newest_first[:agg_n]
    if not agg:
        return "", []
    recent = []
    for iid in agg[:recent_n]:
        title, authors, _, _ = books.get(iid, ("", [], [], ""))
        if not title:
            continue
        names = [author_names[a] for a in authors if a in author_names]
        recent.append(f"{title} by {names[0]}" if names else title)
    a_cnt, s_cnt = Counter(), Counter()
    for iid in agg:
        _, authors, shelves, _ = books.get(iid, ("", [], [], ""))
        for a in authors[:1]:                      # primary author only
            if a in author_names:
                a_cnt[author_names[a]] += 1
        for s in shelves:
            if s not in ubiquitous:
                s_cnt[s] += 1
    parts = []
    if recent:
        parts.append("Recently read (newest first): " + "; ".join(recent))
    fav = [f"{n} ({c})" for n, c in a_cnt.most_common(8) if c >= 2]
    if fav:
        parts.append("Frequent authors: " + ", ".join(fav))
    shelves = [s for s, _ in s_cnt.most_common(15)]
    if shelves:
        parts.append("Common shelves: " + ", ".join(shelves))
    text = " | ".join(parts)
    if max_len > 0:
        text = text[:max_len]
    return text, list(agg)


def legacy_profile(items_file_order, books, max_items, max_len):
    texts, used = [], []
    length = 0
    for iid in items_file_order:
        if len(used) >= max_items:
            break
        prof = books.get(iid, ("", [], [], ""))[3]
        if not prof:
            continue
        sep = 3 if texts else 0                      # " ; "
        if max_len > 0 and length >= max_len:
            break
        texts.append(prof)
        used.append(iid)
        length += sep + len(prof)
    text = " ; ".join(texts)
    if max_len > 0:
        text = text[:max_len]
    return text, used


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=sorted(SPLITS))
    ap.add_argument("--inter", required=True, help="RecBole atomic .inter with timestamp (evaluation source of truth)")
    ap.add_argument("--books_gz", default="goodreads_books_children.json.gz")
    ap.add_argument("--authors_gz", default="goodreads_book_authors.json.gz",
                    help="optional; adds author names to structured profiles")
    ap.add_argument("--style", default="structured", choices=["structured", "legacy"])
    ap.add_argument("--recent_n", type=int, default=10)
    ap.add_argument("--agg_n", type=int, default=50)
    ap.add_argument("--max_items", type=int, default=20, help="legacy style only")
    ap.add_argument("--max_text_len", type=int, default=2000)
    ap.add_argument("--item_max_text_len", type=int, default=2000)
    ap.add_argument("--ubiquitous_df", type=float, default=0.5,
                    help="drop shelves present on more than this fraction of items (structured)")
    ap.add_argument("--out_profiles", required=True)
    ap.add_argument("--out_provenance", required=True)
    args = ap.parse_args()

    df = read_atomic_inter(args.inter)
    phase, safe = assign_phases(df, args.split)
    df["phase"] = phase
    df["safe"] = safe
    users = sorted(str(u) for u in df["user_id"].unique())
    items = sorted(str(i) for i in df["item_id"].unique())
    print(f"[profiles] {args.split}: rows={len(df)} users={len(users)} items={len(items)} "
          f"train={int((phase == 0).sum())} valid={int((phase == 1).sum())} test={int((phase == 2).sum())} "
          f"safe_train={int(safe.sum())}", flush=True)

    books = load_books(args.books_gz, set(items), args.item_max_text_len)
    print(f"[profiles] matched {len(books)}/{len(items)} items in books file", flush=True)
    needed_authors = {a for _, au, _, _ in books.values() for a in au}
    author_names = load_author_names(args.authors_gz, needed_authors) if args.style == "structured" else {}
    if args.style == "structured":
        print(f"[profiles] author names: {len(author_names)}/{len(needed_authors)}"
              + ("" if author_names else " (no --authors_gz; titles only)"), flush=True)

    shelf_df = Counter()
    for _, _, shelves, _ in books.values():
        shelf_df.update(set(shelves))
    ubiquitous = {s for s, c in shelf_df.items() if c > args.ubiquitous_df * max(1, len(books))}
    if ubiquitous:
        print(f"[profiles] ubiquitous shelves dropped from user profiles: {sorted(ubiquitous)}", flush=True)

    tr = df[df["safe"]].copy()
    tr["_row"] = np.arange(len(tr))
    user_profile, prov_rows, n_items_used = {}, [], []
    if args.style == "structured":
        tr = tr.sort_values(["user_id", "timestamp", "_row"], ascending=[True, False, False], kind="mergesort")
    for uid, g in tr.groupby("user_id", sort=False, observed=True):
        uid = str(uid)
        seq = [str(x) for x in g["item_id"].tolist()]
        if args.style == "structured":
            text, used = structured_profile(seq, books, author_names, ubiquitous,
                                            args.recent_n, args.agg_n, args.max_text_len)
        else:
            text, used = legacy_profile(seq, books, args.max_items, args.max_text_len)
        if text:
            user_profile[uid] = text
            prov_rows.extend((uid, i) for i in used)
            n_items_used.append(len(used))

    n_empty = len(users) - len(user_profile)
    print(f"[profiles] users with profile={len(user_profile)} empty={n_empty} "
          f"avg contributing items={np.mean(n_items_used) if n_items_used else 0:.1f}", flush=True)

    # cover CSV: every user and every item appears at least once
    n = max(len(users), len(items))
    rows = []
    for k in range(n):
        u = users[k % len(users)]
        i = items[k % len(items)]
        rows.append((u, user_profile.get(u, ""), i, books.get(i, ("", [], [], ""))[3]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out_profiles)), exist_ok=True)
    pd.DataFrame(rows, columns=["user_id", "user_profile", "item_id", "item_profile"]).to_csv(
        args.out_profiles, index=False)

    prov = pd.DataFrame(prov_rows, columns=["user_id", "item_id"])
    prov.to_csv(args.out_provenance, index=False, compression="gzip")
    meta = {
        "split": args.split, "ratios": SPLITS[args.split][0], "group_by_user": SPLITS[args.split][1],
        "inter": os.path.abspath(args.inter), "style": args.style,
        "recent_n": args.recent_n, "agg_n": args.agg_n, "max_items": args.max_items,
        "max_text_len": args.max_text_len, "n_users": len(users), "n_items": len(items),
        "n_users_with_profile": len(user_profile), "n_users_empty_profile": n_empty,
        "n_provenance_pairs": len(prov),
        "phase_counts": {"train": int((phase == 0).sum()), "valid": int((phase == 1).sum()),
                         "test": int((phase == 2).sum())},
    }
    with open(args.out_provenance + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[profiles] saved -> {args.out_profiles}\n[profiles] provenance -> {args.out_provenance}", flush=True)


if __name__ == "__main__":
    main()
