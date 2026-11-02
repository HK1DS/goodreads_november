#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tiny synthetic Goodreads-format dataset for offline smoke tests.

Writes (in --out_dir):
  goodreads_interactions_children.json.gz   raw interactions (Goodreads date strings)
  goodreads_books_children.json.gz          books (title/description/authors/publisher/shelves)
  goodreads_book_authors.json.gz            author names

Users have a latent topic preference so the models have signal to learn.
Timestamps deliberately include ties, empty read_at (date_added fallback),
missing dates and out-of-range years, to exercise the RecBole split replica.
"""
import argparse
import gzip
import json
import os
import random
from datetime import datetime, timedelta, timezone

TOPICS = ["dragons", "animals", "space", "princess", "dinosaurs", "pirates", "school", "friendship"]


def fmt(dt):
    return dt.strftime("%a %b %d %H:%M:%S %z %Y")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--users", type=int, default=400)
    ap.add_argument("--items", type=int, default=250)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    authors = [{"author_id": str(1000 + a), "name": f"Author {a}"} for a in range(60)]
    books = []
    for i in range(args.items):
        topic = TOPICS[i % len(TOPICS)]
        shelves = [{"count": str(rng.randint(500, 900)), "name": "to-read"},
                   {"count": str(rng.randint(300, 600)), "name": "children"},
                   {"count": str(rng.randint(50, 300)), "name": topic},
                   {"count": str(rng.randint(5, 80)), "name": rng.choice(["picture-books", "chapter-books", "series"])}]
        books.append({
            "book_id": str(50000 + i),
            "title": f"The {topic.title()} Book {i}",
            "description": f"A children's story about {topic} and adventure number {i}.",
            "authors": [{"author_id": authors[i % len(authors)]["author_id"], "role": ""}],
            "publisher": f"Pub {i % 12}",
            "popular_shelves": shelves,
        })
    pop = [1.0 / (1 + (i % 97)) ** 0.7 for i in range(args.items)]

    base = datetime(2009, 1, 1, tzinfo=timezone(timedelta(hours=-8)))
    rows = []
    for u in range(args.users):
        uid = f"u{u:05d}"
        fav = rng.sample(TOPICS, 2)
        n = rng.randint(35, 90)
        start = base + timedelta(days=rng.randint(0, 2000))
        chosen = set()
        for k in range(n):
            # 70% from favourite topics, rest popularity-weighted
            if rng.random() < 0.7:
                t = rng.choice(fav)
                cand = [i for i in range(args.items) if TOPICS[i % len(TOPICS)] == t]
                i = rng.choice(cand)
            else:
                i = rng.choices(range(args.items), weights=pop)[0]
            if i in chosen:
                continue
            chosen.add(i)
            day = start + timedelta(days=k * rng.randint(1, 9))
            # ties: several books stamped at the same midnight
            if rng.random() < 0.2:
                day = start + timedelta(days=(k // 3) * 5)
            r = {"user_id": uid, "book_id": books[i]["book_id"], "rating": rng.choice([0, 3, 4, 5]),
                 "date_added": fmt(day), "read_at": fmt(day + timedelta(days=1))}
            roll = rng.random()
            if roll < 0.10:
                r["read_at"] = ""                       # date_added fallback
            elif roll < 0.12:
                r["read_at"] = ""; r["date_added"] = ""  # missing -> clipped floor
            elif roll < 0.13:
                r["read_at"] = fmt(day.replace(year=2031))   # future garbage -> clipped ceil
            rows.append(r)
    rng.shuffle(rows)  # raw file order is NOT chronological

    with gzip.open(os.path.join(args.out_dir, "goodreads_interactions_children.json.gz"), "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with gzip.open(os.path.join(args.out_dir, "goodreads_books_children.json.gz"), "wt", encoding="utf-8") as f:
        for b in books:
            f.write(json.dumps(b) + "\n")
    with gzip.open(os.path.join(args.out_dir, "goodreads_book_authors.json.gz"), "wt", encoding="utf-8") as f:
        for a in authors:
            f.write(json.dumps(a) + "\n")
    print(f"[synthetic] users={args.users} items={args.items} interactions={len(rows)} -> {args.out_dir}")


if __name__ == "__main__":
    main()
