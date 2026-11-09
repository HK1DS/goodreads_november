#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Which users gain or lose from the IKGR facets? (TODO #7, no training needed)

Joins the per-user test metrics written by eval_slices.py (per_user_{SPLIT}/*.csv.gz)
with user features computed from the TRAIN split only:

  activity        number of train interactions (same as activity_train in the dumps)
  n_intents       number of LLM intents attached to the user in kg_pack.pt
  intent_disp     intent dispersion = 1 - mean pairwise cosine of the user's intent embeddings
                  (low = focused / redundant intents, high = diverse); unlike 1-||mean|| it does not
                  grow mechanically with the number of intents
  shelf_entropy   Shannon entropy (bits) of the shelf (category) distribution over the user's
                  train books, ignoring shelves on more than --ubiquitous_df of all books
                  (e.g. 'childrens', 'fiction'), which carry no category information

For each feature, users are split into quintiles and the paired per-user difference
A - B is reported with a 95% bootstrap CI (users resampled). Metrics are averaged over
the seeds available for each model first (as in paired_compare.py). Because all three
diversity features grow with activity, an OLS of the per-user difference on the
standardised features (activity included) separates their effects.

  python analyze_user_segments.py --run_dir run_k30_TO --split TO \
      --inter data/kc_k30/ikgr-custom/ikgr-custom.inter --out run_k30_TO/user_segments_TO.md
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ikgr_core.temporal_split import assign_phases, read_atomic_inter  # noqa: E402

FEATURES = [
    ("activity", "train interactions"),
    ("n_intents", "LLM intent count"),
    ("intent_disp", "intent dispersion, 1 - mean pairwise cosine"),
    ("shelf_entropy", "shelf (category) entropy, bits"),
]


# ----------------------------------------------------------------------------- features
def user_features(inter_path, split, kg_pack, meta_kg_pack, ubiquitous_df):
    import torch
    df = read_atomic_inter(inter_path)
    phase, _ = assign_phases(df, split)
    tr = df.loc[phase == 0, ["user_id", "item_id"]]
    users = tr["user_id"].astype(str)
    items = tr["item_id"].astype(str)
    feats = pd.DataFrame({"activity": users.value_counts()})

    kg = torch.load(kg_pack, map_location="cpu", weights_only=False)
    emb = kg["intent_emb"].float()
    emb = emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-12)
    n_int, disp = {}, {}
    for u, ids in kg["user_intents"].items():
        ids = ids.long()
        n = int(ids.numel())
        n_int[str(u)] = n
        # mean pairwise cosine via ||sum||^2 = n + sum_{i!=j} cos_ij (no n-dependence, unlike 1-||mean||)
        disp[str(u)] = (float(1.0 - (emb[ids].sum(0).pow(2).sum() - n) / (n * (n - 1)))
                        if n >= 2 else np.nan)
    feats["n_intents"] = pd.Series(n_int)
    feats["intent_disp"] = pd.Series(disp)

    meta = torch.load(meta_kg_pack, map_location="cpu", weights_only=False)
    item_sh = {str(k): v.numpy() for k, v in meta["item_shelves"].items()}
    n_books = len(meta["item_authors"])
    dfreq = np.zeros(meta["n_shelves"], dtype=np.int64)
    for v in item_sh.values():
        dfreq[v] += 1
    keep = dfreq <= ubiquitous_df * n_books
    dropped = [meta["shelf_vocab"][i] for i in np.flatnonzero(~keep)]
    # user x shelf counts over train books = (user x item) @ (item x shelf), all sparse
    from scipy.sparse import csr_matrix
    u_codes, u_index = pd.factorize(users)
    i_codes, i_index = pd.factorize(items)
    UI = csr_matrix((np.ones(len(u_codes), dtype=np.float64), (u_codes, i_codes)),
                    shape=(len(u_index), len(i_index)))
    ir, ic = [], []
    for j, it in enumerate(i_index):
        sh = item_sh.get(it)
        if sh is not None:
            sh = sh[keep[sh]]
            ir.append(np.full(sh.size, j))
            ic.append(sh)
    ir = np.concatenate(ir)
    ic = np.concatenate(ic)
    IS = csr_matrix((np.ones(ir.size, dtype=np.float64), (ir, ic)), shape=(len(i_index), meta["n_shelves"]))
    M = (UI @ IS).tocsr()
    del UI, IS
    ent = np.zeros(len(u_index))
    for i in range(M.shape[0]):
        d = M.data[M.indptr[i]:M.indptr[i + 1]]
        if d.size:
            p = d / d.sum()
            ent[i] = float(-(p * np.log2(p)).sum())
    feats["shelf_entropy"] = pd.Series(ent, index=u_index.astype(str))
    return feats, dropped


# ----------------------------------------------------------------------------- metrics
def load_model(per_user_dir, name, seeds=None):
    """Per-user metrics of one result name, averaged over its seeds (or the given ones)."""
    paths = sorted(glob.glob(os.path.join(per_user_dir, f"{name}__s*__test.csv.gz")))
    frames = []
    for p in paths:
        seed = int(re.search(r"__s(\d+)__", os.path.basename(p)).group(1))
        if seeds and seed not in seeds:
            continue
        frames.append(pd.read_csv(p, dtype={"user_id": str}).assign(seed=seed))
    if not frames:
        raise SystemExit(f"no per-user files for {name} {seeds or ''} in {per_user_dir}")
    allf = pd.concat(frames)
    used = sorted(allf["seed"].unique())
    m = allf.groupby("user_id")[["ndcg@10", "tail_recall@10", "n_rel_tail"]].mean()
    return m, used, allf


def boot_ci(x, n_boot, rng):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return np.nan, np.nan
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt(v, d=4):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.{d}f}"


def bucket_table(df, feat, a_col, b_col, rng, n_boot, per_seed=None, q=5):
    """Quintile table of the paired difference a - b along one feature."""
    d = df[[feat, a_col, b_col]].dropna()
    labels = pd.qcut(d[feat].rank(method="first"), q, labels=[f"Q{i + 1}" for i in range(q)])
    lines = ["| bucket | feature range | users | A | B | A − B | 95% CI | win % | loss % |"
             + (" per-seed A − B |" if per_seed is not None else ""),
             "|---|---|---:|---:|---:|---:|---|---:|---:|" + ("---|" if per_seed is not None else "")]
    for lab in labels.cat.categories:
        g = d[labels == lab]
        diff = g[a_col] - g[b_col]
        lo, hi = boot_ci(diff, n_boot, rng)
        rng_txt = f"{g[feat].min():.3g} – {g[feat].max():.3g}"
        row = (f"| {lab} | {rng_txt} | {len(g):,} | {g[a_col].mean():.4f} | {g[b_col].mean():.4f} | "
               f"{fmt(diff.mean())} | [{fmt(lo)}, {fmt(hi)}] | {100 * (diff > 0).mean():.0f} | {100 * (diff < 0).mean():.0f} |")
        if per_seed is not None:
            row += " " + " / ".join(fmt(per_seed(g.index, s)) for s in per_seed.seeds) + " |"
        lines.append(row)
    return "\n".join(lines)


def within_activity_table(df, ycol, feats, rng, n_boot, q=5):
    """Inside each activity quintile, split users at the median of a feature and compare
    the mean A - B of the high half with the low half (activity held roughly fixed)."""
    d = df.dropna(subset=[ycol] + feats).copy()
    d["aq"] = pd.qcut(d["activity"].rank(method="first"), q, labels=[f"Q{i + 1}" for i in range(q)])
    head = "| activity bucket | users | " + " | ".join(f"{f}: low / high half" for f in feats) + " |"
    lines = [head, "|---|---:|" + "---|" * len(feats)]
    gaps = {f: [] for f in feats}
    for lab in d["aq"].cat.categories:
        g = d[d["aq"] == lab]
        cells = []
        for f in feats:
            hi = g[f] > g[f].median()
            lo_m, hi_m = g.loc[~hi, ycol].mean(), g.loc[hi, ycol].mean()
            gaps[f].append((g.loc[~hi, ycol].to_numpy(), g.loc[hi, ycol].to_numpy()))
            cells.append(f"{fmt(lo_m)} / {fmt(hi_m)}")
        lines.append(f"| {lab} | {len(g):,} | " + " | ".join(cells) + " |")
    # pooled within-bucket gap (high - low), CI by resampling users inside each half
    row = []
    for f in feats:
        est = np.mean([h.mean() - l.mean() for l, h in gaps[f]])
        boots = np.empty(n_boot)
        for b in range(n_boot):
            boots[b] = np.mean([h[rng.integers(0, h.size, h.size)].mean() - l[rng.integers(0, l.size, l.size)].mean()
                                for l, h in gaps[f]])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        row.append(f"**{fmt(est)}** [{fmt(lo)}, {fmt(hi)}]")
    lines.append("| high − low (mean over buckets, 95% CI) | | " + " | ".join(row) + " |")
    return "\n".join(lines)


def ols_table(df, ycol, feats, rng, n_boot):
    d = df[[ycol] + feats].dropna()
    X = d[feats].copy()
    X["activity"] = np.log1p(X["activity"])
    X = (X - X.mean()) / X.std(ddof=0)
    Xm = np.column_stack([np.ones(len(X)), X.to_numpy()])
    y = d[ycol].to_numpy()
    beta = np.linalg.lstsq(Xm, y, rcond=None)[0]
    boots = np.empty((n_boot, Xm.shape[1]))
    for b in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        boots[b] = np.linalg.lstsq(Xm[i], y[i], rcond=None)[0]
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    lines = ["| term (per +1 SD) | coefficient | 95% CI |", "|---|---:|---|"]
    names = ["intercept (average user)"] + [("log activity" if f == "activity" else f) for f in feats]
    for n, bt, l, h in zip(names, beta, lo, hi):
        star = "" if l <= 0 <= h else " *"
        lines.append(f"| {n} | {fmt(bt, 5)}{star} | [{fmt(l, 5)}, {fmt(h, 5)}] |")
    return "\n".join(lines), len(y)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="run_k30_TO")
    ap.add_argument("--split", default="TO")
    ap.add_argument("--inter", default="data/kc_k30/ikgr-custom/ikgr-custom.inter")
    ap.add_argument("--kg_pack", default=None, help="default: <run_dir>/kg_pack.pt")
    ap.add_argument("--meta_kg_pack", default=None, help="default: <run_dir>/meta_kg_pack.pt")
    ap.add_argument("--per_user_dir", default=None, help="default: <run_dir>/per_user_<split>")
    ap.add_argument("--a", default="IKGR_lgcn_dyn_nointent")
    ap.add_argument("--b", default="LightGCN")
    ap.add_argument("--intent_a", default="IKGR_lgcn_dyn", help="intent ablation: this minus --a on shared seeds")
    ap.add_argument("--ubiquitous_df", type=float, default=0.5)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--n_boot_ols", type=int, default=200)
    ap.add_argument("--out", default=None, help="default: <run_dir>/user_segments_<split>.md")
    args = ap.parse_args()
    kg_pack = args.kg_pack or os.path.join(args.run_dir, "kg_pack.pt")
    meta = args.meta_kg_pack or os.path.join(args.run_dir, "meta_kg_pack.pt")
    pud = args.per_user_dir or os.path.join(args.run_dir, f"per_user_{args.split}")
    out = args.out or os.path.join(args.run_dir, f"user_segments_{args.split}.md")
    rng = np.random.default_rng(0)

    feats, dropped = user_features(args.inter, args.split, kg_pack, meta, args.ubiquitous_df)
    A, a_seeds, a_all = load_model(pud, args.a)
    B, b_seeds, b_all = load_model(pud, args.b)
    df = feats.join(A.add_prefix("A_"), how="inner").join(B.add_prefix("B_"), how="inner")
    if (df["A_n_rel_tail"] != df["B_n_rel_tail"]).any():
        raise SystemExit("per-user files disagree on the test set (n_rel_tail) - different splits?")
    df["d_ndcg"] = df["A_ndcg@10"] - df["B_ndcg@10"]
    df["d_tail"] = df["A_tail_recall@10"] - df["B_tail_recall@10"]
    tail = df[df["A_n_rel_tail"] > 0]

    shared = sorted(set(a_seeds) & set(b_seeds))

    def seed_diff(metric):
        piv_a = a_all.pivot_table(index="user_id", columns="seed", values=metric)
        piv_b = b_all.pivot_table(index="user_id", columns="seed", values=metric)

        def f(idx, s):
            return float((piv_a.loc[idx, s] - piv_b.loc[idx, s]).mean())
        f.seeds = shared
        return f

    feat_names = [f for f, _ in FEATURES]
    corr = df[feat_names].corr(method="spearman")

    L = []
    L.append(f"# User segments ({args.split}, test) — who gains from {args.a} vs {args.b}\n")
    L.append(f"Generated by `analyze_user_segments.py`. A = `{args.a}` (seeds {', '.join(map(str, a_seeds))}), "
             f"B = `{args.b}` (seeds {', '.join(map(str, b_seeds))}); per-user metrics averaged over seeds, "
             f"{len(df):,} test users ({len(tail):,} with a long-tail test item). "
             f"CI = 95% bootstrap over users ({args.n_boot} resamples). "
             f"Per-seed column: A − B within each shared seed ({', '.join(map(str, shared))}).\n")
    L.append("Features use the TRAIN split only: " + "; ".join(f"`{f}` = {d}" for f, d in FEATURES)
             + f". Shelves on more than {args.ubiquitous_df:.0%} of books are ignored for the entropy: "
             + ", ".join(f"`{s}`" for s in dropped) + ". "
             + f"If A does not use the LLM intents (`{args.a}`), the intent features only describe the user "
             "(what the LLM read in their train profile); they are not an input of A.\n")
    L.append("## Feature correlations (Spearman)\n")
    L.append("| | " + " | ".join(feat_names) + " |")
    L.append("|---|" + "---:|" * len(feat_names))
    for f in feat_names:
        L.append(f"| {f} | " + " | ".join(f"{corr.loc[f, g]:.2f}" for g in feat_names) + " |")
    L.append("")

    for metric, data, col_a, col_b, title in [
        ("ndcg@10", df, "A_ndcg@10", "B_ndcg@10", "NDCG@10"),
        ("tail_recall@10", tail, "A_tail_recall@10", "B_tail_recall@10", "tail Recall@10 (users with a tail test item)"),
    ]:
        L.append(f"## {title}: A − B by user segment\n")
        sd = seed_diff(metric)
        for f, desc in FEATURES:
            L.append(f"**by {f}** ({desc})\n")
            L.append(bucket_table(data, f, col_a, col_b, rng, args.n_boot, per_seed=sd))
            L.append("")

    others = [f for f in feat_names if f != "activity"]
    L.append("## Holding activity fixed: high vs low half of each feature inside each activity quintile\n")
    L.append("Each cell is the mean per-user A − B for users below / above the feature's median within that "
             "activity bucket. The last row averages (high − low) over the five buckets "
             f"({args.n_boot_ols} resamples).\n")
    for ycol, data, title in [("d_ndcg", df, "NDCG@10"), ("d_tail", tail, "tail Recall@10")]:
        L.append(f"**{title}**\n")
        L.append(within_activity_table(data, ycol, others, rng, args.n_boot_ols))
        L.append("")

    L.append("## Controlling for activity: OLS of the per-user difference on standardised features\n")
    L.append("Coefficient = change in the average per-user A − B for +1 SD of the feature, holding the others fixed "
             "(activity enters as log(1+activity)). `*` = 95% bootstrap CI excludes 0 "
             f"({args.n_boot_ols} resamples).\n")
    for ycol, data, title in [("d_ndcg", df, "NDCG@10"), ("d_tail", tail, "tail Recall@10")]:
        tab, n = ols_table(data, ycol, feat_names, rng, args.n_boot_ols)
        L.append(f"**{title}** (n = {n:,})\n")
        L.append(tab)
        L.append("")

    # intent ablation on the seeds both runs share
    try:
        Ia, ia_seeds, _ = load_model(pud, args.intent_a)
        base, base_seeds, _ = load_model(pud, args.a, seeds=set(ia_seeds))
        di = feats.join(Ia.add_prefix("I_"), how="inner").join(base.add_prefix("N_"), how="inner")
        L.append(f"## Intent ablation by intent features: `{args.intent_a}` − `{args.a}` "
                 f"(seed {', '.join(map(str, base_seeds))} only)\n")
        L.append("Does the LLM-intent channel help users whose intents are many or diverse?\n")
        for f in ("n_intents", "intent_disp"):
            L.append(f"**NDCG@10 by {f}**\n")
            L.append(bucket_table(di, f, "I_ndcg@10", "N_ndcg@10", rng, args.n_boot))
            L.append("")
    except SystemExit as e:
        L.append(f"_(intent ablation skipped: {e})_\n")

    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[segments] wrote {out} ({len(df):,} users)")


if __name__ == "__main__":
    main()
