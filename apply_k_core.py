import pandas as pd
import numpy as np
import argparse
import os

def filter_k_core(df, k):
    """Iteratively filter dataframe to obtain the (k, k)-core (rows = interactions).

    Works on integer category codes with a boolean row mask, so a 10M-row file
    needs a few hundred MB instead of several GB. The k-core is the unique
    maximal fixed point, so the kept rows (and their original order) are
    identical to the previous pandas value_counts/isin implementation.
    """
    u = df['user_id'].astype('category').cat.codes.to_numpy()
    i = df['item_id'].astype('category').cat.codes.to_numpy()
    nu, ni = int(u.max()) + 1, int(i.max()) + 1
    keep = np.ones(len(df), dtype=bool)
    iteration = 0
    while True:
        before = int(keep.sum())
        ucnt = np.bincount(u[keep], minlength=nu)          # user core filter
        keep &= ucnt[u] >= k
        icnt = np.bincount(i[keep], minlength=ni)          # item core filter
        keep &= icnt[i] >= k
        after = int(keep.sum())
        iteration += 1
        print(f"Iteration {iteration}: Users {len(np.unique(u[keep]))}, "
              f"Items {len(np.unique(i[keep]))}, Rows {after}", flush=True)
        if after == before:
            break
    return df[keep]


def collect_profiles_streaming(profiles_in, kcore_users, kcore_items, chunksize=200_000):
    """
    Stream the (potentially huge) profiles CSV in chunks and collect ONE profile
    text per k-core user and per k-core item. This avoids loading the full
    multi-GB/GB file into memory.

    Returns (user_prof_map, item_prof_map).
    """
    user_prof_map = {}
    item_prof_map = {}
    n_users_target = len(kcore_users)
    n_items_target = len(kcore_items)

    reader = pd.read_csv(
        profiles_in,
        usecols=["user_id", "user_profile", "item_id", "item_profile"],
        chunksize=chunksize,
        dtype=str,
        keep_default_na=False,
    )

    rows_scanned = 0
    for chunk in reader:
        rows_scanned += len(chunk)
        for uid, uprof, iid, iprof in zip(
            chunk["user_id"], chunk["user_profile"],
            chunk["item_id"], chunk["item_profile"]
        ):
            if uid in kcore_users and uid not in user_prof_map:
                user_prof_map[uid] = uprof
            if iid in kcore_items and iid not in item_prof_map:
                item_prof_map[iid] = iprof

        print(f"  scanned {rows_scanned:,} profile rows | "
              f"users {len(user_prof_map)}/{n_users_target} | "
              f"items {len(item_prof_map)}/{n_items_target}", flush=True)

        # Early exit once every k-core entity has a profile.
        if len(user_prof_map) >= n_users_target and len(item_prof_map) >= n_items_target:
            print("  all k-core profiles found, stopping early.")
            break

    return user_prof_map, item_prof_map


def build_cover_profiles(user_prof_map, item_prof_map):
    """
    Build a compact 'cover' profiles dataframe in which every k-core user and
    every k-core item appears at least once. The pipeline only needs each entity
    represented once (intents are extracted per unique profile), so this is
    sufficient and keeps all downstream steps tractable.
    """
    users = sorted(user_prof_map.keys())
    items = sorted(item_prof_map.keys())
    if not users or not items:
        raise RuntimeError("No profiles collected for k-core users/items. "
                           "Check that profiles CSV matches the interactions IDs.")

    n = max(len(users), len(items))
    rows = []
    for idx in range(n):
        uid = users[idx % len(users)]
        iid = items[idx % len(items)]
        rows.append({
            "user_id": uid,
            "user_profile": user_prof_map[uid],
            "item_id": iid,
            "item_profile": item_prof_map[iid],
        })
    return pd.DataFrame(rows, columns=["user_id", "user_profile", "item_id", "item_profile"])


def main():
    parser = argparse.ArgumentParser(description="Apply iterative K-Core filtering to Goodreads CSV datasets.")
    parser.add_argument("--profiles_in", default="data/profiles.csv", help="Path to unfiltered profiles.csv")
    parser.add_argument("--interactions_in", default="data/interactions.csv", help="Path to unfiltered interactions.csv")
    parser.add_argument("--k", type=int, default=20, help="k value for k-core filtering (default: 20)")
    parser.add_argument("--out_dir", default="data/k_core", help="Output directory for filtered files")
    parser.add_argument("--chunksize", type=int, default=200_000, help="Chunk size for streaming the profiles CSV")
    parser.add_argument("--no_profiles", action="store_true",
                        help="only k-core filter the interactions (legacy profiles.csv is time-agnostic/leaky)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading interactions...")
    if not os.path.exists(args.interactions_in):
        raise SystemExit(f"Error: unfiltered interactions '{args.interactions_in}' not found. "
                         "Run goodreads_preprocess.py first.")
    use_profiles = bool(args.profiles_in) and not args.no_profiles
    if use_profiles and not os.path.exists(args.profiles_in):
        raise SystemExit(f"Error: '{args.profiles_in}' not found. Pass --no_profiles to build only the "
                         "k-core interactions (default pipeline: build_trainonly_profiles.py makes profiles).")

    # category dtype keeps each distinct id string once (memory-safe for ~10M rows)
    df_inter = pd.read_csv(args.interactions_in, dtype={"user_id": "category", "item_id": "category"})
    print(f"Original interactions: {df_inter.shape}")

    # 1) k-core filter on interactions (fits in memory; ~432MB file)
    print(f"\nFiltering interactions with k={args.k}...")
    filtered_inter = filter_k_core(df_inter, args.k)

    kcore_users = set(filtered_inter["user_id"].astype(str))
    kcore_items = set(filtered_inter["item_id"].astype(str))
    print(f"k-core: {len(kcore_users)} users, {len(kcore_items)} items, {len(filtered_inter)} interactions")

    interactions_out = os.path.join(args.out_dir, f"interactions_k{args.k}.csv")
    filtered_inter.to_csv(interactions_out, index=False)
    print(f"[Saved] interactions -> {interactions_out}")
    if not use_profiles:
        print("[Success] interactions only (--no_profiles).")
        return

    # 2) Stream the huge profiles file to collect one profile per k-core entity
    print(f"\nStreaming profiles from '{args.profiles_in}' (chunksize={args.chunksize:,})...")
    user_prof_map, item_prof_map = collect_profiles_streaming(
        args.profiles_in, kcore_users, kcore_items, chunksize=args.chunksize
    )
    missing_u = len(kcore_users) - len(user_prof_map)
    missing_i = len(kcore_items) - len(item_prof_map)
    if missing_u or missing_i:
        print(f"[warn] missing profiles for {missing_u} users / {missing_i} items "
              f"(they will be skipped from the cover).")

    # 3) Build compact cover profiles and save
    cover = build_cover_profiles(user_prof_map, item_prof_map)
    profiles_out = os.path.join(args.out_dir, f"profiles_k{args.k}.csv")
    cover.to_csv(profiles_out, index=False)

    print(f"\n[Success] k={args.k} dataset saved to '{args.out_dir}':")
    print(f"- Interactions: {filtered_inter.shape} -> {interactions_out}")
    print(f"- Profiles (cover): {cover.shape} -> {profiles_out}")
    print(f"Unique Users: {len(kcore_users)}, Unique Items: {len(kcore_items)}")


if __name__ == "__main__":
    main()
