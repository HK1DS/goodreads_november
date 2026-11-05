#!/usr/bin/env python3
"""Reassemble and extract the k=30 TO data bundle shipped in data_bundle/.

The bundle holds everything needed to run the evaluation stage (E) without any
LLM/API call: data/kc_k30/ (k-core interactions, RecBole .inter, train-only
profiles + provenance) and run_k30_TO/{kg_pack.pt, meta_kg_pack.pt}. It is split
into <100 MB parts because GitHub rejects larger files.

Usage (from the repository root):
    python scripts/unpack_data_bundle.py            # verify sha256, then extract
    python scripts/unpack_data_bundle.py --check    # verify only
"""
import argparse
import glob
import hashlib
import os
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_DIR = os.path.join(ROOT, "data_bundle")
PREFIX = "goodreads_k30_TO_bundle.tgz.part"
EXPECTED = {
    "files": [
        "data/kc_k30/ikgr-custom/ikgr-custom.inter",
        "data/kc_k30/interactions_k30.csv",
        "data/kc_k30/profiles_k30_TO.csv",
        "data/kc_k30/profile_provenance_k30_TO.csv.gz",
        "data/kc_k30/profile_provenance_k30_TO.csv.gz.meta.json",
        "run_k30_TO/kg_pack.pt",
        "run_k30_TO/meta_kg_pack.pt",
    ],
}


class _Concat:
    """Read-only file object over the ordered parts (no 600 MB temp file)."""

    def __init__(self, paths):
        self._files = [open(p, "rb") for p in paths]
        self._i = 0

    def read(self, n=-1):
        out = bytearray()
        while self._i < len(self._files) and (n < 0 or len(out) < n):
            chunk = self._files[self._i].read(-1 if n < 0 else n - len(out))
            if chunk:
                out += chunk
            else:
                self._i += 1
        return bytes(out)

    def close(self):
        for f in self._files:
            f.close()


def parts():
    ps = sorted(glob.glob(os.path.join(BUNDLE_DIR, PREFIX + "*")))
    if not ps:
        raise SystemExit(f"no bundle parts found in {BUNDLE_DIR}")
    return ps


def verify(ps):
    want = open(os.path.join(BUNDLE_DIR, "bundle.sha256"), encoding="ascii").read().split()[0]
    h = hashlib.sha256()
    for p in ps:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    got = h.hexdigest()
    if got != want:
        raise SystemExit(f"sha256 mismatch: {got} != {want} (re-download data_bundle/)")
    print(f"[bundle] {len(ps)} parts OK (sha256 {got[:16]}...)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only, do not extract")
    args = ap.parse_args()
    ps = parts()
    verify(ps)
    if args.check:
        return
    src = _Concat(ps)
    try:
        with tarfile.open(fileobj=src, mode="r|gz") as tar:
            for member in tar:
                name = os.path.normpath(member.name)
                if name.startswith("..") or os.path.isabs(name):
                    raise SystemExit(f"unsafe path in bundle: {member.name}")
                if hasattr(tarfile, "data_filter"):
                    tar.extract(member, ROOT, filter="data")
                else:
                    tar.extract(member, ROOT)
                if member.isfile():
                    print(f"[bundle] {member.name} ({member.size / 1e6:.1f} MB)")
    finally:
        src.close()
    missing = [f for f in EXPECTED["files"] if not os.path.exists(os.path.join(ROOT, f))]
    if missing:
        raise SystemExit(f"extraction incomplete, missing: {missing}")
    print("[bundle] done. Run e.g.: python run_pipeline.py --k 30 --split TO --steps ER "
          "--seeds 2022 --specs IKGR_lgcn_dyn_nointent,LightGCN")


if __name__ == "__main__":
    main()
