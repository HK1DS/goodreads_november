# data_bundle

k=30 Goodreads Children, per-user temporal split (TO) inputs used for the results in `run_k30_TO/`:
`data/kc_k30/` + `run_k30_TO/kg_pack.pt` (LLM intent KG) + `run_k30_TO/meta_kg_pack.pt` (metadata KG).
Split into <100 MB parts for GitHub. Restore from the repository root:

    python scripts/unpack_data_bundle.py

Derived from the UCSD Goodreads dataset (Wan & McAuley, RecSys 2018; Wan et al., ACL 2019), academic use only.
