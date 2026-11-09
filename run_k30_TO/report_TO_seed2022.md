# Goodreads k=30 TO — team seed 2022

Run on the `architect` branch at commit `89579d3` with `--steps ER --seeds 2022 --specs IKGR_lgcn_dyn_nointent,LightGCN`. Hardware: GTX 1060 6GB; PyTorch 2.3.0+cu121. The run used the bundled Goodreads k=30 inputs and made no LLM/API calls.

| Model | NDCG@10 | Tail Recall@10 | Coverage@10 | Best epoch |
|---|---:|---:|---:|---:|
| IKGR_lgcn_dyn_nointent | 0.0863 | 0.0161 | 0.4547 | 35 |
| LightGCN | 0.0879 | 0.0101 | 0.2840 | 39 |

For IKGR relative to LightGCN, the per-user paired comparison gives NDCG@10 difference −0.0017 (95% bootstrap CI [−0.0025, −0.0008]) and tail Recall@10 difference +0.0060 (95% CI [+0.0052, +0.0069]).

`slice_eval_TO_seed2022_result.json` holds the full seed 2022 metrics for both models and their popularity-penalty variants. `per_user_TO/*__s2022__test.csv.gz` holds the eight per-user test tables needed for paired comparisons.

The existing `slice_eval_TO_result.json` and `report_TO.md` retain the earlier seed 2020 results. The seed 2022 run produced different code hashes in its run signatures, so these results are kept separate until the code-version difference is reviewed; this file is not a three-seed aggregate.
