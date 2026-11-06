# Paired comparisons (TO, test)

Unit = user (metric averaged over seeds first). CI = 95% bootstrap over users; p = Wilcoxon signed-rank; W/T/L = % users where A wins/ties/loses.

| A vs B | metric | users | A | B | diff (rel) | 95% CI | p | W/T/L % |
|---|---|---:|---:|---:|---:|---|---:|---|
| IKGR_lgcn_dyn_nointent vs LightGCN | ndcg@10 | 62,142 | 0.0865 | 0.0877 | -0.0012 (-1.4%) | [-0.0020, -0.0004] | 1.1e-08 | 21/57/22 |
| IKGR_lgcn_dyn_nointent vs LightGCN | tail_recall@10 | 37,746 | 0.0150 | 0.0100 | +0.0050 (+50.2%) | [+0.0043, +0.0057] | 2.5e-09 | 3/96/1 |
| IKGR_lgcn_dyn vs IKGR_lgcn_dyn_nointent | ndcg@10 | 62,142 | 0.0864 | 0.0865 | -0.0002 (-0.2%) | [-0.0007, +0.0004] | 4.9e-11 | 18/61/21 |
| IKGR_lgcn_dyn vs IKGR_lgcn_dyn_nointent | tail_recall@10 | 37,746 | 0.0150 | 0.0150 | -0.0000 (-0.3%) | [-0.0006, +0.0005] | 0.31 | 1/97/2 |

**IKGR_lgcn_dyn_nointent vs LightGCN — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 30 | 12,496 | +0.0016 | 19 | 20 |
| Q2 | 39 | 12,820 | +0.0015 | 20 | 19 |
| Q3 | 54 | 12,344 | -0.0008 | 20 | 21 |
| Q4 | 89 | 12,069 | -0.0028 | 21 | 23 |
| Q5 | 7137 | 12,413 | -0.0056 | 23 | 27 |

**IKGR_lgcn_dyn_nointent vs LightGCN — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 33 | 7,923 | +0.0127 | 4 | 1 |
| Q2 | 48 | 7,719 | +0.0061 | 3 | 1 |
| Q3 | 71 | 7,175 | +0.0029 | 2 | 1 |
| Q4 | 124 | 7,411 | +0.0022 | 2 | 1 |
| Q5 | 7137 | 7,518 | +0.0006 | 2 | 1 |

**IKGR_lgcn_dyn vs IKGR_lgcn_dyn_nointent — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 30 | 12,496 | +0.0008 | 17 | 18 |
| Q2 | 39 | 12,820 | -0.0004 | 17 | 18 |
| Q3 | 54 | 12,344 | +0.0003 | 18 | 19 |
| Q4 | 89 | 12,069 | -0.0011 | 19 | 22 |
| Q5 | 7137 | 12,413 | -0.0004 | 22 | 25 |

**IKGR_lgcn_dyn vs IKGR_lgcn_dyn_nointent — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 33 | 7,923 | -0.0001 | 2 | 2 |
| Q2 | 48 | 7,719 | -0.0013 | 1 | 2 |
| Q3 | 71 | 7,175 | +0.0005 | 1 | 1 |
| Q4 | 124 | 7,411 | +0.0003 | 1 | 1 |
| Q5 | 7137 | 7,518 | +0.0005 | 2 | 2 |

# Accuracy / exposure Pareto (TO, seed means)

★ = not dominated on (NDCG@10, tail Recall@10); ◆ = not dominated on (NDCG@10, Coverage@10).

| result | seeds | NDCG@10 | ±sd | tail R@10 | Cov@10 | best epoch | frontier |
|---|---:|---:|---:|---:|---:|---:|---|
| LightGCN | 2 | 0.0877 | 0.0001 | 0.0100 | 0.2853 | 39 | ★◆ |
| IKGR_lgcn_dyn_nointent | 2 | 0.0865 | 0.0006 | 0.0150 | 0.4589 | 38 | ★◆ |
| IKGR_lgcn_dyn | 1 | 0.0864 | 0.0000 | 0.0150 | 0.4353 | 29 |  |
| LightGCN_pop0p25 | 2 | 0.0811 | 0.0001 | 0.0146 | 0.5009 | 39 | ◆ |
| LightGCN_pop0p5 | 2 | 0.0695 | 0.0001 | 0.0209 | 0.7574 | 39 | ★◆ |
| IKGR_lgcn_dyn_pop0p25 | 1 | 0.0680 | 0.0000 | 0.0237 | 0.7164 | 29 | ★ |
| IKGR_lgcn_dyn_nointent_pop0p25 | 2 | 0.0673 | 0.0003 | 0.0250 | 0.7593 | 38 | ★◆ |
| LightGCN_pop1p0 | 2 | 0.0388 | 0.0005 | 0.0263 | 0.6000 | 39 | ★ |
| IKGR_lgcn_dyn_pop0p5 | 1 | 0.0378 | 0.0000 | 0.0301 | 0.8103 | 29 | ★◆ |
| IKGR_lgcn_dyn_nointent_pop0p5 | 2 | 0.0364 | 0.0002 | 0.0317 | 0.8374 | 38 | ★◆ |
| IKGR_lgcn_dyn_nointent_pop1p0 | 2 | 0.0104 | 0.0001 | 0.0276 | 0.5096 | 38 |  |
| IKGR_lgcn_dyn_pop1p0 | 1 | 0.0100 | 0.0000 | 0.0266 | 0.4762 | 29 |  |
