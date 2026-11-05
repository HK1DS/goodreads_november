# Paired comparisons (TO, test)

Unit = user (metric averaged over seeds first). CI = 95% bootstrap over users; p = Wilcoxon signed-rank; W/T/L = % users where A wins/ties/loses.

| A vs B | metric | users | A | B | diff (rel) | 95% CI | p | W/T/L % |
|---|---|---:|---:|---:|---:|---|---:|---|
| IKGR_lgcn_dyn_nointent vs LightGCN | ndcg@10 | 62,142 | 0.0871 | 0.0878 | -0.0007 (-0.8%) | [-0.0015, +0.0002] | 1.3e-08 | 17/64/19 |
| IKGR_lgcn_dyn_nointent vs LightGCN | tail_recall@10 | 37,746 | 0.0153 | 0.0104 | +0.0049 (+47.4%) | [+0.0041, +0.0057] | 0.00015 | 2/97/1 |
| IKGR_lgcn_dyn vs IKGR_lgcn_dyn_nointent | ndcg@10 | 62,142 | 0.0864 | 0.0871 | -0.0007 (-0.8%) | [-0.0014, -0.0001] | 0.013 | 16/67/17 |
| IKGR_lgcn_dyn vs IKGR_lgcn_dyn_nointent | tail_recall@10 | 37,746 | 0.0150 | 0.0153 | -0.0003 (-2.0%) | [-0.0010, +0.0004] | 0.85 | 1/98/1 |

**IKGR_lgcn_dyn_nointent vs LightGCN — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 30 | 12,496 | +0.0020 | 16 | 17 |
| Q2 | 39 | 12,820 | +0.0026 | 16 | 16 |
| Q3 | 54 | 12,344 | -0.0010 | 16 | 18 |
| Q4 | 89 | 12,069 | -0.0020 | 18 | 19 |
| Q5 | 7137 | 12,413 | -0.0051 | 19 | 23 |

**IKGR_lgcn_dyn_nointent vs LightGCN — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 33 | 7,923 | +0.0123 | 3 | 1 |
| Q2 | 48 | 7,719 | +0.0058 | 2 | 1 |
| Q3 | 71 | 7,175 | +0.0031 | 1 | 1 |
| Q4 | 124 | 7,411 | +0.0023 | 1 | 1 |
| Q5 | 7137 | 7,518 | +0.0005 | 1 | 1 |

**IKGR_lgcn_dyn vs IKGR_lgcn_dyn_nointent — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 30 | 12,496 | +0.0007 | 14 | 14 |
| Q2 | 39 | 12,820 | -0.0019 | 14 | 15 |
| Q3 | 54 | 12,344 | +0.0003 | 16 | 16 |
| Q4 | 89 | 12,069 | -0.0018 | 17 | 18 |
| Q5 | 7137 | 12,413 | -0.0010 | 20 | 21 |

**IKGR_lgcn_dyn vs IKGR_lgcn_dyn_nointent — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 33 | 7,923 | -0.0004 | 1 | 1 |
| Q2 | 48 | 7,719 | -0.0015 | 1 | 1 |
| Q3 | 71 | 7,175 | -0.0001 | 1 | 1 |
| Q4 | 124 | 7,411 | +0.0001 | 1 | 1 |
| Q5 | 7137 | 7,518 | +0.0004 | 1 | 1 |

# Accuracy / exposure Pareto (TO, seed means)

★ = not dominated on (NDCG@10, tail Recall@10); ◆ = not dominated on (NDCG@10, Coverage@10).

| result | seeds | NDCG@10 | ±sd | tail R@10 | Cov@10 | best epoch | frontier |
|---|---:|---:|---:|---:|---:|---:|---|
| LightGCN | 1 | 0.0878 | 0.0000 | 0.0104 | 0.2876 | 39 | ★◆ |
| IKGR_lgcn_dyn_nointent | 1 | 0.0871 | 0.0000 | 0.0153 | 0.4616 | 39 | ★◆ |
| IKGR_lgcn_dyn | 1 | 0.0864 | 0.0000 | 0.0150 | 0.4353 | 29 |  |
| LightGCN_pop0p25 | 1 | 0.0813 | 0.0000 | 0.0151 | 0.5036 | 39 | ◆ |
| LightGCN_pop0p5 | 1 | 0.0695 | 0.0000 | 0.0210 | 0.7613 | 39 | ★◆ |
| IKGR_lgcn_dyn_pop0p25 | 1 | 0.0680 | 0.0000 | 0.0237 | 0.7164 | 29 | ★ |
| IKGR_lgcn_dyn_nointent_pop0p25 | 1 | 0.0676 | 0.0000 | 0.0254 | 0.7666 | 39 | ★◆ |
| LightGCN_pop1p0 | 1 | 0.0393 | 0.0000 | 0.0267 | 0.6061 | 39 | ★ |
| IKGR_lgcn_dyn_pop0p5 | 1 | 0.0378 | 0.0000 | 0.0301 | 0.8103 | 29 | ★◆ |
| IKGR_lgcn_dyn_nointent_pop0p5 | 1 | 0.0366 | 0.0000 | 0.0319 | 0.8418 | 39 | ★◆ |
| IKGR_lgcn_dyn_nointent_pop1p0 | 1 | 0.0105 | 0.0000 | 0.0278 | 0.5163 | 39 |  |
| IKGR_lgcn_dyn_pop1p0 | 1 | 0.0100 | 0.0000 | 0.0266 | 0.4762 | 29 |  |
