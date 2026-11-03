# Paired comparisons (TO, test)

Unit = user (metric averaged over seeds first). CI = 95% bootstrap over users; p = Wilcoxon signed-rank; W/T/L = % users where A wins/ties/loses.

| A vs B | metric | users | A | B | diff (rel) | 95% CI | p | W/T/L % |
|---|---|---:|---:|---:|---:|---|---:|---|
| IKGR_dyn vs BPR | ndcg@10 | 62,142 | 0.0835 | 0.0791 | +0.0044 (+5.6%) | [+0.0037, +0.0051] | 1e-20 | 28/46/26 |
| IKGR_dyn vs BPR | tail_recall@10 | 37,746 | 0.0168 | 0.0163 | +0.0006 (+3.6%) | [-0.0001, +0.0012] | 0.17 | 4/93/3 |
| IKGR_dyn vs LightGCN | ndcg@10 | 62,142 | 0.0835 | 0.0860 | -0.0026 (-3.0%) | [-0.0033, -0.0018] | 2e-22 | 25/48/27 |
| IKGR_dyn vs LightGCN | tail_recall@10 | 37,746 | 0.0168 | 0.0131 | +0.0038 (+28.9%) | [+0.0031, +0.0044] | 1.4e-09 | 4/94/2 |
| IKGR_dyn vs IKGR_kgoff | ndcg@10 | 62,142 | 0.0835 | 0.0776 | +0.0059 (+7.6%) | [+0.0050, +0.0068] | 9.8e-174 | 30/51/20 |
| IKGR_dyn vs IKGR_kgoff | tail_recall@10 | 37,746 | 0.0168 | 0.0160 | +0.0008 (+5.2%) | [+0.0000, +0.0017] | 7.4e-10 | 4/94/2 |
| IKGR_dyn vs IKGR_dyn_nointent | ndcg@10 | 62,142 | 0.0835 | 0.0841 | -0.0006 (-0.8%) | [-0.0009, -0.0004] | 0.0042 | 23/52/24 |
| IKGR_dyn vs IKGR_dyn_nointent | tail_recall@10 | 37,746 | 0.0168 | 0.0174 | -0.0005 (-3.1%) | [-0.0009, -0.0002] | 0.3 | 2/96/2 |

**IKGR_dyn vs BPR — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 30 | 12,496 | +0.0059 | 26 | 24 |
| Q2 | 39 | 12,820 | +0.0054 | 26 | 24 |
| Q3 | 54 | 12,344 | +0.0054 | 28 | 25 |
| Q4 | 89 | 12,069 | +0.0020 | 29 | 27 |
| Q5 | 7137 | 12,413 | +0.0031 | 32 | 30 |

**IKGR_dyn vs BPR — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 33 | 7,923 | +0.0015 | 4 | 3 |
| Q2 | 48 | 7,719 | +0.0010 | 3 | 3 |
| Q3 | 71 | 7,175 | -0.0001 | 3 | 3 |
| Q4 | 124 | 7,411 | +0.0004 | 3 | 3 |
| Q5 | 7137 | 7,518 | -0.0000 | 5 | 4 |

**IKGR_dyn vs LightGCN — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 30 | 12,496 | +0.0033 | 23 | 24 |
| Q2 | 39 | 12,820 | +0.0009 | 23 | 24 |
| Q3 | 54 | 12,344 | -0.0018 | 24 | 26 |
| Q4 | 89 | 12,069 | -0.0064 | 26 | 28 |
| Q5 | 7137 | 12,413 | -0.0091 | 28 | 32 |

**IKGR_dyn vs LightGCN — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 33 | 7,923 | +0.0081 | 4 | 2 |
| Q2 | 48 | 7,719 | +0.0057 | 4 | 2 |
| Q3 | 71 | 7,175 | +0.0026 | 3 | 2 |
| Q4 | 124 | 7,411 | +0.0018 | 3 | 2 |
| Q5 | 7137 | 7,518 | +0.0003 | 5 | 3 |

**IKGR_dyn vs IKGR_kgoff — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 30 | 12,496 | +0.0087 | 27 | 19 |
| Q2 | 39 | 12,820 | +0.0066 | 27 | 19 |
| Q3 | 54 | 12,344 | +0.0067 | 29 | 19 |
| Q4 | 89 | 12,069 | +0.0035 | 31 | 21 |
| Q5 | 7137 | 12,413 | +0.0040 | 35 | 22 |

**IKGR_dyn vs IKGR_kgoff — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 33 | 7,923 | +0.0022 | 4 | 3 |
| Q2 | 48 | 7,719 | +0.0010 | 4 | 3 |
| Q3 | 71 | 7,175 | -0.0001 | 3 | 2 |
| Q4 | 124 | 7,411 | +0.0011 | 3 | 2 |
| Q5 | 7137 | 7,518 | -0.0002 | 5 | 3 |

**IKGR_dyn vs IKGR_dyn_nointent — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 30 | 12,496 | -0.0007 | 21 | 21 |
| Q2 | 39 | 12,820 | -0.0008 | 22 | 22 |
| Q3 | 54 | 12,344 | -0.0008 | 23 | 23 |
| Q4 | 89 | 12,069 | -0.0002 | 25 | 26 |
| Q5 | 7137 | 12,413 | -0.0008 | 28 | 29 |

**IKGR_dyn vs IKGR_dyn_nointent — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 33 | 7,923 | -0.0010 | 2 | 2 |
| Q2 | 48 | 7,719 | -0.0009 | 1 | 2 |
| Q3 | 71 | 7,175 | -0.0005 | 2 | 2 |
| Q4 | 124 | 7,411 | -0.0002 | 2 | 2 |
| Q5 | 7137 | 7,518 | -0.0000 | 3 | 3 |

# Accuracy / exposure Pareto (TO, seed means)

★ = not dominated on (NDCG@10, tail Recall@10); ◆ = not dominated on (NDCG@10, Coverage@10).

| result | seeds | NDCG@10 | ±sd | tail R@10 | Cov@10 | best epoch | frontier |
|---|---:|---:|---:|---:|---:|---:|---|
| LightGCN | 3 | 0.0860 | 0.0015 | 0.0131 | 0.4361 | 38 | ★◆ |
| IKGR_dyn_nointent | 3 | 0.0841 | 0.0009 | 0.0174 | 0.5302 | 10 | ★◆ |
| IKGR_dyn | 3 | 0.0835 | 0.0004 | 0.0168 | 0.5292 | 10 |  |
| BPR | 3 | 0.0791 | 0.0006 | 0.0163 | 0.5188 | 36 |  |
| IKGR_full_hetero | 1 | 0.0780 | 0.0000 | 0.0159 | 0.5908 | 33 | ◆ |
| IKGR_kgoff | 1 | 0.0776 | 0.0000 | 0.0160 | 0.5178 | 37 |  |
| IKGR_dyn_nointent_pop0p25 | 3 | 0.0649 | 0.0005 | 0.0269 | 0.7574 | 10 | ★◆ |
| IKGR_dyn_pop0p25 | 3 | 0.0643 | 0.0004 | 0.0264 | 0.7549 | 10 |  |
| LightGCN_pop0p25 | 3 | 0.0597 | 0.0009 | 0.0261 | 0.7793 | 38 | ◆ |
| BPR_pop0p25 | 3 | 0.0593 | 0.0005 | 0.0261 | 0.7527 | 36 |  |
| IKGR_kgoff_pop0p25 | 1 | 0.0593 | 0.0000 | 0.0256 | 0.7451 | 37 |  |
| IKGR_dyn_nointent_pop0p5 | 3 | 0.0394 | 0.0006 | 0.0334 | 0.8596 | 10 | ★◆ |
| IKGR_kgoff_pop0p5 | 1 | 0.0390 | 0.0000 | 0.0344 | 0.8836 | 37 | ★◆ |
| IKGR_dyn_pop0p5 | 3 | 0.0390 | 0.0009 | 0.0328 | 0.8567 | 10 |  |
| BPR_pop0p5 | 3 | 0.0370 | 0.0001 | 0.0341 | 0.8933 | 36 | ◆ |
| LightGCN_pop0p5 | 3 | 0.0240 | 0.0000 | 0.0345 | 0.8735 | 38 | ★ |
| IKGR_kgoff_pop1p0 | 1 | 0.0165 | 0.0000 | 0.0354 | 0.7948 | 37 | ★ |
| BPR_pop1p0 | 3 | 0.0158 | 0.0002 | 0.0357 | 0.8260 | 36 | ★ |
| IKGR_dyn_nointent_pop1p0 | 3 | 0.0139 | 0.0002 | 0.0323 | 0.7002 | 10 |  |
| IKGR_dyn_pop1p0 | 3 | 0.0137 | 0.0006 | 0.0317 | 0.6987 | 10 |  |
| LightGCN_pop1p0 | 3 | 0.0092 | 0.0001 | 0.0268 | 0.5596 | 38 |  |
