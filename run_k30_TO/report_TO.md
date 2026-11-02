# Paired comparisons (TO, test)

Unit = user (metric averaged over seeds first). CI = 95% bootstrap over users; p = Wilcoxon signed-rank; W/T/L = % users where A wins/ties/loses.

| A vs B | metric | users | A | B | diff (rel) | 95% CI | p | W/T/L % |
|---|---|---:|---:|---:|---:|---|---:|---|
| IKGR_dyn vs BPR | ndcg@10 | 62,142 | 0.0840 | 0.0786 | +0.0054 (+6.9%) | [+0.0044, +0.0064] | 8.7e-23 | 21/61/18 |
| IKGR_dyn vs BPR | tail_recall@10 | 37,746 | 0.0168 | 0.0167 | +0.0002 (+1.1%) | [-0.0008, +0.0011] | 0.48 | 2/96/2 |
| IKGR_dyn vs LightGCN | ndcg@10 | 62,142 | 0.0840 | 0.0875 | -0.0035 (-4.0%) | [-0.0045, -0.0025] | 5.9e-34 | 18/61/21 |
| IKGR_dyn vs LightGCN | tail_recall@10 | 37,746 | 0.0168 | 0.0128 | +0.0041 (+31.7%) | [+0.0031, +0.0050] | 0.0029 | 2/96/1 |
| IKGR_dyn vs IKGR_kgoff | ndcg@10 | 62,142 | 0.0840 | 0.0776 | +0.0064 (+8.3%) | [+0.0054, +0.0074] | 9.3e-27 | 21/61/18 |
| IKGR_dyn vs IKGR_kgoff | tail_recall@10 | 37,746 | 0.0168 | 0.0160 | +0.0008 (+5.2%) | [-0.0001, +0.0018] | 0.27 | 2/96/2 |
| IKGR_dyn vs IKGR_dyn_nointent | ndcg@10 | 62,142 | 0.0840 | 0.0853 | -0.0013 (-1.5%) | [-0.0017, -0.0008] | 0.0041 | 14/71/15 |
| IKGR_dyn vs IKGR_dyn_nointent | tail_recall@10 | 37,746 | 0.0168 | 0.0175 | -0.0006 (-3.6%) | [-0.0012, -0.0001] | 0.6 | 1/98/1 |

**IKGR_dyn vs BPR — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 30 | 12,496 | +0.0072 | 19 | 17 |
| Q2 | 39 | 12,820 | +0.0056 | 19 | 17 |
| Q3 | 54 | 12,344 | +0.0068 | 21 | 18 |
| Q4 | 89 | 12,069 | +0.0034 | 22 | 19 |
| Q5 | 7137 | 12,413 | +0.0040 | 23 | 20 |

**IKGR_dyn vs BPR — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 33 | 7,923 | +0.0013 | 2 | 2 |
| Q2 | 48 | 7,719 | +0.0010 | 2 | 2 |
| Q3 | 71 | 7,175 | -0.0012 | 2 | 2 |
| Q4 | 124 | 7,411 | -0.0004 | 1 | 1 |
| Q5 | 7137 | 7,518 | +0.0000 | 3 | 2 |

**IKGR_dyn vs LightGCN — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 30 | 12,496 | +0.0033 | 18 | 19 |
| Q2 | 39 | 12,820 | -0.0014 | 17 | 19 |
| Q3 | 54 | 12,344 | -0.0029 | 17 | 21 |
| Q4 | 89 | 12,069 | -0.0073 | 18 | 23 |
| Q5 | 7137 | 12,413 | -0.0093 | 19 | 25 |

**IKGR_dyn vs LightGCN — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 33 | 7,923 | +0.0095 | 3 | 1 |
| Q2 | 48 | 7,719 | +0.0057 | 2 | 1 |
| Q3 | 71 | 7,175 | +0.0026 | 2 | 1 |
| Q4 | 124 | 7,411 | +0.0018 | 2 | 1 |
| Q5 | 7137 | 7,518 | +0.0001 | 3 | 2 |

**IKGR_dyn vs IKGR_kgoff — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 30 | 12,496 | +0.0096 | 20 | 17 |
| Q2 | 39 | 12,820 | +0.0068 | 19 | 17 |
| Q3 | 54 | 12,344 | +0.0069 | 21 | 18 |
| Q4 | 89 | 12,069 | +0.0038 | 22 | 19 |
| Q5 | 7137 | 12,413 | +0.0048 | 23 | 20 |

**IKGR_dyn vs IKGR_kgoff — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 33 | 7,923 | +0.0027 | 2 | 2 |
| Q2 | 48 | 7,719 | +0.0013 | 2 | 2 |
| Q3 | 71 | 7,175 | -0.0004 | 2 | 2 |
| Q4 | 124 | 7,411 | +0.0004 | 2 | 1 |
| Q5 | 7137 | 7,518 | -0.0001 | 3 | 2 |

**IKGR_dyn vs IKGR_dyn_nointent — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 30 | 12,496 | -0.0015 | 12 | 13 |
| Q2 | 39 | 12,820 | -0.0021 | 13 | 13 |
| Q3 | 54 | 12,344 | -0.0013 | 14 | 15 |
| Q4 | 89 | 12,069 | -0.0005 | 15 | 16 |
| Q5 | 7137 | 12,413 | -0.0010 | 17 | 17 |

**IKGR_dyn vs IKGR_dyn_nointent — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 33 | 7,923 | -0.0004 | 1 | 1 |
| Q2 | 48 | 7,719 | -0.0011 | 1 | 1 |
| Q3 | 71 | 7,175 | -0.0008 | 1 | 1 |
| Q4 | 124 | 7,411 | -0.0007 | 1 | 1 |
| Q5 | 7137 | 7,518 | -0.0002 | 1 | 1 |

# Accuracy / exposure Pareto (TO, seed means)

★ = not dominated on (NDCG@10, tail Recall@10); ◆ = not dominated on (NDCG@10, Coverage@10).

| result | seeds | NDCG@10 | ±sd | tail R@10 | Cov@10 | best epoch | frontier |
|---|---:|---:|---:|---:|---:|---:|---|
| LightGCN | 1 | 0.0875 | 0.0000 | 0.0128 | 0.4310 | 37 | ★◆ |
| IKGR_dyn_nointent | 1 | 0.0853 | 0.0000 | 0.0175 | 0.5194 | 9 | ★◆ |
| IKGR_dyn | 1 | 0.0840 | 0.0000 | 0.0168 | 0.5160 | 9 |  |
| BPR | 1 | 0.0786 | 0.0000 | 0.0167 | 0.5209 | 37 | ◆ |
| IKGR_full_hetero | 1 | 0.0780 | 0.0000 | 0.0159 | 0.5908 | 33 | ◆ |
| IKGR_kgoff | 1 | 0.0776 | 0.0000 | 0.0160 | 0.5178 | 37 |  |
| IKGR_dyn_pop0p25 | 1 | 0.0647 | 0.0000 | 0.0268 | 0.7412 | 9 | ★◆ |
| LightGCN_pop0p25 | 1 | 0.0608 | 0.0000 | 0.0259 | 0.7747 | 37 | ◆ |
| IKGR_kgoff_pop0p25 | 1 | 0.0593 | 0.0000 | 0.0256 | 0.7451 | 37 |  |
| BPR_pop0p25 | 1 | 0.0587 | 0.0000 | 0.0264 | 0.7556 | 37 |  |
| IKGR_kgoff_pop0p5 | 1 | 0.0390 | 0.0000 | 0.0344 | 0.8836 | 37 | ★◆ |
| IKGR_dyn_pop0p5 | 1 | 0.0379 | 0.0000 | 0.0325 | 0.8421 | 9 |  |
| BPR_pop0p5 | 1 | 0.0369 | 0.0000 | 0.0343 | 0.8956 | 37 | ◆ |
| LightGCN_pop0p5 | 1 | 0.0241 | 0.0000 | 0.0337 | 0.8685 | 37 |  |
| IKGR_kgoff_pop1p0 | 1 | 0.0165 | 0.0000 | 0.0354 | 0.7948 | 37 | ★ |
| BPR_pop1p0 | 1 | 0.0158 | 0.0000 | 0.0356 | 0.8273 | 37 | ★ |
| IKGR_dyn_pop1p0 | 1 | 0.0129 | 0.0000 | 0.0311 | 0.6681 | 9 |  |
| LightGCN_pop1p0 | 1 | 0.0091 | 0.0000 | 0.0265 | 0.5540 | 37 |  |
