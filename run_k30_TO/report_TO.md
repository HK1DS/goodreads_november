# Paired comparisons (TO, test)

> **2026-11-13 검증 추가:** 아래 기존 main 집계에 이어, [팀원 실험 검증 및 추가 결과](#팀원-실험-검증-및-추가-결과-2026-11-13)에 TODO #4의 seed 2022와 3-seed 통합, TODO #5·#8의 MF 결과를 추가했다. 기존 첫 비교는 seeds 2020·2021의 이력이며, intent ablation은 양쪽에 공통으로 존재하는 **seed 2020만** 사용하도록 본문 두 행과 활동량 표를 정정했다. 기존 #7 사용자 구간 분석을 3-seed 분석으로 바꾸지는 않았다.

Unit = user (metric averaged over seeds first). CI = 95% bootstrap over users; p = Wilcoxon signed-rank; W/T/L = % users where A wins/ties/loses.

| A vs B | metric | users | A | B | diff (rel) | 95% CI | p | W/T/L % |
|---|---|---:|---:|---:|---:|---|---:|---|
| IKGR_lgcn_dyn_nointent vs LightGCN | ndcg@10 | 62,142 | 0.0865 | 0.0877 | -0.0012 (-1.4%) | [-0.0020, -0.0004] | 1.1e-08 | 21/57/22 |
| IKGR_lgcn_dyn_nointent vs LightGCN | tail_recall@10 | 37,746 | 0.0150 | 0.0100 | +0.0050 (+50.2%) | [+0.0043, +0.0057] | 2.5e-09 | 3/96/1 |
| IKGR_lgcn_dyn vs IKGR_lgcn_dyn_nointent | ndcg@10 | 62,142 | 0.0864 | 0.0871 | -0.0007 (-0.8%) | [-0.0014, -0.0001] | 0.013 | 16/67/17 |
| IKGR_lgcn_dyn vs IKGR_lgcn_dyn_nointent | tail_recall@10 | 37,746 | 0.0150 | 0.0153 | -0.0003 (-2.0%) | [-0.0010, +0.0004] | 0.85 | 1/98/1 |

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

# Accuracy / exposure Pareto (TO, 기존 main snapshot)

기존 main의 조건별 seed 수를 보존한 표다. 새 3-seed 비교는 아래 추가 결과를 사용한다.

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

---

# 팀원 실험 검증 및 추가 결과 (2026-11-13)

검토 기준: main `f7c86b6`, architect `14e8c4d`. 학습을 다시 실행하지 않고 실행 기록·설정·소스 지문과 사용자별 원본을 대조했다. 결과의 일관성과 약속한 프로토콜 충족을 확인했으며, 아래 범위에서 보고서에 채택한다.

| TODO | 팀원 담당 실험 | 확인 결과 |
|---|---|---|
| #4 | IKGR_lgcn_dyn_nointent·LightGCN, seed 2022 | 2회 학습, 인기도 재랭킹 포함 8개 평가 결과 확인; 기존 main의 2020·2021과 3-seed 집계 |
| #5 | BPR·IKGR_kgoff·IKGR_dyn_nointent, seeds 2020·2021·2022 | MF 64차원 대조군 9회 학습 완료 |
| #8 | MF IKGR_dyn의 intent 제거율 0/5/10/20%, 같은 3 seed | 12회 학습 완료; 대표 제거율은 validation만으로 선택한 5% |

#6의 seed 2020 재현 확인은 TODO에서 선택 항목이며, 이번 팀원 완료 실험에 포함하지 않는다.

## 검증한 증빙

- 원본 사용자별 gzip 176개를 확인했다. #5·#8의 168개는 SHA256·CRC·파일 크기·각 62,142행을 모두 검증했으며, #4의 8개는 Git LFS OID와 원본 SHA256이 일치한다.
- #5·#8의 21회 학습 및 84개 평가 완료 기록이 JSON·CSV·로그와 일치한다. 원본 test 지표 평균 672건, 최고 체크포인트 validation 점수 21건, 제거율 선택에 쓰인 validation 값 12건을 재계산했다. 주요 paired 비교 4건도 bootstrap과 Wilcoxon을 재계산해 저장값과 일치했다.
- 전달 증빙 59개와 실행 소스·설정·프로토콜 8개의 SHA256이 명세와 일치한다. train-only leakage guard는 모든 결과에서 통과했다. 필터 제거 수와 5% ⊆ 10% ⊆ 20% 관계도 확인했다.
- #4의 main 2020·2021 로컬 원본 16개와 architect 2022 원본 8개는 각 JSON의 반올림 지표와 일치하며, 사용자 ID·train 활동량·전체/long-tail 정답 수가 모두 같다. 기록된 입력 지문과 실행 설정도 일치한다. 코드·실행 지문 차이는 줄바꿈(LF/CRLF)과 머신 절대경로로 모두 설명된다.
- 로컬 main의 실제 `.inter` SHA256은 `0ddbc7447a1406add989194711814599d0ab32136dde679dbee9b9b697fb9fa5`이며 #5·#8 명세의 atomic_inter 지문과 일치한다. #4의 구 schema 3에는 해당 파일 지문이 없으므로 당시 입력 전체의 바이트 동일성까지 독립 증명한 것으로 해석하지 않는다.
- 소스 검증: `pytest -q tests/` 35 passed, 1 skipped(선택 CUDA 검사); 가짜 LLM/인코더를 쓰는 TO·TO_GLOBAL 오프라인 E2E, 재실행 skip 및 누수 방지 음성 검사가 통과했다.

## TODO #4 — LightGCN 계열 seed 2022 및 3-seed 통합

k=30, per-user TO 80/10/10, train-only 프로필·recency, 64차원, 최대 40 epoch, validation NDCG@10 체크포인트 선택이다. 팀원 seed 2022는 GTX 1060 6GB, 기존 main은 RTX 3060 Ti에서 수행했다. 머신 차이를 숨기지 않으며 하드웨어 간 비트 단위 재현성을 주장하지 않는다.

### seed별 기본 모델

| 모델 | seed 2020 NDCG@10 | seed 2021 | seed 2022 | best epoch (0-based, 2020/2021/2022) |
|---|---:|---:|---:|---|
| LightGCN | 0.0878 | 0.0877 | 0.0879 | 39 / 39 / 39 |
| IKGR_lgcn_dyn_nointent | 0.0871 | 0.0859 | 0.0863 | 39 / 37 / 35 |

### 3-seed 전체 집계

seeds 2020·2021·2022, 평균 ± 모집단 표준편차(ddof=0). 각 seed JSON에 저장된 반올림 지표를 집계한 값이므로 아래 사용자별 원본 기반 paired 평균과 미세한 차이가 있다. 인기도 페널티는 추가 학습 없이 평가한 조건이다.

| 모델/평가 조건 | NDCG@10 | tail Recall@10 | Coverage@10 |
|---|---:|---:|---:|
| LightGCN | 0.08780 ± 0.00008 | 0.01003 | 0.28487 |
| IKGR_lgcn_dyn_nointent | 0.08643 ± 0.00050 | 0.01537 | 0.45750 |
| LightGCN_pop0p25 | 0.08113 ± 0.00012 | 0.01460 | 0.50067 |
| LightGCN_pop0p5 | 0.06943 ± 0.00005 | 0.02087 | 0.75817 |
| IKGR_lgcn_dyn_nointent_pop0p25 | 0.06733 ± 0.00025 | 0.02530 | 0.75757 |
| LightGCN_pop1p0 | 0.03857 ± 0.00054 | 0.02610 | 0.59983 |
| IKGR_lgcn_dyn_nointent_pop0p5 | 0.03613 ± 0.00041 | 0.03177 | 0.83530 |
| IKGR_lgcn_dyn_nointent_pop1p0 | 0.01040 ± 0.00008 | 0.02740 | 0.50300 |

### 동일 3 seed의 사용자 단위 paired 비교

각 사용자의 3 seed 지표를 먼저 평균한 뒤 bootstrap 10,000회(95% CI, seed=0)와 Wilcoxon(zsplit)을 적용했다. A는 IKGR_lgcn_dyn_nointent다.

| 비교 (A − B) | 지표 | 사용자 수 | 평균 차이 | 상대 차이 | 95% bootstrap CI | Wilcoxon p |
|---|---|---:|---:|---:|---|---:|
| A − LightGCN | ndcg@10 | 62,142 | -0.0013494 | -1.54% | [-0.0020933, -0.0006122] | 2.38218e-05 |
| A − LightGCN | tail_recall@10 | 37,746 | +0.0053599 | +53.47% | [+0.0046582, +0.0060593] | 1.45854e-14 |
| A − LightGCN_pop0p25 | ndcg@10 | 62,142 | +0.0053073 | +6.54% | [+0.0045756, +0.0060320] | 5.80034e-70 |
| A − LightGCN_pop0p25 | tail_recall@10 | 37,746 | +0.0008063 | +5.53% | [+0.0001221, +0.0014858] | 0.981599 |

3 seed에서도 no-intent IKGR은 LightGCN보다 overall NDCG가 낮고 tail recall·coverage가 높다. 단순 인기도 페널티 0.25의 LightGCN과 비교하면 NDCG와 tail recall의 평균 차이 모두 양수이고 사용자 bootstrap CI도 0을 포함하지 않는다. 다만 이 tail 비교의 Wilcoxon p≈0.982로 평균 차이 CI와 판정이 다르므로 일관된 유의성으로 표현하지 않는다. 이는 명시한 페널티 지점과의 비교이며 모든 가능한 페널티에 대한 우위를 뜻하지 않는다.

LightGCN은 3 seed 모두 best epoch 39로 상한에 도달했다. 따라서 수렴 후 최종 성능이 아니라 동일 최대 40 epoch 예산의 비교로 해석한다. intent 채널은 seed 2020만 있어 3-seed intent 효과를 주장하지 않는다.

## TODO #5·#8 — MF 64차원 대조군과 intent 필터

동일 RTX 2070 SUPER에서 7개 기본 조건 × 3 seed = 21회 학습했다. 최대 40 epoch, train batch 2048, eval_step=2, stopping_step=3, lr=0.001, reg=1e-6, validation NDCG@10 기준 최고 모델을 사용했다. validation/test 각각 62,142명 전체와 전체 아이템을 사용자 128명/배치로 평가했다.
RecBole의 eval_batch_size는 128 × 27,976 = 3,580,928 score-count다. 기존 LightGCN 평가와 배치·장비가 다르고 근접 동률 순위 차이가 관측됐으므로 이 MF 표를 LightGCN 표와 합쳐 모델 우위를 주장하지 않는다.

필터는 train에 등장한 고유 아이템의 intent 연결 수를 사용했다. positive-degree intent 114,794개를 degree 내림차순·ID 오름차순으로 정렬하고, floor 비율로 0 / 5,739 / 11,479 / 22,958개를 제거했다. 사용자·아이템 양쪽 연결을 기존 cap32 전에 제거하며, intent ID·임베딩 행·남은 엣지 순서를 보존하고 intent 임베딩은 frozen이다.

### 3-seed test 지표

평균 ± 모집단 표준편차(ddof=0), 반올림된 seed JSON 기준.

| 조건 | NDCG@10 | Recall@10 | tail Recall@10 | Coverage@10 |
|---|---:|---:|---:|---:|
| BPR | 0.08180 ± 0.00179 | 0.08750 | 0.01130 | 0.50397 |
| IKGR_kgoff | 0.08110 ± 0.00150 | 0.08700 | 0.01110 | 0.50167 |
| IKGR_dyn_nointent | 0.08493 ± 0.00103 | 0.09167 | 0.01283 | 0.45123 |
| IKGR_dyn | 0.08450 ± 0.00051 | 0.09117 | 0.01320 | 0.46633 |
| IKGR_dyn_filter5 | 0.08500 ± 0.00131 | 0.09157 | 0.01347 | 0.49660 |
| IKGR_dyn_filter10 | 0.08540 ± 0.00134 | 0.09150 | 0.01270 | 0.45437 |
| IKGR_dyn_filter20 | 0.08513 ± 0.00083 | 0.09167 | 0.01297 | 0.49003 |

### validation 기준 대표 조건 선택

| intent 제거율 | 3-seed validation NDCG@10 평균 (사용자 원본) | 선택 |
|---|---:|---|
| 0% | 0.1083432356 | — |
| 5% | 0.1089530748 | 대표 조건 |
| 10% | 0.1088031931 | — |
| 20% | 0.1085618388 | — |

대표 조건은 validation 평균이 가장 높은 **5% 제거**다. test NDCG 평균이 가장 높은 10%를 사후 선택하지 않았다.

### 대표 5% 제거의 사용자 단위 비교

| 비교 (A − B) | 지표 | 평균 차이 | 95% bootstrap CI | Wilcoxon p |
|---|---|---:|---|---:|
| IKGR_dyn_filter5 − IKGR_dyn | ndcg@10 | +0.0005058 | [+0.0001866, +0.0008228] | 0.000444547 |
| IKGR_dyn_filter5 − IKGR_dyn | tail_recall@10 | +0.0002522 | [-0.0000705, +0.0005840] | 0.463778 |
| IKGR_dyn_filter5 − IKGR_dyn_nointent | ndcg@10 | +0.0000767 | [-0.0003339, +0.0004806] | 0.573315 |
| IKGR_dyn_filter5 − IKGR_dyn_nointent | tail_recall@10 | +0.0006168 | [+0.0002037, +0.0010378] | 0.343954 |

5% 필터는 무필터 IKGR_dyn보다 NDCG@10을 약 **0.60%** 회복하지만, no-intent보다 높은 정확도를 입증하지는 못했다(약 +0.09%, CI에 0 포함). no-intent 대비 tail 평균 차이의 bootstrap CI는 양수이나 Wilcoxon p≈0.344로 판정이 다르므로 일관된 유의성으로 표현하지 않는다.
추가 인기도 페널티 0.25/0.5/1.0까지 포함해 28개 평가 조건 × 3 seed = 84개 결과가 있다. 필터 선택과 위 핵심 비교에는 기본 모델 결과만 사용했다.

### 시간·중단 기록

2026-11-10 02:11:30–08:25:24 KST, 경과 약 6시간 14분이다. filter5 seed 2020의 미완료 실행이 한 차례 소실되어 해당 fit만 epoch 0부터 재실행했고, 이미 완료한 12회는 재사용했다. 로그상 최종 중복·누락은 없고 completed/exit 0이다. 완료된 21회 fit 시간 합계는 **19,490초(5시간 24분 50초)**이며 준비·최종 내보내기·중단 손실·재시작 공백은 제외한다. 정식 실행에서는 요청 pause/resume을 사용하지 않았다.

## 해석 범위와 재현 자료

- 사용자 bootstrap CI는 고정된 3 seed의 사용자 평균을 대상으로 한다. seed 재추출이나 하드웨어 차이에 대한 불확실성을 포함하지 않으며, p값은 다중 비교 보정 전 탐색적 값이다.
- TO에서는 train 이력이 0인 사용자가 없으므로 이 실험으로 true cold-start 효과를 주장하지 않는다. #7의 기존 사용자 구간 분석은 2-seed 결과로 유지한다.
- 이번 검증은 제공된 실행 기록과 원본 결과의 일관성 검증이다. 전체 GPU 학습이나 체크포인트로부터 예측을 다시 생성하지 않았다.
- 기존 main의 intent-vs-no-intent paired 행은 공통 seed 2020으로 정정했다. 모델별로 모든 가용 seed를 독립 평균하면 서로 다른 seed 구성이 섞이므로, 향후 비교도 공통 seed를 명시해야 한다.

기존 main의 2020·2021 사용자별 원본은 로컬 검증 입력이며 이번 보고서 갱신에 업로드하지 않았다. 아래 저장소 링크에는 seed 2022 사용자별 원본과 3 seed의 요약 JSON이 있지만, 3-seed paired 통계를 완전히 재현하려면 기존 2020·2021 사용자별 파일도 별도로 필요하다.

- TODO #4: [seed 2022 보고서](https://github.com/HK1DS/goodreads_november/blob/14e8c4d4c9f77ef52dd77c524f0102161e520520/run_k30_TO/report_TO_seed2022.md), [원본 JSON](https://github.com/HK1DS/goodreads_november/blob/14e8c4d4c9f77ef52dd77c524f0102161e520520/run_k30_TO/slice_eval_TO_seed2022_result.json), [사용자별 원본](https://github.com/HK1DS/goodreads_november/tree/14e8c4d4c9f77ef52dd77c524f0102161e520520/run_k30_TO/per_user_TO), [기존 main JSON](https://github.com/HK1DS/goodreads_november/blob/f7c86b610828b5dc1bef531622a1e34e85fc2ce0/run_k30_TO/slice_eval_TO_result.json).
- TODO #5·#8: [상세 보고서](https://github.com/HK1DS/goodreads_november/blob/14e8c4d4c9f77ef52dd77c524f0102161e520520/run_k30_TO/todo_5_8/README.md), [원본 JSON](https://github.com/HK1DS/goodreads_november/blob/14e8c4d4c9f77ef52dd77c524f0102161e520520/run_k30_TO/todo_5_8/slice_eval_TO_result.json), [전체 평균·표준편차](https://github.com/HK1DS/goodreads_november/blob/14e8c4d4c9f77ef52dd77c524f0102161e520520/run_k30_TO/todo_5_8/all_results_mean_std.csv), [사용자별 원본](https://github.com/HK1DS/goodreads_november/tree/14e8c4d4c9f77ef52dd77c524f0102161e520520/run_k30_TO/todo_5_8/per_user_TO).
- 프로토콜·실행 증빙: [전용 설정](https://github.com/HK1DS/goodreads_november/blob/14e8c4d4c9f77ef52dd77c524f0102161e520520/configs/todo_5_8.yaml), [프로토콜](https://github.com/HK1DS/goodreads_november/blob/14e8c4d4c9f77ef52dd77c524f0102161e520520/docs/todo_5_8_protocol.md), [실행 명세](https://github.com/HK1DS/goodreads_november/blob/14e8c4d4c9f77ef52dd77c524f0102161e520520/run_k30_TO/todo_5_8/delivery/evidence/formal-run-manifest.json), [증빙 SHA256](https://github.com/HK1DS/goodreads_november/blob/14e8c4d4c9f77ef52dd77c524f0102161e520520/run_k30_TO/todo_5_8/delivery/sha256.json), [원본 SHA256·행 수](https://github.com/HK1DS/goodreads_november/blob/14e8c4d4c9f77ef52dd77c524f0102161e520520/run_k30_TO/todo_5_8/raw_manifest.json), [paired 통계](https://github.com/HK1DS/goodreads_november/blob/14e8c4d4c9f77ef52dd77c524f0102161e520520/run_k30_TO/todo_5_8/paired_statistics.json).

원본 링크는 검토한 architect 커밋에 고정했다. 이 보고서 반영 자체가 실험 코드나 결과 디렉터리 전체의 main 병합을 뜻하지 않는다.
