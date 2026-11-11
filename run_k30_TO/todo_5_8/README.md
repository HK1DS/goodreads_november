# TODO #5·#8 실험 결과 — MF 64차원, 3 seed

실험 완료: **2026-11-10 08:25 KST**. 저장소 반영: 2026-11-11, `architect` 브랜치.
첨부된 정식 실행 산출물을 가져온 보고서이며, 이번 저장소 반영 과정에서 학습을 다시 실행하지 않았다.

## 결론

- **#5 완료:** BPR / IKGR_kgoff / IKGR_dyn_nointent × seeds 2020·2021·2022, 총 9회 학습.
- **#8 완료:** MF 기반 IKGR_dyn의 intent 제거율 0/5/10/20% × 같은 3 seed, 총 12회 학습.
- 대표 제거율은 **validation NDCG@10 평균으로 선택한 5%**다. Test에서 가장 높은 10%를 사후 선택하지 않았다.
- 5% 제거는 무필터 대비 test NDCG@10을 약 0.60% 개선했다. 그러나 no-intent 대비 차이는 약 0.09%이고 bootstrap CI가 0을 포함하므로, intent의 정확도 기여를 입증했다고 해석할 수 없다.
- 이 결과는 **MF 백본**에 한정된다. 기존 LightGCN 결과 및 512차원·레거시 결과와 합산하지 않는다.

## 저장 위치와 파일 역할

| 경로 | 내용 |
|---|---|
| `run_k30_TO/todo_5_8/README.md` | 이 보고서 |
| `slice_eval_TO_result.json` | 28개 조건 × 3 seed = 84개 평가 결과 |
| `all_results_by_seed.csv`, `all_results_mean_std.csv` | seed별 지표 및 평균·모집단 표준편차 |
| `validation_selection.csv` | validation 기반 제거율 선택 |
| `paired_statistics.csv/json` | 사용자 단위 paired 통계 |
| `fit_times.csv` | 완료된 21회 학습 시간 |
| `intent_filter_*.json` | 조건·seed별 제거 intent ID/degree 및 통계 |
| `per_user_TO/*.csv.gz` | 168개 validation/test 원본, Git LFS로 관리 |
| `raw_manifest.json` | 사용자별 파일 크기·행 수·SHA256 |
| `delivery/` | 첨부 results ZIP을 그대로 보존한 원본 증빙; `sha256.json`으로 검증 가능 |
| `../../configs/todo_5_8.yaml` | 전용 실험 설정 |
| `../../scripts/run_todo_5_8.py` | 점검·실행·요청 시 일시중지/재개 |
| `../../ikgr_core/intent_filter.py` | train 아이템 degree 기반 intent 필터 |
| `../../docs/todo_5_8_protocol.md` | 실험 준비 시점의 프로토콜 기록 |

위 표에서 파일명만 있는 경로는 이 디렉터리 기준이다. `delivery/repository/`의 TODO·ARCHITECTURE 및 문서의 과거 업로드/PR 관련 문구는 원본 스냅샷이며 현재 진행 상태나 작업 지시가 아니다. 실행 코드·설정은 저장소의 표준 위치에도 반영했다. 기존 `run_k30_TO/`의 다른 결과는 덮어쓰지 않았다.

## 공통 조건

k=30, 사용자별 시간순 TO 80/10/10, train-only 프로필·recency, 64차원, RTX 2070 SUPER 동일 장비. 최대 40 epoch, train batch 2048, lr=0.001, reg=1e-6, eval_step=2, stopping_step=3. 최고 validation NDCG@10 체크포인트로 test 평가.

평가는 validation/test 각각 **62,142명 전체와 전체 아이템**을 대상으로 사용자 128명/배치로 수행했다. RecBole 설정의 score-count 기준으로는 3,580,928(128 × padding 포함 27,976 아이템)이다. 1명/128명 배치 간 근접 동률 순위 차이가 관측되어 모든 정식 조건에 128을 동일 적용했다. RecBole은 미개선 횟수가 3을 **초과**할 때 종료하고 동점은 갱신하므로 단순한 '3회 연속 미개선'과 구분해야 한다.

필터는 train에 등장한 고유 아이템 연결 수로 intent를 정렬한다. positive-degree intent **114,794개**를 분모로 degree 내림차순·ID 오름차순, floor 비율을 적용한다. 제거 수는 0 / 5,739 / 11,479 / 22,958개다. 사용자·아이템 양쪽 연결을 기존 cap32 적용 전에 제거하고 ID·임베딩 행·남은 연결 순서를 유지하며 intent 임베딩은 frozen이다.

## 주요 test 결과

3 seed 평균 ± 모집단 표준편차(ddof=0). 원본 JSON의 seed별 반올림 지표를 집계했으므로 아래 값과 사용자별 원본 기반 paired 평균에 미세한 차이가 있다.

| 모델 | NDCG@10 | Recall@10 | tail Recall@10 | Coverage@10 | validation NDCG@10 |
|---|---:|---:|---:|---:|---:|
| BPR | 0.08180 ± 0.00179 | 0.08750 | 0.01130 | 0.50397 | — |
| IKGR_kgoff | 0.08110 ± 0.00150 | 0.08700 | 0.01110 | 0.50167 | — |
| IKGR_dyn_nointent | 0.08493 ± 0.00103 | 0.09167 | 0.01283 | 0.45123 | — |
| IKGR_dyn | 0.08450 ± 0.00051 | 0.09117 | 0.01320 | 0.46633 | 0.108343 |
| IKGR_dyn_filter5 | 0.08500 ± 0.00131 | 0.09157 | 0.01347 | 0.49660 | 0.108953 |
| IKGR_dyn_filter10 | 0.08540 ± 0.00134 | 0.09150 | 0.01270 | 0.45437 | 0.108803 |
| IKGR_dyn_filter20 | 0.08513 ± 0.00083 | 0.09167 | 0.01297 | 0.49003 | 0.108562 |

`—`는 선택용 집계 CSV에 값이 없다는 의미다. 제거율 선택의 validation 평균은 사용자별 원본에서 계산했으며, 최고 checkpoint의 반올림된 fit 점수와 구분한다.

### seed별 NDCG@10

| 모델 | 2020 | 2021 | 2022 |
|---|---:|---:|---:|
| BPR | 0.0837 | 0.0794 | 0.0823 |
| IKGR_kgoff | 0.0827 | 0.0791 | 0.0815 |
| IKGR_dyn_nointent | 0.0861 | 0.0851 | 0.0836 |
| IKGR_dyn | 0.0850 | 0.0838 | 0.0847 |
| IKGR_dyn_filter5 | 0.0867 | 0.0835 | 0.0848 |
| IKGR_dyn_filter10 | 0.0868 | 0.0836 | 0.0858 |
| IKGR_dyn_filter20 | 0.0863 | 0.0844 | 0.0847 |

## 사용자 단위 비교와 해석

각 사용자의 3 seed 지표를 먼저 평균한 뒤 사용자 단위 bootstrap 10,000회(95% CI, seed=0)와 Wilcoxon(zsplit)을 적용했다. NDCG는 62,142명, tail recall은 유효 사용자 37,746명 기준이다. p값은 다중 비교 보정 전 탐색적 값이며, bootstrap 평균 차이와 Wilcoxon은 서로 다른 통계이므로 판정이 다를 수 있다.

| 비교 (A − B) | 지표 | 평균 차이 | 95% bootstrap CI | Wilcoxon p |
|---|---|---:|---:|---:|
| IKGR_dyn_filter5 − IKGR_dyn | ndcg@10 | +0.0005058 | [+0.0001866, +0.0008228] | 0.000445 |
| IKGR_dyn_filter5 − IKGR_dyn | tail_recall@10 | +0.0002522 | [-0.0000705, +0.0005840] | 0.463778 |
| IKGR_dyn_filter5 − IKGR_dyn_nointent | ndcg@10 | +0.0000767 | [-0.0003339, +0.0004806] | 0.573315 |
| IKGR_dyn_filter5 − IKGR_dyn_nointent | tail_recall@10 | +0.0006168 | [+0.0002037, +0.0010378] | 0.343954 |

5% 제거는 무필터보다 정확도를 소폭 회복한다. no-intent와의 정확도 차이는 불확실하다. no-intent 대비 tail 평균 차이의 bootstrap CI는 양수지만 Wilcoxon p≈0.344이므로 일관된 유의성으로 표현하지 않는다. 10% 제거는 test NDCG 평균이 가장 높지만 대표 선택 조건이 아니며, 무필터 대비 tail recall 평균은 감소했다.

인기도 페널티 0.25/0.5/1.0은 각 학습 모델에서 추가 학습 없이 평가했다. 7개 기본 조건 + 21개 재랭킹 조건의 전체 수치는 [전체 집계](all_results_mean_std.csv), [seed별 원본 집계](all_results_by_seed.csv)에 있다. 기본 비교와 재랭킹 결과를 혼합하지 않는다.

## 실행 시간·중단 기록

최초 시작 2026-11-10 02:11:30, 완료 08:25:24 KST로 경과 약 6시간 14분이다. 05:26경 5% 제거 seed2020의 epoch12 기록 이후 프로세스가 소실됐으며 원인은 확인되지 않았다. 완료 12회는 재사용하고 해당 미완료 학습만 epoch0부터 다시 실행했다. 05:30:51 재시작 이후 정상 종료했다.

`fit_times.csv`는 완료된 학습+주기 검증+최고 모델 저장 시간이며 데이터 준비·최종 평가·내보내기·중단 손실 및 재시작 공백은 포함하지 않는다. 요청 시 저장/재개 기능은 준비 단계에서 검증됐지만 정식 실행 중 요청 일시중지는 사용하지 않았다.

## 재현과 검증

원본 기준 커밋은 `e75941a17ecc4b6dbf51a3a6eefd1f4d82bfd221` (`architect`). Python 3.11.18 / PyTorch 2.3.0+cu121 / RecBole 1.2.0 환경이며 전체 의존성은 [환경 기록](delivery/evidence/environment-requirements.txt), 코드·입력 지문은 [실행 명세](delivery/evidence/formal-run-manifest.json)와 결과 JSON에 있다.

```bash
git lfs install
git lfs pull --include="run_k30_TO/todo_5_8/per_user_TO/*.gz"
python scripts/unpack_data_bundle.py
python scripts/run_todo_5_8.py --check
python -m pytest -q tests/
```

학습 재실행은 별도의 깨끗한 checkout/출력 디렉터리에서 `python scripts/run_todo_5_8.py --run --phase all`을 사용한다. `--phase 5`/`--phase 8`로 분리 가능하다. 배포된 결과 파일 위에서 학습을 재실행하면 지문 검증·캐시 판단에 영향을 줄 수 있으므로 원본 결과를 보존한다. 체크포인트와 대규모 입력은 첨부에 없으며 입력은 `data_bundle/`에서 복원한다. 준비 당시 전용 환경 경로는 현재 환경에 맞춰 조정해야 한다.

첨부 최종 테스트 로그에는 **35 passed**, smoke 및 pause 검증 기록이 있다. 이번 반영에서는 모든 전달 파일의 SHA256, 168개 gzip의 무결성·행 수(각 62,142행), 실행 소스 지문, 84개 seed별 NDCG와 28개 평균·표준편차, validation 선택값을 확인했다. 실행 소스는 원본과 바이트 단위로 일치한다. 현재 반영 환경에는 PyTorch/RecBole/pytest가 없어 GPU 학습 및 해당 테스트 suite를 재실행하지 않았으며, Python 구문 검사만 별도로 수행했다.
