# TODO — Goodreads k=30 TO 실험 (2026-11-09 기준)

모든 실험은 `architect` 브랜치 + `data_bundle/`만으로 **LLM/API 호출 없이** 돌릴 수 있다.
설계·프로토콜·현재 결과는 [`ARCHITECTURE.md`](ARCHITECTURE.md) 참고.

## 0. 준비 (처음 한 번)

```powershell
git clone -b architect https://github.com/HK1DS/goodreads_november
cd goodreads_november
powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1      # Windows + NVIDIA GPU (venv: %USERPROFILE%\.venvs\goodreads-ikgr)
& "$env:USERPROFILE\.venvs\goodreads-ikgr\Scripts\Activate.ps1"
python scripts/unpack_data_bundle.py                                   # sha256 확인 후 data/kc_k30/, run_k30_TO/*.pt 복원
python -m pytest -q tests/                                             # (선택) 11 passed 확인
```
Linux: `pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121 && pip install -r requirements-pipeline.txt`.

**매 실행 전에** (새 터미널마다) 아래 환경변수를 설정한다. 빠뜨리면 새 모델의 인기도 페널티 대조군이 계산되지 않는다.
```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:IKGR_POP_RERANK_SPECS = "IKGR_kgoff,BPR,LightGCN,IKGR_dyn,IKGR_dyn_nointent,IKGR_lgcn_dyn_nointent,IKGR_lgcn_dyn"
```
bash: `export PYTHONIOENCODING=utf-8 IKGR_POP_RERANK_SPECS=IKGR_kgoff,BPR,LightGCN,IKGR_dyn,IKGR_dyn_nointent,IKGR_lgcn_dyn_nointent,IKGR_lgcn_dyn`

### 공통 규칙
- 중단: `Ctrl+C` 한 번. 재개: **같은 커맨드 다시 입력** → 끝난 모델×seed는 건너뛰고, 돌던 것만 epoch 0부터 다시.
- 손해 없이 끊는 시점: 콘솔에 새 `===== <모델> seed=<n> (TO) =====` 헤더가 뜬 직후.
- 실행 중에는 코드와 `config.yaml`을 수정하지 않는다 (코드/설정 해시가 바뀌면 끝난 결과까지 무효화).
- 한 seed 안에서 비교하는 모델들은 **같은 PC**에서 돌린다 (GPU가 다르면 같은 seed라도 수치가 조금 다를 수 있음).
- 끝나면 `run_k30_TO/slice_eval_TO_result.json`과 `run_k30_TO/per_user_TO/` 폴더를 공유 → 결과를 합쳐 3-seed 리포트를 만든다.
- 시간은 RTX 3060 Ti(8GB) 기준. 64차원: IKGR_lgcn 계열 ~4.2시간/seed, LightGCN ~5.3시간/seed.

## 1. 진행 상태

| # | 실험 | 담당 | 상태 | 예상 시간 |
|---|---|---|---|---|
| 1 | seed 2020: LightGCN, IKGR_lgcn_dyn_nointent, IKGR_lgcn_dyn | 진욱 PC | ✅ 완료 (결과: `run_k30_TO/`) | — |
| 2 | seed 2021: LightGCN | 진욱 PC | ✅ 완료 (11/5) | 5.3h |
| 3 | seed 2021: IKGR_lgcn_dyn_nointent | 진욱 PC | ✅ 완료 (11/6) | 4.2h |
| 4 | **seed 2022: IKGR_lgcn_dyn_nointent + LightGCN** | **팀원 요청 1순위** | ⬜ | 9.5h |
| 5 | **64차원 베이스라인: BPR, IKGR_kgoff, IKGR_dyn_nointent × seed 2020·2021·2022** | **팀원 요청 2순위** | ⬜ | ~9h (추정) |
| 6 | 재현 확인: IKGR_lgcn_dyn_nointent seed 2020 (기준값 NDCG@10 0.0871) | 팀원 (선택, 안해도 될듯) | ⬜ | 4.2h |
| 7 | 사용자별 구분: intent 수·카테고리(shelf) 다양성 구간별 비교 (GPU 불필요) | 진욱 | ✅ 완료 (11/9, 2 seed) → `run_k30_TO/user_segments_TO.md` | ~1분 |
| 8 | intent 필터: 연결 아이템 수 상위 5/10/20% intent 제거 후 정확도 | 미정(~7까지 하고 괜찮은지 확인후 세부적으로 진행) | ⬜ 코드 준비 필요 | 비율당 1~4.5h |
| 9 | late fusion λ ∈ {0, .25, .5, .75, 1}: LightGCN × IKGR_lgcn 점수 혼합 (평가만) | 미정 | ⬜ 스크립트 준비 필요 | — |

## 2. 커맨드

### #3 (진욱 PC) seed 2021 IKGR_lgcn_dyn_nointent
```powershell
python run_pipeline.py --k 30 --split TO --steps ER --seeds 2020,2021 --specs IKGR_lgcn_dyn_nointent
```

### #4 (팀원 1순위) seed 2022 핵심 비교 — 약 9.5시간
```powershell
python run_pipeline.py --k 30 --split TO --steps ER --seeds 2022 --specs IKGR_lgcn_dyn_nointent,LightGCN
```
4~5시간 단위로 나누려면 `--specs IKGR_lgcn_dyn_nointent`와 `--specs LightGCN`을 따로 실행한다.

### #5 (팀원 2순위) 64차원 베이스라인 3 seed — 약 9시간(추정)
```powershell
python run_pipeline.py --k 30 --split TO --steps ER --seeds 2020,2021,2022 --specs BPR,IKGR_kgoff,IKGR_dyn_nointent
```
seed 하나씩 하려면 `--seeds 2020`처럼 바꿔서 나눠 실행한다.

### #6 (선택) 재현 확인
```powershell
python run_pipeline.py --k 30 --split TO --steps ER --seeds 2020 --specs IKGR_lgcn_dyn_nointent
```
→ `[IKGR_lgcn_dyn_nointent s2020] ndcg@10=...`가 0.0871 근처인지 확인 (같은 GPU·드라이버면 동일할 가능성이 높고, 다르면 소수점 넷째 자리 정도 차이가 날 수 있음).

### 결과 확인
```powershell
python paired_compare.py --run_dir run_k30_TO --split TO --report     # → run_k30_TO/report_TO.md
python analyze_user_segments.py --run_dir run_k30_TO --split TO       # #7 사용자 구간 분석 → run_k30_TO/user_segments_TO.md (~1분, GPU 불필요)
```
`analyze_user_segments.py`는 `per_user_TO/`에 있는 seed를 모두 평균하므로, seed 2022가 끝난 뒤 다시 실행하면 3-seed 버전이 된다.

## 3. 남은 결정 / 메모

- **2 seed 결과 (11/6):** IKGR_lgcn_dyn_nointent는 LightGCN 대비 NDCG −1.4%(유의), tail recall +50%, coverage +61%.
  seed 2020만 봤을 때의 "정확도 동률"은 성립하지 않는다. 인기도 페널티 대조군보다는 같은 tail에서 NDCG가 약 6.5% 높다.
  자세한 표는 `ARCHITECTURE.md` 4절(사용자 구간별 표는 4.1절), 사용자 단위 비교는 `run_k30_TO/report_TO.md`.

- **사용자별 구분 (#7, 11/9, 2 seed):** IKGR_lgcn_dyn_nointent − LightGCN을 사용자 특성 구간별로 비교.
  - 가장 큰 요인은 **활동량**: 하위 40%는 IKGR이 NDCG +0.0014~0.0016(두 seed 같은 방향, 구간별 CI는 0 포함), 상위 40%는 유의하게 낮음(−0.0027, −0.0056). tail 이득은 하위 20%에서 +0.0127로 가장 큼.
  - **intent 다양성**(intent 임베딩 간 평균 코사인 거리): intent가 한쪽으로 몰린 사용자(하위 20%)는 NDCG −0.0036으로 손해, 다양한 사용자(상위 40%)는 LightGCN과 동률. 활동량을 고정해도 다양한 쪽이 +0.0021 [+0.0003, +0.0036] 유리하다. tail 이득도 절대값으로는 다양한 사용자가 더 큼(+0.0083 vs +0.0037). 상대 %로는 반대(+43% vs +70%)라 LightGCN 기준값 차이 때문일 수 있음.
  - **카테고리(shelf) 다양성**: 그냥 보면 다양할수록 손해지만 활동량과 상관이 커서(ρ=0.52) 생긴 착시다. 활동량을 고정하면 오히려 약하게 유리하다(OLS +0.0013/SD, CI 하한이 0에 가까움).
  - **intent 수**는 intent 다양성과 상관(ρ=−0.56; intent가 많은 사용자는 비슷한 intent가 반복됨)이라 독립적인 신호가 약하다.
  - LLM intent 채널(seed 2020)은 어느 intent 수·다양성 구간에서도 NDCG를 올리지 않았다. 즉 "intent가 풍부한 사용자에게는 도움이 된다"는 가설도 성립하지 않는다.
  - 표 여러 개를 동시에 본 결과라 활동량 외의 경계선 효과(CI 끝이 0 근처)는 seed 2022로 다시 확인해야 한다.
- **LLM intent:** LightGCN 백본(seed 2020)·MF 백본(3 seed) 모두 intent를 넣으면 NDCG −0.8%로 기여 없음. #8(범용 intent 제거)이 이 결론을 뒤집을 수 있는 남은 실험.
- **40 epoch 상한:** LightGCN과 IKGR_lgcn_dyn_nointent 모두 best epoch 39 → "동일 40 epoch 예산"으로 명시. 상한 확대는 모든 모델 재실행이 필요해 보류.
- **Amazon/MovieLens:** `IKGR-main` 파이프라인 `eval_slices.py`의 `dok_matrix._update = dict.update`를 이 브랜치의 `_dok_update`로 바꾼 뒤 LightGCN을 다시 돌려야 한다 (기존 LightGCN 수치는 전파 없는 버전일 가능성이 높음).
- **512차원 결과:** `main` 브랜치(`e03f50a`)에 보관. 64차원 결과와 비교하지 않는다.
