# Goodreads Children 실험: LLM intent · 메타데이터 · recency를 CF에 보완 신호로 쓰는 추천 파이프라인

한양대 데이터사이언스 졸업프로젝트(지도교수 김현준, 2026.09–12)의 Goodreads 도메인 실험 레포다.
IKGR / DynLLM / CORONA에서 착안한 구성요소(LLM intent KG, 메타데이터 KG, recency 동적 프로필,
그래프 후보생성·재랭킹)를 RecBole 위에 재구현하고, **언제 이 신호들이 협업 필터링(CF)을 보완하는지**를
시간순 평가로 검증한다. MovieLens 25M / Amazon Clothing 실험은 별도 레포에 있다.

> **현재 상태 (2026-11-03)**
> 이전 Goodreads 결과(`run_k30/`, `run_k50/`, `IKGR_REPORT.md`, `DYNLLM_REPORT.md`, `CORONA_REPORT.md`,
> Progress Report 2의 Goodreads† 행)는 **user profile이 split과 무관하게 만들어진 레거시 프로토콜**의 결과다.
> 프로필에 검증/테스트 기간 책이 들어갈 수 있었으므로 **최종 보고서에는 인용하지 말고**, 아래의 train-only
> 파이프라인으로 다시 돌린 결과(`run_k{K}_{SPLIT}/`)를 쓴다. k=30 TO 실데이터 재실행 완료: 6개 spec seed 2020 + IKGR_dyn_nointent·IKGR_dyn·LightGCN·BPR seed 2021·2022
> (결과 `run_k30_TO/slice_eval_TO_result.json`, 유저 단위 비교 `run_k30_TO/report_TO.md`).
>
> **⚠️ 2026-11-04 — LightGCN 베이스라인 버그 (레거시 + 2026-11-03 결과 모두 해당)**
> RecBole 1.2.0의 LightGCN은 `scipy.sparse.dok_matrix._update()`로 인접행렬을 만든다. SciPy ≥ 1.13에는 이 메서드가
> 없어서 `eval_slices.py`/`run_baselines.py`가 `_update = dict.update`로 대체했는데, SciPy 1.13부터 dok 값은
> `self._dict`에 저장되므로 이 대체는 **빈 인접행렬**을 만든다(에러 없음). 즉 지금까지의 모든 "LightGCN" 결과는 그래프
> 전파가 없는 MF 변형이다(`requirements-pipeline.txt`의 scipy==1.13.1에서 재현). shim을 `self._dict`에 쓰도록 고쳤고,
> `tests/test_lgcn_backbone.py`가 인접행렬 nnz와 RecBole LightGCN 점수 일치를 검사한다. LightGCN은 다시 돌려야 한다
> (`scripts\run_lgcn_backbone.bat`). 같은 스크립트가 새 spec `IKGR_lgcn_dyn_nointent`/`IKGR_lgcn_dyn`(메타데이터 KG +
> recency ± intent를 LightGCN 백본 위에 얹은 모델, `lgcn_layers: 2`)도 돌린다.
>
> **2026-11-04 — 공통 임베딩 차원 512 → 64.** 고친 LightGCN은 배치마다 그래프 전체를 전파해서 512차원에선
> epoch당 약 24분(seed당 최대 ~17시간)이 걸렸다. 그래서 모든 spec의 `embedding_size`를 RecBole LightGCN 기본값인
> 64로 낮추고 처음부터 다시 돌린다(`scripts\run_emb64.bat`). 512차원 결과(LightGCN은 전파 없는 버전)는
> `run_k30_TO/archive_emb512/`에 보관했다. 두 차원의 수치는 서로 비교하지 않는다.

---

## 1. 무엇이 바뀌었나

| 문제 (레거시) | 수정 |
|---|---|
| user profile = 원본 파일 순서상 앞 20권, 시간·split 무시, 2,000자 컷(사실상 책 1–2권). 검증/테스트 책이 섞일 수 있음. TO_GLOBAL의 train 0개 유저도 (train 밖의 책으로 만든) 프로필 보유 | `build_trainonly_profiles.py`: RecBole split을 그대로 재현해 **해당 split의 train 구간 중, 그 유저(TO)/전역(TO_GLOBAL) 첫 비-train 시점보다 엄격히 이전인 상호작용만** 사용. train 이력이 없으면 빈 프로필 → intent 없음 |
| 프로필이 split 하나에만 맞음 (TO용 프로필을 TO_GLOBAL에 쓰면 누수) | split마다 별도 프로필·intent·KG·결과 디렉터리 `run_k{K}_{SPLIT}/` |
| 누수 여부를 확인하는 장치 없음 | `eval_slices.py` **누수 가드**: 프로필에 쓰인 모든 (user, item)이 RecBole의 실제 train split에 있는지, KG에 intent가 있는 모든 유저가 train-only 프로필을 갖는지 매 실행 검사. 다른 split용 프로필이면 중단 |
| 모든 모델 12 epoch 고정 (LightGCN 등 미수렴), 모델 선택 지표 MRR@10 | 공통 프로토콜: 최대 40 epoch, 2 epoch마다 검증, 3회 연속 개선 없으면 정지, **검증 NDCG@10 최고 체크포인트**로 테스트. 모든 spec 동일 |
| long-tail/cold 정의가 테스트 유저의 train+valid 이력 기준 | **train split 기준** 인기도·활동도 (`definitions` 필드로 결과 JSON에 기록) |
| "KG가 다양성을 올린다"를 검증할 대조군 없음 | **인기도 페널티 재랭킹 대조군**: `s − λ·σ_u·z(log(1+pop_train))`, λ∈{0.25, 0.5, 1.0}을 MF/BPR/LightGCN/IKGR_dyn에 적용(재학습 없음). Pareto 표로 비교 |
| intent의 독립 기여가 분리 안 됨 | `IKGR_dyn_nointent`: IKGR_dyn과 같은 융합 구조에서 intent 채널만 끈 ablation |
| 3시드 std만 보고 | 유저별 지표 덤프 + `paired_compare.py`: 유저 단위 paired bootstrap 95% CI, Wilcoxon p, win/tie/loss, 활동도 구간별 차이 |
| 37 GB `profiles.csv` 필요 | 불필요(기본 파이프라인에서 생성 안 함). 레거시 재현용으로만 `--legacy_profiles` |

구조 프로필(기본값)은 유저의 가장 최근 train 도서 50권에서 만든다:
`Recently read (newest first): 제목 by 저자; … | Frequent authors: … | Common shelves: …`
(`--profile_style legacy`를 주면 예전 규칙에 시간 제한만 더한 방식).

## 2. 빠른 실행 (Windows, RTX 3060 Ti 기준)

```powershell
# .env 에 LUXIA_API_KEY 설정 후
powershell -ExecutionPolicy Bypass -File scripts\run_goodreads_revalidation.ps1             # k=30, TO, 3 seeds
powershell -ExecutionPolicy Bypass -File scripts\run_goodreads_revalidation.ps1 -Seeds 2020 # 먼저 1 seed만
powershell -ExecutionPolicy Bypass -File scripts\run_goodreads_revalidation.ps1 -SkipLLM    # 무료 단계 + 비용 추정만
```

**Python 환경**: `-Python`을 주지 않고 레포에 `.venv`도 없으면, 첫 실행 때 `scripts\setup_venv.ps1`이
`%USERPROFILE%\.venvs\goodreads-ikgr`에 venv를 만든다(OneDrive 밖, 약 5GB, 10–20분). Python 3.10–3.12를
`py` 런처 → anaconda base → PATH 순으로 찾고, torch 2.3.0(CUDA 12.1 빌드) + `requirements-pipeline.txt`를 설치한 뒤
import·CUDA를 점검한다. 이미 쓰던 환경이 있으면 `-Python <python.exe 경로>`로 지정하면 된다.

스크립트 순서: 환경 점검(실패 시 유료 단계 전에 중단) → 원본 다운로드(레포의 `*.json.gz`는 Git-LFS 포인터라 실제 파일로 교체) → 무료 전처리 →
**LLM 비용 추정 후 y/N 확인** → step1·step2(유료) → KG → 학습·평가 → 리포트. 캐시와 완료된 seed는
재사용되므로 중간에 끊겨도 같은 명령으로 이어진다. 준비 단계(0·A·T·P) 산출물이 이미 있으면 건너뛴다(다시 만들려면 `--force_prep`).

직접 실행할 때:

```bash
python run_pipeline.py --k 30 --split TO --steps 0ATPS        # 무료 준비 + 비용 추정
python run_pipeline.py --k 30 --split TO --steps BCDER        # LLM(확인 프롬프트) → KG → 평가 → 리포트
python run_pipeline.py --k 30 --split TO --steps ER --seeds 2021,2022   # 평가만 이어서
```

### 중단과 재개

언제든 Ctrl+C(또는 창 닫기)로 멈추고 **같은 명령을 다시 실행**하면 이어서 진행된다.

| 멈춘 시점 | 다시 실행하면 |
|---|---|
| 준비(0·A·T·P) | 이미 만든 파일은 건너뜀 (`--force_prep`으로 재생성) |
| step1·step2 (LLM) | 캐시(50건마다 + Ctrl+C 시 즉시 저장)에 있는 프로필은 호출 안 함 → 남은 것만 호출. 창을 강제로 닫으면 마지막 저장 이후 최대 50건 정도만 다시 호출 |
| step1·step2·KG 완료 후 | `run_k{K}_{SPLIT}/.stage_{B,C,D}.json`(입력 내용 해시)로 통째로 건너뜀 (`--redo D` 등으로 강제) |
| 학습·평가 | 끝난 spec×seed 결과는 유지, 돌던 spec×seed 하나만 epoch 0부터 다시 |

주의: 실험 도중 `eval_slices.py`·`ikgr_core/model_ikgr.py`·`ikgr_core/corona_retriever.py`나 설정을 바꾸면,
이미 끝난 결과도 "입력이 바뀜"으로 보고 다시 학습한다(의도된 안전장치). 코드 업데이트는 실험 사이에 하자.

### 비용·시간 예상 (k=30, TO)

- 엔티티: 유저 약 62k, 아이템 약 28k → step1 ≈ 90k 호출, step2 ≈ 90k 호출(유저 프로필이 비어 있으면 호출 없음).
- 비용: gpt-4o-mini 정가 기준 약 $30대, 과거 관측 단가(k=30 신규 26,168 호출 ≈ $13–16)로는 최대 약 $100.
  정확한 값은 `S` 단계가 캐시를 반영해 출력한다(`run_k30_TO/llm_cost_estimate.json`). `--max_usd`로 상한 지정 가능.
  아이템 intent는 split과 무관하므로, TO를 먼저 돌리면 TO_GLOBAL의 step1 아이템 호출은 캐시로 재사용된다.
  더 싸게: k=50(약 51k 엔티티) 또는 k=100(약 18k 엔티티).
- GPU: spec×seed 1회에 대략 45–70분(조기종료 시점에 따라 다름) → 6 spec × 3 seed ≈ 15–20시간.
  `-Seeds 2020`으로 전 spec을 먼저 돌려 확인하고 나머지 seed를 붙이는 것을 권장.

## 3. 파이프라인

| 단계 | 스크립트 | 산출물 |
|---|---|---|
| 0 | `goodreads_preprocess.py` | `data/interactions.csv` (user, item, rating, timestamp) |
| A | `apply_k_core.py --no_profiles` | `data/kc_k{K}/interactions_k{K}.csv` |
| T | `make_temporal_inter.py` | `data/kc_k{K}/ikgr-custom/ikgr-custom.inter` (timestamp를 [2006, 2017]로 clip, **평가의 기준 파일**) |
| P | `build_trainonly_profiles.py` | `profiles_k{K}_{SPLIT}.csv`, `profile_provenance_k{K}_{SPLIT}.csv.gz` (+`.meta.json`) |
| S | `estimate_llm_cost.py` | `run_k{K}_{SPLIT}/llm_cost_estimate.json` |
| B | `step1.py` (LLM) | exact intents, `step1_cache.json` (프로필 텍스트 → intents) |
| C | `step2.py` (LLM + RAG) | related intents, `intent_vocab.json`, `intents_emb.npy` |
| D | `build_kg.py`, `build_meta_kg.py` | `kg_pack.pt`, `meta_kg_pack.pt` (메타 KG는 k-core 전체 아이템 대상) |
| E | `eval_slices.py` | `slice_eval_{SPLIT}_result.json`, `per_user_{SPLIT}/*.csv.gz` |
| R | `paired_compare.py --report` | `report_{SPLIT}.md` (paired CI + Pareto) |

`ikgr_core/temporal_split.py`가 RecBole 1.2.0의 split(float32 timestamp, stable 정렬, `_calcu_split_ids` 반올림)을
그대로 재현한다. 합성 데이터에서 실제 RecBole 결과와 행 단위로 일치함을 테스트로 확인했다.

## 4. 평가 프로토콜

- **Split**: `TO` = 유저별 시간순 80/10/10(모든 평가 유저가 train 이력 보유). `TO_GLOBAL` = 전역 시간 컷 70/10/20(cold 유저 포함).
- **학습/선택**: 위 공통 프로토콜(`config.yaml`의 `recbole:` 섹션, `IKGR_EPOCHS`/`IKGR_EVAL_STEP`/`IKGR_STOPPING_STEP`/`IKGR_VALID_METRIC`로 변경 가능). 결과 JSON에 `fit.best_epoch` 기록.
- **지표**: NDCG/Recall/Hit@{10,30}, tail Recall@{10,30}(train 인기도 상위 20% 컷 이하 아이템), Coverage@10, novelty(train 인기도 기준 self-information, train에 없던 아이템은 최대값).
- **cold 버킷**: train 상호작용 수 기준 분위 버킷 + 절대 버킷(0 / 1–5 / 6–20 / >20).
- **주요 spec**: `IKGR_kgoff`(MF), `BPR`, `LightGCN`, `IKGR_full_hetero`(intent+메타 KG), `IKGR_dyn_nointent`(메타+recency), `IKGR_dyn`(intent+메타+recency). 선택: `IKGR_meta_only`, `IKGR_cand_db`, `IKGR_full`, `IKGR_rerank_db_rel` 등.
- **재랭킹 대조군**: `{spec}_pop0p25`, `_pop0p5`, `_pop1p0` (같은 학습 모델에서 평가만 다르게).
- **유의성**: `python paired_compare.py --run_dir run_k30_TO --split TO --a IKGR_dyn --b BPR LightGCN IKGR_dyn_nointent --by_activity`

결과 해석 원칙: KG 계열이 long-tail/coverage를 올렸다고 말하려면 같은 NDCG 수준에서 **CF + 인기도 페널티 곡선보다 위**에 있어야 한다(`report_*.md`의 Pareto 표 ★/◆ 표시).

## 5. 테스트

```bash
python -m pytest -q tests/                        # split 재현(실제 RecBole 대조) + 프로필 train-only 검증 + 기존 rerank 테스트
python tests/smoke_e2e.py --workdir /tmp/ikgr_smoke   # 합성 데이터 · 가짜 LLM/인코더로 0→R 전 단계 + 누수 가드 음성 테스트
```

`tests/smoke_e2e.py`는 실제 API를 호출하지 않는다(`tests/fake_llm_server.py`, `tests/fakes/sentence_transformers`).
누수 가드가 (a) 테스트 기간 쌍이 섞인 provenance, (b) 다른 split용 프로필을 각각 거부하는지 확인한다.

## 6. 레거시 자료

- `run_k30/`, `run_k50/`의 결과 JSON, `IKGR_REPORT.md`, `DYNLLM_REPORT.md`, `CORONA_REPORT.md`, `AGENTS.md`의 표:
  레거시 프로필 프로토콜 + 12 epoch 고정 + train+valid 기반 슬라이스 정의. 기록용으로만 보관한다.
  특히 TO_GLOBAL cold0 "약 20–31배" 결과는 train 이력이 0인 유저가 train split 밖의 책(주로 검증/테스트 기간)으로 만든 프로필을 가졌던 결과라 폐기 대상이다.
- `run_pipeline_auto.py`, `step3.py`, `run_baselines.py`, `verify_*`, `run_rerank_orchestrator.py`: 이전 단계 도구(레거시 경로 `run/` 기준).

## 7. 구조

```
build_trainonly_profiles.py   train-only 유저 프로필 + provenance
ikgr_core/temporal_split.py   RecBole temporal split 재현
eval_slices.py                학습·평가·누수 가드·per-user 덤프·재랭킹 대조군
paired_compare.py             paired CI + Pareto 리포트
estimate_llm_cost.py          LLM 호출/비용 추정
run_pipeline.py               단계 오케스트레이터 (--k, --split)
scripts/run_goodreads_revalidation.ps1   Windows 원샷 실행
ikgr_core/model_ikgr.py       IKGR 모델 (intent/메타 KG, recency, late fusion)
ikgr_core/corona_retriever.py 그래프 후보생성 / soft prior
step1.py, step2.py            LLM intent 추출 / RAG 확장
tests/                        단위·스모크 테스트, 합성 데이터 생성기
```

데이터: Goodreads Children (Wan & McAuley, RecSys 2018; Wan et al., ACL 2019). 학술 용도 한정, 재배포 금지.
