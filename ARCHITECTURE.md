# 아키텍처 · 평가 프로토콜 요약 (다른 도메인 실험용 참고 문서)

Goodreads Children(k=30, per-user 시간순 TO 분할)에서 검증한 현재 구조와 평가 방식을 정리한 문서다.
Amazon Clothing 등 다른 도메인에 옮길 때 이 문서와 아래 파일들을 기준으로 삼는다. (2026-11-05 기준)

## 1. 먼저 알아야 할 버그 — LightGCN 베이스라인이 그래프를 안 쓰고 있었다

- RecBole 1.2.0 LightGCN은 `scipy.sparse.dok_matrix._update()`로 인접행렬을 만든다. SciPy ≥ 1.13에는 이 메서드가 없다.
- 예전 코드는 `dok_matrix._update = dict.update`로 대체했는데, SciPy 1.13부터 dok 값은 `self._dict`에 저장되므로
  **에러 없이 빈 인접행렬**이 만들어진다. 그 결과 "LightGCN"이 전파 없는 MF처럼 학습됐다.
- 수정: `eval_slices.py`, `run_baselines.py` 상단의 `_dok_update` (값을 `self._dict`에 쓴다).
- 검사: `python -m pytest -q tests/test_lgcn_backbone.py` (RecBole LightGCN 인접행렬 nnz > 0 확인 포함).
- **dataset-agnostic 파이프라인(`IKGR-main`)의 `eval_slices.py`에도 같은 대체 코드가 있다.** 그 파이프라인으로 돌린
  LightGCN 결과(Amazon/MovieLens 포함)는 다시 확인이 필요하고, 새로 돌리기 전에 같은 수정을 옮겨야 한다.

## 2. 모델 구조 (`ikgr_core/model_ikgr.py`, 클래스 `IKGR`)

점수 = ⟨u, i⟩ 내적, BPR loss. 사용자/아이템 표현은 **CF 백본 + 가산 facet**이다.

| 구성 | 설정 키 | 내용 |
|---|---|---|
| CF 백본 | `lgcn_layers` (0 = MF, 2 = LightGCN) | `lgcn_layers > 0`이면 **train 상호작용만으로 만든** 대칭 정규화 인접행렬로 RecBole LightGCN과 동일하게 전파(layer 평균). facet을 모두 끄면 점수가 RecBole LightGCN과 일치한다(테스트로 검증). |
| 메타데이터 KG (아이템) | `use_meta_kg`, `meta_kg_path` | 아이템 → 메타 노드(Goodreads: 저자·출판사·shelf) 임베딩 평균을 관계별 학습 가중치로 더한다. LLM 불필요. |
| recency 동적 프로필 (사용자) | `use_dynamic` | 사용자의 최근 train 상호작용 50개 아이템 표현의 시간감쇠(τ=180일) 가중합. train split만 사용. |
| LLM intent KG (선택) | `use_kg`, `kg_pack_path`, `intent_learnable: False` | LLM이 뽑은 사용자/아이템 intent 노드(mpnet 임베딩, 고정 랜덤 투영) 평균을 더한다. |

`eval_slices.py`의 spec 이름:

| spec | 구성 |
|---|---|
| `IKGR_lgcn_dyn_nointent` | **LightGCN 백본 + 메타 KG + recency** (현재 주력 모델) |
| `IKGR_lgcn_dyn` | 위 + LLM intent |
| `IKGR_dyn_nointent` / `IKGR_dyn` | MF 백본 버전 (intent 없음/있음) |
| `IKGR_kgoff` | MF only (ablation 기준선) |
| `BPR`, `LightGCN` | RecBole 베이스라인 |

### 메타 KG 파일 형식 (다른 도메인에 옮길 때)

`build_meta_kg.py`는 Goodreads 책 메타데이터 전용이다. 모델은 아래 키를 가진 `meta_kg_pack.pt`만 읽으므로,
다른 도메인은 같은 형식으로 만들면 된다(예: Amazon brand → `item_authors` 슬롯, category → `item_shelves` 슬롯).

```
{
  "item_authors":    {item_token(str): LongTensor[node ids]},   # 관계 1
  "item_publishers": {item_token(str): LongTensor[node ids]},   # 관계 2
  "item_shelves":    {item_token(str): LongTensor[node ids]},   # 관계 3
  "n_authors": int, "n_publishers": int, "n_shelves": int,
}
```
빈 관계는 그대로 두면 자동으로 건너뛴다. 아이템당 노드는 `kg_cap`(32)개까지 쓴다.

## 3. 평가 프로토콜 (모든 spec 동일, `config.yaml`의 `recbole` 섹션)

- **누수 방지:** 사용자 프로필(→ LLM intent)은 해당 분할의 train 구간에서만 만든다(`build_trainonly_profiles.py`).
  `eval_slices.py`가 매 실행 provenance ⊆ RecBole train split인지 검사한다(누수 가드).
- **학습:** `embedding_size: 64`, `learning_rate: 1e-3`, `reg_weight: 1e-6`, 최대 40 epoch, 2 epoch마다 검증,
  검증 NDCG@10이 3회 연속 안 오르면 정지, 검증 최고 체크포인트로 테스트.
  - 512차원에서는 진짜 LightGCN이 epoch당 ~24분(RTX 3060 Ti)이라 64로 낮췄다. 64차원 LightGCN은 epoch당 ~7분.
- **지표:** NDCG/Recall@10·30, train 인기도 기준 long-tail recall, coverage@10, novelty.
- **대조군:** 인기도 페널티 재랭킹 `s − λ·σ_u·z(log(1+pop_train))`, λ ∈ {0.25, 0.5, 1.0} (재학습 없음).
  "롱테일이 늘었다"는 주장은 이 대조군보다 나은지로 판단한다.
- **통계:** `paired_compare.py` — 사용자 단위 paired bootstrap 95% CI, Wilcoxon p, 활동도 구간별 차이.
- **재현성:** seed 고정 + deterministic 설정. 같은 seed 재실행 시 결과가 소수점까지 같다.

## 4. 현재 결과 (Goodreads k=30 TO, 64차원 — `run_k30_TO/`, 2026-11-09)

| 모델 | seeds | NDCG@10 | Recall@10 | tail R@10 | Coverage@10 | best epoch |
|---|---:|---:|---:|---:|---:|---:|
| LightGCN (수정본) | 2020, 2021 | 0.0878 / 0.0877 | 0.0909 / 0.0909 | 0.0104 / 0.0096 | 0.288 / 0.283 | 39 / 39 |
| **IKGR_lgcn_dyn_nointent** | 2020, 2021 | 0.0871 / 0.0859 | 0.0930 / 0.0923 | 0.0153 / 0.0147 | 0.462 / 0.456 | 39 / 37 |
| IKGR_lgcn_dyn (+ LLM intent) | 2020 | 0.0864 | 0.0927 | 0.0150 | 0.435 | 29 |
| LightGCN + 인기도 페널티 λ=0.25 | 2020, 2021 | 0.0813 / 0.0810 | 0.0848 / 0.0847 | 0.0151 / 0.0142 | 0.504 / 0.498 | 39 / 39 |

- **IKGR_lgcn_dyn_nointent vs LightGCN (2 seed, 사용자 단위 paired):** NDCG −1.4% (95% CI [−0.0020, −0.0004], 유의),
  tail recall +50% (CI [+0.0043, +0.0057]), coverage +61%. Recall@10은 두 seed 모두 약 +2%.
  seed 2020만으로는 NDCG 차이가 유의하지 않았지만(−0.8%) seed 2021(−2.1%)을 더하면 유의하게 낮다.
- **인기도 페널티 대조군:** LightGCN에 λ=0.25를 줘서 tail을 같은 수준(0.0147)으로 맞추면 NDCG가 0.0812로 떨어진다.
  IKGR_lgcn_dyn_nointent는 같은 tail(0.0150)을 NDCG 0.0865로 낸다(약 +6.5%). → 롱테일을 얻는 비용이 대조군보다 작다.
- **사용자별:** 저활동 사용자와 취향(intent)이 다양한 사용자에게 유리하고, 고활동 사용자에게 불리하다. 표는 4.1절.
- **LLM intent 추가 (seed 2020):** NDCG −0.8% (CI [−0.0014, −0.0001]), tail −2% (유의하지 않음) → 기여 없음.
  MF 백본(512차원, 3 seed)에서도 같은 방향이었다. intent spec은 seed 2021 이후로 돌리지 않았다.
- **한계:** seed 2개(2022는 TODO #4), LightGCN·IKGR 모두 40 epoch 상한 근처에서 최고점, 검증과 테스트의 순위가 다를 수 있어
  테스트로만 판단(검증 NDCG는 IKGR 0.112 vs LightGCN 0.101). seed 간 변동은 IKGR(0.0012)이 LightGCN(0.0001)보다 크다.
- 512차원 결과(전파 없는 LightGCN 포함)는 커밋 `e03f50a`에 있다. 64차원 결과와 비교하지 않는다.

### 4.1 사용자 구간별 효과 (TODO #7, 2 seed, 2026-11-09)

Δ = IKGR_lgcn_dyn_nointent − LightGCN이다. 사용자별로 두 seed(2020, 2021)를 평균한 뒤 비교했다.
각 특성으로 사용자를 5분위로 나눴다(Q1 = 낮음, Q5 = 높음).
`*` = 사용자 bootstrap 95% CI가 0을 포함하지 않음. 괄호는 LightGCN 대비 상대 변화다.
CI 전체와 seed별 값은 `run_k30_TO/user_segments_TO.md`에 있고, `python analyze_user_segments.py --run_dir run_k30_TO --split TO`로 다시 만든다(~1분, GPU 불필요).

| 특성 | 정의 (train 구간만 사용) | Q1 범위 → Q5 범위 |
|---|---|---|
| 활동량 | train 상호작용 수 | 24–30 → 89–7,137 |
| intent 다양성 | 1 − 사용자 intent 임베딩(mpnet) 간 평균 코사인 | 0.16–0.29 → 0.41–0.73 |
| 카테고리 다양성 | 사용자 책들의 shelf 분포 엔트로피(bit). 책 50% 이상에 붙는 shelf 5개(`childrens`, `fiction` 등)는 제외 | 2.98–5.00 → 5.75–7.37 |
| intent 수 | LLM이 뽑은 사용자 intent 개수 | 8–19 → 32–110 |

특성 간 상관(Spearman): 활동량–카테고리 다양성 0.52, intent 수–intent 다양성 −0.56, 나머지는 |ρ| ≤ 0.05다.
주력 모델은 intent를 입력으로 쓰지 않으므로, intent 특성은 LLM이 train 프로필을 읽고 판단한 사용자 성향 지표로만 쓰였다.

**표 1. NDCG@10 Δ (사용자 62,142명)**

| 구간 | 활동량 | intent 다양성 | 카테고리 다양성 | intent 수 |
|---|---:|---:|---:|---:|
| Q1 (낮음) | +0.0016 (+1.5%) | −0.0036\* (−4.1%) | −0.0009 (−0.7%) | −0.0011 (−1.3%) |
| Q2 | +0.0014 (+1.4%) | −0.0015 (−1.8%) | −0.0012 (−1.2%) | +0.0002 (+0.2%) |
| Q3 | −0.0006 (−0.7%) | −0.0008 (−0.9%) | +0.0002 (+0.2%) | −0.0008 (−0.9%) |
| Q4 | −0.0027\* (−3.6%) | +0.0002 (+0.2%) | −0.0017\* (−2.4%) | −0.0002 (−0.2%) |
| Q5 (높음) | −0.0056\* (−7.7%) | +0.0000 (0.0%) | −0.0023\* (−4.7%) | −0.0038\* (−4.2%) |

**표 2. tail Recall@10 Δ (테스트에 롱테일 아이템이 있는 사용자 37,746명; 분위 경계는 이 사용자들 기준)**

| 구간 | 활동량 | intent 다양성 | 카테고리 다양성 | intent 수 |
|---|---:|---:|---:|---:|
| Q1 (낮음) | +0.0127\* (+61%) | +0.0037\* (+70%) | +0.0089\* (+30%) | +0.0045\* (+37%) |
| Q2 | +0.0064\* (+43%) | +0.0035\* (+63%) | +0.0067\* (+81%) | +0.0054\* (+49%) |
| Q3 | +0.0031\* (+36%) | +0.0051\* (+65%) | +0.0050\* (+79%) | +0.0051\* (+64%) |
| Q4 | +0.0022\* (+60%) | +0.0042\* (+37%) | +0.0030\* (+71%) | +0.0040\* (+47%) |
| Q5 (높음) | +0.0006\* (+27%) | +0.0083\* (+43%) | +0.0015\* (+107%) | +0.0057\* (+59%) |

**표 3. 활동량을 고정했을 때의 효과**

- 중앙값 분할: 활동량 5분위 각각 안에서 특성 상위 절반의 Δ에서 하위 절반의 Δ를 뺀 값을 5구간 평균한 것.
- OLS: 사용자별 Δ를 log 활동량과 나머지 세 특성(표준화)에 회귀한 계수. 특성이 +1 SD 늘 때 Δ의 변화다.

| 특성 | NDCG@10 중앙값 분할 | NDCG@10 OLS | tail R@10 중앙값 분할 | tail R@10 OLS |
|---|---:|---:|---:|---:|
| intent 다양성 | +0.0021\* [+0.0003, +0.0036] | +0.0007 [−0.0003, +0.0017] | +0.0013 [−0.0002, +0.0028] | +0.0021\* [+0.0010, +0.0030] |
| 카테고리 다양성 | +0.0013 [−0.0003, +0.0030] | +0.0013\* [+0.0002, +0.0024] | −0.0004 [−0.0018, +0.0010] | −0.0007 [−0.0018, +0.0005] |
| intent 수 | −0.0011 [−0.0027, +0.0005] | −0.0008 [−0.0016, +0.0002] | −0.0006 [−0.0022, +0.0007] | +0.0012\* [+0.0004, +0.0019] |
| log 활동량 | — | −0.0034\* [−0.0042, −0.0025] | — | −0.0031\* [−0.0039, −0.0023] |

**표 4. LLM intent 채널의 효과 (IKGR_lgcn_dyn − IKGR_lgcn_dyn_nointent, seed 2020, NDCG@10 Δ)**

| 구간 | intent 수 | intent 다양성 |
|---|---:|---:|
| Q1 (낮음) | −0.0015\* | −0.0005 |
| Q2 | −0.0006 | −0.0003 |
| Q3 | −0.0010 | −0.0007 |
| Q4 | −0.0007 | −0.0008 |
| Q5 (높음) | +0.0001 | −0.0015 (CI 상한 0.0000) |

**해석**

- **활동량이 가장 확실한 요인이다.**
  - 하위 40%에서 IKGR이 NDCG +1.4~1.5%로 앞선다. 두 seed가 같은 방향이지만 구간별 CI는 0을 포함한다.
  - 상위 40%에서는 유의하게 낮다(−3.6%, −7.7%). OLS에서도 가장 큰 계수다.
  - tail 이득의 절대 크기는 저활동 사용자에게 몰려 있다(Q1 +0.0127).
- **intent 다양성:** intent가 한 방향으로 몰린 사용자(Q1)는 NDCG −4.1%로 손해이고, 다양한 사용자(Q4–Q5)는 LightGCN과 동률이다.
  - 활동량을 고정한 중앙값 분할에서도 유의하다(+0.0021).
  - OLS에서는 유의하지 않다. intent 수와 상관(ρ=−0.56)이 있어 효과가 나뉘기 때문으로 보인다. 방향은 일관되고 강도는 중간이다.
- **카테고리 다양성:** 표 1에서 Q4–Q5가 손해로 보이는 것은 활동량과의 상관(ρ=0.52) 때문이다.
  - 활동량을 고정하면 부호가 +로 바뀐다. 다만 OLS에서만 유의하므로 약한 근거다.
- **tail:** 모든 구간에서 유의하게 늘었다.
  - 절대 이득은 intent가 다양한 사용자에서 크다(Q5 +0.0083 vs Q1 +0.0037).
  - 상대(%) 이득은 반대로 Q1(+70%)이 Q5(+43%)보다 크다. LightGCN의 tail 기준값이 다양한 사용자에서 훨씬 높기 때문이다(0.0193 vs 0.0053).
    그래서 "다양한 사용자일수록 tail 이득이 크다"는 절대값 기준으로만 말할 수 있다.
- **LLM intent 채널:** intent가 많거나 다양한 사용자 구간에서도 NDCG를 올리지 못했다(표 4). "intent가 풍부한 사용자에게는 도움이 된다"는 가설도 성립하지 않는다.
- **한계:** 표 여러 개(특성 4개 × 5구간)를 동시에 봤고 다중비교 보정은 하지 않았다. 활동량 외에 CI 끝이 0 근처인 효과는 seed 2022(TODO #4)로 다시 확인해야 한다.

## 5. 실행

### 데이터 (API 호출 없이 바로 실행)

`data_bundle/`에 우리가 실제로 쓴 입력 데이터가 들어 있다(gzip 압축 후 100MB 미만 7조각, 약 610MB).
풀면 `data/kc_k30/`(k-core 상호작용, RecBole `.inter`, train-only 프로필 + provenance)와
`run_k30_TO/kg_pack.pt`(LLM intent KG), `run_k30_TO/meta_kg_pack.pt`(메타데이터 KG)가 생긴다.
LLM intent는 이미 들어 있으므로 **intent spec까지 API 키 없이** 돌릴 수 있다.

```powershell
python scripts/unpack_data_bundle.py      # sha256 확인 후 압축 해제 (약 1.4GB)
```

### 학습·평가

```powershell
# Windows, venv: scripts\setup_venv.ps1 로 생성
$env:IKGR_POP_RERANK_SPECS = "IKGR_kgoff,BPR,LightGCN,IKGR_dyn,IKGR_dyn_nointent,IKGR_lgcn_dyn_nointent,IKGR_lgcn_dyn"
python run_pipeline.py --k 30 --split TO --steps ER --seeds 2022 --specs IKGR_lgcn_dyn_nointent,LightGCN
```
- `--steps ER`: 번들을 푼 상태에서는 학습·평가(E)와 리포트(R)만 돌리면 된다. LLM 단계(B/C)와 KG 빌드(D)는 건너뛴다.
- 결과: `run_k30_TO/slice_eval_TO_result.json`, `run_k30_TO/per_user_TO/`, `run_k30_TO/report_TO.md`.
  한 seed 안에서 비교하는 모델들은 같은 PC에서 돌린다(GPU가 다르면 같은 seed라도 수치가 조금 다를 수 있다).
- 단계: B(step1 LLM intent) / C(step2 RAG) / D(KG 빌드) / E(학습·평가) / R(paired 리포트). 끝난 단계·모델×seed는 재실행 시 건너뛴다.
- intent를 쓰지 않는 spec(`*_nointent`, `LightGCN`, `BPR`)은 LLM 출력을 쓰지 않는다. 다만 D 단계의 intent KG 빌드는
  step2 출력이 필요하므로, LLM 없이 돌리려면 메타 KG만 `build_meta_kg.py --inter <RecBole .inter> --out <run_dir>/meta_kg_pack.pt`
  형식으로 만든 뒤 `--steps E`로 평가 단계만 실행한다.
- 테스트: `python -m pytest -q tests/`, `python tests/smoke_e2e.py --workdir <tmp>` (가짜 LLM, GPU 불필요).
