# repair_exp — 궤적 수리(Trajectory Repair) 확장 실험

`main_exp`(논문 메인 실험 재현 코드)를 통째로 복사한 뒤, 확장 실험용 코드를 얹은 폴더입니다.
0장은 새로 추가한 궤적 수리 실험을, 1장부터는 기반이 된 main_exp의 내용(A/B/C/D 재현)을
그대로 담고 있습니다 — 궤적 수리는 (C)/(D)가 찾아낸 결정적 오류 지점을 입력으로 쓰므로,
1~6장의 파이프라인이 선행되어야 합니다.

---

## 0. 확장 실험: 궤적 수리 및 복구율(Recovery Rate)

### 0.1 아이디어

(C)/(D) 방법론이 찾아낸 **결정적 오류 단계(s\*)** 에서 그 단계의 출력을 교정하고,
그 이후 궤적 전체를 다시 생성했을 때 과제가 실제로 성공으로 전환되는지를 측정합니다.

```
원본 실패 궤적:  Step0 ... Step(s*-1)  [Step s* 오류]  Step(s*+1) ... StepN → 실패
수리 후 궤적:    Step0 ... Step(s*-1)  [Step s* 교정]  Step(s*+1)' ... StepN' → 성공/실패 판정
```

**복구율(recovery rate) = 수리 후 성공으로 판정된 파일 수 / 시도한 파일 수**

### 0.2 정답(ground truth) 누설 방지 — 기본값은 아예 W/O GT

처음엔 "GT를 보고 사유를 설명하되 정답만 지워서 넘기자"는 방식(프롬프트 지시 + 검증 게이트)으로
설계했는데, 그 경우 게이트가 뚫릴 잔여 위험이 항상 남습니다. 그래서 **기본값 자체를 아예
GT를 전혀 안 쓰는 파이프라인으로 바꿨습니다** — 게이트가 필요 없어지므로 훨씬 단순해집니다.
GT-informed 버전은 `--use_gt_reason` 플래그로 켜는 선택적 2차 실험(ablation)으로만 남겨뒀습니다.

| 단계 | 기본값 (GT-free) | `--use_gt_reason` 켰을 때 |
|---|---|---|
| 귀인(Phase 1, 기존 C/D) | ✅ GT 사용 | 동일 (main_exp의 정상 동작 — 귀인은 원래 GT로 판단) |
| s\* 이전 궤적(0..s\*-1) 맥락 | GT 아님(원본 로그 재사용) | 동일 |
| "왜 틀렸는지" 사유 | **생성 자체를 안 함** — `repair_step`에 "이 지점이 결정적 오류로 지목됐다"는 사실만 전달, 진단은 모델이 스스로 | GT로 사유를 생성하되, 하드 게이트(0.2.1 참고) 통과분만 전달 |
| s\* 교정(repair_step) | GT 없음. query + 이전 맥락 + "여기가 의심 지점"이라는 사실만 입력 | GT 없음. 대신 게이트 통과한 사유가 추가로 입력됨 |
| s\* 이후 궤적 재생성 | GT 없음, **원본 s\*+1.. 내용도 안 줌** | 동일 |
| 검증(Verifier) | ✅ GT 사용 (채점 목적, evaluate.py와 같은 성격) | 동일 |

**기본값(GT-free)에서는 파이프라인 전체에서 GT가 쓰이는 곳이 귀인과 채점 딱 두 곳뿐**이고,
그 사이(교정·재생성)에는 GT가 들어갈 통로 자체가 존재하지 않습니다 — 필터링이 아니라 애초에
안 넣는 것이므로 "게이트가 뚫릴 가능성"이라는 논의 자체가 성립하지 않습니다. 대신 모델이
받는 힌트가 "여기 어딘가 문제가 있다"는 위치 정보뿐이라, `--use_gt_reason` 버전보다 복구율은
더 낮게 나올 가능성이 높습니다 — 이건 결함이 아니라 **"정확한 위치 지정(귀인)만으로 얼마나
복구가 되는가"를 순수하게 측정**하는 것이라 오히려 논문 스토리에는 더 깔끔한 지표입니다.

#### 0.2.1 (선택) `--use_gt_reason`을 켰을 때의 하드 게이트

GT-informed 사유를 2차 실험으로 보고 싶을 때만 이 경로를 씁니다. GT가 실제로 들어오는 곳은
사유 생성 한 곳뿐이고, 그 사유가 곧바로 `repair_step`으로 흘러가므로, 생성 직후 다음 게이트를
통과한 텍스트만 넘깁니다:

1. 사유 생성
2. **정규식으로 정답 문자열이 그대로 포함됐는지 검사** (`_contains_ground_truth`)
3. 통과해도 **별도 LLM judge에게 "이 문장으로 정답을 유추할 수 있는가"를 재확인** (`_judge_leaks_answer`)
4. 둘 중 하나라도 걸리면 "방금 정답을 흘렸다, 절차적 결함만 다시 설명하라"는 경고를 추가해 최대 3회 재시도
5. **3회 모두 실패하면**, 어떤 모델 출력도 아닌 고정 템플릿 문장(`_safe_fallback_reason`)으로 강제 대체 — 이 문장은 GT에서 유도된 정보가 전혀 없어 leak이 구조적으로 불가능

기본값이든 `--use_gt_reason`이든 공통으로, s\* 이후 단계 재생성 시 같은 에이전트가 원래 그
뒤에 뭐라고 말했는지는 절대 보여주지 않고 오직 새로 생성된 내용만 누적해서 다음 단계의
맥락으로 사용합니다.

### 0.3 단순화한 부분 (특허 문서와의 차이)

특허 명세서 초안에 있는 아이디어(양방향 추적 기반 귀인 확장, 동적 오케스트레이터의 에이전트 역할
신규 정의, 재실행 가능/불가 단계 구분과 부작용 방지, 실패 시 상류로 재분류하는 폐루프) 중
논문 확장 실험에 필수적이지 않은 부분은 반영하지 않았습니다:
- 귀인(Phase 1)은 특허의 새 역방향 추적이 아니라 **main_exp에 이미 구현된 (C)/(D) 결과를 그대로 재사용**합니다.
- 하류 재생성은 동적 오케스트레이터가 아니라 **원본 로그의 에이전트 순서를 그대로 따라가며 내용만 새로 생성**합니다 (누가 다음에 말할지는 구조적 정보이지 "정답"이 아니라고 보고, 순서 자체는 재사용).
- 폐루프(실패 시 s\*를 상속점으로 재분류해 상류로 재시도)는 없습니다 — 1회 수리·1회 판정만 수행합니다. 필요하면 나중에 별도 실험으로 얹을 수 있습니다.
- 이 데이터셋(Algorithm-Generated)은 이미 종료된 로그이므로 실제 도구(웹 브라우저 등)를 다시 실행할 수 없습니다. 하류 재생성은 전부 **LLM이 원래 에이전트 역할을 롤플레이하며 텍스트를 새로 만드는 시뮬레이션**이며, 실제 재실행이 아닙니다.

### 0.4 새로 추가된 파일

```
Automated_FA/
├── repair.py                # 궤적 수리 실험 진입점
├── Lib/repair_utils.py      # (기본 GT-free) 교정 → 하류 재생성 → 검증, (선택) GT-informed 사유+게이트
├── score_repair.py          # results_*.jsonl → 복구율 집계
└── outputs_repair/          # repair.py 실행 로그 + 채점용 results_*.jsonl (실행 후 생성됨)
```

### 0.5 실행 순서 (GT-free 기준, 전부 `repair_exp/Automated_FA/`에서 실행)

귀인(Phase 1, C/D)은 원래도 GT를 쓰는 게 정상 설계이므로 그대로 두고, 그 결과를 OpenRouter로
새로 뽑아 GT-free 수리의 입력으로 씁니다. 커밋된 `outputs/reference/` 로그(GPT-4o 직접 호출,
윈도우 3)를 그대로 써도 되지만, 아래는 이 폴더의 OpenRouter 파이프라인으로 처음부터 재현하는
순서입니다.

**0) 사전 확인**
```bash
cd repair_exp/Automated_FA
cat .env   # OPENROUTER_API_KEY 확인 (main_exp에서 복사되어 있음)
```
윈도우 크기를 논문 메인 실험(Table 3)과 맞추려면 `Lib/utils.py`의 `FINAL_WINDOW_RADIUS`를
5 → 3으로 변경 (선택 사항, §1.3 참고).

**1) 오류 귀인 재수행 (Phase 1, C/D)** — 여기는 GT 사용이 정상
```bash
python inference.py --method all_at_once --model openai/gpt-4o --two_stage_window \
    --directory_path ../Who_and_When/Algorithm-Generated --is_handcrafted False
python inference.py --method step_by_step --model openai/gpt-4o --two_stage_window \
    --directory_path ../Who_and_When/Algorithm-Generated --is_handcrafted False
```
→ `outputs/all_at_once_openai_gpt-4o_alg_generated.txt` (A+C), `outputs/step_by_step_openai_gpt-4o_alg_generated.txt` (B+D)

**2) (선택) 귀인 품질 확인**
```bash
DATA=../Who_and_When/Algorithm-Generated
python evaluate.py --data_path $DATA --eval_file outputs/all_at_once_openai_gpt-4o_alg_generated.txt   # (C)
python evaluate.py --data_path $DATA --eval_file outputs/step_by_step_openai_gpt-4o_alg_generated.txt  # (D)
```

**3) 궤적 수리 (GT-free, 기본값 — 플래그 없음)** — (D) 기준 예시, (C)도 `--attribution_log`만 바꾸면 동일
```bash
# 스모크 테스트 (5건)
python repair.py --pivot_source attribution_log \
    --attribution_log outputs/step_by_step_openai_gpt-4o_alg_generated.txt \
    --directory_path ../Who_and_When/Algorithm-Generated \
    --model openai/gpt-4o --limit 5

# 전체 실행
python repair.py --pivot_source attribution_log \
    --attribution_log outputs/step_by_step_openai_gpt-4o_alg_generated.txt \
    --directory_path ../Who_and_When/Algorithm-Generated \
    --model openai/gpt-4o
```

**4) 비교군 (oracle 상한선 / random 대조군)** — 동일하게 GT-free
```bash
python repair.py --pivot_source oracle --model openai/gpt-4o
python repair.py --pivot_source random --model openai/gpt-4o
```

**5) 복구율(수리율) 비교 — 귀인 정답/오답별 breakdown 포함**
```bash
python score_repair.py --results_file \
    outputs_repair/results_openai_gpt-4o_attribution_log.jsonl \
    outputs_repair/results_openai_gpt-4o_oracle.jsonl \
    outputs_repair/results_openai_gpt-4o_random.jsonl
```
제안 방법론(attribution_log)의 복구율이 random보다 유의하게 높고 oracle에 근접할수록,
"정확한 귀인 → 실제 수리 성공"이라는 인과관계를 뒷받침하는 근거가 됩니다.

`score_repair.py`는 각 pivot_source의 전체 복구율뿐 아니라, **s\*가 데이터셋 라벨(mistake_step)과
정확히 일치했는지에 따른 breakdown**도 자동으로 함께 보여줍니다 (`--no_breakdown`으로 끌 수 있음):

```
outputs_repair/results_openai_gpt-4o_attribution_log.jsonl
  [attribution_log] overall                                126      58          46.03%
    attribution correct (s* == mistake_step)                 82      46          56.10%
    attribution incorrect (s* != mistake_step)                44      12          27.27%
```

"귀인이 정확히 맞았을 때의 수리율"과 "틀렸을 때의 수리율"을 이렇게 바로 비교할 수 있습니다 —
`repair.py`가 파일마다 `mistake_step`/`attribution_correct`를 `results_*.jsonl`에 함께 기록해두기
때문에 재계산 없이 바로 나옵니다 (`Lib/repair_utils.py`의 `attribution_is_correct()`).

> **비용/시간 팁**: 1번은 파일 126건 × 2회 호출, 3~4번은 pivot_source당 파일당 여러 호출
> (교정 + 하류 재생성 + 검증)이 발생합니다. 전체를 돌리기 전에 3~4번은 `--limit 5`로 먼저
> 검증하는 걸 권장합니다.

**(선택) 2차 실험 — GT-informed 사유 버전과 비교**
```bash
python repair.py --pivot_source attribution_log \
    --attribution_log outputs/step_by_step_openai_gpt-4o_alg_generated.txt \
    --directory_path ../Who_and_When/Algorithm-Generated \
    --model openai/gpt-4o --use_gt_reason
```
결과 파일명에 `_gtreason`이 자동으로 붙어 GT-free 실행과 구분됩니다
(예: `results_openai_gpt-4o_attribution_log_gtreason.jsonl`).

> **비용 참고**: 기본값(GT-free)은 파일당 대략 (교정 1회 + 하류 재생성 N회 + 최종 검증 1회)만큼
> LLM 호출이 발생합니다 (N ≈ 원본 궤적에서 s\* 이후 남은 스텝 수, 보통 2~5). `--use_gt_reason`을
> 켜면 사유 설명 1~3회 × 누설검증 judge 1~3회가 추가됩니다(대부분 1회차에 통과). 126건 전체를
> 3가지 pivot_source로 돌리면 호출량이 꽤 커지므로, 먼저 `--limit`으로 소규모 검증을 권장합니다.

### 0.6 여러 모델로 비교 실험 (Claude Haiku / Gemini Flash / GPT-5 mini 등)

`--model`은 OpenRouter 모델 ID를 그대로 받으므로, 위 1~5단계를 모델만 바꿔서 반복하면 됩니다.
**정확한 슬러그(slug)는 모델이 계속 추가/변경되니 실행 전에 https://openrouter.ai/models 에서
직접 검색해 확인하세요** — 아래는 명명 패턴 기준 추정치이며 실제와 다를 수 있습니다:

```bash
# 예시 (정확한 slug는 openrouter.ai/models에서 확인 후 교체)
MODEL=anthropic/claude-haiku-4.5      # Claude Haiku 계열
MODEL=google/gemini-3.0-flash         # Gemini Flash 계열 (버전 표기 확인 필요)
MODEL=openai/gpt-5-mini               # GPT-5 mini

# 1) 귀인 재수행 → 2) (선택) 확인 → 3) 수리(GT-free) → 4) 비교군 → 5) 채점, 모델만 교체해서 반복
python inference.py --method all_at_once --model $MODEL --two_stage_window \
    --directory_path ../Who_and_When/Algorithm-Generated --is_handcrafted False
python inference.py --method step_by_step --model $MODEL --two_stage_window \
    --directory_path ../Who_and_When/Algorithm-Generated --is_handcrafted False

python repair.py --pivot_source attribution_log \
    --attribution_log "outputs/step_by_step_$(echo $MODEL | tr '/' '_')_alg_generated.txt" \
    --directory_path ../Who_and_When/Algorithm-Generated \
    --model $MODEL --limit 5   # 스모크 테스트 먼저

python repair.py --pivot_source attribution_log \
    --attribution_log "outputs/step_by_step_$(echo $MODEL | tr '/' '_')_alg_generated.txt" \
    --directory_path ../Who_and_When/Algorithm-Generated \
    --model $MODEL
python repair.py --pivot_source oracle --model $MODEL
python repair.py --pivot_source random --model $MODEL

python score_repair.py --results_file outputs_repair/results_$(echo $MODEL | tr '/' '_')_*.jsonl
```

세 모델 각각 이 순서를 돌리면 `outputs_repair/`에 모델별 `results_<model>_*.jsonl`이 쌓이므로,
`score_repair.py`에 여러 모델의 파일을 한 번에 넘겨 모델 간 복구율까지 나란히 비교할 수 있습니다.

> **주의**: 귀인(1번)에 사용하는 모델과 수리(3~4번)에 사용하는 모델을 다르게 섞는 것도 가능합니다
> (예: GPT-4o로 귀인 → Claude Haiku로 수리). 다만 `--model`은 두 단계에 각각 독립적으로 넘기는
> 값이니, 실험 설계상 "같은 모델로 귀인+수리를 통일"할지 "귀인은 논문 기준(GPT-4o)으로 고정하고
> 수리 모델만 바꿔가며 비교"할지는 명확히 정하고 실행하는 걸 권장합니다.

---

## 1. main_exp 기반 파이프라인 (궤적 수리의 전제 조건)

궤적 수리는 (C)/(D)가 만든 귀인 로그를 입력으로 쓰므로, 아래는 main_exp 그대로의 내용입니다
— 이미 main_exp에서 실행해 봤다면 건너뛰고 0장으로 돌아가도 됩니다.

| 프로세스 | 방법 | 설명 |
|---|---|---|
| (A) | All-at-Once | 전통적 전역 분석 (baseline) |
| (B) | Step-by-Step | 전통적 순차 분석 (baseline) |
| (C) | (A) + Window | (A) 결과를 pivot으로 윈도우 증강 + 윈도우 집중 식별 |
| (D) | (B) + Window | (B) 결과를 pivot으로 윈도우 증강 + 윈도우 집중 식별 |

한 번의 `--two_stage_window` 실행이 **1-stage 예측(= A/B)** 과 **윈도우 재식별(= C/D)** 을
같은 로그 파일에 함께 기록하므로, 실제 추론은 **2회**(all_at_once, step_by_step)만 하면 됩니다.

### 1.0 논문 실험 결과 위치 (Window-Augmentation... .pdf 기준)

| 구분 | 논문 위치 | 내용 | 이 폴더로 재현 가능? |
|---|---|---|---|
| **Table 2** | p.62, §IV.1 Experiment Overview | H/W·S/W 실험 환경 (Python 3.12.8, OpenAI SDK 1.59.3, GPT-4o) | 참고용 |
| **Fig. 8** | p.62, §IV.1 | 전체 성능 평가 프로세스 (A~D 흐름도) | 이 폴더의 파이프라인과 동일 |
| **Table 3** | p.62, §IV.2 Experimental Results and Analysis | **메인 결과**: (A)(B)(C)(D) Step/Agent Accuracy, `Algorithm-Generated` 126건, GPT-4o, 윈도우 크기 3 | ✅ 이 폴더의 재현 대상 |
| **Fig. 9** | p.62–63, §IV.3 Performance Variation With Window Sizes | 윈도우 크기 1~5에 따른 (C)/(D) 성능 변화 (크기 3에서 최고 성능) | ✅ `FINAL_WINDOW_RADIUS` 를 1~5로 바꿔가며 재현 (§3 참고) |
| **Table 4** | p.63, §IV.4 Generalization and Robustness Analysis | 모델을 GPT-4o-mini로 교체한 (A)(B)(C)(D) 결과 (일반화 검증) | ✅ `--model openai/gpt-4o-mini` 로 재현 |
| **Table 5** | p.63, §IV.4 | `Hand-Crafted` 데이터 58건(GPT-4o-mini, 윈도우 크기 5) 결과 (견고성 검증) | ⚠️ 데이터 미포함 — 아래 "Table 5 재현" 참고 |

> 논문 본문에는 Binary Search / Chunk-Parallel 방법론이 등장하지 않습니다. 이 폴더에서 해당
> 코드를 제거한 것은 논문 내용과 정확히 일치합니다.

---

### 1.1 폴더 구조

```
repair_exp/
├── README.md
├── Who_and_When/
│   └── Algorithm-Generated/                     # 데이터셋 126건 (원본 그대로)
└── Automated_FA/
    ├── inference.py                             # 귀인 추론 진입점 (OpenRouter 호출) — main_exp와 동일
    ├── evaluate_base.py                         # (A), (B) 채점
    ├── evaluate.py                              # (C), (D) 채점
    ├── evaluate_1stage.py                       # (C)/(D) 대체 파서 (포맷 자동 감지)
    ├── evaluate_1stage_alg.py                   # (C)/(D) 대체 파서 (에이전트명 정규화 포함)
    ├── repair.py                                # [신규] 궤적 수리 실험 진입점 (0장 참고)
    ├── score_repair.py                          # [신규] 복구율 집계
    ├── Lib/
    │   ├── utils.py                             # A/B/C/D 방법론 본체 — main_exp와 동일
    │   └── repair_utils.py                      # [신규] 사유 설명 → 교정 → 하류 재생성 → 검증
    ├── outputs/
    │   └── reference/                           # 논문에 쓰인 원본 실행 로그 (윈도우 크기 3)
    │       ├── all_at_once_gpt-4o_alg_generated (WIN3).txt
    │       └── step_by_step_gpt-4o_alg_generated (WIN=3).txt
    └── outputs_repair/                          # [신규] repair.py 실행 로그 + results_*.jsonl (실행 후 생성)
```

### 파일 출처

| 파일 | 원본 위치 |
|---|---|
| `Automated_FA/inference.py`, `evaluate*.py`, `Lib/utils.py` | `KTserverbackup/Automated_FA/` (Binary Search / Chunk-Parallel 코드 제거, OpenRouter 연동으로 수정) |
| `Who_and_When/Algorithm-Generated/` | `KTserverbackup/Who_and_When/Algorithm-Generated/` (Elice 백업과 바이트 동일, 무수정) |
| `outputs/reference/all_at_once_gpt-4o_alg_generated (WIN3).txt` | `KTserverbackup/Automated_FA/outputs/[FINAL]all-at-once/Algorithm-Generated/` (무수정) |
| `outputs/reference/step_by_step_gpt-4o_alg_generated (WIN=3).txt` | `KTserverbackup/Automated_FA/outputs/[FINAL]step-by-step/Algorithm-generated/` (무수정) |

> `inference.py`의 `--method` 선택지는 `all_at_once` / `step_by_step` 두 가지만 남겼습니다
> (원본에는 이 실험과 무관한 `binary_search`, `chunk_parallel` 옵션도 있었습니다). 로컬 GPU에서
> Llama/Qwen을 돌리던 `Lib/local_model.py` 도 제거했습니다 — 논문의 모든 실험(Table 3/4/5)은
> GPT-4o / GPT-4o-mini만 사용했고, 필요하면 OpenRouter로 같은 모델 계열도 그대로 호출할 수 있습니다.

---

### 1.2 환경 (OpenRouter + .env)

- Python 3.12 (논문: 3.12.8)
- 필수 패키지: `openai`, `python-dotenv`, `tqdm`

```bash
pip install openai python-dotenv tqdm
```

- `inference.py`는 [OpenRouter](https://openrouter.ai/)의 OpenAI 호환 엔드포인트를 사용합니다.
  `--model`에 OpenRouter 모델 ID(예: `openai/gpt-4o`, `openai/gpt-4o-mini`,
  `anthropic/claude-3.5-sonnet`, `meta-llama/llama-3.1-70b-instruct`)를 넘기면 되며,
  전체 목록은 https://openrouter.ai/models 참고.
- `repair_exp/Automated_FA/` (또는 상위 디렉터리)에 `.env` 파일을 두고 아래 값을 설정하세요.
  (main_exp에서 이미 설정해뒀다면, 폴더 복사 시 `.env`도 함께 복사되어 그대로 사용됩니다.)

```dotenv
# repair_exp/Automated_FA/.env
OPENROUTER_API_KEY=sk-or-v1-...

# 선택: 기본값은 https://openrouter.ai/api/v1
# OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# 선택: OpenRouter 랭킹에 앱을 표시하기 위한 헤더 (https://openrouter.ai/docs)
# OPENROUTER_SITE_URL=https://your-app-or-repo-url
# OPENROUTER_SITE_NAME=your-app-name
```

`.env` 대신 각 명령에 `--api_key ...` / `--base_url ...` 를 직접 전달할 수도 있습니다
(CLI 인자가 `.env` 값보다 우선합니다).

---

### 1.3 윈도우 크기 설정 (중요)

`Automated_FA/Lib/utils.py`의 `FINAL_WINDOW_RADIUS`:

```python
FINAL_WINDOW_RADIUS = 5   # ← 원본 그대로 두었음
```

논문 §4.1 은 메인 실험(Table 3)에서 **윈도우 증강 범위를 3** 으로 고정합니다.
따라서 **Table 3 을 재현하려면 이 값을 `3` 으로 바꾼 뒤** 추론을 실행하세요.
(윈도우 크기 분석 Fig. 9 는 이 값을 1~5 로 바꿔가며 반복 실행한 결과입니다.
 `outputs/reference/` 의 로그는 이미 3 으로 실행된 것입니다. Table 5(Hand-Crafted)는 5로 실행됩니다.)

---

### 1.4 추론 실행

작업 디렉터리: `repair_exp/Automated_FA/`

```bash
cd repair_exp/Automated_FA

# (A) + (C) : all-at-once  및  all-at-once + window
python inference.py --method all_at_once --model openai/gpt-4o --two_stage_window \
    --directory_path ../Who_and_When/Algorithm-Generated --is_handcrafted False

# (B) + (D) : step-by-step  및  step-by-step + window
python inference.py --method step_by_step --model openai/gpt-4o --two_stage_window \
    --directory_path ../Who_and_When/Algorithm-Generated --is_handcrafted False
```

`--model`만 바꾸면 다른 LLM으로도 동일한 파이프라인을 재현할 수 있습니다. 예를 들어
논문 Table 4(일반화 검증)를 재현하려면:

```bash
python inference.py --method all_at_once --model openai/gpt-4o-mini --two_stage_window \
    --directory_path ../Who_and_When/Algorithm-Generated --is_handcrafted False
python inference.py --method step_by_step --model openai/gpt-4o-mini --two_stage_window \
    --directory_path ../Who_and_When/Algorithm-Generated --is_handcrafted False
```

결과 로그 저장 위치 (파일명은 `--model`의 `/`를 `_`로 치환하여 생성됩니다):

```
Automated_FA/outputs/all_at_once_openai_gpt-4o_alg_generated.txt
Automated_FA/outputs/step_by_step_openai_gpt-4o_alg_generated.txt
```

각 로그에는 `Prediction for X.json:` (1-stage = A/B) 블록과
`=== Final Prediction for X.json ===` (window = C/D) 블록이 함께 들어갑니다.

---

### 1.5 채점

```bash
cd repair_exp/Automated_FA
DATA=../Who_and_When/Algorithm-Generated
LOG_A=outputs/all_at_once_openai_gpt-4o_alg_generated.txt
LOG_B=outputs/step_by_step_openai_gpt-4o_alg_generated.txt

# (A)  all-at-once 로그의 1-stage 블록
python evaluate_base.py --data_path $DATA --eval_file $LOG_A

# (B)  step-by-step 로그의 1-stage 블록
python evaluate_base.py --data_path $DATA --eval_file $LOG_B

# (C)  all-at-once 로그의 window 블록
python evaluate.py --data_path $DATA --eval_file $LOG_A

# (D)  step-by-step 로그의 window 블록
python evaluate.py --data_path $DATA --eval_file $LOG_B
```

`evaluate_base.py` / `evaluate.py` 모두 `Step Accuracy` 와 `Agent Accuracy` 를
정답 레이블(`mistake_step`, `mistake_agent`)과 비교해 126건 기준으로 출력합니다.

### Table 5 재현 (참고)

`Hand-Crafted` 데이터셋은 이 폴더에 포함되어 있지 않습니다(`KTserverbackup/Who_and_When/Hand-Crafted/`
에 58건이 있습니다). 재현하려면 해당 데이터를 `Who_and_When/Hand-Crafted/` 로 복사하고,
`FINAL_WINDOW_RADIUS = 5`, `--model openai/gpt-4o-mini`, `--is_handcrafted True`,
`--directory_path ../Who_and_When/Hand-Crafted` 로 위 4~5단계를 동일하게 반복하면 됩니다.

---

### 1.6 원본 로그로 논문 수치 확인 (API 재호출 불필요)

`outputs/reference/` 의 윈도우 크기 3 로그를 그대로 채점 (이 로그는 GPT-4o 직접 호출로
생성된 논문 당시 원본 로그이며, OpenRouter 마이그레이션과 무관하게 그대로 사용 가능합니다):

```bash
cd repair_exp/Automated_FA
DATA=../Who_and_When/Algorithm-Generated
AL="outputs/reference/all_at_once_gpt-4o_alg_generated (WIN3).txt"
SS="outputs/reference/step_by_step_gpt-4o_alg_generated (WIN=3).txt"

python evaluate_base.py --data_path $DATA --eval_file "$AL"   # (A)
python evaluate_base.py --data_path $DATA --eval_file "$SS"   # (B)
python evaluate.py      --data_path $DATA --eval_file "$AL"   # (C)
python evaluate.py      --data_path $DATA --eval_file "$SS"   # (D)
```

### 검증 결과 (이 저장소에서 실제 실행)

| Process | 채점 방식 | 논문 Table 3 | 원본 로그 재채점 | 일치 |
|---|---|---|---|---|
| (A) All-at-Once      | `evaluate_base` on `all_at_once (WIN3)`   | 0.1746 / 0.4444 | 0.0952 / 0.4365 | 근사 (아래 주1) |
| (B) Step-by-Step     | `evaluate_base` on `step_by_step (WIN=3)` | 0.2460 / 0.3571 | **0.2460 / 0.3571** | 정확히 일치 |
| (C) (A) + Window     | `evaluate` on `all_at_once (WIN3)`        | 0.4444 / 0.5873 | **0.4444 / 0.5873** | 정확히 일치 |
| (D) (B) + Window     | `evaluate` on `step_by_step (WIN=3)`      | 0.4524 / 0.6270 | **0.4524 / 0.6270** | 정확히 일치 |

(수치는 Step Accuracy / Agent Accuracy)

> **주1 — (A) baseline:** B/C/D 는 커밋된 로그로 논문 수치가 정확히 재현되지만,
> (A) 는 그렇지 않습니다. `all_at_once (WIN3)` 로그의 1-stage 부분을 채점하면
> Agent 0.4365 / Step 0.0952 로, 논문의 0.4444 / 0.1746 과 다릅니다.
> 커밋된 다른 all-at-once 로그 중에서는 `(WIN1)` 의 1-stage 가 논문 (A) 에 가장 가깝습니다
> (Agent 0.4444 일치, Step 0.1667 vs 0.1746). 즉 논문 (A) 행을 만든 정확한 실행 로그는
> 저장소에 남아 있지 않으며, baseline 의 run-to-run LLM 변동으로 보입니다.
> 1-stage 예측은 윈도우 크기와 무관하므로, 재추론 시 `--two_stage_window` 실행 로그의
> 1-stage 부분을 그대로 (A)/(B) 로 채점하면 됩니다.

> LLM 응답은 완전 결정론적이지 않으므로(`temperature=0.6`, `seed=42` 를 고정하지만
> 모델·프로바이더에 따라 비트 단위 재현을 보장하지 않음) 재추론 시 수치가 소폭 달라질 수 있습니다.
> OpenRouter 경유 시에는 실제 요청을 처리하는 업스트림 프로바이더에 따라 이 변동폭이
> 조금 더 커질 수 있습니다.
