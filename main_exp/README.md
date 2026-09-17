# main_exp — 논문 메인 실험 재현 코드

논문 Table 3 (프로세스 **A / B / C / D**, `Algorithm-Generated` 126건, 윈도우 크기 3)
을 재현하는 데 필요한 파일만 모아둔 폴더입니다.
A/B/C/D(baseline + window augmentation)와 무관한 실험 코드(Binary Search, Chunk-Parallel,
로컬 GPU 모델 경로)는 제거하였고, 대신 **OpenRouter** 를 통해 어떤 모델로도(`--model` 변경만으로)
같은 파이프라인을 재현·비교할 수 있도록 구성했습니다.

| 프로세스 | 방법 | 설명 |
|---|---|---|
| (A) | All-at-Once | 전통적 전역 분석 (baseline) |
| (B) | Step-by-Step | 전통적 순차 분석 (baseline) |
| (C) | (A) + Window | (A) 결과를 pivot으로 윈도우 증강 + 윈도우 집중 식별 |
| (D) | (B) + Window | (B) 결과를 pivot으로 윈도우 증강 + 윈도우 집중 식별 |

한 번의 `--two_stage_window` 실행이 **1-stage 예측(= A/B)** 과 **윈도우 재식별(= C/D)** 을
같은 로그 파일에 함께 기록하므로, 실제 추론은 **2회**(all_at_once, step_by_step)만 하면 됩니다.

---

## 0. 논문 실험 결과 위치 (Window-Augmentation... .pdf 기준)

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

## 1. 폴더 구조

```
main_exp/
├── README.md
├── Who_and_When/
│   └── Algorithm-Generated/                     # 데이터셋 126건 (원본 그대로)
└── Automated_FA/
    ├── inference.py                             # 추론 진입점 (OpenRouter 호출)
    ├── evaluate_base.py                         # (A), (B) 채점 — "Prediction for X.json:" 블록 파싱
    ├── evaluate.py                              # (C), (D) 채점 — "=== Final Prediction for X.json ===" 블록 파싱
    ├── evaluate_1stage.py                       # (C)/(D) 대체 파서 (포맷 자동 감지)
    ├── evaluate_1stage_alg.py                   # (C)/(D) 대체 파서 (에이전트명 정규화 포함)
    ├── Lib/
    │   └── utils.py                             # A/B/C/D 방법론 본체
    └── outputs/
        └── reference/                           # 논문에 쓰인 원본 실행 로그 (윈도우 크기 3)
            ├── all_at_once_gpt-4o_alg_generated (WIN3).txt
            └── step_by_step_gpt-4o_alg_generated (WIN=3).txt
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

## 2. 환경 (OpenRouter + .env)

- Python 3.12 (논문: 3.12.8)
- 필수 패키지: `openai`, `python-dotenv`, `tqdm`

```bash
pip install openai python-dotenv tqdm
```

- `inference.py`는 [OpenRouter](https://openrouter.ai/)의 OpenAI 호환 엔드포인트를 사용합니다.
  `--model`에 OpenRouter 모델 ID(예: `openai/gpt-4o`, `openai/gpt-4o-mini`,
  `anthropic/claude-3.5-sonnet`, `meta-llama/llama-3.1-70b-instruct`)를 넘기면 되며,
  전체 목록은 https://openrouter.ai/models 참고.
- `main_exp/Automated_FA/` (또는 상위 디렉터리)에 `.env` 파일을 두고 아래 값을 설정하세요.

```dotenv
# main_exp/Automated_FA/.env
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

## 3. 윈도우 크기 설정 (중요)

`Automated_FA/Lib/utils.py`의 `FINAL_WINDOW_RADIUS`:

```python
FINAL_WINDOW_RADIUS = 5   # ← 원본 그대로 두었음
```

논문 §4.1 은 메인 실험(Table 3)에서 **윈도우 증강 범위를 3** 으로 고정합니다.
따라서 **Table 3 을 재현하려면 이 값을 `3` 으로 바꾼 뒤** 추론을 실행하세요.
(윈도우 크기 분석 Fig. 9 는 이 값을 1~5 로 바꿔가며 반복 실행한 결과입니다.
 `outputs/reference/` 의 로그는 이미 3 으로 실행된 것입니다. Table 5(Hand-Crafted)는 5로 실행됩니다.)

---

## 4. 추론 실행

작업 디렉터리: `main_exp/Automated_FA/`

```bash
cd main_exp/Automated_FA

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

## 5. 채점

```bash
cd main_exp/Automated_FA
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

## 6. 원본 로그로 논문 수치 확인 (API 재호출 불필요)

`outputs/reference/` 의 윈도우 크기 3 로그를 그대로 채점 (이 로그는 GPT-4o 직접 호출로
생성된 논문 당시 원본 로그이며, OpenRouter 마이그레이션과 무관하게 그대로 사용 가능합니다):

```bash
cd main_exp/Automated_FA
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
