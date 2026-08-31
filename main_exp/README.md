# main_exp — 논문 메인 실험 재현 코드

논문 Table 3 (프로세스 **A / B / C / D**, `Algorithm-Generated` 126건, GPT-4o, 윈도우 크기 3)
을 재현하는 데 필요한 파일만 **원본 그대로(무수정)** 모아둔 폴더입니다.
코드 구조·프롬프트는 일절 손대지 않았습니다.

| 프로세스 | 방법 | 설명 |
|---|---|---|
| (A) | All-at-Once | 전통적 전역 분석 (baseline) |
| (B) | Step-by-Step | 전통적 순차 분석 (baseline) |
| (C) | (A) + Window | (A) 결과를 pivot으로 윈도우 증강 + 윈도우 집중 식별 |
| (D) | (B) + Window | (B) 결과를 pivot으로 윈도우 증강 + 윈도우 집중 식별 |

한 번의 `--two_stage_window` 실행이 **1-stage 예측(= A/B)** 과 **윈도우 재식별(= C/D)** 을
같은 로그 파일에 함께 기록하므로, 실제 추론은 **2회**(all_at_once, step_by_step)만 하면 됩니다.

---

## 1. 폴더 구조

```
main_exp/
├── README.md
├── Who_and_When/
│   └── Algorithm-Generated/                     # 데이터셋 126건 (원본 그대로)
└── Automated_FA/
    ├── inference.py                             # 추론 진입점
    ├── evaluate_base.py                         # (A), (B) 채점 — "Prediction for X.json:" 블록 파싱
    ├── evaluate.py                              # (C), (D) 채점 — "=== Final Prediction for X.json ===" 블록 파싱
    ├── evaluate_1stage.py                       # (C)/(D) 대체 파서 (포맷 자동 감지)
    ├── evaluate_1stage_alg.py                   # (C)/(D) 대체 파서 (에이전트명 정규화 포함)
    ├── Lib/
    │   ├── utils.py                             # A/B/C/D 방법론 본체 (GPT)
    │   └── local_model.py                       # 로컬 모델 경로 (inference.py import 의존성)
    └── outputs/
        └── reference/                           # 논문에 쓰인 원본 실행 로그 (윈도우 크기 3)
            ├── all_at_once_gpt-4o_alg_generated (WIN3).txt
            └── step_by_step_gpt-4o_alg_generated (WIN=3).txt
```

### 파일 출처 (전부 무수정 복사본)

| 파일 | 원본 위치 |
|---|---|
| `Automated_FA/inference.py`, `evaluate*.py`, `Lib/utils.py` | `KTserverbackup/Automated_FA/` |
| `Automated_FA/Lib/local_model.py` | `Eliceserverbackup/who&when_2stage/Automated_FA/Lib/local_model (1).py` <br>(KT 백업에 이 파일이 빠져 있어 같은 계열 백업에서 그대로 가져옴) |
| `Who_and_When/Algorithm-Generated/` | `KTserverbackup/Who_and_When/Algorithm-Generated/` (Elice 백업과 바이트 동일) |
| `outputs/reference/all_at_once_gpt-4o_alg_generated (WIN3).txt` | `KTserverbackup/Automated_FA/outputs/[FINAL]all-at-once/Algorithm-Generated/` |
| `outputs/reference/step_by_step_gpt-4o_alg_generated (WIN=3).txt` | `KTserverbackup/Automated_FA/outputs/[FINAL]step-by-step/Algorithm-generated/` |

---

## 2. 환경

- Python 3.12 (논문: 3.12.8)
- 필수 패키지: `openai` (논문: SDK 1.59.3), `python-dotenv`, `tqdm`
- `inference.py` 최상단이 `torch` / `transformers` / `Lib.local_model` 을 import 하므로
  **GPT-4o 실험만 돌려도 `torch`, `transformers` 는 설치되어 있어야 합니다** (GPU 불필요).
- `main_exp/` 또는 상위에 `.env` 파일을 두고 `OPENAI_API_KEY=...` 설정
  (또는 각 명령에 `--api_key ...` 전달).

```bash
pip install openai python-dotenv tqdm torch transformers
```

---

## 3. 윈도우 크기 설정 (중요)

`Automated_FA/Lib/utils.py` 351행:

```python
FINAL_WINDOW_RADIUS = 5   # ← 원본 그대로 두었음
```

논문 §4.1 은 메인 실험(Table 3)에서 **윈도우 증강 범위를 3** 으로 고정합니다.
따라서 **Table 3 을 재현하려면 이 값을 `3` 으로 바꾼 뒤** 추론을 실행하세요.
(윈도우 크기 분석 Fig. 9 는 이 값을 1~5 로 바꿔가며 반복 실행한 결과입니다.
 `outputs/reference/` 의 로그는 이미 3 으로 실행된 것입니다.)

---

## 4. 추론 실행

작업 디렉터리: `main_exp/Automated_FA/`

```bash
cd main_exp/Automated_FA

# (A) + (C) : all-at-once  및  all-at-once + window
python inference.py --method all_at_once --model gpt-4o --two_stage_window \
    --directory_path ../Who_and_When/Algorithm-Generated --is_handcrafted False

# (B) + (D) : step-by-step  및  step-by-step + window
python inference.py --method step_by_step --model gpt-4o --two_stage_window \
    --directory_path ../Who_and_When/Algorithm-Generated --is_handcrafted False
```

결과 로그 저장 위치:

```
Automated_FA/outputs/all_at_once_gpt-4o_alg_generated.txt
Automated_FA/outputs/step_by_step_gpt-4o_alg_generated.txt
```

각 로그에는 `Prediction for X.json:` (1-stage = A/B) 블록과
`=== Final Prediction for X.json ===` (window = C/D) 블록이 함께 들어갑니다.

---

## 5. 채점

```bash
cd main_exp/Automated_FA
DATA=../Who_and_When/Algorithm-Generated

# (A)  all-at-once 로그의 1-stage 블록
python evaluate_base.py --data_path $DATA --eval_file outputs/all_at_once_gpt-4o_alg_generated.txt

# (B)  step-by-step 로그의 1-stage 블록
python evaluate_base.py --data_path $DATA --eval_file outputs/step_by_step_gpt-4o_alg_generated.txt

# (C)  all-at-once 로그의 window 블록
python evaluate.py --data_path $DATA --eval_file outputs/all_at_once_gpt-4o_alg_generated.txt

# (D)  step-by-step 로그의 window 블록
python evaluate.py --data_path $DATA --eval_file outputs/step_by_step_gpt-4o_alg_generated.txt
```

`evaluate_base.py` / `evaluate.py` 모두 `Step Accuracy` 와 `Agent Accuracy` 를
정답 레이블(`mistake_step`, `mistake_agent`)과 비교해 126건 기준으로 출력합니다.

---

## 6. 원본 로그로 논문 수치 확인 (API 재호출 불필요)

`outputs/reference/` 의 윈도우 크기 3 로그를 그대로 채점:

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
| (B) Step-by-Step     | `evaluate_base` on `step_by_step (WIN=3)` | 0.2460 / 0.3571 | **0.2460 / 0.3571** | ✅ 정확히 일치 |
| (C) (A) + Window     | `evaluate` on `all_at_once (WIN3)`        | 0.4444 / 0.5873 | **0.4444 / 0.5873** | ✅ 정확히 일치 |
| (D) (B) + Window     | `evaluate` on `step_by_step (WIN=3)`      | 0.4524 / 0.6270 | **0.4524 / 0.6270** | ✅ 정확히 일치 |

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
> GPT-4o 는 비트 단위 재현을 보장하지 않음) 재추론 시 수치가 소폭 달라질 수 있습니다.
