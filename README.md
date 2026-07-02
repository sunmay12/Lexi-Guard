# Lexi-Guard

Lexi-Guard는 한국어 법률 텍스트의 **원문 충실성(faithfulness)** 을 판별하는 경량 검증기입니다. 원문 법령 조문(`context`)과 후보 조문 또는 답변(`answer`)을 함께 입력하면, 후보가 원문에 충실한지 `Faithful` / `Not_Faithful`로 분류합니다.

이 프로젝트는 법률 RAG/GraphRAG 시스템에서 자주 발생하는 "근거는 맞게 찾았지만 생성 답변이 숫자, 조건, 적용 범위, 법적 효과를 미묘하게 바꾸는 문제"를 잡기 위해 만들어졌습니다.

## 핵심 기능

- 법령 원문과 후보 텍스트를 pair로 비교하는 binary classifier
- 숫자 조작, 조건 추가/삭제, 정보 누락, 법적 효과 반전 등 법률 특화 환각 유형 탐지
- `klue/roberta-large` 기반 fine-tuned 모델 제공
- 단일/배치 CLI 추론과 Flask 기반 로컬 웹 데모 제공
- synthetic hallucination dataset 생성, 검증, 학습, 평가 파이프라인 포함

## 프로젝트 의의

일반적인 RAG 평가는 검색 근거의 관련성이나 답변의 표면 품질에 집중하는 경우가 많습니다. 하지만 법률 문장에서는 작은 변화가 큰 의미 차이를 만듭니다.

예를 들어 다음 두 문장은 겉보기에는 비슷하지만, 후보 조문은 원문에 없는 적용 조건을 추가합니다.

```text
원문:
제91조(청구의 방식) 심사의 청구는 대통령령으로 정하는 바에 따라 문서로 하여야 한다.

후보:
제91조(청구의 방식) 심사의 청구는 대통령령으로 정하는 바에 따라 문서로 하여야 한다.
다만, 상시근로자 50인 이상 사업장에 한한다.
```

Lexi-Guard는 이런 차이를 단순 문장 유사도 문제가 아니라 **법률 원문 대비 후보 텍스트의 충실성 검증 문제**로 정의합니다. 특히 일반 zero-shot NLI 모델이 약한 조건 삭제, 정보 누락, 범위 조작 같은 유형을 법률 도메인 데이터로 직접 학습한다는 점에 의미가 있습니다.

## 입력과 출력

입력:

```json
{
  "context": "원문 법령 조문",
  "answer": "검증할 후보 조문 또는 답변"
}
```

출력:

```json
{
  "label": "Not_Faithful",
  "confidence": 0.98,
  "probabilities": {
    "Faithful": 0.02,
    "Not_Faithful": 0.98
  }
}
```

현재 모델은 자연어 질문까지 함께 이해하는 QA 평가기가 아니라, `context + answer` pair를 비교하는 충실성 분류기입니다.

## 빠른 시작

### 1. 설치

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 웹 데모 실행

```bash
python app.py --model-dir ./lexi-guard-roberta
```

브라우저에서 아래 주소를 엽니다.

```text
http://127.0.0.1:8501
```

웹 데모에서는 원문 법령과 후보 조문을 입력하고 `Faithful` / `Not_Faithful` 확률, confidence, 표면 차이 분석을 확인할 수 있습니다.

**웹 데모 화면**
이미지1
이미지2

### 3. 단일 예시 추론

```bash
python src/inference.py \
  --model-dir ./lexi-guard-roberta \
  --context "제91조(청구의 방식) 심사의 청구는 대통령령으로 정하는 바에 따라 문서로 하여야 한다." \
  --answer "제91조(청구의 방식) 심사의 청구는 대통령령으로 정하는 바에 따라 문서로 하여야 한다. 다만, 상시근로자 50인 이상 사업장에 한한다."
```

예상 출력:

```json
{
  "label": "Not_Faithful",
  "confidence": 0.98,
  "probabilities": {
    "Faithful": 0.02,
    "Not_Faithful": 0.98
  }
}
```

### 4. 배치 추론

`examples.json`:

```json
[
  {
    "context": "원문 법령 조문",
    "answer": "검증할 후보 조문"
  }
]
```

실행:

```bash
python src/inference.py \
  --model-dir ./lexi-guard-roberta \
  --input-json examples.json \
  --output-json predictions.json
```

## 프로젝트 구조

```text
Lexi-Guard/
├── app.py                         # Flask 기반 로컬 웹 데모 서버
├── requirements.txt               # Python 의존성
├── README.md
├── config/
│   └── prompts.yaml               # LLM 환각 생성/검증 프롬프트 설정
├── templates/
│   └── index.html                 # 웹 데모 UI
├── static/
│   └── style.css                  # 웹 데모 스타일
├── notebooks/
│   └── Lexi_Guard.ipynb           # 실험/분석 노트북
├── lexi-guard-roberta/            # fine-tuned 모델 아티팩트
├── data/
│   ├── raw/                       # 원본 법령 JSON
│   ├── processed/                 # 생성/검증된 데이터셋, split, 평가 결과
│   ├── labeled/                   # 추가 라벨링 데이터
│   └── logs/                      # 데이터 생성 로그
└── src/
    ├── law_fetcher.py             # 법령 데이터 수집/정리 보조
    ├── question_generator.py      # 법령 기반 질문 생성
    ├── hallucination_types.py     # 환각 유형 정의
    ├── llm_hallucinator.py        # LLM 기반 환각 샘플 생성
    ├── hallucinator.py            # 환각 생성 로직
    ├── validators/
    │   └── rule_validator.py      # rule-based 검증
    ├── labeler.py                 # 라벨링/검증 보조
    ├── markers.py                 # 변경 span/marker 처리
    ├── selective_repair.py        # 일부 샘플 보정
    ├── train.py                   # KLUE-RoBERTa fine-tuning
    ├── evaluate.py                # 저장 모델 평가
    ├── inference.py               # CLI 추론
    ├── baseline.py                # baseline 실험
    ├── tfidf_baseline.py          # TF-IDF baseline
    ├── kluenli.py                 # NLI baseline 실험
    ├── rag.py                     # RAG 관련 실험 코드
    └── utils/                     # parser, logger, severity 유틸
```

일부 대용량 산출물(`data/processed`, `data/raw`, `lexi-guard-roberta`)은 `.gitignore` 대상입니다. 로컬 작업 디렉터리에는 포함되어 있지만 새 환경에서는 별도 공유 또는 재생성이 필요할 수 있습니다.

## 데이터셋

대상 법령:

- 근로기준법
- 남녀고용평등법
- 고용보험법

환각 유형:

| Type | 설명 |
|---|---|
| `Number_Manipulation` | 숫자, 기간, 비율, 날짜 등 수량 정보 변경 |
| `Condition_Addition` | 원문에 없는 조건 또는 단서 추가 |
| `Condition_Deletion` | 원문에 있는 조건, 단서, 예외 삭제 |
| `Information_Omission` | 핵심 법적 정보 일부 누락 |
| `Legal_Effect_Reversal` | 의무, 금지, 허용 등 법적 효과 역전 |
| `Scope_Manipulation` | 적용 대상, 범위, 기간, 지역 등 조작 |
| `Entity_Substitution` | 권리/의무 주체 또는 객체 치환 |
| `No_Hallucination` | 원문과 충실하게 일치하는 faithful 샘플 |

데이터셋 규모:

```text
Total:        1,606
Faithful:       263
Not_Faithful: 1,343
```

Split:

```text
Train: 1,285
Val:     151
Test:    170
```

조문 단위로 split하여 같은 조문에서 파생된 샘플이 train/test에 동시에 들어가는 leakage를 줄였습니다.

## 데이터 생성과 품질 관리

LLM 생성 결과를 그대로 사용하지 않고, 다음 절차를 거쳐 학습 데이터로 정제했습니다.

```text
법령 조문 수집
→ LLM 기반 환각 후보 생성
→ JSON parsing
→ changed_span 추출
→ rule-based validation
→ LLM judge scoring
→ severity assignment
→ verified dataset 구축
→ train/val/test split
```

각 샘플에는 다음과 같은 메타데이터가 포함됩니다.

- `label_gt`: `Faithful` 또는 `Not_Faithful`
- `hallucination_type`: 환각 유형
- `changed_span`: 원문 대비 변경된 핵심 구간
- `severity`: 법적 의미 변화의 심각도
- `plausibility_score`: 그럴듯함 점수
- `subtlety_score`: 미묘함 점수
- `judge_reason`: 검증 판단 이유


## 모델 학습 및 추론 파이프라인

```text
Verified Dataset
        │
        ▼
Train / Validation / Test Split
        │
        ▼
KLUE-RoBERTa Fine-tuning
        │
        ▼
Lexi-Guard
        │
        ▼
Input:
(Context, Answer)
        │
        ▼
Output:
Faithful / Not_Faithful
```
본 프로젝트는 생성된 검증 데이터셋을 이용하여 KLUE-RoBERTa를 sequence-pair classification 형태로 파인튜닝하였다. 학습된 Lexi-Guard 모델은 원문 법령(context)과 후보 텍스트(answer)를 입력받아 두 문장 간의 충실성을 판단하고, Faithful 또는 Not_Faithful을 출력한다.

## 모델

- Base model: `klue/roberta-large`
- Task: binary sequence-pair classification
- Input format: `[CLS] context [SEP] answer [SEP]`
- Labels: `Faithful`, `Not_Faithful`
- Loss: weighted cross entropy
- Selection metric: macro F1
- Max length: 512 tokens

## 학습

```bash
python src/train.py
```

기본 설정:

- `MODEL_NAME = "klue/roberta-large"`
- `OUTPUT_DIR = "./lexi-guard-roberta"`
- `BATCH_SIZE = 8`
- `EPOCHS = 5`
- `LR = 2e-5`

`src/train.py`는 `data/processed/splits/train.json`, `val.json`, `test.json`를 읽어 학습하고, 가장 좋은 모델을 `lexi-guard-roberta/`에 저장합니다.

## 평가

```bash
python src/evaluate.py \
  --model-dir ./lexi-guard-roberta \
  --test-path data/processed/splits/test.json \
  --output-dir data/processed/eval
```

생성 파일:

- `data/processed/eval/metrics.json`
- `data/processed/eval/errors.json`

## 성능

### Lexi-Guard Test Set

Test Set Performance

| Class         | Precision |   Recall | F1-score | Support |
| ------------- | --------: | -------: | -------: | ------: |
| Faithful      |      0.90 | **1.00** |     0.95 |      28 |
| Not_Faithful  |  **1.00** |     0.98 |     0.99 |     142 |
| **Accuracy**  |         - |        - | **0.98** |     170 |
| **Macro Avg** |      0.95 |     0.99 | **0.97** |     170 |

Confusion Matrix

| Gold \ Pred      | Faithful | Not_Faithful |
| ---------------- | -------: | -----------: |
| **Faithful**     |   **28** |            0 |
| **Not_Faithful** |        3 |      **139** |


Balanced Test Set

| Metric   |     Value |
| -------- | --------: |
| Samples  |        56 |
| Macro F1 | **0.982** |

### Results

| Model                         |  Accuracy |  Macro F1 | Faithful F1 | Not_Faithful F1 |
| ----------------------------- | --------: | --------: | ----------: | --------------: |
| TF-IDF + Logistic Regression  |     0.700 |     0.580 |       0.350 |           0.800 |
| Zero-shot DeBERTa NLI         |     0.729 |     0.678 |       0.549 |           0.807 |
| **Lexi-Guard (KLUE-RoBERTa)** | **0.982** | **0.972** |   **0.950** |       **0.990** |


### 유형별 Detection Recall

| Type | Detection Recall |
|---|---:|
| Condition_Addition | 0.920 |
| Condition_Deletion | 1.000 |
| Entity_Substitution | 1.000 |
| Information_Omission | 1.000 |
| Legal_Effect_Reversal | 1.000 |
| Number_Manipulation | 0.909 |
| Scope_Manipulation | 1.000 |
| No_Hallucination acceptance | 1.000 |

주요 실패 사례는 **긴 조문 안에 아주 짧은 조건이 추가**되는 경우와, 개정일자 수준의 **미세한 숫자 변경**이었습니다.

### Zero-Shot NLI Baseline 비교

Baseline: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`

```text
Accuracy:        0.729
Macro F1:        0.678
Faithful F1:     0.549
Not_Faithful F1: 0.807
```

| Model | Accuracy | Macro F1 | Faithful F1 | Not_Faithful F1 |
|---|---:|---:|---:|---:|
| Zero-shot DeBERTa NLI | 0.729 | 0.678 | 0.549 | 0.807 |
| Lexi-Guard KLUE-RoBERTa | 0.982 | 0.970 | 0.950 | 0.990 |

NLI baseline은 `Legal_Effect_Reversal`, `Number_Manipulation`처럼 표면적 모순이 강한 유형은 비교적 잘 잡았지만, `Condition_Deletion`, `Information_Omission`처럼 법률적으로 중요한 누락/삭제형 환각에 취약했습니다.

## 예시

### Faithful 예시

```text
context:
제91조(청구의 방식) 심사의 청구는 대통령령으로 정하는 바에 따라 문서로 하여야 한다.

answer:
제91조(청구의 방식) 심사의 청구는 대통령령으로 정하는 바에 따라 문서로 하여야 한다.

prediction:
Faithful
```

### Not_Faithful 예시: 조건 추가

```text
context:
제91조(청구의 방식) 심사의 청구는 대통령령으로 정하는 바에 따라 문서로 하여야 한다.

answer:
제91조(청구의 방식) 심사의 청구는 대통령령으로 정하는 바에 따라 문서로 하여야 한다.
다만, 상시근로자 50인 이상 사업장에 한한다.

prediction:
Not_Faithful
```

### Not_Faithful 예시: 숫자 변경

```text
context:
사용자는 근로자에게 1주에 평균 1회 이상의 유급휴일을 보장하여야 한다.

answer:
사용자는 근로자에게 1개월에 평균 1회 이상의 유급휴일을 보장하여야 한다.

prediction:
Not_Faithful
```

## 한계

- 현재 데이터셋은 synthetic hallucination 기반이므로 실제 RAG 답변 전체에 대한 일반화 성능을 직접 보장하지 않습니다.
- 입력은 `context + answer` pair이며, 질문까지 포함한 end-to-end QA 평가기는 아닙니다.
- 긴 조문은 512 token 기준으로 truncation될 수 있습니다.
- Faithful 샘플 수가 Not_Faithful보다 적어, 더 균형 잡힌 데이터 확장이 필요합니다.
- 모델의 예측은 법률 자문이 아니라 자동 검증 보조 신호로 사용해야 합니다.

## 향후 개선 방향

- 실제 법률 QA/RAG 답변 기반 out-of-distribution 평가셋 구축
- 긴 조문 대응을 위한 sliding window 또는 changed-span 중심 입력 실험
- 숫자 조작 유형을 본문 수량 변경과 메타데이터/개정일자 변경으로 세분화
- faithful paraphrase 샘플 확장
- TF-IDF, logistic regression, Korean NLI 등 추가 baseline 정리
- RAG pipeline의 post-generation verifier로 통합
