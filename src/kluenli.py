import json
import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
    precision_score, recall_score
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 데이터 로드 =====
def load_split(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

test_data = load_split("data/processed/splits/test.json")

LABEL2ID = {"Faithful": 0, "Not_Faithful": 1}

# ===== KLUE-NLI 모델 =====
# 이 태스크 구조:
#   premise   = context  (원본 조문)
#   hypothesis = answer  (모델 답변)
#   entailment → Faithful / contradiction → Not_Faithful
NLI_MODEL = "klue/roberta-large-nli"  # KLUE NLI fine-tuned
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
nli_model     = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
nli_model.to(device)
nli_model.eval()

# KLUE-NLI 레이블 인덱스 확인
# 모델에 따라 순서가 다를 수 있으므로 id2label로 직접 확인
print("NLI label map:", nli_model.config.id2label)
# 예상: {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

# ===== 추론 =====
nli_preds  = []
nli_labels = []

for d in test_data:
    inputs = nli_tokenizer(
        d["context"],          # premise
        d["answer"],           # hypothesis
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = nli_model(**inputs).logits  # shape: (1, 3)

    probs = torch.softmax(logits, dim=-1)[0]

    # entailment 확률 vs contradiction 확률로 이진 분류
    # neutral은 "판단 보류"이므로 Not_Faithful 쪽으로 보수적으로 처리
    entail_idx = [
        k for k, v in nli_model.config.id2label.items()
        if v.lower() == "entailment"
    ][0]
    contra_idx = [
        k for k, v in nli_model.config.id2label.items()
        if v.lower() == "contradiction"
    ][0]

    entail_prob = probs[entail_idx].item()
    contra_prob = probs[contra_idx].item()

    # entailment가 contradiction보다 높으면 Faithful
    pred = 0 if entail_prob > contra_prob else 1
    nli_preds.append(pred)
    nli_labels.append(LABEL2ID[d["label_gt"]])

# ===== 평가 =====
print("\n===== KLUE-NLI (zero-shot) =====")
print(classification_report(
    nli_labels, nli_preds,
    target_names=["Faithful", "Not_Faithful"],
))
nli_acc  = accuracy_score(nli_labels,  nli_preds)
nli_f1   = f1_score(nli_labels,        nli_preds, average="macro")
nli_prec = precision_score(nli_labels, nli_preds, average="macro")
nli_rec  = recall_score(nli_labels,    nli_preds, average="macro")
print(f"Accuracy: {nli_acc:.4f} / Macro F1: {nli_f1:.4f} / Precision: {nli_prec:.4f} / Recall: {nli_rec:.4f}")

# ===== 유형별 분석 =====
type_preds_map  = defaultdict(list)
type_labels_map = defaultdict(list)

for i, d in enumerate(test_data):
    h_type = d["hallucination_type_gt"]
    type_preds_map[h_type].append(nli_preds[i])
    type_labels_map[h_type].append(nli_labels[i])

print("\n[유형별 Not_Faithful F1]")
for h_type in sorted(type_preds_map.keys()):
    preds_t  = type_preds_map[h_type]
    labels_t = type_labels_map[h_type]
    n = len(labels_t)
    if len(set(labels_t)) < 2:
        only = "Faithful" if labels_t[0] == 0 else "Not_Faithful"
        acc = sum(p == l for p, l in zip(preds_t, labels_t)) / n
        print(f"  {h_type}: Acc={acc:.3f} (n={n}, {only} only)")
    else:
        f1 = f1_score(labels_t, preds_t, pos_label=1, average="binary")
        print(f"  {h_type}: F1={f1:.3f} (n={n})")