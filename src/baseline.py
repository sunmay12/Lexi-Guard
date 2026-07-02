# Baseline 평가 + 파인튜닝 결과 기록
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, f1_score, accuracy_score
import torch, json

with open("data/processed/splits/test.json", encoding="utf-8") as f:
    test_data = json.load(f)

LABEL2ID = {"Faithful": 0, "Not_Faithful": 1}

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
baseline_model = AutoModelForSequenceClassification.from_pretrained(
    "klue/roberta-large",
    num_labels=2,
    id2label={0: "Faithful", 1: "Not_Faithful"},
    label2id={"Faithful": 0, "Not_Faithful": 1},
)
baseline_model.eval()
if torch.cuda.is_available():
    baseline_model = baseline_model.cuda()

baseline_preds, baseline_labels = [], []

for d in test_data:
    inputs = tokenizer(
        d["context"], d["answer"],
        truncation=True, max_length=512,
        padding="max_length", return_tensors="pt",
    )
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        logits = baseline_model(**inputs).logits
    baseline_preds.append(torch.argmax(logits, dim=-1).item())
    baseline_labels.append(LABEL2ID[d["label_gt"]])

print("===== Baseline (파인튜닝 전) =====")
print(classification_report(
    baseline_labels, baseline_preds,
    target_names=["Faithful", "Not_Faithful"],
))
baseline_f1  = f1_score(baseline_labels, baseline_preds, average="macro")
baseline_acc = accuracy_score(baseline_labels, baseline_preds)
print(f"Accuracy: {baseline_acc:.4f} / Macro F1: {baseline_f1:.4f}")

# 파인튜닝 결과 — 셀 8 평가 후 여기에 직접 기입
ours_results = {
    "accuracy": 0.9824,
    "macro_f1": 0.9721,
}

# 최종 비교표
print("=" * 65)
print(f"{'Model':<35} {'Accuracy':>10} {'Macro F1':>10}")
print("=" * 65)
print(f"{'KLUE-RoBERTa (baseline)':<35} {baseline_acc:>10.4f} {baseline_f1:>10.4f}")
print(f"{'KLUE-RoBERTa (fine-tuned)':<35} "
      f"{ours_results['accuracy']:>10.4f} "
      f"{ours_results['macro_f1']:>10.4f}")
print("=" * 65)