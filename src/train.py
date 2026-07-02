# ===== Lexi-Guard Fine-tuning =====
# klue/roberta-large 기반 Faithful / Not_Faithful 이진 분류
# Google Colab T4 기준

# 설치
# !pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 -q
# !pip install transformers==4.52.4 datasets==3.6.0 accelerate==1.7.0 scikit-learn -q

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report, f1_score, precision_score, recall_score
)
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset

# 설정
MODEL_NAME   = "klue/roberta-large"
MAX_LENGTH   = 512
BATCH_SIZE   = 8
EPOCHS       = 5
LR           = 2e-5
OUTPUT_DIR   = "./lexi-guard-roberta"
LABEL2ID     = {"Faithful": 0, "Not_Faithful": 1}
ID2LABEL     = {0: "Faithful", 1: "Not_Faithful"}

# 데이터 로드
def load_split(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

train_data = load_split("data/processed/splits/train.json")
val_data   = load_split("data/processed/splits/val.json")
test_data  = load_split("data/processed/splits/test.json")

# 전처리 + 토크나이저
# 입력 형식: [CLS] context [SEP] answer [SEP]
# context = 원본 법령 조문, answer = 모델이 생성한 답변
def make_records(data):
    return [
        {
            "text_a": d["context"],
            "text_b": d["answer"],
            "label":  LABEL2ID[d["label_gt"]],
        }
        for d in data
    ]

train_records = make_records(train_data)
val_records   = make_records(val_data)
test_records  = make_records(test_data)

train_dataset = Dataset.from_list(train_records)
val_dataset   = Dataset.from_list(val_records)
test_dataset  = Dataset.from_list(test_records)

# 토크나이저
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text_a"],
        batch["text_b"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset   = val_dataset.map(tokenize, batched=True)
test_dataset  = test_dataset.map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch",   columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch",  columns=["input_ids", "attention_mask", "label"])

# 클래스 가중치 (불균형 보정)
labels_train = [d["label"] for d in train_records]
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=labels_train,
)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print(f"Class weights: Faithful={class_weights[0]:.3f}, Not_Faithful={class_weights[1]:.3f}")

# 커스텀 Trainer (가중치 손실 적용)
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(
            weight=class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# 평가 함수
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1":        f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro"),
        "recall":    recall_score(labels, preds, average="macro"),
        "f1_not_faithful": f1_score(labels, preds, pos_label=1),
    }

# 모델 + 학습
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

# 학습 설정
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    warmup_ratio=0.1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=50,
    fp16=True,           # T4는 fp16 지원
    report_to="none",    # wandb 끄기
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# 학습
trainer.train()

# 테스트셋 평가
print("\n===== Test Set 평가 =====")
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

print(classification_report(
    labels, preds,
    target_names=["Faithful", "Not_Faithful"]
))

# 유형별 분석 (Error Analysis용)
print("\n===== 유형별 F1 (Not_Faithful 기준)=====")
from collections import defaultdict

type_preds = defaultdict(list)
type_labels = defaultdict(list)

for i, d in enumerate(test_data):
    h_type = d.get("hallucination_type_gt") or d.get("hallucination_type", "No_Hallucination")
    type_preds[h_type].append(preds[i])
    type_labels[h_type].append(labels[i])

for h_type in sorted(type_preds.keys()):
    if len(set(type_labels[h_type])) < 2:
        continue
    f1 = f1_score(type_labels[h_type], type_preds[h_type], average="binary", pos_label=1)
    n  = len(type_labels[h_type])
    print(f"  {h_type}: F1={f1:.3f} (n={n})")

# 모델 저장
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n모델 저장: {OUTPUT_DIR}")