import json
import random
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score

random.seed(42)

# 데이터 로드
def load_split(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

train_data = load_split("data/processed/splits/train.json")
val_data   = load_split("data/processed/splits/val.json")
test_data  = load_split("data/processed/splits/test.json")

LABEL2ID = {"Faithful": 0, "Not_Faithful": 1}

def make_features(data):
    # context + answer를 [SEP] 토큰으로 이어붙임 — RoBERTa와 동일한 입력 구조
    texts  = [f"{d['context']} [SEP] {d['answer']}" for d in data]
    labels = [LABEL2ID[d["label_gt"]] for d in data]
    return texts, labels

X_train, y_train = make_features(train_data)
X_val,   y_val   = make_features(val_data)
X_test,  y_test  = make_features(test_data)

# TF-IDF 벡터화
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),   # unigram + bigram
    max_features=50000,
    sublinear_tf=True,    # log(1+tf) — 빈도 스케일 압축
)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec   = vectorizer.transform(X_val)
X_test_vec  = vectorizer.transform(X_test)

# Logistic Regression
clf = LogisticRegression(
    C=1.0,
    class_weight="balanced",   # Faithful/Not_Faithful 불균형 보정
    max_iter=1000,
    random_state=42,
)
clf.fit(X_train_vec, y_train)

# Val 성능 (하이퍼파라미터 참고용)
val_preds = clf.predict(X_val_vec)
val_f1    = f1_score(y_val, val_preds, average="macro")
print(f"Val Macro F1: {val_f1:.4f}")

# Test 평가
test_preds = clf.predict(X_test_vec)

print("\n===== TF-IDF + Logistic Regression =====")
print(classification_report(
    y_test, test_preds,
    target_names=["Faithful", "Not_Faithful"],
))
tfidf_acc  = accuracy_score(y_test, test_preds)
tfidf_f1   = f1_score(y_test,       test_preds, average="macro")
tfidf_prec = precision_score(y_test, test_preds, average="macro")
tfidf_rec  = recall_score(y_test,   test_preds, average="macro")
print(f"Accuracy: {tfidf_acc:.4f} / Macro F1: {tfidf_f1:.4f} / Precision: {tfidf_prec:.4f} / Recall: {tfidf_rec:.4f}")

# 유형별 분석
from collections import defaultdict

type_preds_map  = defaultdict(list)
type_labels_map = defaultdict(list)

for i, d in enumerate(test_data):
    h_type = d.get("hallucination_type_gt") or d.get("hallucination_type", "No_Hallucination")
    type_preds_map[h_type].append(test_preds[i])
    type_labels_map[h_type].append(y_test[i])

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