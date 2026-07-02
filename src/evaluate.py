import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


LABEL2ID = {"Faithful": 0, "Not_Faithful": 1}
ID2LABEL = {0: "Faithful", 1: "Not_Faithful"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="저장된 Lexi-Guard 모델을 test split에서 재평가합니다."
    )
    parser.add_argument("--model-dir", default="./lexi-guard-roberta")
    parser.add_argument("--test-path", default="data/processed/splits/test.json")
    parser.add_argument("--output-dir", default="data/processed/eval")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_json(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_hallucination_type(row: dict) -> str:
    return (
        row.get("hallucination_type_gt")
        or row.get("hallucination_type")
        or "No_Hallucination"
    )


def make_records(data: list[dict]) -> list[dict]:
    return [
        {
            "text_a": row["context"],
            "text_b": row["answer"],
            "label": LABEL2ID[row["label_gt"]],
        }
        for row in data
    ]


def build_dataset(records: list[dict], tokenizer, max_length: int) -> Dataset:
    dataset = Dataset.from_list(records)

    def tokenize(batch):
        return tokenizer(
            batch["text_a"],
            batch["text_b"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return dataset


def metric_summary(labels: np.ndarray, preds: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "macro_precision": float(precision_score(labels, preds, average="macro")),
        "macro_recall": float(recall_score(labels, preds, average="macro")),
        "not_faithful_f1": float(f1_score(labels, preds, pos_label=1)),
        "not_faithful_precision": float(precision_score(labels, preds, pos_label=1)),
        "not_faithful_recall": float(recall_score(labels, preds, pos_label=1)),
        "faithful_f1": float(f1_score(labels, preds, pos_label=0)),
        "faithful_precision": float(precision_score(labels, preds, pos_label=0)),
        "faithful_recall": float(recall_score(labels, preds, pos_label=0)),
    }


def per_type_detection_recall(
    data: list[dict],
    labels: np.ndarray,
    preds: np.ndarray,
) -> dict:
    type_labels: dict[str, list[int]] = defaultdict(list)
    type_preds: dict[str, list[int]] = defaultdict(list)

    for i, row in enumerate(data):
        h_type = get_hallucination_type(row)
        type_labels[h_type].append(int(labels[i]))
        type_preds[h_type].append(int(preds[i]))

    result = {}
    for h_type in sorted(type_labels):
        y_true = np.array(type_labels[h_type])
        y_pred = np.array(type_preds[h_type])
        label_counts = Counter(ID2LABEL[int(label)] for label in y_true)

        if set(y_true) == {1}:
            score_name = "not_faithful_detection_recall"
            score = float(recall_score(y_true, y_pred, pos_label=1))
        elif set(y_true) == {0}:
            score_name = "faithful_acceptance_recall"
            score = float(recall_score(y_true, y_pred, pos_label=0))
        else:
            score_name = "binary_f1"
            score = float(f1_score(y_true, y_pred, pos_label=1))

        wrong_indices = [
            i for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred))
            if true_label != pred_label
        ]
        result[h_type] = {
            "n": int(len(y_true)),
            "label_counts": dict(label_counts),
            score_name: score,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "wrong": int(len(wrong_indices)),
        }

    return result


def balanced_subset_indices(
    data: list[dict],
    labels: np.ndarray,
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        by_label[int(label)].append(i)

    if len(by_label) < 2:
        return list(range(len(labels)))

    sample_size = min(len(indices) for indices in by_label.values())
    chosen = []
    for label in sorted(by_label):
        indices = by_label[label][:]
        rng.shuffle(indices)
        chosen.extend(indices[:sample_size])
    chosen.sort()
    return chosen


def collect_errors(
    data: list[dict],
    labels: np.ndarray,
    preds: np.ndarray,
) -> list[dict]:
    errors = []
    for i, row in enumerate(data):
        if int(labels[i]) == int(preds[i]):
            continue
        errors.append({
            "index": i,
            "id": row.get("id"),
            "law_name": row.get("law_name"),
            "article_no": row.get("article_no"),
            "article_title": row.get("article_title"),
            "hallucination_type": get_hallucination_type(row),
            "severity": row.get("severity"),
            "gold": ID2LABEL[int(labels[i])],
            "pred": ID2LABEL[int(preds[i])],
            "changed_span": row.get("changed_span"),
            "change_description": row.get("change_description"),
            "context": row.get("context"),
            "answer": row.get("answer"),
        })
    return errors


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_data = load_json(args.test_path)
    records = make_records(test_data)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    test_dataset = build_dataset(records, tokenizer, args.max_length)

    eval_args = TrainingArguments(
        output_dir=str(output_dir / "trainer_tmp"),
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        # tokenizer=tokenizer,
    )
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    report_text = classification_report(
        labels,
        preds,
        target_names=[ID2LABEL[0], ID2LABEL[1]],
    )
    metrics = metric_summary(labels, preds)
    per_type = per_type_detection_recall(test_data, labels, preds)
    errors = collect_errors(test_data, labels, preds)

    balanced_indices = balanced_subset_indices(test_data, labels, args.seed)
    balanced_labels = labels[balanced_indices]
    balanced_preds = preds[balanced_indices]
    balanced_metrics = metric_summary(balanced_labels, balanced_preds)
    balanced_cm = confusion_matrix(balanced_labels, balanced_preds, labels=[0, 1])

    payload = {
        "model_dir": args.model_dir,
        "test_path": args.test_path,
        "n_test": len(test_data),
        "label_distribution": dict(Counter(row["label_gt"] for row in test_data)),
        "metrics": metrics,
        "confusion_matrix": {
            "labels": [ID2LABEL[0], ID2LABEL[1]],
            "matrix": cm.tolist(),
        },
        "balanced_subset": {
            "n": len(balanced_indices),
            "label_distribution": dict(
                Counter(ID2LABEL[int(label)] for label in balanced_labels)
            ),
            "metrics": balanced_metrics,
            "confusion_matrix": {
                "labels": [ID2LABEL[0], ID2LABEL[1]],
                "matrix": balanced_cm.tolist(),
            },
        },
        "per_type": per_type,
        "num_errors": len(errors),
    }

    save_json(output_dir / "metrics.json", payload)
    save_json(output_dir / "errors.json", errors)

    print("\n===== Test Set 평가 =====")
    print(report_text)
    print("Confusion matrix rows=gold, cols=pred [Faithful, Not_Faithful]")
    print(cm)

    print("\n===== Balanced Subset 평가 =====")
    print(f"n={len(balanced_indices)}")
    print(balanced_metrics)
    print(balanced_cm)

    print("\n===== 유형별 Detection Recall =====")
    for h_type, row in per_type.items():
        score_keys = [
            key for key in row
            if key.endswith("_recall") or key == "binary_f1"
        ]
        score = row[score_keys[0]]
        print(
            f"{h_type:26s}: {score_keys[0]}={score:.3f} "
            f"accuracy={row['accuracy']:.3f} wrong={row['wrong']}/{row['n']} "
            f"labels={row['label_counts']}"
        )

    print(f"\n오답 저장: {output_dir / 'errors.json'} ({len(errors)}개)")
    print(f"메트릭 저장: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
