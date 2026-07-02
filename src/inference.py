import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ID2LABEL = {0: "Faithful", 1: "Not_Faithful"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lexi-Guard 모델로 원문 법령 대비 후보 조문의 충실성을 판별합니다."
    )
    parser.add_argument("--model-dir", default="./lexi-guard-roberta")
    parser.add_argument("--context", help="원문 법령 조문")
    parser.add_argument("--answer", help="검증할 후보 조문 또는 답변")
    parser.add_argument(
        "--input-json",
        help=(
            "context/answer 필드를 가진 JSON 파일. 리스트 또는 단일 객체를 지원합니다."
        ),
    )
    parser.add_argument("--output-json", help="배치 예측 결과 저장 경로")
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def load_inputs(args: argparse.Namespace) -> list[dict]:
    if args.input_json:
        with open(args.input_json, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        return data

    if not args.context or not args.answer:
        raise SystemExit("--context와 --answer를 함께 주거나 --input-json을 사용하세요.")

    return [{"context": args.context, "answer": args.answer}]


def predict_one(
    model,
    tokenizer,
    context: str,
    answer: str,
    max_length: int,
) -> dict:
    inputs = tokenizer(
        context,
        answer,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)

    pred_id = int(torch.argmax(probs).item())
    return {
        "label": ID2LABEL[pred_id],
        "confidence": float(probs[pred_id].item()),
        "probabilities": {
            ID2LABEL[i]: float(probs[i].item())
            for i in range(len(ID2LABEL))
        },
    }


def main() -> None:
    args = parse_args()
    rows = load_inputs(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()

    results = []
    for row in rows:
        result = predict_one(
            model,
            tokenizer,
            row["context"],
            row["answer"],
            args.max_length,
        )
        results.append({**row, **result})

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"saved: {output_path}")
        return

    print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
