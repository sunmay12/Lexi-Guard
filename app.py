import argparse
import difflib
import re
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "lexi-guard-roberta"
DEFAULT_MAX_LENGTH = 512
ID2LABEL = {0: "Faithful", 1: "Not_Faithful"}

CONDITION_HINTS = [
    "다만", "단,", "경우에 한하여", "경우에만", "한한다", "한하여",
    "제외", "승인", "신청", "의결",
]

LEGAL_EFFECT_HINTS = [
    "하여야 한다", "할 수 있다", "할 수 없다", "하지 못한다",
    "하여서는 아니 된다", "의무", "금지", "허용", "거부",
]

_model = None
_tokenizer = None
_device = None
_model_dir = DEFAULT_MODEL_DIR

app = Flask(__name__)
app.config["MAX_LENGTH"] = DEFAULT_MAX_LENGTH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lexi-Guard demo server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    return parser.parse_args()


def get_model():
    global _model, _tokenizer, _device
    if _model is None or _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(_model_dir)
        _model = AutoModelForSequenceClassification.from_pretrained(_model_dir)
        _device = "mps" if torch.backends.mps.is_available() else "cpu"
        _model.to(_device)
        _model.eval()
    return _model, _tokenizer, _device


def predict(context: str, answer: str, max_length: int) -> dict:
    model, tokenizer, device = get_model()
    inputs = tokenizer(
        context,
        answer,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = {key: value.to(device) for key, value in inputs.items()}

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
        "explanation": build_explanation(context, answer, ID2LABEL[pred_id]),
    }


def compact(text: str, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def changed_chunks(context: str, answer: str) -> tuple[list[str], list[str]]:
    matcher = difflib.SequenceMatcher(None, context, answer)
    removed = []
    added = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in {"delete", "replace"}:
            chunk = context[i1:i2].strip()
            if chunk:
                removed.append(compact(chunk))
        if tag in {"insert", "replace"}:
            chunk = answer[j1:j2].strip()
            if chunk:
                added.append(compact(chunk))
    return removed[:3], added[:3]


def build_explanation(context: str, answer: str, label: str) -> dict:
    removed, added = changed_chunks(context, answer)
    hints = []

    if context == answer:
        summary = "원문과 후보 조문이 동일합니다."
    elif len(answer) > len(context):
        summary = "후보 조문에 원문에 없던 문구가 추가된 것으로 보입니다."
    elif len(answer) < len(context):
        summary = "후보 조문에서 원문의 일부 문구가 누락된 것으로 보입니다."
    else:
        summary = "원문과 후보 조문 사이에 치환 또는 재작성 차이가 있습니다."

    added_text = " ".join(added)
    removed_text = " ".join(removed)
    combined_diff = f"{added_text} {removed_text}"

    if re.search(r"\d", combined_diff):
        hints.append("숫자, 날짜, 기간 또는 조문번호 변경 가능성")
    if any(marker in added_text for marker in CONDITION_HINTS):
        hints.append("조건 또는 단서 추가 가능성")
    if any(marker in removed_text for marker in CONDITION_HINTS):
        hints.append("조건 또는 단서 삭제 가능성")
    if any(marker in combined_diff for marker in LEGAL_EFFECT_HINTS):
        hints.append("의무, 금지, 허용 등 법적 효과 표현 변화 가능성")

    if not hints and label == "Not_Faithful":
        hints.append("표면 차이는 작지만 모델이 원문 충실성 저하를 감지했습니다.")
    if not hints and label == "Faithful":
        hints.append("큰 표면 차이가 감지되지 않았습니다.")

    return {
        "summary": summary,
        "added": added,
        "removed": removed,
        "hints": hints,
        "disclaimer": "이 설명은 모델 내부 근거가 아니라 원문과 후보 조문의 표면 차이를 기반으로 한 보조 분석입니다.",
    }


@app.route("/", methods=["GET"])
@app.route("/index.html", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        context = str(payload.get("context", "")).strip()
        answer = str(payload.get("answer", "")).strip()
        if not context or not answer:
            return jsonify({"error": "context와 answer가 필요합니다."}), 400

        max_length = app.config["MAX_LENGTH"]
        return jsonify(predict(context, answer, max_length)), 200
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


def main() -> None:
    global _model_dir
    args = parse_args()
    _model_dir = Path(args.model_dir)
    app.config["MAX_LENGTH"] = args.max_length

    if not _model_dir.exists():
        raise SystemExit(f"model dir not found: {_model_dir}")

    print(f"Lexi-Guard demo: http://{args.host}:{args.port}")
    print(f"model: {_model_dir}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()