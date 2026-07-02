import json

from src.config import LOG_DIR

PARSING_ERROR_LOG = LOG_DIR / "parsing_error.jsonl"


def log_parsing_error(h_type: str, raw_response: str, article: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    entry = {
        "type": h_type,
        "raw_response": raw_response,
        "article": f"{article.get('법령명', '')} 제{article.get('조문번호', '')}조",
    }

    with open(PARSING_ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")