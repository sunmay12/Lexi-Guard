import re
import json

def parse_json_response(text: str) -> dict | None:
    """
    LLM 응답에서 JSON 객체를 추출해 dict로 반환.
    Generator / Judge 양쪽에서 공통 사용.
    파싱 실패 또는 필수 키 누락 시 None 반환.
    """
    text = re.sub(r'```json|```', '', text).strip()
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())

        if not isinstance(parsed, dict):
            return None

        return parsed

    except json.JSONDecodeError:
        return None

def extract_changed_span(change_description: str) -> dict[str, str]:
    """change_description에서 before / after 텍스트를 추출."""
    quoted_match = re.search(
        r"[\"']([^\"']+)[\"']\s*→\s*[\"']([^\"']+)[\"']",
        change_description,
    )
    if quoted_match:
        return {
            "before": quoted_match.group(1).strip(),
            "after": quoted_match.group(2).strip(),
        }

    normalized = re.sub(r"^유형\s*[A-F]\s*-\s*", "", change_description).strip()
    match = re.search(
        r"([^→]+?)\s*→\s*([^→]+?)\s*(변경|치환|삭제|추가|역전|축소|확대)?$",
        normalized,
    )
    if match:
        return {
            "before": match.group(1).strip().strip("'\""),
            "after":  match.group(2).strip().strip("'\""),
        }
    return {"before": "", "after": ""}
