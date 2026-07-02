import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.config import PROJECT_ROOT
from src.hallucination_types import HallucinationType
from src.utils.parser import parse_json_response


env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

TEACHER_MODEL = os.getenv("TEACHER_MODEL", "gpt-4o-mini")
API_DELAY = 0.5

QUESTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "questions.json"
HALLUCINATIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "llm_hallucination_final_dataset.json"
)
HARD_NEGATIVES_PATH = PROJECT_ROOT / "data" / "labeled" / "hard_negatives.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "labeled" / "dataset.json"
BACKUP_PATH = PROJECT_ROOT / "data" / "labeled" / "dataset_backup.json"


_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        _client = OpenAI(api_key=api_key)
    return _client


PROMPT = """
당신은 법률 전문가이자 AI 평가 전문가입니다.
아래 [조문], [질문], [답변]을 읽고 답변의 신실성(Faithfulness)을 평가하세요.

[조문]
{context}

[질문]
{question}

[답변]
{answer}

평가 기준:
- score: 답변이 조문에 근거하는 정도를 0~100점으로 평가
  * 100: 조문 내용과 완전히 일치
  * 70~99: 대체로 일치하나 일부 부정확
  * 30~69: 부분적으로 일치하나 중요한 오류 포함
  * 0~29: 조문과 거의 일치하지 않음

- label:
  * "Faithful"
  * "Not_Faithful"

- hallucination_type:
  * "No_Hallucination"
  * "Number_Manipulation"
  * "Condition_Addition"
  * "Condition_Deletion"
  * "Information_Omission"
  * "Scope_Manipulation"
  * "Legal_Effect_Reversal"
  * "Entity_Substitution"
  * "Article_Mixing"

반드시 JSON만 출력하세요:
{{
  "score": 85,
  "label": "Faithful",
  "hallucination_type": "No_Hallucination",
  "reason": "답변이 조문의 내용과 정확히 일치함"
}}
"""


def load_json(path: Path, default: list | None = None) -> list:
    if default is None:
        default = []

    if not path.exists() or path.stat().st_size == 0:
        return default

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_question_map(questions: list[dict]) -> dict[tuple[str, str], list[dict]]:
    question_map: dict[tuple[str, str], list[dict]] = {}

    for q in questions:
        key = (q.get("법령명", ""), q.get("조문번호", ""))
        if not all(key):
            continue
        question_map.setdefault(key, []).append(q)

    return question_map


LEGACY_TYPE_MAP = {
    "Partial_Answer": HallucinationType.INFORMATION_OMISSION.value,
    "Historical_Regulation_Confusion": HallucinationType.NUMBER_MANIPULATION.value,
    "Cross_Act_Mixing": HallucinationType.ARTICLE_MIXING.value,
    "해당없음": HallucinationType.NO_HALLUCINATION.value,
}


def normalize_hallucination_type(h_type: str) -> str:
    return LEGACY_TYPE_MAP.get(h_type, h_type)


def normalize_hallucination_sample(h: dict) -> dict:
    """
    새 스키마를 우선 사용하되, 예전 rule-generated 스키마도 읽을 수 있게 변환.
    """
    if "law_name" in h:
        h_type = normalize_hallucination_type(h.get("hallucination_type", ""))
        return {
            "id": h.get("id"),
            "law_name": h.get("law_name", ""),
            "article_no": h.get("article_no", ""),
            "article_title": h.get("article_title", ""),
            "context": h.get("context", ""),
            "answer": h.get("answer", ""),
            "hallucination_type_gt": h_type,
            "label_gt": h.get("label_gt", "Not_Faithful"),
            "source": h.get("source", "unknown"),
            "severity": h.get("severity"),
            "changed_span": h.get("changed_span", {"before": "", "after": ""}),
            "change_description": h.get("change_description", ""),
            "plausibility_score": h.get("plausibility_score"),
            "judge_reason": h.get("judge_reason", ""),
            "needs_revalidation": h.get("needs_revalidation", False),
        }

    raw_type = h.get("type", "")
    h_type = normalize_hallucination_type(raw_type)
    context = h.get("원본", "")
    answer = h.get("hallucinated", "")
    sample_id = h.get("id") or hashlib.md5(
        (
            f"{h.get('법령명', '')}|{h.get('조문번호', '')}|"
            f"{h_type}|{answer}"
        ).encode()
    ).hexdigest()

    return {
        "id": sample_id,
        "law_name": h.get("법령명", ""),
        "article_no": h.get("조문번호", ""),
        "article_title": h.get("조문제목", ""),
        "context": context,
        "answer": answer,
        "hallucination_type_gt": h_type,
        "label_gt": (
            "Faithful"
            if h_type == HallucinationType.NO_HALLUCINATION.value
            else "Not_Faithful"
        ),
        "source": h.get("source", "legacy_hard_negative"),
        "severity": h.get("severity"),
        "changed_span": h.get("changed_span", {"before": "", "after": ""}),
        "change_description": h.get("change_description", h.get("note", "")),
        "plausibility_score": h.get("plausibility_score"),
        "judge_reason": h.get("judge_reason", ""),
        "needs_revalidation": h.get("needs_revalidation", False),
    }


def build_dataset(
    questions_path: Path = QUESTIONS_PATH,
    hallucinations_path: Path = HALLUCINATIONS_PATH,
    hard_negatives_path: Path = HARD_NEGATIVES_PATH,
) -> list[dict]:
    questions = load_json(questions_path)
    hallucinations = load_json(hallucinations_path)
    hard_negatives = load_json(hard_negatives_path)

    question_map = make_question_map(questions)
    dataset = []

    for q in questions:
        source = q.get("source_article", "")
        if not source:
            continue

        dataset.append({
            "id": hashlib.md5(
                (
                    f"{q.get('법령명', '')}|{q.get('조문번호', '')}|"
                    f"{q.get('question', '')}|faithful"
                ).encode()
            ).hexdigest(),
            "law_name": q.get("법령명", ""),
            "article_no": q.get("조문번호", ""),
            "article_title": q.get("조문제목", ""),
            "context": source,
            "answer": source,
            "hallucination_type_gt": HallucinationType.NO_HALLUCINATION.value,
            "label_gt": "Faithful",
            "source": "faithful_original",
            "severity": "none",
            "changed_span": {"before": "", "after": ""},
            "change_description": "",
            "plausibility_score": None,
            "judge_reason": "",
            "needs_revalidation": False,
            "question": q.get("question", ""),
            "question_type": q.get("question_type"),
            "teacher_score": None,
            "teacher_label": None,
            "teacher_hallucination_type": None,
            "teacher_reason": None,
            "is_hard_negative": False,
        })

    for raw_h in hallucinations:
        h = normalize_hallucination_sample(raw_h)
        key = (h["law_name"], h["article_no"])
        qs = question_map.get(key, [])

        if not qs:
            continue

        for q in qs:
            dataset.append({
                **h,
                "id": hashlib.md5(
                    f"{h.get('id', '')}|{q.get('question', '')}".encode()
                ).hexdigest(),
                "question": q.get("question", ""),
                "question_type": q.get("question_type"),
                "teacher_score": None,
                "teacher_label": None,
                "teacher_hallucination_type": None,
                "teacher_reason": None,
                "is_hard_negative": False,
            })

    for hn in hard_negatives:
        h = normalize_hallucination_sample(hn)
        dataset.append({
            **h,
            "question": f"{h['article_title']}에 관한 질문",
            "question_type": "hard_negative",
            "teacher_score": None,
            "teacher_label": None,
            "teacher_hallucination_type": None,
            "teacher_reason": None,
            "is_hard_negative": True,
        })

    return dataset


def label_single(context: str, question: str, answer: str, max_attempts: int = 3) -> dict | None:
    prompt = PROMPT.format(
        context=context[:2000],
        question=question,
        answer=answer,
    )

    for attempt in range(max_attempts):
        try:
            response = get_client().chat.completions.create(
                model=TEACHER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return parse_json_response(response.choices[0].message.content)

        except Exception as e:
            if "429" in str(e) and attempt < max_attempts - 1:
                wait = 60 * (attempt + 1)
                print(f"429 → {wait}초 대기")
                time.sleep(wait)
                continue
            raise

    return None


def validate_teacher_result(result: dict | None) -> bool:
    if not result:
        return False

    if result.get("label") not in {"Faithful", "Not_Faithful"}:
        return False

    if result.get("hallucination_type") not in HallucinationType.values():
        return False

    score = result.get("score")
    if not isinstance(score, (int, float)):
        return False

    return 0 <= score <= 100


def save_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    print("데이터셋 구성 중...")
    dataset = build_dataset()
    print(f"총 {len(dataset)}개 라벨링 대상\n")

    labeled = []
    skipped = 0

    for i, item in enumerate(dataset):
        try:
            result = label_single(
                context=item["context"],
                question=item["question"],
                answer=item["answer"],
            )

            if not validate_teacher_result(result):
                print(f"[{i + 1}] teacher 응답 검증 실패 → 스킵")
                skipped += 1
                continue

            item.update({
                "teacher_score": int(result["score"]),
                "teacher_label": result["label"],
                "teacher_hallucination_type": result["hallucination_type"],
                "teacher_reason": result.get("reason", ""),
            })

            labeled.append(item)

            if (i + 1) % 10 == 0:
                save_json(BACKUP_PATH, labeled)
                print(f"[{i + 1}/{len(dataset)}] 백업 저장")

            print(
                f"[{i + 1}/{len(dataset)}] "
                f"{item['teacher_label']} ({item['teacher_score']}점)"
            )

            time.sleep(API_DELAY)

        except Exception as e:
            print(f"[{i + 1}] 오류: {e}")
            skipped += 1
            time.sleep(1)

    save_json(OUTPUT_PATH, labeled)

    print(f"\n라벨링 완료: {len(labeled)}개")
    print(f"스킵: {skipped}개")

    labels = Counter(d["teacher_label"] for d in labeled)
    types = Counter(d["teacher_hallucination_type"] for d in labeled)

    print("\n[Teacher 라벨 분포]")
    for k, v in labels.items():
        print(f"  {k}: {v}개")

    print("\n[Teacher 환각 유형 분포]")
    for k, v in types.items():
        print(f"  {k}: {v}개")

    scores = [d["teacher_score"] for d in labeled if d["teacher_score"] is not None]
    if scores:
        print(f"\n평균 score: {sum(scores) / len(scores):.1f}")
