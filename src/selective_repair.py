import json
import os
import time
import re
from pathlib import Path
from dotenv import load_dotenv
from google import genai

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PROMPT_EXCEPTION_ONLY = """
당신은 법률 전문가입니다.
아래 법령 조문을 읽고, 예외조항/단서조항에 관한 질문 1개만 생성하세요.

[필수 조건]
- 조문의 단서조항 또는 예외규정 적용 여부를 판단해야 하는 질문
- 질문은 반드시 "근로자", "사업주", "사용자" 중 하나로 시작할 것
- 조문에서 도출 가능한 범위 내에서 구체적인 상황 포함
- 문장 끝은 반드시 "인가?", "하는가?", "가능한가?" 형태

[출력 형식 - JSON만 출력]
{{
  "question_type": "예외적용형",
  "question": "구체적 상황을 포함한 질문"
}}

조문번호: {num}
조문제목: {title}
조문내용: {content}
"""


def validate_question(q: str) -> bool:
    q = q.strip()
    if len(q) < 25:
        return False
    if not any(s in q for s in ["근로자", "사업주", "사용자"]):
        return False
    if len(q.split()) < 8:
        return False
    if len(set(q)) < 10:
        return False
    VALID_ENDINGS = ["인가", "하는가", "가능한가", "해야 하는가", "할 수 있는가"]
    if not any(q.endswith(e) or q.endswith(e + "?") for e in VALID_ENDINGS):
        return False
    return True


def parse_response(text: str) -> dict:
    text = re.sub(r'```json|```', '', text).strip()
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    # 기존 questions.json 로드
    with open("data/processed/questions.json", encoding="utf-8") as f:
        dataset = json.load(f)

    # article 단위로 collapse
    article_exception_map = {}
    for d in dataset:
        key = (d["법령명"], d["조문번호"])
        if key not in article_exception_map:
            article_exception_map[key] = d["has_exception"]

    # 예외조항 있는 조문 키 셋
    exception_articles_in_dataset = {
        key for key, has_exc in article_exception_map.items()
        if has_exc == True
    }

    # 예외적용형 질문 이미 있는 조문 키 셋
    has_exception_type = {
        (d["법령명"], d["조문번호"])
        for d in dataset
        if d["question_type"] == "예외적용형"
    }

    # 전체 법령 로드
    all_articles = []
    for law in ["근로기준법", "남녀고용평등법", "고용보험법"]:
        with open(f"data/raw/{law}.json", encoding="utf-8") as f:
            articles = json.load(f)
            for a in articles:
                a["법령명"] = law
            all_articles.extend(articles)

    # repair 대상: 예외조항 있는데 예외적용형 없는 조문
    repair_targets = [
        a for a in all_articles
        if (a["법령명"], a["조문번호"]) in exception_articles_in_dataset
        and (a["법령명"], a["조문번호"]) not in has_exception_type
    ]

    print(f"repair 대상: {len(repair_targets)}개 조문\n")

    added = 0
    for i, article in enumerate(repair_targets):
        try:
            prompt = PROMPT_EXCEPTION_ONLY.format(
                num=article["조문번호"],
                title=article["조문제목"],
                content=article["조문내용"]
            )

            response = client.models.generate_content(
                model="gemini-1.5-flash-8b",
                contents=prompt
            )
            parsed = parse_response(response.text)

            if not parsed:
                print(f"[{i+1}] 제{article['조문번호']}조 파싱 실패")
                time.sleep(2)
                continue

            q = parsed.get("question", "")
            if not validate_question(q):
                print(f"[{i+1}] 제{article['조문번호']}조 품질 실패: {q[:50]}")
                time.sleep(2)
                continue

            dataset.append({
                "법령명": article["법령명"],
                "조문번호": article["조문번호"],
                "조문제목": article["조문제목"],
                "source_article": article["조문내용"],
                "question_type": "예외적용형",
                "question": q,
                "has_exception": True,
                "law_category": "labor",
                "evaluation_group": "exception",
                "article_length": len(article["조문내용"])
            })

            added += 1
            print(f"[{i+1}/{len(repair_targets)}] {article['법령명']} 제{article['조문번호']}조 ✅")

            # 백업
            with open("data/processed/exception_backup.json",
                      "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)

            time.sleep(5)

        except Exception as e:
            if "429" in str(e):
                print(f"429 → 60초 대기")
                time.sleep(60)
            else:
                print(f"[{i+1}] 오류: {e}")
                time.sleep(2)

    # # 최종 저장
    # with open("data/processed/repair_exception.json", "w", encoding="utf-8") as f:
    #     json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n예외적용형 {added}개 추가")
    print(f"최종 총 {len(dataset)}개")

    # 유형 분포 확인
    from collections import Counter
    types = Counter(d["question_type"] for d in dataset)
    print("\n[질문 유형 분포]")
    for k, v in sorted(types.items()):
        print(f"  {k}: {v}개")