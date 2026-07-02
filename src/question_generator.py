import json
import os
import time
import re
import logging
import yaml

from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

with open("config/prompts.yaml", "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)

TEST_MODE = True
TEST_SIZE = False

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# logging.basicConfig(
#     filename="question_generation.log",
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

# 예외조항 존재 여부 판단
EXCEPTION_KEYWORDS = [
    "다만",
    "단,",
    "예외",
    "제외한다",
    "적용하지 아니",
    "그러하지 아니하다"
]

SITUATION_KEYWORDS = [
    "근로자", "사업주", "사용자", "개월", "일",
    "경우", "상황", "신청", "요구", "가능"
]


# def has_exception(content: str) -> bool:
#     return any(kw in content for kw in EXCEPTION_KEYWORDS)
# 수정 (3개 핵심만)
def has_exception(content: str) -> bool:
    return "단서" in content or "다만" in content or "예외" in content


def parse_response(text: str) -> dict:
    text = re.sub(r'```json|```', '', text).strip()
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def validate_question(q: str) -> bool:
    q = q.strip()

    if len(q) < 25:
        return False

    SUBJECTS = ["근로자", "사업주", "사용자"]
    if not any(s in q for s in SUBJECTS):
        return False

    if len(q.split()) < 8:
        return False

    # 단조로운 질문 필터
    if len(set(q)) < 10:
        return False

    VALID_ENDINGS = ["인가", "하는가", "가능한가", "해야 하는가", "할 수 있는가"]
    if not any(q.endswith(e) or q.endswith(e + "?") for e in VALID_ENDINGS):
        return False

    return True


def generate_questions(articles: list, max_retries: int = 1) -> list:
    dataset = []
    
    for i, article in enumerate(articles):
        content = article["조문내용"]
        
        # 예외조항 여부에 따라 프롬프트 선택
        has_exc = has_exception(content)
        prompt_template = PROMPTS["WITH_EXCEPTION"] if has_exc else PROMPTS["WITHOUT_EXCEPTION"]
        expected_types = (
            {"권리판단형", "의무판단형", "예외적용형"}
            if has_exc else
            {"권리판단형", "의무판단형", "경계사례형"}
        )
        
        success = False
        
        for attempt in range(max_retries):
            try:
                prompt = prompt_template.format(
                    num=article["조문번호"],
                    title=article["조문제목"],
                    content=content
                )
                
                response = client.models.generate_content(
                    model="models/gemini-3.1-flash-lite",
                    contents=prompt
                )
                parsed = parse_response(response.text)
                
                if not parsed:
                    print(f"[{i+1}] 제{article['조문번호']}조 파싱 실패 (시도 {attempt+1})")
                    time.sleep(5)
                    continue
                
                questions = parsed.get("questions", [])
                
                # 개수 검증
                if len(questions) != 3:
                    print(f"[{i+1}] 제{article['조문번호']}조 {len(questions)}개 → 재생성")
                    time.sleep(5)
                    continue
                
                # 유형 검증
                types = {q.get("question_type") for q in questions}
                if not expected_types.issubset(types):
                    print(f"[{i+1}] 제{article['조문번호']}조 유형 불일치 → 재생성")
                    time.sleep(5)
                    continue
                
                # 품질 검증
                valid_questions = []
                for q in questions:
                    if validate_question(q.get("question", "")):
                        valid_questions.append(q)

                if len(valid_questions) != 3:
                    print(
                        f"[{i+1}] 제{article['조문번호']}조 "
                        f"품질 검증 실패 ({len(valid_questions)}/3) → 재생성"
                    )
                    # time.sleep(3)
                    continue

                questions = valid_questions

                # gold_answer 저장 안 함 -> question + source_article만
                for q in questions:
                    dataset.append({
                        "법령명": article["법령명"],
                        "조문번호": article["조문번호"],
                        "조문제목": article["조문제목"],
                        "source_article": article["조문내용"],
                        "question_type": q.get("question_type"),
                        "question": q.get("question"),
                        "has_exception": has_exc,
                        # 추가
                        "law_category": "labor",
                        "evaluation_group":
                            "exception"
                            if has_exc
                            else "normal",

                        "article_length": len(article["조문내용"])
                    })
                with open(
                    "data/processed/questions_backup.json",
                    "w",
                    encoding="utf-8"
                ) as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)
                
                exc_label = "예외O" if has_exc else "예외X"
                print(f"[{i+1}/{len(articles)}] 제{article['조문번호']}조 ({exc_label}) ✅")
                success = True
                break
            
            except Exception as e:
                error_str = str(e)
                if "429" in error_str:
                    wait_time = 60
                    print(f"할당량 초과(429). {wait_time}초간 휴식...")
                    time.sleep(wait_time)
                    continue 
                else:
                    print(f"[{i+1}] 제{article['조문번호']}조 오류: {error_str}")
                    time.sleep(5)
        
        if not success:
            print(f"[{i+1}] 제{article['조문번호']}조 ❌ 스킵")
        
        time.sleep(5)
    
    return dataset


def remove_duplicates(dataset: list, threshold: float = 0.94) -> list:
    """semantic_search 기반 중복 제거"""
    try:
        from sentence_transformers import SentenceTransformer, util
        import torch
        
        print("\n중복 제거 중...")
        emb_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
        
        questions = [d["question"] for d in dataset]
        embeddings = emb_model.encode(
            questions,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        # semantic_search 기반 중복 제거
        keep_indices = []
        removed = 0
        
        for i in range(len(dataset)):
            if i == 0:
                keep_indices.append(i)
                continue
            
            kept_embeddings = embeddings[keep_indices]
            scores = util.cos_sim(embeddings[i], kept_embeddings)[0]
            
            if scores.max().item() < threshold:
                keep_indices.append(i)
            else:
                removed += 1
        
        print(f"중복 제거: {removed}개 제거 → {len(keep_indices)}개 남음")
        return [dataset[i] for i in keep_indices]
    
    except ImportError:
        print("sentence-transformers 미설치 → 중복 제거 스킵")
        return dataset


if __name__ == "__main__":
    all_articles = []
    
    for law in ["근로기준법", "남녀고용평등법", "고용보험법"]:
        with open(f"data/raw/{law}.json", encoding="utf-8") as f:
            articles = json.load(f)
            for article in articles:
                article["법령명"] = law
            all_articles.extend(articles)
    
    # 기존 결과 로드
    existing_dataset = []
    existing_keys = set()
    
    if os.path.exists("data/processed/questions.json"):
        with open("data/processed/questions.json", encoding="utf-8") as f:
            existing_dataset = json.load(f)
        # 이미 처리된 (법령명 + 조문번호) 조합
        existing_keys = {
            (d["법령명"], d["조문번호"])
            for d in existing_dataset
        }
        print(f"기존 질문 {len(existing_dataset)}개 로드")
    
    # 이미 처리된 조문 스킵
    remaining_articles = [
        a for a in all_articles
        if (a["법령명"], a["조문번호"]) not in existing_keys
    ]
    print(f"남은 조문: {len(remaining_articles)}개\n")
    
    # 나머지만 생성
    new_dataset = generate_questions(remaining_articles)
    
    # 합치기
    dataset = remove_duplicates(existing_dataset + new_dataset)
    
    with open("data/processed/questions.json",
              "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n최종 {len(dataset)}개 질문 저장 완료")