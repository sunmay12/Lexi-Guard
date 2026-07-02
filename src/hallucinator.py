import hashlib
import json
import random

from src.hallucination_types import HallucinationType

# 환각 유형별 생성 함수
def hallucinate_number(text: str) -> str:
    """1. 숫자변경: 조문 내 숫자를 다른 숫자로 교체"""
    
    import re

    # 조문번호 제거 (제5조, 제10조 등)
    clean_text = re.sub(r'제\d+조', '', text)

    numbers = re.findall(r'\d+', clean_text)

    if not numbers:
        return None

    target = random.choice(numbers)
    original = int(target)

    if original <= 0:
        return None

    while True:
        delta = random.choice([-1, 1]) * random.randint(
            max(1, int(original * 0.2)),
            max(2, int(original * 0.5))
        )

        fake = original + delta

        if fake > 0 and fake != original:
            break

    pattern = rf'(?<!\d){target}(?!\d)'

    result = re.sub(
        pattern,
        str(fake),
        text,
        count=1
    )

    if result == text:
        return None

    return result


def hallucinate_delete_condition(text: str) -> str:
    """2. 조건삭제: 조건절 제거"""
    import re
    patterns = [
        r'단,[^.]*\.',
        r'[^.]*경우에는[^.]*\.',
        r'[^.]*한 때에는[^.]*\.',
        r'[^.]*이상인[^.]*\.',
        r'[^.]*이하인[^.]*\.',
        r'[^.]*경우에 한하여[^.]*\.',
        r'[^.]*경우에만[^.]*\.',
        r'[^.]*에 한한다[^.]*\.',
        r'[^.]*를 초과하는[^.]*\.',
        r'[^.]*미만인[^.]*\.',
        r'[^.]*이상[^.]*\.',
        r'[^.]*이하[^.]*\.',
        r'[^.]*다만,[^.]*\.',
        r'[^.]*경우[^.]*\.',
        r'[^.]*때에는[^.]*\.',
        r'[^.]*한하여[^.]*\.',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match and match.group() != text:
            result = text.replace(match.group(), ' ').strip()
            if result and result != text:
                return result
    return None


def hallucinate_add_condition(text: str) -> str:
    """3. 조건추가: 없는 조건 삽입"""
    fake_conditions = [

        " 단, 5인 미만 사업장은 제외한다.",
        " 단, 계약직 근로자는 해당하지 않는다.",
        " 단, 수습기간 중에는 적용하지 않는다.",
        " 단, 사용자의 동의가 있는 경우에 한한다.",
        " 단, 상시근로자 10인 이상 사업장에 한한다.",
        " 단, 공공기관 근로자에 한하여 적용한다.",
        " 단, 외국인 근로자는 제외한다.",
        " 단, 노조의 동의를 받은 경우에만 가능하다.",
        " 단, 사업장 내규에 따른다.",
        " 단, 긴급한 경우에는 적용하지 않는다.",
        " 단, 사용자가 승인한 경우에만 가능하다.",
        " 단, 연장근로에는 적용되지 않는다.",
    ]

    condition = random.choice(
        fake_conditions
    )

    if '.' in text:
        return (
            text.rsplit('.', 1)[0]
            + condition
        )

    return None


def hallucinate_mix_articles(article1: str, article2: str) -> str:
    """4. 조문혼합: 앞 조문 앞문장 + 뒷 조문 뒷문장"""
    
    s1 = [s.strip() for s in article1.split('.') if s.strip()]
    s2 = [s.strip() for s in article2.split('.') if s.strip()]
    
    # 각각 문장이 2개 이상일 때만 혼합
    if len(s1) < 2 or len(s2) < 2:
        return None
    
    # 앞 조문 앞 절반 + 뒷 조문 뒷 절반
    front = s1[:len(s1)//2]
    back = s2[len(s2)//2:]
    
    mixed = '. '.join(front + back) + '.'
    
    # 원본이랑 같으면 None
    if mixed == article1 or mixed == article2:
        return None
    
    return mixed


def hallucinate_partial(text: str) -> str:
    """5. 부분정답: 내용 절반만 남기기"""
    # 마침표 기준
    sentences = [
        s.strip()
        for s in text.split('.')
        if s.strip()
    ]

    if len(sentences) >= 2:

        half = max(
            1,
            len(sentences) // 2
        )

        return (
            '. '.join(sentences[:half])
            + '.'
        )

    return None

def make_sample(
    article: dict,
    answer: str,
    hallucination_type: str,
    label_gt: str,
    source: str = "rule_generated",
) -> dict:
    content = article["조문내용"]

    sample_id = hashlib.md5(
        f"{article['법령명']}|{article['조문번호']}|{hallucination_type}|{answer}".encode()
    ).hexdigest()

    return {
        "id": sample_id,
        "law_name": article["법령명"],
        "article_no": article["조문번호"],
        "article_title": article["조문제목"],
        "context": content,
        "answer": answer,
        "hallucination_type": hallucination_type,
        "severity": None,
        "label_gt": label_gt,
        "changed_span": {"before": "", "after": ""},
        "change_description": "",
        "plausibility_score": None,
        "judge_reason": "",
        "needs_revalidation": False,
        "source": source,
    }

# 메인 함수
def generate_hallucinations(articles: list) -> list:
    """
    조문 리스트 -> 환각 답변 데이터셋 생성
    각 조문당 5가지 유형 시도
    """
    dataset = []
    
    valid_articles = [a for a in articles if len(a["조문내용"].strip()) >= 30]

    for i, article in enumerate(valid_articles):
        content = article["조문내용"]
        
        # 1. 숫자변경
        result = hallucinate_number(content)
        if result and result != content:
            dataset.append(make_sample(
                article,
                result,
                HallucinationType.NUMBER_MANIPULATION.value,
                "Not_Faithful",
            ))
        
        # 2. 조건삭제
        result = hallucinate_delete_condition(content)
        if result and result != content:
            dataset.append(make_sample(
                article,
                result,
                HallucinationType.CONDITION_DELETION.value,
                "Not_Faithful",
            ))
        
        # 3. 조건추가
        result = hallucinate_add_condition(content)
        if result and result != content:
            dataset.append(make_sample(
                article,
                result,
                HallucinationType.CONDITION_ADDITION.value,
                "Not_Faithful",
            ))
        
        # 4. 조문혼합 (다른 조문과 섞기)
        if i + 1 < len(valid_articles):
            # 같은 조문번호 제외 + 내용이 다른 조문 선택
            candidates = [
                a for j, a in enumerate(valid_articles)
                if j != i 
                and a["조문번호"] != article["조문번호"]
                and a["조문내용"] != content
            ]
            if candidates:
                mix_target = random.choice(candidates)
                result = hallucinate_mix_articles(
                    content, 
                    mix_target["조문내용"]
                )
                if result and result != content:
                    dataset.append(make_sample(
                        article,
                        result,
                        HallucinationType.ARTICLE_MIXING.value,
                        "Not_Faithful",
                    ))

        
        # 5. 부분정답
        result = hallucinate_partial(content)
        if result and result != content:
            dataset.append(make_sample(
                article,
                result,
                HallucinationType.INFORMATION_OMISSION.value,
                "Not_Faithful",
            ))
        
        # 원본도 포함 (Faithful 케이스)
        dataset.append(make_sample(
            article,
            content,
            HallucinationType.NO_HALLUCINATION.value,
            "Faithful",
        ))
    
    return dataset


if __name__ == "__main__":
    import json
    
    # 전체 법령 로드
    all_articles = []
    for law in ["근로기준법", "남녀고용평등법", "고용보험법"]:
        with open(f"data/raw/{law}.json", encoding="utf-8") as f:
            articles = json.load(f)
            # 법령명 추가
            for article in articles:
                article["법령명"] = law
            all_articles.extend(articles)
    
    print(f"전체 조문 수: {len(all_articles)}개")
    
    result = generate_hallucinations(all_articles)
    
    # 유형별 통계
    from collections import Counter
    types = Counter(r["hallucination_type"] for r in result)
    print("\n유형별 생성 수:")
    for t, c in types.items():
        print(f"  {t}: {c}개")
    
    with open("data/processed/hallucinations.json",
              "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n총 {len(result)}개 저장 완료")