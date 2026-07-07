import hashlib
import json
import random
import time
import re
import argparse
import sys
from datetime import datetime
from collections import Counter
from pathlib import Path
from typing import Literal
from google import genai
from src.config import (
    get_gemini_api_key,
    GENERATOR_MODEL,
    JUDGE_MODEL,
    TEST_MODE,
    TEST_SIZE,
    API_DELAY,
    ARTICLE_DELAY,
    CHECKPOINT_INTERVAL,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)
from src.hallucination_types import HallucinationType
from src.validators.rule_validator import validate
from src.utils.parser import parse_json_response, extract_changed_span
from src.utils.severity import calculate_severity
from src.utils.logger import log_parsing_error
from src.prompts.hallucination_prompts import PROMPTS, JUDGE_PROMPT

class _Tee:
    """stdout과 로그 파일에 동시 출력"""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

"""
환각 데이터셋 생성 파이프라인
단계:
  1. generate_hallucination()  — Generator LLM 호출
  2. validate_hallucination()  — Rule-based 검증
  3. judge_hallucination()     — Judge LLM 평가
  4. save_checkpoint()         — 중간 저장
  실패 원인 4분류: null_response / api_error / parse_error / no_change
  Condition_Addition 유형 강제 샘플링 (_pick_condition_type)
  fail_reason_count Counter → 유형별 실패 원인 통계 출력
"""

HALLUCINATION_TYPES = HallucinationType.generation_values()

# 실패 원인 타입
GenFailReason = Literal["null_response", "api_error", "parse_error", "no_change"]

PipelineFailReason = Literal[
    "null_response", "api_error", "parse_error", "no_change", "duplicate",
    "rule_fail", "span_invalid", "judge_plausibility", "judge_subtlety",
    "key_error", "unknown", "no_number", "no_modal_verb",
]

# Gemini 클라이언트 (모듈 레벨 싱글톤)
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=get_gemini_api_key())
    return _client


# Scope_Manipulation 강제 lexical sampling
SCOPE_SAMPLING_SEED = 42
_scope_rng = random.Random(SCOPE_SAMPLING_SEED)

SCOPE_PHRASE_POOL: dict[str, list[str]] = {
    "A": [
        "6개월 이상 계속 근로한", "1년 미만 근속한", "취업 후 3개월을 초과한",
        "2년 이상 근속한", "1년 이상 계속 근로한", "3개월 미만 근속한",
        "취업 후 1년이 지난", "6개월 미만 근속한", "5년 이상 근속한",
        "취업 후 2년을 초과한", "1년을 초과하여 근속한",
    ],
    "B": [
        "만 18세 이상의", "만 55세 미만의", "만 60세 이상의", "만 24세 미만의",
        "만 65세를 초과한", "만 19세 미만의", "만 50세 이상의", "만 30세 미만의",
        "만 62세를 초과한", "만 15세 이상의", "만 70세 이상의", "만 25세 미만의",
    ],
    "C": [
        "생산직", "사무직", "일용직", "전문직인", "현장직", "기술직",
        "서비스직", "영업직", "관리직", "연구직인", "운전직",
    ],
    "D": [
        "정규직", "기간제", "파견근로자인", "단시간근로자인", "수습",
        "계약직", "촉탁직", "임시직", "교대근무를 하는", "재택근무를 하는",
    ],
    "E": [
        "상시 5인 이상 사업장에 종사하는", "상시 30인 미만 사업장에 종사하는",
        "중소기업에 종사하는", "상시 50인 이상 사업장에 종사하는",
        "상시 10인 미만 사업장에 종사하는", "상시 100인 이상 사업장에 종사하는",
        "대기업에 종사하는", "상시 20인 이상 사업장에 종사하는",
        "상시 300인 이상 사업장에 종사하는", "영세 사업장에 종사하는",
        "상시 4인 이하 사업장에 종사하는",
    ],
    "F": [
        "수도권 사업장에 종사하는", "국내 사업장에 종사하는",
        "제조업 사업장에 종사하는", "비수도권 사업장에 종사하는",
        "지방 사업장에 종사하는", "특별시·광역시 소재 사업장에 종사하는",
        "농어촌 지역 사업장에 종사하는", "산업단지 내 사업장에 종사하는",
        "해외 사업장에 종사하는", "도서·산간 지역 사업장에 종사하는",
    ],
}

_scope_pick_queue: list[tuple[str, str]] = []


def _refill_scope_queue() -> None:
    all_pairs = [
        (type_key, phrase)
        for type_key, phrases in SCOPE_PHRASE_POOL.items()
        for phrase in phrases
    ]
    _scope_rng.shuffle(all_pairs)
    _scope_pick_queue.extend(all_pairs)


def _pick_scope_constraint() -> tuple[str, str]:
    if not _scope_pick_queue:
        _refill_scope_queue()
    return _scope_pick_queue.pop()


# 신규: Condition_Addition 유형 강제 샘플링
# Scope_Manipulation과 동일한 stratified 방식
# LLM 자율 선택 시 A 유형 편향이 관찰되어 코드 레벨 강제로 전환
CONDITION_ADDITION_TYPES = ["A", "B", "C", "D", "E", "F"]
_condition_rng = random.Random(SCOPE_SAMPLING_SEED + 1)  # scope와 다른 시드
_condition_pick_queue: list[str] = []


def _refill_condition_queue() -> None:
    """A~F를 2세트 만들어 셔플 후 큐에 채운다."""
    pool = CONDITION_ADDITION_TYPES * 2
    _condition_rng.shuffle(pool)
    _condition_pick_queue.extend(pool)


def _pick_condition_type() -> str:
    """
    Condition_Addition에서 사용할 유형(A~F)을 결정론적으로 뽑는다.
    큐가 비면 재셔플하여 다시 채운다 (stratified 방식).
    """
    if not _condition_pick_queue:
        _refill_condition_queue()
    return _condition_pick_queue.pop()


# 1. API 호출
# google-api-core가 설치돼 있으면 정식 예외 클래스 사용
# 없으면 문자열 fallback — import 실패를 조용히 처리
try:
    from google.api_core.exceptions import ResourceExhausted as _ResourceExhausted
    from google.api_core.exceptions import GoogleAPIError as _GoogleAPIError
except ImportError:
    _ResourceExhausted = None
    _GoogleAPIError = None

def _is_rate_limit_error(e: Exception) -> bool:
    """429 / ResourceExhausted 여부를 타입과 문자열 양쪽으로 확인."""
    if _ResourceExhausted is not None and isinstance(e, _ResourceExhausted):
        return True
    return "429" in str(e) or "resource exhausted" in str(e).lower()

def _is_daily_quota_exhausted(e: Exception) -> bool:
    """일일 한도 초과는 재시도해도 무의미하므로 별도 판정."""
    msg = str(e)
    return "PerDay" in msg or "generate_content_free_tier_requests" in msg

def call_with_retry(
    prompt: str,
    model: str = GENERATOR_MODEL,
    max_attempts: int = 3,
) -> str:
    for attempt in range(max_attempts):
        try:
            response = _get_client().models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.4,
                },
            )
            return response.text
        except Exception as e:
            if _is_daily_quota_exhausted(e):
                print("    ⛔ 일일 할당량 초과. 즉시 중단")
                raise
            if _is_rate_limit_error(e):
                wait = 60 * (attempt + 1)
                print(f"    RateLimit → {wait}초 대기 (attempt {attempt+1}/{max_attempts})")
                time.sleep(wait)
                if attempt == max_attempts - 1:
                    raise
            else:
                raise

# 2. 생성
def generate_hallucination(
    content: str,
    h_type: str,
    article: dict,
) -> tuple[dict | None, GenFailReason | None]:
    """
    Generator LLM을 호출해 환각 문장을 생성.

    반환 타입 변경: (결과 dict, 실패 원인) 튜플
    - 성공 시:  (dict, None)
    - 실패 시:  (None, FailReason)
      - "api_error"      : API 호출 자체가 실패
      - "parse_error"    : JSON 파싱 실패
      - "null_response"  : LLM이 적절히 null을 반환 (숫자 없음 등 정상 케이스)
      - "no_change"      : 생성됐으나 원본과 동일

    Condition_Addition: forced_condition_type 강제 주입
    Scope_Manipulation: forced_type, forced_phrase 강제 주입 (기존 유지)
    """
    # 유형별 프롬프트 포맷
    if h_type == HallucinationType.SCOPE_MANIPULATION.value:
        forced_type, forced_phrase = _pick_scope_constraint()
        prompt = PROMPTS[h_type].format(
            content=content,
            forced_type=forced_type,
            forced_phrase=forced_phrase,
        )
    elif h_type == HallucinationType.CONDITION_ADDITION.value:
        forced_condition_type = _pick_condition_type()
        prompt = PROMPTS[h_type].format(
            content=content,
            forced_condition_type=forced_condition_type,
        )
    else:
        prompt = PROMPTS[h_type].format(content=content)

    # API 호출
    try:
        raw = call_with_retry(prompt)
    except Exception as e:
        print(f"    [API 오류] {h_type}: {e}")
        log_parsing_error(h_type, "", article)
        return None, "api_error"

    # JSON 파싱
    parsed = parse_json_response(raw)
    if not parsed:
        log_parsing_error(h_type, raw, article)
        return None, "parse_error"

    # null 반환 확인
    if "hallucinated" not in parsed:
        return None, "null_response"

    hallucinated_value = parsed["hallucinated"]
    if hallucinated_value is None:
        return None, "null_response"

    hallucinated = str(hallucinated_value).strip()

    if not hallucinated or hallucinated == content.strip():
        return None, "no_change"

    change_description = parsed.get("change_description", "")

    # changed_span: LLM 직접 출력 우선, 없으면 텍스트 파싱 fallback
    llm_span = parsed.get("changed_span")
    if (
        isinstance(llm_span, dict)
        and "before" in llm_span
        and "after" in llm_span
    ):
        changed_span = {
            "before": str(llm_span["before"]).strip(),
            "after":  str(llm_span["after"]).strip(),
        }
    else:
        changed_span = extract_changed_span(change_description)

    if h_type == HallucinationType.SCOPE_MANIPULATION.value:
        changed_span["forced_type"] = forced_type
        changed_span["forced_phrase"] = forced_phrase

    return {
        "hallucinated":       hallucinated,
        "change_description": change_description,
        "changed_span":       changed_span,
    }, None


# 3. Rule-based 검증
def validate_hallucination(
    h_type: str,
    original: str,
    hallucinated: str,
    changed_span: dict[str, str],
) -> tuple[bool, str]:
    """rule_validator에 위임."""
    return validate(h_type, original, hallucinated, changed_span)

# 4. Judge LLM 평가
def judge_hallucination(
    original: str,
    hallucinated: str,
    h_type: str,
) -> dict:
    """
    Judge LLM으로 그럴듯함(plausibility)과 오류 미묘함(subtlety) 평가.
    """
    prompt = JUDGE_PROMPT.format(
        original=original,
        hallucinated=hallucinated,
        hallucination_type=h_type,
    )
    try:
        raw = call_with_retry(prompt, model=JUDGE_MODEL)
    except Exception as e:
        print(f"    [Judge API 오류]: {e}")
        return {
            "plausibility_score": None,
            "subtlety_score":     None,
            "judge_reason":       f"judge_api_failure: {e}",
            "needs_revalidation": True,
        }

    parsed = parse_json_response(raw)
    if parsed is None:
        print(f"    [Judge 파싱 오류]: JSON 추출 실패")
        return {
            "plausibility_score": None,
            "subtlety_score":     None,
            "judge_reason":       "parse_failure",
            "needs_revalidation": True,
        }

    p_score = parsed.get("plausibility_score")
    s_score = parsed.get("subtlety_score")
    reason  = parsed.get("judge_reason", "")

    if not isinstance(p_score, (int, float)):
        print(f"    [Judge 오류]: plausibility_score 타입 이상 ({type(p_score)})")
        return {
            "plausibility_score": None,
            "subtlety_score":     int(s_score) if isinstance(s_score, (int, float)) else None,
            "judge_reason":       reason or "parse_failure",
            "needs_revalidation": True,
        }

    needs_rev = False
    if not isinstance(s_score, (int, float)):
        print(f"    [Judge 오류]: subtlety_score 타입 이상 ({type(s_score)}) → None 유지")
        s_score   = None
        needs_rev = True
        reason    = reason or "subtlety_omission"

    return {
        "plausibility_score": int(p_score),
        "subtlety_score":     int(s_score) if s_score is not None else None,
        "judge_reason":       reason,
        "needs_revalidation": needs_rev,
    }

# 5. 체크포인트 저장
def save_checkpoint(dataset: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"  [체크포인트 저장: {len(dataset)}개]")

def _validate_single_span(h_type: str, span: dict, original: str) -> str | None:
    """위반 시 오류 메시지, 통과 시 None"""
    if h_type == "Condition_Addition" and span.get("before") != "":
        return f"before가 비어있지 않음: '{span['before'][:30]}'"
    if h_type == "Information_Omission" and span.get("after") != "":
        return f"after가 비어있지 않음: '{span['after'][:30]}'"
    if h_type in ("Number_Manipulation", "Legal_Effect_Reversal", "Entity_Substitution"):
        if not span.get("before") or not span.get("after"):
            return f"before/after 중 하나가 비어있음"
    if h_type == "Entity_Substitution":
        before = span.get("before", "")
        if "..." in before or "…" in before:
            return f"before에 생략 표기(...) 포함 — 비연속 span 의심: '{before[:30]}'"
        if before and before not in original:
            return f"before가 원본의 연속 부분문자열이 아님: '{before[:30]}'"
    return None

def _has_number(text: str) -> bool:
    # "제2조제1항제3호" 같은 연쇄 참조를 완전히 제거하기 위해 반복 적용
    # 단일 re.sub은 "제2조"만 제거하고 "제1항제3호"가 잔존할 수 있음
    pattern = re.compile(r"제\s*\d+\s*(조|항|호|장|절)(\s*의\s*\d+)?")
    prev = None
    text_wo_citation = text
    while prev != text_wo_citation:
        prev = text_wo_citation
        text_wo_citation = pattern.sub("", text_wo_citation)

    return bool(
        re.search(
            r"\d+\s*(일|개월|년|세|인|명|원|%|퍼센트|배|회|차)"
            r"|[일이삼사오육칠팔구십백천만]+\s*(일|개월|년|세|인|명|배|회)",
            text_wo_citation,
        )
    )

def _article_sort_key(article_no: str) -> tuple[int, int]:
    match = re.match(r"^\s*(\d+)(?:\s*의\s*(\d+))?\s*$", str(article_no))
    if not match:
        return (10**9, 0)
    return (int(match.group(1)), int(match.group(2) or 0))


def _parse_article_bound(article_no: str | None) -> tuple[int, int] | None:
    if article_no is None:
        return None
    parsed = _article_sort_key(article_no)
    if parsed[0] == 10**9:
        raise ValueError(f"조문번호 형식을 해석할 수 없습니다: {article_no}")
    return parsed


def _load_articles(laws: list[str]) -> list[dict]:
    all_articles: list[dict] = []
    for law in laws:
        with open(RAW_DATA_DIR / f"{law}.json", encoding="utf-8") as f:
            articles = json.load(f)
            for article in articles:
                article["법령명"] = law
            all_articles.extend(articles)
    return all_articles


def _filter_articles(
    articles: list[dict],
    start_article: str | None = None,
    end_article: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    start_key = _parse_article_bound(start_article)
    end_key = _parse_article_bound(end_article)

    filtered = []
    for article in articles:
        key = _article_sort_key(article.get("조문번호", ""))
        if start_key and key < start_key:
            continue
        if end_key and key > end_key:
            continue
        filtered.append(article)

    filtered.sort(key=lambda a: (a.get("법령명", ""), _article_sort_key(a.get("조문번호", ""))))
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM 기반 법령 환각 데이터셋을 생성합니다."
    )
    parser.add_argument(
        "--law",
        action="append",
        dest="laws",
        help="생성 대상 법령명. 여러 번 지정 가능. 기본값: 3개 법령 전체",
    )
    parser.add_argument(
        "--start-article",
        help="시작 조문번호. 예: 1, 18의2",
    )
    parser.add_argument(
        "--end-article",
        help="끝 조문번호. 예: 10, 19의2",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="필터링 후 앞에서 N개 조문만 실행",
    )
    parser.add_argument(
        "--ignore-test-mode",
        action="store_true",
        help="config.py의 TEST_MODE/TEST_SIZE 제한을 무시",
    )
    return parser.parse_args()

# severity와 subtlety_score 간 불일치 감지
# severity: high/medium/low → 각각 subtlety 기대 범위 정의
_SEVERITY_SUBTLETY_EXPECTED: dict[str, tuple[int, int]] = {
    "high":   (3, 5),
    "medium": (2, 4),
    "low":    (1, 3),
}

# 유형별 override — severity 기반 기본값보다 우선 적용
# Legal_Effect_Reversal: 역전은 구조상 눈에 띄는 게 정상 → subtlety 낮아도 무방
# Number_Manipulation:   숫자 변경은 미묘해야 품질 좋은 샘플 → subtlety 높아야 함
_TYPE_SUBTLETY_EXPECTED: dict[str, tuple[int, int]] = {
    HallucinationType.LEGAL_EFFECT_REVERSAL.value: (1, 3),
    HallucinationType.NUMBER_MANIPULATION.value:   (3, 5),
    # 근로자/사용자 직접 치환은 구조상 눈에 잘 띔 (10건 표본 중 6건 subtlety=2 일관 재현)
    HallucinationType.ENTITY_SUBSTITUTION.value:   (1, 3),
}

def _is_severity_subtlety_mismatch(
    severity: str,
    subtlety_score: int | None,
    h_type: str = "",
) -> bool:
    if subtlety_score is None:
        return False
    # 유형별 override가 있으면 우선 적용
    if h_type and h_type in _TYPE_SUBTLETY_EXPECTED:
        low, high = _TYPE_SUBTLETY_EXPECTED[h_type]
        return not (low <= subtlety_score <= high)
    # 없으면 severity 기반 기본값 사용
    expected = _SEVERITY_SUBTLETY_EXPECTED.get(severity)
    if expected is None:
        return False
    low, high = expected
    return not (low <= subtlety_score <= high)

# 6. 메인 루프
def generate_llm_hallucinations(
    articles: list[dict],
) -> tuple[list[dict], dict]:
    """전체 파이프라인 실행."""
    dataset:            list[dict]              = []
    seen:               set[str]                = set()
    success_count:      Counter                 = Counter()
    revalidation_count: Counter                 = Counter()
    fail_count:         Counter                 = Counter()
    # 신규: 유형별 실패 원인 세부 카운터
    fail_reason_count:  dict[str, Counter]      = {
        h: Counter() for h in HALLUCINATION_TYPES
    }
    last_checkpoint = 0

    checkpoint_path = PROCESSED_DATA_DIR / "checkpoint.json"

    for i, article in enumerate(articles):
        content = article["조문내용"]
        print(f"\n[{i+1}/{len(articles)}] {article['법령명']} 제{article['조문번호']}조")

        for h_type in HALLUCINATION_TYPES:
            if h_type == HallucinationType.NUMBER_MANIPULATION.value:
                if not _has_number(content):
                    print(f"  {h_type}: 스킵 (숫자 없는 조문)")
                    fail_count[h_type] += 1
                    fail_reason_count[h_type]["no_number"] += 1
                    continue
            try:
                gen, fail_reason = generate_hallucination(content, h_type, article)
                time.sleep(API_DELAY)

                if gen is None:
                    print(f"  {h_type}: 스킵 ({fail_reason})")
                    fail_count[h_type] += 1
                    fail_reason_count[h_type][fail_reason] += 1
                    continue

                hallucinated       = gen["hallucinated"]
                change_description = gen["change_description"]
                changed_span       = gen["changed_span"]

                dedup_key = hashlib.md5(
                    f"{h_type}|{content.strip()}|{hallucinated}".encode()
                ).hexdigest()
                if dedup_key in seen:
                    print(f"  {h_type}: 스킵 (중복)")
                    fail_count[h_type] += 1
                    fail_reason_count[h_type]["duplicate"] += 1
                    continue
                seen.add(dedup_key)

                is_valid, reason = validate_hallucination(
                    h_type, content, hallucinated, changed_span
                )
                if not is_valid:
                    print(f"  {h_type}: 스킵 (Rule 검증 실패 — {reason})")
                    fail_count[h_type] += 1
                    fail_reason_count[h_type]["rule_fail"] += 1
                    # 디버그: Condition_Deletion 실패 시 원문/마커 매칭 상태 출력
                    # if h_type == HallucinationType.CONDITION_DELETION.value:
                    #     from src.markers import CONDITION_DELETION_MARKERS
                    #     present = [m for m in CONDITION_DELETION_MARKERS if m in content]
                    #     print(f"    [DEBUG] 원문: {content[:200]}")
                    #     print(f"    [DEBUG] 원문에 존재하는 마커: {present}")
                    continue

                # 추가: span 제약 즉시 검증
                span_error = _validate_single_span(h_type, changed_span, content)
                if span_error:
                    print(f"  {h_type}: 스킵 (changed_span 제약 위반 — {span_error})")
                    fail_count[h_type] += 1
                    fail_reason_count[h_type]["span_invalid"] += 1
                    continue

                severity = calculate_severity(
                    h_type, content, hallucinated, changed_span
                )

                judge_result = judge_hallucination(content, hallucinated, h_type)
                time.sleep(API_DELAY)

                needs_rev = judge_result["needs_revalidation"]
                p = judge_result["plausibility_score"]
                s = judge_result["subtlety_score"]

                if p is not None:
                    if p < 3:
                        print(f"  {h_type}: Judge plausibility {p}점 → 스킵 (문체 부자연)")
                        fail_count[h_type] += 1
                        fail_reason_count[h_type]["judge_plausibility"] += 1
                        continue
                    if s is not None and s < 2:
                        print(f"  {h_type}: Judge subtlety {s}점 → 스킵 (오류 너무 명백)")
                        fail_count[h_type] += 1
                        fail_reason_count[h_type]["judge_subtlety"] += 1
                        continue

                if needs_rev:
                    print(f"  {h_type}: needs_revalidation 마킹 ({judge_result['judge_reason']})")

                sample_id = hashlib.md5(
                    f"{article['법령명']}|{article['조문번호']}|{h_type}|{hallucinated}".encode()
                ).hexdigest()

                mismatch = _is_severity_subtlety_mismatch(severity, s, h_type)
                
                dataset.append({
                    "id": sample_id,
                    "law_name":      article["법령명"],
                    "article_no":    article["조문번호"],
                    "article_title": article["조문제목"],
                    "context": content,
                    "answer":  hallucinated,
                    "hallucination_type": h_type,
                    "severity":           severity,
                    "subtlety_score":     judge_result["subtlety_score"],
                    "severity_subtlety_mismatch": mismatch,   # 추가
                    "label_gt":           "Not_Faithful",
                    "changed_span":       changed_span,
                    "change_description": change_description,
                    "plausibility_score": judge_result["plausibility_score"],
                    "judge_reason":       judge_result["judge_reason"],
                    "needs_revalidation": needs_rev,
                    "source":             "llm_generated",
                })

                if needs_rev:
                    revalidation_count[h_type] += 1
                    print(
                        f"  {h_type} ⚠️  ({severity}) | {judge_result['judge_reason']} | {change_description}"
                    )
                else:
                    success_count[h_type] += 1
                    print(
                        f"  {h_type}✅ ({severity}) "
                        f"| plausibility {p} subtlety {s} "
                        f"| {change_description}"
                    )

                if mismatch:
                    print(
                        f"    ↳ severity/subtlety 불일치: "
                        f"severity={severity}, subtlety_score={s} "
                        f"(기대 범위: {_SEVERITY_SUBTLETY_EXPECTED.get(severity)})"
                    )

                if len(dataset) - last_checkpoint >= CHECKPOINT_INTERVAL:
                    save_checkpoint(dataset, checkpoint_path)
                    last_checkpoint = len(dataset)

            except KeyError as e:
                print(f"  {h_type} [키 오류 — 코드 버그 가능성]: {e}")
                fail_count[h_type] += 1
                fail_reason_count[h_type]["key_error"] += 1
            except Exception as e:
                print(f"  {h_type} [알 수 없는 오류]: {type(e).__name__}: {e}")
                fail_count[h_type] += 1
                fail_reason_count[h_type]["unknown"] += 1

        time.sleep(ARTICLE_DELAY)

    return dataset, {
        "success":      success_count,
        "revalidation": revalidation_count,
        "fail":         fail_count,
        "fail_reason":  fail_reason_count,  # 신규
    }

# 7. 엔트리포인트
if __name__ == "__main__":
    import time as _time

    args = _parse_args()
    log_dir = PROCESSED_DATA_DIR.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_dir / f"run_{timestamp}.log"
    log_file = open(log_filename, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log_file)
    print(f"[로그 파일] {log_filename}")
    
    target_laws = args.laws or ["근로기준법", "남녀고용평등법", "고용보험법"]
    all_articles = _load_articles(target_laws)
    all_articles = _filter_articles(
        all_articles,
        start_article=args.start_article,
        end_article=args.end_article,
        limit=args.limit,
    )

    if TEST_MODE and not args.ignore_test_mode and args.limit is None:
        all_articles = all_articles[:TEST_SIZE]

    selected = ", ".join(
        f"{a['법령명']} 제{a['조문번호']}조" for a in all_articles[:10]
    )
    if len(all_articles) > 10:
        selected += f" 외 {len(all_articles) - 10}개"

    print(f"대상 법령: {', '.join(target_laws)}")
    if args.start_article or args.end_article:
        print(
            "조문 범위: "
            f"{args.start_article or '처음'} ~ {args.end_article or '끝'}"
        )
    print(f"선택 조문: {selected or '(없음)'}")
    print(f"전체 조문: {len(all_articles)}개")
    print(f"예상 생성: 최대 {len(all_articles) * len(HALLUCINATION_TYPES)}개\n")

    if not all_articles:
        raise SystemExit("선택된 조문이 없습니다. --law 또는 조문 범위를 확인하세요.")

    dataset, counts = generate_llm_hallucinations(all_articles)

    types       = Counter(d["hallucination_type"]  for d in dataset)
    severities  = Counter(d["severity"]            for d in dataset)
    p_scores    = Counter(d["plausibility_score"]  for d in dataset)
    s_scores    = Counter(d["subtlety_score"]      for d in dataset)

    span_filled = sum(
        1 for d in dataset
        if d["changed_span"].get("before") or d["changed_span"].get("after")
    )

    # 유형별 채움률 추가
    span_filled_by_type: dict[str, tuple[int, int]] = {}
    for h_type in HALLUCINATION_TYPES:
        type_samples = [d for d in dataset if d["hallucination_type"] == h_type]
        filled = sum(
            1 for d in type_samples
            if d["changed_span"].get("before") or d["changed_span"].get("after")
        )
        span_filled_by_type[h_type] = (filled, len(type_samples))

    print(f"changed_span 채움률 (전체): {span_filled}/{len(dataset)} "
        f"({span_filled/max(len(dataset),1):.1%})")
    print("\n[유형별 changed_span 채움률]")
    for h_type, (filled, total) in span_filled_by_type.items():
        if total:
            print(f"  {h_type}: {filled}/{total} ({filled/total:.1%})")

    print("\n[유형별]")
    for k, v in sorted(types.items()):
        print(f"  {k}: {v}개")

    print("\n[severity별]")
    for k, v in sorted(severities.items()):
        print(f"  {k}: {v}개")

    print("\n[Judge plausibility 점수별]")
    for k, v in sorted(p_scores.items()):
        print(f"  {k}점: {v}개")

    print("\n[Judge subtlety 점수별]")
    for k, v in sorted(s_scores.items()):
        print(f"  {k}점: {v}개")

    scope_descs = [
        d["change_description"] for d in dataset
        if d["hallucination_type"] == "Scope_Manipulation"
    ]
    if scope_descs:
        print("\n[Scope_Manipulation 생성 분포]")
        for k, v in sorted(Counter(scope_descs).items()):
            print(f"  {k}: {v}개")

    print("\n[유형별 성공률]")
    for h_type in HALLUCINATION_TYPES:
        s = counts["success"][h_type]
        r = counts["revalidation"][h_type]
        f = counts["fail"][h_type]
        total = s + r + f
        if total:
            print(
                f"  {h_type}: "
                f"검증완료 {s/total:.1%} | "
                f"재검증필요 {r/total:.1%} | "
                f"실패 {f/total:.1%} ({total}건)"
            )

    # 신규: 유형별 실패 원인 상세 통계
    print("\n[유형별 실패 원인 상세]")
    for h_type in HALLUCINATION_TYPES:
        reasons = counts["fail_reason"][h_type]
        if reasons:
            reason_str = " | ".join(f"{k}: {v}건" for k, v in sorted(reasons.items()))
            print(f"  {h_type}: {reason_str}")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 엔트리포인트 통계 출력부에 추가
    print("\n[실패 원인 전체 요약 top-3]")
    total_reasons: Counter = Counter()
    for h_type in HALLUCINATION_TYPES:
        total_reasons.update(counts["fail_reason"][h_type])
    for reason, cnt in total_reasons.most_common(3):
        print(f"  {reason}: {cnt}건")

    verified     = [d for d in dataset if not d.get("needs_revalidation")]
    revalidation = [d for d in dataset if d.get("needs_revalidation")]

    with open(
        PROCESSED_DATA_DIR / "llm_hallucinations_verified.json", "w", encoding="utf-8"
    ) as f:
        json.dump(verified, f, ensure_ascii=False, indent=2)

    with open(
        PROCESSED_DATA_DIR / "llm_hallucinations_needs_revalidation.json",
        "w", encoding="utf-8",
    ) as f:
        json.dump(revalidation, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료")
    print(f"✅ verified           → {PROCESSED_DATA_DIR}/llm_hallucinations_verified.json ({len(verified)}개)")
    print(f"📁 needs_revalidation → {PROCESSED_DATA_DIR}/llm_hallucinations_needs_revalidation.json ({len(revalidation)}개)")

    if revalidation:
        print(f"\n⚠️  재검증 필요: {len(revalidation)}개")

    stats = {
        "timestamp":           _time.strftime("%Y-%m-%d %H:%M:%S"),
        "generator_model":     GENERATOR_MODEL,
        "judge_model":         JUDGE_MODEL,
        "total_generated":     len(dataset),
        "verified":            len(verified),
        "needs_revalidation":  len(revalidation),
        "changed_span_fill_rate": f"{span_filled/max(len(dataset),1):.1%}",
            "changed_span_fill_rate_by_type": {
                h_type: {
                    "filled": filled,
                    "total": total,
                    "rate": f"{filled/total:.1%}" if total else "0.0%",
                }
                for h_type, (filled, total) in span_filled_by_type.items()
            },        "success_rate": {
            h_type: {
                "verified":           counts["success"][h_type],
                "needs_revalidation": counts["revalidation"][h_type],
                "fail":               counts["fail"][h_type],
                "fail_reason":        dict(counts["fail_reason"][h_type]),  # 신규
                "total": (
                    counts["success"][h_type]
                    + counts["revalidation"][h_type]
                    + counts["fail"][h_type]
                ),
            }
            for h_type in HALLUCINATION_TYPES
        },
    }
    with open(
        PROCESSED_DATA_DIR / "generation_stats.json", "w", encoding="utf-8"
    ) as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"stats → {PROCESSED_DATA_DIR}/generation_stats.json")
