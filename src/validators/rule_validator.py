"""
rule_validator.py — Rule-based 환각 검증

각 환각 유형별로 최소한의 구조적 조건을 검사.
LLM Judge 호출 전 빠른 필터링이 목적이므로,
False Negative(통과시켜야 할 걸 막음)보다
False Positive(잘못된 걸 통과시킴)를 줄이는 방향으로 설계.

반환값: (is_valid: bool, reason: str)

변경사항:
  - _validate_condition_deletion: 마커 확장 (프롬프트와 동기화)
  - _validate_condition_addition: 약한 단일 명사 마커 제거, 강한 구문 패턴만 인정
  - _validate_information_omission: after 비어있지 않으면 차단 (재작성 방지)
                                     누락 길이 판정을 before 텍스트 기준으로 변경
  - _validate_legal_effect_reversal: "하여서는 아니 된다" 계열 pair 추가
  - _validate_scope_manipulation: after 자체에도 scope 키워드 존재 확인
  - _validate_entity_substitution: 주체 순서 치환 차단 추가
"""

import re
from src.hallucination_types import HallucinationType
from src.markers import CONDITION_DELETION_MARKERS


# 공통 헬퍼

def _has_content(text: str) -> bool:
    return bool(text and text.strip())


def _texts_differ(a: str, b: str) -> bool:
    """공백·따옴표를 정규화한 뒤 실제로 달라졌는지 확인."""
    normalize = lambda s: re.sub(r"\s+", " ", s).strip().strip("'\"")
    return normalize(a) != normalize(b)


def _extract_numbers(text: str) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", text)]


# 유형별 검증

def _validate_number_manipulation(
    original: str, hallucinated: str, changed_span: dict
) -> tuple[bool, str]:
    orig_nums = _extract_numbers(original)
    hall_nums = _extract_numbers(hallucinated)

    if not orig_nums:
        return False, "원본에 숫자가 없음"
    if orig_nums == hall_nums:
        return False, "숫자가 실제로 변경되지 않음"

    # changed_span이 있으면 before 값이 숫자인지 추가 검증
    before = changed_span.get("before", "")
    if before and not re.search(r"\d", before):
        return False, f"changed_span.before에 숫자 없음: '{before}'"

    return True, ""


def _validate_condition_deletion(
    original: str, hallucinated: str, changed_span: dict
) -> tuple[bool, str]:
    """
    이전 버전 문제:
        원문에 마커가 있고 환각문이 짧으면 통과 → 문장 뒷부분 잘라내도 통과.
    수정:
        원문에 있던 조건 마커가 환각문에서 실제로 사라졌는지 확인.
        단순 truncation과 구분하기 위해 환각문이 문장 부호로 자연스럽게 끝나야 함.
    """
    present_markers = [m for m in CONDITION_DELETION_MARKERS if m in original]
    if not present_markers:
        return False, "원본에 삭제 가능한 조건절·단서조항 없음"

    if len(hallucinated) >= len(original):
        return False, "환각문이 원본보다 짧아지지 않음 (조건 미삭제 의심)"

    if not _texts_differ(original, hallucinated):
        return False, "원본과 환각문이 동일"

    # 마커가 한 조문에 여러 번 등장할 수 있으므로(예: "경우에는"),
    # 단순 존재 여부가 아니라 등장 횟수가 줄었는지로 판단.
    # changed_span.before가 있으면 그게 정확히 사라졌는지 우선 확인.
    before = changed_span.get("before", "")
    if before.strip():
        if before in hallucinated:
            return False, f"changed_span.before('{before[:30]}')가 환각문에 여전히 존재"
    else:
        # before가 없으면 마커 등장 횟수 비교로 fallback
        reduced = [
            m for m in present_markers
            if original.count(m) <= hallucinated.count(m)
        ]
        if reduced and len(reduced) == len(present_markers):
            return False, (
                f"삭제 대상 마커 '{present_markers[0]}'의 등장 횟수가 "
                f"줄지 않음 (원문 {original.count(present_markers[0])}회 → "
                f"환각문 {hallucinated.count(present_markers[0])}회)"
            )

    # 단순 truncation 감지: 환각문이 문장 종결 부호 없이 끝나면 의심
    if not re.search(r"[.。!?다라]$", hallucinated.strip()):
        return False, "환각문이 문장 종결 없이 끊겨 있음 (단순 truncation 의심)"

    return True, ""


def _validate_condition_addition(
    original: str, hallucinated: str, changed_span: dict
) -> tuple[bool, str]:
    """
    이전 버전 문제:
        `len(hallucinated) > len(original)`만 확인 → 문장 끝에 아무 텍스트나 붙여도 통과.
    수정:
        환각문에 조건 마커가 실제로 추가됐는지 확인.
        원문에는 없던 마커가 환각문에 생겼어야 함.
    """
    if len(hallucinated) <= len(original):
        return False, "환각문이 원본보다 길어지지 않음 (조건 미추가 의심)"
    if not _texts_differ(original, hallucinated):
        return False, "원본과 환각문이 동일"

    # 강한 패턴만 마커로 사용. "승인", "신청", "서면" 같은 단일 명사는
    # 원문에 우연히 존재할 수 있는 일반 단어라 신호로 부적합.
    # "~에 한하여", "~에 한한다" 처럼 조건절을 구성하는 구문 패턴만 인정.
    condition_markers = [
        "다만", "단,", "단 ",
        "경우에 한하여", "경우에만", "경우를 제외", "경우에는",
        "한하여", "한함", "에 한한다", "에 한정",
        "예외적으로",
    ]

    # 원문에 없던 마커가 환각문에 새로 등장해야 진짜 조건 추가
    new_markers = [
        m for m in condition_markers
        if m not in original and m in hallucinated
    ]
    if not new_markers:
        return False, "환각문에 새로운 조건 마커('다만', '한하여' 등)가 추가되지 않음"

    # changed_span.after가 있으면 환각문에 실제 포함됐는지 확인
    after = changed_span.get("after", "")
    if after and after not in hallucinated:
        return False, f"추가된 조건('{after[:30]}')이 환각문에 없음"

    # 참고: 삽입 위치로 인한 문장 구조 붕괴(예: 동사 앞에 조건을 끼워
    # 넣어 발생하는 의미적 중복·비문)는 표면 패턴 매칭으로는 신뢰성 있게
    # 탐지하기 어려움 (이런 경우 표현 등장 횟수 자체는 정상으로 카운트됨).
    # 이 문제는 rule_validator가 아니라 프롬프트의 삽입 위치 제약으로
    # 예방하는 것이 더 적절한 방어선임 (semantic 검증은 Judge LLM 영역).

    return True, ""


def _validate_information_omission(
    original: str, hallucinated: str, changed_span: dict
) -> tuple[bool, str]:
    """
    변경:
        changed_span.after가 비어있지 않으면 차단.
        Information_Omission은 "순수 삭제"만 허용되며,
        after에 텍스트가 있다는 건 재작성(다른 표현으로 대체)이 일어났다는 신호.
        재작성은 Legal_Effect_Reversal / Entity_Substitution 영역과 겹쳐
        라벨 오염을 일으키므로 여기서 명확히 차단.
    """
    after = changed_span.get("after", "")
    if after.strip():
        return False, (
            f"Information_Omission인데 after가 비어있지 않음 "
            f"(순수 삭제가 아닌 재작성 의심): '{after[:30]}'"
        )

    if len(hallucinated) >= len(original):
        return False, "환각문이 원본보다 짧아지지 않음 (정보 미누락 의심)"
    if not _texts_differ(original, hallucinated):
        return False, "원본과 환각문이 동일"

    # before가 비어있으면 LLM이 무엇을 누락시켰는지 스스로 특정
    # 못한 것이므로 fallback 없이 즉시 차단. fallback(전체 길이 차이)은
    # "진짜 누락"과 "사소한 표현 차이"를 구분하지 못해 품질 낮은
    # 샘플(예: change_description="누락할 중요한 내용이 없음")을 통과시킴.
    before = changed_span.get("before", "")
    if not before.strip():
        return False, "changed_span.before가 비어있음 (누락 대상 특정 불가 — fallback 차단)"
    if len(before.strip()) < 2:
        return False, f"누락된 텍스트가 너무 짧음 (changed_span.before='{before}')"
    else:
        # before가 비어있으면 fallback으로 전체 길이 차이 사용
        omitted_len = len(original) - len(hallucinated)
        if omitted_len < 5:
            return False, f"누락 길이가 너무 짧음 ({omitted_len}자)"

    # 문장 종결 검사: 쉼표·조사 등으로 어색하게 끊기면 차단.
    # 프롬프트에서 어미 다듬기를 요구하지만, 모델이 이를 따르지 않고
    # 단순 truncation만 하는 경우가 있어 rule 레벨에서 추가 방어.
    if not re.search(r"[.。!?다라]$", hallucinated.strip()):
        return False, "환각문이 문장 종결 없이 끊겨 있음 (미완성 문장, 어미 미조정 의심)"

    return True, ""


def _validate_legal_effect_reversal(
    original: str, hallucinated: str, changed_span: dict
) -> tuple[bool, str]:
    """
    원문의 법적 효과 표현(orig_phrase)이 환각문에서 역전 표현(rev_phrase)으로
    실제로 바뀌었는지 확인.

    이전 버전 문제:
        `orig_phrase in original`만 보고 return True → 환각문이 뭐든 통과.
    수정:
        orig_phrase가 원문에 있고 rev_phrase가 환각문에 있어야 통과.
        단, changed_span이 채워진 경우 해당 쌍을 우선 검증.
    """
    before = changed_span.get("before", "")
    after  = changed_span.get("after", "")
    if before and after:
        if before not in original:
            return False, f"before('{before}')가 원본에 없음"
        if after not in hallucinated:
            return False, f"after('{after}')가 환각문에 없음"
        if before == after:
            return False, "changed_span.before와 after가 동일함"
        return True, ""

    reversal_pairs = [
        ("할 수 있다",    "할 수 없다"),
        ("할 수 없다",    "할 수 있다"),
        ("하여야 한다",   "하지 아니한다"),
        ("하여야 한다",   "할 필요가 없다"),
        ("하여야 한다",   "하여서는 아니 된다"),
        ("하지 아니한다", "하여야 한다"),
        ("할 수 있다",    "하여서는 아니 된다"),
        ("금지한다",      "허용한다"),
        ("허용한다",      "금지한다"),
        ("의무가 있다",   "의무가 없다"),
        ("의무가 없다",   "의무가 있다"),
    ]

    matched_pairs = []
    for orig_phrase, rev_phrase in reversal_pairs:
        if orig_phrase in original:
            matched_pairs.append((orig_phrase, rev_phrase))

    if not matched_pairs:
        return False, "원문에 역전 가능한 법적 효과 표현 없음"

    # rev_phrase가 환각문에 있고, orig_phrase가 환각문에서 사라져야 통과
    for orig_phrase, rev_phrase in matched_pairs:
        if rev_phrase in hallucinated and orig_phrase not in hallucinated:
            return True, ""

    # 실패 원인을 구체적으로 분기
    for orig_phrase, rev_phrase in matched_pairs:
        if rev_phrase in hallucinated and orig_phrase in hallucinated:
            return False, (
                f"rev_phrase('{rev_phrase}')가 추가됐으나 "
                f"orig_phrase('{orig_phrase}')가 환각문에 여전히 존재 (부분 치환 의심)"
            )

    found_phrases = [p for p, _ in matched_pairs]
    return False, f"원문의 '{found_phrases[0]}' 등이 환각문에서 역전 표현으로 치환되지 않음"

def _validate_scope_manipulation(
    original: str, hallucinated: str, changed_span: dict
) -> tuple[bool, str]:
    """
    Scope_Manipulation은 생성 단계에서 지정한 forced_phrase가 실제 결과에
    반영됐는지를 우선 확인한다. forced_phrase가 없는 legacy 샘플은
    기존 범위 키워드 휴리스틱으로 fallback한다.
    """
    if not _texts_differ(original, hallucinated):
        return False, "원본과 환각문이 동일"

    before = changed_span.get("before", "")
    after  = changed_span.get("after", "")
    forced_phrase = changed_span.get("forced_phrase", "")

    if before and before not in original:
        return False, f"before('{before}')가 원본에 없음"
    if after and after not in hallucinated:
        return False, f"after('{after}')가 환각문에 없음"

    if forced_phrase:
        if forced_phrase in after or forced_phrase in hallucinated:
            return True, ""
        return False, (
            f"지정된 Scope 표현('{forced_phrase}')이 "
            "changed_span.after 또는 환각문에 없음"
        )

    scope_keywords = [
        "모든", "전체", "일부", "이상", "이하", "초과", "미만",
        "이내", "이전", "이후", "한정", "한하여", "에 한함",
        "만 ", "세 ", "인 ", "개월", "주간", "년간", "년",
        "정규직", "비정규직", "단시간", "통상", "기간제", "파견",
        "상시", "사업장", "수도권", "국내", "제조업",
    ]

    # changed_span이 채워진 경우: before는 원문에, after는 환각문에 있어야 함
    if before and after:
        # after 자체에 scope 키워드가 없으면 단순 텍스트 수정일 가능성
        # (예: "근로자" → "행복한 근로자" 같은 무관한 수식어 추가 차단)
        if not any(kw in after for kw in scope_keywords):
            return False, f"after('{after}')에 범위·수량 관련 키워드 없음 (Scope 변경 아닐 가능성)"
        return True, ""

    # changed_span이 없는 경우: 원문 또는 환각문에 범위 관련 키워드가 있어야 통과
    combined = original + hallucinated
    if not any(kw in combined for kw in scope_keywords):
        return False, "범위·수량 관련 키워드가 원문/환각문 어디에도 없음 (Scope 변경 아닐 가능성)"

    return True, ""

_NON_ENTITY_TERMS = {"근로시간", "임금", "보험료", "기간", "급여", "수당"}
def _validate_entity_substitution(
    original: str, hallucinated: str, changed_span: dict
) -> tuple[bool, str]:
    """
    변경:
        주체 순서 치환("근로자와 사용자" → "사용자와 근로자")은
        법적 의미 변화가 없으므로 차단 추가
        "사용자"는 근로기준법 2조상 사업주·사업경영담당자·사업주를 위해
        행위하는 자를 포괄하는 상위개념 ->
        "사업주"로 치환되면 범위가 좁아지는 진짜 의미 변화
        => {"사용자", "고용주", "사업주"} 동의어 그룹에서 분리
    """
    if not _texts_differ(original, hallucinated):
        return False, "원본과 환각문이 동일"

    before = changed_span.get("before", "")
    after  = changed_span.get("after", "")

    # 신규: before가 주체가 아닌 추상 개념이면 차단
    if before in _NON_ENTITY_TERMS:
        return False, f"before('{before}')는 법적 주체가 아닌 개념/수량 표현 — Entity_Substitution 부적합"
    
    # 수정 코드
    if before and after:
        synonyms = [
            {"근로자", "직원", "종업원"},
        ]
        for group in synonyms:
            if before in group and after in group:
                return False, f"동의어 치환 감지: '{before}' → '{after}'"

        before_tokens = {t.strip() for t in re.split(r"\s*[와과]\s*", before) if t.strip()}
        after_tokens  = {t.strip() for t in re.split(r"\s*[와과]\s*", after)  if t.strip()}
        if len(before_tokens) >= 2 and before_tokens == after_tokens:
            return False, f"주체 순서 치환만 발생 (법적 의미 변화 없음): '{before}' → '{after}'"

        # 신규: 양방향 동시 맞교환 감지
        # before가 원문에서 차지하던 위치에 after가 들어가는 동시에,
        # after가 원문에서 차지하던 위치에 before가 들어가면
        # changed_span 하나로는 설명 안 되는 이중 치환 (의미가 바뀌어도 추적 불가능한 데이터라 차단)
        core_subjects = {"근로자", "사용자", "사업주", "고용노동부장관"}
        if before in core_subjects and after in core_subjects:
            if after in original and before in hallucinated:
                return False, (
                    f"양방향 동시 치환 감지: '{before}'↔'{after}' "
                    f"두 주체가 서로 자리를 맞바꿈 (changed_span으로 추적 불가능한 이중 변경)"
                )

        if before not in original:
            return False, f"before('{before}')가 원본에 없음"
        if after not in hallucinated:
            return False, f"after('{after}')가 환각문에 없음"

    return True, ""

# 디스패치 테이블
_VALIDATORS = {
    HallucinationType.NUMBER_MANIPULATION.value:  _validate_number_manipulation,
    HallucinationType.CONDITION_DELETION.value:   _validate_condition_deletion,
    HallucinationType.CONDITION_ADDITION.value:   _validate_condition_addition,
    HallucinationType.INFORMATION_OMISSION.value: _validate_information_omission,
    HallucinationType.LEGAL_EFFECT_REVERSAL.value:_validate_legal_effect_reversal,
    HallucinationType.SCOPE_MANIPULATION.value:   _validate_scope_manipulation,
    HallucinationType.ENTITY_SUBSTITUTION.value:  _validate_entity_substitution,
}


# 공개 API

def validate(
    h_type: str,
    original: str,
    hallucinated: str,
    changed_span: dict[str, str],
) -> tuple[bool, str]:
    """
    유형별 Rule 검증 진입점.

    Args:
        h_type:       HallucinationType 값 문자열
        original:     원본 조문
        hallucinated: 환각 조문
        changed_span: {"before": ..., "after": ...}
                      비어있어도 동작하지만, 채워질수록 검증 정확도 향상

    Returns:
        (True, "")               — 검증 통과
        (False, "실패 이유")      — 검증 실패
    """
    # 공통: 원본과 환각문 존재 여부
    if not _has_content(original):
        return False, "원본 조문이 비어있음"
    if not _has_content(hallucinated):
        return False, "환각 조문이 비어있음"

    # 공통: 완전 동일 체크
    if not _texts_differ(original, hallucinated):
        return False, "원본과 환각문이 완전히 동일함"

    validator = _VALIDATORS.get(h_type)
    if validator is None:
        # 알 수 없는 유형은 기본 통과 (새 유형 추가 시 여기에 등록)
        return True, ""

    return validator(original, hallucinated, changed_span)
