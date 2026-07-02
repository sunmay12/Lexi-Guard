import re

from src.hallucination_types import HallucinationType
from src.markers import (
    HIGH_IMPACT_CONDITION_DELETION_MARKERS,
    HIGH_RISK_SUBJECT_PAIRS,
    INFORMATION_OMISSION_HIGH_IMPACT_MARKERS,
    INFORMATION_OMISSION_MEDIUM_IMPACT_MARKERS,
)
"""
severity.py
변경사항:
  - Entity_Substitution: count 기반 zeroed_out 로직 제거.
    "근로자→사용자" 또는 "사용자→근로자" 직접 치환 여부를 명시적으로 확인.
    count 기반 추정은 문맥/역할을 무시해 오판 가능성이 있었음.
  - CORE_LEGAL_SUBJECTS 확장: 근로자/사용자 외 사업주, 고용노동부장관 추가.
    데이터 범위가 넓어지면 두 개만으로는 깨짐.
  - Condition_Deletion: 마커별로 "원문에 있었는데 사라졌는지"를
    명시적으로 순회 확인 (이전 버전은 로직이 불명확했음).
  - 분류 기준:
      High   — 권리/의무 역전, 주체 역전, 금지↔허용, 처벌 조항 변경
      Medium — 범위 제한, 숫자 변경, 조건 추가
      Low    — 부가설명 삭제, 정의 일부 누락
"""

def calculate_severity(
    h_type: str,
    original: str,
    hallucinated: str,
    changed_span: dict | None = None,
) -> str:
    if h_type == HallucinationType.ARTICLE_MIXING.value:
        return "high"

    if h_type == HallucinationType.NO_HALLUCINATION.value:
        return "none"

    if h_type == HallucinationType.LEGAL_EFFECT_REVERSAL.value:
        return "high"

    if h_type == HallucinationType.NUMBER_MANIPULATION.value:
        return _number_severity(original, hallucinated)

    if h_type == HallucinationType.CONDITION_DELETION.value:
        return _condition_deletion_severity(original, hallucinated)

    if h_type == HallucinationType.INFORMATION_OMISSION.value:
        return _information_omission_severity(changed_span)

    if h_type == HallucinationType.ENTITY_SUBSTITUTION.value:
        return _entity_substitution_severity(original, hallucinated)

    if h_type == HallucinationType.SCOPE_MANIPULATION.value:
        return "medium"

    if h_type == HallucinationType.CONDITION_ADDITION.value:
        return "medium"

    return "medium"

def _number_severity(original: str, hallucinated: str) -> str:
    orig_nums = [int(n) for n in re.findall(r'\d+', original)]
    hall_nums = [int(n) for n in re.findall(r'\d+', hallucinated)]

    if not orig_nums or not hall_nums:
        return "medium"

    if len(orig_nums) != len(hall_nums):
        return "medium"

    changed_ratios = []

    for o, h in zip(orig_nums, hall_nums):
        if o == h:
            continue
        if o == 0:
            return "high"
        changed_ratios.append(abs(o - h) / o)

    if not changed_ratios:
        return "low"

    max_ratio = max(changed_ratios)
    if max_ratio < 0.2:
        return "low"
    elif max_ratio < 0.5:
        return "medium"
    else:
        return "high"

def _entity_substitution_severity(original: str, hallucinated: str) -> str:
    """
    주체 역전 판정 — 직접 치환 탐지 방식으로 단순화.

    이전 버전 문제:
        count() 기반으로 "원문엔 있었는데 환각문엔 0개"를 추정했으나
        문맥/역할을 무시한 휴리스틱이라 오판 가능성이 있었음.

    수정:
        원문에 주체 A가 있고 환각문에 주체 B가 등장하는
        "직접 역전 패턴"이 보이면 high로 판정.
        그 외 일반 명사·객체 치환은 medium.
    """
    for subject_a, subject_b in HIGH_RISK_SUBJECT_PAIRS:
        if (subject_a in original and subject_b in hallucinated) or \
           (subject_b in original and subject_a in hallucinated):
            return "high"

    return "medium"

def _information_omission_severity(changed_span: dict | None = None) -> str:
    if not changed_span:
        return "low"

    omitted = str(changed_span.get("before", ""))
    if any(marker in omitted for marker in INFORMATION_OMISSION_HIGH_IMPACT_MARKERS):
        return "high"
    if any(marker in omitted for marker in INFORMATION_OMISSION_MEDIUM_IMPACT_MARKERS):
        return "medium"

    return "low"

def _condition_deletion_severity(original: str, hallucinated: str) -> str:
    """
    조건/단서 삭제 중에서도 핵심 권리·의무 요건 삭제는 high,
    일반적인 단서조항·예외 삭제는 medium.

    변경:
        마커별로 "원문에 존재했는데 환각문에서 사라졌는지"를
        명시적으로 순회하며 확인 (이전 any() 중첩 표현은 모호했음).
    """
    for marker in HIGH_IMPACT_CONDITION_DELETION_MARKERS:
        was_present = marker in original
        still_present = marker in hallucinated
        if was_present and not still_present:
            return "high"

    return "medium"
