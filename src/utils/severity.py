import re

from src.hallucination_types import HallucinationType
from src.markers import (
    HIGH_IMPACT_CONDITION_DELETION_MARKERS,
    HIGH_RISK_SUBJECT_PAIRS,
    INFORMATION_OMISSION_HIGH_IMPACT_MARKERS,
    INFORMATION_OMISSION_MEDIUM_IMPACT_MARKERS,
)

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
    주체 역전 판정 — 직접 치환 탐지 방식으로 단순화
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
    마커별로 "원문에 존재했는데 환각문에서 사라졌는지"를 명시적으로 순회하며 확인
    """
    for marker in HIGH_IMPACT_CONDITION_DELETION_MARKERS:
        was_present = marker in original
        still_present = marker in hallucinated
        if was_present and not still_present:
            return "high"

    return "medium"
