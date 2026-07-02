from itertools import combinations


# 핵심 법적 주체. 이들 사이의 치환은 권리·의무 주체를 바꿀 가능성이 높다.
CORE_LEGAL_SUBJECTS = {"근로자", "사용자", "사업주", "고용노동부장관"}
HIGH_RISK_SUBJECT_PAIRS = list(combinations(CORE_LEGAL_SUBJECTS, 2))


# 조건 삭제 검증과 severity 평가가 공유하는 마커.
CONDITION_DELETION_MARKERS = [
    "다만", "단,", "단 ", "단서", "예외", "경우에는", "경우를 제외",
    "동등한 지위에서", "자유의사에 따라", "성실하게", "헌법에 따라",
]

HIGH_IMPACT_CONDITION_DELETION_MARKERS = [
    "자유의사에 따라", "동등한 지위에서", "성실하게", "헌법에 따라",
]


INFORMATION_OMISSION_HIGH_IMPACT_MARKERS = [
    "금지", "금지한다", "하여서는 아니 된다", "할 수 없다",
    "하지 못한다", "거부하지 못한다", "차별적 처우를 하지 못한다",
    "강요하지 못한다", "폭행하지 못한다",
]

INFORMATION_OMISSION_MEDIUM_IMPACT_MARKERS = [
    "하여야 한다", "의무", "권리", "청구", "신청", "보장",
    "지급", "차별", "강요", "거부", "다만",
]
