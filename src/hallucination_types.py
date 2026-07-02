from enum import Enum


class HallucinationType(Enum):
    NUMBER_MANIPULATION = "Number_Manipulation"
    CONDITION_DELETION = "Condition_Deletion"
    CONDITION_ADDITION = "Condition_Addition"
    INFORMATION_OMISSION = "Information_Omission"
    LEGAL_EFFECT_REVERSAL = "Legal_Effect_Reversal"
    SCOPE_MANIPULATION = "Scope_Manipulation"
    ENTITY_SUBSTITUTION = "Entity_Substitution"

    ARTICLE_MIXING = "Article_Mixing"
    NO_HALLUCINATION = "No_Hallucination"

    @classmethod
    def generation_values(cls):
        """LLM으로 생성할 환각 유형만 반환."""
        return [
            cls.NUMBER_MANIPULATION.value,
            cls.CONDITION_DELETION.value,
            cls.CONDITION_ADDITION.value,
            cls.INFORMATION_OMISSION.value,
            cls.LEGAL_EFFECT_REVERSAL.value,
            cls.SCOPE_MANIPULATION.value,
            cls.ENTITY_SUBSTITUTION.value,
        ]

    @classmethod
    def values(cls):
        """전체 라벨 공간 반환."""
        return [m.value for m in cls]