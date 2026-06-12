from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, Field


class FollowupSignal(BaseModel):
    has_followup_marker: bool = False
    asks_reason: bool = False
    asks_recommended_actions: bool = False
    asks_recommended_action_item: bool = False
    item_index: Optional[int] = None
    asks_recap: bool = False
    is_short_ambiguous: bool = False
    is_clear_new_concept: bool = False
    markers: List[str] = Field(default_factory=list)


class FollowupSignalDetector:
    """Detects candidate follow-up signals.

    This class intentionally does not choose a route. It only converts surface
    wording into structured signals that ContextResolver can validate against
    AnswerMemory.
    """

    FOLLOWUP_MARKERS = ['이것', '이걸', '이건', '그것', '그걸', '그건', '방금', '앞에서', '직전', '위에서', '그중', '그 중']
    REASON_MARKERS = ['왜', '이유', '왜 그렇게', '그게 왜', '필요한데', '중요']
    ACTION_MARKERS = ['권장조치', '권장 조치', '조치', '점검', '순서', '우선순위', '중요한 순서', '먼저']
    RECAP_MARKERS = ['다시', '정리', '요약', '설명']
    NEW_CONCEPT_MARKERS = ['뭐야', '무엇', '정의', '란', '이란']

    @classmethod
    def detect(cls, message: str) -> FollowupSignal:
        compact = cls._compact(message)
        markers: List[str] = []

        def contains(terms: list[str]) -> bool:
            found = [term for term in terms if cls._compact(term) in compact]
            markers.extend(found)
            return bool(found)

        item_index = cls._item_index(message)
        return FollowupSignal(
            has_followup_marker=contains(cls.FOLLOWUP_MARKERS),
            asks_reason=contains(cls.REASON_MARKERS),
            asks_recommended_actions=contains(cls.ACTION_MARKERS),
            asks_recommended_action_item=item_index is not None,
            item_index=item_index,
            asks_recap=contains(cls.RECAP_MARKERS),
            is_short_ambiguous=len(compact) <= 8 and contains(cls.REASON_MARKERS + cls.FOLLOWUP_MARKERS),
            is_clear_new_concept=contains(cls.NEW_CONCEPT_MARKERS) and not item_index,
            markers=list(dict.fromkeys(markers)),
        )

    @staticmethod
    def _compact(text: str) -> str:
        return re.sub(r'\s+', '', text or '').lower()

    @staticmethod
    def _item_index(message: str) -> Optional[int]:
        match = re.search(r'(?:그\s*중|그중|위\s*항목|항목)\s*(\d+)\s*번', message or '')
        if match:
            return int(match.group(1))
        match = re.search(r'(\d+)\s*번(?:은|이|을|를)?\s*(?:왜|이유|필요)', message or '')
        if match:
            return int(match.group(1))
        return None
