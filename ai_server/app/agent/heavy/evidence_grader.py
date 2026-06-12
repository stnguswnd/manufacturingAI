from __future__ import annotations

import re

from app.schemas import RagChunk
from app.agent.heavy.rag_schemas import EvidenceGrade


class EvidenceGrader:
    """Grades evidence relevance only; it does not retrieve or build citations."""

    STOPWORDS = {
        'the', 'and', 'for', 'with', 'what', 'how',
        '어떤', '있어', '대한', '알려줘', '찾아줘', '확인', '점검', '절차',
        '제조', '관련', '질의', '질문', '해야', '하는지',
    }

    def contexts_match_user_terms(self, question: str, contexts: list[RagChunk]) -> bool:
        terms = self._salient_terms(question)
        if not terms:
            return True
        blob = ' '.join([f'{chunk.document_title} {chunk.text} {chunk.source} {chunk.section or ""} {chunk.doc_type or ""}' for chunk in contexts]).lower()
        return any(term in blob for term in terms)

    def grade(self, question: str, contexts: list[RagChunk]) -> EvidenceGrade:
        if not contexts:
            return EvidenceGrade(
                usable=False,
                usable_chunks=0,
                weak_reason='no_retrieved_evidence',
                relevance_label='empty',
            )
        if not self.contexts_match_user_terms(question, contexts):
            return EvidenceGrade(
                usable=False,
                usable_chunks=len(contexts),
                weak_reason='no_direct_user_term_overlap',
                relevance_label='weak',
            )
        return EvidenceGrade(
            usable=True,
            usable_chunks=len(contexts),
            weak_reason=None,
            relevance_label='usable',
        )

    def _salient_terms(self, question: str) -> set[str]:
        terms: set[str] = set()
        for token in re.findall(r'[가-힣A-Za-z0-9_+#.-]+', question or ''):
            lowered = token.lower()
            stripped = lowered.rstrip('이가은는을를에의와과도')
            is_korean = bool(re.search(r'[가-힣]', lowered))
            if lowered in self.STOPWORDS or stripped in self.STOPWORDS:
                continue
            if (is_korean and len(stripped) >= 2) or (not is_korean and len(stripped) >= 3):
                terms.add(stripped)
        return terms
