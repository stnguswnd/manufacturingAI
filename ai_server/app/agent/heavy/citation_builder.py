from __future__ import annotations

from typing import Any

from app.agent.heavy.rag_schemas import EvidenceGrade
from app.schemas import RagChunk


class CitationBuilder:
    """Builds citation metadata from already-filtered and graded evidence."""

    def build(self, chunks: list[RagChunk], grade: EvidenceGrade | None = None) -> list[dict[str, Any]]:
        if grade is not None and not grade.usable:
            return []
        citations: list[dict[str, Any]] = []
        for chunk in chunks or []:
            citations.append({
                'source': chunk.source,
                'document': chunk.document_title,
                'chunk_id': chunk.chunk_id,
                'url': chunk.url,
                'reason': 'graded_evidence' if grade else 'retrieved_evidence',
            })
        return citations

