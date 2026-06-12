from __future__ import annotations

from app.schemas import RagChunk


class EvidenceFilter:
    """Filters retrieval output without assigning final relevance grades."""

    def filter(self, chunks: list[RagChunk]) -> list[RagChunk]:
        seen: set[str] = set()
        filtered: list[RagChunk] = []
        for chunk in chunks or []:
            if not chunk.text.strip():
                continue
            key = chunk.chunk_id or f'{chunk.source}:{chunk.document_title}:{chunk.text[:80]}'
            if key in seen:
                continue
            seen.add(key)
            filtered.append(chunk)
        return filtered

