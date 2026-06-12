from __future__ import annotations

from app.agent.heavy.rag_schemas import RetrievalRequest
from app.schemas import RagChunk
from app.services.rag_service import RagService


class Retriever:
    """Executes retrieval requests against the configured RAG service."""

    def __init__(self, rag_service: RagService):
        self.rag_service = rag_service

    def retrieve(self, request: RetrievalRequest) -> list[RagChunk]:
        return self.rag_service.search(request.query, top_k=request.top_k, filters=request.filters)

