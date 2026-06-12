# Feature Registry

기능을 추가할 때마다 이 문서에 누적한다.

## F-001: LLM-only Agent Runtime

- Status: done
- User value: 실제 LLM 기반 Agent 동작만 유지해 데모/운영 경로를 명확히 함
- Main files: `ai_server/app/services/llm_service.py`, `ai_server/app/agent/graph.py`, `ai_server/app/schemas.py`, `streamlit_app.py`
- API/UI entry: `/agent/send`, Streamlit Agent tab
- Data dependency: `OPENAI_API_KEY`
- Cost impact: 모든 Agent 실행이 실제 LLM 비용을 발생시킬 수 있음
- Safety impact: template/mock 응답으로 안전 검증을 흐리지 않음
- Observability: LLM usage span, agent run span
- Tests: compileall, pytest, `/health`, `/llm/models`
- Demo steps: Streamlit에서 모델 선택 후 Agent 실행
- Follow-up: 요청당 max cost guard 추가

## F-002: Supervisor Re-plan Loop

- Status: done
- User value: 근거 부족, 파싱 실패, 안전 검증 실패 시 재계획 후 재시도
- Main files: `ai_server/app/services/supervisor_service.py`, `ai_server/app/agent/graph.py`
- API/UI entry: `/agent/send/stream`, Streamlit 진행 trace
- Data dependency: RAG search results, safety validation errors
- Cost impact: re-plan과 retry가 LLM 호출 수를 늘릴 수 있음
- Safety impact: 안전 검증 실패를 반영한 재시도 가능
- Observability: `llm_usage.replan_count`
- Tests: RAG weak-context scenario, safety validator test
- Demo steps: 근거가 약한 질문에서 `Supervisor Re-plan` trace 확인
- Follow-up: re-plan 사유별 통계 추가

## F-003: Token/Cost Meter

- Status: done
- User value: 요청별 LLM 사용량과 예상 비용 확인
- Main files: `ai_server/app/services/llm_service.py`, `ai_server/app/services/observability_service.py`, `streamlit_app.py`
- API/UI entry: Agent response `llm_usage`, Streamlit metric
- Data dependency: OpenAI response `usage`
- Cost impact: 비용 가시화
- Safety impact: 없음
- Observability: `gen_ai.usage.*`, `gen_ai.request.model`
- Tests: 실제 LLM 호출 시 usage 확인, compileall, pytest
- Demo steps: Agent 실행 후 Estimated cost 확인
- Follow-up: 누적 budget, 사용자별 비용 집계

## F-004: Model Selection Policy

- Status: done
- User value: 사용 가능한 모델을 선택하되 고비용 모델은 비활성화
- Main files: `ai_server/app/config.py`, `ai_server/app/services/llm_service.py`, `ai_server/app/main.py`, `streamlit_app.py`
- API/UI entry: `/llm/models`, Streamlit model selectbox
- Data dependency: `LLM_MODEL_CATALOG`
- Cost impact: 고비용 모델 선택 방지
- Safety impact: 없음
- Observability: selected model recorded in usage records
- Tests: `/llm/models`에서 `gpt-5.5 selectable=false` 확인
- Demo steps: 모델 선택창에서 비활성 모델 목록 확인
- Follow-up: org/user role 기반 모델 정책

## F-005: User-scoped Context Engineering

- Status: done
- User value: 유저별 이전 실행 이력과 장기 memory를 활용해 더 일관된 Agent 답변 제공
- Main files: `schemas.py`, `main.py`, `storage`, `user_service.py`, `context_service.py`, `memory_service.py`, `streamlit_app.py`
- API/UI entry: `/users`, `/users/{user_id}/context`, `/agent/send`
- Data dependency: SQLite users/sessions/memories/agent_runs
- Cost impact: context injection으로 input token 증가 가능
- Safety impact: 과거 context가 현재 safety gate를 덮어쓰지 못하도록 우선순위 필요
- Observability: user hash, context count, estimated context tokens
- Tests: user isolation, delete cascade, context budget, memory extraction, agent history integration
- Demo steps: 유저 A/B를 만들고 각기 다른 history가 context에 반영되는지 확인
- Follow-up: embedding 기반 similar run retrieval

## F-006: Chroma PDF Ingestion

- Status: planned
- User value: PDF 제조 문서를 업로드하면 semantic RAG 검색에 즉시 반영
- Main files: `chroma_rag_service.py`, `embedding_service.py`, `document_ingestion_service.py`, `sqlite_store.py`, `streamlit_app.py`
- API/UI entry: `/rag/documents/upload`, `/rag/documents`, `/rag/search`, Streamlit RAG tab
- Data dependency: PDF files, OpenAI embeddings or fake embedding in tests
- Cost impact: embedding 생성 비용 발생
- Safety impact: safety procedure 문서 metadata filter와 citation 품질 개선
- Observability: embedding token usage, indexed chunk count, vector search latency
- Tests: PDF extraction, Chroma upsert/search, metadata filter, delete cascade
- Demo steps: PDF 업로드 후 관련 질문에서 새 문서 citation 확인
- Follow-up: hybrid BM25 + vector search, OCR fallback
