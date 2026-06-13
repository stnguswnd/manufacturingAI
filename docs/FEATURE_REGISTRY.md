# Feature Registry

현재 포트폴리오 기준으로 유지하는 핵심 기능 목록입니다. 과거 계획성 항목이나 제거된 Report/legacy graph 경로는 포함하지 않습니다.

## F-001: LangGraph Root + SubAgent Runtime

- Status: done
- User value: 제조 Agent 실행 흐름을 책임별 StateGraph로 분리해 추적과 유지보수를 쉽게 함
- Main files: `ai_server/app/agent/root_graph.py`, `ai_server/app/agent/*_subagent/`, `ai_server/app/agent/rag_evidence/`
- API/UI entry: `/agent/send`, Streamlit Agent tab
- Data dependency: request, user/session context, domain YAML
- Safety impact: safety gate와 response validator를 Root flow에 통합
- Observability: route, trace, warnings, replan, llm_usage
- Tests: root graph, subagent, RAG/safety tests
- Demo steps: Streamlit trace에서 SubAgent 순서 확인

## F-002: AI4I Prediction Routing and Feature Audit

- Status: done
- User value: AI4I feature가 완전할 때만 예측하고, 불완전하면 누락값만 요청
- Main files: `context_subagent`, `root_graph.py`, `PredictionService`, `schemas/agent.py`
- API/UI entry: `/agent/send`
- Data dependency: Type, Air temperature, Process temperature, Rotational speed, Torque, Tool wear
- Safety impact: 예측하지 않은 값을 답변에서 지어내지 않음
- Observability: `prediction_called`, `prediction_skip_reason`, `missing_features`, `parsed_ai4i_features`
- Tests: complete feature prediction, missing feature clarification
- Demo steps: Type/Torque만 넣고 clarification 확인

## F-003: RAG Evidence SubAgent

- Status: done
- User value: 문서 근거 검색, 필터링, grading, citation을 독립 LangGraph SubAgent로 실행
- Main files: `ai_server/app/agent/rag_evidence/state.py`, `nodes.py`, `subagent.py`
- API/UI entry: `/agent/send`; `/rag/search`는 debug seam
- Data dependency: Chroma `manufacturing_rag`, `rag_chunks.jsonl`
- Safety impact: 문서 근거와 safety metadata를 함께 선택
- Observability: adaptive profile, selected sources, citation count, warnings
- Tests: RAG evidence orchestration, Chroma runtime RAG
- Demo steps: Haas troubleshooting 질문에서 citation 확인

## F-004: Adaptive RAG Profiles

- Status: done
- User value: 질문 유형별로 retrieval 전략을 달리해 문서 선택 오류와 출력 노이즈를 줄임
- Main files: `RagQueryPlanner`, `RagFanoutPolicy`, `rag_evidence/nodes.py`, `RagService`
- API/UI entry: `/agent/send`
- Data dependency: user question, diagnostic plan, safety gate metadata, prediction context
- Safety impact: safety 질문에서 safety gate/title metadata supplement 적용
- Observability: `retrieval_profile`, query spec names, selected safety gates
- Tests: planned query sanitize, RAG-only safety, prediction_plus_rag
- Demo steps: 드릴기 안전 질문과 AI4I prediction 질문 비교

## F-005: Safety Gate and Response Validation

- Status: done
- User value: LOTO, 회전부 방호, 정비 자격 등 필수 안전 확인을 빠뜨리지 않음
- Main files: `safety_subagent`, `safety_gate_matrix.yaml`, `SafetyValidationService`
- API/UI entry: `/agent/send`
- Data dependency: domain YAML, selected evidence, manufacturing context
- Safety impact: 금지 표현 또는 필수 gate 누락 시 replan/차단
- Observability: safety gates, validation status, blocked reason
- Tests: RAG and safety tests
- Demo steps: 회전부/공구 교체 질문에서 safety validation trace 확인

## F-006: Context and Memory

- Status: done
- User value: user/session별 이전 context를 참고하되 현재 입력과 safety를 우선함
- Main files: `context_subagent`, `memory_subagent`, `context_service.py`, `memory_service.py`, storage services
- API/UI entry: `/agent/send`, `/users`, history endpoints
- Data dependency: SQLite users/sessions/memories/agent_runs
- Safety impact: 과거 memory가 현재 safety gate를 덮어쓰지 않도록 supporting context로만 사용
- Observability: context summary, memory output, warnings
- Tests: context engineering, memory/focus tests
- Demo steps: 같은 session에서 follow-up 질문 실행

## F-007: LLM Model Policy and Cost Tracking

- Status: done
- User value: 모델 선택, 고비용 모델 제한, 요청별 token/cost 확인
- Main files: `config.py`, `llm_service.py`, `observability_service.py`, `streamlit_app.py`
- API/UI entry: `/llm/models`, `/agent/send`, Streamlit metrics/debug panel
- Data dependency: OpenAI-compatible response usage
- Cost impact: usage 기반 estimated cost 계산
- Safety impact: 모델 정책을 서버에서 강제
- Observability: model, input/output tokens, estimated cost
- Tests: model policy, usage records
- Demo steps: `/llm/models`와 Streamlit cost panel 확인

## F-008: RAG Index Rebuild Runbook

- Status: done
- User value: git ignored vector DB를 새 환경에서 재현 가능하게 함
- Main files: `ai_server/scripts/index_rag_chunks_chroma.py`, `docs/RAG_INDEX_RUNBOOK.md`
- API/UI entry: 운영 명령
- Data dependency: `ai_server/data/processed/rag_chunks.jsonl`, OpenAI embedding API
- Safety impact: 없음
- Observability: indexed chunk count, collection name, persist dir
- Tests: Chroma runtime RAG tests
- Demo steps: count 727 확인, Haas spindle query 확인

## F-009: Public Answer / Debug Separation

- Status: done
- User value: 사용자 답변은 간결하게 유지하고, run/debug/cost 정보는 상세 패널에만 노출
- Main files: `root_graph.py`, response schemas, `streamlit_app.py`, formatter/packager code
- API/UI entry: `/agent/send`, Streamlit Agent tab
- Data dependency: run trace, citations, llm_usage
- Safety impact: 내부 금지표현 목록이나 gate id가 사용자 답변에 새지 않음
- Observability: debug/trace/history에는 유지
- Tests: public answer does not expose debug metadata
- Demo steps: answer와 debug panel 분리 확인
