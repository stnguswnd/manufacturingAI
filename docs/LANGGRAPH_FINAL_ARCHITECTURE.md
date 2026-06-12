# LangGraph 최종 오케스트레이션 구조 정리

이 문서는 현재 제조 AI Agent에 구현된 LangGraph 기반 구조와, 다음 확장 설계를 논의하기 위한 기준 문서다.

핵심 방향은 다음이다.

```text
Raw User Message
→ Request Context
→ Intent Gateway
→ Fast Path 또는 Supervisor Planning
→ 필요한 제조 Subgraph만 조건부 실행
→ Response / Documentation
→ Focus Update
→ Audit / Persistence
```

현재 구현은 기존 `ManufacturingAgentGraph.run()` 중심 직렬 구조에서 벗어나, `RootManufacturingGraph`가 LangGraph `StateGraph`로 상위 orchestration을 담당하는 형태까지 전환되어 있다. MVP 안정화를 위해 SQLite persistent checkpointer, `user_id:session_id` 기반 thread 격리, process_data 오염 방지, No-LLM Fast Path를 우선 적용했다. RAG 내부의 query planner/retriever/grader/citation builder 분리와 `Send` 기반 병렬 검색은 아직 다음 확장 단계다.

---

## 1. 핵심 코드 위치

| 역할 | 파일 |
|---|---|
| Root LangGraph orchestration | `ai_server/app/agent/root_graph.py` |
| 공유 AgentState | `ai_server/app/agent/state.py` |
| 표준 trace helper | `ai_server/app/agent/trace.py` |
| 기존 제조 도메인 service/legacy graph | `ai_server/app/agent/graph.py` |
| Intent Gateway | `ai_server/app/services/intent_gateway_service.py` |
| Follow-up reference resolution | `ai_server/app/services/reference_resolution_service.py` |
| User-scoped context | `ai_server/app/services/context_service.py` |
| User memory update | `ai_server/app/services/memory_service.py` |
| API entrypoint | `ai_server/app/main.py` |
| 회귀 테스트 | `ai_server/tests/test_intent_gateway.py` |

---

## 2. 현재 Root Graph 구조

현재 `RootManufacturingGraph._build_graph()`는 다음 노드들을 등록한다.

```python
graph.add_node('request_context', self._request_context_node)
graph.add_node('intent_gateway', self._intent_gateway_node)
graph.add_node('fast_concept_answer', self._fast_concept_answer_node)
graph.add_node('lightweight_rag_answer', self._lightweight_rag_answer_node)
graph.add_node('unsupported_or_clarification', self._unsupported_or_clarification_node)
graph.add_node('supervisor_planning', self._supervisor_planning_node)
graph.add_node('manufacturing_analysis', self._manufacturing_analysis_node)
graph.add_node('evidence_retrieval', self._evidence_retrieval_node)
graph.add_node('safety', self._safety_node)
graph.add_node('response_synthesis', self._response_synthesis_node)
graph.add_node('documentation', self._documentation_node)
graph.add_node('response_packager', self._response_packager_node)
graph.add_node('focus_updater', self._focus_updater_node)
graph.add_node('audit_persistence', self._audit_persistence_node)
```

그래프 흐름은 아래와 같다.

```text
START
  ↓
request_context
  ↓
intent_gateway
  ├─ fast_concept_answer
  ├─ lightweight_rag_answer
  ├─ unsupported_or_clarification
  └─ supervisor_planning
        ↓
      manufacturing_analysis?
        ↓
      evidence_retrieval?
        ↓
      safety?
        ↓
      response_synthesis
        ↓
      documentation?
        ↓
      response_packager
  ↓
focus_updater
  ↓
audit_persistence
  ↓
END
```

`?`가 붙은 노드는 `AgentPlan`과 routing function에 따라 조건부로 실행된다.

---

## 3. Short-Term Memory 구조

현재 short-term memory는 LangGraph SQLite checkpointer와 `thread_id = user_id:session_id` 원칙을 따른다.

```python
checkpointer = SqliteSaver.from_conn_string(str(checkpoint_path))
return graph.compile(checkpointer=self.checkpointer)
```

실행 시 config:

```python
config = {
    'configurable': {
        'thread_id': f'{user_id}:{session_id}',
        'user_id': user_id,
        'session_id': session_id,
    }
}
final_state = self.graph.invoke(state, config=config)
```

중요한 점:

- `run_id`는 매 실행마다 새로 생성된다.
- `user_id:session_id`는 LangGraph `thread_id`로 사용된다.
- 같은 `user_id + session_id` 조합이면 `messages`, `last_focus`, `recent_entities`가 이어진다.
- 다른 사용자가 같은 `session_id`를 써도 thread가 분리된다.
- 후속 질문의 “이것/그것/지난번” 해석은 history DB보다 먼저 `last_focus`를 본다.
- 서버 재시작 후에도 SQLite checkpoint DB가 남아 있으면 같은 user/session의 short-term state를 복원할 수 있다.

---

## 4. AgentState

현재 `AgentState`는 graph 전체에서 공유되는 실행 상태다.

```python
class AgentState(TypedDict):
    run_id: str
    user_id: str
    session_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    current_question: str
    original_question: str
    send_request: AgentSendRequest

    request: NotRequired[AgentRequest]
    user_context: NotRequired[dict[str, Any]]
    resolved_reference: NotRequired[dict[str, Any]]
    last_focus: NotRequired[dict[str, Any] | None]
    recent_entities: NotRequired[list[dict[str, Any]]]
    current_turn_process_data: NotRequired[dict[str, Any] | None]
    previous_turn_process_data: NotRequired[dict[str, Any] | None]
    session_last_process_data: NotRequired[dict[str, Any] | None]
    process_data_reference_policy: NotRequired[dict[str, Any]]

    intent_gateway: NotRequired[dict[str, Any]]
    selected_path: NotRequired[SelectedPath]

    plan: NotRequired[AgentPlan]
    route: NotRequired[list[str]]
    prediction: NotRequired[PredictionResponse | None]
    manufacturing_context: NotRequired[ManufacturingContext]
    retrieved_documents: NotRequired[list[RagChunk]]
    recommended_actions: NotRequired[list[str]]
    safety_guidance: NotRequired[str | None]
    answer: NotRequired[str]
    report: NotRequired[str | None]

    response: NotRequired[AgentResponse]
    warnings: list[str]
    errors: list[dict[str, Any]]
    usage_records: list[LLMUsageRecord]
    trace: list[dict[str, Any]]
    replan_count: int
```

설계 원칙:

- 노드는 전체 state를 새로 만들지 않고 필요한 key만 갱신한다.
- `messages`는 `add_messages` reducer로 누적된다.
- `last_focus`는 follow-up 질문 해석의 1순위 신호다.
- `user_context`는 supporting evidence이며 현재 입력, 현재 RAG, 현재 safety gate보다 우선하지 않는다.
- 이전 턴의 `process_data`는 session state에 저장하되, 단순 개념 질문에는 자동 주입하지 않는다.

---

## 5. Request Context Node

`request_context` 노드는 다음을 처리한다.

```text
User validation
→ session upsert
→ user_context build
→ reference resolution
→ AgentRequest 생성
```

핵심 동작:

```python
user_context = self.context_service.build(
    user_id=req.user_id,
    session_id=session_id,
    request=base,
)

resolution = self.reference_resolution_service.resolve(
    user_id=req.user_id,
    session_id=session_id,
    question=req.message,
    context=user_context,
    last_focus=state.get('last_focus'),
    recent_entities=state.get('recent_entities') or [],
    messages=state.get('messages') or [],
)

user_context['current_turn'] = resolution.model_dump()
request = self._to_agent_request(
    req,
    session_id=session_id,
    user_context=user_context,
    question=resolution.resolved_question,
)
```

여기서 중요한 점은 원문 질문과 해석 결과가 분리된다는 것이다.

```json
{
  "original_question": "그렇다면 이것의 단점은?",
  "resolved_question": "토크의 단점은?",
  "resolved": true,
  "resolved_target": {
    "label": "토크",
    "type": "concept",
    "source": "last_focus",
    "confidence": 0.95
  }
}
```

---

## 6. Intent Gateway

`intent_gateway`는 모든 질문을 무겁게 처리하지 않기 위한 1차 분기점이다.

대표 분기:

| 질문 유형 | selected_path | Prediction | RAG | Safety |
|---|---|---:|---:|---:|
| `토크가 뭐야?` | `fast_concept_answer` | false | false | false |
| `이것의 단점은?` after `토크가 뭐야?` | `fast_concept_answer` | false | false | false |
| `LOTO 기준 문서로 알려줘` | `lightweight_rag_answer` | false | true | optional |
| `이 토크 값 위험해?` + process_data | `supervisor_planning` | true | optional | possible |
| `설비 멈춰줘` | `unsupported_or_clarification` | false | false | false |

현재 테스트로 보장하는 정책:

- `process_data`가 있어도 `토크가 뭐야?`는 prediction을 실행하지 않는다.
- `이 토크 값 위험해?`는 prediction path로 간다.
- `토크가 뭐야? → 이것의 단점은?`은 `last_focus=토크`로 해석한다.
- `토크와 공구 마모의 차이는? → 이것의 단점은?`은 clarification으로 보낸다.

---

## 7. Fast Path

Fast Path는 일반 개념 질문을 처리한다.

```text
request_context
→ intent_gateway
→ fast_concept_answer
→ focus_updater
→ audit_persistence
```

사용하는 것:

- `resolved_question`
- `resolved_target`
- 일반 제조/기계 지식
- 최소 LLM payload

사용하지 않는 것:

- Prediction Tool
- Manufacturing Analysis
- Safety Gate
- Report Agent

Fast Path system prompt의 핵심 정책:

```text
정의, 장단점, 한계, 원리 질문에는 일반 제조/기계 지식으로 답한다.
현재 설비 상태, 고장 확률, 안전 상태는 공정 데이터와 검증 근거 없이는 단정하지 않는다.
```

---

## 8. Heavy Path

복합 제조 업무 질문은 `supervisor_planning`부터 heavy path로 들어간다.

```text
supervisor_planning
→ manufacturing_analysis
→ evidence_retrieval
→ safety
→ response_synthesis
→ documentation
→ response_packager
```

### 8.1 Supervisor Planning

`supervisor_planning`은 `SupervisorService.plan()`을 호출해서 `AgentPlan`을 만든다.

```python
plan = self.heavy_graph.supervisor.plan(request, usage_callback=collect_usage)
state['plan'] = plan
state['route'] = list(plan.required_nodes)
```

`AgentPlan`의 주요 필드:

```python
prediction_required: bool
rag_required: bool
safety_required: bool
report_required: bool
asset_context_required: bool
process_condition_required: bool
failure_mode_required: bool
safety_gate_required: bool
action_plan_required: bool
required_nodes: list[str]
layers: list[AgentLayer]
rag_query: str
```

### 8.2 Manufacturing Analysis

`manufacturing_analysis`는 현재 다음을 수행한다.

```text
Prediction Tool
→ Domain Context build
→ Asset / Process / Failure / Risk context 생성
```

코드 수준:

```python
if plan.prediction_required and request.process_data:
    prediction = self.heavy_graph.prediction_service.predict(request.process_data)

manufacturing_context = self.heavy_graph.domain_service.build_context(
    request,
    prediction,
    doc_count=0,
)
```

결과는 state에 저장된다.

```python
state['prediction'] = prediction
state['manufacturing_context'] = manufacturing_context
```

### 8.3 Evidence Retrieval

`evidence_retrieval`은 현재 RAG 검색과 evidence grading을 한 노드 안에서 처리한다.

```python
retrieval_request = self.rag_query_planner.plan(
    request=request,
    planned_query=plan.rag_query,
    prediction=prediction,
    manufacturing_context=manufacturing_context,
    top_k=top_k,
    filters=plan.rag_filters,
)
contexts = self.retriever.retrieve(retrieval_request)
```

근거가 없거나 약하면 local replan을 수행한다.

```python
while plan.rag_required and (not contexts or weak_contexts) and replan_attempt < AGENT_MAX_REPLAN_ATTEMPTS:
    plan = self.heavy_graph.supervisor.replan(...)
    contexts = self.rag_service.search(...)
```

현재 trace node:

```text
retrieval.document_retriever
retrieval.local_replan
retrieval.evidence_grader
```

### 8.4 Safety

`safety`는 제조 context의 safety gates를 바탕으로 action과 safety guidance를 만든다.

```python
actions = self.recommendation_builder.collect_action_phrases(prediction, manufacturing_context)
safety_guidance = self.safety_gate_builder.safety_guidance(manufacturing_context)
```

현재 trace node:

```text
safety.safety_subgraph
```

다음 확장에서는 이 노드를 아래처럼 더 쪼개는 것이 적합하다.

```text
safety.gate_builder
safety.constraint_injector
safety.action_validator
safety.answer_validator
safety.report_validator
```

### 8.5 Response Synthesis

`response_synthesis`는 LLM 답변 생성과 안전 검증을 담당한다.

```python
llm_result = self.llm_service.generate_json(
    schema_name='manufacturing_domain_agent_response',
    schema=ANSWER_SCHEMA,
    system_prompt=self.heavy_graph._answer_system_prompt(plan.report_required),
    payload=self.structured_payload_builder.build(...),
    operation='answer_generation',
)
```

LLM 답변 후 safety validation:

```python
validation = self.heavy_graph.safety_validator.validate_answer(
    answer or '',
    manufacturing_context,
)
```

검증 실패 시 parent replan:

```python
plan = self.heavy_graph.supervisor.replan(
    request,
    plan,
    validation.errors,
    attempt=llm_attempt + 1,
)
```

현재 trace node:

```text
response.answer_composer
supervisor.parent_replan
```

### 8.6 Documentation

`documentation`은 보고서가 필요할 때만 실행된다.

```python
if not state.get('report'):
    state['report'] = self.heavy_graph.report_service.make_report(...)
```

현재 trace node:

```text
documentation.report_composer
```

### 8.7 Response Packager

`response_packager`는 최종 `AgentResponse`를 만든다.

포함 항목:

- `answer`
- `prediction`
- `manufacturing_context`
- `retrieved_documents`
- `safety_guidance`
- `report`
- `citations`
- `warnings`
- `trace`
- `plan`
- `llm_usage`
- `context_used`

---

## 9. Focus Updater

`focus_updater`는 다음 턴의 follow-up 처리를 위해 대화 초점을 저장한다.

예:

```json
{
  "last_focus": {
    "label": "토크",
    "type": "concept",
    "confidence": 0.95,
    "source": "current_turn",
    "session_id": "session_demo",
    "source_run_id": "..."
  }
}
```

정책:

- resolved target이 있으면 그것을 focus로 둔다.
- 일반 개념 질문에서 단일 entity가 있으면 그 entity를 focus로 둔다.
- 고장모드 분석 응답이면 주요 failure mode를 focus 후보로 둔다.
- 보고서 생성 응답이면 `직전 점검 보고서`를 focus로 둘 수 있다.
- 비교 질문처럼 entity가 여러 개면 단일 focus를 강제로 만들지 않는다.

이 구조 덕분에 아래 대화가 동작한다.

```text
User: 토크가 뭐야?
Agent: ...
User: 그렇다면 이것의 단점은?
Agent: 여기서 "이것"은 직전 질문의 "토크"를 의미한다고 보고 답변...
```

---

## 10. Audit / Persistence

`audit_persistence`는 다음을 수행한다.

```text
trace 정규화
usage summary 보정
history 저장
memory update
context metadata 보정
```

현재 저장 대상:

- `agent_runs`
- `user_memories`
- `user_sessions`

주의:

- Fast Path 응답도 history에 저장된다.
- Memory update는 raw private data를 그대로 저장하지 않고 rule-based summary 중심으로 저장한다.
- trace에는 raw user_id 대신 hash를 쓰는 방향이 운영 기준이다. 현재 observability 확장 시 이 정책을 유지해야 한다.

---

## 11. API 진입점

현재 주요 API:

```text
POST /agent/send
POST /agent/send/stream
POST /agent/intent
POST /agent/plan
GET  /users/{user_id}/context
GET  /users/{user_id}/history
```

`/agent/send`는 실제 LangGraph root를 실행한다.

```python
@app.post('/agent/send')
def send_agent(req: AgentSendRequest):
    return root_graph.run(req)
```

`/agent/send/stream`은 trace event를 newline-delimited JSON으로 흘려준다.

```json
{"type": "start"}
{"type": "trace", "step": {"step": "supervisor.route_planner", "detail": "..."}}
{"type": "final", "response": {...}}
```

---

## 12. 현재 테스트 보장 범위

현재 `test_intent_gateway.py`에서 보장하는 핵심 회귀:

```text
1. 명시적 개념 질문은 follow-up으로 오해하지 않음
2. 설비 제어 요청은 gateway에서 차단
3. process_data가 있어도 일반 개념 질문은 prediction 실행 안 함
4. 현재 값 위험도 질문은 prediction 필요
5. 같은 session의 last_focus로 "이것" 해석
6. 비교 질문 뒤의 "이것"은 ambiguous 처리
7. heavy path가 Subgraph trace node를 생성
```

heavy path trace 테스트는 아래 노드를 검증한다.

```python
assert 'supervisor.route_planner' in trace_steps
assert 'manufacturing.prediction_tool' in trace_steps
assert 'manufacturing.analysis_subgraph' in trace_steps
assert 'retrieval.evidence_grader' in trace_steps
assert 'safety.safety_subgraph' in trace_steps
assert 'response.answer_composer' in trace_steps
assert 'documentation.report_composer' in trace_steps
```

---

## 13. 현재 구현의 한계

현재 구조는 “Root Graph 계층화”까지 들어간 상태다. 하지만 완전한 최종형까지는 아래가 남아 있다.

### 13.1 RAG 내부가 아직 완전 분리되지 않음

현재:

```text
evidence_retrieval_node 안에서
query 생성, retrieve, weak evidence 판단, local replan 수행
```

목표:

```text
Evidence Retrieval Subgraph
 ├─ rag_query_planner
 ├─ dynamic_retrieval_dispatcher
 ├─ document_retriever
 ├─ evidence_filter
 ├─ evidence_grader
 ├─ citation_builder
 └─ retrieval_replan
```

### 13.2 LangGraph Send 기반 병렬 검색 미구현

현재 RAG query는 단일 query 중심이다.

목표:

```python
from langgraph.types import Send

def dispatch_retrieval(state: AgentState):
    return [
        Send("document_retriever", {"retrieval_request": req})
        for req in state["retrieval_requests"]
    ]
```

사용처:

- 정비 문서 query
- 안전 문서 query
- 설비 매뉴얼 query
- 보고서 템플릿 query

### 13.3 Safety validator가 아직 하나로 묶여 있음

현재:

```text
safety_subgraph에서 safety guidance 생성
response_synthesis에서 validate_answer 실행
```

목표:

```text
Safety Subgraph
 ├─ gate_builder
 ├─ constraint_injector
 ├─ action_safety_validator
 ├─ answer_safety_validator
 ├─ report_safety_validator
 └─ escalation_decision
```

### 13.4 Documentation validation 미분리

현재 보고서 생성은 있지만 completeness checker가 별도 노드가 아니다.

목표:

```text
Documentation Subgraph
 ├─ report_planner
 ├─ report_evidence_binder
 ├─ report_composer
 ├─ report_safety_validator
 └─ report_completeness_checker
```

### 13.5 Checkpointer는 MVP용 SQLite

현재:

```python
self.checkpointer = SqliteSaver.from_conn_string(...)
```

운영 목표:

```text
SQLite / Postgres / Redis checkpointer
```

현재 SQLite persistent checkpointer는 Local/MVP에는 충분하지만, 동시 요청이 많아지는 운영 환경에서는 lock, connection pooling, horizontal scaling을 고려해 Postgres/Redis checkpointer 전환이 필요하다.

---

## 14. 다음 AI와 의논할 확장 과제

아래 순서로 확장하는 것이 가장 안전하다.

### Phase A. RAG Subgraph 실제 분리

목표:

```text
evidence_retrieval_node를 여러 LangGraph node로 분해
```

추가 state:

```python
rag_query_plan: dict
retrieval_requests: list[dict]
retrieved_documents: list[RagChunk]
evidence_candidates: list[dict]
evidence_scores: list[dict]
citations: list[dict]
```

완료 기준:

- trace에 `retrieval.rag_query_planner`, `retrieval.document_retriever`, `retrieval.evidence_grader`, `retrieval.citation_builder`가 분리 표시된다.
- weak evidence일 때 RAG 내부에서 local replan한다.

### Phase B. Chroma PDF Ingestion 연결

목표:

```text
PDF upload
→ chunking
→ embedding
→ Chroma collection 저장
→ RAG retriever가 Chroma 사용
```

추가 API 후보:

```text
POST /documents/upload
POST /documents/ingest
GET  /documents
DELETE /documents/{document_id}
```

추가 UI:

```text
Streamlit Documents 탭
PDF 업로드
인덱싱 상태
문서별 chunk count
검색 테스트
```

### Phase C. Safety Subgraph 세분화

목표:

```text
조치 계획, 최종 답변, 보고서를 각각 검증
```

완료 기준:

- action list는 안전하지만 답변이 위험한 경우를 잡는다.
- 답변은 안전하지만 보고서가 LOTO를 누락한 경우를 잡는다.
- 반복 실패 시 human escalation으로 보낸다.

### Phase D. 운영용 Checkpointer 전환

목표:

```text
동시 요청과 배포 환경을 견디는 persistent checkpointer
```

고려안:

- Postgres checkpointer
- Redis checkpointer
- user hard delete 시 checkpoint namespace 삭제 정책

완료 기준:

- 같은 `user_id + session_id`로 재접속해도 `last_focus`가 유지된다.
- user hard delete 시 checkpoint도 함께 삭제된다.

### Phase E. Async Job / Event Stream

목표:

긴 작업을 `/agent/send` 동기 request에서 분리한다.

후보 API:

```text
POST /agent/jobs
GET  /agent/jobs/{job_id}
GET  /agent/jobs/{job_id}/events
POST /agent/jobs/{job_id}/cancel
```

사용처:

- 긴 보고서 생성
- 다중 문서 RAG 검색
- PDF ingestion
- multi-step evaluation

---

## 15. 다른 AI에게 전달할 핵심 요약

현재 제조 AI Agent는 `RootManufacturingGraph`를 중심으로 LangGraph `StateGraph`를 사용한다. `user_id:session_id`를 `thread_id`로 사용하고 SQLite persistent checkpointer를 붙여 같은 user/session의 `messages`, `last_focus`, `recent_entities`, `session_last_process_data`를 이어간다. `request_context`에서 user-scoped context와 follow-up reference resolution을 수행하고, `intent_gateway`에서 단순 개념 질문은 Fast Path로 보내며, 현재 공정 판단/점검/보고서 요청만 heavy path로 보낸다.

Fast Path는 토크/공구 마모/스핀들/회전수 같은 MVP 핵심 용어에 대해 glossary/template 기반 No-LLM 응답을 우선 사용한다. 따라서 단순 개념 질문은 Prediction/RAG/Safety/Report/LLM을 호출하지 않는다. 이전 턴의 `process_data`는 “방금 데이터”, “이 토크 값”, “그 조건”처럼 명시 참조가 있을 때만 prediction에 사용된다.

Heavy path는 현재 `supervisor_planning → manufacturing_analysis → evidence_retrieval → safety → response_synthesis → documentation → response_packager`로 분리되어 trace에 Subgraph 단위로 표시된다. Prediction, domain context, RAG 검색, safety guidance, LLM answer generation, report generation은 state에 중간 산출물로 저장된다.

다음 확장의 핵심은 `evidence_retrieval_node`와 `safety_node`를 더 작은 LangGraph node/subgraph로 분해하고, Chroma 기반 문서 검색과 LangGraph `Send` 기반 병렬 retrieval을 붙이는 것이다. 운영 품질을 위해서는 SQLite checkpointer를 Postgres/Redis checkpointer로 전환하고, user hard delete 시 checkpoint/history/memory/vector namespace를 함께 삭제하는 정책이 필요하다.
