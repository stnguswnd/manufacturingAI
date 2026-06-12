# 제조 AI 에이전트 현재 오케스트레이션 및 노드 구조

이 문서는 현재 코드베이스의 실제 실행 구조를 다른 AI 또는 설계자에게 전달해 전략을 다시 세울 수 있도록 정리한 것이다. 핵심은 현재 시스템이 LangGraph 기반의 명시적 그래프라기보다, `ManufacturingAgentGraph.run()` 안에서 순차 실행과 조건부 재계획을 직접 제어하는 imperative orchestration 구조라는 점이다.

## 1. 핵심 요약

현재 런타임의 중심 오케스트레이터는 다음 파일이다.

```text
ai_server/app/agent/graph.py
```

주요 클래스:

```python
ManufacturingAgentGraph
```

실제 실행 메서드:

```python
ManufacturingAgentGraph.run(req, progress_callback=None)
```

현재 구조는 이름상 graph이지만 실제로는 LangGraph의 node/edge/checkpoint 구조가 아니라, Python 코드 안에서 다음 순서로 직접 실행된다.

```text
입력 정규화
-> supervisor plan
-> 예측
-> 제조 도메인 컨텍스트 1차 구성
-> RAG 검색
-> RAG 품질이 약하면 supervisor re-plan
-> 제조 도메인 컨텍스트 2차 구성
-> safety/action 구성
-> LLM 답변 생성
-> safety validation 실패 시 supervisor re-plan 후 재시도
-> report 생성
-> usage/cost 집계
-> history 저장
-> FastAPI layer에서 memory 업데이트
```

즉, 현재 시스템의 "노드"는 LangGraph 노드가 아니라 trace step으로 표현되는 논리 노드에 가깝다.

## 2. 주요 진입점

### 2.1 `/agent/send`

동기 실행 API다.

파일:

```text
ai_server/app/main.py
```

흐름:

```text
AgentSendRequest
-> prepare_agent_request()
-> ManufacturingAgentGraph.run()
-> MemoryService.update_from_run()
-> AgentResponse 반환
```

특징:

- `user_id`가 필수다.
- `session_id`가 없으면 자동 생성된다.
- 실행 전 유저 컨텍스트가 구성된다.
- 실행 후 유저별 장기 memory가 업데이트된다.
- 응답에는 답변, 예측 결과, 제조 컨텍스트, RAG 문서, 안전 가이드, 리포트, trace, usage, context_used가 포함된다.

### 2.2 `/agent/send/stream`

Streamlit에서 진행 상태를 보여주기 위한 스트리밍 API다.

흐름은 `/agent/send`와 거의 같지만, `progress_callback`을 통해 trace step이 NDJSON event로 흘러간다.

스트림 이벤트 종류:

```text
start
trace
final
error
```

이 API 덕분에 Streamlit UI에서 "현재 어떤 단계가 실행 중인지" 표시할 수 있다.

### 2.3 `/agent/plan`

agent를 실제로 끝까지 실행하지 않고 supervisor plan을 미리 확인하는 API다.

주의점:

- `prepare_agent_request()`를 거치므로 user/session/context 구성은 수행된다.
- supervisor LLM refinement가 켜져 있으면 이 API도 LLM 호출 비용이 발생할 수 있다.

## 3. FastAPI 사전 준비 단계

`ManufacturingAgentGraph.run()`에 들어가기 전, FastAPI layer에서 다음 작업이 먼저 수행된다.

파일:

```text
ai_server/app/main.py
```

함수:

```python
prepare_agent_request(req: AgentSendRequest)
```

흐름:

```text
1. UserService.validate_user(user_id)
2. session_id가 없으면 session_{uuid} 생성
3. user_sessions upsert
4. AgentSendRequest -> AgentRequest 변환
5. ContextService.build(user_id, session_id, request)
6. AgentRequest.user_context에 주입
7. ManufacturingAgentGraph.run()으로 전달
```

이 단계는 현재 trace에 포함되지 않는다. 따라서 UI에서 보이는 agent trace는 실제 전체 파이프라인 중 graph 내부 단계만 보여준다.

## 4. 전체 실행 흐름 다이어그램

```text
Client / Streamlit
  |
  v
FastAPI /agent/send or /agent/send/stream
  |
  v
prepare_agent_request()
  |
  +--> UserService.validate_user()
  +--> user_sessions upsert
  +--> ContextService.build()
  |
  v
ManufacturingAgentGraph.run()
  |
  +--> Input Normalizer
  +--> SupervisorService.plan()
  |      |
  |      +--> deterministic plan
  |      +--> optional LLM refinement
  |
  +--> Prediction Tool
  +--> Evidence Tool
  +--> DomainKnowledgeService.build_context() 1차
  |      +--> Asset Context Agent
  |      +--> Process Condition Agent
  |      +--> Failure Mode Agent
  |      +--> Risk & Priority Agent
  |
  +--> RAG query 생성
  +--> RagService.search()
  |      |
  |      +--> weak/no context이면 Supervisor Re-plan loop
  |
  +--> DomainKnowledgeService.build_context() 2차
  |      +--> Safety Gate Agent
  |      +--> Action Planner Agent
  |
  +--> Safety Ops Agent
  +--> LLMService.generate_json()
  |      |
  |      +--> Explanation Agent
  |      +--> SafetyValidationService.validate_answer()
  |      +--> 실패 시 Supervisor Re-plan 후 재시도
  |
  +--> final Safety Validator
  +--> Report Agent
  +--> Evaluation / Audit Agent
  +--> usage summary
  +--> SQLiteStore.append(agent_run)
  |
  v
FastAPI layer
  |
  +--> MemoryService.update_from_run()
  |
  v
AgentResponse
```

## 5. 현재 노드 목록

아래 노드들은 실제 코드에서 trace로 표시되거나 supervisor plan route에 포함되는 논리 노드다.

| 순서 | 노드명 | 실제 소유 코드 | 주요 입력 | 주요 출력 | 조건 |
|---:|---|---|---|---|---|
| 1 | Input Normalizer | `ManufacturingAgentGraph.run()` | `AgentRequest` | normalized request trace | 항상 |
| 2 | Manufacturing Supervisor / Router | `SupervisorService.plan()` | request | `SupervisorPlan` | 항상 |
| 3 | Manufacturing Intent Classifier | `DiagnosticPlanner` | request | route node | plan metadata |
| 4 | Manufacturing Route Planner | `DiagnosticPlanToAgentPlanTranslator` | deterministic signals | route node | plan metadata |
| 5 | Prediction Tool | `PredictionService.predict()` | `process_data` | `PredictionOutput` | `prediction_required and process_data` |
| 6 | Evidence Tool | `ManufacturingAgentGraph.run()` | prediction | evidence trace | prediction 수행 시 |
| 7 | Asset Context Agent | `DomainKnowledgeService.build_context()` | request, prediction | asset context | domain context 생성 시 |
| 8 | Process Condition Agent | `DomainKnowledgeService.build_context()` | process data | condition summary | condition 존재 시 |
| 9 | Failure Mode Agent | `DomainKnowledgeService.build_context()` | prediction/domain rules | failure modes | failure modes 존재 시 |
| 10 | Risk & Priority Agent | `DomainKnowledgeService.build_context()` | prediction, failure modes | risk level | domain context 생성 시 |
| 11 | RAG Query Builder | `RagQueryPlanner.plan()` | question, context | retrieval request | plan metadata 또는 RAG 전 |
| 12 | Procedure Retrieval Agent | `RagService.search()` | query, filters | `RagChunk[]` | `rag_required` |
| 13 | Supervisor Re-plan | `SupervisorService.replan()` | plan, findings | revised plan | RAG 약함, safety fail, LLM parse fail |
| 14 | Safety Gate Agent | `DomainKnowledgeService.build_context()` | risk/failure/process | safety gates | safety gate 필요 시 |
| 15 | Action Planner Agent | graph helper | safety/context/docs | recommended actions | action plan 필요 시 |
| 16 | Safety Ops Agent | `SafetyGateBuilder.safety_guidance()` | gates, actions | safety guidance | 항상 또는 safety 관련 |
| 17 | Explanation Agent | `LLMService.generate_json()` | LLM payload | answer/report/actions | LLM 답변 생성 |
| 18 | LLM Usage Meter | `collect_usage()` | OpenAI usage | usage trace | LLM 호출마다 |
| 19 | Safety Validator | `SafetyValidationService.validate_answer()` | answer, context | pass/fail, issues | LLM 답변 후 |
| 20 | Report Agent | `ReportService.make_report()` 또는 LLM report | answer/context | report | report required |
| 21 | Evaluation / Audit Agent | graph final audit | response ingredients | audit trace | 항상 |
| 22 | History Store | `SQLiteStore.append()` | response/request | agent_runs row | 항상 |
| 23 | Memory Writer | `MemoryService.update_from_run()` | run response | user_memories upsert | graph 성공 후 |

중요한 점:

- `Memory Writer`는 현재 graph 내부 trace node가 아니다.
- `ContextService.build()`도 현재 trace node가 아니다.
- `RAG Query Builder`는 plan route에는 나오지만 실제 trace에서는 별도 step으로 항상 표시되지는 않는다.
- `LLM Usage Meter`는 실제 trace에는 나오지만 supervisor의 초기 route에는 포함되지 않는다.

## 6. Supervisor 구조

파일:

```text
ai_server/app/services/supervisor_service.py
```

주요 클래스:

```python
SupervisorService
```

주요 메서드:

```python
plan(req, usage_callback=None)
replan(req, previous_plan, findings, attempt, usage_callback=None)
```

### 6.1 초기 plan 생성

`SupervisorService.plan()`은 기본적으로 deterministic plan을 먼저 만든다.

```text
_deterministic_plan(req)
```

이후 설정이 켜져 있으면 LLM으로 refinement를 시도한다.

```text
AGENT_SUPERVISOR_LLM_REFINEMENT=true
```

흐름:

```text
request
-> deterministic signal 계산
-> route/layer 구성
-> optional LLM refinement
-> SupervisorPlan 반환
```

### 6.2 deterministic signal

주요 판단값:

```text
prediction_required
safety_required
report_required
knowledge_required
asset_context_required
process_condition_required
failure_mode_required
safety_gate_required
action_plan_required
rag_required
```

예시:

- `process_data`가 있으면 예측이 필요하다고 본다.
- 안전, 정비, 점검 관련 키워드가 있으면 safety path가 켜진다.
- 보고서 요청 또는 `generate_report=true`면 report path가 켜진다.
- knowledge, procedure, manual, 기준 등 질문이면 RAG path가 켜진다.
- `mode=hybrid`면 prediction, rag, safety, action을 폭넓게 켠다.

### 6.3 supervisor layer 정의

`DiagnosticPlanToAgentPlanTranslator`는 다음 레이어를 구성한다.

| Layer | 이름 | 대표 노드 |
|---:|---|---|
| 0 | Input Layer | Input Normalizer |
| 1 | Manufacturing Supervisor Layer | Manufacturing Intent Classifier, Manufacturing Route Planner |
| 2 | Asset Context Layer | Asset Context Agent |
| 3 | Process Condition Layer | Process Condition Agent |
| 4 | Failure Mode Layer | Failure Mode Agent |
| 5 | Risk & Priority Layer | Risk & Priority Agent |
| 6 | Procedure Retrieval Layer | RAG Query Builder, Procedure Retrieval Agent |
| 7 | Safety Gate Layer | Safety Gate Agent |
| 8 | Action Planning Layer | Action Planner Agent |
| 9 | Reasoning Layer | Explanation Agent |
| 10 | Documentation Layer | Report Agent |
| 11 | Audit & Persistence Layer | Evaluation / Audit Agent, History Store |

이 layer 정보는 UI/응답에서 route를 설명하는 데 쓰이지만, 실제 실행은 `ManufacturingAgentGraph.run()`의 if문과 loop로 수행된다.

## 7. 재계획 구조

현재 재계획은 크게 두 곳에서 발생한다.

### 7.1 RAG 재계획

위치:

```text
ManufacturingAgentGraph.run()
```

조건:

```text
RAG가 필요함
AND 검색 결과가 없음
또는 검색 결과가 사용자 질문 키워드와 약하게만 매칭됨
```

실행:

```text
RagService.search()
-> no/weak contexts 판정
-> SupervisorService.replan(findings)
-> 수정된 query/filter/top_k로 재검색
```

최대 재시도 횟수:

```text
AGENT_MAX_REPLAN_ATTEMPTS
```

현재 동작:

- 검색 결과가 아예 없으면 더 넓게 다시 검색한다.
- 검색 결과가 있지만 질문 키워드와 약하면 weak context로 보고 다시 검색한다.
- 재시도 후에도 약하면 context를 비우고 답변에서 해당 문서를 근거로 쓰지 않도록 한다.

### 7.2 LLM 답변 안전성 재계획

조건:

```text
LLM answer 생성
-> SafetyValidationService.validate_answer()
-> fail
```

실행:

```text
validation issue를 findings로 전달
-> SupervisorService.replan()
-> audit_feedback을 payload에 넣어 LLM 재호출
```

최종 재시도 후에도 safety validation을 통과하지 못하면 `UnsafeResponseError`를 발생시킨다.

### 7.3 LLM parse/schema 오류 재계획

조건:

```text
LLM JSON 응답 파싱 실패
또는 schema 불일치
```

실행:

```text
SupervisorService.replan()
-> output policy와 audit feedback을 강화
-> LLM 재호출
```

## 8. LLM 호출 구조

파일:

```text
ai_server/app/services/llm_service.py
```

LLM 호출은 주로 다음 두 영역에서 발생한다.

```text
1. Supervisor plan refinement
2. Final answer/report generation
```

현재 기본 실행은 LLM 사용을 전제로 한다. 이전 mock/false 흐름은 제거된 상태다.

LLM payload에는 다음 정보가 들어간다.

```text
question
inspection_notes
process_data
supervisor_plan
prediction
manufacturing_context
rag_contexts
recommended_actions
safety_guidance
user_context
audit_feedback
output_policy
```

시스템 프롬프트에는 다음 우선순위가 들어간다.

```text
현재 입력 공정 데이터
현재 검색 문서
현재 safety gate
> 유저 과거 context
```

유저 과거 context는 참고 정보일 뿐이며, 현재 센서값이나 현재 안전 상태를 대체하지 못한다.

## 9. User-scoped Context 구조

현재 user context는 graph 내부 노드가 아니라 FastAPI layer에서 먼저 만들어져 `AgentRequest.user_context`로 주입된다.

파일:

```text
ai_server/app/services/context_service.py
ai_server/app/services/user_service.py
ai_server/app/services/memory_service.py
```

### 9.1 ContextService.build()

입력:

```text
user_id
session_id
current_request
```

출력 형태:

```json
{
  "user_profile": {},
  "session_context": {},
  "long_term_memory": [],
  "recent_runs": [],
  "similar_runs": [],
  "context_policy": {},
  "estimated_context_tokens": 0
}
```

선택 정책:

```text
profile: 항상 포함
session summary: 같은 session 기준
recent runs: 같은 user의 최신 실행
similar runs: 같은 user의 키워드 유사 실행
long-term memory: importance 높은 순
```

중요한 격리 정책:

```text
user_id 기준으로만 history/memory/session을 가져온다.
다른 user의 실행 이력은 context에 들어가면 안 된다.
```

### 9.2 MemoryService.update_from_run()

위치:

```text
FastAPI /agent/send 실행 후
```

저장 memory 종류:

```text
equipment_preference
recurring_failure_mode
report_preference
safety_note
recent_summary
```

초기 구현은 LLM 기반 memory extraction이 아니라 rule-based extraction이다.

저장하면 안 되는 정보:

```text
API key
개인정보
민감한 사고 원문
근거 없는 추정
안전 상태 보증 표현
```

## 10. RAG 구조

파일:

```text
ai_server/app/services/rag_service.py
```

현재 RAG는 Chroma/FAISS 같은 벡터 DB를 사용하지 않는다.

현재 방식:

```text
chunks.jsonl 로드
-> 메모리 상에서 lexical/BM25 유사 scoring
-> RagChunk 반환
```

현재 RAG의 역할:

- 절차서/기준/도메인 문서 근거 검색
- LLM payload에 `rag_contexts`로 전달
- citation 생성에 사용
- RAG 결과가 질문과 약하면 supervisor re-plan 유발

현재 한계:

- semantic retrieval이 아니다.
- PDF 업로드/임베딩/벡터 저장은 아직 실제 런타임에 연결되지 않았다.
- Chroma PDF ingestion 전략 문서는 있으나 코드 구현은 아직 별도 과제로 남아 있다.

## 11. Prediction 구조

파일:

```text
ai_server/app/services/prediction_service.py
```

Prediction node는 `process_data`가 있을 때 실행된다.

입력은 Pydantic schema를 통과한 공정 데이터다.

주요 필드:

```text
type
air_temperature_k
process_temperature_k
rotational_speed_rpm
torque_nm
tool_wear_min
```

역할:

```text
process_data
-> 학습된 또는 준비된 모델로 failure/risk 예측
-> PredictionOutput 생성
-> failure mode, risk, confidence 등을 domain context로 전달
```

이 결과는 이후 다음 단계에 영향을 준다.

```text
Failure Mode Agent
Risk & Priority Agent
Safety Gate Agent
Action Planner Agent
LLM answer payload
```

## 12. Safety 구조

파일:

```text
ai_server/app/services/safety_validation_service.py
```

Safety는 두 층으로 동작한다.

### 12.1 Safety gate/context 생성

`DomainKnowledgeService.build_context()`에서 제조 도메인 규칙과 prediction 결과를 바탕으로 safety gate를 구성한다.

예:

```text
loto_if_physical_maintenance
ppe_required
stop_machine_if_high_risk
```

### 12.2 LLM 답변 검증

`SafetyValidationService.validate_answer()`는 LLM answer가 다음을 위반하는지 확인한다.

```text
안전 절차 누락
위험 작업을 과도하게 단정
현재 데이터보다 과거 context를 우선
근거 없는 안전 보증
필수 LOTO/PPE/정지 조건 누락
```

검증 실패 시 supervisor re-plan 후 LLM을 다시 호출한다.

최종 실패 시 응답을 내보내지 않고 오류 처리한다.

## 13. Usage / Cost 구조

LLM 호출마다 OpenAI response usage를 읽어 usage record를 만든다.

수집 항목:

```text
input_tokens
output_tokens
cached_tokens
total_tokens
estimated_cost_usd
estimated_cost_krw
usd_krw_exchange_rate
model
operation
latency
```

Graph 내부 helper:

```python
collect_usage(record)
```

이 helper는 다음을 수행한다.

```text
1. usage_records 배열에 추가
2. LLM Usage Meter trace emit
3. 최종 usage summary에 반영
```

Streamlit `Usage` 탭은 agent history에 저장된 usage summary를 집계해 보여준다.

## 14. Persistence 구조

파일:

```text
ai_server/app/storage/sqlite_store.py
```

주요 테이블:

```text
users
user_sessions
user_memories
agent_runs
```

`agent_runs`에는 다음이 저장된다.

```text
run_id
user_id
session_id
request_json
response_json
created_at
```

현재 `JsonLineStore`라는 이름은 backward compatibility alias로 남아 있으며 실제 구현은 SQLite 기반이다.

## 15. 현재 plan route와 실제 trace의 차이

현재 구조에서 주의해야 할 부분이다.

Supervisor plan route는 "실행해야 할 논리 노드 목록"이고, 실제 trace는 `ManufacturingAgentGraph.run()`에서 emit한 step 목록이다. 둘은 완전히 1:1로 일치하지 않는다.

예:

```text
plan route에는 RAG Query Builder가 있지만 실제 trace에는 Procedure Retrieval Agent만 보일 수 있다.
ContextService.build()는 실제 실행되지만 trace에 없다.
MemoryService.update_from_run()도 실제 실행되지만 trace에 없다.
LLM Usage Meter는 trace에 있지만 plan route에는 없다.
```

이 차이 때문에 UI에서 보는 진행 과정은 전체 시스템의 모든 내부 작업을 완전히 표현하지 않는다.

## 16. Optional LangGraph 파일

파일:

```text
ai_server/app/agent/langgraph_optional.py
```

이 파일에는 LangGraph demo skeleton이 존재한다.

구조:

```text
supervisor
-> prediction
-> rag_search
-> safety_ops
-> explanation
-> report
-> END
```

그러나 현재 production runtime에서는 이 구조를 사용하지 않는다.

현재 실제 runtime은 다음이다.

```text
ManufacturingAgentGraph.run()
```

따라서 전략을 다시 세울 때 이 파일을 현재 구조의 근거로 보면 안 된다. 참고용 또는 향후 LangGraph 전환 후보로만 보면 된다.

## 17. 현재 구조의 장점

현재 구조의 장점은 다음과 같다.

```text
1. 코드 흐름이 한 파일에서 직관적으로 추적된다.
2. supervisor plan과 re-plan이 이미 존재한다.
3. RAG 검색 실패 또는 약한 검색 결과에 대해 재계획 루프가 있다.
4. LLM 답변 safety validation 실패 시 재시도 구조가 있다.
5. user-scoped context가 LLM payload에 주입된다.
6. usage/cost가 응답과 UI에 연결되어 있다.
7. Streamlit에서 trace 기반 진행 상황을 확인할 수 있다.
```

## 18. 현재 구조의 한계

전략 재설계 시 가장 중요하게 봐야 할 한계다.

### 18.1 명시적 graph state가 없다

현재 state는 `run()` 내부 local variable에 흩어져 있다.

예:

```text
plan
prediction
manufacturing_context
contexts
actions
safety_guidance
answer
report
warnings
usage_records
```

따라서 노드별 입력/출력 계약이 강하게 고정되어 있지 않다.

### 18.2 node와 trace가 완전히 일치하지 않는다

실제 작업이 일어나지만 trace에 없는 단계가 있다.

```text
ContextService.build()
MemoryService.update_from_run()
session upsert
history query
context budget trimming
```

반대로 trace에는 있지만 plan route에는 없는 것도 있다.

```text
LLM Usage Meter
Supervisor Re-plan
```

### 18.3 Context engineering이 graph 외부에 있다

현재 user context는 FastAPI layer에서 graph 실행 전에 만들어진다.

장점:

```text
API 단에서 user validation과 context injection이 단순하다.
```

한계:

```text
context build가 trace와 node observability에 포함되지 않는다.
context 품질에 따라 supervisor plan이 바뀌는 구조가 약하다.
```

### 18.4 Memory update가 graph 외부에 있다

실행 후 memory update는 FastAPI handler에서 수행된다.

한계:

```text
memory update 실패가 agent trace에 드러나지 않는다.
memory writer가 명시적 노드가 아니다.
```

### 18.5 RAG가 아직 semantic vector retrieval이 아니다

현재는 lexical scoring이다.

향후 PDF 업로드, chunking, embedding, Chroma indexing이 들어오면 다음 구조가 필요하다.

```text
Document Ingestion
-> Chunker
-> Embedder
-> Vector Store
-> Retriever
-> Evidence Grader
```

### 18.6 supervisor plan은 있지만 완전한 multi-agent handoff는 아니다

현재 supervisor는 plan/replan을 만들지만, 각 agent node가 독립 실행자처럼 메시지를 주고받는 구조는 아니다.

실제 구조:

```text
central orchestrator가 service를 순차 호출
```

즉, "supervisor가 agent에게 일을 맡기고 결과를 보고 다시 보내는" 구조의 초기 형태는 있으나, 완전한 agent handoff graph는 아니다.

### 18.7 비동기 job queue가 없다

현재 요청은 API request lifecycle 안에서 처리된다.

한계:

```text
장시간 RAG ingestion
장시간 LLM multi-step 실행
재시도 많은 작업
동시 사용자 증가
```

이런 경우 Celery/RQ/Arq/Background task 등 job queue 구조가 필요할 수 있다.

## 19. 전략 재설계를 위한 권장 방향

다른 AI에게 전략을 맡길 때 아래 방향을 요구하면 좋다.

### 19.1 명시적 AgentState 도입

현재 local variable을 하나의 typed state로 정리한다.

예:

```python
class AgentState(TypedDict):
    run_id: str
    user_id: str
    session_id: str
    request: AgentRequest
    user_context: dict
    plan: SupervisorPlan
    prediction: PredictionOutput | None
    manufacturing_context: ManufacturingContext | None
    rag_contexts: list[RagChunk]
    safety_guidance: SafetyGuidance | None
    answer: str | None
    report: dict | None
    warnings: list[str]
    trace: list[AgentTraceStep]
    usage_records: list[LLMUsageRecord]
    replan_count: int
```

### 19.2 ContextBuilder와 MemoryWriter를 정식 노드로 승격

현재 graph 외부에 있는 단계를 graph 내부 trace에 포함한다.

권장 노드:

```text
User Validation Node
Session Manager Node
Context Builder Node
Memory Writer Node
```

### 19.3 RAG를 세분화

현재 RAG는 하나의 `Procedure Retrieval Agent`로 보인다.

Chroma 도입 시 다음으로 나누는 것이 좋다.

```text
RAG Query Planner
Document Retriever
Evidence Filter
Evidence Grader
Citation Builder
```

### 19.4 Safety를 독립 subgraph로 강화

현재 safety는 domain context와 final validator에 나뉘어 있다.

권장 구조:

```text
Safety Gate Builder
Safety Constraint Injector
Answer Safety Validator
Action Safety Validator
Report Safety Validator
```

### 19.5 route와 trace의 용어를 통일

현재 plan route와 실제 trace step이 다르다. UI/운영/테스트를 위해 아래가 필요하다.

```text
node_id
node_name
node_type
status
started_at
ended_at
latency_ms
input_summary
output_summary
error
```

### 19.6 LangGraph 전환 검토

현재 구조를 LangGraph로 옮기려면 조건부 edge가 필요하다.

예상 graph:

```text
prepare_context
-> supervisor_plan
-> prediction_or_skip
-> domain_context_pass_1
-> rag_or_skip
-> evidence_grade
-> replan_if_needed
-> domain_context_pass_2
-> safety_guidance
-> answer_generation
-> safety_validate
-> replan_or_finish
-> report_or_skip
-> persist_run
-> memory_update
```

조건부 edge:

```text
if plan.prediction_required
if plan.rag_required
if evidence weak
if answer unsafe
if report required
```

## 20. AI 전략 수립용 압축 입력

아래는 다른 AI에게 그대로 전달하기 좋은 압축 요약이다.

```text
현재 제조 AI 에이전트는 FastAPI + SQLite + Streamlit 기반이다.

실제 오케스트레이션은 LangGraph가 아니라 `ManufacturingAgentGraph.run()` 안의 imperative flow다.

FastAPI layer에서 먼저 `user_id` 검증, session upsert, user-scoped context build를 수행하고, 그 결과를 `AgentRequest.user_context`에 넣어 graph로 넘긴다.

Graph 내부는 다음 순서로 실행된다.

Input Normalizer
-> SupervisorService.plan()
-> optional PredictionService.predict()
-> DomainKnowledgeService.build_context() 1차
-> optional RagService.search()
-> weak/no RAG context이면 SupervisorService.replan() 후 RAG 재시도
-> DomainKnowledgeService.build_context() 2차
-> safety guidance/action plan 구성
-> LLMService.generate_json()으로 answer/report 생성
-> SafetyValidationService.validate_answer()
-> 실패 시 SupervisorService.replan() 후 LLM 재시도
-> ReportService fallback report
-> usage/cost summary
-> SQLiteStore.append()
-> FastAPI layer에서 MemoryService.update_from_run()

현재 trace node는 UI 진행상황 표시용이며, 실제 graph node와 완전히 일치하지 않는다.
ContextService와 MemoryService는 실제 실행되지만 현재 trace에는 없다.
Supervisor plan route에는 RAG Query Builder 같은 노드가 있지만 실제 trace에는 생략될 수 있다.

현재 RAG는 Chroma/FAISS가 아니라 chunks.jsonl 기반 lexical/BM25-ish search다.
Chroma PDF ingestion은 전략 문서만 있고 아직 실제 런타임에 구현되지 않았다.

현재 re-plan 구조는 두 가지다.
1. RAG 검색 결과가 없거나 약하면 supervisor가 plan을 수정하고 재검색한다.
2. LLM 답변이 safety validation을 통과하지 못하면 supervisor가 plan을 수정하고 LLM 답변을 재생성한다.

현재 한계는 명시적 AgentState 부재, route/trace 불일치, context/memory가 graph 외부에 있음, RAG가 semantic retrieval이 아님, 완전한 multi-agent handoff가 아님, async job queue가 없음이다.

전략 방향은 AgentState 도입, ContextBuilder/MemoryWriter를 정식 노드로 승격, RAG를 QueryPlanner/Retriever/EvidenceGrader로 분리, Safety subgraph 강화, route와 trace 용어 통일, 필요 시 LangGraph conditional graph로 전환하는 것이다.
```
