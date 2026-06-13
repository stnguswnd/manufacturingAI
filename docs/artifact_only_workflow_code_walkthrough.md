# Artifact-Only Bounded Workflow Code Walkthrough

작성일: 2026-06-13

이 문서는 제조 AI Agent 서버의 v3 runtime을 코드 기준으로 설명한다. 핵심 메시지는 다음이다.

```text
이 프로젝트는 자율형 멀티 에이전트가 아니다.
제조 안전 도메인에 맞춰 Prediction, Adaptive RAG Evidence, Safety Contract,
Answer Text Review를 Artifact contract와 LangGraph conditional edge로 제어하는
bounded agentic workflow다.
```

목표는 agent 이름을 늘리는 것이 아니라, runtime state와 graph edge에서 실제 책임 경계가 보이게 만드는 것이다.

- Root state는 artifact-only다.
- `planning_router`는 artifact 준비 상태를 보고 다음 node를 고른다.
- `rag_evidence_subagent`는 Adaptive RAG 실행 단위다.
- `safety_contract_subagent`는 RAG를 직접 실행하지 않고 `EvidenceArtifact`를 소비한다.
- Artifact quality gate는 producer node 직후 `ValidationReport.next_action`을 만든다.
- `answer_text_review`는 최종 public answer 문장만 검증한다.
- `rerun_rag`, `rerun_safety`, `rewrite_only`는 graph edge로 흐른다.
- stale artifact는 `None` tombstone으로 무효화한다.
- `/agent/send` public `AgentResponse` schema는 유지한다.

## 1. State 위치

Root runtime state는 `ai_server/app/agent/state.py`의 `ManufacturingAgentState`다.

```python
class ManufacturingAgentState(TypedDict, total=False):
    state_schema_version: int
    run_id: str
    user_id: str
    session_id: str
    thread_id: str

    request: RequestArtifact | None
    context: ContextArtifact | None
    planning: PlanningArtifact | None
    prediction: PredictionArtifact | None
    evidence: EvidenceArtifact | None
    safety: SafetyArtifact | None
    draft: AnswerDraft | None
    validation: ValidationReport | None
    response: ResponseArtifact | None
    memory: MemoryArtifact | None
    audit: AuditArtifact | None
    runtime: RuntimeArtifact | None
```

Root-level에 다시 생기면 안 되는 runtime field:

```text
plan
retrieved_documents
citations
safety_guidance
safety_warnings
structured_answer_payload
manufacturing_context
answer
report
usage_records
trace
replan_count
rag_evidence
evidence_grade
formatter_context
```

외부 `AgentResponse` projection에는 기존 public schema를 유지하기 위해 `retrieved_documents`, `citations`, `safety_guidance`, `plan`, `trace` 등이 있을 수 있다. 금지 대상은 내부 root runtime state다.

## 2. Artifact 모델

Artifact 모델은 `ai_server/app/agent/artifacts.py`에 있다.

### RequestArtifact

```python
class RequestArtifact(BaseModel):
    user_id: str
    session_id: str
    question: str
    original_message: str
    normalized_message: str | None = None
    process_data: dict[str, Any] | None = None
    inspection_notes: str | None = None
    top_k: int | None = None
    mode: str = 'auto'
    llm_model: str | None = None
```

요청 원문과 normalized question, AI4I process data, top_k, model override를 담는다.

### ContextArtifact

```python
class ContextArtifact(BaseModel):
    recent_turns: list[dict[str, Any]] = Field(default_factory=list)
    recent_turn_routes: list[dict[str, Any]] = Field(default_factory=list)
    rolling_summary: str = ''
    last_answer_memory: dict[str, Any] = Field(default_factory=dict)
    context_resolution: dict[str, Any] = Field(default_factory=dict)
    context_packs: dict[str, Any] = Field(default_factory=dict)
    compressed_context: dict[str, Any] = Field(default_factory=dict)
    user_context: dict[str, Any] = Field(default_factory=dict)
    turn_context: dict[str, Any] = Field(default_factory=dict)
    context_validation_warnings: list[str] = Field(default_factory=list)
    ai4i_feature_status: dict[str, Any] = Field(default_factory=dict)
    turn_process_data: dict[str, Any] | None = None
    previous_turn_process_data: dict[str, Any] | None = None
    session_last_process_data: dict[str, Any] | None = None
    process_data_reference_policy: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
```

`ContextSubAgent` 내부 local state는 root에 펼치지 않는다. RootGraph에는 이 artifact 하나로만 들어온다.

### PlanningArtifact

```python
class PlanningArtifact(BaseModel):
    selected_path: str
    answer_type: str
    intent: str | None = None
    needs_prediction: bool = False
    needs_rag: bool = False
    needs_safety: bool = False
    draft_ready: bool = False
    clarification_required: bool = False
    fast_answer_ready: bool = False
    missing_features: list[str] = Field(default_factory=list)
    completed_nodes: set[str] = Field(default_factory=set)
    next_node: str | None = None
    agent_plan: dict[str, Any] = Field(default_factory=dict)
    diagnostic_plan: dict[str, Any] = Field(default_factory=dict)
    route: list[str] = Field(default_factory=list)
    reasoning_summary: str | None = None
    warnings: list[str] = Field(default_factory=list)
```

`PlanningArtifact`가 router의 단일 truth source다. `completed_nodes`와 실제 artifact 존재 여부를 함께 보고 다음 node를 결정한다.

### PredictionArtifact

```python
class PredictionArtifact(BaseModel):
    called: bool = False
    result: dict[str, Any] | None = None
    skip_reason: str | None = None
    parsed_features: dict[str, Any] = Field(default_factory=dict)
    missing_features: list[str] = Field(default_factory=list)
    ambiguous_features: list[str] = Field(default_factory=list)
    invalid_features: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
```

AI4I prediction은 이 artifact 안에만 들어간다. AI4I CSV/process data는 RAG corpus가 아니다.

### EvidenceArtifact

```python
class EvidenceArtifact(BaseModel):
    profile: str | None = None
    queries: list[str] = Field(default_factory=list)
    documents: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    selected_source_ids: list[str] = Field(default_factory=list)
    required_safety_gates: list[str] = Field(default_factory=list)
    evidence_covers_required_gates: bool = True
    missing_gate_evidence: list[str] = Field(default_factory=list)
    generic_document_downgraded: bool = False
    retrieval_diagnostics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
```

Adaptive RAG output은 이 artifact 하나로 root state에 저장된다. `profile`은 `prediction_plus_rag`, `rag_only_safety`, `troubleshooting_rag`, `concept_explanation` 중 하나다.

### SafetyArtifact

```python
class SafetyArtifact(BaseModel):
    required_gates: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    forbidden_actions: list[str] = Field(default_factory=list)
    required_checks: list[str] = Field(default_factory=list)
    public_guidance: str | None = None
    structured_payload: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
```

Safety gate id는 내부 metadata로 유지한다. Public answer에는 자연어 안전 확인 항목으로만 반영한다.

### Draft / Validation / Response / Runtime

```python
class AnswerDraft(BaseModel):
    text: str
    route: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    llm_used: bool = False
    llm_error: str | None = None
    recommended_actions: list[str] = Field(default_factory=list)
    safety_guidance: str | None = None


class ValidationReport(BaseModel):
    passed: bool
    failures: list[ValidationFailure] = Field(default_factory=list)
    retryable: bool = False
    next_action: ReviewAction = 'pass'
    required_reexecution: list[str] = Field(default_factory=list)


class ResponseArtifact(BaseModel):
    answer: str
    route: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    public_citations: list[dict[str, Any]] = Field(default_factory=list)
    saved: bool = True
    response_type: str = 'answer'
    llm_used: bool = False
    llm_error: str | None = None
    report: str | None = None
    safe_fallback_used: bool = False


class RuntimeArtifact(BaseModel):
    review_iteration: int = 0
    replan_count: int = 0
    rewrite_attempts: int = 0
    rag_rerun_attempts: int = 0
    safety_rerun_attempts: int = 0
    evidence_signatures: list[str] = Field(default_factory=list)
    previous_missing_gate_evidence: list[str] = Field(default_factory=list)
    quality_gate_reports: list[dict[str, Any]] = Field(default_factory=list)
    usage_records: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
```

`RuntimeArtifact`만 trace, token/cost usage, retry budget, quality gate report, runtime error를 담는다.

## 3. Dependency Container

`RootManufacturingGraph`가 하위 객체 생성 로직을 계속 직접 갖지 않도록 `AgentRuntimeDeps`를 둔다.

```python
@dataclass(frozen=True)
class AgentRuntimeDeps:
    store: SQLiteStore
    prediction_service: PredictionService
    domain_service: DomainKnowledgeService
    safety_validator: SafetyValidationService
    llm_service: LLMService
    intent_classifier: IntentClassifierService
    intent_gateway: IntentGatewayService
    glossary_answer_service: GlossaryAnswerService
    formatter_registry: FormatterRegistry
    context_subagent: ContextSubAgent
    diagnostic_planner: DiagnosticPlanner
    planning_subagent: PlanningSubAgent
    citation_builder: CitationBuilder
    rag_evidence_subagent: RagEvidenceSubAgent
    recommendation_builder: RecommendationBuilder
    safety_subagent: SafetySubAgent
    memory_subagent: MemorySubAgent
    structured_payload_builder: StructuredAnswerPayloadBuilder
    answer_composer: AnswerComposer
    answer_rewriter: AnswerRewriter
    answer_review_loop: AnswerReviewLoop
```

조립은 `AgentRuntimeDeps.from_services()`에서 한다.

```python
rag_evidence_subagent = RagEvidenceSubAgent(RagEvidenceDeps(
    query_planner=RagQueryPlanner(),
    fanout_policy=RagFanoutPolicy(),
    rag_service=rag_service,
    evidence_filter=EvidenceFilter(),
    evidence_grader=EvidenceGrader(),
    citation_builder=citation_builder,
    domain_service=domain_service,
))

safety_subagent = SafetySubAgent(SafetyDeps(
    domain_service=domain_service,
    recommendation_builder=recommendation_builder,
    safety_gate_builder=SafetyGateBuilder(),
))

answer_review_loop = AnswerReviewLoop(
    citation_verifier=CitationVerifier(),
    safety_critic=SafetyCritic(safety_validator),
)
```

여기서 중요한 경계는 safety deps에 `RagService`, Chroma retriever, `CitationBuilder`가 없다는 점이다.

## 4. RootGraph Graph Node / Edge

`RootManufacturingGraph._build_graph()`가 전체 흐름을 정의한다.

```python
graph = StateGraph(ManufacturingAgentState)
graph.add_node('request_context', self._request_context_node)
graph.add_node('planning_router', self._planning_router_node)
graph.add_node('prediction_node', self._prediction_node)
graph.add_node('prediction_quality_gate', self._prediction_quality_gate_node)
graph.add_node('rag_evidence_subagent', self._rag_evidence_node)
graph.add_node('evidence_quality_gate', self._evidence_quality_gate_node)
graph.add_node('safety_contract_subagent', self._safety_contract_node)
graph.add_node('safety_contract_gate', self._safety_contract_gate_node)
graph.add_node('fast_answer', self._fast_answer_node)
graph.add_node('output_policy_gate', self._output_policy_gate_node)
graph.add_node('clarification_response', self._clarification_response_node)
graph.add_node('answer_compose', self._answer_compose_node)
graph.add_node('answer_text_review', self._answer_text_review_node)
graph.add_node('invalidate_rewrite', self._invalidate_rewrite_node)
graph.add_node('answer_rewrite', self._answer_rewrite_node)
graph.add_node('invalidate_rag_downstream', self._invalidate_rag_downstream_node)
graph.add_node('invalidate_safety_downstream', self._invalidate_safety_downstream_node)
graph.add_node('safe_block_response', self._safe_block_response_node)
graph.add_node('safe_review_fallback', self._safe_review_fallback_node)
graph.add_node('response_packager', self._response_packager_node)
graph.add_node('memory_writer', self._memory_writer_node)
graph.add_node('audit_persistence', self._audit_persistence_node)
```

앞쪽 루프는 artifact 준비 + artifact 품질 검증 루프다.

```python
graph.add_edge('request_context', 'planning_router')
graph.add_conditional_edges(
    'planning_router',
    self._route_after_planning,
    {
        'prediction_node': 'prediction_node',
        'rag_evidence_subagent': 'rag_evidence_subagent',
        'safety_contract_subagent': 'safety_contract_subagent',
        'answer_compose': 'answer_compose',
        'clarification_response': 'clarification_response',
        'fast_answer': 'fast_answer',
    },
)
graph.add_edge('prediction_node', 'prediction_quality_gate')
graph.add_edge('prediction_quality_gate', 'planning_router')
graph.add_edge('rag_evidence_subagent', 'evidence_quality_gate')
graph.add_edge('safety_contract_subagent', 'safety_contract_gate')
graph.add_edge('fast_answer', 'output_policy_gate')
graph.add_edge('output_policy_gate', 'response_packager')
graph.add_conditional_edges(
    'evidence_quality_gate',
    self._route_after_quality_gate,
    {
        'pass': 'planning_router',
        'rerun_rag': 'invalidate_rag_downstream',
        'clarification_required': 'clarification_response',
        'max_retry_exceeded': 'safe_review_fallback',
    },
)
graph.add_conditional_edges(
    'safety_contract_gate',
    self._route_after_quality_gate,
    {
        'pass': 'planning_router',
        'rerun_rag': 'invalidate_rag_downstream',
        'rerun_safety': 'invalidate_safety_downstream',
        'clarification_required': 'clarification_response',
        'max_retry_exceeded': 'safe_review_fallback',
    },
)
```

뒤쪽 루프는 최종 답변 문장 검증 루프다.

```python
graph.add_edge('answer_compose', 'answer_text_review')
graph.add_conditional_edges(
    'answer_text_review',
    self._route_after_text_review,
    {
        'pass': 'response_packager',
        'rewrite_only': 'invalidate_rewrite',
        'rerun_rag': 'invalidate_rag_downstream',
        'rerun_safety': 'invalidate_safety_downstream',
        'clarification_required': 'clarification_response',
        'block': 'safe_block_response',
        'max_retry_exceeded': 'safe_review_fallback',
    },
)
graph.add_edge('invalidate_rewrite', 'answer_rewrite')
graph.add_edge('answer_rewrite', 'answer_text_review')
graph.add_edge('invalidate_rag_downstream', 'rag_evidence_subagent')
graph.add_edge('invalidate_safety_downstream', 'safety_contract_subagent')
```

즉 RAG/Safety 재실행은 root 내부 helper 호출이 아니라 graph edge로 보인다.

## 5. 새 Turn 시작 시 Checkpoint Stale Artifact 제거

LangGraph checkpoint는 같은 `user_id/session_id`의 이전 state를 들고 있을 수 있다. 새 요청에서 이전 `planning`, `evidence`, `safety`, `response`가 섞이면 안 된다.

그래서 root는 turn-scoped artifact key를 따로 정의한다.

```python
TURN_SCOPED_ARTIFACT_KEYS = {
    'planning',
    'prediction',
    'evidence',
    'safety',
    'draft',
    'validation',
    'response',
    'audit',
}
```

`request_context` 시작 시 이전 context/memory는 읽되, turn-scoped artifact는 `None`으로 초기화한다.

```python
def _request_context_node(self, state):
    request = self._request_artifact(state)
    previous_context = self._context_artifact(state)
    previous_memory = self._memory_artifact(state)
    self._clear_turn_scoped_artifacts(state)
    ...
```

```python
def _clear_turn_scoped_artifacts(state):
    for key in TURN_SCOPED_ARTIFACT_KEYS:
        state[key] = None
```

이 정책 때문에 같은 session에서 새 질문이 들어와도 이전 turn의 `meta_feedback`, stale evidence, stale safety contract가 다음 turn router에 영향을 주지 않는다.

## 6. Planning Router

`planning_router`는 현재 state의 artifact를 읽고 다음 node를 하나 정한다.

```python
def _planning_router_node(self, state):
    request = self._request_artifact(state)
    context = self._context_artifact(state)
    previous = self._planning_artifact_or_none(state)
    completed = set(previous.completed_nodes if previous else set())
    if state.get('prediction'):
        completed.add('prediction_node')
    if state.get('evidence'):
        completed.add('rag_evidence_subagent')
    if state.get('safety'):
        completed.add('safety_contract_subagent')

    if previous and previous.selected_path:
        planning = previous
    else:
        planning = self._initial_planning(request, context, state)

    if planning.needs_prediction and not self._agent_request(state).process_data:
        planning = planning.model_copy(update={
            'clarification_required': True,
            'needs_prediction': False,
            'needs_rag': False,
            'needs_safety': False,
            'missing_features': list(context.ai4i_feature_status.get('missing_features') or planning.missing_features),
            'reasoning_summary': context.ai4i_feature_status.get('prediction_skip_reason') or 'AI4I prediction requires all six features.',
        })

    next_node = self._next_planning_node(planning, completed, state)
    planning = planning.model_copy(update={
        'completed_nodes': completed,
        'next_node': next_node,
        'draft_ready': next_node == 'answer_compose',
    })
    state['planning'] = planning.model_dump(mode='json')
    return self._return_state(state)
```

결정 정책은 다음과 같이 고정되어 있다.

```python
def _next_planning_node(planning, completed, state):
    if planning.clarification_required:
        return 'clarification_response'
    if planning.fast_answer_ready and not (planning.needs_prediction or planning.needs_rag or planning.needs_safety):
        return 'fast_answer'
    if planning.needs_prediction and 'prediction_node' not in completed and not state.get('prediction'):
        return 'prediction_node'
    if planning.needs_rag and 'rag_evidence_subagent' not in completed and not state.get('evidence'):
        return 'rag_evidence_subagent'
    if planning.needs_safety and 'safety_contract_subagent' not in completed and not state.get('safety'):
        return 'safety_contract_subagent'
    return 'answer_compose'
```

중요한 점:

- `selected_path != "supervisor_planning"`이라는 이유만으로 fast answer로 보내지 않는다.
- `fast_answer_ready=True`여도 prediction/RAG/safety가 필요하면 heavy node가 먼저 실행된다.
- prediction intent가 있고 AI4I feature가 부족하면 clarification으로 종료한다.

## 7. AI4I Feature Completeness Gate

AI4I prediction은 6개 feature가 모두 유효해야 실행한다.

```text
Type
Air temperature
Process temperature
Rotational speed
Torque
Tool wear
```

예측 의도가 있는데 feature가 부족하면 router가 다음 상태를 만든다.

```python
planning = planning.model_copy(update={
    'clarification_required': True,
    'needs_prediction': False,
    'needs_rag': False,
    'needs_safety': False,
    'missing_features': list(context.ai4i_feature_status.get('missing_features') or planning.missing_features),
})
```

결과:

```text
prediction_node 실행 안 함
rag_evidence_subagent 실행 안 함
safety_contract_subagent 실행 안 함
answer_compose 실행 안 함
clarification_response -> response_packager
```

이 정책은 feature 부족 요청을 RAG-only safety 답변으로 우회하지 않기 위한 hard gate다.

## 8. Node별 Write Contract

각 node는 자기 artifact만 쓴다. Trace/usage는 예외적으로 `RuntimeArtifact`에만 기록한다.

### prediction_node

```python
state['prediction'] = PredictionArtifact(
    called=bool(prediction),
    result=prediction.model_dump(mode='json') if prediction else None,
    skip_reason=skip_reason,
    parsed_features=dict(context.ai4i_feature_status.get('parsed_ai4i_features') or {}),
    missing_features=list(context.ai4i_feature_status.get('missing_features') or []),
    ambiguous_features=list(context.ai4i_feature_status.get('ambiguous_features') or []),
    invalid_features=list(context.ai4i_feature_status.get('invalid_features') or []),
    warnings=warnings,
).model_dump(mode='json')
```

### rag_evidence_subagent node

```python
output = self.deps.rag_evidence_subagent.invoke(RagEvidenceInput(
    request=request,
    plan=plan,
    prediction=prediction,
    manufacturing_context=manufacturing_context,
    top_k=min(max(request.top_k or 5, 1), AGENT_MAX_RAG_TOP_K),
))
state['evidence'] = output.evidence_artifact.model_dump(mode='json')
```

Root state에 `retrieved_documents`, `citations`, `evidence_grade`를 따로 쓰지 않는다.

### safety_contract_subagent node

```python
evidence = self._evidence_artifact(state)
docs = [RagChunk.model_validate(item) for item in evidence.documents]
output = self.deps.safety_subagent.invoke(SafetyInput(
    request=request,
    prediction=prediction,
    manufacturing_context=self._manufacturing_context(state),
    retrieved_documents=docs,
    structured_answer_payload={},
))
state['safety'] = output.safety_artifact.model_dump(mode='json')
```

Safety node는 이미 선택된 `EvidenceArtifact.documents`만 소비한다. Chroma/RAG 검색은 하지 않는다.

### answer_compose

```python
result = self.deps.answer_composer.compose_artifact(
    request_artifact=self._request_artifact(state),
    context_artifact=self._context_artifact(state),
    planning_artifact=self._planning_artifact(state),
    prediction_artifact=self._prediction_artifact_or_none(state),
    evidence_artifact=self._evidence_artifact_or_none(state),
    safety_artifact=self._safety_artifact_or_none(state),
    manufacturing_context=self._manufacturing_context(state),
    action_titles=self._recommended_action_titles(state),
    usage_callback=lambda record: self._record_usage(state, record),
    system_prompt=self._answer_system_prompt(),
)
state['draft'] = result.draft.model_dump(mode='json')
```

AnswerComposer는 draft만 만든다. Validation, next_action, rerun 판단을 하지 않는다.

### answer_text_review

```python
report = self.deps.answer_review_loop.review(
    draft=draft,
    manufacturing_context=self._manufacturing_context(state),
    evidence_artifact=self._evidence_artifact_or_none(state),
    safety_artifact=self._safety_artifact_or_none(state),
    needs_rag=planning.needs_rag,
)
...
state['runtime'] = runtime.model_dump(mode='json')
state['validation'] = report.model_dump(mode='json')
```

Text review node는 `ValidationReport`와 retry counter만 쓴다. RAG/Safety를 직접 호출하지 않는다. RAG artifact 품질과 SafetyArtifact 품질은 각각 `evidence_quality_gate`, `safety_contract_gate`가 먼저 판단한다.

## 9. Adaptive RAG Evidence SubAgent

`rag_evidence_subagent` 내부 state는 subgraph-local이다. Root에는 최종 `EvidenceArtifact`만 들어온다.

RAG subgraph input/output은 `ai_server/app/agent/rag_evidence/state.py`에 있다.

```python
class RagEvidenceInput(BaseModel):
    request: AgentRequest
    plan: AgentPlan
    prediction: PredictionResponse | None = None
    manufacturing_context: ManufacturingContext
    top_k: int = Field(default=5, ge=1, le=20)


class RagEvidenceOutput(BaseModel):
    plan: AgentPlan
    route: list[str] = Field(default_factory=list)
    retrieved_documents: list[RagChunk] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    evidence_grade: EvidenceGrade
    evidence_artifact: EvidenceArtifact = Field(default_factory=EvidenceArtifact)
    manufacturing_context: ManufacturingContext
    warnings: list[str] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)
    replan_count_delta: int = 0
```

내부 흐름:

```text
plan_queries
  -> retrieve
  -> filter
  -> grade
  -> cite
  -> build_payload
  -> build_trace
  -> EvidenceArtifact
```

Profile은 `RagFanoutPolicy.profile()`이 결정한다.

```python
if request.process_data or manufacturing_context.failure_modes or manufacturing_context.process_conditions:
    return 'prediction_plus_rag'
if plan.safety_required or plan.safety_gate_required or manufacturing_context.safety_gates:
    return 'rag_only_safety'
if 'troubleshooting_guide' in set(plan.document_scope or []):
    return 'troubleshooting_rag'
return 'concept_explanation'
```

Fan-out spec은 bounded maximum 4개다.

```python
specs = [{
    'name': 'primary',
    'query': base,
    'top_k': primary_top_k,
    'filters': filters,
    'intent': 'primary',
    'profile': profile,
}]

if needs_safety or gates:
    specs.append({
        'name': f'safety_{gate.gate_id}',
        'intent': 'safety_gate',
        'profile': profile,
        'safety_gate_id': gate.gate_id,
        'metadata_terms': ...,
    })

if needs_troubleshooting:
    specs.append({'name': 'troubleshooting', ...})

if profile == 'prediction_plus_rag' and modes:
    specs.append({'name': 'failure_mode', ...})
```

최종 trace와 health diagnostics는 `EvidenceArtifact`에 같이 들어간다.

```python
trace.update({
    'query_spec_names': [spec.get('name') for spec in state.get('query_specs') or [] if spec.get('name')],
    'raw_count': int(trace.get('raw_count') or len(state.get('raw_chunks') or [])),
    'filtered_count': int(trace.get('filtered_count') or len(state.get('filtered_chunks') or [])),
    'selected_count': len(selected),
    'citation_count': len(citations),
    'corpus_count_mismatch': bool(mismatch),
    'warnings': list(dict.fromkeys(warnings)),
})
state_with_final_diagnostics = {
    **state,
    'trace': trace,
    'warnings': trace['warnings'],
}
evidence_artifact = _evidence_artifact(
    state=state_with_final_diagnostics,
    grade=grade,
    selected=selected,
    citations=citations,
)
```

Artifact 생성에서 safety gate coverage를 계산한다.

```python
required_gates = [gate.gate_id for gate in context.safety_gates] if (plan.safety_required or plan.safety_gate_required) else []
covered_gates = {
    gate.gate_id
    for gate in context.safety_gates
    if any(_matches_gate_context(chunk, gate) for chunk in selected)
}
missing_gate_evidence = [gate_id for gate_id in required_gates if gate_id not in covered_gates]

return EvidenceArtifact(
    profile=diagnostics.get('retrieval_profile'),
    queries=[str(spec.get('query') or '') for spec in query_specs if spec.get('query')],
    documents=[chunk.model_dump() for chunk in selected],
    citations=citations,
    selected_source_ids=list(dict.fromkeys([item for item in citation_ids if item])),
    warnings=list(state.get('warnings') or []),
    retrieval_diagnostics=diagnostics,
    required_safety_gates=required_gates,
    evidence_covers_required_gates=not missing_gate_evidence,
    missing_gate_evidence=missing_gate_evidence,
    generic_document_downgraded=False,
)
```

Safety 질문에서 gate evidence가 부족하면 `evidence_quality_gate`가 먼저 `rerun_rag`를 선택한다. 최종 문장 검증 중 뒤늦게 근거 누락이 발견된 경우에만 `answer_text_review`가 `rerun_rag`를 다시 선택할 수 있다.

## 10. SafetyContractSubAgent

Safety contract node는 `EvidenceArtifact`를 소비한다.

```text
EvidenceArtifact.documents
  -> RagChunk list
  -> SafetyInput.retrieved_documents
  -> SafetySubAgent
  -> SafetyArtifact
```

정상 dependency:

```python
safety_subagent = SafetySubAgent(SafetyDeps(
    domain_service=domain_service,
    recommendation_builder=recommendation_builder,
    safety_gate_builder=SafetyGateBuilder(),
))
```

금지 dependency:

```text
RagService
Chroma
ChromaRetriever
metadata_search
search_with_diagnostics
CitationBuilder
```

이 경계는 `tests/test_artifact_only_graph.py`에서 static test로도 막는다.

```python
source = inspect.getsource(safety_nodes)
forbidden = ['RagService', 'Chroma', 'ChromaRetriever', 'metadata_search', 'search_with_diagnostics', 'CitationBuilder']
assert not any(token in source for token in forbidden)
```

## 11. Artifact Quality Gate And Answer Text Review Mapping

Artifact quality gate는 producer artifact 직후 품질을 검증한다. `AnswerReviewLoop`는 최종 문장 검증 단계에서 `CitationVerifier`와 `SafetyCritic` 결과를 merge한다.

```python
class AnswerReviewLoop:
    ACTION_PRIORITY = {
        'block': 5,
        'clarification_required': 4,
        'rerun_rag': 3,
        'rerun_safety': 2,
        'rewrite_only': 1,
        'pass': 0,
    }

    def review(...):
        citation_report = self.citation_verifier.verify(draft, evidence_artifact, needs_rag=needs_rag)
        safety_report = self.safety_critic.review(
            draft,
            manufacturing_context=manufacturing_context,
            safety_artifact=safety_artifact,
            evidence_artifact=evidence_artifact,
        )
        return self._merge([citation_report, safety_report])
```

`CitationVerifier`의 주요 mapping:

```python
if any(f.code == 'missing_required_gate_evidence' for f in failures):
    return ValidationReport(
        passed=False,
        retryable=True,
        next_action='rerun_rag',
        required_reexecution=['rag_evidence', 'safety_contract'],
    )
if any(f.source == 'debug_leak' for f in failures):
    return ValidationReport(passed=False, retryable=True, next_action='rewrite_only')
if not citations:
    return ValidationReport(
        passed=False,
        retryable=True,
        next_action='rerun_rag',
        required_reexecution=['rag_evidence', 'safety_contract'],
    )
```

`SafetyCritic`의 주요 mapping:

```python
if any(f.code == 'forbidden_action' and f.severity == 'critical' for f in failures):
    return ValidationReport(passed=False, retryable=False, next_action='block')

if any(f.code == 'public_safety_gate_id_leak' for f in failures):
    return ValidationReport(passed=False, retryable=True, next_action='rewrite_only')

missing_gate = any(f.code == 'required_safety_gate_missing' for f in failures)
missing_evidence = bool(evidence_artifact and not evidence_artifact.evidence_covers_required_gates)
if missing_gate and missing_evidence:
    return ValidationReport(
        passed=False,
        retryable=True,
        next_action='rerun_rag',
        required_reexecution=['rag_evidence', 'safety_contract'],
    )

if missing_gate and not (safety_artifact and safety_artifact.required_gates):
    return ValidationReport(
        passed=False,
        retryable=True,
        next_action='rerun_safety',
        required_reexecution=['safety_contract'],
    )
```

정리하면:

| Failure | next_action | 이유 |
| --- | --- | --- |
| debug/internal metadata leak | `rewrite_only` | evidence/safety 재실행 불필요 |
| citation 없음 | `rerun_rag` | 근거 artifact 부족 |
| required gate evidence 부족 | `rerun_rag` | RAG가 safety gate 문서를 다시 찾아야 함 |
| safety contract 자체 누락 | `rerun_safety` | Evidence는 보존하고 safety contract만 재생성 |
| forbidden action | `block` | rewrite loop를 계속 돌리지 않음 |
| retry limit 초과 | `max_retry_exceeded` | safe fallback으로 종료 |

## 12. Review Retry Limit

Quality gate와 `answer_text_review`는 retry 가능한 action만 per-action budget을 증가시킨다.

```python
if not report.passed and report.next_action in {'rewrite_only', 'rerun_rag', 'rerun_safety'}:
    if not self._retry_budget_allows(runtime, report.next_action):
        report = report.model_copy(update={'next_action': 'max_retry_exceeded', 'retryable': False})
    else:
        self._increment_retry_budget(runtime, report.next_action)
```

`max_retry_exceeded`가 되면 graph는 `safe_review_fallback`으로 간다.

## 13. Stale Artifact Invalidation

Compiled LangGraph에서는 반환 dict에서 key를 생략하는 것이 삭제를 의미하지 않는다. 그래서 `pop()`이 아니라 `None` tombstone을 쓴다.

### rewrite_only

```python
def _invalidate_rewrite_node(self, state):
    self._stash_validation_failures(state)
    for key in ['draft', 'validation', 'response']:
        state[key] = None
    return self._return_state(state)
```

보존:

```text
request
context
planning
prediction
evidence
safety
runtime
```

### rerun_rag

```python
def _invalidate_rag_downstream_node(self, state):
    self._stash_validation_failures(state)
    for key in ['evidence', 'safety', 'draft', 'validation', 'response']:
        state[key] = None
    self._remove_completed(state, {'rag_evidence_subagent', 'safety_contract_subagent'})
    return self._return_state(state)
```

RAG가 바뀌면 evidence에 의존하는 safety, draft, validation, response가 모두 stale이다.

### rerun_safety

```python
def _invalidate_safety_downstream_node(self, state):
    self._stash_validation_failures(state)
    for key in ['safety', 'draft', 'validation', 'response']:
        state[key] = None
    self._remove_completed(state, {'safety_contract_subagent'})
    return self._return_state(state)
```

Evidence는 보존하고 safety 이후 artifact만 무효화한다.

## 14. ResponsePackager

`response_packager`는 최종 `ResponseArtifact`를 만드는 유일한 packaging node다.

```python
def _response_packager_node(self, state):
    response = self._response_artifact_or_none(state)
    if not response:
        draft = self._draft_artifact(state)
        response = ResponseArtifact(
            answer=self._append_reference_details(draft.text, draft.citations, self._manufacturing_context(state)),
            route=['answer.compose', 'answer.review'],
            warnings=draft.warnings,
            public_citations=draft.citations,
            llm_used=draft.llm_used,
            llm_error=draft.llm_error,
        )
    public_citations = response.public_citations or self._citations(state)
    route = response.route or self._trace_route(state)
    state['response'] = response.model_copy(update={
        'answer': self._sanitize_public_answer(
            self._append_reference_details(response.answer, public_citations, self._manufacturing_context(state))
        ),
        'route': route,
        'warnings': self._public_warning_lines(response.warnings + self._artifact_warnings(state)),
        'public_citations': public_citations,
    }).model_dump(mode='json')
```

Root state에는 `AgentResponse`를 저장하지 않는다. 저장되는 것은 `ResponseArtifact`다.

## 15. Public AgentResponse Projection

외부 API 응답은 기존 `AgentResponse` schema를 유지한다. 변환은 `_agent_response_from_state()`에서 한다.

```python
return AgentResponse(
    run_id=state['run_id'],
    user_id=request.user_id,
    session_id=request.session_id,
    route=response.route,
    answer=self._sanitize_public_answer(response.answer),
    prediction=prediction,
    manufacturing_context=self._manufacturing_context(state),
    retrieved_documents=[RagChunk.model_validate(item) for item in self._evidence_artifact(state).documents],
    safety_guidance=self._safety_artifact(state).public_guidance,
    report=response.report,
    citations=response.public_citations,
    warnings=self._public_warning_lines(response.warnings),
    trace=to_agent_trace_steps(runtime.trace),
    saved=response.saved,
    plan=AgentPlan.model_validate(planning.agent_plan) if planning and planning.agent_plan else None,
    llm_used=response.llm_used or bool(runtime.usage_records),
    llm_provider=LLM_PROVIDER,
    llm_model=request.llm_model or self.deps.llm_service.model,
    llm_usage=self._usage_summary(runtime.usage_records),
    llm_error=response.llm_error,
    context_used=self._context_metadata(context, request.user_id),
    prediction_called=bool(prediction_artifact and prediction_artifact.called),
    ...
)
```

이 projection 때문에 public API schema는 유지되지만, internal runtime은 artifact-only다.

## 16. Public / Debug Separation

Public answer sanitizer는 내부 debug 문자열을 제거한다.

```python
blocked = [
    'resolved=false', 'resolved_target', 'question_kind', 'context_policy',
    'internal_reason', 'run_id', 'run id', 'llm usage', 'model=', 'llm_model',
    'tokens', 'token', 'cost', 'calls=', 'replans', 'trace', 'raw error',
    'raw_score', 'chunk_id', 'safety_gate_id', 'gate id',
    'forbidden_agent_actions',
]
```

Answer system prompt에도 같은 정책을 넣는다.

```python
'run id, model, token, cost, call count, trace, chunk id, safety gate id 같은 debug 정보를 답변 본문에 쓰지 마세요. '
'safety gate id는 내부 metadata이므로 자연어 안전 확인 항목으로만 반영하세요. '
```

내부 trace, usage, citation metadata는 structured field에 남길 수 있다. Public answer text에 섞이면 실패다.

## 17. Memory Writer

Memory writer도 artifact 기반 입력을 받는다.

```python
output = self.deps.memory_subagent.invoke(MemoryInput(
    run_id=state['run_id'],
    request=self._request_artifact(state),
    context=context,
    planning=self._planning_artifact_or_none(state),
    prediction=self._prediction_artifact_or_none(state),
    evidence=self._evidence_artifact_or_none(state),
    safety=self._safety_artifact_or_none(state),
    draft=AnswerDraft.model_validate(state['draft']) if state.get('draft') else None,
    response=self._response_artifact_or_none(state) or ResponseArtifact(answer=''),
    runtime=self._runtime_artifact(state),
    user_id=state['user_id'],
))
state['memory'] = MemoryArtifact(...).model_dump(mode='json')
```

`MemoryInput` 구조:

```python
class MemoryInput(BaseModel):
    run_id: str
    request: RequestArtifact
    context: ContextArtifact = Field(default_factory=ContextArtifact)
    planning: PlanningArtifact | None = None
    prediction: PredictionArtifact | None = None
    evidence: EvidenceArtifact | None = None
    safety: SafetyArtifact | None = None
    draft: AnswerDraft | None = None
    response: ResponseArtifact
    runtime: RuntimeArtifact = Field(default_factory=RuntimeArtifact)
    user_id: str
```

내부 기존 `MemoryService`와 `AnswerMemoryWriter` 재사용을 위해 transient `AgentResponse` projection을 만들 수 있지만, root runtime state에는 저장하지 않는다.

## 18. Checkpoint v3

초기 state:

```python
state: ManufacturingAgentState = {
    'state_schema_version': STATE_SCHEMA_VERSION,
    'run_id': str(uuid4()),
    'user_id': req.user_id,
    'session_id': session_id,
    'thread_id': build_thread_id(user_id=req.user_id, session_id=session_id),
    'request': RequestArtifact(...).model_dump(mode='json'),
    'runtime': RuntimeArtifact().model_dump(mode='json'),
}
```

Checkpoint 읽기는 v3만 인정한다.

```python
values = dict(snapshot.values or {})
if values.get('state_schema_version') != STATE_SCHEMA_VERSION:
    return {}
return values
```

기본 checkpoint 파일명은 v3다.

```python
self.checkpoint_path = checkpoint_path or LANGGRAPH_CHECKPOINT_DB.with_name(
    f'{LANGGRAPH_CHECKPOINT_DB.stem}_v3{LANGGRAPH_CHECKPOINT_DB.suffix}'
)
```

v2 checkpoint migration layer는 없다.

## 19. 테스트

핵심 테스트 파일:

- `ai_server/tests/test_artifact_only_graph.py`
- `ai_server/tests/test_answer_review_contracts.py`
- `ai_server/tests/test_rag_evidence_orchestration.py`
- `ai_server/tests/test_context_engineering.py`

Artifact-only checkpoint 검증:

```python
values = root._checkpoint_values(user_id=user['user_id'], session_id='artifact_only')
assert values['state_schema_version'] == 3
assert set(values) <= {'state_schema_version', 'run_id', 'user_id', 'session_id', 'thread_id', *ARTIFACT_KEYS}
assert not {'plan', 'retrieved_documents', 'citations', 'safety_guidance', 'answer', 'structured_answer_payload'} & set(values)
```

Router fast path regression 검증:

```python
planning = PlanningArtifact(
    selected_path='rag_only_safety',
    answer_type='safety',
    needs_rag=True,
    needs_safety=True,
    fast_answer_ready=False,
)
assert RootManufacturingGraph._next_planning_node(planning, set(), {}) == 'rag_evidence_subagent'

planning = planning.model_copy(update={'fast_answer_ready': True})
assert RootManufacturingGraph._next_planning_node(planning, set(), {}) == 'rag_evidence_subagent'
```

Compiled LangGraph tombstone 검증:

```python
graph = StateGraph(ManufacturingAgentState)
graph.add_node('invalidate_rag_downstream', root._invalidate_rag_downstream_node)
graph.set_entry_point('invalidate_rag_downstream')
graph.add_edge('invalidate_rag_downstream', END)
compiled = graph.compile()

out = compiled.invoke(base_state())

for key in ['evidence', 'safety', 'draft', 'validation', 'response']:
    assert out.get(key) is None
```

Safety boundary 검증:

```python
source = inspect.getsource(safety_nodes)
forbidden = ['RagService', 'Chroma', 'ChromaRetriever', 'metadata_search', 'search_with_diagnostics', 'CitationBuilder']
assert not any(token in source for token in forbidden)
```

RAG diagnostics 검증:

```python
output = agent(FakeRagService([[chunk('c1')]] * 4, chroma_count=701)).invoke(evidence_input())

assert output.trace['corpus_count_mismatch'] is True
assert output.evidence_artifact.generic_document_downgraded is False
assert output.evidence_artifact.retrieval_diagnostics['corpus_count_mismatch'] is True
assert any('Chroma collection count mismatch' in item for item in output.evidence_artifact.warnings)
```

Review mapping 검증:

```python
assert CitationVerifier().verify(...).next_action == 'rerun_rag'
assert CitationVerifier().verify(debug_leak_draft, ...).next_action == 'rewrite_only'
assert SafetyCritic(...).review(missing_contract, ...).next_action == 'rerun_safety'
assert SafetyCritic(...).review(forbidden_action, ...).next_action == 'block'
```

현재 전체 테스트 결과:

```text
LANGSMITH_TRACING=false .venv/bin/python -m pytest -q
124 passed
```

## 20. 파일별 역할 요약

| 파일 | 역할 |
| --- | --- |
| `ai_server/app/agent/state.py` | root `ManufacturingAgentState` 정의 |
| `ai_server/app/agent/artifacts.py` | Pydantic artifact contract |
| `ai_server/app/agent/root_graph.py` | LangGraph node/edge, router, invalidation, packaging |
| `ai_server/app/agent/runtime_deps.py` | service/subagent dependency assembly |
| `ai_server/app/agent/answer_composer.py` | AnswerDraft 생성과 rewrite |
| `ai_server/app/agent/answer_review.py` | Citation/Safety review merge |
| `ai_server/app/agent/validators/citation_verifier.py` | citation/debug leak deterministic critic |
| `ai_server/app/agent/validators/safety_critic.py` | safety failure to next_action mapping |
| `ai_server/app/agent/rag_evidence/nodes.py` | Adaptive RAG profile/fan-out/retrieve/filter/grade/cite/artifact |
| `ai_server/app/agent/safety_subagent/nodes.py` | SafetyArtifact 생성 |
| `ai_server/app/agent/memory_subagent/state.py` | artifact-based memory input |
| `ai_server/app/agent/checkpointing/reset.py` | v3 checkpoint reset helper |

## 21. 현재 남은 리스크

Root runtime은 artifact-only이고 graph-level edge로 재실행된다. 남은 리스크는 두 가지다.

1. `MemorySubAgent` 내부는 기존 persistence service 재사용 때문에 transient `AgentResponse` projection을 만든다. Root state에는 저장하지 않지만, memory service까지 완전히 artifact-native로 낮추는 작업은 후속으로 분리할 수 있다.
2. `DiagnosticPlanner`는 아직 `SupervisorService`의 LLM refinement/replan helper를 일부 사용한다. Primary runtime의 다음 node 결정은 `PlanningRouter`가 소유하지만, planning 내부 구현을 더 줄이려면 planner dependency도 별도 `PlanningPolicy`로 낮출 수 있다.

이 두 가지는 bounded workflow의 현재 동작을 깨지는 않지만, 다음 구조 정리 대상이다.
