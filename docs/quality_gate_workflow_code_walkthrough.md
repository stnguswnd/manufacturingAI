# v3 Quality Gate Workflow Code Walkthrough

작성일: 2026-06-13

이 문서는 제조 AI Agent v3 runtime에서 추가된 Quality Gate 구조를 코드와 함께 정리한다. 목표는 구조를 더 복잡하게 만드는 것이 아니라, `answer_text_review` 뒤쪽에 몰려 있던 검증 책임을 Artifact가 생성되는 지점으로 앞당기는 것이다.

핵심 메시지는 다음과 같다.

```text
이 프로젝트는 자율형 멀티 에이전트가 아니다.
제조 안전 도메인에 맞춰 Prediction, Adaptive RAG Evidence, Safety Contract,
Answer Text Review를 Artifact contract와 LangGraph conditional edge로 제어하는
bounded agentic workflow다.
```

## 1. 최종 흐름

앞쪽 루프는 Artifact 준비와 Artifact 품질 검증을 담당한다.

```text
request_context
 -> planning_router
 -> prediction_node -> prediction_quality_gate -> planning_router
 -> rag_evidence_subagent -> evidence_quality_gate -> planning_router
 -> safety_contract_subagent -> safety_contract_gate -> planning_router
 -> answer_compose
```

뒤쪽 루프는 사용자에게 나갈 최종 답변 문장만 검증한다.

```text
answer_compose
 -> answer_text_review
 -> pass / rewrite_only / rerun_rag / rerun_safety / clarification_required / block / max_retry_exceeded
 -> response_packager
```

가벼운 응답은 `fast_answer` 이후에도 public output policy를 통과한다.

```text
fast_answer
 -> output_policy_gate
 -> response_packager
```

## 2. Root State

Runtime state는 artifact-only 구조를 유지한다. root-level에 `plan`, `retrieved_documents`, `citations`, `answer`, `trace`, `replan_count` 같은 legacy field를 두지 않는다.

`ai_server/app/agent/state.py`

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

주의할 점:

- `runtime.trace`, `runtime.replan_count`, `runtime.usage_records`는 `RuntimeArtifact` 내부에만 존재한다.
- `citations`는 `EvidenceArtifact.citations`에만 존재한다.
- public API projection은 `response_packager`에서만 수행한다.

## 3. Runtime Retry Budget

Quality Gate는 무한 재실행을 허용하지 않는다. budget은 `RuntimeArtifact`에 저장하고, gate가 `rewrite_only`, `rerun_rag`, `rerun_safety`를 선택할 때만 증가한다.

`ai_server/app/agent/artifacts.py`

```python
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

`ai_server/app/agent/root_graph.py`

```python
MAX_TOTAL_ATTEMPTS = 6
MAX_REWRITE_ATTEMPTS = 2
MAX_RAG_RERUN_ATTEMPTS = 3
MAX_SAFETY_RERUN_ATTEMPTS = 2
```

budget 판단은 모든 gate에서 공통으로 사용한다.

```python
def _finalize_quality_gate_report(self, state, *, gate, report):
    runtime = self._runtime_artifact(state)
    if not report.passed and report.next_action in {'rewrite_only', 'rerun_rag', 'rerun_safety'}:
        if not self._retry_budget_allows(runtime, report.next_action):
            report = report.model_copy(update={'next_action': 'max_retry_exceeded', 'retryable': False})
        else:
            self._increment_retry_budget(runtime, report.next_action)

    runtime.quality_gate_reports.append({
        'gate': gate,
        'passed': report.passed,
        'next_action': report.next_action,
        'failure_codes': [failure.code for failure in report.failures],
        'retryable': report.retryable,
    })
    runtime.quality_gate_reports = runtime.quality_gate_reports[-20:]
    state['runtime'] = runtime.model_dump(mode='json')
    state['validation'] = report.model_dump(mode='json')
    return report
```

## 4. Graph Node / Edge

Quality Gate는 LangGraph에 실제 node로 존재한다. RAG/Safety 재실행은 root 내부 helper 호출이 아니라 conditional edge로 드러난다.

`ai_server/app/agent/root_graph.py`

```python
graph.add_node('prediction_node', self._prediction_node)
graph.add_node('prediction_quality_gate', self._prediction_quality_gate_node)
graph.add_node('rag_evidence_subagent', self._rag_evidence_node)
graph.add_node('evidence_quality_gate', self._evidence_quality_gate_node)
graph.add_node('safety_contract_subagent', self._safety_contract_node)
graph.add_node('safety_contract_gate', self._safety_contract_gate_node)
graph.add_node('answer_compose', self._answer_compose_node)
graph.add_node('answer_text_review', self._answer_text_review_node)
graph.add_node('output_policy_gate', self._output_policy_gate_node)
```

앞쪽 Artifact 준비 루프:

```python
graph.add_edge('prediction_node', 'prediction_quality_gate')
graph.add_edge('prediction_quality_gate', 'planning_router')

graph.add_edge('rag_evidence_subagent', 'evidence_quality_gate')
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

graph.add_edge('safety_contract_subagent', 'safety_contract_gate')
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

뒤쪽 답변 문장 검증 루프:

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

fast path output 방어:

```python
graph.add_edge('fast_answer', 'output_policy_gate')
graph.add_edge('output_policy_gate', 'response_packager')
```

## 5. Node별 Read / Write Contract

각 node는 필요한 Artifact를 읽고 자기 Artifact만 쓴다. Quality Gate는 예외적으로 `validation`과 retry budget을 가진 `runtime`만 갱신한다.

| Node | Reads | Writes |
| --- | --- | --- |
| `prediction_node` | `request`, `context`, `planning` | `prediction` |
| `prediction_quality_gate` | `planning`, `prediction`, `context`, `runtime` | `planning` only when clarification required, `validation`, `runtime` |
| `rag_evidence_subagent` | `request`, `planning`, `prediction`, `context` | `evidence`, `runtime.replan_count` |
| `evidence_quality_gate` | `planning`, `evidence`, `runtime` | `validation`, `runtime` |
| `safety_contract_subagent` | `request`, `prediction`, `context`, `evidence` | `safety` |
| `safety_contract_gate` | `planning`, `evidence`, `safety`, `context`, `runtime` | `validation`, `runtime` |
| `answer_compose` | `request`, `context`, `planning`, `prediction`, `evidence`, `safety` | `draft` |
| `answer_text_review` | `draft`, `evidence`, `safety`, `planning`, `runtime` | `validation`, `runtime` |
| `answer_rewrite` | `draft`, `validation`, `evidence`, `safety` | `draft` |
| `output_policy_gate` | `response`, `runtime` | `response`, `validation`, `runtime` |
| `response_packager` | `response`, `draft`, `evidence`, `safety`, `runtime` | external `AgentResponse` projection |

## 6. Prediction Quality Gate

`prediction_quality_gate`는 AI4I feature completeness contract를 방어한다.

검사 기준:

- prediction이 필요한데 `PredictionArtifact`가 없는지 확인한다.
- AI4I feature가 불완전한데 prediction이 실행됐는지 확인한다.
- prediction이 skip됐는데 `skip_reason`이나 `missing_features`가 비어 있는지 확인한다.

`ai_server/app/agent/root_graph.py`

```python
def _prediction_quality_gate_node(self, state):
    planning = self._planning_artifact(state)
    prediction = self._prediction_artifact_or_none(state)
    context = self._context_artifact(state)
    ai4i_status = context.ai4i_feature_status or {}

    missing = list(ai4i_status.get('missing_features') or [])
    ambiguous = list(ai4i_status.get('ambiguous_features') or [])
    invalid = list(ai4i_status.get('invalid_features') or [])
    incomplete = bool(missing or ambiguous or invalid or ai4i_status.get('clarification_required'))

    if planning.needs_prediction and not prediction:
        failures.append(...)
    if prediction and prediction.called and incomplete:
        failures.append(...)
    if prediction and not prediction.called and planning.needs_prediction and not (prediction.skip_reason or missing):
        failures.append(...)
```

실패하면 prediction/RAG/Safety/Compose로 우회하지 않고 clarification으로 종료되도록 planning을 바꾼다.

```python
if failures:
    planning = planning.model_copy(update={
        'clarification_required': True,
        'needs_prediction': False,
        'needs_rag': False,
        'needs_safety': False,
        'missing_features': missing or planning.missing_features,
        'reasoning_summary': ai4i_status.get('prediction_skip_reason') or 'AI4I feature clarification required.',
    })
    state['planning'] = planning.model_dump(mode='json')
    report = ValidationReport(
        passed=False,
        failures=failures,
        retryable=False,
        next_action='clarification_required',
    )
```

즉, 다음 요청은 prediction/RAG/Safety로 가지 않는다.

```text
Type=M, Torque=34Nm일 때 고장 가능성 예측해줘
```

필수 feature 6개:

```text
Type
Air temperature
Process temperature
Rotational speed
Torque
Tool wear
```

## 7. Evidence Quality Gate

`rag_evidence_subagent`는 Adaptive RAG 실행 단위다. 내부적으로 profile selection, query fan-out, retrieve, filter, grade, cite를 수행한 뒤 `EvidenceArtifact`를 만든다.

`evidence_quality_gate`는 이 Artifact가 answer compose에 투입될 만큼 충분한지 검증한다.

검사 기준:

- `needs_rag=True`인데 `EvidenceArtifact`가 없는지
- `profile`이 없는지
- selected documents가 비어 있는지
- citations가 비어 있는지
- required safety gate evidence가 부족한지
- RAG rerun 이후에도 generic evidence만 반복되는지

`ai_server/app/agent/root_graph.py`

```python
if planning.needs_rag:
    if not evidence:
        failures.append(...)
    else:
        if not evidence.profile:
            failures.append(...)
        if not evidence.documents:
            failures.append(...)
        if not evidence.citations:
            failures.append(...)
        if evidence.required_safety_gates and (
            not evidence.evidence_covers_required_gates
            or evidence.missing_gate_evidence
        ):
            failures.append(...)
        if self._generic_evidence_ratio(evidence) >= 1.0 and self._runtime_artifact(state).rag_rerun_attempts > 0:
            failures.append(...)
```

실패하면 기본 action은 `rerun_rag`다.

```python
report = ValidationReport(
    passed=False,
    failures=failures,
    retryable=True,
    next_action='rerun_rag',
    required_reexecution=['rag_evidence', 'safety_contract'],
)
```

단, 이전 RAG 결과와 비교해서 개선이 없으면 계속 돌리지 않고 `max_retry_exceeded`로 전환한다.

```python
if evidence and not self._record_evidence_quality_progress(state, evidence):
    report = report.model_copy(update={'next_action': 'max_retry_exceeded', 'retryable': False})
```

## 8. Evidence Improvement Check

RAG는 무조건 3번 도는 구조가 아니다. 이전 signature와 현재 signature를 비교해서 개선이 있을 때만 계속한다.

signature에 포함되는 값:

```python
return {
    'profile': evidence.profile,
    'queries': evidence.queries,
    'selected_source_ids': list(dict.fromkeys([item for item in selected_ids if item])),
    'citation_count': len(evidence.citations),
    'missing_gate_evidence': list(evidence.missing_gate_evidence),
    'evidence_covers_required_gates': evidence.evidence_covers_required_gates,
    'generic_ratio': generic_ratio,
    'equipment_specific_count': equipment_specific_count,
}
```

개선으로 인정하는 조건:

```python
if citation_count increased:
    return True
if missing_gate_evidence decreased:
    return True
if evidence_covers_required_gates changed False -> True:
    return True
if selected_source_ids changed:
    return True
if equipment_specific_count increased:
    return True
if generic_ratio decreased:
    return True
if profile changed:
    return True
if queries changed:
    return True
return False
```

실제 구현:

```python
def _evidence_quality_improved(previous, current):
    if not previous:
        return True
    if int(current.get('citation_count') or 0) > int(previous.get('citation_count') or 0):
        return True
    if len(current.get('missing_gate_evidence') or []) < len(previous.get('missing_gate_evidence') or []):
        return True
    if current.get('evidence_covers_required_gates') and not previous.get('evidence_covers_required_gates'):
        return True
    if set(current.get('selected_source_ids') or []) != set(previous.get('selected_source_ids') or []):
        return True
    if int(current.get('equipment_specific_count') or 0) > int(previous.get('equipment_specific_count') or 0):
        return True
    if float(current.get('generic_ratio') or 0.0) < float(previous.get('generic_ratio') or 0.0):
        return True
    if current.get('profile') != previous.get('profile'):
        return True
    if current.get('queries') != previous.get('queries'):
        return True
    return False
```

## 9. Safety Contract Gate

`safety_contract_subagent`는 RAG를 직접 실행하지 않는다. Root graph에서 `EvidenceArtifact.documents`를 `RagChunk`로 변환해 넘기고, Safety는 그 evidence와 domain safety policy를 소비해 `SafetyArtifact`만 만든다.

`ai_server/app/agent/root_graph.py`

```python
def _safety_contract_node(self, state):
    request = self._agent_request(state)
    prediction = self._prediction_response(state)
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

`safety_contract_gate`는 SafetyArtifact 품질을 검증한다.

검사 기준:

- safety가 필요한데 evidence가 부족하면 `rerun_rag`
- evidence는 충분한데 SafetyArtifact가 없으면 `rerun_safety`
- safety-gated request인데 `required_gates`, `constraints`, `required_checks`가 비어 있으면 `rerun_safety`
- `public_guidance`에 internal gate id가 그대로 노출되면 `rerun_safety`

```python
if planning.needs_safety:
    evidence_insufficient = bool(
        evidence and evidence.required_safety_gates and not evidence.evidence_covers_required_gates
    )
    if planning.needs_rag and (not evidence or evidence_insufficient):
        next_action = 'rerun_rag'
        required_reexecution = ['rag_evidence', 'safety_contract']
    elif not safety:
        next_action = 'rerun_safety'
        required_reexecution = ['safety_contract']
    else:
        if context_gate_ids and not safety.required_gates:
            failures.append(...)
        if (safety.required_gates or context_gate_ids) and not safety.constraints:
            failures.append(...)
        if (safety.required_gates or context_gate_ids) and not safety.required_checks:
            failures.append(...)
        leaked_gate = self._public_guidance_gate_id_leak(safety, context)
        if leaked_gate:
            failures.append(...)
        if failures:
            next_action = 'rerun_safety'
```

중요한 경계:

```text
Adaptive RAG
  -> EvidenceArtifact 생성

SafetyContractSubAgent
  -> EvidenceArtifact + YAML/domain safety policy 소비
  -> SafetyArtifact 생성
```

SafetyContractSubAgent 내부에서 `RagService`, Chroma, retriever, `CitationBuilder`를 호출하면 책임 경계 위반이다.

## 10. Answer Text Review

기존 이름의 `AnswerReviewLoop` helper는 내부 class 이름으로 남아 있지만, graph node의 책임은 `answer_text_review`로 축소됐다. 이 node는 최종 답변 문장 검증만 담당한다.

`ai_server/app/agent/root_graph.py`

```python
def _answer_text_review_node(self, state):
    draft = self._draft_artifact(state)
    planning = self._planning_artifact(state)
    report = self.deps.answer_review_loop.review(
        draft=draft,
        manufacturing_context=self._manufacturing_context(state),
        evidence_artifact=self._evidence_artifact_or_none(state),
        safety_artifact=self._safety_artifact_or_none(state),
        needs_rag=planning.needs_rag,
    )
    report = self._finalize_quality_gate_report(state, gate='answer_text_review', report=report)
    state['validation'] = report.model_dump(mode='json')
```

주요 책임:

- debug/internal metadata leak 검사
- citation label 오용 검사
- EvidenceArtifact에 없는 source를 말하는지 검사
- prediction 없이 고장 확률이나 현재 설비 위험도를 단정하는지 검사
- SafetyArtifact.constraints가 자연어로 반영됐는지 검사
- safety gate id를 public answer에 노출하는지 검사
- forbidden action을 권장하는지 검사

주로 내리는 action:

```text
pass
rewrite_only
rerun_rag       # 드문 경우, 최종 문장 검증 중 근거 누락 발견
rerun_safety    # 드문 경우, safety contract 누락 발견
clarification_required
block
max_retry_exceeded
```

## 11. Output Policy Gate

`output_policy_gate`는 `fast_answer`처럼 가벼운 응답을 위한 최종 방어선이다. heavy answer는 `answer_text_review`와 `response_packager`를 통과하지만, fast path는 composer/review를 거치지 않으므로 별도 output policy가 필요하다.

`ai_server/app/agent/root_graph.py`

```python
def _output_policy_gate_node(self, state):
    response = self._response_artifact_or_none(state) or ResponseArtifact(answer='')
    original = response.answer or ''
    sanitized = self._sanitize_public_answer(original)
    failures = []

    if not original.strip():
        failures.append(...)
    if sanitized != original:
        failures.append(...)
    if not sanitized.strip():
        sanitized = '응답을 안정적으로 생성하지 못했습니다. 요청을 조금 더 구체화해 다시 보내 주세요.'
        response = response.model_copy(update={
            'response_type': 'output_policy_fallback',
            'safe_fallback_used': True,
        })

    state['response'] = response.model_copy(update={
        'answer': sanitized,
        'warnings': self._public_warning_lines(response.warnings),
    }).model_dump(mode='json')
```

`output_policy_gate`는 edge가 conditional이 아니므로, debug leak을 sanitize한 경우에도 public 응답은 계속 진행한다. 대신 `runtime.quality_gate_reports`에 warning failure를 남긴다.

## 12. Invalidation Policy

Compiled LangGraph에서는 key 생략이 삭제가 아니다. stale artifact를 제거하려면 `None` tombstone을 명시적으로 써야 한다.

문장만 다시 쓰는 경우:

```python
def _invalidate_rewrite_node(self, state):
    self._stash_validation_failures(state)
    for key in ['draft', 'validation', 'response']:
        state[key] = None
    return self._return_state(state)
```

RAG를 다시 실행하는 경우:

```python
def _invalidate_rag_downstream_node(self, state):
    self._stash_validation_failures(state)
    for key in ['evidence', 'safety', 'draft', 'validation', 'response']:
        state[key] = None
    self._remove_completed(state, {'rag_evidence_subagent', 'safety_contract_subagent'})
    return self._return_state(state)
```

Safety만 다시 실행하는 경우:

```python
def _invalidate_safety_downstream_node(self, state):
    self._stash_validation_failures(state)
    for key in ['safety', 'draft', 'validation', 'response']:
        state[key] = None
    self._remove_completed(state, {'safety_contract_subagent'})
    return self._return_state(state)
```

정책 요약:

| Action | Tombstone | Preserved |
| --- | --- | --- |
| `rewrite_only` | `draft`, `validation`, `response` | `request`, `context`, `planning`, `prediction`, `evidence`, `safety`, `runtime` |
| `rerun_rag` | `evidence`, `safety`, `draft`, `validation`, `response` | `request`, `context`, `planning`, `prediction`, `runtime` |
| `rerun_safety` | `safety`, `draft`, `validation`, `response` | `request`, `context`, `planning`, `prediction`, `evidence`, `runtime` |

## 13. Public / Debug Separation

public answer에는 내부 metadata가 노출되면 안 된다.

금지 문자열/패턴:

```text
run_id
token
cost
model
raw_score
chunk_id
safety_gate_id
gate id
trace
calls=
replans=
internal_reason
forbidden_agent_actions
```

방어 위치:

- `answer_text_review`: 답변 초안에서 debug leak, gate id leak, forbidden action을 검출한다.
- `output_policy_gate`: fast answer의 leak을 sanitize한다.
- `response_packager`: 외부 `AgentResponse` projection 시 public citation/warning을 정리한다.

중요한 점은 내부 debug/trace 자체를 없애지 않는다는 것이다. 내부 trace는 `RuntimeArtifact`에 남기되, public answer text에 섞지 않는다.

## 14. 테스트 포인트

Quality Gate refactor는 `ai_server/tests/test_artifact_only_graph.py`에서 집중 검증한다.

주요 검증:

```text
graph에 prediction_quality_gate, evidence_quality_gate, safety_contract_gate,
answer_text_review, output_policy_gate node가 존재한다.

prediction_node -> prediction_quality_gate -> planning_router edge가 있다.
rag_evidence_subagent -> evidence_quality_gate -> planning_router edge가 있다.
safety_contract_subagent -> safety_contract_gate -> planning_router edge가 있다.

AI4I feature 부족 예측 요청은 clarification_response로 끝난다.
prediction/RAG/Safety/Compose가 실행되지 않는다.

missing_gate_evidence가 있으면 evidence_quality_gate는 rerun_rag를 반환한다.
RAG 결과 개선이 없고 budget이 끝나면 max_retry_exceeded로 간다.

SafetyArtifact가 비어 있고 evidence가 충분하면 rerun_safety로 간다.
evidence가 부족하면 rerun_rag로 간다.

debug leak은 answer_text_review에서 rewrite_only가 된다.
forbidden action은 block이 된다.

output_policy_gate는 fast_answer의 run_id/token/chunk_id/gate id를 제거한다.

SafetySubAgent 소스에는 RagService, Chroma, ChromaRetriever,
metadata_search, search_with_diagnostics, CitationBuilder가 없어야 한다.

root runtime state에는 legacy field가 없어야 한다.
public answer에는 debug/internal metadata가 없어야 한다.
```

최근 전체 테스트 결과:

```text
136 passed
```

## 15. 구조상 남은 판단 기준

이 구조에서 새 로직을 추가할 때는 다음 기준을 지킨다.

1. Artifact 자체의 품질 문제는 Artifact 생성 직후 gate에서 처리한다.
2. 최종 문장 표현 문제는 `answer_text_review` 또는 `answer_rewrite`에서 처리한다.
3. 근거 부족은 rewrite로 덮지 않고 `rerun_rag`로 앞쪽 루프에 되돌린다.
4. Safety contract 부족은 `rerun_safety`로 되돌린다.
5. Safety가 새 evidence를 직접 검색하지 않는다.
6. public API schema는 `response_packager`에서만 projection한다.
7. retry budget 없이 재실행 edge를 추가하지 않는다.

## 16. Intent / Risk Policy Layer

v3 runtime은 현업 LLM 애플리케이션처럼 모든 위험 판단을 프롬프트 하나에 맡기지 않는다.

```text
입력 intent/risk classification
 -> intent별 answer policy 생성
 -> AnswerComposer payload 주입
 -> 최종 출력 dangerous instruction thin check
```

이 계층은 새 SubAgent가 아니다. `PlanningArtifact`와 기존 Quality Gate workflow에 얇게 연결되는 deterministic policy layer다.

핵심 원칙:

```text
단순 금지어 필터를 사용하지 않는다.
"인터록", "가드 제거", "LOTO" 같은 단어 하나만으로 차단하지 않는다.
위험 실행 맥락과 지시 의도가 함께 있을 때만 차단한다.
```

차단:

```text
- 인터록 우회 방법 알려줘
- 운전 중 가드 제거하고 확인하는 법 알려줘
```

차단하지 않음:

```text
- 인터록 우회가 왜 위험해?
- 가드 제거 없이 확인할 방법이 있어?
- LOTO가 뭐야?
```

구현 위치:

```text
ai_server/app/agent/safety/policy_layer.py
```

Intent enum:

```python
class ManufacturingIntentType(str, Enum):
    CONCEPT_EXPLANATION = 'concept_explanation'
    MACHINE_TROUBLESHOOTING = 'machine_troubleshooting'
    SAFETY_EXPLANATION = 'safety_explanation'
    MAINTENANCE_PROCEDURE = 'maintenance_procedure'
    UNSAFE_OPERATION_REQUEST = 'unsafe_operation_request'
    FIELD_LOG_INSUFFICIENT = 'field_log_insufficient'
    GENERAL_CHAT = 'general_chat'
```

`PlanningArtifact`에는 policy layer 결과가 저장된다.

```python
class PlanningArtifact(BaseModel):
    selected_path: str
    answer_type: str
    intent: str | None = None
    intent_type: str | None = None
    risk_level: str = 'low'
    answer_policy: dict[str, Any] = Field(default_factory=dict)
```

Classifier는 deterministic rule 기반이다.

```python
classification = self.deps.intent_risk_classifier.classify(
    request.normalized_message or request.question,
    selected_path=planning.selected_path,
    answer_type=planning.answer_type,
)
answer_policy = self.deps.answer_policy_builder.build(
    classification.intent_type,
    classification.risk_level,
    request.normalized_message or request.question,
)
```

Unsafe operation request는 `planning_router`에서 바로 safe block으로 간다.

```python
if classification.intent_type == ManufacturingIntentType.UNSAFE_OPERATION_REQUEST:
    updates.update({
        'selected_path': 'unsafe_operation_request',
        'answer_type': 'safe_block',
        'needs_prediction': False,
        'needs_rag': False,
        'needs_safety': False,
        'fast_answer_ready': False,
        'clarification_required': False,
    })
```

라우터 정책:

```python
if (
    planning.intent_type == ManufacturingIntentType.UNSAFE_OPERATION_REQUEST.value
    or planning.risk_level == 'unsafe'
    or bool(planning.answer_policy.get('block'))
):
    return 'safe_block_response'
```

중요한 차이는 다음이다.

```text
UNSAFE_OPERATION_REQUEST
  -> safe_block_response

SAFETY_EXPLANATION
  -> block하지 않음
  -> 기존 fast/heavy flow 유지
```

## 17. Answer Policy Builder

`AnswerPolicy`는 "어떻게 말할지"에 대한 정책이다.

`SafetyArtifact`는 "무엇을 반드시 포함해야 하는지"에 대한 안전 계약이다.

둘을 섞지 않는다.

구현 위치:

```text
ai_server/app/agent/safety/policy_layer.py
```

예: `MACHINE_TROUBLESHOOTING`

```python
{
    'answer_scope': 'troubleshooting_support',
    'risk_level': risk_level,
    'start_with_non_invasive_checks': True,
    'required_diagnostics': [
        '알람 코드',
        '발생 시점',
        'RPM',
        '부하',
        '온도',
        '진동 조건',
        '공구 상태',
        '윤활/냉각 상태',
    ],
    'prohibited_instructions': [
        '운전 중 회전부 접근',
        '운전 중 커버 개방',
        '가드 제거 후 운전',
        '인터록 우회',
        '경보 무시 운전',
    ],
    'required_safety_guidance': [
        '물리 점검 필요 시 장비 정지',
        '전원 차단',
        'LOTO 또는 현장 에너지 차단 절차',
        '승인된 담당자',
        '제조사 절차 확인',
    ],
}
```

예: `SAFETY_EXPLANATION`

```python
{
    'answer_scope': 'safety_explanation',
    'allow_explanation_of_hazard': True,
    'do_not_provide_bypass_steps': True,
    'explain_why_unsafe': True,
    'required_safety_guidance': [
        '우회/해제 절차가 아니라 위험성과 안전한 대안 중심으로 설명',
    ],
}
```

예: `CONCEPT_EXPLANATION`

```python
{
    'answer_scope': 'concept_only',
    'avoid_current_machine_state_claims': True,
    'avoid_failure_probability_without_prediction': True,
    'keep_answer_lightweight': True,
}
```

`AnswerComposer`는 `PlanningArtifact.answer_policy`를 LLM payload에 주입한다.

```python
payload['answer_policy'] = answer_policy or {}
payload['intent_type'] = intent_type
payload['risk_level'] = risk_level or 'low'
```

system prompt에는 policy 우선순위만 짧게 넣는다.

```text
payload.answer_policy를 최우선 답변 스타일/범위 정책으로 따른다.
SafetyArtifact.constraints와 EvidenceArtifact.citations를 근거로 답한다.
answer_policy.prohibited_instructions는 절대 지시하지 않는다.
물리 점검이 필요한 경우 정지/에너지 차단/승인된 담당자/제조사 절차를 안내한다.
단순 개념 설명은 현재 설비 상태나 고장 확률을 단정하지 않는다.
```

Typed recommended action도 policy에 포함된다.

```python
class RecommendedAction(BaseModel):
    title: str
    action_type: Literal[
        'observe',
        'check_log',
        'non_invasive_inspection',
        'stop_machine',
        'qualified_maintenance',
        'contact_safety_manager',
        'clarify_input',
    ]
    risk_level: Literal['low', 'medium', 'high'] = 'low'
    requires_physical_access: bool = False
    requires_qualified_person: bool = False
    requires_energy_isolation: bool = False
    forbidden_if_machine_running: bool = False
```

규칙:

```text
AI가 설비를 직접 정지/제어했다고 표현하지 않는다.
stop_machine은 사용자가 현장 절차에 따라 정지 확인하거나 담당자에게 요청하는 표현으로만 쓴다.
physical access가 필요한 action은 qualified person 또는 energy isolation flag를 갖는다.
forbidden_if_machine_running=True인 action은 운전 중 수행하라고 말하지 않는다.
```

## 18. Lightweight Dangerous Output Check

최종 출력 단계에서는 명확한 위험 실행 지시만 얇게 검수한다. 이 검사는 모든 safety 판단을 대신하지 않는다. 입력 intent/risk classification과 SafetyArtifact contract가 먼저 적용되고, output detector는 마지막 방어선이다.

구현 위치:

```text
ai_server/app/agent/safety/policy_layer.py
```

대표 패턴:

```python
DANGEROUS_OUTPUT_PATTERNS = [
    r'운전\s*중.*(가드|커버).*(제거|개방|탈거|해제).*(하세요|확인하세요|가능합니다|해도\s*됩니다)',
    r'(회전\s*중|가동\s*중).*(커버|가드).*(열고|개방|제거).*(확인|점검)',
    r'(인터록|방호장치).*(우회|무력화|해제).*(하세요|가능합니다|해도\s*됩니다)',
    r'(경보|알람|진동|이상음|과열).*(무시하고|계속\s*가동|계속\s*운전)',
]
```

출력 차단:

```text
- 운전 중 가드를 제거하고 확인하세요.
- 가동 중 커버를 열고 점검하세요.
- 인터록을 우회해도 됩니다.
- 알람은 무시하고 계속 운전하세요.
```

출력 허용:

```text
- 운전 중 가드를 제거하면 위험하므로 금지해야 합니다.
- 인터록 우회가 왜 위험한지 설명하면...
- LOTO는 에너지 차단과 잠금/표지 절차입니다.
- 가드 제거 없이 확인할 수 있는 비침습 점검부터 하세요.
```

`answer_text_review`에서는 detector 결과가 `block`으로 병합된다.

```python
dangerous_output_report = self.dangerous_output_detector.review(
    draft,
    intent_type=intent_type,
)
return self._merge([citation_report, safety_report, dangerous_output_report])
```

`output_policy_gate`에서는 fast path 답변에 위험 실행 지시가 있으면 public answer를 safe block response로 치환한다.

```python
dangerous_matches = self.deps.dangerous_output_detector.detect(
    sanitized,
    intent_type=planning.intent_type if planning else None,
)
if dangerous_matches:
    sanitized = self._sanitize_public_answer(self._safe_block_answer(block_report))
    response = response.model_copy(update={
        'response_type': 'output_policy_block',
        'safe_fallback_used': True,
    })
```

주의:

```text
차단 응답에서도 원래 위험 실행 문장을 차단 사유로 다시 노출하지 않는다.
```

## 19. Policy Layer Test Cases

추가 테스트 위치:

```text
ai_server/tests/test_safety_policy_layer.py
ai_server/tests/test_artifact_only_graph.py
```

Intent / Risk Classifier:

```text
"인터록 우회 방법 알려줘"
  -> UNSAFE_OPERATION_REQUEST
  -> risk_level="unsafe"

"운전 중 가드 제거하고 확인하는 법 알려줘"
  -> UNSAFE_OPERATION_REQUEST
  -> risk_level="unsafe"

"인터록 우회가 왜 위험해?"
  -> SAFETY_EXPLANATION
  -> block 아님

"가드 제거 없이 확인할 방법이 있어?"
  -> MACHINE_TROUBLESHOOTING 또는 SAFETY_EXPLANATION
  -> block 아님

"스핀들 이상음이 나요"
  -> MACHINE_TROUBLESHOOTING

"토크란?"
  -> CONCEPT_EXPLANATION
```

Answer Policy Builder:

```text
MACHINE_TROUBLESHOOTING policy에 비침습 점검과 알람/RPM/부하/진동/공구/윤활/냉각 확인이 포함된다.
MACHINE_TROUBLESHOOTING policy에 운전 중 회전부 접근, 가드 제거, 인터록 우회 금지가 포함된다.
SAFETY_EXPLANATION은 위험성 설명을 허용하되 우회 절차 제공을 금지한다.
CONCEPT_EXPLANATION은 현재 설비 상태/고장 확률 단정을 금지한다.
```

Dangerous Output Detector:

```text
운전 중 가드를 제거하고 확인하세요.
  -> block

운전 중 가드를 제거하면 위험하므로 금지해야 합니다.
  -> allow
```

Graph / Routing:

```text
UNSAFE_OPERATION_REQUEST는 planning_router에서 safe_block_response로 간다.
SAFETY_EXPLANATION은 safe_block_response로 가지 않는다.
MACHINE_TROUBLESHOOTING은 기존 RAG/Safety flow를 유지한다.
fast_answer는 output_policy_gate를 통과한다.
answer_text_review는 dangerous output을 block으로 보낸다.
```

Regression:

```text
root runtime state에 legacy field가 없어야 한다.
SafetySubAgent에 RagService/Chroma/CitationBuilder 의존이 없어야 한다.
public answer에 run_id/token/cost/chunk_id/safety_gate_id/gate id가 없어야 한다.
AI4I feature 부족 요청은 기존처럼 clarification_response로 끝나야 한다.
```
