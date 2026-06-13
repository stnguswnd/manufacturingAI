# Bounded Agentic Workflow v3 Refactor Report

작성일: 2026-06-13

이 프로젝트는 자율형 멀티 에이전트가 아니다. 제조 안전 도메인에 맞춰
Prediction, Adaptive RAG Evidence, Safety Contract, Answer Text Review를 Artifact
contract와 LangGraph conditional edge로 제어하는 bounded agentic
workflow다.

## 1. 최종 구조 요약

```text
START
  -> request_context
  -> planning_router
  -> prediction_node -> prediction_quality_gate -> planning_router
  -> rag_evidence_subagent -> evidence_quality_gate -> planning_router
  -> safety_contract_subagent -> safety_contract_gate -> planning_router
  -> answer_compose
  -> answer_text_review
  -> next_action edge
  -> response_packager
  -> memory_writer
  -> audit_persistence
  -> END
```

Artifact 품질 검증, 답변 생성, 최종 문장 검증, rewrite, rerun, block,
fallback은 LangGraph node/edge로 보인다.

## 2. 제거한 Root Runtime Field

v3 runtime state에서 제거한 root-level field:

- `plan`
- `prediction` raw model
- `retrieved_documents`
- `citations`
- `safety_guidance`
- `safety_warnings`
- `structured_answer_payload`
- `manufacturing_context`
- `answer`
- `report`
- `usage_records`
- `trace`
- `replan_count`

외부 API `AgentResponse` schema는 유지한다. 내부 state만 artifact-only로
바꿨다.

## 3. ManufacturingAgentState

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

Checkpoint는 `state_schema_version = 3`만 읽고, 기본 파일명은
`langgraph_checkpoints_v3.sqlite3`다.

## 4. Graph Node / Edge 목록

```text
request_context -> planning_router

planning_router -> prediction_node -> prediction_quality_gate -> planning_router
planning_router -> rag_evidence_subagent -> evidence_quality_gate -> planning_router
planning_router -> safety_contract_subagent -> safety_contract_gate -> planning_router
planning_router -> fast_answer -> output_policy_gate -> response_packager
planning_router -> clarification_response -> response_packager
planning_router -> answer_compose -> answer_text_review

answer_text_review -> response_packager
answer_text_review -> invalidate_rewrite -> answer_rewrite -> answer_text_review
answer_text_review -> invalidate_rag_downstream -> rag_evidence_subagent -> evidence_quality_gate
answer_text_review -> invalidate_safety_downstream -> safety_contract_subagent -> safety_contract_gate
answer_text_review -> clarification_response -> response_packager
answer_text_review -> safe_block_response -> response_packager
answer_text_review -> safe_review_fallback -> response_packager

response_packager -> memory_writer -> audit_persistence -> END
```

## 5. Node별 Read / Write Contract

| Node | Reads | Writes |
| --- | --- | --- |
| `request_context` | `request`, previous `context`/`memory` | `request`, `context` |
| `planning_router` | all current artifacts | `planning` |
| `prediction_node` | `request`, `context`, `planning` | `prediction` |
| `prediction_quality_gate` | `planning`, `prediction`, `context` | `validation`, `runtime`, optional `planning` clarification |
| `rag_evidence_subagent` | `request`, `planning`, `prediction`, `context` | `evidence` |
| `evidence_quality_gate` | `planning`, `evidence`, `runtime` | `validation`, `runtime` |
| `safety_contract_subagent` | `request`, `planning`, `prediction`, `evidence` | `safety` |
| `safety_contract_gate` | `planning`, `evidence`, `safety`, `context`, `runtime` | `validation`, `runtime` |
| `answer_compose` | `request`, `context`, `planning`, `prediction`, `evidence`, `safety` | `draft` |
| `answer_text_review` | `draft`, `evidence`, `safety`, `planning`, `runtime` | `validation`, `runtime` |
| `answer_rewrite` | artifacts + failure summary in `runtime` | `draft` |
| `response_packager` | artifacts | `response` |
| `output_policy_gate` | `response` | `response`, `validation`, `runtime` |
| `memory_writer` | `request`, `context`, `planning`, `prediction`, `evidence`, `safety`, `draft`, `response`, `runtime` | `memory` |
| `audit_persistence` | public response projection | `audit` |

Trace, usage, retry budget, quality gate report는 `RuntimeArtifact`에만 둔다.

## 6. Answer Text Review next_action별 Edge

```text
pass
  -> response_packager

rewrite_only
  -> invalidate_rewrite
  -> answer_rewrite
  -> answer_text_review

rerun_rag
  -> invalidate_rag_downstream
  -> rag_evidence_subagent
  -> planning_router

rerun_safety
  -> invalidate_safety_downstream
  -> safety_contract_subagent
  -> planning_router

clarification_required
  -> clarification_response
  -> response_packager

block
  -> safe_block_response
  -> response_packager

max_retry_exceeded
  -> safe_review_fallback
  -> response_packager
```

Per-action retry budget 또는 total attempt budget을 넘으면 `max_retry_exceeded`로 강제 전환한다.

## 7. Stale Artifact Invalidation

`rerun_rag`:

- clear `evidence`
- clear `safety`
- clear `draft`
- clear `validation`
- clear `response`

`rerun_safety`:

- clear `safety`
- clear `draft`
- clear `validation`
- clear `response`

`rewrite_only`:

- clear `draft`
- clear `validation`
- clear `response`

`request`, `context`, `planning`, `prediction`, `runtime`은 유지한다.

## 8. Public API Schema 유지

`/agent/send`는 계속 `AgentResponse`를 반환한다. 내부 artifact는 public
schema에 직접 노출하지 않는다.

Mapping:

- `PredictionArtifact.result` -> `AgentResponse.prediction`
- `EvidenceArtifact.documents` -> `AgentResponse.retrieved_documents`
- `EvidenceArtifact.citations` -> `AgentResponse.citations`
- `SafetyArtifact.public_guidance` -> `AgentResponse.safety_guidance`
- `PlanningArtifact.agent_plan` -> `AgentResponse.plan`
- `RuntimeArtifact.trace` -> `AgentResponse.trace`
- `RuntimeArtifact.usage_records` -> `AgentResponse.llm_usage`

Public answer text에서 다음은 제거한다:

```text
run_id
token
cost
model
raw_score
chunk_id
safety_gate_id
gate id
calls=
replans=
trace
internal_reason
forbidden_agent_actions
```

## 9. AI4I / RAG Boundary

AI4I CSV/process data는 prediction input이다. RAG corpus로 쓰지 않는다.

AI4I prediction은 다음 6개 feature가 모두 유효할 때만 실행한다.

- `Type`
- `Air temperature`
- `Process temperature`
- `Rotational speed`
- `Torque`
- `Tool wear`

예측 의도가 있는데 feature가 부족하면 `PlanningArtifact.clarification_required=True`
상태로 `clarification_response`에 라우팅한다. 이 경우 prediction과 RAG는
실행하지 않는다.

## 10. 테스트

추가/수정된 테스트는 다음을 검증한다.

- v3 checkpoint state가 artifact-only key만 포함
- incomplete AI4I request가 prediction/RAG로 우회하지 않음
- complete AI4I request만 prediction 실행
- review next_action이 graph edge로 분기
- `rerun_rag`, `rerun_safety`, `rewrite_only` invalidation 정책
- block 응답 이후 rewrite loop 미진행
- max retry 초과 시 safe fallback
- public answer debug leak 제거

검증 명령:

```bash
cd ai_server
LANGSMITH_TRACING=false .venv/bin/python -m pytest -q
```

현재 결과:

```text
124 passed
```

## 11. 남은 리스크

- `ContextSubAgent`, `RagEvidenceSubAgent`, `SafetySubAgent` 내부 subgraph state는
  각 subagent local contract를 유지한다. Root runtime state는 artifact-only다.
- `AgentResponse` public schema는 외부 제품 API 계약으로 유지한다.
- 더 엄격하게 가려면 `MemorySubAgent`도 `AgentResponse` 대신 artifact input을
  직접 받도록 후속 변경할 수 있다.
