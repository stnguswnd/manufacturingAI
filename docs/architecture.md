# Manufacturing AI Agent Architecture

이 프로젝트는 자율형 멀티 에이전트가 아니다.
제조 안전 도메인에 맞춰 Prediction, Adaptive RAG Evidence, Safety Contract,
Answer Text Review를 Artifact contract와 LangGraph conditional edge로 제어하는
bounded agentic workflow다.

## Runtime Flow

```text
START
  -> request_context
  -> planning_router
  -> conditional:
       prediction_node -> prediction_quality_gate -> planning_router
       rag_evidence_subagent -> evidence_quality_gate -> planning_router
       safety_contract_subagent -> safety_contract_gate -> planning_router
       fast_answer -> output_policy_gate -> response_packager
       clarification_response -> response_packager
       answer_compose
  -> answer_text_review
  -> conditional:
       pass -> response_packager
       rewrite_only -> invalidate_rewrite -> answer_rewrite -> answer_text_review
       rerun_rag -> invalidate_rag_downstream -> rag_evidence_subagent -> planning_router
       rerun_safety -> invalidate_safety_downstream -> safety_contract_subagent -> planning_router
       clarification_required -> clarification_response -> response_packager
       block -> safe_block_response -> response_packager
       max_retry_exceeded -> safe_review_fallback -> response_packager
  -> memory_writer
  -> audit_persistence
  -> END
```

## Two Bounded Loops

앞쪽 루프는 Artifact 준비 + Artifact 품질 검증 루프다.

```text
request_context
  -> planning_router
  -> prediction_node -> prediction_quality_gate
  -> rag_evidence_subagent -> evidence_quality_gate
  -> safety_contract_subagent -> safety_contract_gate
  -> planning_router
```

`planning_router`가 artifact 준비 상태를 보고 다음 실행 node를 고른다.
`rag_evidence_subagent`는 Adaptive RAG 실행 단위이며 profile selection,
query planning, fan-out, retrieve, filter, grade, cite, `EvidenceArtifact`
생성을 담당한다.

뒤쪽 루프는 최종 답변 문장 검증 루프다.

```text
answer_compose
  -> answer_text_review
  -> answer_rewrite 또는 upstream rerun edge
```

RAG 품질과 SafetyArtifact 품질은 앞쪽 quality gate에서 먼저 검증한다.
`answer_text_review`는 public answer 문장이 근거, safety contract, output
policy를 지켰는지 확인한다. 문장 문제는 `rewrite_only`로 뒤쪽 루프 안에서
해결하고, 뒤늦게 발견된 근거/Safety 문제만 `rerun_rag` 또는 `rerun_safety`로
앞쪽 루프에 되돌린다.

`RootManufacturingGraph`는 `StateGraph(ManufacturingAgentState)`를 연결하는
top-level orchestration boundary다. 하위 서비스와 subagent 생성은
`AgentRuntimeDeps`에서 조립하고, RootGraph는 node/edge 연결과 artifact
전달에 집중한다.

## ManufacturingAgentState

Runtime state는 v3 artifact-only 구조다. Root-level runtime field는 artifact key로 제한한다.

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

삭제한 root-level runtime field:

- `plan`
- `prediction` as raw `PredictionResponse`
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

필요한 값은 각 artifact 내부에 둔다. 예를 들어 citation은
`EvidenceArtifact.citations`, 답변 초안은 `AnswerDraft.text`, public answer는
`ResponseArtifact.answer`, trace/usage/replan count는 `RuntimeArtifact`에 둔다.

## Artifact Contract

- `RequestArtifact`: user/session/question/process data/top_k/model.
- `ContextArtifact`: recent turns, rolling summary, context resolution,
  context packs, AI4I feature status, process data reference policy.
- `PlanningArtifact`: selected path, answer type, execution needs,
  completed nodes, next node, full `AgentPlan` payload.
- `PredictionArtifact`: prediction called/result/skip reason, parsed/missing
  AI4I features.
- `EvidenceArtifact`: Adaptive RAG retrieval profile, queries, selected documents,
  citations, required safety gate evidence coverage.
- `SafetyArtifact`: required gates, natural language constraints, forbidden
  actions, required checks, public guidance.
- `AnswerDraft`: generated or rewritten draft before validation.
- `ValidationReport`: quality gate/text review failures and bounded `next_action`.
- `ResponseArtifact`: sanitized public answer payload before mapping to
  `AgentResponse`.
- `RuntimeArtifact`: review iteration, per-action retry budget, evidence
  signatures, quality gate reports, usage records, trace, runtime errors.

## Node Read/Write Contract

| Node | Reads | Writes |
| --- | --- | --- |
| `request_context` | `request`, previous `context`/`memory` | `request`, `context` |
| `planning_router` | `request`, `context`, `planning`, `prediction`, `evidence`, `safety`, `validation`, `runtime` | `planning` |
| `prediction_node` | `request`, `context`, `planning` | `prediction` |
| `prediction_quality_gate` | `planning`, `prediction`, `context` | `validation`, `runtime`, optional `planning` clarification |
| `rag_evidence_subagent` | `request`, `planning`, `prediction`, `context` | `evidence` |
| `evidence_quality_gate` | `planning`, `evidence`, `runtime` | `validation`, `runtime` |
| `safety_contract_subagent` | `request`, `planning`, `prediction`, `evidence` | `safety` |
| `safety_contract_gate` | `planning`, `evidence`, `safety`, `context`, `runtime` | `validation`, `runtime` |
| `answer_compose` | `request`, `context`, `planning`, `prediction`, `evidence`, `safety` | `draft` |
| `answer_text_review` | `draft`, `evidence`, `safety`, `planning`, `runtime` | `validation`, `runtime` |
| `answer_rewrite` | artifacts plus previous failure summary in `runtime` | `draft` |
| `response_packager` | artifacts | `response` |
| `output_policy_gate` | `response` | `response`, `validation`, `runtime` |
| `memory_writer` | `request`, `context`, `response` | `memory` |
| `audit_persistence` | public `AgentResponse` projection | `audit` |

`RuntimeArtifact` is the cross-cutting operational artifact for trace, usage,
review iteration, and errors.

## Planning Router

`planning_router` is deterministic policy plus bounded planner output. It
does not run an autonomous loop. It reads completed artifacts and chooses one
`planning.next_node`.

Rules:

- AI4I prediction requires all six features:
  `Type`, `Air temperature`, `Process temperature`, `Rotational speed`,
  `Torque`, `Tool wear`.
- If prediction intent exists but features are missing/ambiguous/invalid,
  `planning.clarification_required=True` and the graph routes to
  `clarification_response`.
- Incomplete AI4I requests never fall through to RAG-only answers.
- If prediction is complete, `prediction_node` runs first.
- If document evidence is needed, `rag_evidence_subagent` runs.
- If safety constraints are needed, `safety_contract_subagent` runs.
- When required artifacts exist, the graph routes to `answer_compose`.

## Quality Gate And Text Review Edges

Artifact quality gates run immediately after their producer nodes.

```text
prediction_node -> prediction_quality_gate -> planning_router
rag_evidence_subagent -> evidence_quality_gate -> planning_router
safety_contract_subagent -> safety_contract_gate -> planning_router
fast_answer -> output_policy_gate -> response_packager
```

`answer_text_review` runs final public-text checks, then writes
`ValidationReport.next_action`.

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

Reexecution is visible in graph edges and is not handled by a root-internal
upstream rerun helper.

## Stale Artifact Invalidation

Before retrying upstream work, stale downstream artifacts are tombstoned with
`None`. This matters because compiled LangGraph channels do not treat an
omitted key as deletion.

`rerun_rag` clears:

- `evidence`
- `safety`
- `draft`
- `validation`
- `response`

`rerun_safety` clears:

- `safety`
- `draft`
- `validation`
- `response`

`rewrite_only` clears:

- `draft`
- `validation`
- `response`

`request`, `context`, `planning`, `prediction`, and `runtime` are preserved.
The planning artifact's `completed_nodes` set is updated so the router does not
reuse stale RAG or safety completion.

## AI4I And RAG Boundary

AI4I CSV/process data is prediction input only. It is not a RAG corpus.

```text
AI4I process data
  -> PredictionService
```

Document evidence comes from the manufacturing RAG corpus:

```text
OSHA / Haas / KOSHA documents
  -> RAG Evidence SubAgent
```

RAG-only safety questions must not include AI4I probability language or failure
mode probability language.

## Public API Mapping

The external `/agent/send` response model remains `AgentResponse`. The internal
state is artifact-only, and `response_packager` plus the final response
projection map artifacts to the public schema:

- `ResponseArtifact.answer` -> `AgentResponse.answer`
- `PredictionArtifact.result` -> `AgentResponse.prediction`
- `EvidenceArtifact.documents` -> `AgentResponse.retrieved_documents`
- `EvidenceArtifact.citations` -> `AgentResponse.citations`
- `SafetyArtifact.public_guidance` -> `AgentResponse.safety_guidance`
- `PlanningArtifact.agent_plan` -> `AgentResponse.plan`
- `RuntimeArtifact.trace` -> `AgentResponse.trace`
- `RuntimeArtifact.usage_records` -> `AgentResponse.llm_usage`

Public answer text is sanitized before packaging. It must not contain:

- `run_id`
- `token`
- `cost`
- `model`
- `raw_score`
- `chunk_id`
- `safety_gate_id`
- `gate id`
- `calls=`
- `replans=`
- `trace`
- `internal_reason`
- `forbidden_agent_actions`

Internal trace, usage, citations, and context metadata can remain in structured
debug fields of `AgentResponse`.

## Checkpoint v3 Policy

Checkpoint state is v3-only:

- `state_schema_version = 3`
- `thread_id = user_id:session_id`
- v2 checkpoints are not migrated
- non-v3 checkpoints are ignored by `_checkpoint_values`
- default SQLite file uses `langgraph_checkpoints_v3.sqlite3`

Checkpointed values are sanitized to JSON-like primitives only.
