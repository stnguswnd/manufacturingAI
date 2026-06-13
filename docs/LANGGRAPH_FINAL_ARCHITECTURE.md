# LangGraph v3 Final Architecture

현재 제조 AI Agent는 unrestricted autonomous multi-agent system이 아니다.
제조 안전 도메인에 맞춰 Prediction, Adaptive RAG Evidence, Safety Contract,
Answer Text Review를 Artifact contract와 LangGraph conditional edge로 제어하는
bounded agentic workflow다.

## Core Graph

```text
START
  -> request_context
  -> planning_router
  -> prediction_node -> prediction_quality_gate -> planning_router
  -> rag_evidence_subagent(Adaptive RAG) -> evidence_quality_gate -> planning_router
  -> safety_contract_subagent -> safety_contract_gate -> planning_router
  -> answer_compose
  -> answer_text_review
  -> pass / rewrite_only / rerun_rag / rerun_safety / clarification_required / block / max_retry_exceeded
  -> response_packager
  -> memory_writer
  -> audit_persistence
  -> END
```

Artifact quality gate, answer compose, text review, rewrite, stale artifact
invalidation, upstream rerun, block, fallback은 모두 명시적 graph node/edge다.

## Runtime State

Root runtime state는 `ManufacturingAgentState` v3다.

```python
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

Root-level `plan`, `retrieved_documents`, `citations`, `safety_guidance`,
`structured_answer_payload`, `manufacturing_context`, `answer`, `trace`,
`usage_records`는 runtime state에 없다.

## Quality Gate And Text Review Edges

```text
prediction_node -> prediction_quality_gate -> planning_router
rag_evidence_subagent -> evidence_quality_gate -> planning_router
safety_contract_subagent -> safety_contract_gate -> planning_router
fast_answer -> output_policy_gate -> response_packager

pass -> response_packager
rewrite_only -> invalidate_rewrite -> answer_rewrite -> answer_text_review
rerun_rag -> invalidate_rag_downstream -> rag_evidence_subagent -> planning_router
rerun_safety -> invalidate_safety_downstream -> safety_contract_subagent -> planning_router
clarification_required -> clarification_response -> response_packager
block -> safe_block_response -> response_packager
max_retry_exceeded -> safe_review_fallback -> response_packager
```

## Public API

`/agent/send`는 기존 `AgentResponse` schema를 유지한다. 내부 artifact는
public schema에 직접 노출하지 않고 `response_packager`와 final projection에서
public field로 변환한다.

Public answer text에는 run id, token, cost, model, raw score, chunk id,
safety gate id, gate id, calls, replans, trace, internal reason을 노출하지
않는다.
