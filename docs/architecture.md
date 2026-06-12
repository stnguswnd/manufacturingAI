# Manufacturing AI Agent Architecture

## Runtime Flow

```text
User Request
  -> RootManufacturingGraph
  -> Checkpoint v2 thread lookup (thread_id = user_id:session_id)
  -> ContextCompressor
  -> ContextResolver
  -> ContextPackBuilder
  -> IntentGateway / structured classifier
  -> selected path
  -> answer node or heavy manufacturing path
  -> FormatterRegistry
  -> AnswerMemoryWriter
  -> checkpoint v2 save
```

`root_graph.py` is kept as the orchestration boundary. Context resolution,
context packing, answer memory, formatting, safety policy, and checkpointing
are handled by dedicated modules.

## ContextResolution

`ContextResolver` receives only:

- `current_user_message`
- `last_answer_memory`
- `recent_turns`
- `rolling_summary`

It does not receive raw RAG documents, full conversation history, formatter
context, or legacy focus fields. It returns a structured `ContextResolution`
with:

- `is_followup`
- `followup_type`
- `followup_target`
- `followup_item_index`
- `standalone_query`
- `context_needed`
- `confidence`
- `reason`

Examples:

- `왜?` with answer memory -> `previous_answer_reason`
- `방금 권장조치 순서대로 알려줘` -> `previous_recommended_actions`
- `그중 2번은 왜 필요한데?` -> `previous_recommended_action_item`
- `LOTO가 뭐야?` -> standalone concept question, even if the previous answer was heavy analysis

## ContextPackBuilder

`ContextPackBuilder` creates node-specific context contracts:

- `classifier_context`
- `answer_context`
- `rag_context`
- `safety_context`
- `formatter_context`
- `memory_writer_context`

The classifier context is intentionally small. It can include the current
message, standalone query, follow-up metadata, last answer summary/focus, recent
turn intents, and glossary summary. It must not include retrieved document
text, raw messages, full conversation history, safety manuals, raw exceptions,
or the full root state.

`recommended_actions` is only included in `formatter_context` when
`answer_type == recommended_action_recap`. A single action item is only included
when `answer_type == recommended_action_item_explanation`.

## FormatterRegistry

`FormatterRegistry` renders already-decided routes. Formatters do not infer
intent from the user question.

Current formatter keys:

- `fast_concept_answer`
- `general_lightweight_answer`
- `recommended_action_recap`
- `recommended_action_item_explanation`
- `safety_answer`
- `clarification`
- `fallback`

This prevents the old problem where a lightweight answer could accidentally
inherit the heavy analysis format (`판정`, `위험도`, `권장 조치`, etc.).

## SafetyPolicy

Safety routing is not decided by the formatter. Once a route requires safety,
`SafetyPolicy` builds a `SafetyContext` containing:

- `must_include`
- `forbidden`
- `disclaimer_level`
- `requires_professional_review`
- `allowed_scope`

`SafetyFormatter` renders this safety context. Fast concept answers should not
receive a full safety checklist unless the route explicitly requires it.

## AnswerMemory

Legacy state fields are removed:

- `last_focus`
- `last_answer_claims`
- `last_answer_key_phrases`
- top-level `recommended_actions`

The single answer-memory object is:

- `selected_path`
- `answer_type`
- `user_intent`
- `short_summary`
- `focus`
- `key_points`
- `claims`
- `recommended_actions`
- `decisions`
- `code_changes`
- `tables`
- `mentioned_entities`
- `unresolved_questions`
- `source_refs`
- `safety_level`
- `created_at`
- `expires_after_turns`

`AnswerMemory.recommended_actions` is always a list of structured
`RecommendedAction` objects:

- `id`
- `title`
- `rationale`
- `safety_note`
- `priority`

String actions from existing tools or LLM output are normalized before being
stored internally.

## Checkpoint v2 Policy

Checkpoint state is v2-only:

- `state_schema_version = 2`
- `thread_id = user_id:session_id`
- v1 checkpoints are not migrated
- v1 checkpoints are ignored by `_checkpoint_values`
- default SQLite file uses a v2 name: `langgraph_checkpoints_v2.sqlite3`

Checkpointed state is sanitized to primitives only:

- allowed: `dict`, `list`, `str`, `int`, `float`, `bool`, `None`
- not allowed: Pydantic models, `AgentRequest`, `AgentResponse`, LangChain
  messages, or arbitrary Python objects

The reset helper removes only the v2 checkpoint file:

```bash
cd ai_server
../.venv312/bin/python -m app.agent.checkpointing.reset
```

## Hard Gate Audit

Hard gates are retained only where they protect safety, tool scope, or obvious
no-LLM paths. Semantic follow-up decisions are handled by `ContextResolver` and
the intent classifier. Gate definitions live under `app/agent/routing/`.

| Location | Terms | Purpose | Misclassification Prevented | Final Decision or Candidate Signal | Tests |
| --- | --- | --- | --- | --- | --- |
| `ControlScopeGate` | stop/start/control terms | refuse direct machine control | treating unsafe control as normal Q&A | final safety/scope gate | gateway tests |
| `MetaFeedbackGate` | bug/context/routing feedback | route product feedback away from manufacturing analysis | heavy report format on system feedback | final meta route | `test_meta_feedback_preserves_focus_policy` |
| `ProcessDataDiagnosisGate` | risk/failure/current condition terms with process data | force real process-data questions to heavy analysis | lightweight answer for risk judgement | final hard gate only when `process_data` exists | `test_process_data_risk_question_is_hard_gated_to_supervisor` |
| `ReportGate` | report/documentation terms or `generate_report` | route report generation to heavy/report flow | losing report request in lightweight path | final hard gate | route behavior tests |
| `SafetyRequestGate` | safety/maintenance terms | require safety-aware planning | unsafe lightweight maintenance advice | final safety hard gate | `test_safety_request_is_not_lightweight` |
| `DocumentRequestGate` | document/source terms | route document-backed question to RAG | unsupported citation request | final lightweight RAG gate | gateway tests |
| `GlossaryConceptGate` | simple concept terms + glossary hit | no-LLM glossary fast path | unnecessary LLM/heavy path for concepts | final no-LLM fast path, only with glossary hit | `test_no_llm_glossary_concept_uses_fast_path` |
| `RecommendedActionFollowupGate` | `ContextResolution.followup_type` | route action recap/item follow-ups to dedicated formatters | action recap falling into `general_lightweight_answer` | final follow-up gate after ContextResolver | `test_recommended_action_followup_routes_to_recap_not_lightweight` |
| `FollowupCandidateGate` | pronoun/reason/action signals | expose a candidate signal only | candidate signal accidentally selecting a path | candidate signal, never final | `test_followup_candidate_gate_does_not_make_final_route_decision` |
| `followup_signals.py` markers | pronoun/reason/action signals | produce resolver input signals | treating short follow-up as standalone | candidate signal only | `ContextResolver` tests |
| `DiagnosticFallbackPolicy` | manufacturing planning hints | deterministic heavy-route fallback policy | missing RAG/safety/prediction stages in heavy path | structured `DiagnosticPlan`, then converted to `AgentPlan` | `test_supervisor_keyword_policy_is_hidden_behind_diagnostic_planner`, `test_diagnostic_planner_returns_structured_plan` |

Risk note: deterministic manufacturing planning still exists, but it is hidden
behind `DiagnosticPlanner` and represented as a structured `DiagnosticPlan`
before the rest of the graph sees it. Root graph callers should not inspect
keyword matches directly.

## Heavy Manufacturing Path Audit

The heavy path is partially separated by LangGraph nodes:

- `supervisor_planning`: diagnostic planning/data requirement decision
- `manufacturing_analysis`: prediction and domain context
- `evidence_retrieval`: RAG retrieval and weak-evidence replan
- `safety`: safety gate/action payload preparation
- `response_synthesis`: LLM answer composition and safety validation
- `documentation`: report generation
- `response_packager`: public response object

Remaining coupling:

- `SupervisorService` still owns LLM refinement and replan behavior, but
  deterministic fallback planning is isolated behind `DiagnosticPlanner`.
- RAG is now split by role, but future work can promote these classes into
  first-class LangGraph subgraph nodes with independent retries and metrics.

Current heavy modules:

1. `DiagnosticPlanner`
2. `RagQueryPlanner`
3. `Retriever`
4. `EvidenceFilter`
5. `EvidenceGrader`
6. `CitationBuilder`
7. `SafetyGateBuilder`
8. `RecommendationBuilder`
9. `StructuredAnswerPayloadBuilder`

RAG role contracts:

| Component | Responsibility | Explicit Non-Responsibility |
| --- | --- | --- |
| `RagQueryPlanner` | build retrieval request/query | no retrieval execution |
| `Retriever` | execute search against `RagService` | no query planning or grading |
| `EvidenceFilter` | dedupe and remove empty evidence | no relevance grading |
| `EvidenceGrader` | decide usable/weak evidence | no citation building |
| `CitationBuilder` | build citations from graded evidence | no retrieval or grading |

ManufacturingAgentGraph removed method audit:

| Removed Method | Replacement |
| --- | --- |
| removed RAG query helper | `RagQueryPlanner.plan` |
| removed evidence term helper | `EvidenceGrader.grade` |
| removed action phrase helper | `RecommendationBuilder.collect_action_phrases` |
| removed safety guidance helper | `SafetyGateBuilder.safety_guidance` |
| removed warning helper | `SafetyGateBuilder.warnings` |
| removed payload helper | `StructuredAnswerPayloadBuilder.build` |

Recommended next work:

1. promote RAG query/retrieve/filter/grade/citation classes into LangGraph
   subgraph nodes with local replan edges
2. make `RecommendationBuilder` produce rich rationale/safety notes from domain
   catalog metadata
3. keep `ManufacturingAgentGraph` wrappers thin or remove them once all callers
   use dedicated modules directly

## Architecture Contract Tests

| Contract | Test |
| --- | --- |
| classifier context excludes retrieved docs and raw messages | `test_context_pack_builder_excludes_raw_docs_and_messages_from_classifier_context` |
| new concept question is not polluted by previous heavy memory | `test_previous_heavy_memory_does_not_pollute_new_concept_question`, `test_context_resolver_new_concept_not_polluted_by_heavy_memory` |
| recommended action recap uses dedicated formatter | `test_integrated_recommended_action_recap_uses_dedicated_formatter`, `test_recommended_action_followup_routes_to_recap_not_lightweight` |
| recommended action item uses action rationale/safety note | `test_integrated_recommended_action_item_uses_action_rationale`, `test_recommended_action_item_formatter_uses_item_rationale_and_safety_note` |
| checkpoint state is primitive-only | `test_checkpoint_state_contains_only_json_like_values` |
| thread id separates user/session memory | `test_user_session_checkpoint_memory_isolation`, `test_checkpoint_thread_id_policy_and_reset` |
| general lightweight formatter does not infer recap from action presence | `test_formatter_does_not_infer_recap_from_actions_on_general_path` |
| fast concept formatter does not leak heavy format | `test_fast_concept_formatter_does_not_leak_heavy_format` |
| follow-up candidate gate is not final | `test_followup_candidate_gate_does_not_make_final_route_decision` |
| supervisor keyword fallback is hidden behind diagnostic planner | `test_supervisor_keyword_policy_is_hidden_behind_diagnostic_planner` |
| diagnostic planner returns structured plan | `test_diagnostic_planner_returns_structured_plan` |
| RAG query planner does not retrieve directly | `test_rag_query_planner_does_not_retrieve_directly` |
| evidence grader does not build citations | `test_evidence_grader_does_not_build_citations` |
| citation builder uses graded evidence | `test_citation_builder_uses_graded_evidence` |
| ManufacturingAgentGraph wrappers stay thin | `test_manufacturing_graph_wrappers_are_thin` |

## Current Guarantees

- New concept questions are not polluted by prior heavy answers.
- Follow-up action recap uses structured `AnswerMemory.recommended_actions`.
- Action item explanations use the selected action's `rationale` and
  `safety_note`.
- Checkpoint snapshots are JSON-like and include `state_schema_version = 2`.
- Different `user_id/session_id` pairs do not share `last_answer_memory`.
- Deterministic heavy-path keyword planning is represented as `DiagnosticPlan`
  behind `DiagnosticPlanner`.
- RAG responsibilities are separated into query planning, retrieval, filtering,
  grading, and citation building classes.

## Remaining Risks

- `DiagnosticFallbackPolicy` is still deterministic. It is hidden behind
  `DiagnosticPlanner`, but the next step is to make planning consume richer
  `IntentResult` and `ContextPacks` signals.
- RAG roles are split into classes, but they are not yet fully separate
  LangGraph subgraph nodes with independent retry, latency, and quality metrics.
- `ManufacturingAgentGraph` no longer exposes helper wrappers for RAG, safety,
  actions, or payload building. Root graph and heavy graph call dedicated
  modules directly.
