# Current Backend Architecture Audit

This audit reflects the current runtime after legacy execution paths were
removed.

## 1. Runtime entrypoints

- Main product endpoint: `POST /agent/send`.
- Streaming product endpoint: `POST /agent/send/stream`.
- RAG API/debug seam: `POST /rag/search`.
- Health/readiness: `GET /health`, `GET /ready`.
- User/domain/history/evaluation endpoints remain.
- The legacy run endpoint has been removed.

## 2. FastAPI endpoint map

| Endpoint | Runtime role | Primary callee |
| --- | --- | --- |
| `GET /health` | coarse service health | `RagService.corpus_health`, domain/LLM checks |
| `GET /ready` | readiness check | prediction/domain/history/RAG checks |
| `GET /llm/models` | LLM catalog | `LLMService` config |
| `GET /domain/*` | domain metadata | `DomainKnowledgeService` |
| `POST /users`, `GET/PATCH/DELETE /users/*` | user CRUD | `UserService` |
| `GET /users/{id}/context` | context preview | `ContextService` |
| `POST /agent/intent` | route preview | `RootManufacturingGraph.preview_route` |
| `POST /agent/plan` | plan preview | `RootManufacturingGraph.diagnostic_planner` |
| `POST /predict` | AI4I prediction only | `PredictionService` |
| `POST /rag/search` | RAG API/debug lookup | `RagService.search` |
| `POST /agent/send` | main agent answer | `RootManufacturingGraph.run` |
| `POST /agent/send/stream` | streaming main agent answer | `RootManufacturingGraph.run` |
| `GET /history*` | run history | `SQLiteStore` |
| `POST /evaluation/score` | answer evaluation | `evaluate_answer` |

## 3. Agent runtime flow

```text
FastAPI /agent/send
  -> AgentSendRequest validation
  -> RootManufacturingGraph.run
  -> request_context
  -> planning_router
  -> prediction_node -> prediction_quality_gate
  -> RagEvidenceSubAgent -> evidence_quality_gate
  -> SafetyContractSubAgent -> safety_contract_gate
  -> answer_compose
  -> answer_text_review
  -> next_action conditional edge
  -> response_packager
  -> memory_writer
  -> audit_persistence
  -> AgentResponse
```

`RootManufacturingGraph` is a real LangGraph
`StateGraph(ManufacturingAgentState)`. Runtime state is v3 artifact-only. The
root graph owns top-level node/edge orchestration and response projection;
subagents own context, RAG evidence, safety, and memory local workflows.

## 4. RAG Evidence SubAgent flow

`ai_server/app/agent/rag_evidence/subagent.py` builds a real LangGraph
`StateGraph(RagEvidenceState)`:

```text
plan_queries -> retrieve -> filter -> grade -> cite -> build_payload -> trace -> END
```

The root graph only builds `RagEvidenceInput`, invokes the SubAgent, and stores
`EvidenceArtifact` in `state["evidence"]`.

Other SubAgents follow the same boundary:

| SubAgent | Flow |
| --- | --- |
| `ContextSubAgent` | request/session context -> conversation resolution -> context packs -> validation -> output |
| `PlanningSubAgent` | planning context -> diagnostic planner -> validation -> output |
| `SafetySubAgent` | safety context -> safety policy -> validation -> output |
| `MemorySubAgent` | answer memory extraction -> focus update -> memory write -> output |

## 5. Services responsibility map

| Service | Current responsibility | Runtime usage |
| --- | --- | --- |
| `PredictionService` | load existing AI4I model bundle and predict | `/predict`, `prediction_node` |
| `RagService` | public RAG seam and Chroma-backed search | `/rag/search`, RAG SubAgent |
| `ChromaRetriever` | Chroma connection, embedding, query, diagnostics | `RagService`; located in `app.services` |
| `LLMService` | OpenAI-compatible JSON generation | gateway/planning/answer compose |
| `DomainKnowledgeService` | YAML domain catalog and manufacturing context | root graph, Safety/RAG SubAgents |
| `IntentGatewayService` | hard gate + classifier + policy validation | root graph |
| `SupervisorService` | bounded `replan()` support plus LLM service holder for `DiagnosticPlanner` | `DiagnosticPlanner.replan` |
| `ContextService` | user/session context pack | ContextSubAgent |
| `MemoryService` | extract/store answer memory | MemorySubAgent |
| `UserService` | user CRUD/validation | user endpoints |
| `SafetyValidationService` | answer safety validation | SafetyCritic / answer review |

## 6. Schema/state boundary

- `app.schemas.agent`: agent request/response/plan/trace/LLM usage.
- `app.schemas.prediction`: AI4I process data and prediction result.
- `app.schemas.domain`: manufacturing context, risk, safety gates, action plan.
- `app.schemas.rag`: RAG search request and chunk model.
- `app.schemas.user`: user CRUD schemas.
- `app.schemas.evaluation`: evaluation request/response.
- `app.agent.state`: root graph runtime `TypedDict`.
- `app.agent.context_subagent.state`: Context SubAgent state and boundary
  models.
- `app.agent.planning_subagent.state`: Planning SubAgent state and boundary
  models.
- `app.agent.rag_evidence.state`: RAG SubAgent runtime state and boundary
  models.
- `app.agent.safety_subagent.state`: Safety SubAgent state and boundary models.
- `app.agent.memory_subagent.state`: Memory SubAgent state and boundary models.

`app.schemas` is a package marker only, not a barrel re-export.

## 7. Storage and external dependencies

- Primary runtime store: `SQLiteStore`.
- Root graph checkpointer: LangGraph SQLite checkpointing by user/session
  thread id.
- Model artifact: `ai_server/storage/models/ai4i_model_bundle.joblib`.
- RAG source of truth: `ai_server/data/processed/rag_chunks.jsonl`.
- Chroma persist dir: `ai_server/data/vector_db/chroma`.
- Chroma collection: `manufacturing_rag`.
- Embedding model: `text-embedding-3-small`.

Prediction runtime does not auto-train missing models. Run
`scripts/train_ai4i_model.py` explicitly.

## 8. Tests map

`pytest` currently passes 106 tests.

| Test file | Production path covered |
| --- | --- |
| `test_rag_evidence_orchestration.py` | RAG Evidence SubAgent flow and root handoff |
| `test_artifact_only_graph.py` | v3 artifact-only state, graph edges, invalidation, public/debug separation |
| `test_chroma_runtime_rag.py` | Chroma retriever/filter/grader/citation behavior with fakes |
| `test_rag_and_safety.py` | explicit JSONL dev RAG behavior and safety validation |
| `test_context_engineering.py` | context, formatter registry, checkpoint, diagnostic planner, boundaries |
| `test_intent_gateway.py` | hard gates, classifier degrade behavior, follow-up routing |
| `test_user_context.py`, `test_memory_service.py` | user isolation and memory |
| ingestion tests | KOSHA parsing, document building, chunking |

## 9. Dead code / legacy candidates

Remaining candidates for later cleanup:

1. Historical docs that describe removed pre-StateGraph execution.
2. Broad `except Exception` blocks in LLM, intent, RAG diagnostic, and
   checkpoint paths.
3. Remaining broad defensive paths in classifier/LLM services.
4. Ignored pycache files from removed modules.

## 10. Risky defensive code candidates

- `LLMService.generate_json(...)` returns `None` with `last_error`; callers must
  surface explicit failures.
- Intent classifier degrade paths are allowed only when the route/warning is
  explicit.
- RAG retrieval failure returns empty evidence with warnings. It must not use
  JSONL fallback when Chroma is enabled.
- Observability errors may be swallowed because they must not affect product
  state.

## 11. Refactor priorities

### P0

- Audit remaining broad exception handlers for explicit diagnostics.
- Tighten classifier/LLM defensive paths so failures remain observable.
- Tighten LLM failure behavior where callers still rely on `None`.

### P1

- Move historical records under a clear archive path.
- Add a dependency direction check for service-to-agent imports.

### P2

- Corpus versioning and index health automation.
- Admin/debug UI for corpus health.
- Optional response synthesis subgraph if root graph grows again.

## 12. Do-not-touch list

- AI4I model bundle.
- SQLite runtime history/checkpoint files.
- Processed RAG JSONL/report artifacts.
- Chroma vector DB files.
- Ingestion outputs.
- `/rag/search` public API/debug seam without explicit product-surface review.
