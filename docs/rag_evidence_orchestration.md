# Adaptive RAG Evidence SubAgent

`rag_evidence_subagent` is the Adaptive RAG execution unit in the bounded
workflow. It is separate from the AI4I prediction tool: AI4I process data drives
risk prediction, while OSHA, Haas, and KOSHA documents provide maintenance,
safety, and troubleshooting evidence.

## Runtime Shape

```text
RootManufacturingGraph
  -> planning_router
  -> RagEvidenceSubAgent.invoke(...)
       -> profile selection
       -> plan_queries
       -> retrieve
       -> filter
       -> grade
       -> cite
       -> build_payload / EvidenceArtifact
       -> trace
  -> planning_router
  -> evidence_quality_gate
  -> planning_router
  -> safety_contract_subagent / answer_compose
```

The root graph does not call RAG internals directly. It converts
`RequestArtifact`, `PlanningArtifact`, `PredictionArtifact`, and
`ContextArtifact` into `RagEvidenceInput`, invokes the compiled subagent graph,
and stores only `state["evidence"] = EvidenceArtifact`.

`EvidenceArtifact` is first consumed by `evidence_quality_gate`. After it passes,
SafetyContractSubAgent and AnswerComposer can consume it. It is not a public
answer text surface.

## State

Graph state is request-scoped and lives in `app.agent.rag_evidence.state`.
It carries only the active request, plan, prediction/context snapshot, query
specs, retrieved chunks, filtered/selected chunks, grade, citations, warnings,
trace, and output.

No request-specific state is stored on services.

## Adaptive Profile And Query Fan-Out

Fan-out is deterministic and bounded to four query specs. The active profile is
stored as `trace["retrieval_profile"]` and `EvidenceArtifact.profile`.

Profiles:

- `prediction_plus_rag`
- `rag_only_safety`
- `troubleshooting_rag`
- `concept_explanation`

Possible query spec names:

- `primary`
- `safety_{gate_id}`
- `troubleshooting`
- `failure_mode`

Safety gates can add gate-specific metadata terms and title supplements. OSF/TWF
or troubleshooting profiles can add troubleshooting and failure-mode specs. The
policy does not call an LLM.

## Selection And Trace

Evidence selection prefers usable, relevant chunks first, then failure/signal or
safety-gate alignment, high/medium priority, and finally limited diversity.

The trace is compact and log-safe. It includes query spec names, backend,
counts, selected sources, selected safety gates, warnings, and corpus count
mismatch status. It does not include raw chunk text, API keys, full prompts, or
large local paths.

Evidence diagnostics include deterministic critic fields:

- `citation_coverage_ok`
- `selected_chunk_ids_unique`
- `selected_doc_ids_unique`
- `evidence_grade_selected_consistent`
- `required_safety_gates`
- `evidence_covers_required_gates`
- `missing_gate_evidence`
- `generic_document_downgraded`

For safety questions, missing required gate evidence is reported in
`EvidenceArtifact` so `evidence_quality_gate` can choose `rerun_rag` before
answer composition. It does not silently mark unrelated evidence as usable.

SafetyContractSubAgent does not call Chroma, retrievers, or citation builders.
It consumes `EvidenceArtifact` and turns selected evidence plus safety gate YAML
into `SafetyArtifact`.

Chroma failures do not fall back to JSONL search in the RAG Evidence path.
Failures produce empty evidence plus explicit warnings.

## Chroma Health

The expected local corpus has 727 JSONL chunks and 727 Chroma vectors after
rebuild. If a different environment reports a mismatch, runtime requests emit:

```text
Chroma collection count mismatch: expected 727, actual <actual>. Retrieval continues. Reindex corpus separately.
```

This subagent does not sync, reindex, download, or mutate the corpus.

## Not Included

- Streamlit upload/vectorize UI
- corpus versioning
- ingestion redesign
- Chroma sync or automatic reindex
- Safety or formatter subgraph migration
