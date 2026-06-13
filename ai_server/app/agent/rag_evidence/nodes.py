from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from app.agent.artifacts import EvidenceArtifact
from app.agent.heavy.citation_builder import CitationBuilder
from app.agent.heavy.evidence_filter import EvidenceFilter
from app.agent.heavy.evidence_grader import EvidenceGrader
from app.agent.heavy.rag_query_planner import RagFanoutPolicy, RagQueryPlanner
from app.agent.heavy.rag_schemas import EvidenceGrade
from app.config import RAG_CORPUS_EXPECTED_COUNT
from app.schemas.agent import AgentPlan, AgentRequest
from app.schemas.domain import ManufacturingContext
from app.schemas.prediction import PredictionResponse
from app.schemas.rag import RagChunk
from app.services.domain_service import DomainKnowledgeService
from app.services.rag_service import RagService

from .state import RagEvidenceState


@dataclass(frozen=True)
class RagEvidenceDeps:
    query_planner: RagQueryPlanner
    fanout_policy: RagFanoutPolicy
    rag_service: RagService
    evidence_filter: EvidenceFilter
    evidence_grader: EvidenceGrader
    citation_builder: CitationBuilder
    domain_service: DomainKnowledgeService
    expected_count: int = RAG_CORPUS_EXPECTED_COUNT


def plan_queries(state: RagEvidenceState, deps: RagEvidenceDeps) -> RagEvidenceState:
    request, plan, prediction, context = _models(state)
    primary = deps.query_planner.plan(
        request=request,
        planned_query=plan.rag_query,
        prediction=prediction,
        manufacturing_context=context,
        top_k=int(state.get('top_k') or 5),
        filters=plan.rag_filters,
    )
    specs = deps.fanout_policy.build(primary, request=request, plan=plan, manufacturing_context=context)
    trace = dict(state.get('trace') or {})
    trace['retrieval_profile'] = (specs[0].get('profile') if specs else None) or 'concept_explanation'
    return {'query_specs': specs[: deps.fanout_policy.max_query_specs], 'trace': trace}


def retrieve(state: RagEvidenceState, deps: RagEvidenceDeps) -> RagEvidenceState:
    raw: list[RagChunk] = []
    spec_traces: list[dict[str, Any]] = []
    backends: list[str] = []
    warnings = list(state.get('warnings') or [])
    for spec in state.get('query_specs') or []:
        try:
            result = deps.rag_service.search_with_diagnostics(spec['query'], top_k=int(spec.get('top_k') or 5), filters=spec.get('filters'))
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            warnings.append(f'RAG retrieval failed for query spec {spec.get("name") or "unknown"}: {error}. Retrieval continues.')
            backends.append('error')
            spec_traces.append({
                'name': spec.get('name'),
                'intent': spec.get('intent'),
                'returned_count': 0,
                'backend': 'error',
                'error': error,
            })
            continue
        chunks = [chunk if isinstance(chunk, RagChunk) else RagChunk.model_validate(chunk) for chunk in result.get('chunks') or []]
        diagnostics = result.get('diagnostics') or {}
        if diagnostics.get('error'):
            warnings.append(f'RAG retrieval failed for query spec {spec.get("name") or "unknown"}: {diagnostics["error"]}. Retrieval continues.')
        if spec.get('metadata_terms'):
            supplement = _metadata_supplement(deps.rag_service, spec)
            if supplement.get('diagnostics', {}).get('error'):
                warnings.append(f'RAG metadata supplement failed for query spec {spec.get("name") or "unknown"}: {supplement["diagnostics"]["error"]}. Retrieval continues.')
            chunks.extend(chunk if isinstance(chunk, RagChunk) else RagChunk.model_validate(chunk) for chunk in supplement.get('chunks') or [])
        raw.extend(chunks)
        backends.append(str(diagnostics.get('backend') or 'empty'))
        spec_traces.append({
            'name': spec.get('name'),
            'intent': spec.get('intent'),
            'profile': spec.get('profile'),
            'returned_count': len(chunks),
            'backend': diagnostics.get('backend'),
            'error': diagnostics.get('error'),
        })
    deduped = _dedupe(raw)
    trace = dict(state.get('trace') or {})
    trace['retrieval_backend'] = _backend_label(backends, deduped)
    trace['raw_count'] = len(deduped)
    return {'raw_chunks': [chunk.model_dump() for chunk in deduped], 'trace': trace, 'warnings': warnings}


def _metadata_supplement(rag_service: RagService, spec: dict[str, Any]) -> dict[str, Any]:
    search = getattr(rag_service, 'metadata_search', None)
    if not callable(search):
        return {'chunks': [], 'diagnostics': {'backend': 'unavailable', 'error': None}}
    return search(
        spec.get('metadata_terms') or [],
        top_k=min(int(spec.get('top_k') or 5), 8),
        allow_restricted=bool(spec.get('allow_restricted')),
    )


def filter_evidence(state: RagEvidenceState, deps: RagEvidenceDeps) -> RagEvidenceState:
    chunks = _chunks(state.get('raw_chunks') or [])
    filters = _merged_filters(state.get('query_specs') or [])
    filtered = deps.evidence_filter.filter(chunks, filters=filters)
    trace = dict(state.get('trace') or {})
    trace['filtered_count'] = len(filtered)
    return {'filtered_chunks': [chunk.model_dump() for chunk in filtered], 'trace': trace}


def _merged_filters(query_specs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not query_specs:
        return None
    merged = dict(query_specs[0].get('filters') or {})
    if any(spec.get('allow_restricted') for spec in query_specs):
        merged['include_restricted'] = True
    return merged


def grade_evidence(state: RagEvidenceState, deps: RagEvidenceDeps) -> RagEvidenceState:
    request = AgentRequest.model_validate(state['request'])
    chunks = _chunks(state.get('filtered_chunks') or [])
    grade = deps.evidence_grader.grade(request.question, chunks)
    return {'evidence_grade': grade.model_dump()}


def build_citations(state: RagEvidenceState, deps: RagEvidenceDeps) -> RagEvidenceState:
    grade = EvidenceGrade.model_validate(state.get('evidence_grade') or {'usable': False, 'weak_reason': 'no_grade'})
    chunks = _chunks(state.get('filtered_chunks') or [])
    citations = deps.citation_builder.build(chunks, grade) if grade.usable else []
    return {'citations': citations}


def build_payload(state: RagEvidenceState, deps: RagEvidenceDeps) -> RagEvidenceState:
    request, plan, prediction, context = _models(state)
    grade = EvidenceGrade.model_validate(state.get('evidence_grade') or {'usable': False, 'weak_reason': 'no_grade'})
    filtered = _chunks(state.get('filtered_chunks') or [])
    profile = (state.get('trace') or {}).get('retrieval_profile') or 'concept_explanation'
    selected = _select_evidence(filtered, request=request, plan=plan, context=context, profile=profile) if grade.usable else []
    citations_by_id = {item.get('chunk_id'): item for item in (state.get('citations') or [])}
    citations = [citations_by_id[chunk.chunk_id] for chunk in selected if chunk.chunk_id in citations_by_id]
    manufacturing_context = deps.domain_service.build_context(request, prediction, doc_count=len(selected))
    trace = dict(state.get('trace') or {})
    trace.update({
        'selected_count': len(selected),
        'citation_count': len(citations),
        'citation_coverage_ok': len(citations) == len(selected),
        'selected_chunk_ids_unique': len({chunk.chunk_id for chunk in selected if chunk.chunk_id}) == len([chunk for chunk in selected if chunk.chunk_id]),
        'selected_doc_ids_unique': len({_doc_key(chunk) for chunk in selected}) == len(selected),
        'evidence_grade_selected_consistent': grade.usable or (not selected and not citations),
        'selected_sources': _unique([chunk.source for chunk in selected]),
        'selected_safety_gates': _unique([chunk.safety_gate for chunk in selected if chunk.safety_gate]),
        'evidence_judge': _evidence_judge_decision(selected, profile=profile, plan=plan, context=context),
    })
    return {
        'selected_chunks': [chunk.model_dump() for chunk in selected],
        'citations': citations,
        'manufacturing_context': manufacturing_context.model_dump(),
        'trace': trace,
    }


def build_trace(state: RagEvidenceState, deps: RagEvidenceDeps) -> RagEvidenceState:
    trace = dict(state.get('trace') or {})
    warnings = list(state.get('warnings') or [])
    selected = _chunks(state.get('selected_chunks') or [])
    citations = list(state.get('citations') or [])
    try:
        health = deps.rag_service.corpus_health()
    except Exception as exc:
        health = {'backend': 'error', 'chroma_count': None, 'error': f'{type(exc).__name__}: {exc}'}
    actual = health.get('chroma_count')
    mismatch = actual is not None and deps.expected_count and actual != deps.expected_count
    if mismatch:
        warnings.append(f'Chroma collection count mismatch: expected {deps.expected_count}, actual {actual}. Retrieval continues. Reindex corpus separately.')
    if health.get('error'):
        warnings.append(f'Chroma health check failed: {health["error"]}. Retrieval continues.')
    trace.update({
        'query_spec_names': [spec.get('name') for spec in state.get('query_specs') or [] if spec.get('name')],
        'raw_count': int(trace.get('raw_count') or len(state.get('raw_chunks') or [])),
        'filtered_count': int(trace.get('filtered_count') or len(state.get('filtered_chunks') or [])),
        'selected_count': len(selected),
        'citation_count': len(citations),
        'selected_sources': _unique([chunk.source for chunk in selected]),
        'selected_safety_gates': _unique([chunk.safety_gate for chunk in selected if chunk.safety_gate]),
        'corpus_count_mismatch': bool(mismatch),
        'warnings': list(dict.fromkeys(warnings)),
    })
    state_with_final_diagnostics = {
        **state,
        'trace': trace,
        'warnings': trace['warnings'],
    }
    grade = EvidenceGrade.model_validate(state.get('evidence_grade') or {'usable': False, 'weak_reason': 'no_grade'})
    evidence_artifact = _evidence_artifact(
        state=state_with_final_diagnostics,
        grade=grade,
        selected=selected,
        citations=citations,
    )
    trace['evidence_artifact'] = {
        'profile': evidence_artifact.profile,
        'required_safety_gates': evidence_artifact.required_safety_gates,
        'evidence_covers_required_gates': evidence_artifact.evidence_covers_required_gates,
        'missing_gate_evidence': evidence_artifact.missing_gate_evidence,
    }
    output = {
        'plan': state['plan'],
        'route': list((state.get('plan') or {}).get('required_nodes') or []),
        'retrieved_documents': state.get('selected_chunks') or [],
        'citations': state.get('citations') or [],
        'evidence_grade': grade.model_dump(),
        'evidence_artifact': evidence_artifact.model_dump(),
        'manufacturing_context': state['manufacturing_context'],
        'warnings': trace['warnings'],
        'trace': trace,
        'replan_count_delta': 0,
    }
    return {'trace': trace, 'warnings': trace['warnings'], 'output': output}


def _models(state: RagEvidenceState) -> tuple[AgentRequest, AgentPlan, PredictionResponse | None, ManufacturingContext]:
    request = AgentRequest.model_validate(state['request'])
    plan = AgentPlan.model_validate(state['plan'])
    prediction = PredictionResponse.model_validate(state['prediction']) if state.get('prediction') else None
    context = ManufacturingContext.model_validate(state['manufacturing_context'])
    return request, plan, prediction, context


def _chunks(rows: list[dict[str, Any]]) -> list[RagChunk]:
    return [RagChunk.model_validate(row) for row in rows]


def _dedupe(chunks: list[RagChunk]) -> list[RagChunk]:
    seen: set[str] = set()
    out: list[RagChunk] = []
    for chunk in chunks:
        key = chunk.chunk_id or f'{chunk.doc_id}:{chunk.document_title}:{hashlib.sha1(chunk.text[:300].encode("utf-8")).hexdigest()}'
        if key in seen:
            continue
        seen.add(key)
        out.append(chunk)
    return out


def _select_evidence(chunks: list[RagChunk], *, request: AgentRequest, plan: AgentPlan, context: ManufacturingContext, profile: str) -> list[RagChunk]:
    relevant = [chunk for chunk in chunks if _is_relevant(chunk, request=request, context=context)]
    candidates = sorted(
        relevant or [],
        key=lambda chunk: _selection_score(chunk, request=request, plan=plan, context=context, profile=profile),
        reverse=True,
    )
    selected: list[RagChunk] = []
    if profile == 'rag_only_safety':
        _pick_by_group(selected, candidates, lambda c: _matches_safety_context(c, context), lambda c: c.safety_gate or _doc_key(c), limit=3)
    elif profile == 'prediction_plus_rag':
        _pick(selected, candidates, lambda c: _matches_failure_mode(c, context))
        _pick(selected, candidates, lambda c: (c.doc_type or '').strip().lower() == 'troubleshooting')
        if plan.safety_required or plan.safety_gate_required:
            _pick_by_group(selected, candidates, lambda c: _matches_safety_context(c, context), lambda c: c.safety_gate or _doc_key(c), limit=2)
    elif profile == 'troubleshooting_rag':
        _pick(selected, candidates, lambda c: (c.doc_type or '').strip().lower() == 'troubleshooting')
        if plan.safety_required or plan.safety_gate_required:
            _pick_by_group(selected, candidates, lambda c: _matches_safety_context(c, context), lambda c: c.safety_gate or _doc_key(c), limit=2)
    else:
        _pick(selected, candidates, lambda c: True)
    if not selected:
        selected_doc_keys: set[str] = set()
        for chunk in candidates:
            if len(selected) >= 3:
                break
            if _doc_key(chunk) not in selected_doc_keys:
                selected.append(chunk)
                selected_doc_keys.add(_doc_key(chunk))
    _fill_to_limit(selected, candidates, limit=3)
    return selected[:3]


def _selection_score(chunk: RagChunk, *, request: AgentRequest, plan: AgentPlan, context: ManufacturingContext, profile: str) -> float:
    blob = _chunk_blob(chunk)
    metadata_blob = _chunk_metadata_blob(chunk)
    title_blob = _chunk_title_blob(chunk)
    score = float(chunk.score or 0.0) * 8.0
    score += {'high': 18, 'medium': 10, 'low': 0}.get((chunk.project_priority or '').strip().lower(), 3)
    gate = (chunk.safety_gate or '').strip().lower()
    doc_type = (chunk.doc_type or '').strip().lower()
    source = (chunk.source or '').strip().lower()
    if (chunk.retrieval_scope or '').strip().lower() == 'default':
        score += 8
    if (chunk.retrieval_scope or '').strip().lower() == 'restricted':
        score -= 2
    if profile in {'prediction_plus_rag', 'troubleshooting_rag'} and doc_type == 'troubleshooting':
        score += 34
    elif doc_type == 'troubleshooting':
        score += 12
    if profile == 'troubleshooting_rag' and source == 'haas':
        score += 12
    if gate == 'maintenance_check':
        score += 10
    if (plan.safety_required or plan.safety_gate_required) and _matches_safety_context(chunk, context):
        score += 34
    if (plan.safety_required or plan.safety_gate_required) and doc_type in {'safety_standard', 'safety_procedure', 'korean_machine_safety', 'korean_safety_reference'}:
        score += 16
    if _matches_failure_mode(chunk, context):
        score += 30
    terms = _context_terms(request, context)
    title_hits = sum(1 for term in terms if term in title_blob)
    metadata_hits = sum(1 for term in terms if term in metadata_blob)
    body_hits = sum(1 for term in terms if term in blob)
    score += min(title_hits, 5) * 14
    score += min(metadata_hits, 5) * 12
    score += min(body_hits, 8) * 3
    if plan.safety_required or plan.safety_gate_required:
        safety_terms = _safety_terms(context)
        score += min(sum(1 for term in safety_terms if term in title_blob), 5) * 16
        score += min(sum(1 for term in safety_terms if term in metadata_blob), 5) * 14
        score += min(sum(1 for term in safety_terms if term in blob), 6) * 4
    if title_hits == 0 and metadata_hits <= 1 and doc_type not in _preferred_doc_types(profile):
        score -= 18
    return score


def _pick(selected: list[RagChunk], chunks: list[RagChunk], predicate) -> None:
    selected_ids = {chunk.chunk_id for chunk in selected}
    selected_doc_keys = {_doc_key(chunk) for chunk in selected}
    selected_gates = {chunk.safety_gate for chunk in selected if chunk.safety_gate}
    for chunk in chunks:
        gate = chunk.safety_gate
        if (
            chunk.chunk_id not in selected_ids
            and _doc_key(chunk) not in selected_doc_keys
            and (not gate or gate not in selected_gates)
            and predicate(chunk)
        ):
            selected.append(chunk)
            return


def _pick_by_group(selected: list[RagChunk], chunks: list[RagChunk], predicate, group_key, *, limit: int) -> None:
    selected_ids = {chunk.chunk_id for chunk in selected}
    selected_doc_keys = {_doc_key(chunk) for chunk in selected}
    selected_groups = {group_key(chunk) for chunk in selected}
    for chunk in chunks:
        if len(selected_groups) >= limit:
            return
        group = group_key(chunk)
        if (
            group
            and group not in selected_groups
            and chunk.chunk_id not in selected_ids
            and _doc_key(chunk) not in selected_doc_keys
            and predicate(chunk)
        ):
            selected.append(chunk)
            selected_groups.add(group)
            selected_ids.add(chunk.chunk_id)
            selected_doc_keys.add(_doc_key(chunk))


def _fill_to_limit(selected: list[RagChunk], chunks: list[RagChunk], *, limit: int) -> None:
    selected_ids = {chunk.chunk_id for chunk in selected}
    selected_doc_keys = {_doc_key(chunk) for chunk in selected}
    selected_gates = {chunk.safety_gate for chunk in selected if chunk.safety_gate}
    for chunk in chunks:
        if len(selected) >= limit:
            return
        if chunk.chunk_id in selected_ids or _doc_key(chunk) in selected_doc_keys:
            continue
        if chunk.safety_gate and chunk.safety_gate in selected_gates and len(selected) < max(1, limit - 1):
            continue
        selected.append(chunk)
        selected_ids.add(chunk.chunk_id)
        selected_doc_keys.add(_doc_key(chunk))
        if chunk.safety_gate:
            selected_gates.add(chunk.safety_gate)


def _doc_key(chunk: RagChunk) -> str:
    return chunk.doc_id or f'{chunk.source}:{chunk.document_title or chunk.title}'


def _matches_safety_context(chunk: RagChunk, context: ManufacturingContext) -> bool:
    metadata = ' '.join([
        chunk.safety_gate or '',
        chunk.doc_type or '',
        chunk.document_title or '',
        chunk.title or '',
    ]).lower().replace('_', ' ')
    return any(term in metadata for term in _safety_identity_terms(context))


def _matches_failure_mode(chunk: RagChunk, context: ManufacturingContext) -> bool:
    modes = {mode.code for mode in context.failure_modes}
    return bool(modes and _contains_any(chunk.failure_modes, modes))


def _is_relevant(chunk: RagChunk, *, request: AgentRequest, context: ManufacturingContext) -> bool:
    blob = _chunk_blob(chunk)
    context_terms = _context_terms(request, context)
    if context_terms:
        return any(token in blob.replace('_', ' ') for token in context_terms)
    if any(token in blob for token in _tokens(request.question)):
        return True
    if _contains_any(chunk.failure_modes, {mode.code for mode in context.failure_modes}):
        return True
    return bool(chunk.safety_gate and chunk.safety_gate in {gate.gate_id for gate in context.safety_gates})


def _chunk_blob(chunk: RagChunk) -> str:
    return ' '.join([
        chunk.document_title or '',
        chunk.title or '',
        chunk.doc_type or '',
        chunk.safety_gate or '',
        chunk.text or '',
    ]).lower().replace('_', ' ')


def _chunk_metadata_blob(chunk: RagChunk) -> str:
    return ' '.join([
        chunk.document_title or '',
        chunk.title or '',
        chunk.doc_type or '',
        chunk.safety_gate or '',
    ]).lower().replace('_', ' ')


def _chunk_title_blob(chunk: RagChunk) -> str:
    return ' '.join([
        chunk.document_title or '',
        chunk.title or '',
    ]).lower().replace('_', ' ')


def _preferred_doc_types(profile: str) -> set[str]:
    if profile == 'rag_only_safety':
        return {'safety_standard', 'safety_procedure', 'korean_machine_safety', 'korean_safety_reference', 'korean_maintenance_guidance'}
    if profile in {'prediction_plus_rag', 'troubleshooting_rag'}:
        return {'troubleshooting', 'korean_maintenance_guidance', 'safety_standard', 'safety_procedure'}
    return {'korean_maintenance_guidance', 'manual', 'reference'}


def _safety_terms(context: ManufacturingContext) -> set[str]:
    terms: set[str] = set()
    for gate in context.safety_gates:
        terms.update(_tokens(gate.gate_id))
        terms.update(_tokens(gate.name_ko))
        terms.update(_tokens(gate.description_ko))
        for check in gate.required_checks:
            terms.update(_tokens(check))
        for trigger in gate.triggered_by:
            terms.update(_tokens(trigger))
        for term in gate.document_search_terms:
            terms.update(_tokens(term))
    return terms


def _safety_identity_terms(context: ManufacturingContext) -> set[str]:
    terms: set[str] = set()
    for gate in context.safety_gates:
        terms.update(_tokens(gate.gate_id))
        terms.update(_tokens(gate.name_ko))
        terms.update(_tokens(gate.description_ko))
        for trigger in gate.triggered_by:
            terms.update(_tokens(trigger))
        for term in gate.document_search_terms:
            terms.update(_tokens(term))
    return terms


def _context_terms(request: AgentRequest, context: ManufacturingContext) -> set[str]:
    terms = _tokens(request.question)
    for condition in context.process_conditions:
        terms.update(_tokens(f'{condition.tag} {condition.label_ko} {condition.source_feature or ""}'))
    for mode in context.failure_modes:
        terms.update(_tokens(' '.join([mode.code, mode.name_ko, *mode.recommended_checks[:3]])))
    for gate in context.safety_gates:
        terms.update(_tokens(f'{gate.gate_id} {gate.name_ko}'))
        for term in gate.document_search_terms:
            terms.update(_tokens(term))
    return terms


def _evidence_judge_decision(selected: list[RagChunk], *, profile: str, plan: AgentPlan, context: ManufacturingContext) -> dict[str, Any]:
    reasons: list[str] = []
    if len(selected) < 2:
        reasons.append('evidence_count_lt_2')
    if (plan.safety_required or plan.safety_gate_required) and not any(_matches_safety_context(chunk, context) for chunk in selected):
        reasons.append('missing_safety_aligned_evidence')
    if profile == 'prediction_plus_rag' and context.failure_modes and not any(_matches_failure_mode(chunk, context) for chunk in selected):
        reasons.append('missing_failure_mode_evidence')
    if selected:
        first = selected[0]
        title_terms = _context_terms(AgentRequest(question=' '.join(context.document_search_terms)), context)
        title_blob = _chunk_title_blob(first)
        if not any(term in title_blob for term in title_terms):
            reasons.append('top_evidence_low_title_directness')
    for chunk in selected:
        if not chunk.text.strip():
            reasons.append('selected_evidence_missing_text')
        if not (chunk.doc_id or chunk.document_title or chunk.title):
            reasons.append('selected_evidence_missing_doc_identity')
        scope = (chunk.retrieval_scope or '').strip().lower()
        if scope == 'emergency_only' and profile != 'rag_only_safety':
            reasons.append('emergency_only_evidence_selected_without_emergency_profile')
    return {
        'called': False,
        'reason': 'deterministic_confident' if not reasons else 'ambiguous_top_candidates',
        'would_call_on': list(dict.fromkeys(reasons)),
    }


def _evidence_artifact(*, state: RagEvidenceState, grade: EvidenceGrade, selected: list[RagChunk], citations: list[dict[str, Any]]) -> EvidenceArtifact:
    plan = AgentPlan.model_validate(state['plan'])
    context = ManufacturingContext.model_validate(state['manufacturing_context'])
    query_specs = state.get('query_specs') or []
    required_gates = [gate.gate_id for gate in context.safety_gates] if (plan.safety_required or plan.safety_gate_required) else []
    covered_gates = {
        gate.gate_id
        for gate in context.safety_gates
        if any(_matches_gate_context(chunk, gate) for chunk in selected)
    }
    missing_gate_evidence = [gate_id for gate_id in required_gates if gate_id not in covered_gates]
    citation_ids = [str(item.get('doc_id') or item.get('chunk_id') or item.get('label') or '') for item in citations if item]
    diagnostics = dict(state.get('trace') or {})
    diagnostics.update({
        'evidence_grade': grade.model_dump(),
        'citation_coverage_ok': len(citations) == len(selected),
    })
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


def _matches_gate_context(chunk: RagChunk, gate) -> bool:
    if chunk.safety_gate == getattr(gate, 'gate_id', None):
        return True
    metadata = ' '.join([
        chunk.safety_gate or '',
        chunk.doc_type or '',
        chunk.document_title or '',
        chunk.title or '',
    ]).lower().replace('_', ' ')
    terms: set[str] = set()
    terms.update(_tokens(getattr(gate, 'gate_id', '')))
    terms.update(_tokens(getattr(gate, 'name_ko', '')))
    terms.update(_tokens(getattr(gate, 'description_ko', '')))
    for term in getattr(gate, 'document_search_terms', None) or []:
        terms.update(_tokens(term))
    return bool(terms and any(term in metadata for term in terms))


def _tokens(text: str) -> set[str]:
    stopwords = {
        'ai4i', 'type', 'air', 'temperature', 'process', 'rotational', 'speed',
        '어떤', '있어', '대한', '확인', '점검', '절차', '해야', '하는지', '높고', '큰데',
        '데이터', '가능성', '예측', '교체', '전에', '항목', '알려줘',
        'check', 'high', 'low', 'normal', 'min', 'max', 'data', 'model', 'false', 'true',
    }
    terms: set[str] = set()
    for token in re.findall(r'[가-힣A-Za-z0-9_+#.-]+', text or ''):
        lowered = token.lower().replace('_', ' ')
        for part in lowered.split():
            stripped = part.rstrip('이가은는을를에의와과도')
            if len(stripped) >= 2 and stripped not in stopwords and not stripped.replace('.', '', 1).isdigit():
                terms.add(stripped)
                if re.fullmatch(r'[가-힣]{3,}', stripped):
                    terms.add(stripped[:2])
    return terms


def _contains_any(value: object, expected: set[str]) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        values = {item.strip() for item in value.replace(';', ',').split(',') if item.strip()}
    elif isinstance(value, (list, tuple, set)):
        values = {str(item).strip() for item in value if str(item).strip()}
    else:
        values = {str(value).strip()}
    return bool(values.intersection(expected))


def _backend_label(backends: list[str], chunks: list[RagChunk]) -> str:
    if any(backend == 'error' for backend in backends):
        return 'error'
    if any(backend == 'chroma' for backend in backends):
        return 'chroma'
    if any(backend == 'jsonl_dev' for backend in backends):
        return 'jsonl_dev'
    return 'empty' if not chunks else 'unknown'


def _unique(values: list[str | None]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
