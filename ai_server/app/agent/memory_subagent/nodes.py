from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent.context import AnswerMemoryWriter
from app.agent.artifacts import ContextArtifact, EvidenceArtifact, PlanningArtifact, PredictionArtifact, RequestArtifact, ResponseArtifact, RuntimeArtifact, SafetyArtifact
from app.config import LLM_PROVIDER
from app.schemas.agent import AgentPlan, AgentRequest, AgentResponse
from app.schemas.prediction import PredictionResponse
from app.schemas.rag import RagChunk
from app.agent.trace import to_agent_trace_steps
from app.services.domain_service import DomainKnowledgeService
from app.services.memory_service import MemoryService

from .state import MemoryState


@dataclass(frozen=True)
class MemoryDeps:
    answer_memory_writer: AnswerMemoryWriter
    memory_service: MemoryService
    domain_service: DomainKnowledgeService


def extract_memory_candidates(state: MemoryState, deps: MemoryDeps) -> MemoryState:
    response = _agent_response_from_artifacts(state, deps)
    planning = PlanningArtifact.model_validate(state.get('planning') or {'selected_path': 'unknown', 'answer_type': 'unknown'})
    context = ContextArtifact.model_validate(state.get('context') or {})
    safety = SafetyArtifact.model_validate(state.get('safety') or {})
    memory_state = {
        'request': state.get('request'),
        'selected_path': planning.selected_path,
        'intent_gateway': {
            'selected_path': planning.selected_path,
            'answer_type': planning.answer_type,
            'turn_type': planning.intent or planning.answer_type,
            'reason': planning.reasoning_summary,
        },
        'structured_answer_payload': safety.structured_payload,
        'context_resolution': context.context_resolution,
    }
    answer_memory = deps.answer_memory_writer.build(state=memory_state, response=response)
    return {'answer_memory': answer_memory.model_dump()}


def update_focus(state: MemoryState, deps: MemoryDeps) -> MemoryState:
    request = _agent_request_from_artifacts(state)
    response = _agent_response_from_artifacts(state, deps)
    answer_memory = dict(state.get('answer_memory') or {})
    context = ContextArtifact.model_validate(state.get('context') or {})
    route_history = list(context.recent_turn_routes)
    route_history.append({
        'selected_path': answer_memory.get('selected_path'),
        'answer_type': answer_memory.get('answer_type'),
        'summary': answer_memory.get('short_summary'),
    })
    turns = list(context.recent_turns)
    turns.append({'role': 'user', 'content': request.question})
    turns.append({'role': 'assistant', 'content': response.answer or ''})
    return {
        'recent_turn_routes': route_history[-10:],
        'recent_turns': turns[-10:],
        'session_last_process_data': context.turn_process_data,
    }


def write_answer_memory(state: MemoryState, deps: MemoryDeps) -> MemoryState:
    request = _agent_request_from_artifacts(state)
    response = _agent_response_from_artifacts(state, deps)
    warnings = list(state.get('warnings') or [])
    diagnostics = dict(state.get('diagnostics') or {})
    try:
        diagnostics['memory_update'] = deps.memory_service.update_from_run(
            user_id=state.get('user_id') or request.user_id or '',
            request=request,
            response=response,
        )
    except Exception as exc:
        diagnostics['memory_update_error'] = f'{type(exc).__name__}: {exc}'
        warnings.append('Memory update failed; response generation completed without persistent memory update.')
    return {'warnings': list(dict.fromkeys(warnings)), 'diagnostics': diagnostics}


def emit_memory_output(state: MemoryState, deps: MemoryDeps) -> MemoryState:
    answer_memory = dict(state.get('answer_memory') or {})
    trace = {
        'focus': answer_memory.get('focus'),
        'recommended_action_count': len(answer_memory.get('recommended_actions') or []),
        'memory_warning_count': len(state.get('warnings') or []),
    }
    output = {
        'last_answer_memory': answer_memory,
        'recent_turn_routes': state.get('recent_turn_routes') or [],
        'recent_turns': state.get('recent_turns') or [],
        'session_last_process_data': state.get('session_last_process_data'),
        'warnings': state.get('warnings') or [],
        'diagnostics': state.get('diagnostics') or {},
        'trace': trace,
    }
    return {'trace': trace, 'output': output}


def _agent_request_from_artifacts(state: MemoryState) -> AgentRequest:
    request = RequestArtifact.model_validate(state['request'])
    context = ContextArtifact.model_validate(state.get('context') or {})
    return AgentRequest.model_validate({
        'user_id': request.user_id,
        'session_id': request.session_id,
        'question': request.normalized_message or request.question,
        'process_data': request.process_data,
        'inspection_notes': request.inspection_notes,
        'top_k': request.top_k or 5,
        'mode': request.mode,
        'llm_model': request.llm_model,
        'user_context': context.user_context,
    })


def _agent_response_from_artifacts(state: MemoryState, deps: MemoryDeps) -> AgentResponse:
    request_artifact = RequestArtifact.model_validate(state['request'])
    request = _agent_request_from_artifacts(state)
    response = ResponseArtifact.model_validate(state['response'])
    context = ContextArtifact.model_validate(state.get('context') or {})
    planning = PlanningArtifact.model_validate(state.get('planning') or {'selected_path': 'unknown', 'answer_type': 'unknown'})
    prediction_artifact = PredictionArtifact.model_validate(state.get('prediction') or {})
    evidence = EvidenceArtifact.model_validate(state.get('evidence') or {})
    safety = SafetyArtifact.model_validate(state.get('safety') or {})
    runtime = RuntimeArtifact.model_validate(state.get('runtime') or {})
    prediction = PredictionResponse.model_validate(prediction_artifact.result) if prediction_artifact.result else None
    manufacturing_context = deps.domain_service.build_context(request, prediction, doc_count=len(evidence.documents))
    return AgentResponse(
        run_id=state['run_id'],
        user_id=request_artifact.user_id,
        session_id=request_artifact.session_id,
        route=response.route,
        answer=response.answer,
        prediction=prediction,
        manufacturing_context=manufacturing_context,
        retrieved_documents=[RagChunk.model_validate(item) for item in evidence.documents],
        safety_guidance=safety.public_guidance,
        report=response.report,
        citations=response.public_citations or evidence.citations,
        warnings=response.warnings,
        trace=to_agent_trace_steps(runtime.trace),
        saved=response.saved,
        plan=AgentPlan.model_validate(planning.agent_plan) if planning.agent_plan else None,
        llm_used=response.llm_used or bool(runtime.usage_records),
        llm_provider=LLM_PROVIDER,
        llm_model=request_artifact.llm_model,
        context_used={
            'user_id': request_artifact.user_id,
            'session_id': request_artifact.session_id,
            'ai4i_feature_status': context.ai4i_feature_status,
            'context_resolution': context.context_resolution,
        },
        prediction_called=prediction_artifact.called,
        prediction_skip_reason=prediction_artifact.skip_reason if prediction_artifact.skip_reason in {'missing_ai4i_features', 'ambiguous_ai4i_features', 'invalid_ai4i_features'} else None,
        missing_features=[] if prediction_artifact.called else prediction_artifact.missing_features,
        ambiguous_features=[] if prediction_artifact.called else prediction_artifact.ambiguous_features,
        parsed_ai4i_features=prediction_artifact.parsed_features,
    )
