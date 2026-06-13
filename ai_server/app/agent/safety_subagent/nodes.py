from __future__ import annotations

from dataclasses import dataclass

from app.agent.artifacts import SafetyArtifact
from app.agent.heavy import RecommendationBuilder, SafetyGateBuilder
from app.schemas.agent import AgentRequest
from app.schemas.domain import ManufacturingContext
from app.schemas.prediction import PredictionResponse
from app.schemas.rag import RagChunk
from app.services.domain_service import DomainKnowledgeService

from .state import SafetyState


@dataclass(frozen=True)
class SafetyDeps:
    domain_service: DomainKnowledgeService
    recommendation_builder: RecommendationBuilder
    safety_gate_builder: SafetyGateBuilder


def build_safety_context(state: SafetyState, deps: SafetyDeps) -> SafetyState:
    request = AgentRequest.model_validate(state['request'])
    prediction = PredictionResponse.model_validate(state['prediction']) if state.get('prediction') else None
    context = ManufacturingContext.model_validate(state['manufacturing_context']) if state.get('manufacturing_context') else None
    if not context:
        doc_count = len([RagChunk.model_validate(item) for item in state.get('retrieved_documents') or []])
        context = deps.domain_service.build_context(request, prediction, doc_count=doc_count)
    return {'manufacturing_context': context.model_dump()}


def apply_safety_policy(state: SafetyState, deps: SafetyDeps) -> SafetyState:
    context = ManufacturingContext.model_validate(state['manufacturing_context'])
    prediction = PredictionResponse.model_validate(state['prediction']) if state.get('prediction') else None
    actions = deps.recommendation_builder.to_action_dicts(deps.recommendation_builder.collect_action_phrases(prediction, context))
    payload = dict(state.get('structured_answer_payload') or {})
    payload['recommended_actions'] = actions
    safety_artifact = _build_safety_artifact(context)
    return {
        'structured_answer_payload': payload,
        'safety_artifact': safety_artifact.model_dump(),
        'safety_guidance': deps.safety_gate_builder.safety_guidance(context) if context.safety_gates else None,
        'safety_warnings': deps.safety_gate_builder.warnings(context),
    }


def validate_safety_output(state: SafetyState, deps: SafetyDeps) -> SafetyState:
    warnings = list(dict.fromkeys(state.get('safety_warnings') or []))
    context = ManufacturingContext.model_validate(state['manufacturing_context'])
    payload = state.get('structured_answer_payload') or {}
    if context.safety_gates and not state.get('safety_guidance'):
        warnings.append('Safety guidance is missing despite required safety gates.')
    if context.action_plan and not payload.get('recommended_actions'):
        warnings.append('Recommended actions are missing despite a non-empty action plan.')
    return {'safety_warnings': warnings}


def emit_safety_output(state: SafetyState, deps: SafetyDeps) -> SafetyState:
    context = ManufacturingContext.model_validate(state['manufacturing_context'])
    actions = (state.get('structured_answer_payload') or {}).get('recommended_actions') or []
    retrieved_gate_ids = _retrieved_safety_gate_ids(state.get('retrieved_documents') or [])
    trace = {
        'safety_gate_count': len(context.safety_gates),
        'safety_gate_ids': [gate.gate_id for gate in context.safety_gates],
        'high_severity_gate_count': len([gate for gate in context.safety_gates if gate.severity in {'high', 'critical'}]),
        'has_safety_guidance': bool(state.get('safety_guidance')),
        'recommended_action_count': len(actions),
        'action_requires_loto_count': len([action for action in context.action_plan if bool(action.requires_loto)]),
        'retrieved_safety_gate_ids': retrieved_gate_ids,
        'warning_count': len(state.get('safety_warnings') or []),
    }
    artifact = SafetyArtifact.model_validate(state.get('safety_artifact') or {})
    artifact = artifact.model_copy(update={
        'public_guidance': state.get('safety_guidance'),
        'structured_payload': state.get('structured_answer_payload') or {},
        'warnings': list(dict.fromkeys(artifact.warnings + list(state.get('safety_warnings') or []))),
    })
    output = {
        'manufacturing_context': context.model_dump(),
        'structured_answer_payload': state.get('structured_answer_payload') or {},
        'safety_artifact': artifact.model_dump(mode='json'),
        'safety_guidance': state.get('safety_guidance'),
        'safety_warnings': state.get('safety_warnings') or [],
        'trace': trace,
    }
    return {'trace': trace, 'output': output}


def _build_safety_artifact(context: ManufacturingContext) -> SafetyArtifact:
    constraints: list[str] = []
    forbidden: list[str] = []
    checks: list[str] = []
    for gate in context.safety_gates:
        constraints.append(f'{gate.name_ko}: {gate.description_ko}')
        checks.extend(gate.required_checks)
        forbidden.extend(gate.forbidden_agent_actions)
    return SafetyArtifact(
        required_gates=[gate.gate_id for gate in context.safety_gates],
        constraints=list(dict.fromkeys(constraints)),
        forbidden_actions=list(dict.fromkeys(forbidden)),
        required_checks=list(dict.fromkeys(checks)),
        public_guidance=None,
        structured_payload={},
        warnings=list(context.audit_notes),
    )


def _retrieved_safety_gate_ids(rows: list[dict]) -> list[str]:
    ids: list[str] = []
    for row in rows:
        gate = row.get('safety_gate') if isinstance(row, dict) else None
        if gate and gate not in ids:
            ids.append(str(gate))
    return ids
