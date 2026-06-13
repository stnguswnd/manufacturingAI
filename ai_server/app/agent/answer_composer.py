from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from app.agent.artifacts import AnswerDraft, ContextArtifact, EvidenceArtifact, PlanningArtifact, PredictionArtifact, RequestArtifact, SafetyArtifact, ValidationReport
from app.agent.heavy.structured_answer_payload_builder import StructuredAnswerPayloadBuilder
from app.schemas.agent import AgentPlan, AgentRequest, LLMUsageRecord
from app.schemas.domain import ManufacturingContext
from app.schemas.prediction import PredictionResponse
from app.schemas.rag import RagChunk
from app.services.llm_service import ANSWER_SCHEMA, LLMService


@dataclass
class AnswerComposeResult:
    draft: AnswerDraft | None
    action_titles: list[str] = field(default_factory=list)
    safety_guidance: str | None = None
    warnings: list[str] = field(default_factory=list)
    llm_error: str | None = None
    llm_used: bool = False


class AnswerComposer:
    """Builds the bounded answer draft from already-produced artifacts."""

    def __init__(self, llm_service: LLMService, payload_builder: StructuredAnswerPayloadBuilder):
        self.llm_service = llm_service
        self.payload_builder = payload_builder

    def compose_artifact(
        self,
        *,
        request_artifact: RequestArtifact,
        context_artifact: ContextArtifact,
        planning_artifact: PlanningArtifact,
        prediction_artifact: PredictionArtifact | None,
        evidence_artifact: EvidenceArtifact | None,
        safety_artifact: SafetyArtifact | None,
        manufacturing_context: ManufacturingContext,
        action_titles: list[str],
        audit_feedback: list[str] | None = None,
        usage_callback: Callable[[LLMUsageRecord], None] | None = None,
        system_prompt: str,
    ) -> AnswerComposeResult:
        request = _request_from_artifact(request_artifact, context_artifact)
        plan = _plan_from_artifact(planning_artifact)
        prediction = _prediction_from_artifact(prediction_artifact)
        contexts = _rag_chunks_from_artifact(evidence_artifact)
        result = self.compose(
            request=request,
            plan=plan,
            prediction=prediction,
            manufacturing_context=manufacturing_context,
            contexts=contexts,
            action_titles=action_titles,
            safety_guidance=safety_artifact.public_guidance if safety_artifact else None,
            safety_artifact=safety_artifact,
            audit_feedback=audit_feedback,
            usage_callback=usage_callback,
            system_prompt=system_prompt,
        )
        if result.draft:
            result.draft = result.draft.model_copy(update={
                'route': 'answer_compose',
                'citations': list(evidence_artifact.citations if evidence_artifact else []),
                'llm_used': result.llm_used,
                'llm_error': result.llm_error,
                'recommended_actions': list(result.action_titles),
                'safety_guidance': result.safety_guidance,
            })
        return result

    def compose(
        self,
        *,
        request: AgentRequest,
        plan: AgentPlan,
        prediction: PredictionResponse | None,
        manufacturing_context: ManufacturingContext,
        contexts: list[RagChunk],
        action_titles: list[str],
        safety_guidance: str | None,
        safety_artifact: SafetyArtifact | None = None,
        audit_feedback: list[str] | None = None,
        usage_callback: Callable[[LLMUsageRecord], None] | None = None,
        system_prompt: str,
    ) -> AnswerComposeResult:
        payload = self.payload_builder.build(
            request=request,
            plan=plan,
            prediction=prediction,
            manufacturing_context=manufacturing_context,
            contexts=contexts,
            action_titles=action_titles,
            safety_guidance=safety_guidance,
            audit_feedback=audit_feedback,
        )
        if safety_artifact:
            payload['safety_contract'] = safety_artifact.model_dump()
        result = self.llm_service.generate_json(
            schema_name='manufacturing_domain_agent_response',
            schema=ANSWER_SCHEMA,
            system_prompt=system_prompt,
            payload=payload,
            model=request.llm_model,
            operation='answer_generation',
            usage_callback=usage_callback,
        )
        if not result:
            return AnswerComposeResult(
                draft=None,
                action_titles=action_titles,
                safety_guidance=safety_guidance,
                llm_error=self.llm_service.last_error,
                llm_used=False,
            )
        answer = str(result.get('answer') or '').strip()
        if not answer:
            return AnswerComposeResult(
                draft=None,
                action_titles=action_titles,
                safety_guidance=safety_guidance,
                llm_error='LLM returned an empty answer',
                llm_used=False,
            )
        llm_actions = result.get('recommended_actions') or []
        merged_titles = action_titles
        if isinstance(llm_actions, list):
            merged_titles = list(dict.fromkeys([str(item) for item in llm_actions if str(item).strip()] + action_titles))
        llm_warnings = result.get('warnings') or []
        warnings = [str(item) for item in llm_warnings if str(item).strip()] if isinstance(llm_warnings, list) else []
        return AnswerComposeResult(
            draft=AnswerDraft(text=answer, route='answer_generation', warnings=warnings),
            action_titles=merged_titles,
            safety_guidance=result.get('safety_guidance') or safety_guidance,
            warnings=warnings,
            llm_error=None,
            llm_used=True,
        )


class AnswerRewriter:
    """Rewrites only the answer text when reviewers determine artifacts are sufficient."""

    def __init__(self, composer: AnswerComposer):
        self.composer = composer

    def rewrite(
        self,
        *,
        request: AgentRequest,
        plan: AgentPlan,
        prediction: PredictionResponse | None,
        manufacturing_context: ManufacturingContext,
        contexts: list[RagChunk],
        action_titles: list[str],
        safety_guidance: str | None,
        safety_artifact: SafetyArtifact | None,
        validation_report: ValidationReport,
        usage_callback: Callable[[LLMUsageRecord], None] | None,
        system_prompt: str,
    ) -> AnswerComposeResult:
        feedback = [f'{failure.source}:{failure.code}: {failure.message}' for failure in validation_report.failures]
        return self.composer.compose(
            request=request,
            plan=plan,
            prediction=prediction,
            manufacturing_context=manufacturing_context,
            contexts=contexts,
            action_titles=action_titles,
            safety_guidance=safety_guidance,
            safety_artifact=safety_artifact,
            audit_feedback=feedback,
            usage_callback=usage_callback,
            system_prompt=system_prompt,
        )

    def rewrite_artifact(
        self,
        *,
        request_artifact: RequestArtifact,
        context_artifact: ContextArtifact,
        planning_artifact: PlanningArtifact,
        prediction_artifact: PredictionArtifact | None,
        evidence_artifact: EvidenceArtifact | None,
        safety_artifact: SafetyArtifact | None,
        manufacturing_context: ManufacturingContext,
        action_titles: list[str],
        validation_report: ValidationReport,
        usage_callback: Callable[[LLMUsageRecord], None] | None,
        system_prompt: str,
    ) -> AnswerComposeResult:
        feedback = [f'{failure.source}:{failure.code}: {failure.message}' for failure in validation_report.failures]
        return self.composer.compose_artifact(
            request_artifact=request_artifact,
            context_artifact=context_artifact,
            planning_artifact=planning_artifact,
            prediction_artifact=prediction_artifact,
            evidence_artifact=evidence_artifact,
            safety_artifact=safety_artifact,
            manufacturing_context=manufacturing_context,
            action_titles=action_titles,
            audit_feedback=feedback,
            usage_callback=usage_callback,
            system_prompt=system_prompt,
        )


def _request_from_artifact(request: RequestArtifact, context: ContextArtifact) -> AgentRequest:
    return AgentRequest.model_validate({
        'user_id': request.user_id,
        'session_id': request.session_id,
        'question': request.question,
        'process_data': request.process_data,
        'inspection_notes': request.inspection_notes,
        'top_k': request.top_k or 5,
        'mode': request.mode,
        'llm_model': request.llm_model,
        'user_context': context.user_context,
    })


def _plan_from_artifact(planning: PlanningArtifact) -> AgentPlan:
    if planning.agent_plan:
        return AgentPlan.model_validate(planning.agent_plan)
    return AgentPlan(
        intent=planning.intent or 'general',
        prediction_required=planning.needs_prediction,
        rag_required=planning.needs_rag,
        safety_required=planning.needs_safety,
        safety_gate_required=planning.needs_safety,
        required_nodes=list(planning.route),
        rationale=planning.reasoning_summary or '',
    )


def _prediction_from_artifact(prediction: PredictionArtifact | None) -> PredictionResponse | None:
    if not prediction or not prediction.result:
        return None
    return PredictionResponse.model_validate(prediction.result)


def _rag_chunks_from_artifact(evidence: EvidenceArtifact | None) -> list[RagChunk]:
    if not evidence:
        return []
    return [RagChunk.model_validate(item) for item in evidence.documents]
