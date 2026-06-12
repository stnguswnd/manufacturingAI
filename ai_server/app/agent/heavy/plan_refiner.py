from __future__ import annotations

from typing import Callable

from app.agent.heavy.diagnostic_planner import DiagnosticPlan
from app.agent.heavy.plan_translator import DiagnosticPlanToAgentPlanTranslator
from app.schemas import AgentPlan, AgentRequest, LLMUsageRecord
from app.services.llm_service import LLMService, PLAN_SCHEMA


class PlanRefiner:
    """Optional LLM refinement for an already-structured diagnostic plan."""

    def __init__(self, llm_service: LLMService, translator: DiagnosticPlanToAgentPlanTranslator | None = None):
        self.llm_service = llm_service
        self.translator = translator or DiagnosticPlanToAgentPlanTranslator()

    def refine(
        self,
        *,
        request: AgentRequest,
        diagnostic_plan: DiagnosticPlan,
        base_plan: AgentPlan,
        usage_callback: Callable[[LLMUsageRecord], None] | None = None,
    ) -> AgentPlan | None:
        payload = {
            'question': request.question,
            'has_process_data': request.process_data is not None,
            'has_inspection_notes': bool(request.inspection_notes),
            'generate_report': request.generate_report,
            'requested_mode': request.mode,
            'diagnostic_plan': diagnostic_plan.model_dump(),
            'base_plan': base_plan.model_dump(),
            'manufacturing_policy': {
                'must_keep_prediction_if_process_data': True,
                'must_include_safety_gate_for_physical_maintenance': True,
                'must_not_remove_domain_context': True,
            },
        }
        system = (
            '당신은 제조 AI 시스템의 Manufacturing Planner입니다. '
            '이미 구조화된 diagnostic_plan을 참고해 prediction, RAG, safety, report 필요 여부를 JSON으로 조정하세요. '
            'process_data가 있으면 prediction_required=true를 유지하고, 물리 점검/정비가 있으면 safety_gate_required=true를 유지해야 합니다.'
        )
        data = self.llm_service.generate_json(
            schema_name='manufacturing_route_plan',
            schema=PLAN_SCHEMA,
            system_prompt=system,
            payload=payload,
            model=request.llm_model,
            operation='diagnostic_plan_refine',
            usage_callback=usage_callback,
        )
        if not data:
            return None
        refined_diagnostic = diagnostic_plan.model_copy(update={
            'requires_prediction': diagnostic_plan.requires_prediction or bool(data.get('prediction_required')),
            'requires_rag': diagnostic_plan.requires_rag or bool(data.get('rag_required')) or bool(data.get('safety_required')) or bool(data.get('report_required')),
            'requires_safety': diagnostic_plan.requires_safety or bool(data.get('safety_required')),
            'requires_report': diagnostic_plan.requires_report or bool(data.get('report_required')),
            'requires_asset_context': True,
            'requires_process_condition': diagnostic_plan.requires_process_condition or (bool(data.get('prediction_required')) and bool(request.process_data)),
            'requires_failure_mode': diagnostic_plan.requires_failure_mode or bool(data.get('prediction_required')),
            'requires_safety_gate': diagnostic_plan.requires_safety_gate or bool(data.get('safety_required')) or bool(data.get('prediction_required')),
            'requires_action_plan': True,
            'rag_query': (data.get('rag_query') or diagnostic_plan.rag_query).strip(),
            'confidence': float(data.get('confidence') or 0.88),
            'reason': data.get('rationale') or diagnostic_plan.reason,
            'source': 'llm_refined',
        })
        return self.translator.translate(refined_diagnostic)

