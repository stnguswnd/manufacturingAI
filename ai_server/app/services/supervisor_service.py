from __future__ import annotations

from typing import Callable

from app.config import AGENT_SUPERVISOR_LLM_REFINEMENT
from app.schemas import AgentLayer, AgentPlan, AgentRequest, LLMUsageRecord
from app.services.llm_service import LLMService, PLAN_SCHEMA

class SupervisorService:
    """Manufacturing-specific hierarchical supervisor.

    This version is not a generic keyword router. It creates a manufacturing
    execution graph that separates asset context, process conditions, failure
    modes, risk/priority, procedure retrieval, safety gates, action planning,
    report generation, and audit/persistence.
    """

    def __init__(self, llm_service: LLMService | None = None):
        self.llm_service = llm_service or LLMService()

    def plan(self, req: AgentRequest, usage_callback: Callable[[LLMUsageRecord], None] | None = None) -> AgentPlan:
        base = self._deterministic_plan(req)
        if self._resolved_turn(req):
            return base
        if not AGENT_SUPERVISOR_LLM_REFINEMENT:
            return base
        refined = self._llm_refine(req, base, usage_callback=usage_callback)
        return refined or base

    def replan(self, req: AgentRequest, previous: AgentPlan, audit_findings: list[str], attempt: int) -> AgentPlan:
        """Create a bounded correction plan after an audit/validator failure.

        This does not blindly restart the whole graph. It preserves mandatory
        manufacturing stages and strengthens only the parts that commonly need a
        second pass: retrieval, safety gate coverage, action planning, and final
        explanation/report generation.
        """
        findings = [str(x) for x in audit_findings if str(x).strip()]
        finding_blob = ' '.join(findings).lower()
        prediction_required = previous.prediction_required or bool(req.process_data)
        safety_required = True if any(k in finding_blob for k in ['safety', '안전', 'gate', 'loto', '검증']) else previous.safety_required
        report_required = previous.report_required
        rag_required = True if any(k in finding_blob for k in ['rag', '문서', '근거', 'citation', 'retrieval']) else previous.rag_required
        safety_gate_required = True if safety_required or prediction_required else previous.safety_gate_required
        action_plan_required = True
        process_condition_required = prediction_required and bool(req.process_data)
        failure_mode_required = prediction_required
        document_scope = self._document_scope(prediction_required, safety_required, report_required, rag_required)
        correction_terms = [
            'audit retry',
            'required safety gate checks',
            'LOTO energy isolation',
            'machine guarding rotating parts',
            'qualified maintenance personnel',
            'evidence citation',
        ]
        if any(k in finding_blob for k in ['rag', '문서', '근거', 'retrieval']):
            correction_terms.extend(['operator manual', 'safety procedure', 'preventive maintenance', 'troubleshooting'])
        if any(k in finding_blob for k in ['loto', '안전', 'safety', 'gate']):
            correction_terms.extend(['lockout tagout', '전원 차단', '무전압 확인', '방호장치', '비상정지'])
        rag_query = ' '.join(
            p
            for p in [
                previous.rag_query,
                req.question,
                req.inspection_notes or '',
                ' '.join(document_scope),
                ' '.join(correction_terms),
            ]
            if p
        ).strip()
        layers = self._layers(
            asset_context_required=True,
            process_condition_required=process_condition_required,
            failure_mode_required=failure_mode_required,
            risk_priority_required=True,
            rag_required=rag_required,
            safety_gate_required=safety_gate_required,
            action_plan_required=action_plan_required,
            report_required=report_required,
        )
        required_nodes = [node for layer in layers for node in layer.nodes]
        return AgentPlan(
            intent='hybrid' if previous.intent != 'general' else previous.intent,
            confidence=max(previous.confidence - 0.05, 0.5),
            prediction_required=prediction_required,
            rag_required=rag_required,
            safety_required=safety_required,
            report_required=report_required,
            domain_context_required=True,
            asset_context_required=True,
            process_condition_required=process_condition_required,
            failure_mode_required=failure_mode_required,
            risk_priority_required=True,
            safety_gate_required=safety_gate_required,
            action_plan_required=action_plan_required,
            required_nodes=required_nodes,
            layers=layers,
            rag_query=rag_query,
            rag_filters=previous.rag_filters,
            document_scope=document_scope,
            rationale=f'재계획 attempt={attempt}: audit/validator 실패 사유를 반영했습니다. findings=' + '; '.join(findings[:5]),
            supervisor_source=previous.supervisor_source,
        )

    def _deterministic_plan(self, req: AgentRequest) -> AgentPlan:
        from app.agent.heavy.diagnostic_planner import DiagnosticFallbackPolicy

        diagnostic = DiagnosticFallbackPolicy().plan(req)
        intent = self._intent(
            diagnostic.requires_prediction,
            diagnostic.requires_safety,
            diagnostic.requires_report,
            diagnostic.requires_knowledge,
            diagnostic.requires_rag,
        )
        layers = self._layers(
            asset_context_required=diagnostic.requires_asset_context,
            process_condition_required=diagnostic.requires_process_condition,
            failure_mode_required=diagnostic.requires_failure_mode,
            risk_priority_required=True,
            rag_required=diagnostic.requires_rag,
            safety_gate_required=diagnostic.requires_safety_gate,
            action_plan_required=diagnostic.requires_action_plan,
            report_required=diagnostic.requires_report,
        )
        required_nodes = [node for layer in layers for node in layer.nodes]
        return AgentPlan(
            intent=intent,
            confidence=diagnostic.confidence,
            prediction_required=diagnostic.requires_prediction,
            rag_required=diagnostic.requires_rag,
            safety_required=diagnostic.requires_safety,
            report_required=diagnostic.requires_report,
            domain_context_required=True,
            asset_context_required=diagnostic.requires_asset_context,
            process_condition_required=diagnostic.requires_process_condition,
            failure_mode_required=diagnostic.requires_failure_mode,
            risk_priority_required=True,
            safety_gate_required=diagnostic.requires_safety_gate,
            action_plan_required=diagnostic.requires_action_plan,
            required_nodes=required_nodes,
            layers=layers,
            rag_query=diagnostic.rag_query,
            rag_filters=None,
            document_scope=diagnostic.document_scope,
            rationale=diagnostic.reason,
            supervisor_source='deterministic',
        )

    @staticmethod
    def _resolved_turn(req: AgentRequest) -> dict:
        current_turn = (req.user_context or {}).get('current_turn') or {}
        if not isinstance(current_turn, dict):
            return {}
        reason = current_turn.get('reason')
        if current_turn.get('resolved') or current_turn.get('clarification_question') or reason not in {None, '', 'not_followup'}:
            return current_turn
        return {}

    def _llm_refine(self, req: AgentRequest, base: AgentPlan, usage_callback: Callable[[LLMUsageRecord], None] | None = None) -> AgentPlan | None:
        payload = {
            'question': req.question,
            'has_process_data': req.process_data is not None,
            'has_inspection_notes': bool(req.inspection_notes),
            'generate_report': req.generate_report,
            'requested_mode': req.mode,
            'base_plan': base.model_dump(),
            'manufacturing_policy': {
                'must_keep_prediction_if_process_data': True,
                'must_keep_safety_if_maintenance_or_safety_terms': True,
                'must_include_safety_gate_for_physical_maintenance': True,
                'must_not_remove_domain_context': True,
            },
        }
        system = (
            '당신은 제조 AI 시스템의 Manufacturing Supervisor입니다. '
            '요청을 prediction, knowledge_qa, safety_ops, documentation, hybrid 중 하나로 분류하고, '
            '제조 업무 단계(Asset, Process, Failure Mode, Risk, Retrieval, Safety Gate, Action, Report)를 고려해 JSON으로 반환하세요. '
            '단 process_data가 있으면 prediction_required=true를 유지하고, 정비/안전 키워드가 있으면 safety_gate_required=true를 유지해야 합니다.'
        )
        data = self.llm_service.generate_json(
            schema_name='manufacturing_route_plan',
            schema=PLAN_SCHEMA,
            system_prompt=system,
            payload=payload,
            model=req.llm_model,
            operation='supervisor_plan_refine',
            usage_callback=usage_callback,
        )
        if not data:
            return None
        prediction_required = base.prediction_required or bool(data.get('prediction_required'))
        safety_required = base.safety_required or bool(data.get('safety_required'))
        report_required = base.report_required or bool(data.get('report_required'))
        rag_required = base.rag_required or bool(data.get('rag_required')) or safety_required or report_required
        asset_context_required = True
        process_condition_required = prediction_required and bool(req.process_data)
        failure_mode_required = prediction_required
        safety_gate_required = base.safety_gate_required or safety_required or prediction_required
        action_plan_required = base.action_plan_required or prediction_required or safety_required or report_required or rag_required
        document_scope = self._document_scope(prediction_required, safety_required, report_required, rag_required)
        layers = self._layers(asset_context_required, process_condition_required, failure_mode_required, True, rag_required, safety_gate_required, action_plan_required, report_required)
        required_nodes = [node for layer in layers for node in layer.nodes]
        intent = data.get('intent') if data.get('intent') in {'prediction','knowledge_qa','safety_ops','documentation','hybrid','general'} else base.intent
        if sum([prediction_required, safety_required, report_required, intent == 'knowledge_qa']) >= 2:
            intent = 'hybrid'
        return AgentPlan(
            intent=intent,
            confidence=float(data.get('confidence') or 0.88),
            prediction_required=prediction_required,
            rag_required=rag_required,
            safety_required=safety_required,
            report_required=report_required,
            domain_context_required=True,
            asset_context_required=asset_context_required,
            process_condition_required=process_condition_required,
            failure_mode_required=failure_mode_required,
            risk_priority_required=True,
            safety_gate_required=safety_gate_required,
            action_plan_required=action_plan_required,
            required_nodes=required_nodes,
            layers=layers,
            rag_query=(data.get('rag_query') or base.rag_query or self._build_query(req, document_scope)).strip(),
            rag_filters=base.rag_filters,
            document_scope=document_scope,
            rationale=data.get('rationale') or base.rationale,
            supervisor_source='llm_refined',
        )

    @staticmethod
    def _intent(prediction: bool, safety: bool, report: bool, knowledge: bool, rag: bool) -> str:
        if sum([prediction, safety, report, knowledge]) >= 2:
            return 'hybrid'
        if prediction:
            return 'prediction'
        if safety:
            return 'safety_ops'
        if report:
            return 'documentation'
        if knowledge or rag:
            return 'knowledge_qa'
        return 'general'

    @staticmethod
    def _document_scope(prediction: bool, safety: bool, report: bool, knowledge: bool) -> list[str]:
        scope: list[str] = []
        if prediction:
            scope += ['troubleshooting', 'preventive_maintenance']
        if safety:
            scope += ['safety_standard', 'safety_procedure']
        if report:
            scope += ['technical_document_guide', 'preventive_maintenance']
        if knowledge:
            scope += ['operator_manual', 'troubleshooting', 'equipment_taxonomy']
        return list(dict.fromkeys(scope)) or ['operator_manual', 'troubleshooting', 'safety_standard']

    def _layers(self, asset_context_required: bool, process_condition_required: bool, failure_mode_required: bool, risk_priority_required: bool, rag_required: bool, safety_gate_required: bool, action_plan_required: bool, report_required: bool) -> list[AgentLayer]:
        layers: list[AgentLayer] = [
            AgentLayer(name='0. Input Layer', nodes=['Input Normalizer'], purpose='질문, 공정 데이터, 점검 메모를 표준 상태로 정리'),
            AgentLayer(name='1. Manufacturing Supervisor Layer', nodes=['Manufacturing Intent Classifier', 'Manufacturing Route Planner'], purpose='제조 업무 관점으로 의도와 실행 순서를 결정'),
        ]
        if asset_context_required:
            layers.append(AgentLayer(name='2. Asset Context Layer', nodes=['Asset Context Agent'], purpose='설비, 하위 시스템, 부품, hazard를 식별'))
        if process_condition_required:
            layers.append(AgentLayer(name='3. Process Condition Layer', nodes=['Process Condition Agent'], purpose='온도, 회전수, 토크, 공구 마모 등 운전 조건 분석'))
        if failure_mode_required:
            layers.append(AgentLayer(name='4. Failure Mode Layer', nodes=['Failure Mode Agent'], purpose='AI4I 고장모드와 도메인 카탈로그를 연결'))
        if risk_priority_required:
            layers.append(AgentLayer(name='5. Risk & Priority Layer', nodes=['Risk & Priority Agent'], purpose='품질/설비/안전/생산 위험도를 분리 산정'))
        if rag_required:
            layers.append(AgentLayer(name='6. Procedure Retrieval Layer', nodes=['RAG Query Builder', 'Procedure Retrieval Agent'], purpose='매뉴얼, 안전규정, 점검표, 기술문서 검색'))
        if safety_gate_required:
            layers.append(AgentLayer(name='7. Safety Gate Layer', nodes=['Safety Gate Agent'], purpose='LOTO, 방호장치, 비상정지, 전기/고온 위험 확인'))
        if action_plan_required:
            layers.append(AgentLayer(name='8. Action Planning Layer', nodes=['Action Planner Agent'], purpose='실행 가능한 점검 순서와 승인 필요 여부 구조화'))
        layers.append(AgentLayer(name='9. Reasoning Layer', nodes=['Explanation Agent'], purpose='예측, 위험도, 문서 근거, 안전 게이트를 결합해 답변 생성'))
        if report_required:
            layers.append(AgentLayer(name='10. Documentation Layer', nodes=['Report Agent'], purpose='점검/정비 보고서 초안 생성'))
        layers.append(AgentLayer(name='11. Audit & Persistence Layer', nodes=['Evaluation / Audit Agent', 'History Store'], purpose='금지 표현, 안전 게이트 준수, 실행 이력 저장'))
        return layers

    def _build_query(self, req: AgentRequest, document_scope: list[str]) -> str:
        parts = [req.question or '']
        if req.process_data:
            parts.extend(['공정 데이터', '토크', '공구 마모', '온도', '회전수', '정비 점검', 'CNC', 'Spindle', 'Tool Changer'])
        if req.inspection_notes:
            parts.append(req.inspection_notes)
        if req.generate_report:
            parts.extend(['점검 보고서', '정비 보고서', '점검 항목', '기술문서'])
        parts.extend(document_scope)
        return ' '.join(p for p in parts if p).strip() or 'manufacturing maintenance safety troubleshooting'
