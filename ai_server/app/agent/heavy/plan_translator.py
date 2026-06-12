from __future__ import annotations

from app.schemas import AgentLayer, AgentPlan

from app.agent.heavy.diagnostic_planner import DiagnosticPlan


class DiagnosticPlanToAgentPlanTranslator:
    """Translates a diagnostic contract into the public AgentPlan schema."""

    def translate(self, diagnostic: DiagnosticPlan) -> AgentPlan:
        intent = self.intent_label(diagnostic)
        layers = self._layer_list(diagnostic)
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
            supervisor_source=diagnostic.source,
        )

    @staticmethod
    def intent_label(diagnostic: DiagnosticPlan) -> str:
        active = [
            diagnostic.requires_prediction,
            diagnostic.requires_safety,
            diagnostic.requires_report,
            diagnostic.requires_knowledge,
        ]
        if sum(bool(item) for item in active) >= 2:
            return 'hybrid'
        if diagnostic.requires_prediction:
            return 'prediction'
        if diagnostic.requires_safety:
            return 'safety_ops'
        if diagnostic.requires_report:
            return 'documentation'
        if diagnostic.requires_knowledge or diagnostic.requires_rag:
            return 'knowledge_qa'
        return 'general'

    @staticmethod
    def _layer_list(diagnostic: DiagnosticPlan) -> list[AgentLayer]:
        layers: list[AgentLayer] = [
            AgentLayer(name='0. Input Layer', nodes=['Input Normalizer'], purpose='질문, 공정 데이터, 점검 메모를 표준 상태로 정리'),
            AgentLayer(name='1. Manufacturing Supervisor Layer', nodes=['Manufacturing Intent Classifier', 'Manufacturing Route Planner'], purpose='제조 업무 관점으로 의도와 실행 순서를 결정'),
        ]
        if diagnostic.requires_asset_context:
            layers.append(AgentLayer(name='2. Asset Context Layer', nodes=['Asset Context Agent'], purpose='설비, 하위 시스템, 부품, hazard를 식별'))
        if diagnostic.requires_process_condition:
            layers.append(AgentLayer(name='3. Process Condition Layer', nodes=['Process Condition Agent'], purpose='온도, 회전수, 토크, 공구 마모 등 운전 조건 분석'))
        if diagnostic.requires_failure_mode:
            layers.append(AgentLayer(name='4. Failure Mode Layer', nodes=['Failure Mode Agent'], purpose='AI4I 고장모드와 도메인 카탈로그를 연결'))
        layers.append(AgentLayer(name='5. Risk & Priority Layer', nodes=['Risk & Priority Agent'], purpose='품질/설비/안전/생산 위험도를 분리 산정'))
        if diagnostic.requires_rag:
            layers.append(AgentLayer(name='6. Evidence Retrieval Layer', nodes=['RAG Query Planner', 'Retriever', 'Evidence Filter', 'Evidence Grader', 'Citation Builder'], purpose='문서 근거 검색, 필터링, 평가, citation 생성'))
        if diagnostic.requires_safety_gate:
            layers.append(AgentLayer(name='7. Safety Gate Layer', nodes=['Safety Gate Agent'], purpose='LOTO, 방호장치, 비상정지, 전기/고온 위험 확인'))
        if diagnostic.requires_action_plan:
            layers.append(AgentLayer(name='8. Action Planning Layer', nodes=['Action Planner Agent'], purpose='실행 가능한 점검 순서와 승인 필요 여부 구조화'))
        layers.append(AgentLayer(name='9. Reasoning Layer', nodes=['Explanation Agent'], purpose='예측, 위험도, 문서 근거, 안전 게이트를 결합해 답변 생성'))
        if diagnostic.requires_report:
            layers.append(AgentLayer(name='10. Documentation Layer', nodes=['Report Agent'], purpose='점검/정비 보고서 초안 생성'))
        layers.append(AgentLayer(name='11. Audit & Persistence Layer', nodes=['Evaluation / Audit Agent', 'History Store'], purpose='금지 표현, 안전 게이트 준수, 실행 이력 저장'))
        return layers
