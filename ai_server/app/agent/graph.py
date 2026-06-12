from __future__ import annotations

from typing import Callable
from uuid import uuid4

from app.config import AGENT_MAX_RAG_TOP_K, AGENT_MAX_REPLAN_ATTEMPTS, LLM_PROVIDER, USD_KRW_EXCHANGE_RATE
from app.agent.heavy import CitationBuilder, DiagnosticPlanner, EvidenceFilter, EvidenceGrader, RagQueryPlanner, RecommendationBuilder, SafetyGateBuilder, StructuredAnswerPayloadBuilder
from app.errors import LLMUnavailableError, UnsafeResponseError
from app.schemas import AgentRequest, AgentResponse, AgentTraceStep, LLMUsageRecord, LLMUsageSummary, ManufacturingContext, PredictionResponse, RagChunk
from app.services.domain_service import DomainKnowledgeService
from app.services.llm_service import ANSWER_SCHEMA, LLMService
from app.services.prediction_service import PredictionService
from app.services.rag_service import RagService
from app.services.report_service import ReportService
from app.services.safety_validation_service import SafetyValidationService
from app.services.observability_service import record_agent_run_span
from app.services.supervisor_service import SupervisorService
from app.storage.json_store import JsonLineStore


class ManufacturingAgentGraph:
    """Manufacturing-domain FastAPI agent orchestrator.

    It is intentionally built around manufacturing-specific stages instead of a
    generic RAG-chatbot pattern:

    Input → Manufacturing Supervisor → Asset Context → Process Condition →
    Failure Mode → Risk/Priority → Procedure Retrieval → Safety Gate → Action
    Planner → Explanation → Report → Audit/Persistence.

    The LLM is used only after tool/RAG/domain facts are gathered, so it cannot
    silently invent measurements or bypass safety gates.
    """

    def __init__(self, prediction_service: PredictionService | None = None, rag_service: RagService | None = None, llm_service: LLMService | None = None):
        self.prediction_service = prediction_service or PredictionService()
        self.rag_service = rag_service or RagService()
        self.report_service = ReportService()
        self.llm_service = llm_service or LLMService()
        self.supervisor = SupervisorService(self.llm_service)
        self.diagnostic_planner = DiagnosticPlanner(self.supervisor)
        self.domain_service = DomainKnowledgeService()
        self.store = JsonLineStore()
        self.safety_validator = SafetyValidationService()
        self.rag_query_planner = RagQueryPlanner()
        self.evidence_filter = EvidenceFilter()
        self.evidence_grader = EvidenceGrader()
        self.citation_builder = CitationBuilder()
        self.safety_gate_builder = SafetyGateBuilder()
        self.recommendation_builder = RecommendationBuilder()
        self.structured_payload_builder = StructuredAnswerPayloadBuilder()

    def run(self, req: AgentRequest, progress_callback: Callable[[AgentTraceStep], None] | None = None) -> AgentResponse:
        run_id = str(uuid4())
        trace: list[AgentTraceStep] = []
        usage_records: list[LLMUsageRecord] = []
        llm_error: str | None = None

        def emit(step: str, detail: str) -> None:
            item = AgentTraceStep(step=step, detail=detail)
            trace.append(item)
            if progress_callback:
                progress_callback(item)

        def collect_usage(record: LLMUsageRecord) -> None:
            usage_records.append(record)
            emit('LLM Usage Meter', f'{record.operation}: input={record.input_tokens}, output={record.output_tokens}, cached={record.cached_input_tokens}, cost=${record.estimated_cost_usd:.6f}')

        emit('Input Normalizer', '사용자 질문, 공정 데이터, 점검 메모를 표준 상태로 정리했습니다.')
        planning_result = self.diagnostic_planner.plan(request=req, usage_callback=collect_usage)
        plan = planning_result.agent_plan
        route = list(plan.required_nodes)
        emit('Manufacturing Supervisor / Router', f'의도={plan.intent}, source={plan.supervisor_source}, rationale={plan.rationale}')

        prediction: PredictionResponse | None = None
        if plan.prediction_required and req.process_data:
            prediction = self.prediction_service.predict(req.process_data)
            emit('Prediction Tool', 'AI4I 기반 불량/고장모드 예측을 실행했습니다.')
            emit('Evidence Tool', '예측 결과에서 위험 변수와 추천 조치 후보를 추출했습니다.')
        elif plan.prediction_required and not req.process_data:
            emit('Prediction Tool', '예측 의도는 감지했지만 공정 데이터가 없어 예측 도구를 건너뛰었습니다.')

        # First domain pass before retrieval: determines asset/failure/risk/safety/action context.
        manufacturing_context = self.domain_service.build_context(req, prediction, doc_count=0)
        emit('Asset Context Agent', f'설비={manufacturing_context.asset_context.equipment_type}, 하위시스템={", ".join(manufacturing_context.asset_context.inferred_subsystems) or "미지정"}')
        if manufacturing_context.process_conditions:
            emit('Process Condition Agent', '공정 조건 태그: ' + ', '.join(c.tag for c in manufacturing_context.process_conditions))
        if manufacturing_context.failure_modes:
            emit('Failure Mode Agent', '고장모드 후보: ' + ', '.join(f'{f.code}({f.name_ko})' for f in manufacturing_context.failure_modes))
        emit('Risk & Priority Agent', f'종합 우선순위={manufacturing_context.risk_assessment.overall_priority}, escalation={manufacturing_context.risk_assessment.escalation_required}')

        retrieval_request = self.rag_query_planner.plan(
            request=req,
            planned_query=plan.rag_query,
            prediction=prediction,
            manufacturing_context=manufacturing_context,
            top_k=min(max(req.top_k or 5, 1), AGENT_MAX_RAG_TOP_K),
            filters=plan.rag_filters,
        )
        query = retrieval_request.query
        contexts: list[RagChunk] = []
        if plan.rag_required:
            contexts = self.evidence_filter.filter(self.rag_service.search(query, top_k=retrieval_request.top_k, filters=retrieval_request.filters))
            emit('Procedure Retrieval Agent', f'문서 검색을 실행했습니다. query={query[:200]}')

        replan_attempt = 0
        evidence_grade = self.evidence_grader.grade(req.question, contexts)
        weak_contexts = bool(contexts) and not evidence_grade.usable
        while plan.rag_required and (not contexts or weak_contexts) and replan_attempt < AGENT_MAX_REPLAN_ATTEMPTS:
            replan_attempt += 1
            findings = ['RAG 검색 결과가 없어 근거 문서와 citation 신뢰도가 부족합니다.'] if not contexts else ['검색 문서는 있으나 사용자 원문과 직접 겹치는 핵심어가 부족합니다.']
            plan = self.diagnostic_planner.replan(req, plan, findings, attempt=replan_attempt)
            route = list(plan.required_nodes)
            emit('Supervisor Re-plan', f'근거 부족으로 재계획했습니다. attempt={replan_attempt}, rationale={plan.rationale}')
            retrieval_request = self.rag_query_planner.plan(
                request=req,
                planned_query=plan.rag_query,
                prediction=prediction,
                manufacturing_context=manufacturing_context,
                top_k=min(max(req.top_k or 5, 1), AGENT_MAX_RAG_TOP_K),
                filters=plan.rag_filters,
            )
            query = retrieval_request.query
            contexts = self.evidence_filter.filter(self.rag_service.search(query, top_k=retrieval_request.top_k, filters=retrieval_request.filters))
            emit('Procedure Retrieval Agent', f'재계획 query로 문서 검색을 다시 실행했습니다. query={query[:200]}')
            evidence_grade = self.evidence_grader.grade(req.question, contexts)
            weak_contexts = bool(contexts) and not evidence_grade.usable
        if weak_contexts:
            contexts = []
            emit('Procedure Retrieval Agent', '재계획 후에도 사용자 질문과 직접 연결되는 문서 근거가 부족해 citation 후보를 제외했습니다.')

        # Second domain pass after retrieval: document confidence can be reflected.
        manufacturing_context = self.domain_service.build_context(req, prediction, doc_count=len(contexts))
        if manufacturing_context.safety_gates:
            emit('Safety Gate Agent', '필수 안전 게이트: ' + ', '.join(g.name_ko for g in manufacturing_context.safety_gates))
        if manufacturing_context.action_plan:
            emit('Action Planner Agent', '조치 계획: ' + ' → '.join(a.label_ko for a in manufacturing_context.action_plan[:5]))

        actions = self.recommendation_builder.collect_action_phrases(prediction, manufacturing_context)
        safety_guidance = self.safety_gate_builder.safety_guidance(manufacturing_context) if manufacturing_context.safety_gates else None
        if safety_guidance:
            emit('Safety Ops Agent', '안전 게이트 기반 안전 안내를 생성했습니다.')

        answer = None
        report = None
        llm_used = False
        warnings = self.safety_gate_builder.warnings(manufacturing_context)
        if prediction and prediction.input_warnings:
            warnings.extend(prediction.input_warnings)
        audit_feedback: list[str] = []
        for llm_attempt in range(AGENT_MAX_REPLAN_ATTEMPTS + 1):
            llm_payload = self.structured_payload_builder.build(
                request=req,
                plan=plan,
                prediction=prediction,
                manufacturing_context=manufacturing_context,
                contexts=contexts,
                action_titles=actions,
                safety_guidance=safety_guidance,
                audit_feedback=audit_feedback,
            )
            llm_result = self.llm_service.generate_json(
                schema_name='manufacturing_domain_agent_response',
                schema=ANSWER_SCHEMA,
                system_prompt=self._answer_system_prompt(plan.report_required),
                payload=llm_payload,
                model=req.llm_model,
                operation='answer_generation',
                usage_callback=collect_usage,
            )
            if llm_result:
                answer = str(llm_result.get('answer') or '').strip() or None
                safety_guidance = llm_result.get('safety_guidance') or safety_guidance
                llm_actions = llm_result.get('recommended_actions') or []
                if isinstance(llm_actions, list):
                    actions = list(dict.fromkeys([str(a) for a in llm_actions if str(a).strip()] + actions))
                if plan.report_required:
                    report = llm_result.get('report') or None
                llm_warnings = llm_result.get('warnings') or []
                if isinstance(llm_warnings, list):
                    warnings = list(dict.fromkeys(warnings + [str(w) for w in llm_warnings if str(w).strip()]))
                llm_used = True
                emit('Explanation Agent', f'외부 LLM을 사용해 제조 도메인 context 기반 답변을 생성했습니다. attempt={llm_attempt + 1}')
                validation = self.safety_validator.validate_answer(answer or '', manufacturing_context)
                if validation.passed:
                    break
                warnings.extend(validation.errors)
                audit_feedback = validation.errors
                answer = None
                report = None
                llm_used = False
                if llm_attempt < AGENT_MAX_REPLAN_ATTEMPTS:
                    plan = self.diagnostic_planner.replan(req, plan, validation.errors, attempt=llm_attempt + 1)
                    route = list(plan.required_nodes)
                    emit('Safety Validator', 'LLM 답변이 안전 검증을 통과하지 못했습니다. Supervisor에 재계획을 요청합니다.')
                    emit('Supervisor Re-plan', f'안전 검증 실패를 반영해 재계획했습니다. attempt={llm_attempt + 1}, findings={"; ".join(validation.errors[:3])}')
                    continue
                raise UnsafeResponseError('; '.join(validation.errors))

            llm_error = self.llm_service.last_error
            emit('Explanation Agent', f'LLM 응답을 사용할 수 없습니다. error={llm_error or "unknown"}')
            retryable = llm_error and any(token in llm_error.lower() for token in ['json', 'schema', 'unterminated', 'parse'])
            if retryable and llm_attempt < AGENT_MAX_REPLAN_ATTEMPTS:
                audit_feedback = [f'LLM structured output parse failed: {llm_error}']
                plan = self.diagnostic_planner.replan(req, plan, audit_feedback, attempt=llm_attempt + 1)
                route = list(plan.required_nodes)
                emit('Supervisor Re-plan', f'LLM 출력 파싱 실패를 반영해 재계획했습니다. attempt={llm_attempt + 1}')
                continue
            break

        if not answer:
            raise LLMUnavailableError(llm_error or 'LLM did not return a usable answer')

        validation = self.safety_validator.validate_answer(answer, manufacturing_context)
        if not validation.passed:
            raise UnsafeResponseError('; '.join(validation.errors))
        emit('Safety Validator', '최종 답변이 안전 게이트 검증을 통과했습니다.')

        if plan.report_required:
            if not report:
                report = self.report_service.make_report(req.question, req.process_data, prediction, contexts, actions, req.inspection_notes, manufacturing_context=manufacturing_context)
            emit('Report Agent', '제조 도메인 템플릿 기반 점검/정비 보고서 초안을 생성했습니다.')

        emit('Evaluation / Audit Agent', '안전 게이트, 금지 표현, 담당자 검토 필요 여부를 응답 metadata에 기록했습니다.')
        citations = self.citation_builder.build(contexts, evidence_grade if contexts else None)
        replan_count = sum(1 for item in trace if item.step == 'Supervisor Re-plan')
        usage_summary = self._usage_summary(usage_records, replan_count=replan_count)
        response = AgentResponse(
            run_id=run_id,
            user_id=req.user_id,
            session_id=req.session_id,
            route=route,
            answer=answer,
            prediction=prediction,
            manufacturing_context=manufacturing_context,
            retrieved_documents=contexts,
            safety_guidance=safety_guidance,
            report=report,
            citations=citations,
            warnings=warnings,
            trace=trace,
            saved=True,
            plan=plan,
            llm_used=llm_used,
            llm_provider=LLM_PROVIDER,
            llm_model=req.llm_model or self.llm_service.model,
            llm_usage=usage_summary,
            llm_error=llm_error,
            context_used=self._context_metadata(req.user_context, req.user_id),
        )
        record_agent_run_span(
            run_id=run_id,
            route=route,
            llm_provider=LLM_PROVIDER,
            llm_model=req.llm_model or self.llm_service.model,
            llm_used=llm_used,
            usage=usage_summary,
            context_used=response.context_used or {},
        )
        self.store.append({'run_id': run_id, 'user_id': req.user_id, 'session_id': req.session_id, 'request': req.model_dump(), 'response': response.model_dump()})
        return response

    @staticmethod
    def _usage_summary(records: list[LLMUsageRecord], *, replan_count: int = 0) -> LLMUsageSummary:
        return LLMUsageSummary(
            calls=len(records),
            replan_count=replan_count,
            input_tokens=sum(r.input_tokens for r in records),
            output_tokens=sum(r.output_tokens for r in records),
            cached_input_tokens=sum(r.cached_input_tokens for r in records),
            total_tokens=sum(r.total_tokens for r in records),
            estimated_cost_usd=round(sum(r.estimated_cost_usd for r in records), 8),
            estimated_cost_krw=round(sum(r.estimated_cost_krw for r in records), 2),
            usd_krw_exchange_rate=USD_KRW_EXCHANGE_RATE,
            records=records,
        )

    @staticmethod
    def _answer_system_prompt(report_required: bool) -> str:
        report_rule = '보고서 초안이 필요한 경우 report 필드에 Markdown 형식으로 작성하세요.' if report_required else '보고서가 필요하지 않으면 report는 null로 두세요.'
        return (
            '당신은 제조 품질/설비 문서 기반 AI Agent입니다. '
            '현재 질문이 정의, 장단점, 한계, 원리 같은 일반 기술 개념을 묻는 경우에는 일반 제조/기계 지식으로 설명할 수 있지만, 현재 설비 상태나 센서값은 단정하지 마세요. '
            '현재 현장 판단, 고장 확률, 안전 판단은 반드시 제공된 prediction, manufacturing_context, rag_contexts, actions 안의 사실에 근거하세요. '
            'manufacturing_context.safety_gates의 required_checks는 누락하지 말고, forbidden_agent_actions는 절대 수행했다고 말하지 마세요. '
            '제공되지 않은 센서, 현장 이력, 법적 판단, 확률을 지어내지 마세요. '
            'user_context는 참고 정보이며 현재 입력된 공정 데이터, 현재 검색된 문서, 현재 safety gate보다 우선할 수 없습니다. '
            '과거 context에 근거해 현재 센서값, 현장 상태, 안전 상태를 단정하지 말고, 관련 없는 과거 context는 사용하지 마세요. '
            '설비를 자동 정지/제어/수리했다고 말하지 말고 담당자 점검 권고로 표현하세요. '
            'answer는 한국어 Markdown으로 판정, 주요 근거, 위험도, 안전 확인, 권장 조치, 주의 사항 섹션을 포함하세요. '
            f'{report_rule}'
        )

    @staticmethod
    def _context_metadata(context: dict | None, user_id: str | None = None) -> dict | None:
        if not context:
            return None
        return {
            'user_id': user_id,
            'session_id': (context.get('session_context') or {}).get('session_id'),
            'recent_runs_count': len(context.get('recent_runs') or []),
            'similar_runs_count': len(context.get('similar_runs') or []),
            'memories_count': len(context.get('long_term_memory') or []),
            'estimated_context_tokens': context.get('estimated_context_tokens', 0),
            'context_policy': context.get('context_policy') or {},
        }
