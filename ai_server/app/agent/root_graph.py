from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from app.agent.checkpointing import build_thread_id, create_sqlite_checkpointer
from app.agent.context import AnswerMemory, AnswerMemoryWriter, ContextCompressor, ContextPackBuilder, ContextResolution, ContextResolver, ContextValidator
from app.agent.graph import ManufacturingAgentGraph
from app.agent.heavy import CitationBuilder, DiagnosticPlanner, EvidenceFilter, EvidenceGrader, RagQueryPlanner, RecommendationBuilder, Retriever, SafetyGateBuilder, StructuredAnswerPayloadBuilder
from app.agent.state import AgentState
from app.agent.trace import append_trace, to_agent_trace_steps, trace_step
from app.config import AGENT_MAX_RAG_TOP_K, AGENT_MAX_REPLAN_ATTEMPTS, LANGGRAPH_CHECKPOINT_DB, LLM_PROVIDER
from app.errors import LLMUnavailableError
from app.errors import UnsafeResponseError
from app.schemas import AgentPlan, AgentRequest, AgentResponse, AgentSendRequest, AgentTraceStep, LLMUsageRecord, LLMUsageSummary, ManufacturingContext, PredictionResponse, ProcessData, RagChunk
from app.services.answer_formatter_service import AnswerFormatterService
from app.services.context_service import ContextService
from app.services.glossary_answer_service import GlossaryAnswerService
from app.services.intent_classifier_service import IntentClassifierService
from app.services.intent_gateway_service import IntentGatewayService
from app.services.llm_service import ANSWER_SCHEMA, LLMService
from app.services.memory_service import MemoryService
from app.services.rag_service import RagService
from app.services.user_service import UserService
from app.storage.sqlite_store import SQLiteStore


class RootManufacturingGraph:
    """LangGraph facade for hierarchical orchestration.

    This is the migration bridge from the previous imperative graph to a
    stateful root graph. Fast/unsupported paths are native nodes; complex
    manufacturing work is delegated to the existing ManufacturingAgentGraph as
    a heavy subgraph node until the lower-level subgraphs are split further.
    """

    def __init__(
        self,
        *,
        store: SQLiteStore,
        user_service: UserService,
        context_service: ContextService,
        memory_service: MemoryService,
        heavy_graph: ManufacturingAgentGraph,
        llm_service: LLMService,
        rag_service: RagService,
        intent_classifier: IntentClassifierService | None = None,
        checkpoint_path: Path | None = None,
    ):
        self.store = store
        self.user_service = user_service
        self.context_service = context_service
        self.memory_service = memory_service
        self.heavy_graph = heavy_graph
        self.llm_service = llm_service
        self.rag_service = rag_service
        self.intent_classifier = intent_classifier or IntentClassifierService(llm_service)
        self.intent_gateway = IntentGatewayService(intent_classifier=self.intent_classifier)
        self.glossary_answer_service = GlossaryAnswerService()
        self.answer_formatter = AnswerFormatterService()
        self.context_resolver = ContextResolver()
        self.context_pack_builder = ContextPackBuilder()
        self.context_compressor = ContextCompressor(max_recent_turns=5)
        self.answer_memory_writer = AnswerMemoryWriter()
        self.context_validator = ContextValidator()
        self.diagnostic_planner = DiagnosticPlanner(self.heavy_graph.supervisor)
        self.rag_query_planner = RagQueryPlanner()
        self.retriever = Retriever(rag_service)
        self.evidence_filter = EvidenceFilter()
        self.evidence_grader = EvidenceGrader()
        self.citation_builder = CitationBuilder()
        self.safety_gate_builder = SafetyGateBuilder()
        self.recommendation_builder = RecommendationBuilder()
        self.structured_payload_builder = StructuredAnswerPayloadBuilder()
        self.checkpoint_path = checkpoint_path or LANGGRAPH_CHECKPOINT_DB.with_name(f'{LANGGRAPH_CHECKPOINT_DB.stem}_v2{LANGGRAPH_CHECKPOINT_DB.suffix}')
        self._checkpointer_handle = create_sqlite_checkpointer(self.checkpoint_path)
        self.checkpointer = self._checkpointer_handle.checkpointer
        self.graph = self._build_graph()
        self._progress_callback: Callable[[AgentTraceStep], None] | None = None

    def close(self) -> None:
        self._checkpointer_handle.close()

    def run(self, req: AgentSendRequest, progress_callback: Callable[[AgentTraceStep], None] | None = None) -> AgentResponse:
        self._progress_callback = progress_callback
        session_id = req.session_id or f'session_{uuid4().hex[:12]}'
        thread_id = build_thread_id(user_id=req.user_id, session_id=session_id)
        state: AgentState = {
            'state_schema_version': 2,
            'run_id': str(uuid4()),
            'user_id': req.user_id,
            'session_id': session_id,
            'thread_id': thread_id,
            'current_user_message': req.message,
            'send_request': req.model_dump(),
            'warnings': [],
            'errors': [],
            'usage_records': [],
            'trace': [],
            'replan_count': 0,
        }
        config = self._thread_config(user_id=req.user_id, session_id=session_id)
        try:
            final_state = self.graph.invoke(state, config=config)
            response = final_state.get('response')
            if not response:
                raise LLMUnavailableError('Root graph did not produce a response')
            return self._response_model(response)
        finally:
            self._progress_callback = None

    def preview_route(self, req: AgentSendRequest) -> dict[str, Any]:
        session_id = req.session_id or f'session_{uuid4().hex[:12]}'
        previous = self._checkpoint_values(user_id=req.user_id, session_id=session_id)
        state = self._request_context_node({
            'state_schema_version': 2,
            'run_id': str(uuid4()),
            'user_id': req.user_id,
            'session_id': session_id,
            'thread_id': build_thread_id(user_id=req.user_id, session_id=session_id),
            'current_user_message': req.message,
            'recent_turns': previous.get('recent_turns') or [],
            'rolling_summary': previous.get('rolling_summary') or '',
            'recent_turn_routes': previous.get('recent_turn_routes') or [],
            'last_answer_memory': previous.get('last_answer_memory') or {},
            'session_last_process_data': previous.get('session_last_process_data'),
            'send_request': req.model_dump(),
            'warnings': [],
            'errors': [],
            'usage_records': [],
            'trace': [],
            'replan_count': 0,
        })
        state = self.intent_gateway_node(state)
        return state['intent_gateway']

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node('request_context', self._request_context_node)
        graph.add_node('intent_gateway', self.intent_gateway_node)
        graph.add_node('fast_concept_answer', self._fast_concept_answer_node)
        graph.add_node('general_lightweight_answer', self._general_lightweight_answer_node)
        graph.add_node('recommended_action_recap', self._recommended_action_recap_node)
        graph.add_node('recommended_action_item_explanation', self._recommended_action_item_explanation_node)
        graph.add_node('lightweight_rag_answer', self._lightweight_rag_answer_node)
        graph.add_node('unsupported_or_clarification', self._unsupported_or_clarification_node)
        graph.add_node('meta_feedback', self._meta_feedback_node)
        graph.add_node('supervisor_planning', self._supervisor_planning_node)
        graph.add_node('manufacturing_analysis', self._manufacturing_analysis_node)
        graph.add_node('evidence_retrieval', self._evidence_retrieval_node)
        graph.add_node('safety', self._safety_node)
        graph.add_node('response_synthesis', self._response_synthesis_node)
        graph.add_node('documentation', self._documentation_node)
        graph.add_node('response_packager', self._response_packager_node)
        graph.add_node('focus_updater', self._focus_updater_node)
        graph.add_node('audit_persistence', self._audit_persistence_node)
        graph.set_entry_point('request_context')
        graph.add_edge('request_context', 'intent_gateway')
        graph.add_conditional_edges(
            'intent_gateway',
            self.route_after_gateway,
            {
                'fast_concept_answer': 'fast_concept_answer',
                'general_lightweight_answer': 'general_lightweight_answer',
                'recommended_action_recap': 'recommended_action_recap',
                'recommended_action_item_explanation': 'recommended_action_item_explanation',
                'lightweight_rag_answer': 'lightweight_rag_answer',
                'unsupported_or_clarification': 'unsupported_or_clarification',
                'meta_feedback': 'meta_feedback',
                'report_answer': 'supervisor_planning',
                'supervisor_planning': 'supervisor_planning',
            },
        )
        graph.add_edge('fast_concept_answer', 'focus_updater')
        graph.add_edge('general_lightweight_answer', 'focus_updater')
        graph.add_edge('recommended_action_recap', 'focus_updater')
        graph.add_edge('recommended_action_item_explanation', 'focus_updater')
        graph.add_edge('lightweight_rag_answer', 'focus_updater')
        graph.add_edge('unsupported_or_clarification', 'focus_updater')
        graph.add_edge('meta_feedback', 'audit_persistence')
        graph.add_conditional_edges(
            'supervisor_planning',
            self._route_after_supervisor,
            {
                'manufacturing_analysis': 'manufacturing_analysis',
                'evidence_retrieval': 'evidence_retrieval',
                'safety': 'safety',
                'response_synthesis': 'response_synthesis',
            },
        )
        graph.add_conditional_edges(
            'manufacturing_analysis',
            self._route_after_manufacturing,
            {
                'evidence_retrieval': 'evidence_retrieval',
                'safety': 'safety',
                'response_synthesis': 'response_synthesis',
            },
        )
        graph.add_conditional_edges(
            'evidence_retrieval',
            self._route_after_retrieval,
            {
                'safety': 'safety',
                'response_synthesis': 'response_synthesis',
            },
        )
        graph.add_edge('safety', 'response_synthesis')
        graph.add_conditional_edges(
            'response_synthesis',
            self._route_after_response,
            {
                'documentation': 'documentation',
                'response_packager': 'response_packager',
            },
        )
        graph.add_edge('documentation', 'response_packager')
        graph.add_edge('response_packager', 'focus_updater')
        graph.add_edge('focus_updater', 'audit_persistence')
        graph.add_edge('audit_persistence', END)
        return graph.compile(checkpointer=self.checkpointer)

    def _request_context_node(self, state: AgentState) -> AgentState:
        req = self._send_request_model(state)
        self.user_service.validate(req.user_id)
        session_id = state['session_id']
        req.session_id = session_id
        turn_process_data = req.process_data.model_dump() if req.process_data else None
        session_last_process_data = state.get('session_last_process_data')
        self.user_service.upsert_session(user_id=req.user_id, session_id=session_id, title=req.message[:80] if req.message else None)
        base = self._to_agent_request(req, session_id=session_id)
        user_context = self.context_service.build(user_id=req.user_id, session_id=session_id, request=base)
        last_answer_memory = self._answer_memory_from_state(state.get('last_answer_memory'))
        effective_process_data = req.process_data
        previous_turn_process_data = None
        reference_previous = False
        if not effective_process_data and session_last_process_data and self._references_previous_process_data(req.message):
            previous_turn_process_data = session_last_process_data
            effective_process_data = ProcessData(**session_last_process_data)
            reference_previous = True
        process_policy = {
            'current_process_data_available': bool(turn_process_data),
            'session_last_process_data_available': bool(session_last_process_data),
            'previous_turn_process_data_used': reference_previous,
            'rule': 'Previous process_data is used only for explicit current-value/current-condition follow-up questions.',
        }
        compressed_context = self.context_compressor.compress(
            messages=list(state.get('recent_turns') or []) + [{'role': 'user', 'content': req.message}],
            previous_rolling_summary=state.get('rolling_summary') or '',
        )
        context_resolution = self.context_resolver.resolve(
            current_user_message=req.message,
            last_answer_memory=last_answer_memory,
            recent_turns=compressed_context.recent_turns,
            rolling_summary=compressed_context.rolling_summary,
        )
        context_packs = self.context_pack_builder.build(
            current_user_message=req.message,
            context_resolution=context_resolution,
            compressed_context=compressed_context.model_dump(),
            last_answer_memory=last_answer_memory,
            recent_turn_routes=state.get('recent_turn_routes') or [],
            process_data_policy=process_policy,
        )
        context_validation_warnings = self.context_validator.validate(
            context_resolution=context_resolution.model_dump(),
            context_packs=context_packs.model_dump(),
        )
        turn_context = {
            'original_question': req.message,
            'standalone_query': context_resolution.standalone_query,
            'is_followup': context_resolution.is_followup,
            'followup_type': context_resolution.followup_type,
            'followup_target': context_resolution.followup_target,
            'confidence': context_resolution.confidence,
            'reason': context_resolution.reason,
        }
        user_context['turn_context'] = turn_context
        user_context['process_data_reference_policy'] = process_policy
        user_context['last_answer_memory'] = last_answer_memory.model_dump() if last_answer_memory else {}
        user_context['context_resolution'] = context_resolution.model_dump()
        user_context['context_packs'] = context_packs.model_dump()
        user_context['compressed_context'] = compressed_context.model_dump()
        user_context['context_validation_warnings'] = context_validation_warnings
        request = self._to_agent_request(req, session_id=session_id, user_context=user_context, question=context_resolution.standalone_query, process_data=effective_process_data)
        self._emit_trace(state, trace_step(
            node_id='request_context.context_builder',
            node_name='Request Context Builder',
            node_type='subgraph',
            layer='Request Context',
            status='success',
            input_summary=f'user_id={req.user_id}, session_id={session_id}',
            output_summary=f'followup={context_resolution.is_followup}, type={context_resolution.followup_type}, previous_process_data_used={reference_previous}',
        ))
        state['send_request'] = req.model_dump()
        state['request'] = request.model_dump()
        state['user_context'] = user_context
        state['turn_context'] = turn_context
        state['context_resolution'] = context_resolution.model_dump()
        state['context_packs'] = context_packs.model_dump()
        state['compressed_context'] = compressed_context.model_dump()
        state['rolling_summary'] = compressed_context.rolling_summary
        state['context_validation_warnings'] = context_validation_warnings
        if context_validation_warnings:
            state['warnings'] = list(dict.fromkeys((state.get('warnings') or []) + context_validation_warnings))
        state['turn_process_data'] = turn_process_data
        state['previous_turn_process_data'] = previous_turn_process_data
        state['process_data_reference_policy'] = process_policy
        return self._return_state(state)

    def intent_gateway_node(self, state: AgentState) -> AgentState:
        def collect_usage(record: LLMUsageRecord) -> None:
            state['usage_records'].append(record)
            self._emit_trace(state, trace_step(
                node_id='audit.usage_meter',
                node_name='Usage Meter',
                node_type='metric',
                layer='Audit / Persistence',
                status='success',
                output_summary=f'{record.operation}: input={record.input_tokens}, output={record.output_tokens}, cost=${record.estimated_cost_usd:.6f}',
            ))

        request = self._request_model(state)
        gateway = self.intent_gateway.classify(request=request, user_context=state.get('user_context') or {}, usage_callback=collect_usage)
        self._emit_trace(state, trace_step(
            node_id='intent_gateway.classifier',
            node_name='Intent Gateway',
            node_type='router',
            layer='Intent Gateway',
            status='success',
            input_summary=request.question[:120],
            output_summary=f'turn_type={gateway["turn_type"]}, path={gateway["selected_path"]}',
        ))
        state['intent_gateway'] = gateway
        state['selected_path'] = gateway['selected_path']
        return self._return_state(state)

    @staticmethod
    def route_after_gateway(state: AgentState) -> str:
        return state.get('selected_path') or 'supervisor_planning'

    def _fast_concept_answer_node(self, state: AgentState) -> AgentState:
        request = self._request_model(state)
        glossary_payload = self.glossary_answer_service.answer_payload(request.question)
        if glossary_payload:
            formatter_context = {
                'selected_path': state.get('selected_path') or 'fast_concept_answer',
                'answer_type': (state.get('intent_gateway') or {}).get('answer_type') or 'definition',
                'concept_payload': glossary_payload,
                'reference_note': self._reference_note(state, glossary_payload.get('term')),
            }
            answer = self.answer_formatter.fast_concept_answer(formatter_context)
            self._emit_trace(state, trace_step(
                node_id='fast_path.glossary_answer',
                node_name='No-LLM Glossary Answer',
                node_type='template',
                layer='Fast Path',
                status='success',
                output_summary=f'term={glossary_payload["term"]}',
            ))
            state['response'] = self._response_from_state(
                state,
                answer=answer,
                warnings=[],
                route=['request_context.context_builder', 'intent_gateway.classifier', 'fast_path.glossary_answer'],
            )
            state['formatter_context'] = formatter_context
            state['structured_answer_payload'] = {'concept': glossary_payload}
            return self._return_state(state)

        usage_records: list[LLMUsageRecord] = []

        def collect_usage(record: LLMUsageRecord) -> None:
            usage_records.append(record)

        payload = {
            'question': request.question,
            'context_resolution': (request.user_context or {}).get('context_resolution') or {},
            'intent_gateway': state.get('intent_gateway') or {},
            'policy': {
                'answer_scope': 'general_concept_only',
                'do_not_infer_current_machine_state': True,
                'must_say_process_data_required_for_current_risk': True,
            },
        }
        result = self.llm_service.generate_json(
            schema_name='fast_concept_answer',
            schema=ANSWER_SCHEMA,
            system_prompt=(
                '당신은 제조/기계 개념을 설명하는 AI입니다. '
                '정의, 장단점, 한계, 원리 질문에는 일반 제조/기계 지식으로 간결하게 답하세요. '
                '현재 설비 상태, 고장 확률, 안전 상태는 공정 데이터와 검증 근거 없이는 단정하지 마세요. '
                '답변에는 "현재 설비 상태나 고장 위험은 실제 공정 데이터가 있어야 판단할 수 있습니다."라는 경계 문구를 포함하세요.'
            ),
            payload=payload,
            model=request.llm_model,
            operation='fast_concept_answer',
            usage_callback=collect_usage,
        )
        if not result:
            raise LLMUnavailableError(self.llm_service.last_error or 'Fast concept answer failed')
        state['usage_records'].extend(usage_records)
        self._emit_trace(state, trace_step(
            node_id='fast_path.concept_answer',
            node_name='Concept Answer Node',
            node_type='llm',
            layer='Fast Path',
            status='success',
            output_summary='general concept answer composed',
        ))
        answer = str(result.get('answer') or '').strip()
        warnings = [str(w) for w in (result.get('warnings') or []) if str(w).strip()]
        state['response'] = self._response_from_state(state, answer=answer, warnings=warnings, route=['request_context.context_builder', 'intent_gateway.classifier', 'fast_path.concept_answer'])
        return self._return_state(state)

    def _general_lightweight_answer_node(self, state: AgentState) -> AgentState:
        gateway = state.get('intent_gateway') or {}
        answer_context = ((state.get('context_packs') or {}).get('answer_context') or {})
        memory = answer_context.get('relevant_answer_memory') or {}
        formatter_context = {
            'selected_path': state.get('selected_path') or 'general_lightweight_answer',
            'answer_type': gateway.get('answer_type') or 'explanation',
            'target': (gateway.get('resolved_reference') or {}).get('normalized') or memory.get('focus'),
            'resolved_claim': gateway.get('resolved_claim'),
            'phrase_repair': gateway.get('phrase_repair'),
            'followup_target': (state.get('context_resolution') or {}).get('followup_target'),
        }
        answer = self.answer_formatter.general_lightweight_answer(formatter_context)
        self._emit_trace(state, trace_step(
            node_id='general_lightweight.answer',
            node_name='General Lightweight Answer',
            node_type='template',
            layer='Fast Path',
            status='success',
            output_summary=f'answer_type={formatter_context.get("answer_type")}',
        ))
        state['formatter_context'] = formatter_context
        state['response'] = self._response_from_state(
            state,
            answer=answer,
            warnings=[],
            route=['request_context.context_builder', 'intent_gateway.classifier', 'general_lightweight.answer'],
        )
        return self._return_state(state)

    def _recommended_action_recap_node(self, state: AgentState) -> AgentState:
        formatter_context = ((state.get('context_packs') or {}).get('formatter_context') or {})
        answer = self.answer_formatter.recommended_action_recap(formatter_context)
        self._emit_trace(state, trace_step(
            node_id='response.recommended_action_recap',
            node_name='Recommended Action Recap Formatter',
            node_type='formatter',
            layer='Response Synthesis',
            status='success',
            output_summary=f'actions={len(formatter_context.get("recommended_actions") or [])}',
        ))
        state['formatter_context'] = formatter_context
        state['response'] = self._response_from_state(
            state,
            answer=answer,
            warnings=[],
            route=['request_context.context_builder', 'intent_gateway.classifier', 'response.recommended_action_recap'],
        )
        return self._return_state(state)

    def _recommended_action_item_explanation_node(self, state: AgentState) -> AgentState:
        formatter_context = ((state.get('context_packs') or {}).get('formatter_context') or {})
        answer = self.answer_formatter.recommended_action_item_explanation(formatter_context)
        self._emit_trace(state, trace_step(
            node_id='response.recommended_action_item_explanation',
            node_name='Recommended Action Item Formatter',
            node_type='formatter',
            layer='Response Synthesis',
            status='success',
            output_summary=f'item={formatter_context.get("followup_item_index")}',
        ))
        state['formatter_context'] = formatter_context
        state['response'] = self._response_from_state(
            state,
            answer=answer,
            warnings=[],
            route=['request_context.context_builder', 'intent_gateway.classifier', 'response.recommended_action_item_explanation'],
        )
        return self._return_state(state)

    def _meta_feedback_node(self, state: AgentState) -> AgentState:
        self._emit_trace(state, trace_step(
            node_id='intent_gateway.meta_feedback',
            node_name='Meta Feedback',
            node_type='router',
            layer='Intent Gateway',
            status='success',
            output_summary='사용자 피드백을 제조 분석이 아닌 메타 응답으로 처리',
        ))
        memory = self._answer_memory_from_state(state.get('last_answer_memory'))
        formatter_context = {
            'selected_path': state.get('selected_path') or 'meta_feedback',
            'answer_type': 'meta_feedback',
            'answer_memory_focus': memory.focus if memory and memory.focus else None,
        }
        state['response'] = self._response_from_state(
            state,
            answer=self.answer_formatter.meta_feedback(formatter_context),
            warnings=[],
            route=['request_context.context_builder', 'intent_gateway.classifier', 'intent_gateway.meta_feedback'],
        )
        state['formatter_context'] = formatter_context
        return self._return_state(state)

    def _lightweight_rag_answer_node(self, state: AgentState) -> AgentState:
        request = self._request_model(state)
        contexts = self.rag_service.search(request.question, top_k=min(max(request.top_k or 3, 1), 5))
        usage_records: list[LLMUsageRecord] = []

        def collect_usage(record: LLMUsageRecord) -> None:
            usage_records.append(record)

        result = self.llm_service.generate_json(
            schema_name='lightweight_rag_answer',
            schema=ANSWER_SCHEMA,
            system_prompt=(
                '당신은 제조 문서 기반 Q&A Agent입니다. 제공된 rag_contexts를 우선 사용해 짧게 답하세요. '
                '현재 설비 상태나 고장 확률은 공정 데이터 없이 단정하지 마세요.'
            ),
            payload={
                'question': request.question,
                'rag_contexts': [c.model_dump() for c in contexts],
                'context_resolution': (request.user_context or {}).get('context_resolution') or {},
            },
            model=request.llm_model,
            operation='lightweight_rag_answer',
            usage_callback=collect_usage,
        )
        if not result:
            raise LLMUnavailableError(self.llm_service.last_error or 'Lightweight RAG answer failed')
        state['usage_records'].extend(usage_records)
        self._emit_trace(state, trace_step(
            node_id='retrieval.lightweight_retriever',
            node_name='Lightweight RAG Answer Node',
            node_type='subgraph',
            layer='Evidence Retrieval',
            status='success',
            output_summary=f'{len(contexts)} chunks used',
        ))
        response = self._response_from_state(
            state,
            answer=str(result.get('answer') or '').strip(),
            warnings=[str(w) for w in (result.get('warnings') or []) if str(w).strip()],
            route=['request_context.context_builder', 'intent_gateway.classifier', 'retrieval.lightweight_retriever', 'response.short_answer_composer'],
        )
        response.retrieved_documents = contexts
        state['response'] = response
        return self._return_state(state)

    def _unsupported_or_clarification_node(self, state: AgentState) -> AgentState:
        gateway = state.get('intent_gateway') or {}
        resolution = state.get('context_resolution') or {}
        answer = None
        if resolution.get('followup_type') == 'ambiguous':
            answer = '이전 답변의 어떤 대상을 가리키는지 명확하지 않습니다. 대상이나 항목을 지정해 다시 질문해 주세요.'
        if not answer:
            answer = self.answer_formatter.unsupported_or_clarification(
                reason=gateway.get('reason') or '이 요청은 현재 Agent가 수행할 수 없습니다.',
            )
        else:
            answer = self.answer_formatter.unsupported_or_clarification(reason=str(answer))
        self._emit_trace(state, trace_step(
            node_id='intent_gateway.clarification',
            node_name='Unsupported / Clarification Node',
            node_type='router',
            layer='Intent Gateway',
            status='success',
            output_summary=str(answer)[:160],
        ))
        state['response'] = self._response_from_state(
            state,
            answer=answer,
            warnings=['제조 설비 제어, 안전 보증, 법적 최종 판단은 수행하지 않습니다.'],
            route=['request_context.context_builder', 'intent_gateway.classifier', 'intent_gateway.clarification'],
        )
        return self._return_state(state)

    def _supervisor_planning_node(self, state: AgentState) -> AgentState:
        request = self._request_model(state)

        def collect_usage(record: LLMUsageRecord) -> None:
            state['usage_records'].append(record)
            self._emit_trace(state, trace_step(
                node_id='audit.usage_meter',
                node_name='Usage Meter',
                node_type='metric',
                layer='Audit / Persistence',
                status='success',
                output_summary=f'{record.operation}: input={record.input_tokens}, output={record.output_tokens}, cost=${record.estimated_cost_usd:.6f}',
            ))

        context_resolution = ContextResolution.model_validate(state.get('context_resolution') or {})
        planning_result = self.diagnostic_planner.plan(
            request=request,
            context_resolution=context_resolution,
            intent_result=state.get('intent_gateway') or {},
            usage_callback=collect_usage,
        )
        plan = planning_result.agent_plan
        diagnostic = planning_result.diagnostic_plan
        state['plan'] = plan.model_dump()
        state['diagnostic_plan'] = diagnostic.model_dump()
        state['route'] = list(plan.required_nodes)
        self._emit_trace(state, trace_step(
            node_id='supervisor.route_planner',
            node_name='Supervisor Planning Subgraph',
            node_type='subgraph',
            layer='Supervisor Planning',
            status='success',
            input_summary=f'question={request.question[:120]}',
            output_summary=f'intent={plan.intent}, data={diagnostic.requires_data}, rag={plan.rag_required}, prediction={plan.prediction_required}, safety={plan.safety_required}, report={plan.report_required}',
        ))
        return self._return_state(state)

    @staticmethod
    def _route_after_supervisor(state: AgentState) -> str:
        plan = RootManufacturingGraph._plan_model(state.get('plan'))
        if not plan:
            return 'response_synthesis'
        if plan.asset_context_required or plan.process_condition_required or plan.failure_mode_required or plan.risk_priority_required or plan.prediction_required:
            return 'manufacturing_analysis'
        if plan.rag_required:
            return 'evidence_retrieval'
        if plan.safety_required or plan.safety_gate_required:
            return 'safety'
        return 'response_synthesis'

    def _manufacturing_analysis_node(self, state: AgentState) -> AgentState:
        request = self._request_model(state)
        plan = self._plan_model(state.get('plan'))
        if not plan:
            return self._return_state(state)
        prediction: PredictionResponse | None = None
        if plan.prediction_required and request.process_data:
            prediction = self.heavy_graph.prediction_service.predict(request.process_data)
            self._emit_trace(state, trace_step(
                node_id='manufacturing.prediction_tool',
                node_name='Prediction Tool',
                node_type='tool',
                layer='Manufacturing Analysis',
                status='success',
                output_summary=f'risk={prediction.risk_level}, failure={prediction.predicted_failure}',
            ))
        elif plan.prediction_required:
            self._emit_trace(state, trace_step(
                node_id='manufacturing.prediction_tool',
                node_name='Prediction Tool',
                node_type='tool',
                layer='Manufacturing Analysis',
                status='skipped',
                output_summary='process_data가 없어 예측을 건너뜀',
            ))

        manufacturing_context = self.heavy_graph.domain_service.build_context(request, prediction, doc_count=0)
        state['prediction'] = prediction.model_dump() if prediction else None
        state['manufacturing_context'] = manufacturing_context.model_dump()
        self._emit_trace(state, trace_step(
            node_id='manufacturing.analysis_subgraph',
            node_name='Manufacturing Analysis Subgraph',
            node_type='subgraph',
            layer='Manufacturing Analysis',
            status='success',
            output_summary=(
                f'equipment={manufacturing_context.asset_context.equipment_type}, '
                f'conditions={len(manufacturing_context.process_conditions)}, '
                f'failure_modes={len(manufacturing_context.failure_modes)}, '
                f'priority={manufacturing_context.risk_assessment.overall_priority}'
            ),
        ))
        return self._return_state(state)

    @staticmethod
    def _route_after_manufacturing(state: AgentState) -> str:
        plan = RootManufacturingGraph._plan_model(state.get('plan'))
        if plan and plan.rag_required:
            return 'evidence_retrieval'
        if plan and (plan.safety_required or plan.safety_gate_required):
            return 'safety'
        return 'response_synthesis'

    def _evidence_retrieval_node(self, state: AgentState) -> AgentState:
        request = self._request_model(state)
        plan = self._plan_model(state.get('plan'))
        if not plan:
            return self._return_state(state)
        prediction = self._prediction_model(state.get('prediction'))
        manufacturing_context = self._manufacturing_context_model(state.get('manufacturing_context'))
        if not manufacturing_context:
            manufacturing_context = self.heavy_graph.domain_service.build_context(request, prediction, doc_count=0)

        contexts: list[RagChunk] = []
        retrieval_request = self.rag_query_planner.plan(
            request=request,
            planned_query=plan.rag_query,
            prediction=prediction,
            manufacturing_context=manufacturing_context,
            top_k=min(max(request.top_k or 5, 1), AGENT_MAX_RAG_TOP_K),
            filters=plan.rag_filters,
        )
        query = retrieval_request.query
        if plan.rag_required:
            contexts = self.evidence_filter.filter(self.retriever.retrieve(retrieval_request))
            self._emit_trace(state, trace_step(
                node_id='retrieval.document_retriever',
                node_name='Document Retriever',
                node_type='tool',
                layer='Evidence Retrieval',
                status='success',
                input_summary=query[:180],
                output_summary=f'chunks={len(contexts)}',
            ))

        replan_attempt = 0
        evidence_grade = self.evidence_grader.grade(request.question, contexts)
        state['evidence_grade'] = evidence_grade.model_dump()
        weak_contexts = bool(contexts) and not evidence_grade.usable
        while plan.rag_required and (not contexts or weak_contexts) and replan_attempt < AGENT_MAX_REPLAN_ATTEMPTS:
            replan_attempt += 1
            findings = ['RAG 검색 결과가 없어 근거 문서와 citation 신뢰도가 부족합니다.'] if not contexts else ['검색 문서는 있으나 사용자 원문과 직접 겹치는 핵심어가 부족합니다.']
            plan = self.diagnostic_planner.replan(request, plan, findings, attempt=replan_attempt)
            state['plan'] = plan.model_dump()
            state['route'] = list(plan.required_nodes)
            state['replan_count'] = int(state.get('replan_count') or 0) + 1
            retrieval_request = self.rag_query_planner.plan(
                request=request,
                planned_query=plan.rag_query,
                prediction=prediction,
                manufacturing_context=manufacturing_context,
                top_k=min(max(request.top_k or 5, 1), AGENT_MAX_RAG_TOP_K),
                filters=plan.rag_filters,
            )
            query = retrieval_request.query
            contexts = self.evidence_filter.filter(self.retriever.retrieve(retrieval_request))
            evidence_grade = self.evidence_grader.grade(request.question, contexts)
            state['evidence_grade'] = evidence_grade.model_dump()
            weak_contexts = bool(contexts) and not evidence_grade.usable
            self._emit_trace(state, trace_step(
                node_id='retrieval.local_replan',
                node_name='Retrieval Local Replan',
                node_type='router',
                layer='Evidence Retrieval',
                status='success',
                input_summary='; '.join(findings),
                output_summary=f'attempt={replan_attempt}, chunks={len(contexts)}',
                replan_reason='weak_or_empty_evidence',
            ))
        if weak_contexts:
            contexts = []
            self._emit_trace(state, trace_step(
                node_id='retrieval.evidence_grader',
                node_name='Evidence Grader',
                node_type='validator',
                layer='Evidence Retrieval',
                status='success',
                output_summary='사용자 질문과 직접 연결되는 문서 근거가 부족해 citation 후보 제외',
            ))
        else:
            self._emit_trace(state, trace_step(
                node_id='retrieval.evidence_grader',
                node_name='Evidence Grader',
                node_type='validator',
                layer='Evidence Retrieval',
                status='success',
                output_summary=f'usable_chunks={len(contexts)}',
            ))

        state['retrieved_documents'] = [item.model_dump() for item in contexts]
        state['citations'] = self.citation_builder.build(contexts, evidence_grade)
        state['manufacturing_context'] = self.heavy_graph.domain_service.build_context(request, prediction, doc_count=len(contexts)).model_dump()
        return self._return_state(state)

    @staticmethod
    def _route_after_retrieval(state: AgentState) -> str:
        plan = RootManufacturingGraph._plan_model(state.get('plan'))
        if plan and (plan.safety_required or plan.safety_gate_required):
            return 'safety'
        return 'response_synthesis'

    def _safety_node(self, state: AgentState) -> AgentState:
        manufacturing_context = self._manufacturing_context_model(state.get('manufacturing_context'))
        if not manufacturing_context:
            manufacturing_context = self.heavy_graph.domain_service.build_context(self._request_model(state), self._prediction_model(state.get('prediction')), doc_count=len(self._rag_chunks(state.get('retrieved_documents'))))
            state['manufacturing_context'] = manufacturing_context.model_dump()
        actions = self.recommendation_builder.to_action_dicts(self.recommendation_builder.collect_action_phrases(self._prediction_model(state.get('prediction')), manufacturing_context))
        safety_guidance = self.safety_gate_builder.safety_guidance(manufacturing_context) if manufacturing_context.safety_gates else None
        payload = dict(state.get('structured_answer_payload') or {})
        payload['recommended_actions'] = actions
        state['structured_answer_payload'] = payload
        state['safety_guidance'] = safety_guidance
        self._emit_trace(state, trace_step(
            node_id='safety.safety_subgraph',
            node_name='Safety Subgraph',
            node_type='subgraph',
            layer='Safety',
            status='success',
            output_summary=f'gates={len(manufacturing_context.safety_gates)}, actions={len(actions)}',
        ))
        return self._return_state(state)

    def _response_synthesis_node(self, state: AgentState) -> AgentState:
        request = self._request_model(state)
        plan = self._plan_model(state.get('plan'))
        if not plan:
            return self._return_state(state)
        prediction = self._prediction_model(state.get('prediction'))
        contexts = self._rag_chunks(state.get('retrieved_documents'))
        manufacturing_context = self._manufacturing_context_model(state.get('manufacturing_context')) or self.heavy_graph.domain_service.build_context(request, prediction, doc_count=len(contexts))
        payload = dict(state.get('structured_answer_payload') or {})
        action_items = self.recommendation_builder.to_action_dicts(payload.get('recommended_actions') or self.recommendation_builder.collect_action_phrases(prediction, manufacturing_context))
        action_titles = [item['title'] for item in action_items]
        safety_guidance = state.get('safety_guidance')
        if safety_guidance is None and manufacturing_context.safety_gates:
            safety_guidance = self.safety_gate_builder.safety_guidance(manufacturing_context)
        warnings = self.safety_gate_builder.warnings(manufacturing_context)
        if prediction and prediction.input_warnings:
            warnings.extend(prediction.input_warnings)

        answer: str | None = None
        report: str | None = None
        llm_error: str | None = None
        audit_feedback: list[str] = []
        llm_used = False

        def collect_usage(record: LLMUsageRecord) -> None:
            state['usage_records'].append(record)
            self._emit_trace(state, trace_step(
                node_id='audit.usage_meter',
                node_name='Usage Meter',
                node_type='metric',
                layer='Audit / Persistence',
                status='success',
                output_summary=f'{record.operation}: input={record.input_tokens}, output={record.output_tokens}, cost=${record.estimated_cost_usd:.6f}',
            ))

        for llm_attempt in range(AGENT_MAX_REPLAN_ATTEMPTS + 1):
            llm_result = self.llm_service.generate_json(
                schema_name='manufacturing_domain_agent_response',
                schema=ANSWER_SCHEMA,
                system_prompt=self.heavy_graph._answer_system_prompt(plan.report_required),
                payload=self.structured_payload_builder.build(request=request, plan=plan, prediction=prediction, manufacturing_context=manufacturing_context, contexts=contexts, action_titles=action_titles, safety_guidance=safety_guidance, audit_feedback=audit_feedback),
                model=request.llm_model,
                operation='answer_generation',
                usage_callback=collect_usage,
            )
            if llm_result:
                answer = str(llm_result.get('answer') or '').strip() or None
                safety_guidance = llm_result.get('safety_guidance') or safety_guidance
                llm_actions = llm_result.get('recommended_actions') or []
                if isinstance(llm_actions, list):
                    merged_titles = list(dict.fromkeys([str(a) for a in llm_actions if str(a).strip()] + action_titles))
                    action_items = self.recommendation_builder.to_action_dicts(merged_titles)
                    action_titles = [item['title'] for item in action_items]
                if plan.report_required:
                    report = llm_result.get('report') or None
                llm_warnings = llm_result.get('warnings') or []
                if isinstance(llm_warnings, list):
                    warnings = list(dict.fromkeys(warnings + [str(w) for w in llm_warnings if str(w).strip()]))
                llm_used = True
                validation = self.heavy_graph.safety_validator.validate_answer(answer or '', manufacturing_context)
                self._emit_trace(state, trace_step(
                    node_id='response.answer_composer',
                    node_name='Response Synthesis Subgraph',
                    node_type='llm',
                    layer='Response Synthesis',
                    status='success' if validation.passed else 'failed',
                    output_summary=f'attempt={llm_attempt + 1}, safety_passed={validation.passed}',
                ))
                if validation.passed:
                    break
                warnings.extend(validation.errors)
                audit_feedback = validation.errors
                answer = None
                report = None
                llm_used = False
                if llm_attempt < AGENT_MAX_REPLAN_ATTEMPTS:
                    plan = self.diagnostic_planner.replan(request, plan, validation.errors, attempt=llm_attempt + 1)
                    state['plan'] = plan.model_dump()
                    state['route'] = list(plan.required_nodes)
                    state['replan_count'] = int(state.get('replan_count') or 0) + 1
                    self._emit_trace(state, trace_step(
                        node_id='supervisor.parent_replan',
                        node_name='Supervisor Parent Replan',
                        node_type='router',
                        layer='Supervisor Planning',
                        status='success',
                        output_summary='안전 검증 실패를 반영해 재계획',
                        replan_reason='answer_safety_validation_failed',
                    ))
                    continue
                raise UnsafeResponseError('; '.join(validation.errors))

            llm_error = self.llm_service.last_error
            retryable = llm_error and any(token in llm_error.lower() for token in ['json', 'schema', 'unterminated', 'parse'])
            self._emit_trace(state, trace_step(
                node_id='response.answer_composer',
                node_name='Response Synthesis Subgraph',
                node_type='llm',
                layer='Response Synthesis',
                status='failed',
                output_summary=f'LLM 응답 사용 불가: {llm_error or "unknown"}',
            ))
            if retryable and llm_attempt < AGENT_MAX_REPLAN_ATTEMPTS:
                audit_feedback = [f'LLM structured output parse failed: {llm_error}']
                state['replan_count'] = int(state.get('replan_count') or 0) + 1
                continue
            break

        if not answer:
            raise LLMUnavailableError(llm_error or 'LLM did not return a usable answer')

        state['manufacturing_context'] = manufacturing_context.model_dump()
        payload['recommended_actions'] = action_items
        state['structured_answer_payload'] = payload
        state['safety_guidance'] = safety_guidance
        state['answer'] = answer
        state['report'] = report
        state['warnings'] = list(dict.fromkeys((state.get('warnings') or []) + warnings))
        state['llm_used'] = llm_used
        state['llm_error'] = llm_error
        return self._return_state(state)

    @staticmethod
    def _route_after_response(state: AgentState) -> str:
        plan = RootManufacturingGraph._plan_model(state.get('plan'))
        if plan and plan.report_required:
            return 'documentation'
        return 'response_packager'

    def _documentation_node(self, state: AgentState) -> AgentState:
        plan = self._plan_model(state.get('plan'))
        if not plan:
            return self._return_state(state)
        if not plan.report_required:
            return self._return_state(state)
        if not state.get('report'):
            request = self._request_model(state)
            state['report'] = self.heavy_graph.report_service.make_report(
                request.question,
                request.process_data,
                self._prediction_model(state.get('prediction')),
                self._rag_chunks(state.get('retrieved_documents')),
                self._recommended_action_titles(state),
                request.inspection_notes,
                manufacturing_context=self._manufacturing_context_model(state.get('manufacturing_context')),
            )
        self._emit_trace(state, trace_step(
            node_id='documentation.report_composer',
            node_name='Documentation Subgraph',
            node_type='subgraph',
            layer='Documentation',
            status='success',
            output_summary='점검/정비 보고서 초안 생성',
        ))
        return self._return_state(state)

    def _response_packager_node(self, state: AgentState) -> AgentState:
        request = self._request_model(state)
        contexts = self._rag_chunks(state.get('retrieved_documents'))
        citations = list(state.get('citations') or self.citation_builder.build(contexts))
        response = AgentResponse(
            run_id=state['run_id'],
            user_id=request.user_id,
            session_id=request.session_id,
            route=state.get('route') or [],
            answer=self.answer_formatter.sanitize_public_answer(state.get('answer') or ''),
            prediction=self._prediction_model(state.get('prediction')),
            manufacturing_context=self._manufacturing_context_model(state.get('manufacturing_context')),
            retrieved_documents=contexts,
            safety_guidance=state.get('safety_guidance'),
            report=state.get('report'),
            citations=citations,
            warnings=state.get('warnings') or [],
            trace=to_agent_trace_steps(state.get('trace') or []),
            saved=True,
            plan=self._plan_model(state.get('plan')),
            llm_used=bool(state.get('llm_used')),
            llm_provider=LLM_PROVIDER,
            llm_model=request.llm_model or self.llm_service.model,
            llm_usage=self._usage_summary(state.get('usage_records') or []),
            llm_error=state.get('llm_error'),
            context_used=self._context_metadata(request.user_context, request.user_id),
        )
        self._emit_trace(state, trace_step(
            node_id='response.response_packager',
            node_name='Response Packager',
            node_type='subgraph',
            layer='Response Synthesis',
            status='success',
            output_summary=f'route_nodes={len(response.route)}, citations={len(citations)}',
        ))
        state['citations'] = citations
        state['response'] = response.model_dump()
        return self._return_state(state)

    def _focus_updater_node(self, state: AgentState) -> dict[str, Any]:
        response = self._response_model(state.get('response'))
        answer_memory = self.answer_memory_writer.build(state=state, response=response)
        state['last_answer_memory'] = answer_memory.model_dump()
        state['recent_turn_routes'] = self.append_recent_turn_route(state, answer_memory.model_dump())
        state['recent_turns'] = self._append_recent_turn(state, response.answer or '')
        if state.get('turn_process_data'):
            state['session_last_process_data'] = state['turn_process_data']
        self._emit_trace(state, trace_step(
            node_id='memory.answer_memory_writer',
            node_name='Answer Memory Writer',
            node_type='memory',
            layer='Request Context',
            status='success',
            output_summary=f'focus={answer_memory.focus or "none"}, actions={len(answer_memory.recommended_actions)}',
        ))
        return self._return_state(state)

    def _audit_persistence_node(self, state: AgentState) -> AgentState:
        response = self._response_model(state.get('response'))
        request = self._request_model(state)
        if not state.get('last_answer_memory'):
            answer_memory = self.answer_memory_writer.build(state=state, response=response)
            state['last_answer_memory'] = answer_memory.model_dump()
            state['recent_turn_routes'] = self.append_recent_turn_route(state, answer_memory.model_dump())
            state['recent_turns'] = self._append_recent_turn(state, response.answer or '')
        self._emit_trace(state, trace_step(
            node_id='audit.memory_writer',
            node_name='Memory Writer',
            node_type='storage',
            layer='Audit / Persistence',
            status='success',
            output_summary='history/memory updated',
        ))
        if response.run_id == state['run_id']:
            response.trace = to_agent_trace_steps(state.get('trace') or [])
        else:
            response.trace = to_agent_trace_steps(state.get('trace') or []) + response.trace
        if response.llm_usage and not response.llm_usage.records and state.get('usage_records'):
            response.llm_usage = self._usage_summary(state['usage_records'])
        if response.run_id == state['run_id']:
            self.store.append({'run_id': response.run_id, 'user_id': request.user_id, 'session_id': request.session_id, 'request': request.model_dump(), 'response': response.model_dump()})
        self.memory_service.update_from_run(user_id=request.user_id or state['user_id'], request=request, response=response)
        response.context_used = response.context_used or self._context_metadata(request.user_context, request.user_id)
        state['response'] = response.model_dump()
        return self._return_state(state)

    def _response_from_state(self, state: AgentState, *, answer: str, warnings: list[str], route: list[str]) -> AgentResponse:
        request = self._request_model(state)
        return AgentResponse(
            run_id=state['run_id'],
            user_id=request.user_id,
            session_id=request.session_id,
            route=route,
            answer=self.answer_formatter.sanitize_public_answer(answer),
            warnings=warnings,
            trace=to_agent_trace_steps(state.get('trace') or []),
            saved=True,
            llm_used=bool(state.get('usage_records')),
            llm_provider=LLM_PROVIDER,
            llm_model=request.llm_model or self.llm_service.model,
            llm_usage=self._usage_summary(state.get('usage_records') or []),
            context_used=self._context_metadata(request.user_context, request.user_id),
        )

    @staticmethod
    def _usage_summary(records: list[LLMUsageRecord]) -> LLMUsageSummary:
        normalized = [RootManufacturingGraph._usage_record_model(item) for item in (records or [])]
        return LLMUsageSummary(
            calls=len(normalized),
            input_tokens=sum(item.input_tokens for item in normalized),
            output_tokens=sum(item.output_tokens for item in normalized),
            cached_input_tokens=sum(item.cached_input_tokens for item in normalized),
            total_tokens=sum(item.total_tokens for item in normalized),
            estimated_cost_usd=round(sum(item.estimated_cost_usd for item in normalized), 8),
            estimated_cost_krw=round(sum(item.estimated_cost_krw for item in normalized), 2),
            usd_krw_exchange_rate=normalized[-1].usd_krw_exchange_rate if normalized else 0.0,
            records=normalized,
        )

    @staticmethod
    def _to_agent_request(
        req: AgentSendRequest,
        *,
        session_id: str,
        user_context: dict[str, Any] | None = None,
        question: str | None = None,
        process_data: ProcessData | None = None,
    ) -> AgentRequest:
        return AgentRequest(
            user_id=req.user_id,
            question=req.message if question is None else question,
            process_data=req.process_data if process_data is None else process_data,
            inspection_notes=req.inspection_notes,
            generate_report=req.generate_report,
            top_k=req.top_k,
            session_id=session_id,
            mode=req.mode,
            llm_model=req.llm_model,
            user_context=user_context,
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
            'process_data_reference_policy': context.get('process_data_reference_policy') or {},
            'context_resolution': context.get('context_resolution') or {},
            'context_validation_warnings': context.get('context_validation_warnings') or [],
        }

    @staticmethod
    def append_recent_turn_route(state: AgentState, answer_memory: dict[str, Any]) -> list[dict[str, Any]]:
        current = list(state.get('recent_turn_routes') or [])
        current.append({
            'selected_path': answer_memory.get('selected_path'),
            'answer_type': answer_memory.get('answer_type'),
            'summary': answer_memory.get('short_summary'),
        })
        return current[-10:]

    def _emit_trace(self, state: dict[str, Any], step: dict[str, Any]) -> None:
        append_trace(state, step)
        if self._progress_callback:
            self._progress_callback(to_agent_trace_steps([step])[0])

    @staticmethod
    def _thread_config(*, user_id: str, session_id: str) -> dict[str, Any]:
        thread_id = build_thread_id(user_id=user_id, session_id=session_id)
        return {'configurable': {'thread_id': thread_id, 'user_id': user_id, 'session_id': session_id}}

    @staticmethod
    def _references_previous_process_data(*questions: str) -> bool:
        text = ' '.join(question or '' for question in questions).lower().replace(' ', '')
        reference_terms = [
            '방금데이터',
            '방금그조건',
            '그조건',
            '이조건',
            '이데이터',
            '그데이터',
            '현재값',
            '이수치',
            '이값',
            '이토크값',
            '토크값위험',
            '위험해',
            '위험도',
            '고장확률',
            '가능성',
        ]
        concept_only_terms = ['뭐야', '무엇', '정의', '설명', '장점', '단점', '한계', '원리']
        if any(term in text for term in concept_only_terms) and not any(term in text for term in ['값', '조건', '데이터', '위험', '확률', '가능성']):
            return False
        return any(term in text for term in reference_terms)

    def _checkpoint_values(self, *, user_id: str, session_id: str) -> dict[str, Any]:
        try:
            snapshot = self.graph.get_state(self._thread_config(user_id=user_id, session_id=session_id))
        except Exception:
            return {}
        values = dict(snapshot.values or {})
        if values.get('state_schema_version') != 2:
            return {}
        return values

    @staticmethod
    def _return_state(state: dict[str, Any]) -> dict[str, Any]:
        clean = RootManufacturingGraph._sanitize_value(state)
        clean['state_schema_version'] = 2
        return clean

    @staticmethod
    def _answer_memory_from_state(value: Any) -> AnswerMemory | None:
        if isinstance(value, AnswerMemory):
            return value
        if isinstance(value, dict) and value.get('short_summary'):
            return AnswerMemory.model_validate(value)
        return None

    def _append_recent_turn(self, state: AgentState, answer: str) -> list[dict[str, str]]:
        turns = list(state.get('recent_turns') or [])
        request = self._request_model(state) if state.get('request') else None
        question = request.question if request else state.get('current_user_message', '')
        turns.append({'role': 'user', 'content': str(question or '')})
        turns.append({'role': 'assistant', 'content': str(answer or '')})
        return turns[-10:]

    @staticmethod
    def _recommended_action_titles(state: AgentState) -> list[str]:
        actions = ((state.get('structured_answer_payload') or {}).get('recommended_actions') or [])
        titles: list[str] = []
        for action in actions:
            if isinstance(action, dict):
                title = action.get('title') or action.get('text') or action.get('action')
            else:
                title = str(action)
            if title:
                titles.append(str(title))
        return titles

    @staticmethod
    def _reference_note(state: AgentState, term: str | None = None) -> str | None:
        resolution = state.get('context_resolution') or {}
        if not resolution.get('is_followup'):
            return None
        target = resolution.get('followup_target') or term
        if not target:
            return None
        followup_type = resolution.get('followup_type')
        if followup_type in {'previous_concept', 'previous_answer_reason', 'previous_claim'}:
            return f'직전 답변의 "{target}"{RootManufacturingGraph._object_particle(str(target))} 기준으로 답변하겠습니다.'
        return None

    @staticmethod
    def _object_particle(text: str) -> str:
        if not text:
            return '을'
        code = ord(text[-1])
        if 0xAC00 <= code <= 0xD7A3:
            return '을' if (code - 0xAC00) % 28 else '를'
        return '를'

    @staticmethod
    def _sanitize_value(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return RootManufacturingGraph._sanitize_value(value.model_dump())
        if isinstance(value, dict):
            return {str(k): RootManufacturingGraph._sanitize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [RootManufacturingGraph._sanitize_value(item) for item in value]
        if isinstance(value, tuple):
            return [RootManufacturingGraph._sanitize_value(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    @staticmethod
    def _send_request_model(state: dict[str, Any]) -> AgentSendRequest:
        value = state.get('send_request') or {}
        if isinstance(value, AgentSendRequest):
            return value
        return AgentSendRequest.model_validate(value)

    @staticmethod
    def _request_model(state: dict[str, Any]) -> AgentRequest:
        value = state.get('request') or {}
        if isinstance(value, AgentRequest):
            return value
        return AgentRequest.model_validate(value)

    @staticmethod
    def _response_model(value: Any) -> AgentResponse:
        if isinstance(value, AgentResponse):
            return value
        return AgentResponse.model_validate(value)

    @staticmethod
    def _plan_model(value: Any) -> AgentPlan | None:
        if not value:
            return None
        if isinstance(value, AgentPlan):
            return value
        return AgentPlan.model_validate(value)

    @staticmethod
    def _prediction_model(value: Any) -> PredictionResponse | None:
        if not value:
            return None
        if isinstance(value, PredictionResponse):
            return value
        return PredictionResponse.model_validate(value)

    @staticmethod
    def _manufacturing_context_model(value: Any) -> ManufacturingContext | None:
        if not value:
            return None
        if isinstance(value, ManufacturingContext):
            return value
        return ManufacturingContext.model_validate(value)

    @staticmethod
    def _rag_chunks(value: Any) -> list[RagChunk]:
        chunks: list[RagChunk] = []
        for item in value or []:
            if isinstance(item, RagChunk):
                chunks.append(item)
            else:
                chunks.append(RagChunk.model_validate(item))
        return chunks

    @staticmethod
    def _usage_record_model(value: Any) -> LLMUsageRecord:
        if isinstance(value, LLMUsageRecord):
            return value
        return LLMUsageRecord.model_validate(value)
