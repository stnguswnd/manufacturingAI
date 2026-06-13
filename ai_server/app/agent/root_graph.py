from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from app.agent.artifacts import (
    AnswerDraft,
    AuditArtifact,
    ContextArtifact,
    EvidenceArtifact,
    MemoryArtifact,
    PlanningArtifact,
    PredictionArtifact,
    RequestArtifact,
    ResponseArtifact,
    RuntimeArtifact,
    SafetyArtifact,
    ValidationFailure,
    ValidationReport,
)
from app.agent.checkpointing import build_thread_id, create_sqlite_checkpointer
from app.agent.context_subagent import ContextInput
from app.agent.memory_subagent import MemoryInput
from app.agent.planning_subagent import PlanningInput
from app.agent.rag_evidence import RagEvidenceInput
from app.agent.runtime_deps import AgentRuntimeDeps
from app.agent.safety_subagent import SafetyInput
from app.agent.state import ManufacturingAgentState
from app.agent.trace import to_agent_trace_steps, trace_step
from app.config import AGENT_MAX_RAG_TOP_K, LANGGRAPH_CHECKPOINT_DB, LLM_PROVIDER
from app.errors import LLMUnavailableError
from app.schemas.agent import AgentPlan, AgentRequest, AgentResponse, AgentSendRequest, AgentTraceStep, LLMUsageRecord, LLMUsageSummary
from app.schemas.domain import ManufacturingContext
from app.schemas.prediction import PredictionResponse
from app.schemas.rag import RagChunk
from app.services.context_service import ContextService
from app.services.domain_service import DomainKnowledgeService
from app.services.intent_classifier_service import IntentClassifierService
from app.services.llm_service import LLMService
from app.services.memory_service import MemoryService
from app.services.prediction_service import PredictionService
from app.services.rag_service import RagService
from app.services.safety_validation_service import SafetyValidationService
from app.services.user_service import UserService
from app.storage.sqlite_store import SQLiteStore


STATE_SCHEMA_VERSION = 3
MAX_TOTAL_ATTEMPTS = 6
MAX_REWRITE_ATTEMPTS = 2
MAX_RAG_RERUN_ATTEMPTS = 3
MAX_SAFETY_RERUN_ATTEMPTS = 2

ARTIFACT_KEYS = {
    'request',
    'context',
    'planning',
    'prediction',
    'evidence',
    'safety',
    'draft',
    'validation',
    'response',
    'memory',
    'audit',
    'runtime',
}

TURN_SCOPED_ARTIFACT_KEYS = {
    'planning',
    'prediction',
    'evidence',
    'safety',
    'draft',
    'validation',
    'response',
    'audit',
}


@dataclass(frozen=True)
class ReviewRoute:
    action: str


class RootManufacturingGraph:
    """Artifact-only LangGraph runtime for the manufacturing bounded workflow."""

    def __init__(
        self,
        *,
        store: SQLiteStore,
        user_service: UserService,
        context_service: ContextService,
        memory_service: MemoryService,
        prediction_service: PredictionService,
        domain_service: DomainKnowledgeService,
        safety_validator: SafetyValidationService,
        llm_service: LLMService,
        rag_service: RagService,
        intent_classifier: IntentClassifierService | None = None,
        checkpoint_path: Path | None = None,
        deps: AgentRuntimeDeps | None = None,
    ):
        self.deps = deps or AgentRuntimeDeps.from_services(
            store=store,
            user_service=user_service,
            context_service=context_service,
            memory_service=memory_service,
            prediction_service=prediction_service,
            domain_service=domain_service,
            safety_validator=safety_validator,
            llm_service=llm_service,
            rag_service=rag_service,
            intent_classifier=intent_classifier,
        )
        self.store = self.deps.store
        self.prediction_service = self.deps.prediction_service
        self.domain_service = self.deps.domain_service
        self.safety_validator = self.deps.safety_validator
        self.llm_service = self.deps.llm_service
        self.context_subagent = self.deps.context_subagent
        self.planning_subagent = self.deps.planning_subagent
        self.rag_evidence_subagent = self.deps.rag_evidence_subagent
        self.safety_subagent = self.deps.safety_subagent
        self.memory_subagent = self.deps.memory_subagent
        self.checkpoint_path = checkpoint_path or LANGGRAPH_CHECKPOINT_DB.with_name(f'{LANGGRAPH_CHECKPOINT_DB.stem}_v3{LANGGRAPH_CHECKPOINT_DB.suffix}')
        self._checkpointer_handle = create_sqlite_checkpointer(self.checkpoint_path)
        self.checkpointer = self._checkpointer_handle.checkpointer
        self.graph = self._build_graph()
        self._progress_callback: Callable[[AgentTraceStep], None] | None = None

    def close(self) -> None:
        self._checkpointer_handle.close()

    def run(self, req: AgentSendRequest, progress_callback: Callable[[AgentTraceStep], None] | None = None) -> AgentResponse:
        self._progress_callback = progress_callback
        session_id = req.session_id or f'session_{uuid4().hex[:12]}'
        state: ManufacturingAgentState = {
            'state_schema_version': STATE_SCHEMA_VERSION,
            'run_id': str(uuid4()),
            'user_id': req.user_id,
            'session_id': session_id,
            'thread_id': build_thread_id(user_id=req.user_id, session_id=session_id),
            'request': RequestArtifact(
                user_id=req.user_id,
                session_id=session_id,
                question=req.message,
                original_message=req.message,
                process_data=req.process_data.model_dump() if req.process_data else None,
                inspection_notes=req.inspection_notes,
                top_k=req.top_k,
                mode=req.mode,
                llm_model=req.llm_model,
            ).model_dump(mode='json'),
            'runtime': RuntimeArtifact().model_dump(mode='json'),
        }
        try:
            final_state = self.graph.invoke(state, config=self._thread_config(user_id=req.user_id, session_id=session_id))
            if not final_state.get('response'):
                raise LLMUnavailableError('Root graph did not produce a response artifact')
            return self._agent_response_from_state(final_state)
        finally:
            self._progress_callback = None

    def preview_route(self, req: AgentSendRequest) -> dict[str, Any]:
        session_id = req.session_id or f'session_{uuid4().hex[:12]}'
        previous = self._checkpoint_values(user_id=req.user_id, session_id=session_id)
        state: ManufacturingAgentState = {
            'state_schema_version': STATE_SCHEMA_VERSION,
            'run_id': str(uuid4()),
            'user_id': req.user_id,
            'session_id': session_id,
            'thread_id': build_thread_id(user_id=req.user_id, session_id=session_id),
            'request': RequestArtifact(
                user_id=req.user_id,
                session_id=session_id,
                question=req.message,
                original_message=req.message,
                process_data=req.process_data.model_dump() if req.process_data else None,
                inspection_notes=req.inspection_notes,
                top_k=req.top_k,
                mode=req.mode,
                llm_model=req.llm_model,
            ).model_dump(mode='json'),
            'runtime': RuntimeArtifact().model_dump(mode='json'),
        }
        if previous.get('context'):
            state['context'] = previous['context']
        if previous.get('memory'):
            state['memory'] = previous['memory']
        state = self._request_context_node(state)
        state = self._planning_router_node(state)
        planning = self._planning_artifact(state)
        return {
            'selected_path': planning.selected_path,
            'answer_type': planning.answer_type,
            'turn_type': planning.answer_type,
            'requires_prediction': planning.needs_prediction,
            'requires_rag': planning.needs_rag,
            'requires_safety': planning.needs_safety,
            'reason': planning.reasoning_summary,
        }

    def _build_graph(self):
        graph = StateGraph(ManufacturingAgentState)
        graph.add_node('request_context', self._request_context_node)
        graph.add_node('planning_router', self._planning_router_node)
        graph.add_node('prediction_node', self._prediction_node)
        graph.add_node('prediction_quality_gate', self._prediction_quality_gate_node)
        graph.add_node('rag_evidence_subagent', self._rag_evidence_node)
        graph.add_node('evidence_quality_gate', self._evidence_quality_gate_node)
        graph.add_node('safety_contract_subagent', self._safety_contract_node)
        graph.add_node('safety_contract_gate', self._safety_contract_gate_node)
        graph.add_node('fast_answer', self._fast_answer_node)
        graph.add_node('output_policy_gate', self._output_policy_gate_node)
        graph.add_node('clarification_response', self._clarification_response_node)
        graph.add_node('answer_compose', self._answer_compose_node)
        graph.add_node('answer_text_review', self._answer_text_review_node)
        graph.add_node('invalidate_rewrite', self._invalidate_rewrite_node)
        graph.add_node('answer_rewrite', self._answer_rewrite_node)
        graph.add_node('invalidate_rag_downstream', self._invalidate_rag_downstream_node)
        graph.add_node('invalidate_safety_downstream', self._invalidate_safety_downstream_node)
        graph.add_node('safe_block_response', self._safe_block_response_node)
        graph.add_node('safe_review_fallback', self._safe_review_fallback_node)
        graph.add_node('response_packager', self._response_packager_node)
        graph.add_node('memory_writer', self._memory_writer_node)
        graph.add_node('audit_persistence', self._audit_persistence_node)

        graph.set_entry_point('request_context')
        graph.add_edge('request_context', 'planning_router')
        graph.add_conditional_edges(
            'planning_router',
            self._route_after_planning,
            {
                'prediction_node': 'prediction_node',
                'rag_evidence_subagent': 'rag_evidence_subagent',
                'safety_contract_subagent': 'safety_contract_subagent',
                'answer_compose': 'answer_compose',
                'clarification_response': 'clarification_response',
                'fast_answer': 'fast_answer',
            },
        )
        graph.add_edge('prediction_node', 'prediction_quality_gate')
        graph.add_edge('prediction_quality_gate', 'planning_router')
        graph.add_edge('rag_evidence_subagent', 'evidence_quality_gate')
        graph.add_conditional_edges(
            'evidence_quality_gate',
            self._route_after_quality_gate,
            {
                'pass': 'planning_router',
                'rerun_rag': 'invalidate_rag_downstream',
                'clarification_required': 'clarification_response',
                'max_retry_exceeded': 'safe_review_fallback',
            },
        )
        graph.add_edge('safety_contract_subagent', 'safety_contract_gate')
        graph.add_conditional_edges(
            'safety_contract_gate',
            self._route_after_quality_gate,
            {
                'pass': 'planning_router',
                'rerun_rag': 'invalidate_rag_downstream',
                'rerun_safety': 'invalidate_safety_downstream',
                'clarification_required': 'clarification_response',
                'max_retry_exceeded': 'safe_review_fallback',
            },
        )
        graph.add_edge('fast_answer', 'output_policy_gate')
        graph.add_edge('output_policy_gate', 'response_packager')
        graph.add_edge('clarification_response', 'response_packager')
        graph.add_edge('answer_compose', 'answer_text_review')
        graph.add_conditional_edges(
            'answer_text_review',
            self._route_after_text_review,
            {
                'pass': 'response_packager',
                'rewrite_only': 'invalidate_rewrite',
                'rerun_rag': 'invalidate_rag_downstream',
                'rerun_safety': 'invalidate_safety_downstream',
                'clarification_required': 'clarification_response',
                'block': 'safe_block_response',
                'max_retry_exceeded': 'safe_review_fallback',
            },
        )
        graph.add_edge('invalidate_rewrite', 'answer_rewrite')
        graph.add_edge('answer_rewrite', 'answer_text_review')
        graph.add_edge('invalidate_rag_downstream', 'rag_evidence_subagent')
        graph.add_edge('invalidate_safety_downstream', 'safety_contract_subagent')
        graph.add_edge('safe_block_response', 'response_packager')
        graph.add_edge('safe_review_fallback', 'response_packager')
        graph.add_edge('response_packager', 'memory_writer')
        graph.add_edge('memory_writer', 'audit_persistence')
        graph.add_edge('audit_persistence', END)
        return graph.compile(checkpointer=self.checkpointer)

    def _request_context_node(self, state: ManufacturingAgentState) -> ManufacturingAgentState:
        request = self._request_artifact(state)
        previous_context = self._context_artifact(state)
        previous_memory = self._memory_artifact(state)
        self._clear_turn_scoped_artifacts(state)
        send_request = self._send_request_from_artifact(request)
        output = self.deps.context_subagent.invoke(ContextInput(
            send_request=send_request,
            session_id=request.session_id,
            recent_turns=previous_memory.recent_turns or previous_context.recent_turns,
            rolling_summary=previous_context.rolling_summary,
            recent_turn_routes=previous_memory.recent_turn_routes or previous_context.recent_turn_routes,
            last_answer_memory=previous_memory.last_answer_memory or previous_context.last_answer_memory,
            session_last_process_data=previous_memory.session_last_process_data or previous_context.session_last_process_data,
            warnings=list(previous_context.warnings),
        ))
        context = ContextArtifact(
            recent_turns=list(output.compressed_context.get('recent_turns') or output.rolling_summary and previous_context.recent_turns or previous_context.recent_turns),
            recent_turn_routes=list(previous_context.recent_turn_routes),
            rolling_summary=output.rolling_summary,
            last_answer_memory=previous_memory.last_answer_memory or previous_context.last_answer_memory,
            context_resolution=output.context_resolution,
            context_packs=output.context_packs,
            compressed_context=output.compressed_context,
            user_context=output.user_context,
            turn_context=output.turn_context,
            context_validation_warnings=output.context_validation_warnings,
            ai4i_feature_status=output.ai4i_feature_status,
            turn_process_data=output.turn_process_data,
            previous_turn_process_data=output.previous_turn_process_data,
            session_last_process_data=output.turn_process_data or previous_context.session_last_process_data,
            process_data_reference_policy=output.process_data_reference_policy,
            warnings=output.warnings,
        )
        normalized_request = AgentRequest.model_validate(output.request)
        state['request'] = RequestArtifact(
            user_id=request.user_id,
            session_id=request.session_id,
            question=normalized_request.question,
            original_message=request.original_message,
            normalized_message=normalized_request.question,
            process_data=normalized_request.process_data.model_dump() if normalized_request.process_data else request.process_data,
            inspection_notes=normalized_request.inspection_notes,
            top_k=normalized_request.top_k,
            mode=normalized_request.mode,
            llm_model=normalized_request.llm_model,
        ).model_dump(mode='json')
        state['context'] = context.model_dump(mode='json')
        self._emit_trace(state, trace_step(
            node_id='context.request_context',
            node_name='Request Context',
            node_type='subgraph',
            layer='Context',
            status='success',
            input_summary=request.original_message[:120],
            output_summary=f'followup={output.trace.get("followup")}, ai4i={output.trace.get("ai4i_status")}',
        ))
        return self._return_state(state)

    def _planning_router_node(self, state: ManufacturingAgentState) -> ManufacturingAgentState:
        request = self._request_artifact(state)
        context = self._context_artifact(state)
        previous = self._planning_artifact_or_none(state)
        completed = set(previous.completed_nodes if previous else set())
        if state.get('prediction'):
            completed.add('prediction_node')
        if state.get('evidence'):
            completed.add('rag_evidence_subagent')
        if state.get('safety'):
            completed.add('safety_contract_subagent')

        if previous and previous.selected_path:
            planning = previous
        else:
            planning = self._initial_planning(request, context, state)

        if planning.needs_prediction and not self._agent_request(state).process_data:
            planning = planning.model_copy(update={
                'clarification_required': True,
                'needs_prediction': False,
                'needs_rag': False,
                'needs_safety': False,
                'missing_features': list(context.ai4i_feature_status.get('missing_features') or planning.missing_features),
                'reasoning_summary': context.ai4i_feature_status.get('prediction_skip_reason') or 'AI4I prediction requires all six features.',
            })

        next_node = self._next_planning_node(planning, completed, state)
        planning = planning.model_copy(update={
            'completed_nodes': completed,
            'next_node': next_node,
            'draft_ready': next_node == 'answer_compose',
        })
        state['planning'] = planning.model_dump(mode='json')
        self._emit_trace(state, trace_step(
            node_id='planning.router',
            node_name='Planning Router',
            node_type='router',
            layer='Planning',
            status='success',
            output_summary=f'path={planning.selected_path}, next={next_node}, completed={len(completed)}',
        ))
        return self._return_state(state)

    def _prediction_node(self, state: ManufacturingAgentState) -> ManufacturingAgentState:
        request = self._agent_request(state)
        context = self._context_artifact(state)
        prediction: PredictionResponse | None = None
        warnings: list[str] = []
        skip_reason = None
        if request.process_data:
            prediction = self.deps.prediction_service.predict(request.process_data)
            warnings = list(prediction.input_warnings)
            self._emit_trace(state, trace_step(
                node_id='prediction.ai4i_tool',
                node_name='Prediction Node',
                node_type='tool',
                layer='Prediction',
                status='success',
                output_summary=f'risk={prediction.risk_level}, failure={prediction.predicted_failure}',
            ))
        else:
            skip_reason = context.ai4i_feature_status.get('prediction_skip_reason') or 'missing_ai4i_features'
            self._emit_trace(state, trace_step(
                node_id='prediction.ai4i_tool',
                node_name='Prediction Node',
                node_type='tool',
                layer='Prediction',
                status='skipped',
                output_summary=skip_reason,
            ))
        state['prediction'] = PredictionArtifact(
            called=bool(prediction),
            result=prediction.model_dump(mode='json') if prediction else None,
            skip_reason=skip_reason,
            parsed_features=dict(context.ai4i_feature_status.get('parsed_ai4i_features') or {}),
            missing_features=list(context.ai4i_feature_status.get('missing_features') or []),
            ambiguous_features=list(context.ai4i_feature_status.get('ambiguous_features') or []),
            invalid_features=list(context.ai4i_feature_status.get('invalid_features') or []),
            warnings=warnings,
        ).model_dump(mode='json')
        return self._return_state(state)

    def _prediction_quality_gate_node(self, state: ManufacturingAgentState) -> ManufacturingAgentState:
        planning = self._planning_artifact(state)
        prediction = self._prediction_artifact_or_none(state)
        context = self._context_artifact(state)
        failures: list[ValidationFailure] = []
        ai4i_status = context.ai4i_feature_status or {}
        missing = list(ai4i_status.get('missing_features') or [])
        ambiguous = list(ai4i_status.get('ambiguous_features') or [])
        invalid = list(ai4i_status.get('invalid_features') or [])
        incomplete = bool(missing or ambiguous or invalid or ai4i_status.get('clarification_required'))
        if planning.needs_prediction and not prediction:
            failures.append(ValidationFailure(
                code='prediction_artifact_missing',
                message='Prediction was required but PredictionArtifact was not created.',
                severity='error',
                source='prediction',
            ))
        if prediction and prediction.called and incomplete:
            failures.append(ValidationFailure(
                code='prediction_called_with_incomplete_ai4i_features',
                message='Prediction ran even though AI4I features are incomplete.',
                severity='critical',
                source='prediction',
            ))
        if prediction and not prediction.called and planning.needs_prediction and not (prediction.skip_reason or missing):
            failures.append(ValidationFailure(
                code='prediction_skip_contract_incomplete',
                message='Prediction was skipped without a public skip reason or missing feature list.',
                severity='error',
                source='prediction',
            ))
        if failures:
            planning = planning.model_copy(update={
                'clarification_required': True,
                'needs_prediction': False,
                'needs_rag': False,
                'needs_safety': False,
                'missing_features': missing or planning.missing_features,
                'reasoning_summary': ai4i_status.get('prediction_skip_reason') or 'AI4I feature clarification required.',
            })
            state['planning'] = planning.model_dump(mode='json')
            report = ValidationReport(
                passed=False,
                failures=failures,
                retryable=False,
                next_action='clarification_required',
            )
        else:
            report = ValidationReport.pass_report()
        self._finalize_quality_gate_report(state, gate='prediction_quality_gate', report=report)
        self._emit_trace(state, trace_step(
            node_id='quality.prediction',
            node_name='Prediction Quality Gate',
            node_type='validator',
            layer='Artifact Quality',
            status='success' if report.passed else 'failed',
            output_summary=f'next_action={report.next_action}, failures={len(report.failures)}',
        ))
        return self._return_state(state)

    def _rag_evidence_node(self, state: ManufacturingAgentState) -> ManufacturingAgentState:
        request = self._agent_request(state)
        planning = self._planning_artifact(state)
        plan = self._agent_plan(planning)
        prediction = self._prediction_response(state)
        manufacturing_context = self._manufacturing_context(state)
        output = self.deps.rag_evidence_subagent.invoke(RagEvidenceInput(
            request=request,
            plan=plan,
            prediction=prediction,
            manufacturing_context=manufacturing_context,
            top_k=min(max(request.top_k or 5, 1), AGENT_MAX_RAG_TOP_K),
        ))
        state['evidence'] = output.evidence_artifact.model_dump(mode='json')
        runtime = self._runtime_artifact(state)
        runtime.replan_count += output.replan_count_delta
        state['runtime'] = runtime.model_dump(mode='json')
        self._emit_trace(state, trace_step(
            node_id='rag.rag_evidence_subagent',
            node_name='RAG Evidence SubAgent',
            node_type='subgraph',
            layer='RAG Evidence',
            status='success',
            output_summary=f'selected={len(output.evidence_artifact.documents)}, citations={len(output.evidence_artifact.citations)}',
        ))
        return self._return_state(state)

    def _evidence_quality_gate_node(self, state: ManufacturingAgentState) -> ManufacturingAgentState:
        planning = self._planning_artifact(state)
        evidence = self._evidence_artifact_or_none(state)
        failures: list[ValidationFailure] = []
        if planning.needs_rag:
            if not evidence:
                failures.append(ValidationFailure(
                    code='evidence_artifact_missing',
                    message='RAG evidence was required but EvidenceArtifact is missing.',
                    severity='error',
                    source='citation',
                ))
            else:
                if not evidence.profile:
                    failures.append(ValidationFailure(
                        code='evidence_profile_missing',
                        message='EvidenceArtifact.profile is missing.',
                        severity='error',
                        source='citation',
                    ))
                if not evidence.documents:
                    failures.append(ValidationFailure(
                        code='evidence_documents_missing',
                        message='RAG evidence is required but selected documents are empty.',
                        severity='error',
                        source='citation',
                    ))
                if not evidence.citations:
                    failures.append(ValidationFailure(
                        code='evidence_citations_missing',
                        message='RAG evidence is required but citations are empty.',
                        severity='error',
                        source='citation',
                    ))
                if evidence.required_safety_gates and (not evidence.evidence_covers_required_gates or evidence.missing_gate_evidence):
                    failures.append(ValidationFailure(
                        code='missing_required_gate_evidence',
                        message='Evidence does not cover all required safety gates.',
                        severity='error',
                        source='citation',
                    ))
                if self._generic_evidence_ratio(evidence) >= 1.0 and self._runtime_artifact(state).rag_rerun_attempts > 0:
                    failures.append(ValidationFailure(
                        code='generic_only_evidence_repeated',
                        message='RAG repeatedly selected only generic evidence.',
                        severity='warning',
                        source='citation',
                    ))
        report = ValidationReport.pass_report()
        if failures:
            report = ValidationReport(
                passed=False,
                failures=failures,
                retryable=True,
                next_action='rerun_rag',
                required_reexecution=['rag_evidence', 'safety_contract'],
            )
            if evidence and not self._record_evidence_quality_progress(state, evidence):
                report = report.model_copy(update={'next_action': 'max_retry_exceeded', 'retryable': False})
        elif evidence:
            self._record_evidence_quality_progress(state, evidence)
        report = self._finalize_quality_gate_report(state, gate='evidence_quality_gate', report=report)
        self._emit_trace(state, trace_step(
            node_id='quality.evidence',
            node_name='Evidence Quality Gate',
            node_type='validator',
            layer='Artifact Quality',
            status='success' if report.passed else 'failed',
            output_summary=f'next_action={report.next_action}, failures={len(report.failures)}',
        ))
        return self._return_state(state)

    def _safety_contract_node(self, state: ManufacturingAgentState) -> ManufacturingAgentState:
        request = self._agent_request(state)
        prediction = self._prediction_response(state)
        evidence = self._evidence_artifact(state)
        docs = [RagChunk.model_validate(item) for item in evidence.documents]
        output = self.deps.safety_subagent.invoke(SafetyInput(
            request=request,
            prediction=prediction,
            manufacturing_context=self._manufacturing_context(state),
            retrieved_documents=docs,
            structured_answer_payload={},
        ))
        state['safety'] = output.safety_artifact.model_dump(mode='json')
        self._emit_trace(state, trace_step(
            node_id='safety.safety_contract_subagent',
            node_name='Safety Contract SubAgent',
            node_type='subgraph',
            layer='Safety',
            status='success',
            output_summary=f'gates={len(output.safety_artifact.required_gates)}, checks={len(output.safety_artifact.required_checks)}',
        ))
        return self._return_state(state)

    def _safety_contract_gate_node(self, state: ManufacturingAgentState) -> ManufacturingAgentState:
        planning = self._planning_artifact(state)
        evidence = self._evidence_artifact_or_none(state)
        safety = self._safety_artifact_or_none(state)
        context = self._manufacturing_context(state)
        failures: list[ValidationFailure] = []
        next_action = 'pass'
        required_reexecution: list[str] = []
        if planning.needs_safety:
            evidence_insufficient = bool(evidence and evidence.required_safety_gates and not evidence.evidence_covers_required_gates)
            if planning.needs_rag and (not evidence or evidence_insufficient):
                failures.append(ValidationFailure(
                    code='safety_contract_blocked_by_evidence_quality',
                    message='Safety contract needs better gate-aligned evidence.',
                    severity='error',
                    source='citation',
                ))
                next_action = 'rerun_rag'
                required_reexecution = ['rag_evidence', 'safety_contract']
            elif not safety:
                failures.append(ValidationFailure(
                    code='safety_artifact_missing',
                    message='SafetyArtifact is required but missing.',
                    severity='error',
                    source='safety',
                ))
                next_action = 'rerun_safety'
                required_reexecution = ['safety_contract']
            else:
                context_gate_ids = [gate.gate_id for gate in context.safety_gates]
                if context_gate_ids and not safety.required_gates:
                    failures.append(ValidationFailure(
                        code='safety_required_gates_missing',
                        message='SafetyArtifact.required_gates is empty for a safety-gated request.',
                        severity='error',
                        source='safety',
                    ))
                if (safety.required_gates or context_gate_ids) and not safety.constraints:
                    failures.append(ValidationFailure(
                        code='safety_constraints_missing',
                        message='SafetyArtifact.constraints is empty.',
                        severity='error',
                        source='safety',
                    ))
                if (safety.required_gates or context_gate_ids) and not safety.required_checks:
                    failures.append(ValidationFailure(
                        code='safety_required_checks_missing',
                        message='SafetyArtifact.required_checks is empty.',
                        severity='error',
                        source='safety',
                    ))
                leaked_gate = self._public_guidance_gate_id_leak(safety, context)
                if leaked_gate:
                    failures.append(ValidationFailure(
                        code='safety_public_guidance_gate_id_leak',
                        message=f'SafetyArtifact.public_guidance exposes internal gate id: {leaked_gate}',
                        severity='error',
                        source='debug_leak',
                    ))
                if failures:
                    next_action = 'rerun_safety'
                    required_reexecution = ['safety_contract']
        report = ValidationReport.pass_report()
        if failures:
            report = ValidationReport(
                passed=False,
                failures=failures,
                retryable=next_action in {'rerun_rag', 'rerun_safety'},
                next_action=next_action,  # type: ignore[arg-type]
                required_reexecution=required_reexecution,
            )
        report = self._finalize_quality_gate_report(state, gate='safety_contract_gate', report=report)
        self._emit_trace(state, trace_step(
            node_id='quality.safety_contract',
            node_name='Safety Contract Gate',
            node_type='validator',
            layer='Artifact Quality',
            status='success' if report.passed else 'failed',
            output_summary=f'next_action={report.next_action}, failures={len(report.failures)}',
        ))
        return self._return_state(state)

    def _fast_answer_node(self, state: ManufacturingAgentState) -> ManufacturingAgentState:
        request = self._agent_request(state)
        planning = self._planning_artifact(state)
        context = self._context_artifact(state)
        answer = ''
        warnings: list[str] = []
        route = ['context.request_context', 'planning.router', f'fast.{planning.selected_path}']
        if planning.selected_path == 'fast_concept_answer':
            glossary = self.deps.glossary_answer_service.answer_payload(request.question)
            if glossary:
                formatter_context = {
                    'selected_path': planning.selected_path,
                    'answer_type': planning.answer_type or 'definition',
                    'concept_payload': glossary,
                    'reference_note': self._reference_note(context, glossary.get('term')),
                }
                answer = self.deps.formatter_registry.format('fast_concept_answer', formatter_context)
            else:
                answer = self._lightweight_llm_answer(state)
        elif planning.selected_path == 'general_lightweight_answer':
            answer_context = context.context_packs.get('answer_context') or {}
            memory = answer_context.get('relevant_answer_memory') or {}
            formatter_context = {
                'selected_path': planning.selected_path,
                'answer_type': planning.answer_type or 'explanation',
                'target': memory.get('focus'),
                'followup_target': context.context_resolution.get('followup_target'),
            }
            answer = self.deps.formatter_registry.format('general_lightweight_answer', formatter_context)
        elif planning.selected_path == 'recommended_action_recap':
            answer = self.deps.formatter_registry.format('recommended_action_recap', context.context_packs.get('formatter_context') or {})
        elif planning.selected_path == 'recommended_action_item_explanation':
            answer = self.deps.formatter_registry.format('recommended_action_item_explanation', context.context_packs.get('formatter_context') or {})
        elif planning.selected_path == 'meta_feedback':
            answer = self._meta_feedback_answer(context.last_answer_memory)
        else:
            answer = self._clarification_answer(reason=planning.reasoning_summary or '이 요청은 현재 Agent가 수행할 수 없습니다.')
            warnings.append('제조 설비 제어, 안전 보증, 법적 최종 판단은 수행하지 않습니다.')
        state['response'] = ResponseArtifact(
            answer=self._sanitize_public_answer(answer),
            route=route,
            warnings=self._public_warning_lines(warnings),
            public_citations=[],
            llm_used=bool(self._runtime_artifact(state).usage_records),
            response_type=planning.selected_path,
        ).model_dump(mode='json')
        self._emit_trace(state, trace_step(
            node_id=f'fast.{planning.selected_path}',
            node_name='Fast Answer',
            node_type='formatter',
            layer='Fast Path',
            status='success',
            output_summary=planning.selected_path,
        ))
        return self._return_state(state)

    def _clarification_response_node(self, state: ManufacturingAgentState) -> ManufacturingAgentState:
        planning = self._planning_artifact_or_none(state)
        context = self._context_artifact(state)
        if planning and planning.missing_features:
            answer = self._ai4i_clarification_answer({
                'missing_features': planning.missing_features,
                'ambiguous_features': context.ai4i_feature_status.get('ambiguous_features') or [],
                'invalid_features': context.ai4i_feature_status.get('invalid_features') or [],
                'parsed_ai4i_features': context.ai4i_feature_status.get('parsed_ai4i_features') or {},
            })
        elif context.context_resolution.get('followup_type') == 'ambiguous':
            answer = self._clarification_answer(reason='이전 답변의 어떤 대상을 가리키는지 명확하지 않습니다. 대상이나 항목을 지정해 다시 질문해 주세요.')
        else:
            answer = self._clarification_answer(reason=(planning.reasoning_summary if planning else None) or '요청 의도나 참조 대상을 안정적으로 확정하지 못했습니다.')
        state['response'] = ResponseArtifact(
            answer=self._sanitize_public_answer(answer),
            route=['context.request_context', 'planning.router', 'response.clarification'],
            warnings=[],
            response_type='clarification',
        ).model_dump(mode='json')
        self._emit_trace(state, trace_step(
            node_id='response.clarification',
            node_name='Clarification Response',
            node_type='formatter',
            layer='Response',
            status='success',
            output_summary='clarification response created',
        ))
        return self._return_state(state)

    def _answer_compose_node(self, state: ManufacturingAgentState) -> ManufacturingAgentState:
        result = self.deps.answer_composer.compose_artifact(
            request_artifact=self._request_artifact(state),
            context_artifact=self._context_artifact(state),
            planning_artifact=self._planning_artifact(state),
            prediction_artifact=self._prediction_artifact_or_none(state),
            evidence_artifact=self._evidence_artifact_or_none(state),
            safety_artifact=self._safety_artifact_or_none(state),
            manufacturing_context=self._manufacturing_context(state),
            action_titles=self._recommended_action_titles(state),
            usage_callback=lambda record: self._record_usage(state, record),
            system_prompt=self._answer_system_prompt(),
        )
        if not result.draft:
            runtime = self._runtime_artifact(state)
            runtime.errors.append(result.llm_error or self.deps.llm_service.last_error or 'answer compose failed')
            state['runtime'] = runtime.model_dump(mode='json')
            raise LLMUnavailableError(result.llm_error or self.deps.llm_service.last_error or 'AnswerComposer failed')
        state['draft'] = result.draft.model_dump(mode='json')
        self._emit_trace(state, trace_step(
            node_id='answer.compose',
            node_name='Answer Composer',
            node_type='llm',
            layer='Answer',
            status='success',
            output_summary=f'warnings={len(result.warnings)}',
        ))
        return self._return_state(state)

    def _answer_rewrite_node(self, state: ManufacturingAgentState) -> ManufacturingAgentState:
        runtime = self._runtime_artifact(state)
        failures = [
            ValidationFailure(code='previous_validation_failure', message=item, source='unknown')
            for item in runtime.errors[-5:]
        ]
        validation = ValidationReport(passed=False, failures=failures, retryable=True, next_action='rewrite_only')
        result = self.deps.answer_rewriter.rewrite_artifact(
            request_artifact=self._request_artifact(state),
            context_artifact=self._context_artifact(state),
            planning_artifact=self._planning_artifact(state),
            prediction_artifact=self._prediction_artifact_or_none(state),
            evidence_artifact=self._evidence_artifact_or_none(state),
            safety_artifact=self._safety_artifact_or_none(state),
            manufacturing_context=self._manufacturing_context(state),
            action_titles=self._recommended_action_titles(state),
            validation_report=validation,
            usage_callback=lambda record: self._record_usage(state, record),
            system_prompt=self._answer_system_prompt(),
        )
        if not result.draft:
            raise LLMUnavailableError(result.llm_error or self.deps.llm_service.last_error or 'AnswerRewriter failed')
        state['draft'] = result.draft.model_copy(update={'route': 'answer_rewrite'}).model_dump(mode='json')
        self._emit_trace(state, trace_step(
            node_id='answer.rewrite',
            node_name='Answer Rewriter',
            node_type='llm',
            layer='Answer',
            status='success',
            output_summary='draft rewritten',
        ))
        return self._return_state(state)

    def _answer_text_review_node(self, state: ManufacturingAgentState) -> ManufacturingAgentState:
        draft = self._draft_artifact(state)
        planning = self._planning_artifact(state)
        report = self.deps.answer_review_loop.review(
            draft=draft,
            manufacturing_context=self._manufacturing_context(state),
            evidence_artifact=self._evidence_artifact_or_none(state),
            safety_artifact=self._safety_artifact_or_none(state),
            needs_rag=planning.needs_rag,
        )
        report = self._finalize_quality_gate_report(state, gate='answer_text_review', report=report)
        state['validation'] = report.model_dump(mode='json')
        self._emit_trace(state, trace_step(
            node_id='answer.text_review',
            node_name='Answer Text Review',
            node_type='validator',
            layer='Answer Text Review',
            status='success' if report.passed else 'failed',
            output_summary=f'next_action={report.next_action}, failures={len(report.failures)}',
        ))
        return self._return_state(state)

    def _output_policy_gate_node(self, state: ManufacturingAgentState) -> ManufacturingAgentState:
        response = self._response_artifact_or_none(state) or ResponseArtifact(answer='')
        original = response.answer or ''
        sanitized = self._sanitize_public_answer(original)
        failures: list[ValidationFailure] = []
        if not original.strip():
            failures.append(ValidationFailure(
                code='public_answer_empty',
                message='Public answer is empty.',
                severity='error',
                source='format',
            ))
        if sanitized != original:
            failures.append(ValidationFailure(
                code='public_debug_metadata_removed',
                message='Output policy removed debug/internal metadata from public answer.',
                severity='warning',
                source='debug_leak',
            ))
        if not sanitized.strip():
            sanitized = '응답을 안정적으로 생성하지 못했습니다. 요청을 조금 더 구체화해 다시 보내 주세요.'
            response = response.model_copy(update={'response_type': 'output_policy_fallback', 'safe_fallback_used': True})
        state['response'] = response.model_copy(update={
            'answer': sanitized,
            'warnings': self._public_warning_lines(response.warnings),
        }).model_dump(mode='json')
        report = ValidationReport.pass_report() if not failures else ValidationReport(
            passed=True,
            failures=failures,
            retryable=False,
            next_action='pass',
        )
        self._finalize_quality_gate_report(state, gate='output_policy_gate', report=report)
        self._emit_trace(state, trace_step(
            node_id='quality.output_policy',
            node_name='Output Policy Gate',
            node_type='validator',
            layer='Output Policy',
            status='success' if not failures else 'failed',
            output_summary=f'sanitized={sanitized != original}, failures={len(failures)}',
        ))
        return self._return_state(state)

    def _invalidate_rewrite_node(self, state: ManufacturingAgentState) -> ManufacturingAgentState:
        self._stash_validation_failures(state)
        for key in ['draft', 'validation', 'response']:
            state[key] = None
        self._emit_trace(state, trace_step(
            node_id='invalidate.rewrite',
            node_name='Invalidate Rewrite Downstream',
            node_type='state',
            layer='Review Invalidation',
            status='success',
            output_summary='cleared draft/validation/response',
        ))
        return self._return_state(state)

    def _invalidate_rag_downstream_node(self, state: ManufacturingAgentState) -> ManufacturingAgentState:
        self._stash_validation_failures(state)
        for key in ['evidence', 'safety', 'draft', 'validation', 'response']:
            state[key] = None
        self._remove_completed(state, {'rag_evidence_subagent', 'safety_contract_subagent'})
        self._emit_trace(state, trace_step(
            node_id='invalidate.rag_downstream',
            node_name='Invalidate RAG Downstream',
            node_type='state',
            layer='Review Invalidation',
            status='success',
            output_summary='cleared evidence/safety/draft/validation/response',
        ))
        return self._return_state(state)

    def _invalidate_safety_downstream_node(self, state: ManufacturingAgentState) -> ManufacturingAgentState:
        self._stash_validation_failures(state)
        for key in ['safety', 'draft', 'validation', 'response']:
            state[key] = None
        self._remove_completed(state, {'safety_contract_subagent'})
        self._emit_trace(state, trace_step(
            node_id='invalidate.safety_downstream',
            node_name='Invalidate Safety Downstream',
            node_type='state',
            layer='Review Invalidation',
            status='success',
            output_summary='cleared safety/draft/validation/response',
        ))
        return self._return_state(state)

    def _safe_block_response_node(self, state: ManufacturingAgentState) -> ManufacturingAgentState:
        report = self._validation_report(state)
        answer = self._safe_block_answer(report)
        state['response'] = ResponseArtifact(
            answer=self._sanitize_public_answer(answer),
            route=['answer.text_review', 'response.safe_block'],
            warnings=self._public_warning_lines([failure.message for failure in report.failures]),
            response_type='safe_block',
        ).model_dump(mode='json')
        self._emit_trace(state, trace_step(
            node_id='response.safe_block',
            node_name='Safe Block Response',
            node_type='formatter',
            layer='Response',
            status='success',
            output_summary='blocked unsafe draft',
        ))
        return self._return_state(state)

    def _safe_review_fallback_node(self, state: ManufacturingAgentState) -> ManufacturingAgentState:
        report = self._validation_report(state)
        answer = self._safe_review_fallback_answer(report)
        state['response'] = ResponseArtifact(
            answer=self._sanitize_public_answer(answer),
            route=['answer.text_review', 'response.safe_review_fallback'],
            warnings=self._public_warning_lines([failure.message for failure in report.failures]),
            response_type='safe_review_fallback',
            safe_fallback_used=True,
        ).model_dump(mode='json')
        self._emit_trace(state, trace_step(
            node_id='response.safe_review_fallback',
            node_name='Safe Review Fallback',
            node_type='formatter',
            layer='Response',
            status='success',
            output_summary='max retry exceeded',
        ))
        return self._return_state(state)

    def _response_packager_node(self, state: ManufacturingAgentState) -> ManufacturingAgentState:
        response = self._response_artifact_or_none(state)
        if not response:
            draft = self._draft_artifact(state)
            response = ResponseArtifact(
                answer=self._append_reference_details(draft.text, draft.citations, self._manufacturing_context(state)),
                route=['answer.compose', 'answer.text_review'],
                warnings=draft.warnings,
                public_citations=draft.citations,
                llm_used=draft.llm_used,
                llm_error=draft.llm_error,
            )
        public_citations = response.public_citations or self._citations(state)
        route = response.route or self._trace_route(state)
        state['response'] = response.model_copy(update={
            'answer': self._sanitize_public_answer(self._append_reference_details(response.answer, public_citations, self._manufacturing_context(state))),
            'route': route,
            'warnings': self._public_warning_lines(response.warnings + self._artifact_warnings(state)),
            'public_citations': public_citations,
        }).model_dump(mode='json')
        self._emit_trace(state, trace_step(
            node_id='response.packager',
            node_name='Response Packager',
            node_type='subgraph',
            layer='Response',
            status='success',
            output_summary=f'citations={len(public_citations)}',
        ))
        return self._return_state(state)

    def _memory_writer_node(self, state: ManufacturingAgentState) -> ManufacturingAgentState:
        context = self._context_artifact(state)
        output = self.deps.memory_subagent.invoke(MemoryInput(
            run_id=state['run_id'],
            request=self._request_artifact(state),
            context=context,
            planning=self._planning_artifact_or_none(state),
            prediction=self._prediction_artifact_or_none(state),
            evidence=self._evidence_artifact_or_none(state),
            safety=self._safety_artifact_or_none(state),
            draft=AnswerDraft.model_validate(state['draft']) if state.get('draft') else None,
            response=self._response_artifact_or_none(state) or ResponseArtifact(answer=''),
            runtime=self._runtime_artifact(state),
            user_id=state['user_id'],
        ))
        state['memory'] = MemoryArtifact(
            last_answer_memory=output.last_answer_memory,
            recent_turn_routes=output.recent_turn_routes,
            recent_turns=output.recent_turns,
            session_last_process_data=output.session_last_process_data,
            diagnostics=output.diagnostics,
            warnings=output.warnings,
        ).model_dump(mode='json')
        self._emit_trace(state, trace_step(
            node_id='memory.writer',
            node_name='Memory Writer',
            node_type='memory',
            layer='Memory',
            status='success',
            output_summary=f'focus={output.trace.get("focus") or "none"}',
        ))
        return self._return_state(state)

    def _audit_persistence_node(self, state: ManufacturingAgentState) -> ManufacturingAgentState:
        response = self._agent_response_from_state(state)
        request = self._agent_request(state)
        self.deps.store.append({
            'run_id': response.run_id,
            'user_id': request.user_id,
            'session_id': request.session_id,
            'request': request.model_dump(),
            'response': response.model_dump(),
        })
        state['audit'] = AuditArtifact(persisted=True).model_dump(mode='json')
        self._emit_trace(state, trace_step(
            node_id='audit.persistence',
            node_name='Audit Persistence',
            node_type='storage',
            layer='Audit',
            status='success',
            output_summary='response persisted',
        ))
        return self._return_state(state)

    @staticmethod
    def _route_after_planning(state: ManufacturingAgentState) -> str:
        return PlanningArtifact.model_validate(state['planning']).next_node or 'clarification_response'

    @staticmethod
    def _route_after_text_review(state: ManufacturingAgentState) -> str:
        return ValidationReport.model_validate(state['validation']).next_action

    @staticmethod
    def _route_after_quality_gate(state: ManufacturingAgentState) -> str:
        return ValidationReport.model_validate(state.get('validation') or ValidationReport.pass_report().model_dump()).next_action

    def _initial_planning(self, request_artifact: RequestArtifact, context: ContextArtifact, state: ManufacturingAgentState) -> PlanningArtifact:
        ai4i_status = context.ai4i_feature_status or {}
        if ai4i_status.get('clarification_required'):
            return PlanningArtifact(
                selected_path='ai4i_clarification_required',
                answer_type='ai4i_clarification',
                clarification_required=True,
                missing_features=list(ai4i_status.get('missing_features') or []),
                reasoning_summary=ai4i_status.get('prediction_skip_reason') or 'AI4I feature clarification required.',
            )
        request = self._agent_request_from_artifacts(request_artifact, context)
        usage_records: list[LLMUsageRecord] = []
        gateway = self.deps.intent_gateway.classify(
            request=request,
            user_context=context.user_context,
            usage_callback=lambda record: usage_records.append(record),
        )
        for record in usage_records:
            self._record_usage(state, record)
        selected_path = str(gateway.get('selected_path') or 'supervisor_planning')
        if selected_path != 'supervisor_planning':
            return PlanningArtifact(
                selected_path=selected_path,
                answer_type=str(gateway.get('answer_type') or gateway.get('turn_type') or 'fast_answer'),
                fast_answer_ready=selected_path not in {'unsupported_or_clarification', 'ai4i_clarification_required'},
                clarification_required=selected_path in {'unsupported_or_clarification', 'ai4i_clarification_required'},
                reasoning_summary=str(gateway.get('reason') or ''),
                warnings=[],
            )

        output = self.deps.planning_subagent.invoke(PlanningInput(
            request=request,
            context_resolution=context.context_resolution,
            intent_gateway=gateway,
        ))
        for record in output.usage_records:
            self._record_usage(state, record)
        plan = output.plan
        return PlanningArtifact(
            selected_path='supervisor_planning',
            answer_type=str(gateway.get('answer_type') or plan.intent),
            intent=plan.intent,
            needs_prediction=plan.prediction_required,
            needs_rag=plan.rag_required,
            needs_safety=plan.safety_required or plan.safety_gate_required,
            missing_features=list(ai4i_status.get('missing_features') or []),
            agent_plan=plan.model_dump(mode='json'),
            diagnostic_plan=output.diagnostic_plan,
            route=list(output.route),
            reasoning_summary=plan.rationale,
            warnings=list(output.trace.get('warnings') or []),
        )

    @staticmethod
    def _next_planning_node(planning: PlanningArtifact, completed: set[str], state: ManufacturingAgentState) -> str:
        if planning.clarification_required:
            return 'clarification_response'
        if planning.fast_answer_ready and not (planning.needs_prediction or planning.needs_rag or planning.needs_safety):
            return 'fast_answer'
        if planning.needs_prediction and 'prediction_node' not in completed and not state.get('prediction'):
            return 'prediction_node'
        if planning.needs_rag and 'rag_evidence_subagent' not in completed and not state.get('evidence'):
            return 'rag_evidence_subagent'
        if planning.needs_safety and 'safety_contract_subagent' not in completed and not state.get('safety'):
            return 'safety_contract_subagent'
        return 'answer_compose'

    def _lightweight_llm_answer(self, state: ManufacturingAgentState) -> str:
        request = self._agent_request(state)
        result = self.deps.llm_service.generate_json(
            schema_name='fast_concept_answer',
            schema={'type': 'object', 'properties': {'answer': {'type': 'string'}, 'warnings': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['answer'], 'additionalProperties': True},
            system_prompt='제조/기계 개념을 간결하게 설명하세요. 현재 설비 상태나 안전 상태는 단정하지 마세요.',
            payload={'question': request.question},
            model=request.llm_model,
            operation='fast_concept_answer',
            usage_callback=lambda record: self._record_usage(state, record),
        )
        if not result:
            raise LLMUnavailableError(self.deps.llm_service.last_error or 'Fast answer failed')
        return str(result.get('answer') or '').strip()

    def _record_usage(self, state: ManufacturingAgentState, record: LLMUsageRecord) -> None:
        runtime = self._runtime_artifact(state)
        runtime.usage_records.append(record.model_dump(mode='json'))
        state['runtime'] = runtime.model_dump(mode='json')
        self._emit_trace(state, trace_step(
            node_id='audit.usage_meter',
            node_name='Usage Meter',
            node_type='metric',
            layer='Audit',
            status='success',
            output_summary=f'{record.operation}: input={record.input_tokens}, output={record.output_tokens}, cost=${record.estimated_cost_usd:.6f}',
        ))

    def _agent_response_from_state(self, state: dict[str, Any]) -> AgentResponse:
        request = self._request_artifact(state)
        response = self._response_artifact_or_none(state) or ResponseArtifact(answer='')
        prediction_artifact = self._prediction_artifact_or_none(state)
        prediction = self._prediction_response(state)
        planning = self._planning_artifact_or_none(state)
        runtime = self._runtime_artifact(state)
        context = self._context_artifact(state)
        return AgentResponse(
            run_id=state['run_id'],
            user_id=request.user_id,
            session_id=request.session_id,
            route=response.route,
            answer=self._sanitize_public_answer(response.answer),
            prediction=prediction,
            manufacturing_context=self._manufacturing_context(state),
            retrieved_documents=[RagChunk.model_validate(item) for item in self._evidence_artifact(state).documents],
            safety_guidance=self._safety_artifact(state).public_guidance,
            report=response.report,
            citations=response.public_citations,
            warnings=self._public_warning_lines(response.warnings),
            trace=to_agent_trace_steps(runtime.trace),
            saved=response.saved,
            plan=AgentPlan.model_validate(planning.agent_plan) if planning and planning.agent_plan else None,
            llm_used=response.llm_used or bool(runtime.usage_records),
            llm_provider=LLM_PROVIDER,
            llm_model=request.llm_model or self.deps.llm_service.model,
            llm_usage=self._usage_summary(runtime.usage_records),
            llm_error=response.llm_error,
            context_used=self._context_metadata(context, request.user_id),
            prediction_called=bool(prediction_artifact and prediction_artifact.called),
            prediction_skip_reason=None if prediction_artifact and prediction_artifact.called else self._public_prediction_skip_reason(prediction_artifact, context),
            missing_features=[] if prediction_artifact and prediction_artifact.called else list((prediction_artifact.missing_features if prediction_artifact else context.ai4i_feature_status.get('missing_features') or [])),
            ambiguous_features=[] if prediction_artifact and prediction_artifact.called else list((prediction_artifact.ambiguous_features if prediction_artifact else context.ai4i_feature_status.get('ambiguous_features') or [])),
            parsed_ai4i_features=dict(prediction_artifact.parsed_features if prediction_artifact else context.ai4i_feature_status.get('parsed_ai4i_features') or {}),
        )

    def _agent_request(self, state: dict[str, Any]) -> AgentRequest:
        return self._agent_request_from_artifacts(self._request_artifact(state), self._context_artifact(state))

    @staticmethod
    def _agent_request_from_artifacts(request: RequestArtifact, context: ContextArtifact) -> AgentRequest:
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

    @staticmethod
    def _send_request_from_artifact(request: RequestArtifact) -> AgentSendRequest:
        return AgentSendRequest.model_validate({
            'user_id': request.user_id,
            'session_id': request.session_id,
            'message': request.original_message,
            'process_data': request.process_data,
            'inspection_notes': request.inspection_notes,
            'top_k': request.top_k or 5,
            'mode': request.mode,
            'llm_model': request.llm_model,
        })

    def _agent_plan(self, planning: PlanningArtifact) -> AgentPlan:
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

    def _prediction_response(self, state: dict[str, Any]) -> PredictionResponse | None:
        prediction = self._prediction_artifact_or_none(state)
        if not prediction or not prediction.result:
            return None
        return PredictionResponse.model_validate(prediction.result)

    def _manufacturing_context(self, state: dict[str, Any]) -> ManufacturingContext:
        return self.deps.domain_service.build_context(
            self._agent_request(state),
            self._prediction_response(state),
            doc_count=len(self._evidence_artifact(state).documents),
        )

    @staticmethod
    def _request_artifact(state: dict[str, Any]) -> RequestArtifact:
        return RequestArtifact.model_validate(state['request'])

    @staticmethod
    def _context_artifact(state: dict[str, Any]) -> ContextArtifact:
        return ContextArtifact.model_validate(state.get('context') or {})

    @staticmethod
    def _planning_artifact(state: dict[str, Any]) -> PlanningArtifact:
        return PlanningArtifact.model_validate(state['planning'])

    @staticmethod
    def _planning_artifact_or_none(state: dict[str, Any]) -> PlanningArtifact | None:
        return PlanningArtifact.model_validate(state['planning']) if state.get('planning') else None

    @staticmethod
    def _prediction_artifact_or_none(state: dict[str, Any]) -> PredictionArtifact | None:
        return PredictionArtifact.model_validate(state['prediction']) if state.get('prediction') else None

    @staticmethod
    def _evidence_artifact(state: dict[str, Any]) -> EvidenceArtifact:
        return EvidenceArtifact.model_validate(state.get('evidence') or {})

    @staticmethod
    def _evidence_artifact_or_none(state: dict[str, Any]) -> EvidenceArtifact | None:
        return EvidenceArtifact.model_validate(state['evidence']) if state.get('evidence') else None

    @staticmethod
    def _safety_artifact(state: dict[str, Any]) -> SafetyArtifact:
        return SafetyArtifact.model_validate(state.get('safety') or {})

    @staticmethod
    def _safety_artifact_or_none(state: dict[str, Any]) -> SafetyArtifact | None:
        return SafetyArtifact.model_validate(state['safety']) if state.get('safety') else None

    @staticmethod
    def _draft_artifact(state: dict[str, Any]) -> AnswerDraft:
        return AnswerDraft.model_validate(state['draft'])

    @staticmethod
    def _validation_report(state: dict[str, Any]) -> ValidationReport:
        return ValidationReport.model_validate(state.get('validation') or {'passed': False, 'next_action': 'max_retry_exceeded'})

    @staticmethod
    def _response_artifact_or_none(state: dict[str, Any]) -> ResponseArtifact | None:
        return ResponseArtifact.model_validate(state['response']) if state.get('response') else None

    @staticmethod
    def _memory_artifact(state: dict[str, Any]) -> MemoryArtifact:
        return MemoryArtifact.model_validate(state.get('memory') or {})

    @staticmethod
    def _runtime_artifact(state: dict[str, Any]) -> RuntimeArtifact:
        return RuntimeArtifact.model_validate(state.get('runtime') or {})

    def _recommended_action_titles(self, state: dict[str, Any]) -> list[str]:
        safety = self._safety_artifact_or_none(state)
        prediction = self._prediction_response(state)
        payload_actions = (safety.structured_payload.get('recommended_actions') if safety else None) or []
        titles: list[str] = []
        for item in payload_actions:
            if isinstance(item, dict):
                title = item.get('title') or item.get('text') or item.get('action')
            else:
                title = str(item)
            if title:
                titles.append(str(title))
        if prediction:
            titles.extend(prediction.recommended_actions)
        return list(dict.fromkeys(titles))

    def _artifact_warnings(self, state: dict[str, Any]) -> list[str]:
        warnings: list[str] = []
        warnings.extend(self._context_artifact(state).warnings)
        prediction = self._prediction_artifact_or_none(state)
        evidence = self._evidence_artifact_or_none(state)
        safety = self._safety_artifact_or_none(state)
        draft = AnswerDraft.model_validate(state['draft']) if state.get('draft') else None
        for artifact in [prediction, evidence, safety, draft]:
            if artifact:
                warnings.extend(getattr(artifact, 'warnings', []) or [])
        return list(dict.fromkeys(warnings))

    def _citations(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        if state.get('draft'):
            draft = self._draft_artifact(state)
            if draft.citations:
                return draft.citations
        return self._evidence_artifact(state).citations

    @staticmethod
    def _public_prediction_skip_reason(prediction: PredictionArtifact | None, context: ContextArtifact) -> str | None:
        reason = (prediction.skip_reason if prediction else None) or context.ai4i_feature_status.get('prediction_skip_reason')
        if reason in {'missing_ai4i_features', 'ambiguous_ai4i_features', 'invalid_ai4i_features'}:
            return reason
        status = context.ai4i_feature_status or {}
        if status.get('ambiguous_features'):
            return 'ambiguous_ai4i_features'
        if status.get('invalid_features'):
            return 'invalid_ai4i_features'
        if status.get('missing_features'):
            return 'missing_ai4i_features'
        return None

    def _trace_route(self, state: dict[str, Any]) -> list[str]:
        return [str(item.get('node_id') or item.get('node_name')) for item in self._runtime_artifact(state).trace if item.get('node_id') or item.get('node_name')]

    def _stash_validation_failures(self, state: ManufacturingAgentState) -> None:
        report = self._validation_report(state)
        runtime = self._runtime_artifact(state)
        runtime.errors.extend([f'{failure.source}:{failure.code}: {failure.message}' for failure in report.failures])
        state['runtime'] = runtime.model_dump(mode='json')

    def _remove_completed(self, state: ManufacturingAgentState, nodes: set[str]) -> None:
        planning = self._planning_artifact(state)
        planning.completed_nodes.difference_update(nodes)
        state['planning'] = planning.model_dump(mode='json')

    @staticmethod
    def _clear_turn_scoped_artifacts(state: ManufacturingAgentState) -> None:
        for key in TURN_SCOPED_ARTIFACT_KEYS:
            state[key] = None

    def _finalize_quality_gate_report(self, state: ManufacturingAgentState, *, gate: str, report: ValidationReport) -> ValidationReport:
        runtime = self._runtime_artifact(state)
        if not report.passed and report.next_action in {'rewrite_only', 'rerun_rag', 'rerun_safety'}:
            if not self._retry_budget_allows(runtime, report.next_action):
                report = report.model_copy(update={'next_action': 'max_retry_exceeded', 'retryable': False})
            else:
                self._increment_retry_budget(runtime, report.next_action)
        runtime.quality_gate_reports.append({
            'gate': gate,
            'passed': report.passed,
            'next_action': report.next_action,
            'failure_codes': [failure.code for failure in report.failures],
            'retryable': report.retryable,
        })
        runtime.quality_gate_reports = runtime.quality_gate_reports[-20:]
        state['runtime'] = runtime.model_dump(mode='json')
        state['validation'] = report.model_dump(mode='json')
        return report

    @staticmethod
    def _retry_budget_allows(runtime: RuntimeArtifact, action: str) -> bool:
        total = runtime.rewrite_attempts + runtime.rag_rerun_attempts + runtime.safety_rerun_attempts
        if total >= MAX_TOTAL_ATTEMPTS:
            return False
        if action == 'rewrite_only':
            return runtime.rewrite_attempts < MAX_REWRITE_ATTEMPTS
        if action == 'rerun_rag':
            return runtime.rag_rerun_attempts < MAX_RAG_RERUN_ATTEMPTS
        if action == 'rerun_safety':
            return runtime.safety_rerun_attempts < MAX_SAFETY_RERUN_ATTEMPTS
        return True

    @staticmethod
    def _increment_retry_budget(runtime: RuntimeArtifact, action: str) -> None:
        if action == 'rewrite_only':
            runtime.rewrite_attempts += 1
        elif action == 'rerun_rag':
            runtime.rag_rerun_attempts += 1
        elif action == 'rerun_safety':
            runtime.safety_rerun_attempts += 1
        runtime.review_iteration += 1
        runtime.replan_count += 1

    def _record_evidence_quality_progress(self, state: ManufacturingAgentState, evidence: EvidenceArtifact) -> bool:
        runtime = self._runtime_artifact(state)
        current = self._evidence_quality_signature(evidence)
        previous = self._last_evidence_quality_signature(runtime)
        improved = self._evidence_quality_improved(previous, current)
        runtime.evidence_signatures.append(json.dumps(current, ensure_ascii=False, sort_keys=True))
        runtime.evidence_signatures = runtime.evidence_signatures[-8:]
        runtime.previous_missing_gate_evidence = list(evidence.missing_gate_evidence)
        state['runtime'] = runtime.model_dump(mode='json')
        return improved

    @staticmethod
    def _last_evidence_quality_signature(runtime: RuntimeArtifact) -> dict[str, Any] | None:
        if not runtime.evidence_signatures:
            return None
        try:
            return dict(json.loads(runtime.evidence_signatures[-1]))
        except Exception:
            return None

    def _evidence_quality_signature(self, evidence: EvidenceArtifact) -> dict[str, Any]:
        selected_ids = evidence.selected_source_ids or [
            str(item.get('doc_id') or item.get('chunk_id') or item.get('title') or '')
            for item in evidence.documents
            if item
        ]
        generic_ratio = self._generic_evidence_ratio(evidence)
        equipment_specific_count = sum(1 for item in evidence.documents if self._is_equipment_specific_evidence(item))
        return {
            'profile': evidence.profile,
            'queries': evidence.queries,
            'selected_source_ids': list(dict.fromkeys([item for item in selected_ids if item])),
            'citation_count': len(evidence.citations),
            'missing_gate_evidence': list(evidence.missing_gate_evidence),
            'evidence_covers_required_gates': evidence.evidence_covers_required_gates,
            'generic_ratio': generic_ratio,
            'equipment_specific_count': equipment_specific_count,
        }

    @staticmethod
    def _evidence_quality_improved(previous: dict[str, Any] | None, current: dict[str, Any]) -> bool:
        if not previous:
            return True
        if int(current.get('citation_count') or 0) > int(previous.get('citation_count') or 0):
            return True
        if len(current.get('missing_gate_evidence') or []) < len(previous.get('missing_gate_evidence') or []):
            return True
        if current.get('evidence_covers_required_gates') and not previous.get('evidence_covers_required_gates'):
            return True
        if set(current.get('selected_source_ids') or []) != set(previous.get('selected_source_ids') or []):
            return True
        if int(current.get('equipment_specific_count') or 0) > int(previous.get('equipment_specific_count') or 0):
            return True
        if float(current.get('generic_ratio') or 0.0) < float(previous.get('generic_ratio') or 0.0):
            return True
        if current.get('profile') != previous.get('profile'):
            return True
        if current.get('queries') != previous.get('queries'):
            return True
        return False

    @staticmethod
    def _generic_evidence_ratio(evidence: EvidenceArtifact | None) -> float:
        if not evidence or not evidence.documents:
            return 0.0
        generic = sum(1 for item in evidence.documents if RootManufacturingGraph._is_generic_evidence(item))
        return generic / max(len(evidence.documents), 1)

    @staticmethod
    def _is_generic_evidence(item: dict[str, Any]) -> bool:
        blob = ' '.join(str(item.get(key) or '') for key in ['doc_type', 'source', 'title', 'document_title', 'retrieval_scope']).lower()
        if any(token in blob for token in ['generic', 'industry', 'general']):
            return True
        source = str(item.get('source') or '').lower()
        doc_type = str(item.get('doc_type') or '').lower()
        return source in {'osha', 'kosha'} and not any(token in doc_type for token in ['maintenance', 'troubleshooting', 'procedure'])

    @staticmethod
    def _is_equipment_specific_evidence(item: dict[str, Any]) -> bool:
        blob = ' '.join(str(item.get(key) or '') for key in ['doc_type', 'source', 'title', 'document_title', 'retrieval_scope']).lower()
        return any(token in blob for token in ['haas', 'cnc', 'lathe', 'mill', 'spindle', 'tool', 'maintenance', 'troubleshooting', 'procedure'])

    @staticmethod
    def _public_guidance_gate_id_leak(safety: SafetyArtifact, context: ManufacturingContext) -> str | None:
        guidance = (safety.public_guidance or '').lower()
        if not guidance:
            return None
        for gate in context.safety_gates:
            gate_id = (gate.gate_id or '').strip()
            if gate_id and gate_id.lower() in guidance:
                return gate_id
        return None

    @staticmethod
    def _usage_summary(records: list[dict[str, Any] | LLMUsageRecord]) -> LLMUsageSummary:
        normalized = [item if isinstance(item, LLMUsageRecord) else LLMUsageRecord.model_validate(item) for item in (records or [])]
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
    def _public_warning_lines(warnings: list[str]) -> list[str]:
        blocked = ['응답 생성 시 금지 표현', 'forbidden_agent_actions', '금지 표현:']
        normalized: list[str] = []
        seen: set[str] = set()
        for warning in warnings or []:
            text = ' '.join(str(warning or '').split())
            if not text or any(token in text for token in blocked):
                continue
            key = RootManufacturingGraph._warning_key(text)
            if key not in seen:
                normalized.append(text)
                seen.add(key)
        return normalized[:5]

    @staticmethod
    def _warning_key(text: str) -> str:
        lower = text.lower()
        if ('학습 데이터' in text or 'training data' in lower) and ('공구 마모' in text or 'tool wear' in lower):
            return 'tool_wear_training_range'
        if ('학습 데이터' in text or 'training data' in lower) and ('토크' in text or 'torque' in lower):
            return 'torque_training_range'
        if '실제 설비 제어' in text or '자동 정지' in text or '법적 안전 판단' in text:
            return 'agent_disclaimer'
        if 'LOTO/방호 절차' in text:
            return 'conditional_loto'
        return text

    def _clarification_answer(self, *, reason: str, missing_info: str | None = None) -> str:
        return self.deps.formatter_registry.format('clarification', {
            'public_reason': self._safe_public_reason(reason),
            'missing_info': missing_info,
        })

    @staticmethod
    def _ai4i_clarification_answer(status: dict[str, Any]) -> str:
        missing = list(status.get('missing_features') or [])
        ambiguous = list(status.get('ambiguous_features') or [])
        invalid = list(status.get('invalid_features') or [])
        parsed = status.get('parsed_ai4i_features') or {}
        lines = [
            'AI4I 예측에 필요한 입력이 아직 완전하지 않습니다.',
            '',
            '예측을 실행하려면 아래 6개 feature가 모두 유효해야 합니다.',
        ]
        if missing:
            lines.extend(['', '누락된 값', *[f'- {item}' for item in missing]])
        if ambiguous:
            lines.extend(['', '단위나 해석이 불명확한 값', *[f'- {item}' for item in ambiguous]])
        if invalid:
            lines.extend(['', '유효 범위를 벗어난 값', *[f'- {item}' for item in invalid]])
        if parsed:
            lines.extend(['', '현재까지 인식한 값: ' + ', '.join(f'{key}={value}' for key, value in parsed.items())])
        lines.extend([
            '',
            '다음 형식으로 다시 보내 주세요.',
            'Type=L/M/H, Air temperature=300.2K, Process temperature=309.0K, Rotational speed=1480rpm, Torque=34Nm, Tool wear=235min',
        ])
        return '\n'.join(lines)

    @staticmethod
    def _meta_feedback_answer(memory: dict[str, Any]) -> str:
        focus = memory.get('focus')
        if focus:
            return f'맞습니다. 이 경우 "이걸"은 직전 대화의 "{focus}"를 기준으로 해석하는 것이 자연스럽습니다.'
        return '맞습니다. 이런 경우에는 직전 대화의 핵심 주제를 먼저 참조해서 지시어를 해석해야 합니다.'

    @staticmethod
    def _sanitize_public_answer(answer: str) -> str:
        blocked = [
            'resolved=false', 'resolved_target', 'question_kind', 'context_policy', 'rag_contexts',
            'safety_gates', 'recent_runs', 'similar_runs', 'audit_notes', 'action_plan',
            'current turn information', 'current_turn', 'internal_reason', 'badrequesterror',
            'invalid_json_schema', 'run_id', 'run id', 'llm usage', 'model=', 'llm_model',
            'tokens', 'token', 'cost', 'calls=', 'replans', 're-plans', 'trace', 'raw error',
            'raw_score', 'chunk_id', 'safety_gate_id', 'gate id', 'forbidden_agent_actions',
        ]
        lines = []
        for line in (answer or '').splitlines():
            lowered = line.lower()
            if any(token.lower() in lowered for token in blocked):
                continue
            lines.append(line)
        return '\n'.join(lines).strip()

    @staticmethod
    def _safe_public_reason(reason: str) -> str:
        text = str(reason or '').strip()
        internal_tokens = ['badrequesterror', 'invalid_json_schema', 'stack trace', 'traceback', 'valueerror', 'schema for response_format', 'additionalproperties', 'raw exception']
        if not text or any(token in text.lower() for token in internal_tokens):
            return '요청 의도나 참조 대상을 안정적으로 확정하지 못했습니다.'
        return text

    @staticmethod
    def _answer_system_prompt() -> str:
        return (
            '당신은 제조 품질/설비 문서 기반 AI Agent입니다. '
            '현재 현장 판단, 고장 확률, 안전 판단은 반드시 제공된 prediction, manufacturing_context, rag_contexts, actions 안의 사실에 근거하세요. '
            'Safety contract의 required checks는 누락하지 말고 forbidden actions는 수행했다고 말하지 마세요. '
            'prediction이 없고 RAG/safety 절차만 묻는 질문은 AI4I, 고장 확률, TWF/OSF/HDF/PWF 확률을 쓰지 마세요. '
            'run id, model, token, cost, call count, trace, chunk id, safety gate id 같은 debug 정보를 답변 본문에 쓰지 마세요. '
            '문서 인용은 payload.citation_references의 label만 사용하고, label을 임의로 만들지 마세요. '
            'safety gate id는 내부 metadata이므로 자연어 안전 확인 항목으로만 반영하세요. '
            'report 필드는 항상 null로 두세요.'
        )

    @staticmethod
    def _append_reference_details(answer: str, citations: list[dict[str, Any]], manufacturing_context: ManufacturingContext | None = None) -> str:
        citation_lines = RootManufacturingGraph._citation_reference_lines(citations)
        if citation_lines and '참조 문서' not in (answer or ''):
            return answer.rstrip() + '\n\n참조 문서\n' + '\n'.join(citation_lines)
        return answer

    @staticmethod
    def _citation_reference_lines(citations: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        seen: set[str] = set()
        for index, citation in enumerate(citations or [], start=1):
            label = str(citation.get('label') or f'ref-{index}')
            source = str(citation.get('source') or 'unknown')
            title = str(citation.get('title') or citation.get('document') or citation.get('doc_id') or 'Untitled document')
            key = f'{label}:{source}:{title}'
            if key in seen:
                continue
            seen.add(key)
            details = [f'source={source}']
            if citation.get('doc_type'):
                details.append(f'doc_type={citation["doc_type"]}')
            if citation.get('doc_id'):
                details.append(f'doc_id={citation["doc_id"]}')
            lines.append(f'- [{label}] {title} ({", ".join(details)})')
            if len(lines) >= 3:
                break
        return lines

    @staticmethod
    def _safe_block_answer(report: ValidationReport) -> str:
        details = [failure.message for failure in report.failures if failure.severity in {'error', 'critical'}]
        lines = [
            '요청한 답변 초안에 안전상 허용할 수 없는 표현이나 조치가 포함되어 차단했습니다.',
            '',
            '설비 자동 제어, 안전장치 우회, LOTO 생략, 무자격 정비를 지시하거나 수행한 것처럼 답변할 수 없습니다.',
        ]
        if details:
            lines.extend(['', '차단 사유', *[f'- {item}' for item in details[:3]]])
        lines.extend(['', '안전관리자 또는 승인된 담당자의 절차에 따라 작업 조건을 다시 확인해 주세요.'])
        return '\n'.join(lines)

    @staticmethod
    def _safe_review_fallback_answer(report: ValidationReport) -> str:
        lines = [
            '답변 검증을 통과하는 최종 문장을 안정적으로 생성하지 못했습니다.',
            '',
            '현재 답변은 문서 근거와 안전 확인 항목을 보수적으로 재검토해야 합니다.',
            '설비 제어, 공구 교체, 커버 개방, 회전부 접근 같은 물리 작업은 승인된 담당자와 현장 안전 절차를 먼저 확인해 주세요.',
        ]
        if report.failures:
            lines.extend(['', '검증 실패 요약', *[f'- {failure.message}' for failure in report.failures[:3]]])
        return '\n'.join(lines)

    @staticmethod
    def _reference_note(context: ContextArtifact, term: str | None = None) -> str | None:
        resolution = context.context_resolution
        if not resolution.get('is_followup'):
            return None
        target = resolution.get('followup_target') or term
        if not target:
            return None
        if resolution.get('followup_type') in {'previous_concept', 'previous_answer_reason', 'previous_claim'}:
            return f'직전 답변의 "{target}"를 기준으로 답변하겠습니다.'
        return None

    @staticmethod
    def _context_metadata(context: ContextArtifact, user_id: str | None = None) -> dict | None:
        if not context.user_context:
            return None
        return {
            'user_id': user_id,
            'session_id': (context.user_context.get('session_context') or {}).get('session_id'),
            'recent_runs_count': len(context.user_context.get('recent_runs') or []),
            'similar_runs_count': len(context.user_context.get('similar_runs') or []),
            'memories_count': len(context.user_context.get('long_term_memory') or []),
            'estimated_context_tokens': context.user_context.get('estimated_context_tokens', 0),
            'context_policy': context.user_context.get('context_policy') or {},
            'process_data_reference_policy': context.process_data_reference_policy,
            'ai4i_feature_status': context.ai4i_feature_status,
            'context_resolution': context.context_resolution,
            'context_validation_warnings': context.context_validation_warnings,
        }

    def _emit_trace(self, state: dict[str, Any], step: dict[str, Any]) -> None:
        runtime = self._runtime_artifact(state)
        runtime.trace.append(step)
        state['runtime'] = runtime.model_dump(mode='json')
        if self._progress_callback:
            self._progress_callback(to_agent_trace_steps([step])[0])

    @staticmethod
    def _thread_config(*, user_id: str, session_id: str) -> dict[str, Any]:
        return {'configurable': {'thread_id': build_thread_id(user_id=user_id, session_id=session_id), 'user_id': user_id, 'session_id': session_id}}

    def _checkpoint_values(self, *, user_id: str, session_id: str) -> dict[str, Any]:
        try:
            snapshot = self.graph.get_state(self._thread_config(user_id=user_id, session_id=session_id))
        except Exception:
            return {}
        values = dict(snapshot.values or {})
        if values.get('state_schema_version') != STATE_SCHEMA_VERSION:
            return {}
        return values

    @staticmethod
    def _return_state(state: dict[str, Any]) -> dict[str, Any]:
        clean = RootManufacturingGraph._sanitize_value(state)
        clean['state_schema_version'] = STATE_SCHEMA_VERSION
        return clean

    @staticmethod
    def _sanitize_value(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return RootManufacturingGraph._sanitize_value(value.model_dump(mode='json'))
        if isinstance(value, dict):
            return {str(k): RootManufacturingGraph._sanitize_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [RootManufacturingGraph._sanitize_value(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)
