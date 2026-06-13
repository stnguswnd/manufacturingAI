from __future__ import annotations

from dataclasses import dataclass

from app.agent.answer_composer import AnswerComposer, AnswerRewriter
from app.agent.answer_review import AnswerReviewLoop
from app.agent.context import AnswerMemoryWriter, ContextCompressor, ContextPackBuilder, ContextResolver, ContextValidator
from app.agent.context_subagent import ContextDeps, ContextSubAgent
from app.agent.formatters import FormatterRegistry
from app.agent.heavy import CitationBuilder, DiagnosticPlanner, EvidenceFilter, EvidenceGrader, RagQueryPlanner, RecommendationBuilder, SafetyGateBuilder, StructuredAnswerPayloadBuilder
from app.agent.heavy.rag_query_planner import RagFanoutPolicy
from app.agent.memory_subagent import MemoryDeps, MemorySubAgent
from app.agent.planning_subagent import PlanningDeps, PlanningSubAgent
from app.agent.rag_evidence import RagEvidenceDeps, RagEvidenceSubAgent
from app.agent.safety_subagent import SafetyDeps, SafetySubAgent
from app.agent.validators import CitationVerifier, SafetyCritic
from app.services.context_service import ContextService
from app.services.domain_service import DomainKnowledgeService
from app.services.glossary_answer_service import GlossaryAnswerService
from app.services.intent_classifier_service import IntentClassifierService
from app.services.intent_gateway_service import IntentGatewayService
from app.services.llm_service import LLMService
from app.services.memory_service import MemoryService
from app.services.prediction_service import PredictionService
from app.services.rag_service import RagService
from app.services.safety_validation_service import SafetyValidationService
from app.services.supervisor_service import SupervisorService
from app.services.user_service import UserService
from app.storage.sqlite_store import SQLiteStore


@dataclass(frozen=True)
class AgentRuntimeDeps:
    store: SQLiteStore
    prediction_service: PredictionService
    domain_service: DomainKnowledgeService
    safety_validator: SafetyValidationService
    llm_service: LLMService
    intent_classifier: IntentClassifierService
    intent_gateway: IntentGatewayService
    glossary_answer_service: GlossaryAnswerService
    formatter_registry: FormatterRegistry
    context_subagent: ContextSubAgent
    diagnostic_planner: DiagnosticPlanner
    planning_subagent: PlanningSubAgent
    citation_builder: CitationBuilder
    rag_evidence_subagent: RagEvidenceSubAgent
    recommendation_builder: RecommendationBuilder
    safety_subagent: SafetySubAgent
    memory_subagent: MemorySubAgent
    structured_payload_builder: StructuredAnswerPayloadBuilder
    answer_composer: AnswerComposer
    answer_rewriter: AnswerRewriter
    answer_review_loop: AnswerReviewLoop

    @classmethod
    def from_services(
        cls,
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
    ) -> 'AgentRuntimeDeps':
        classifier = intent_classifier or IntentClassifierService(llm_service)
        glossary = GlossaryAnswerService()
        formatter = FormatterRegistry()
        context_subagent = ContextSubAgent(ContextDeps(
            user_service=user_service,
            context_service=context_service,
            context_resolver=ContextResolver(),
            context_pack_builder=ContextPackBuilder(),
            context_compressor=ContextCompressor(max_recent_turns=5),
            context_validator=ContextValidator(),
        ))
        diagnostic_planner = DiagnosticPlanner(SupervisorService(llm_service))
        planning_subagent = PlanningSubAgent(PlanningDeps(diagnostic_planner=diagnostic_planner))
        citation_builder = CitationBuilder()
        rag_evidence_subagent = RagEvidenceSubAgent(RagEvidenceDeps(
            query_planner=RagQueryPlanner(),
            fanout_policy=RagFanoutPolicy(),
            rag_service=rag_service,
            evidence_filter=EvidenceFilter(),
            evidence_grader=EvidenceGrader(),
            citation_builder=citation_builder,
            domain_service=domain_service,
        ))
        recommendation_builder = RecommendationBuilder()
        safety_subagent = SafetySubAgent(SafetyDeps(
            domain_service=domain_service,
            recommendation_builder=recommendation_builder,
            safety_gate_builder=SafetyGateBuilder(),
        ))
        memory_subagent = MemorySubAgent(MemoryDeps(
            answer_memory_writer=AnswerMemoryWriter(),
            memory_service=memory_service,
            domain_service=domain_service,
        ))
        payload_builder = StructuredAnswerPayloadBuilder()
        answer_composer = AnswerComposer(llm_service, payload_builder)
        return cls(
            store=store,
            prediction_service=prediction_service,
            domain_service=domain_service,
            safety_validator=safety_validator,
            llm_service=llm_service,
            intent_classifier=classifier,
            intent_gateway=IntentGatewayService(intent_classifier=classifier),
            glossary_answer_service=glossary,
            formatter_registry=formatter,
            context_subagent=context_subagent,
            diagnostic_planner=diagnostic_planner,
            planning_subagent=planning_subagent,
            citation_builder=citation_builder,
            rag_evidence_subagent=rag_evidence_subagent,
            recommendation_builder=recommendation_builder,
            safety_subagent=safety_subagent,
            memory_subagent=memory_subagent,
            structured_payload_builder=payload_builder,
            answer_composer=answer_composer,
            answer_rewriter=AnswerRewriter(answer_composer),
            answer_review_loop=AnswerReviewLoop(
                citation_verifier=CitationVerifier(),
                safety_critic=SafetyCritic(safety_validator),
            ),
        )
