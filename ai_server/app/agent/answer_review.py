from __future__ import annotations

from app.agent.artifacts import AnswerDraft, EvidenceArtifact, SafetyArtifact, ValidationFailure, ValidationReport
from app.agent.validators import CitationVerifier, SafetyCritic
from app.schemas.domain import ManufacturingContext


class AnswerReviewLoop:
    """Runs bounded deterministic answer review and chooses the next action."""

    ACTION_PRIORITY = {
        'block': 5,
        'clarification_required': 4,
        'rerun_rag': 3,
        'rerun_safety': 2,
        'rewrite_only': 1,
        'pass': 0,
    }

    def __init__(self, *, citation_verifier: CitationVerifier, safety_critic: SafetyCritic):
        self.citation_verifier = citation_verifier
        self.safety_critic = safety_critic

    def review(
        self,
        *,
        draft: AnswerDraft,
        manufacturing_context: ManufacturingContext,
        evidence_artifact: EvidenceArtifact | None,
        safety_artifact: SafetyArtifact | None,
        needs_rag: bool,
    ) -> ValidationReport:
        citation_report = self.citation_verifier.verify(draft, evidence_artifact, needs_rag=needs_rag)
        safety_report = self.safety_critic.review(
            draft,
            manufacturing_context=manufacturing_context,
            safety_artifact=safety_artifact,
            evidence_artifact=evidence_artifact,
        )
        return self._merge([citation_report, safety_report])

    def _merge(self, reports: list[ValidationReport]) -> ValidationReport:
        failures: list[ValidationFailure] = []
        required_reexecution: list[str] = []
        next_action = 'pass'
        retryable = False
        for report in reports:
            failures.extend(report.failures)
            required_reexecution.extend(report.required_reexecution)
            retryable = retryable or report.retryable
            if self.ACTION_PRIORITY[report.next_action] > self.ACTION_PRIORITY[next_action]:
                next_action = report.next_action
        if not failures:
            return ValidationReport.pass_report()
        return ValidationReport(
            passed=False,
            failures=failures,
            retryable=retryable,
            next_action=next_action,  # type: ignore[arg-type]
            required_reexecution=list(dict.fromkeys(required_reexecution)),
        )
