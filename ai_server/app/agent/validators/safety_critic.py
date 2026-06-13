from __future__ import annotations

import re

from app.agent.artifacts import AnswerDraft, EvidenceArtifact, SafetyArtifact, ValidationFailure, ValidationReport
from app.schemas.domain import ManufacturingContext
from app.services.safety_validation_service import SafetyValidationService


class SafetyCritic:
    """Maps safety validation findings to bounded review actions."""

    def __init__(self, validator: SafetyValidationService):
        self.validator = validator

    def review(
        self,
        draft: AnswerDraft,
        *,
        manufacturing_context: ManufacturingContext,
        safety_artifact: SafetyArtifact | None = None,
        evidence_artifact: EvidenceArtifact | None = None,
    ) -> ValidationReport:
        result = self.validator.validate_answer(draft.text, manufacturing_context)
        failures = [
            ValidationFailure(
                code=self._code(error),
                message=error,
                severity='critical' if '금지된' in error else 'error',
                source='safety',
            )
            for error in result.errors
        ]
        failures.extend(self._public_gate_id_failures(draft.text, manufacturing_context))
        if not failures:
            return ValidationReport.pass_report()
        if any(f.code == 'forbidden_action' and f.severity == 'critical' for f in failures):
            return ValidationReport(passed=False, failures=failures, retryable=False, next_action='block')
        if any(f.code == 'public_safety_gate_id_leak' for f in failures):
            return ValidationReport(passed=False, failures=failures, retryable=True, next_action='rewrite_only')
        missing_gate = any(f.code == 'required_safety_gate_missing' for f in failures)
        missing_evidence = bool(evidence_artifact and not evidence_artifact.evidence_covers_required_gates)
        if missing_gate and missing_evidence:
            return ValidationReport(
                passed=False,
                failures=failures,
                retryable=True,
                next_action='rerun_rag',
                required_reexecution=['rag_evidence', 'safety_contract'],
            )
        if missing_gate and not (safety_artifact and safety_artifact.required_gates):
            return ValidationReport(
                passed=False,
                failures=failures,
                retryable=True,
                next_action='rerun_safety',
                required_reexecution=['safety_contract'],
            )
        return ValidationReport(passed=False, failures=failures, retryable=True, next_action='rewrite_only')

    @staticmethod
    def _code(message: str) -> str:
        if '금지된' in message:
            return 'forbidden_action'
        if '필수 안전 게이트 누락' in message:
            return 'required_safety_gate_missing'
        return 'safety_validation_failed'

    @staticmethod
    def _public_gate_id_failures(answer: str, context: ManufacturingContext) -> list[ValidationFailure]:
        failures: list[ValidationFailure] = []
        lowered = (answer or '').lower()
        for gate in context.safety_gates:
            gate_id = (gate.gate_id or '').strip()
            if not gate_id:
                continue
            if re.search(rf'\b{re.escape(gate_id.lower())}\b', lowered):
                failures.append(ValidationFailure(
                    code='public_safety_gate_id_leak',
                    message=f'Public answer exposes internal safety gate id: {gate_id}',
                    severity='error',
                    source='debug_leak',
                ))
        return failures
