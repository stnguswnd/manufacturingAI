from __future__ import annotations

import re

from app.agent.artifacts import AnswerDraft, EvidenceArtifact, ValidationFailure, ValidationReport


class CitationVerifier:
    """Deterministic public-answer citation and debug-leak checks."""

    DEBUG_PATTERNS = [
        (r'\brun_id\b', 'run_id'),
        (r'\btokens?\b', 'token'),
        (r'\bcost\b', 'cost'),
        (r'\braw_score\b', 'raw_score'),
        (r'\bchunk_id\b', 'chunk_id'),
        (r'\bsafety_gate_id\b', 'safety_gate_id'),
        (r'\bgate id\b', 'gate id'),
        (r'\bcalls=', 'calls='),
        (r'\breplans=', 'replans='),
    ]
    EVIDENCE_CLAIM_TERMS = ['문서', '근거', '출처', '참조', 'osha', 'haas', 'kosha', 'guide', 'manual']

    def verify(self, draft: AnswerDraft, evidence: EvidenceArtifact | None, *, needs_rag: bool = False) -> ValidationReport:
        failures: list[ValidationFailure] = []
        text = draft.text or ''
        lowered = text.lower()
        for pattern, label in self.DEBUG_PATTERNS:
            if re.search(pattern, lowered):
                failures.append(ValidationFailure(
                    code='public_debug_metadata_leak',
                    message=f'Public answer contains debug/internal metadata: {label}',
                    severity='error',
                    source='debug_leak',
                ))
        citations = (evidence.citations if evidence else []) or draft.citations
        if evidence and not evidence.evidence_covers_required_gates:
            failures.append(ValidationFailure(
                code='missing_required_gate_evidence',
                message='Evidence does not cover all required safety gates.',
                severity='error',
                source='citation',
            ))
        claims_document_evidence = any(term in lowered for term in self.EVIDENCE_CLAIM_TERMS)
        if needs_rag and claims_document_evidence and not citations:
            failures.append(ValidationFailure(
                code='citation_required_but_missing',
                message='Answer claims document-backed evidence but has no citation references.',
                severity='error',
                source='citation',
            ))
        return self._report(failures, evidence=evidence, citations=citations)

    @staticmethod
    def _report(failures: list[ValidationFailure], *, evidence: EvidenceArtifact | None, citations: list[dict]) -> ValidationReport:
        if not failures:
            return ValidationReport.pass_report()
        if any(f.code == 'missing_required_gate_evidence' for f in failures):
            return ValidationReport(
                passed=False,
                failures=failures,
                retryable=True,
                next_action='rerun_rag',
                required_reexecution=['rag_evidence', 'safety_contract'],
            )
        if any(f.source == 'debug_leak' for f in failures):
            return ValidationReport(passed=False, failures=failures, retryable=True, next_action='rewrite_only')
        if not citations:
            return ValidationReport(
                passed=False,
                failures=failures,
                retryable=True,
                next_action='rerun_rag',
                required_reexecution=['rag_evidence', 'safety_contract'],
            )
        return ValidationReport(passed=False, failures=failures, retryable=True, next_action='rewrite_only')
