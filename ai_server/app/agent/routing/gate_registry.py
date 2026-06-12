from __future__ import annotations

from app.agent.routing.gate_schemas import GateContext, GateResult
from app.agent.routing.hard_gates import (
    ControlScopeGate,
    DocumentRequestGate,
    EmptyQuestionGate,
    FollowupCandidateGate,
    GlossaryConceptGate,
    HardGate,
    MetaFeedbackGate,
    ProcessDataDiagnosisGate,
    RecommendedActionFollowupGate,
    ReportGate,
    SafetyRequestGate,
)


class GateRegistry:
    def __init__(self, gates: list[HardGate] | None = None):
        self.gates = gates or [
            EmptyQuestionGate(),
            ControlScopeGate(),
            MetaFeedbackGate(),
            RecommendedActionFollowupGate(),
            ProcessDataDiagnosisGate(),
            ReportGate(),
            GlossaryConceptGate(),
            SafetyRequestGate(),
            DocumentRequestGate(),
            FollowupCandidateGate(),
        ]

    def evaluate(self, context: GateContext) -> GateResult:
        for gate in self.gates:
            result = gate.evaluate(context)
            if result.matched and result.is_final:
                return result
        return GateResult(matched=False, gate_name='none', category='empty')

    def evaluate_all(self, context: GateContext) -> list[GateResult]:
        return [result for gate in self.gates if (result := gate.evaluate(context)).matched]
