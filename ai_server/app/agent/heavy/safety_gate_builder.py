from __future__ import annotations

from app.schemas import ManufacturingContext


class SafetyGateBuilder:
    """Builds safety guidance and public warnings from domain safety gates."""

    def safety_guidance(self, manufacturing_context: ManufacturingContext) -> str:
        lines: list[str] = []
        for gate in manufacturing_context.safety_gates:
            lines.append(f'### {gate.name_ko}')
            lines.append(f'- 위험도: {gate.severity}')
            lines.append(f'- 설명: {gate.description_ko}')
            for check in gate.required_checks:
                lines.append(f'- 확인: {check}')
            for forbidden in gate.forbidden_agent_actions[:3]:
                lines.append(f'- 금지: {forbidden}')
            if gate.escalation:
                lines.append(f'- Escalation: {gate.escalation}')
            lines.append('')
        return '\n'.join(lines).strip()

    def warnings(self, manufacturing_context: ManufacturingContext) -> list[str]:
        warnings = ['실제 설비 제어/정비 실행/법적 안전 판단을 대체하지 않습니다.']
        warnings.extend(manufacturing_context.audit_notes)
        forbidden: list[str] = []
        for gate in manufacturing_context.safety_gates:
            forbidden.extend(gate.forbidden_agent_actions)
        if forbidden:
            warnings.append('응답 생성 시 금지 표현: ' + '; '.join(list(dict.fromkeys(forbidden))[:6]))
        return list(dict.fromkeys(warnings))
