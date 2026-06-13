from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel

from app.agent.artifacts import AnswerDraft, ValidationFailure, ValidationReport


class ManufacturingIntentType(str, Enum):
    CONCEPT_EXPLANATION = 'concept_explanation'
    MACHINE_TROUBLESHOOTING = 'machine_troubleshooting'
    SAFETY_EXPLANATION = 'safety_explanation'
    MAINTENANCE_PROCEDURE = 'maintenance_procedure'
    UNSAFE_OPERATION_REQUEST = 'unsafe_operation_request'
    FIELD_LOG_INSUFFICIENT = 'field_log_insufficient'
    GENERAL_CHAT = 'general_chat'


RiskLevel = Literal['low', 'medium', 'high', 'unsafe']


class ManufacturingIntentClassification(BaseModel):
    intent_type: ManufacturingIntentType
    risk_level: RiskLevel = 'low'
    reason: str = ''


class RecommendedAction(BaseModel):
    title: str
    action_type: Literal[
        'observe',
        'check_log',
        'non_invasive_inspection',
        'stop_machine',
        'qualified_maintenance',
        'contact_safety_manager',
        'clarify_input',
    ]
    risk_level: Literal['low', 'medium', 'high'] = 'low'
    requires_physical_access: bool = False
    requires_qualified_person: bool = False
    requires_energy_isolation: bool = False
    forbidden_if_machine_running: bool = False


class ManufacturingIntentClassifier:
    """Deterministic manufacturing intent/risk classifier.

    This classifier intentionally checks unsafe execution intent, not single
    words. Terms such as "인터록", "가드 제거", or "LOTO" are allowed in
    explanatory or safer-alternative questions.
    """

    def classify(self, question: str, *, selected_path: str | None = None, answer_type: str | None = None) -> ManufacturingIntentClassification:
        text = _normalize(question)
        compact = _compact(text)
        if not text:
            return ManufacturingIntentClassification(intent_type=ManufacturingIntentType.GENERAL_CHAT, reason='empty question')

        if self._is_unsafe_operation_request(text, compact):
            return ManufacturingIntentClassification(
                intent_type=ManufacturingIntentType.UNSAFE_OPERATION_REQUEST,
                risk_level='unsafe',
                reason='unsafe operation request',
            )
        if self._is_safety_explanation(text, compact):
            risk = 'medium' if _contains_any(compact, ['인터록', '우회', '가드', '방호장치', '안전장치', 'loto']) else 'low'
            return ManufacturingIntentClassification(
                intent_type=ManufacturingIntentType.SAFETY_EXPLANATION,
                risk_level=risk,  # type: ignore[arg-type]
                reason='safety explanation request',
            )
        if self._is_machine_troubleshooting(text, compact):
            risk = 'high' if _contains_any(compact, ['과열', '연기', '타는냄새', '회전부', '스핀들', '이상음']) else 'medium'
            return ManufacturingIntentClassification(
                intent_type=ManufacturingIntentType.MACHINE_TROUBLESHOOTING,
                risk_level=risk,  # type: ignore[arg-type]
                reason='machine troubleshooting request',
            )
        if self._is_maintenance_procedure(text, compact):
            return ManufacturingIntentClassification(
                intent_type=ManufacturingIntentType.MAINTENANCE_PROCEDURE,
                risk_level='high',
                reason='maintenance procedure request',
            )
        if self._is_concept_explanation(text, compact, selected_path=selected_path, answer_type=answer_type):
            return ManufacturingIntentClassification(
                intent_type=ManufacturingIntentType.CONCEPT_EXPLANATION,
                risk_level='low',
                reason='concept explanation request',
            )
        if _contains_any(compact, ['로그부족', '데이터부족', '정보부족', '현장로그']):
            return ManufacturingIntentClassification(
                intent_type=ManufacturingIntentType.FIELD_LOG_INSUFFICIENT,
                risk_level='low',
                reason='field log insufficient',
            )
        return ManufacturingIntentClassification(
            intent_type=ManufacturingIntentType.GENERAL_CHAT,
            risk_level='low',
            reason='general request',
        )

    @staticmethod
    def _is_unsafe_operation_request(text: str, compact: str) -> bool:
        if _is_explanation_or_safer_alternative(text, compact):
            return False
        method_request = _contains_any(compact, ['방법', '하는법', '절차', '알려줘', '가능해', '해도돼', '해도되', '어떻게'])
        if not method_request:
            return False
        patterns = [
            r'(인터록|방호장치|안전장치).*(우회|무력화|해제|끄고|풀고)',
            r'(우회|무력화|해제).*(인터록|방호장치|안전장치)',
            r'(운전중|가동중|회전중).*(가드|커버|문).*(제거|개방|탈거|열고|열어)',
            r'(가드|커버).*(제거|개방|탈거|열고|열어).*(운전중|가동중|회전중)',
            r'(경보|알람|진동|이상음|과열).*(무시|계속운전|계속가동)',
        ]
        return any(re.search(pattern, compact) for pattern in patterns)

    @staticmethod
    def _is_safety_explanation(text: str, compact: str) -> bool:
        if _contains_any(compact, ['loto', '안전', '인터록', '방호장치', '가드', '위험', '잠금표지']):
            if _contains_any(compact, ['왜위험', '위험해', '위험한지', '뜻', '무엇', '뭐야', '설명', '금지', '하면안', '안되는']):
                return True
        return False

    @staticmethod
    def _is_machine_troubleshooting(text: str, compact: str) -> bool:
        if _contains_any(compact, ['없이확인', '하지않고확인', '비침습', '점검방법']):
            return True
        return _contains_any(compact, [
            '이상음', '소음', '진동', '알람', '경보', '과열', '스핀들', '고장',
            '문제', '점검', '확인할방법', '불량', '멈춤', '떨림',
        ])

    @staticmethod
    def _is_maintenance_procedure(text: str, compact: str) -> bool:
        return _contains_any(compact, ['정비', '교체', '분해', '수리', '윤활', '청소', '조정', '보전'])

    @staticmethod
    def _is_concept_explanation(text: str, compact: str, *, selected_path: str | None, answer_type: str | None) -> bool:
        if selected_path == 'fast_concept_answer' or answer_type in {'definition', 'concept', 'explanation'}:
            return True
        return _contains_any(compact, ['란', '뜻', '정의', '무엇', '뭐야', '설명해줘'])


class AnswerPolicyBuilder:
    def build(self, intent_type: ManufacturingIntentType | str | None, risk_level: str, question: str) -> dict[str, Any]:
        try:
            intent = ManufacturingIntentType(intent_type or ManufacturingIntentType.GENERAL_CHAT)
        except ValueError:
            intent = ManufacturingIntentType.GENERAL_CHAT
        if intent == ManufacturingIntentType.MACHINE_TROUBLESHOOTING:
            return {
                'answer_scope': 'troubleshooting_support',
                'risk_level': risk_level,
                'start_with_non_invasive_checks': True,
                'required_diagnostics': ['알람 코드', '발생 시점', 'RPM', '부하', '온도', '진동 조건', '공구 상태', '윤활/냉각 상태'],
                'prohibited_instructions': ['운전 중 회전부 접근', '운전 중 커버 개방', '가드 제거 후 운전', '인터록 우회', '경보 무시 운전'],
                'required_safety_guidance': ['물리 점검 필요 시 장비 정지', '전원 차단', 'LOTO 또는 현장 에너지 차단 절차', '승인된 담당자', '제조사 절차 확인'],
                'wording_rules': [
                    '가드 제거를 직접 지시하지 말고, 커버 개방 또는 방호장치 접근이 필요한 물리 점검으로 표현',
                    '단순 화면/알람 확인과 물리 점검을 구분',
                ],
                'recommended_action_types': [
                    RecommendedAction(title='알람 코드와 발생 시점 확인', action_type='check_log').model_dump(),
                    RecommendedAction(title='비침습 조건에서 RPM, 부하, 온도, 진동, 공구 상태 확인', action_type='non_invasive_inspection', risk_level='medium').model_dump(),
                    RecommendedAction(
                        title='물리 점검은 현장 절차에 따라 정지/에너지 차단 후 승인된 담당자가 수행',
                        action_type='qualified_maintenance',
                        risk_level='high',
                        requires_physical_access=True,
                        requires_qualified_person=True,
                        requires_energy_isolation=True,
                        forbidden_if_machine_running=True,
                    ).model_dump(),
                ],
            }
        if intent == ManufacturingIntentType.SAFETY_EXPLANATION:
            return {
                'answer_scope': 'safety_explanation',
                'risk_level': risk_level,
                'allow_explanation_of_hazard': True,
                'do_not_provide_bypass_steps': True,
                'explain_why_unsafe': True,
                'required_safety_guidance': ['우회/해제 절차가 아니라 위험성과 안전한 대안 중심으로 설명'],
            }
        if intent == ManufacturingIntentType.MAINTENANCE_PROCEDURE:
            return {
                'answer_scope': 'maintenance_procedure',
                'risk_level': risk_level,
                'requires_qualified_person': True,
                'requires_energy_isolation_if_physical_access': True,
                'required_safety_guidance': ['정지', '에너지 차단', '잔류 에너지 확인', 'LOTO 또는 현장 절차', '승인된 담당자', '제조사 절차'],
                'prohibited_instructions': ['무자격 정비', '운전 중 회전부 접근', '가드 제거 후 운전', '인터록 우회'],
                'recommended_action_types': [
                    RecommendedAction(
                        title='물리 작업 전 현장 절차에 따라 장비 정지와 에너지 차단 상태 확인',
                        action_type='qualified_maintenance',
                        risk_level='high',
                        requires_physical_access=True,
                        requires_qualified_person=True,
                        requires_energy_isolation=True,
                        forbidden_if_machine_running=True,
                    ).model_dump(),
                ],
            }
        if intent == ManufacturingIntentType.CONCEPT_EXPLANATION:
            return {
                'answer_scope': 'concept_only',
                'risk_level': risk_level,
                'avoid_current_machine_state_claims': True,
                'avoid_failure_probability_without_prediction': True,
                'keep_answer_lightweight': True,
            }
        if intent == ManufacturingIntentType.UNSAFE_OPERATION_REQUEST:
            return {
                'answer_scope': 'blocked_unsafe_request',
                'risk_level': risk_level,
                'block': True,
                'safe_alternative_required': True,
                'explain_briefly': True,
            }
        return {
            'answer_scope': 'general',
            'risk_level': risk_level,
            'avoid_current_machine_state_claims_without_evidence': True,
        }


class DangerousOutputDetector:
    DANGEROUS_OUTPUT_PATTERNS = [
        re.compile(r'운전\s*중.*(가드|커버).*(제거|개방|탈거|해제).*(하세요|확인하세요|가능합니다|해도\s*됩니다)'),
        re.compile(r'(회전\s*중|가동\s*중).*(커버|가드).*(열고|개방|제거).*(확인|점검)'),
        re.compile(r'(인터록|방호장치).*(우회|무력화|해제).*(하세요|가능합니다|해도\s*됩니다)'),
        re.compile(r'(경보|알람|진동|이상음|과열).*(무시하고|계속\s*가동|계속\s*운전)'),
    ]

    def detect(self, text: str, *, intent_type: str | None = None) -> list[str]:
        matches: list[str] = []
        for line in (text or '').splitlines() or [text or '']:
            stripped = ' '.join(line.split())
            if not stripped:
                continue
            if self._is_allowed_explanation_line(stripped, intent_type=intent_type):
                continue
            for pattern in self.DANGEROUS_OUTPUT_PATTERNS:
                if pattern.search(stripped):
                    matches.append(stripped)
                    break
        return matches

    def review(self, draft: AnswerDraft, *, intent_type: str | None = None) -> ValidationReport:
        matches = self.detect(draft.text, intent_type=intent_type)
        if not matches:
            return ValidationReport.pass_report()
        return ValidationReport(
            passed=False,
            failures=[
                ValidationFailure(
                    code='dangerous_output_instruction',
                    message=f'Public answer contains dangerous executable instruction: {match}',
                    severity='critical',
                    source='safety',
                )
                for match in matches[:3]
            ],
            retryable=False,
            next_action='block',
        )

    @staticmethod
    def _is_allowed_explanation_line(line: str, *, intent_type: str | None = None) -> bool:
        compact = _compact(line)
        if _contains_any(compact, ['없이', '하지않고', '비침습']):
            return True
        explanation_terms = ['위험', '금지', '하면안', '하지마', '하지않', '안됩니다', '수없습니다', '우회절차가아니라']
        directive_terms = ['하세요', '하십시오', '해도됩니다', '가능합니다', '확인하세요', '점검하세요']
        if _contains_any(compact, explanation_terms) and not _contains_any(compact, directive_terms):
            return True
        if intent_type == ManufacturingIntentType.SAFETY_EXPLANATION.value and _contains_any(compact, explanation_terms):
            return True
        return False


def _normalize(text: str) -> str:
    return ' '.join((text or '').strip().split())


def _compact(text: str) -> str:
    return re.sub(r'[^가-힣a-z0-9]+', '', (text or '').lower())


def _contains_any(text: str, tokens: list[str]) -> bool:
    return any(token in text for token in tokens)


def _is_explanation_or_safer_alternative(text: str, compact: str) -> bool:
    return _contains_any(compact, [
        '왜위험', '위험해', '위험한지', '설명', '금지', '하면안', '안되는',
        '없이', '하지않고', '대안', '비침습', '제거없이', '우회없이',
        'loto가뭐', 'loto란', '무엇', '뭐야',
    ])
