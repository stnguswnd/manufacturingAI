from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas.domain import SafetyGateResult
from app.schemas.domain import ManufacturingContext


CONTROL_FORBIDDEN = [
    '자동으로 정지',
    '정지했습니다',
    '제가 제어',
    '자동 제어',
    '교체했습니다',
    '수리했습니다',
    '안전 상태를 보증',
    '운전 중 점검',
    'LOTO 없이',
    '잠금/표지 생략',
    '가드 제거',
    '비상정지 우회',
    '무전압 확인 없이',
    '전기함을 임의로 열',
]


@dataclass(frozen=True)
class SafetyValidationResult:
    passed: bool
    errors: list[str]


class SafetyValidationService:
    @staticmethod
    def validate_answer(answer: str, manufacturing_context: ManufacturingContext | None) -> SafetyValidationResult:
        errors: list[str] = []
        normalized = answer or ''
        unsafe_lines = [
            line
            for line in normalized.splitlines()
            if not SafetyValidationService._is_negated_or_policy_line(line)
        ]
        unsafe_text = '\n'.join(unsafe_lines)
        for phrase in CONTROL_FORBIDDEN:
            if phrase in unsafe_text:
                errors.append(f'금지된 설비 제어/보증 표현 감지: {phrase}')

        if manufacturing_context and manufacturing_context.safety_gates:
            for gate in manufacturing_context.safety_gates:
                if not SafetyValidationService._gate_content_covered(normalized, gate):
                    errors.append(f'필수 안전 게이트 누락: {gate.gate_id}')

        return SafetyValidationResult(passed=not errors, errors=errors)

    @staticmethod
    def _gate_content_covered(answer: str, gate: SafetyGateResult) -> bool:
        if gate.name_ko and gate.name_ko in answer:
            return True
        if gate.gate_id and gate.gate_id in answer:
            return True
        if any(check and check in answer for check in gate.required_checks[:3]):
            return True

        answer_terms = SafetyValidationService._salient_terms(answer)
        answer_compact = SafetyValidationService._compact(answer)
        for check in gate.required_checks[:3]:
            check_terms = SafetyValidationService._salient_terms(check)
            if check_terms and len(check_terms & answer_terms) >= min(2, len(check_terms)):
                return True

        policy_terms: set[str] = set()
        for text in [gate.name_ko, gate.description_ko, *gate.document_search_terms]:
            if text and SafetyValidationService._compact(text) in answer_compact:
                return True
            policy_terms.update(SafetyValidationService._salient_terms(text))
        return bool(policy_terms and len(policy_terms & answer_terms) >= min(2, len(policy_terms)))

    @staticmethod
    def _is_negated_or_policy_line(line: str) -> bool:
        policy_terms = [
            '금지',
            '하지 않습니다',
            '하지 않는다',
            '하지 않',
            '않습니다',
            '수 없습니다',
            '없습니다',
            '아닙니다',
            '아니며',
            '아니라',
            '대체하지',
            '보증하지',
            '보장하지',
            '수행하지',
            '제어하지',
            '제거 없이',
            '우회 없이',
            '개방 없이',
            '하지 않고',
        ]
        return any(term in line for term in policy_terms)

    @staticmethod
    def _salient_terms(text: str) -> set[str]:
        stopwords = {
            '확인', '상태', '여부', '작업', '필요', '절차', '안전', '정비',
            '전에', '전', '후', '및', '등', '경우', '담당자', '승인된',
            'the', 'and', 'for', 'with', 'check', 'before', 'after',
        }
        terms: set[str] = set()
        for token in re.findall(r'[가-힣A-Za-z0-9]+', text or ''):
            lowered = token.lower().rstrip('이가은는을를에의와과도')
            if len(lowered) < 2 or lowered in stopwords or lowered.isdigit():
                continue
            terms.add(lowered)
            if re.fullmatch(r'[가-힣]{3,}', lowered):
                terms.add(lowered[:2])
        return terms

    @staticmethod
    def _compact(text: str) -> str:
        return re.sub(r'[^가-힣a-z0-9]+', '', (text or '').lower())
