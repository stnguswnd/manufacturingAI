from __future__ import annotations

from typing import Any


GLOSSARY: dict[str, dict[str, Any]] = {
    '토크': {
        'aliases': ['토크', 'torque'],
        'definition': '토크는 물체를 회전시키는 힘의 효과를 뜻합니다.',
        'manufacturing_context': '제조 설비에서는 모터, 스핀들, 축, 공구처럼 회전하는 부품에 걸리는 부하를 이해할 때 사용합니다.',
        'watch_points': ['회전수와 함께 볼 것', '공구 마모와 함께 볼 것', '진동과 온도를 함께 볼 것', '단일 값보다 추세를 볼 것', '장비별 기준값과 알람 이력을 함께 볼 것'],
        'risk_boundary': '현재 설비가 위험한지는 실제 토크 값, 회전수, 공구 마모, 온도, 진동 같은 공정 데이터가 있어야 판단할 수 있습니다.',
        'risks': ['공구 마모 증가', '발열', '진동', '품질 저하', '모터 부하 증가'],
        'related_terms': ['회전수', '스핀들', '공구 마모', '부하'],
    },
    '공구 마모': {
        'aliases': ['공구 마모', 'tool wear', '마모'],
        'definition': '공구 마모는 절삭이나 가공 과정에서 공구의 날이나 표면이 닳는 현상입니다.',
        'manufacturing_context': '공구 마모가 커지면 치수 불량, 표면 품질 저하, 절삭 저항 증가가 발생할 수 있습니다.',
        'watch_points': ['가공 시간과 공구 교체 이력을 함께 볼 것', '토크 상승과 함께 나타나는지 볼 것', '표면 품질과 치수 편차를 확인할 것', '진동과 소음 변화를 함께 볼 것'],
        'risk_boundary': '마모 상태 판단은 실제 가공 조건, 진동, 토크, 온도, 품질 측정값이 함께 필요합니다.',
        'risks': ['품질 저하', '가공 불량', '토크 증가', '공구 파손 가능성'],
        'related_terms': ['토크', '절삭 부하', '스핀들', '진동'],
    },
    '스핀들': {
        'aliases': ['스핀들', 'spindle', 'cnc 스핀들', 'cnc spindle'],
        'definition': '스핀들은 공구나 공작물을 회전시키는 핵심 회전축입니다.',
        'manufacturing_context': '제조 설비에서는 회전수, 진동, 부하, 발열 상태가 품질과 설비 안정성에 영향을 줍니다.',
        'watch_points': ['회전수와 부하 변화를 함께 볼 것', '진동과 발열 추세를 볼 것', '베어링 소음이나 알람 이력을 확인할 것', '공구 상태와 절삭 조건을 함께 볼 것'],
        'risk_boundary': '스핀들 이상 여부는 회전수, 진동, 온도, 부하 데이터가 있어야 판단할 수 있습니다.',
        'risks': ['진동 증가', '가공 정밀도 저하', '베어링 손상', '발열'],
        'related_terms': ['회전수', '토크', '베어링', '진동'],
    },
    '회전수': {
        'aliases': ['회전수', 'rpm', 'rotational speed'],
        'definition': '회전수는 축, 스핀들, 공구가 일정 시간 동안 회전하는 횟수입니다.',
        'manufacturing_context': '제조 공정에서는 절삭 속도, 표면 품질, 발열, 부하와 함께 해석합니다.',
        'watch_points': ['토크와 함께 볼 것', '공구 마모와 함께 볼 것', '소재와 가공 조건에 맞는 기준값을 확인할 것', '갑작스러운 변동보다 지속 추세를 볼 것'],
        'risk_boundary': '현재 회전수가 적정한지는 소재, 공구, 토크, 마모, 품질 측정값이 함께 있어야 판단할 수 있습니다.',
        'risks': ['가공 품질 저하', '발열 증가', '진동', '공구 수명 저하'],
        'related_terms': ['토크', '스핀들', '절삭 속도', '공구 마모'],
    },
    '부하': {
        'aliases': ['부하', 'load'],
        'definition': '부하는 설비나 부품이 작업 중 받는 힘, 저항, 에너지 요구량을 뜻합니다.',
        'manufacturing_context': '제조 설비에서는 모터, 스핀들, 공구에 걸리는 작업 부담을 해석할 때 사용합니다.',
        'watch_points': ['토크와 전류를 함께 볼 것', '회전수와 부하 변화가 동시에 나타나는지 볼 것', '공구 마모나 걸림이 있는지 확인할 것'],
        'risk_boundary': '현재 부하가 위험한지는 설비 기준값, 토크, 전류, 회전수, 알람 이력이 함께 있어야 판단할 수 있습니다.',
        'risks': ['과부하', '발열', '모터 부담 증가', '품질 저하'],
        'related_terms': ['토크', '전류', '회전수', '스핀들'],
    },
    'OSF': {
        'aliases': ['osf', '과부하 고장', 'overstrain'],
        'definition': 'OSF는 AI4I 데이터에서 과부하성 고장 가능성을 나타내는 고장모드입니다.',
        'manufacturing_context': '토크, 공구 마모, 회전 조건이 불리하게 겹칠 때 과부하 신호로 해석할 수 있습니다.',
        'watch_points': ['토크와 공구 마모를 함께 볼 것', '회전수 조건을 확인할 것', '스핀들 부하와 알람 이력을 함께 볼 것'],
        'risk_boundary': 'OSF 가능성은 실제 공정 데이터와 모델 예측 결과가 있어야 판단할 수 있습니다.',
        'risks': ['과부하', '공구 손상', '스핀들 부담 증가'],
        'related_terms': ['토크', '공구 마모', '스핀들', '부하'],
    },
    'TWF': {
        'aliases': ['twf', '공구 마모 고장'],
        'definition': 'TWF는 AI4I 데이터에서 공구 마모와 관련된 고장모드입니다.',
        'manufacturing_context': '공구 사용 시간이 길거나 절삭 저항이 커질 때 품질과 설비 부담에 영향을 줄 수 있습니다.',
        'watch_points': ['공구 사용 시간', '표면 품질', '토크 상승', '진동 변화를 함께 볼 것'],
        'risk_boundary': 'TWF 가능성은 실제 공구 마모 시간과 가공 조건, 예측 결과가 있어야 판단할 수 있습니다.',
        'risks': ['품질 저하', '공구 파손 가능성', '토크 증가'],
        'related_terms': ['공구 마모', '토크', '품질'],
    },
    'HDF': {
        'aliases': ['hdf', '방열 고장', '열 방출'],
        'definition': 'HDF는 AI4I 데이터에서 열 방출 또는 방열 문제와 관련된 고장모드입니다.',
        'manufacturing_context': '공기 온도와 공정 온도 조건이 불리할 때 냉각이나 발열 문제를 의심하는 보조 신호가 됩니다.',
        'watch_points': ['공정 온도와 공기 온도를 함께 볼 것', '냉각 상태를 확인할 것', '발열 추세를 볼 것'],
        'risk_boundary': 'HDF 가능성은 온도 데이터와 설비 냉각 조건이 있어야 판단할 수 있습니다.',
        'risks': ['발열', '품질 저하', '냉각 성능 저하'],
        'related_terms': ['온도', '냉각', '방열'],
    },
    'PWF': {
        'aliases': ['pwf', '동력 고장', '전력 고장'],
        'definition': 'PWF는 AI4I 데이터에서 동력 또는 출력 조건과 관련된 고장모드입니다.',
        'manufacturing_context': '토크와 회전수 조합이 불리할 때 모터나 구동계 부담을 해석하는 보조 신호가 됩니다.',
        'watch_points': ['토크와 회전수를 함께 볼 것', '전류나 구동계 알람을 확인할 것', '부하 변동 추세를 볼 것'],
        'risk_boundary': 'PWF 가능성은 실제 공정 데이터와 구동계 상태 정보가 있어야 판단할 수 있습니다.',
        'risks': ['구동계 부담', '출력 불안정', '생산 품질 저하'],
        'related_terms': ['토크', '회전수', '부하'],
    },
    'LOTO': {
        'aliases': ['loto', 'lockout', 'tagout', '잠금표지', '에너지 차단'],
        'definition': 'LOTO는 정비 전 설비 에너지를 차단하고 잠금표지로 재가동을 방지하는 절차입니다.',
        'manufacturing_context': '물리 점검이나 정비가 필요한 상황에서 작업자 안전을 위해 확인해야 하는 절차입니다.',
        'watch_points': ['정비 전 에너지 차단 확인', '권한 있는 담당자 확인', '잔류 에너지 확인', '임의 재가동 방지'],
        'risk_boundary': 'LOTO 필요 여부와 수행은 현장 안전 절차와 자격 있는 담당자 판단이 필요합니다.',
        'risks': ['예기치 않은 가동', '끼임', '감전', '상해'],
        'related_terms': ['정비', '에너지 차단', '기계 방호'],
    },
    '기계 방호': {
        'aliases': ['기계 방호', '기계방호', 'machine guarding', '방호장치'],
        'definition': '기계 방호는 회전부, 끼임점, 비산물 등으로부터 작업자를 보호하기 위한 장치와 절차입니다.',
        'manufacturing_context': '스핀들, 공구, 회전축 주변 점검이나 정비 시 방호장치 상태가 중요합니다.',
        'watch_points': ['회전부 접근 전 방호장치 확인', '운전 중 접근 금지', '인터록과 가드 상태 확인'],
        'risk_boundary': '방호장치 적정성은 현장 설비 기준과 안전 담당자 확인이 필요합니다.',
        'risks': ['끼임', '절단', '비산', '상해'],
        'related_terms': ['스핀들', '회전수', 'LOTO', '안전장치'],
    },
}


class GlossaryAnswerService:
    def can_answer(self, question: str, resolved_target: dict[str, Any] | None = None) -> bool:
        return self.resolve_term(question, resolved_target=resolved_target) is not None

    def answer_payload(self, question: str, resolved_target: dict[str, Any] | None = None) -> dict[str, Any] | None:
        resolved = self.resolve_term(question, resolved_target=resolved_target)
        if not resolved:
            return None
        term = resolved['canonical']
        data = GLOSSARY[term]
        return {'term': term, 'matched_text': resolved['matched_text'], 'canonical': term, **data}

    def resolve_term(self, question: str, resolved_target: dict[str, Any] | None = None) -> dict[str, str] | None:
        target_label = str((resolved_target or {}).get('normalized') or (resolved_target or {}).get('label') or (resolved_target or {}).get('text') or '').strip()
        target = self._canonical_from_text(target_label)
        if target:
            return target
        text = (question or '').lower()
        return self._canonical_from_text(text)

    @staticmethod
    def canonical_terms() -> dict[str, dict[str, Any]]:
        return GLOSSARY

    @staticmethod
    def _canonical_from_text(text: str) -> dict[str, str] | None:
        source = (text or '').lower()
        if not source:
            return None
        for term, data in GLOSSARY.items():
            if term.lower() == source:
                return {'canonical': term, 'matched_text': text}
            for alias in data.get('aliases', []):
                alias_text = str(alias).lower()
                if alias_text and alias_text in source:
                    return {'canonical': term, 'matched_text': str(alias)}
        return None
