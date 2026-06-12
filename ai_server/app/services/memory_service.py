from __future__ import annotations

from app.schemas import AgentRequest, AgentResponse
from app.storage.sqlite_store import SQLiteStore


class MemoryService:
    def __init__(self, store: SQLiteStore | None = None):
        self.store = store or SQLiteStore()

    def update_from_run(self, *, user_id: str, request: AgentRequest, response: AgentResponse) -> dict:
        updated = 0
        mfg = response.manufacturing_context
        if not mfg:
            return {'updated_count': 0}
        asset = mfg.asset_context
        if asset and asset.equipment_type:
            self.store.upsert_memory(
                user_id=user_id,
                memory_type='equipment_preference',
                memory_key=asset.equipment_type,
                content={'value': asset.equipment_type, 'summary': f'{asset.equipment_label_ko} 관련 질의가 반복되었습니다.'},
                source_run_id=response.run_id,
                importance=4,
            )
            updated += 1
        for failure in mfg.failure_modes:
            self.store.upsert_memory(
                user_id=user_id,
                memory_type='recurring_failure_mode',
                memory_key=failure.code,
                content={'value': failure.code, 'summary': f'{failure.code}({failure.name_ko}) 고장모드가 반복 등장했습니다.'},
                source_run_id=response.run_id,
                importance=4,
            )
            updated += 1
        if request.generate_report or response.report:
            self.store.upsert_memory(
                user_id=user_id,
                memory_type='report_preference',
                memory_key='standard',
                content={'value': 'standard', 'summary': '보고서 초안 생성을 자주 사용합니다.'},
                source_run_id=response.run_id,
                importance=3,
            )
            updated += 1
        for gate in mfg.safety_gates:
            self.store.upsert_memory(
                user_id=user_id,
                memory_type='safety_note',
                memory_key=gate.gate_id,
                content={'value': gate.gate_id, 'summary': f'{gate.name_ko} safety gate가 반복 적용되었습니다.'},
                source_run_id=response.run_id,
                importance=5,
            )
            updated += 1
        if request.session_id:
            self.store.upsert_memory(
                user_id=user_id,
                memory_type='recent_summary',
                memory_key=request.session_id,
                content={'summary': self._session_summary(request, response)},
                source_run_id=response.run_id,
                importance=3,
            )
            updated += 1
        return {'updated_count': updated}

    @staticmethod
    def _session_summary(request: AgentRequest, response: AgentResponse) -> str:
        mfg = response.manufacturing_context
        if not mfg:
            return request.question[:300]
        failures = ', '.join(f.code for f in mfg.failure_modes[:3]) or '미확정'
        gates = ', '.join(g.name_ko for g in mfg.safety_gates[:3]) or '없음'
        return f'최근 질문: {request.question[:160]} / 종합 위험도: {mfg.risk_assessment.overall_priority} / 고장모드: {failures} / 안전 게이트: {gates}'

