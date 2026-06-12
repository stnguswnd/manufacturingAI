from __future__ import annotations

from app.schemas import (
    ActionStep,
    AgentResponse,
    AssetContext,
    ManufacturingContext,
    RiskAssessment,
    RiskAxis,
    SafetyGateResult,
    AgentRequest,
)
from app.services.memory_service import MemoryService
from app.services.user_service import UserService
from app.storage.sqlite_store import SQLiteStore


def test_memory_extraction_from_run(tmp_path):
    store = SQLiteStore(tmp_path / 'test.sqlite3')
    user = UserService(store).create({'display_name': 'A'})
    mfg = ManufacturingContext(
        asset_context=AssetContext(equipment_type='CNC', equipment_label_ko='CNC 계열 설비'),
        risk_assessment=RiskAssessment(
            quality=RiskAxis(axis='quality', level='high', rationale=''),
            equipment=RiskAxis(axis='equipment', level='high', rationale=''),
            safety=RiskAxis(axis='safety', level='high', rationale=''),
            production=RiskAxis(axis='production', level='medium', rationale=''),
            document_confidence=RiskAxis(axis='document_confidence', level='medium', rationale=''),
            overall_priority='high',
        ),
        action_plan=[ActionStep(action_id='a', label_ko='점검', description_ko='', output_phrase='점검')],
        safety_gates=[
            SafetyGateResult(
                gate_id='loto_if_physical_maintenance',
                name_ko='정비 전 LOTO/에너지 차단 확인',
                severity='high',
                description_ko='',
                required_checks=['전원 차단'],
                forbidden_agent_actions=['보증 금지'],
            )
        ],
    )
    response = AgentResponse(run_id='r1', user_id=user['user_id'], route=[], answer='ok', manufacturing_context=mfg, report='report')
    result = MemoryService(store).update_from_run(user_id=user['user_id'], request=AgentRequest(user_id=user['user_id'], question='q', session_id='s1', generate_report=True), response=response)
    memories = store.list_memories(user['user_id'])

    assert result['updated_count'] >= 3
    assert {memory['memory_type'] for memory in memories} >= {'equipment_preference', 'safety_note', 'report_preference', 'recent_summary'}

