from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS_DIR))

from rag_pipeline_utils import classify_kosha_doc, normalize_items, safe_filename


FIXTURES = Path(__file__).resolve().parent / 'fixtures'


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding='utf-8'))


def test_normalize_items_handles_list_and_single_dict():
    list_payload = load_fixture('kosha_sample_response_list.json')
    single_payload = load_fixture('kosha_sample_response_single_item.json')

    assert len(normalize_items(list_payload)) == 2
    assert len(normalize_items(single_payload)) == 1
    assert normalize_items({'body': {'items': {}}}) == []
    assert normalize_items({'body': {}}) == []
    assert normalize_items({}) == []


def test_safe_filename_handles_korean_spaces_special_chars_and_length():
    value = safe_filename(' 프레스 방호장치 / 점검: 기술지침? ' * 20, max_length=80)

    assert '/' not in value
    assert ':' not in value
    assert '?' not in value
    assert ' ' not in value
    assert len(value) <= 80
    assert value.startswith('프레스_방호장치')


def test_classify_kosha_doc_metadata_rules():
    cases = [
        ('프레스 방호장치 점검에 관한 기술지침', 'korean_machine_safety', 'high', 'default'),
        ('공작기계 정비 작업 안전에 관한 지침', 'korean_maintenance_guidance', 'high', 'default'),
        ('작업환경측정 및 소음 관리에 관한 지침', 'korean_work_environment_guidance', 'low', 'restricted'),
        ('건설 현장 추락 예방 지침', 'korean_general_safety_other', 'low', 'restricted'),
        ('알 수 없는 일반 안전 지침', 'korean_safety_reference', 'medium', 'restricted'),
    ]
    for title, doc_type, priority, scope in cases:
        metadata = classify_kosha_doc(title)
        assert metadata['doc_type'] == doc_type
        assert metadata['project_priority'] == priority
        assert metadata['retrieval_scope'] == scope
