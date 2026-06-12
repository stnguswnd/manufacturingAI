from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS_DIR))

from build_rag_documents import build_documents


def test_build_rag_documents_extracts_html_and_keeps_metadata_only(tmp_path, monkeypatch):
    project = tmp_path
    manifest = project / 'ai_server' / 'data' / 'rag_source_manifest.yaml'
    raw_html = project / 'ai_server' / 'data' / 'raw' / 'rag_sources' / 'osha' / 'sample.html'
    processed = project / 'ai_server' / 'data' / 'processed'
    manifest.parent.mkdir(parents=True)
    raw_html.parent.mkdir(parents=True)
    processed.mkdir(parents=True)
    raw_html.write_text('<html><body><nav>skip</nav><main><h1>LOTO</h1><p>정비 전 에너지 차단 확인.</p></main></body></html>', encoding='utf-8')
    manifest.write_text(
        '''
sources:
  - id: sample_osha
    source: OSHA
    url: https://example.com/sample
    save_path: ai_server/data/raw/rag_sources/osha/sample.html
    doc_type: safety_standard
    safety_gate: loto
    failure_modes: [OSF]
    related_signals: [torque_nm]
    project_priority: high
    retrieval_scope: default
    use_case: 테스트
kosha_api: {}
''',
        encoding='utf-8',
    )
    out = processed / 'rag_documents.jsonl'
    monkeypatch.setattr('build_rag_documents.PROJECT_ROOT', project)
    monkeypatch.setattr('build_rag_documents.RAG_DOCUMENTS_PATH', out)
    monkeypatch.setattr('build_rag_documents.load_manifest', lambda: {
        'sources': [
            {
                'id': 'sample_osha',
                'source': 'OSHA',
                'url': 'https://example.com/sample',
                'save_path': 'ai_server/data/raw/rag_sources/osha/sample.html',
                'doc_type': 'safety_standard',
                'safety_gate': 'loto',
                'failure_modes': ['OSF'],
                'related_signals': ['torque_nm'],
                'project_priority': 'high',
                'retrieval_scope': 'default',
                'use_case': '테스트',
            }
        ]
    })

    docs = build_documents()

    assert len(docs) == 1
    assert docs[0]['doc_id'] == 'sample_osha'
    assert docs[0]['source'] == 'OSHA'
    assert docs[0]['doc_type'] == 'safety_standard'
    assert docs[0]['metadata_only'] is False
    assert '정비 전 에너지 차단 확인' in docs[0]['text']
    assert out.exists()


def test_build_rag_documents_marks_hwp_as_metadata_only(tmp_path, monkeypatch):
    project = tmp_path
    processed = project / 'ai_server' / 'data' / 'processed'
    hwp = project / 'ai_server' / 'data' / 'raw' / 'rag_sources' / 'kosha' / 'files' / 'guide.hwp'
    processed.mkdir(parents=True)
    hwp.parent.mkdir(parents=True)
    hwp.write_bytes(b'HWP binary')
    index = {
        'documents': [
            {
                'techGdlnNo': 'KOSHA-1',
                'techGdlnNm': '공작기계 정비 작업 안전에 관한 지침',
                'fileDownlUrl': 'https://example.com/guide.hwp',
                'local_path': 'ai_server/data/raw/rag_sources/kosha/files/guide.hwp',
                'doc_type': 'korean_maintenance_guidance',
                'safety_gate': 'maintenance_check',
                'failure_modes': ['OSF'],
                'related_signals': ['torque_nm'],
                'project_priority': 'high',
                'retrieval_scope': 'default',
            }
        ]
    }
    index_path = processed / 'kosha_download_index.json'
    index_path.write_text(json.dumps(index, ensure_ascii=False), encoding='utf-8')
    out = processed / 'rag_documents.jsonl'
    monkeypatch.setattr('build_rag_documents.PROJECT_ROOT', project)
    monkeypatch.setattr('build_rag_documents.KOSHA_INDEX_JSON_PATH', index_path)
    monkeypatch.setattr('build_rag_documents.RAG_DOCUMENTS_PATH', out)
    monkeypatch.setattr('build_rag_documents.load_manifest', lambda: {'sources': []})

    docs = build_documents()

    assert len(docs) == 1
    assert docs[0]['source'] == 'KOSHA'
    assert docs[0]['metadata_only'] is True
    assert docs[0]['extraction_status'] == 'failed'
