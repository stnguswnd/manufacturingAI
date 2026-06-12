from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS_DIR))

from build_rag_chunks import build_chunks
from rag_pipeline_utils import write_jsonl


def test_chunking_splits_long_documents_and_preserves_metadata(tmp_path, monkeypatch):
    documents_path = tmp_path / 'rag_documents.jsonl'
    chunks_path = tmp_path / 'rag_chunks.jsonl'
    long_text = '토크가 높으면 공구 마모와 발열이 증가할 수 있습니다. ' * 80
    write_jsonl(documents_path, [
        {
            'doc_id': 'doc_a',
            'source': 'KOSHA',
            'title': '공작기계 정비 작업 안전에 관한 지침',
            'url': 'https://example.com/a',
            'local_path': 'raw/a.html',
            'doc_type': 'korean_maintenance_guidance',
            'safety_gate': 'maintenance_check',
            'failure_modes': ['OSF', 'TWF'],
            'related_signals': ['torque_nm'],
            'project_priority': 'high',
            'retrieval_scope': 'default',
            'use_case': '정비 점검',
            'metadata_only': False,
            'text': long_text,
        },
        {
            'doc_id': 'doc_meta',
            'source': 'KOSHA',
            'title': 'metadata only',
            'metadata_only': True,
            'text': '',
        },
    ])
    monkeypatch.setattr('build_rag_chunks.RAG_DOCUMENTS_PATH', documents_path)
    monkeypatch.setattr('build_rag_chunks.RAG_CHUNKS_PATH', chunks_path)

    chunks = build_chunks(chunk_size=500, chunk_overlap=80)

    assert len(chunks) > 1
    assert chunks[0]['chunk_id'] == 'doc_a_0001'
    assert all(chunk['doc_id'] == 'doc_a' for chunk in chunks)
    assert chunks[0]['doc_type'] == 'korean_maintenance_guidance'
    assert chunks[0]['failure_modes'] == ['OSF', 'TWF']
    assert not any(chunk['doc_id'] == 'doc_meta' for chunk in chunks)
