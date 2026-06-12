from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rag_pipeline_utils import (
    KOSHA_INDEX_JSON_PATH,
    PROJECT_ROOT,
    RAG_DOCUMENTS_PATH,
    clean_text,
    ensure_dirs,
    extract_text,
    load_manifest,
    read_jsonl,
    write_jsonl,
)


def static_document(source: dict[str, Any], *, keep_metadata_only: bool) -> dict[str, Any] | None:
    path = PROJECT_ROOT / source['save_path']
    text = ''
    status = 'missing'
    error = 'source file does not exist'
    if path.exists():
        text, status, error = extract_text(path)
    metadata_only = status != 'success' or not text
    if metadata_only and not keep_metadata_only:
        return None
    return {
        'doc_id': source['id'],
        'source': source['source'],
        'title': source.get('title') or source['id'],
        'url': source.get('url'),
        'local_path': source['save_path'],
        'doc_type': source.get('doc_type'),
        'safety_gate': source.get('safety_gate'),
        'failure_modes': source.get('failure_modes') or [],
        'related_signals': source.get('related_signals') or [],
        'project_priority': source.get('project_priority') or 'medium',
        'retrieval_scope': source.get('retrieval_scope') or 'default',
        'use_case': source.get('use_case') or '',
        'search_policy': source.get('search_policy') or '',
        'text': clean_text(text),
        'extraction_status': status,
        'extraction_error': error,
        'metadata_only': metadata_only,
    }


def kosha_documents(*, keep_metadata_only: bool) -> list[dict[str, Any]]:
    if not KOSHA_INDEX_JSON_PATH.exists():
        return []
    import json

    index = json.loads(KOSHA_INDEX_JSON_PATH.read_text(encoding='utf-8'))
    rows = index.get('documents') if isinstance(index, dict) else []
    documents: list[dict[str, Any]] = []
    for idx, row in enumerate(rows or [], 1):
        local_path = row.get('local_path')
        path = PROJECT_ROOT / local_path if local_path else None
        text = ''
        status = 'missing'
        error = 'downloaded file is missing'
        if path and path.exists():
            text, status, error = extract_text(path)
        metadata_only = status != 'success' or not text
        if metadata_only and not keep_metadata_only:
            continue
        doc_no = str(row.get('techGdlnNo') or '').strip()
        doc_id = f'kosha_{doc_no}' if doc_no else f'kosha_{idx:05d}'
        documents.append({
            'doc_id': doc_id,
            'source': 'KOSHA',
            'title': row.get('techGdlnNm') or doc_id,
            'url': row.get('fileDownlUrl'),
            'local_path': local_path,
            'doc_type': row.get('doc_type'),
            'safety_gate': row.get('safety_gate'),
            'failure_modes': row.get('failure_modes') or [],
            'related_signals': row.get('related_signals') or [],
            'project_priority': row.get('project_priority') or 'medium',
            'retrieval_scope': row.get('retrieval_scope') or 'restricted',
            'use_case': row.get('use_case') or '',
            'search_keyword': row.get('search_keyword'),
            'text': clean_text(text),
            'extraction_status': status,
            'extraction_error': error or row.get('download_error'),
            'metadata_only': metadata_only,
        })
    return documents


def build_documents(*, keep_metadata_only: bool = True) -> list[dict[str, Any]]:
    ensure_dirs()
    manifest = load_manifest()
    documents: list[dict[str, Any]] = []
    for source in manifest.get('sources') or []:
        doc = static_document(source, keep_metadata_only=keep_metadata_only)
        if doc:
            documents.append(doc)
    documents.extend(kosha_documents(keep_metadata_only=keep_metadata_only))
    write_jsonl(RAG_DOCUMENTS_PATH, documents)
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description='Build RAG document JSONL from downloaded OSHA/Haas/KOSHA raw files.')
    parser.add_argument('--drop-metadata-only', action='store_true', help='Do not include documents with failed text extraction.')
    args = parser.parse_args()
    documents = build_documents(keep_metadata_only=not args.drop_metadata_only)
    print(f'documents={len(documents)} -> {RAG_DOCUMENTS_PATH}')


if __name__ == '__main__':
    main()
