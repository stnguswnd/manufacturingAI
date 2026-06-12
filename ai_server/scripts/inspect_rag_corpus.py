from __future__ import annotations

import json
from collections import Counter

from rag_pipeline_utils import KOSHA_INDEX_JSON_PATH, RAG_CHUNKS_PATH, RAG_CORPUS_REPORT_PATH, RAG_DOCUMENTS_PATH, counters, read_jsonl


def markdown_counter(title: str, counter: Counter) -> list[str]:
    lines = [f'## {title}', '']
    if not counter:
        return lines + ['- 없음', '']
    for key, count in counter.most_common():
        lines.append(f'- {key}: {count}')
    lines.append('')
    return lines


def build_report() -> str:
    documents = read_jsonl(RAG_DOCUMENTS_PATH)
    chunks = read_jsonl(RAG_CHUNKS_PATH)
    failures = [doc for doc in documents if doc.get('extraction_status') != 'success']
    high_priority = [doc for doc in documents if doc.get('project_priority') == 'high']
    restricted = [doc for doc in documents if doc.get('retrieval_scope') == 'restricted']
    download_failures = []
    if KOSHA_INDEX_JSON_PATH.exists():
        index = json.loads(KOSHA_INDEX_JSON_PATH.read_text(encoding='utf-8'))
        download_failures = (index.get('api_failures') or []) + (index.get('file_failures') or [])

    lines = [
        '# RAG Corpus Report',
        '',
        f'- 전체 문서 수: {len(documents)}',
        f'- 전체 chunk 수: {len(chunks)}',
        '',
    ]
    lines += markdown_counter('source별 문서 수', counters(documents, 'source'))
    lines += markdown_counter('source별 chunk 수', counters(chunks, 'source'))
    lines += markdown_counter('doc_type별 문서 수', counters(documents, 'doc_type'))
    lines += markdown_counter('project_priority별 문서 수', counters(documents, 'project_priority'))
    lines += markdown_counter('retrieval_scope별 문서 수', counters(documents, 'retrieval_scope'))
    lines += markdown_counter('failure_mode별 chunk 수', counters(chunks, 'failure_modes'))
    lines += markdown_counter('safety_gate별 chunk 수', counters(chunks, 'safety_gate'))

    lines += ['## 텍스트 추출 실패 문서 목록', '']
    lines += [f'- {doc.get("doc_id")}: {doc.get("title")} ({doc.get("extraction_error")})' for doc in failures] or ['- 없음']
    lines.append('')
    lines += ['## 다운로드 실패 문서 목록', '']
    lines += [f'- {item.get("title") or item.get("keyword")}: {item.get("error") or item.get("response_preview")}' for item in download_failures] or ['- 없음']
    lines.append('')
    lines += ['## high priority 문서 목록', '']
    lines += [f'- {doc.get("doc_id")}: {doc.get("title")} [{doc.get("source")} / {doc.get("doc_type")}]' for doc in high_priority] or ['- 없음']
    lines.append('')
    lines += ['## restricted 문서 목록', '']
    lines += [f'- {doc.get("doc_id")}: {doc.get("title")} [{doc.get("source")} / {doc.get("doc_type")}]' for doc in restricted] or ['- 없음']
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    report = build_report()
    RAG_CORPUS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAG_CORPUS_REPORT_PATH.write_text(report, encoding='utf-8')
    print(f'wrote {RAG_CORPUS_REPORT_PATH}')


if __name__ == '__main__':
    main()
