from __future__ import annotations

import argparse

from rag_pipeline_utils import RAG_CHUNKS_PATH, RAG_DOCUMENTS_PATH, read_jsonl, sentence_aware_chunks, write_jsonl


def build_chunks(*, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[dict]:
    documents = read_jsonl(RAG_DOCUMENTS_PATH)
    chunks: list[dict] = []
    for doc in documents:
        if doc.get('metadata_only'):
            continue
        text = doc.get('text') or ''
        for index, chunk_text in enumerate(sentence_aware_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap), 1):
            chunks.append({
                'chunk_id': f'{doc["doc_id"]}_{index:04d}',
                'doc_id': doc['doc_id'],
                'source': doc.get('source'),
                'title': doc.get('title'),
                'url': doc.get('url'),
                'local_path': doc.get('local_path'),
                'doc_type': doc.get('doc_type'),
                'safety_gate': doc.get('safety_gate'),
                'failure_modes': doc.get('failure_modes') or [],
                'related_signals': doc.get('related_signals') or [],
                'project_priority': doc.get('project_priority') or 'medium',
                'retrieval_scope': doc.get('retrieval_scope') or 'default',
                'use_case': doc.get('use_case') or '',
                'chunk_index': index - 1,
                'text': chunk_text,
            })
    write_jsonl(RAG_CHUNKS_PATH, chunks)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description='Build chunk JSONL from rag_documents.jsonl.')
    parser.add_argument('--chunk-size', type=int, default=1000)
    parser.add_argument('--chunk-overlap', type=int, default=150)
    args = parser.parse_args()
    chunks = build_chunks(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    print(f'chunks={len(chunks)} -> {RAG_CHUNKS_PATH}')


if __name__ == '__main__':
    main()
