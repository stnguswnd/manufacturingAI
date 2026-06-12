from __future__ import annotations

import argparse

from rag_pipeline_utils import RAG_CHUNKS_PATH, RAG_DATA_DIR, read_jsonl


def default_embedding_function():
    """Return a Chroma embedding function if configured.

    TODO: wire this to the project-level embedding provider. For now this
    script can use Chroma's default embedding behavior when available.
    """
    return None


def index_chunks(*, persist_dir: str | None = None, collection_name: str = 'manufacturing_rag') -> int:
    try:
        import chromadb
    except Exception as exc:
        raise SystemExit('chromadb is not installed. Install it only if you want optional Chroma indexing.') from exc
    chunks = read_jsonl(RAG_CHUNKS_PATH)
    directory = persist_dir or str(RAG_DATA_DIR / 'vector_db' / 'chroma')
    client = chromadb.PersistentClient(path=directory)
    collection = client.get_or_create_collection(name=collection_name, embedding_function=default_embedding_function())
    ids = [row['chunk_id'] for row in chunks]
    documents = [row['text'] for row in chunks]
    metadatas = [
        {k: v for k, v in row.items() if k not in {'text'} and not isinstance(v, list)}
        | {
            'failure_modes': ','.join(row.get('failure_modes') or []),
            'related_signals': ','.join(row.get('related_signals') or []),
        }
        for row in chunks
    ]
    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description='Optionally index rag_chunks.jsonl into Chroma.')
    parser.add_argument('--persist-dir')
    parser.add_argument('--collection', default='manufacturing_rag')
    args = parser.parse_args()
    count = index_chunks(persist_dir=args.persist_dir, collection_name=args.collection)
    print(f'indexed_chunks={count}')


if __name__ == '__main__':
    main()
