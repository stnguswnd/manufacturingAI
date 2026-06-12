# Chroma PDF Ingestion Strategy

작성일: 2026-06-12

## 1. 목표

현재 경량 lexical RAG를 Chroma 기반 semantic RAG로 교체하고, Streamlit에서 PDF를 업로드하면 자동으로 텍스트 추출, chunking, embedding, Chroma 저장, Agent 검색에 반영되도록 만든다.

## 2. 현재 상태

현재 RAG는 진짜 vector DB가 아니다.

```text
data/processed_docs/chunks.jsonl
→ scripts/ingest_docs.py
→ ai_server/storage/vector_store/chunks.jsonl
→ RagService가 메모리 로드
→ BM25 계열 lexical scoring
```

한계:

- embedding 없음
- semantic similarity 없음
- PDF 업로드 API 없음
- 문서 삭제/재색인/metadata 관리 약함
- `vector_store`라는 이름과 실제 구현이 맞지 않음

## 3. Chroma 선택 이유

Chroma는 문서, metadata, embedding을 collection 단위로 함께 관리할 수 있다. 현재 프로젝트의 제조 문서 RAG에는 다음 장점이 있다.

- 로컬 persistent directory로 빠르게 데모 가능
- metadata filter를 유지하기 쉬움
- PDF 업로드 후 chunk를 collection에 upsert하기 쉬움
- `RagService.search()` 경계만 유지하면 Agent 코드는 크게 바꾸지 않아도 됨
- 나중에 Chroma Cloud 또는 pgvector로 이전하기 쉬움

## 4. 전체 처리 흐름

```text
PDF Upload
→ file validation
→ PDF text extraction
→ page-aware chunking
→ chunk metadata 생성
→ embedding 생성
→ Chroma collection upsert
→ document registry 저장
→ RagService.search()에서 Chroma query
→ Agent answer에 citations 반영
```

## 5. 데이터 저장 구조

```text
ai_server/storage/
  chroma/
    manufacturing_docs/        # Chroma persistent collection
  uploaded_docs/
    {document_id}/
      original.pdf
      extracted_text.json
      chunks.jsonl
```

SQLite에는 문서 registry를 추가한다.

```sql
CREATE TABLE IF NOT EXISTS rag_documents (
    document_id TEXT PRIMARY KEY,
    source TEXT,
    document_title TEXT NOT NULL,
    doc_type TEXT,
    equipment_type TEXT,
    section TEXT,
    version TEXT,
    effective_date TEXT,
    language TEXT DEFAULT 'ko',
    file_name TEXT,
    file_path TEXT,
    chunk_count INTEGER DEFAULT 0,
    embedding_model TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

Chroma metadata:

```json
{
  "document_id": "doc_xxx",
  "chunk_id": "doc_xxx_0001",
  "source": "internal",
  "document_title": "CNC 정비 매뉴얼",
  "doc_type": "maintenance_manual",
  "equipment_type": "CNC",
  "section": "spindle",
  "page_start": 3,
  "page_end": 4,
  "language": "ko",
  "version": "v1.0",
  "effective_date": "2026-06-12"
}
```

## 6. API 설계

### 6.1 PDF 업로드

```text
POST /rag/documents/upload
Content-Type: multipart/form-data
```

Form fields:

```text
file: PDF
source
document_title
doc_type
equipment_type
section
version
effective_date
language
```

응답:

```json
{
  "document_id": "doc_xxx",
  "chunk_count": 42,
  "embedding_model": "text-embedding-3-small",
  "status": "indexed"
}
```

### 6.2 문서 목록

```text
GET /rag/documents
```

### 6.3 문서 삭제

```text
DELETE /rag/documents/{document_id}
```

동작:

- Chroma에서 해당 `document_id` chunk 삭제
- SQLite registry 삭제 또는 soft delete
- 업로드 파일 삭제 여부는 `delete_file=true` 옵션으로 제어

### 6.4 검색

기존 API 유지:

```text
POST /rag/search
```

내부 구현만 Chroma query로 교체한다.

## 7. 코드 구조

```text
ai_server/app/services/
  embedding_service.py       # OpenAI embedding adapter
  document_ingestion_service.py
  chroma_rag_service.py
  rag_service.py             # retrieval interface

ai_server/app/storage/
  sqlite_store.py            # rag_documents registry 추가

streamlit_app.py             # PDF upload UI, document list UI
```

## 8. Embedding 전략

기본값:

```env
RAG_BACKEND=chroma
CHROMA_PERSIST_DIR=ai_server/storage/chroma
CHROMA_COLLECTION=manufacturing_docs
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_BATCH_SIZE=64
```

전략:

- OpenAI embedding은 직접 생성해서 Chroma에 `embeddings`로 넣는다.
- Chroma의 default embedding function에 의존하지 않는다.
- 이유:
  - 사용 토큰과 비용을 직접 계산/기록할 수 있음
  - embedding model을 config로 통제 가능
  - 나중에 local embedding으로 교체하기 쉬움

## 9. Chunking 전략

초기 구현:

```text
chunk_size_chars=1200
chunk_overlap_chars=180
page-aware chunking
```

이유:

- 제조 매뉴얼은 표/절차/주의사항이 섞여 있어 너무 작은 chunk는 문맥 손실이 큼
- 너무 큰 chunk는 retrieval precision과 token cost를 악화시킴

추후 개선:

- heading-aware chunking
- table extraction
- OCR fallback
- chunk quality score

## 10. Metadata filter 전략

Agent의 `plan.rag_filters`와 현재 도메인 context를 Chroma `where` filter로 연결한다.

예:

```json
{
  "equipment_type": "CNC",
  "doc_type": "safety_procedure"
}
```

초기에는 exact match만 적용하고, 결과가 너무 적으면 filter를 완화한다.

검색 흐름:

```text
1. strict metadata filter + vector search
2. 결과 부족 시 equipment_type 제거
3. 그래도 부족하면 metadata 없이 vector search
4. score threshold 적용
```

## 11. Hybrid Search 전략

초기 구현은 Chroma semantic search만 사용한다.

다만 제조 용어에는 `LOTO`, `OSF`, `TWF`, 설비 코드처럼 lexical exact match가 중요하므로 이후 아래 구조로 확장한다.

```text
semantic_score = Chroma distance 기반
keyword_score = BM25/term overlap
final_score = 0.75 * semantic_score + 0.25 * keyword_score
```

## 12. Streamlit UI 전략

RAG 탭을 확장한다.

추가 UI:

- PDF uploader
- document_title 입력
- source 입력
- doc_type selectbox
- equipment_type selectbox
- section 입력
- version/effective_date 입력
- Upload & Index 버튼
- Indexed documents table
- Delete document 버튼
- Search query 테스트

Agent 탭은 기존처럼 RAG를 자동 사용한다.

## 13. Migration 전략

1. `chromadb` 의존성 추가
2. 기존 `chunks.jsonl` 샘플 문서를 Chroma로 재색인하는 script 추가
3. `RagService`를 backend switch 가능하게 변경
4. 기본값을 `RAG_BACKEND=chroma`로 전환
5. 기존 lexical RAG는 fallback/test backend로 유지

## 14. 테스트 전략

필수 테스트:

- PDF text extraction test
- chunking overlap test
- Chroma upsert/search test
- metadata filter test
- document delete cascade test
- `/rag/documents/upload` API test
- Streamlit 없이 service-level upload test
- Agent RAG integration test

테스트에서는 OpenAI embedding 비용이 들지 않게 deterministic fake embedding을 사용한다.

## 15. 구현 순서

### Phase 1: Chroma backend skeleton

- `chromadb` 추가
- `ChromaRagService` 추가
- fake embedding 기반 테스트 추가
- 기존 sample chunks를 Chroma에 넣는 script 추가

### Phase 2: OpenAI embedding

- `EmbeddingService` 추가
- `EMBEDDING_MODEL`, batch size config 추가
- embedding usage/cost record 추가

### Phase 3: PDF upload API

- multipart upload endpoint 추가
- PDF 저장
- text extraction
- chunking
- Chroma upsert
- SQLite document registry 저장

### Phase 4: Streamlit UI

- PDF upload UI
- document list/delete UI
- Chroma search result 표시

### Phase 5: Agent integration hardening

- metadata filter relaxation
- hybrid search 준비
- citation 품질 개선
- golden retrieval tests

## 16. 포트폴리오 문장

```text
기존 JSONL 기반 lexical RAG를 Chroma 기반 semantic vector search로 확장하고, PDF 업로드 시 텍스트 추출·chunking·embedding·metadata indexing을 자동화했습니다. 제조 문서의 설비 유형, 문서 종류, 섹션, 버전 정보를 metadata로 유지해 Agent가 현재 공정 상황과 safety gate에 맞는 근거 문서를 검색하도록 설계했습니다.
```
