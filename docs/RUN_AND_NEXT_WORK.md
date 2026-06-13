# 구동 방법 및 추가 작업 가이드

작성일: 2026-06-12

## 이번에 반영한 보완 사항

- API 보안 설정 추가
  - `API_AUTH_ENABLED`, `API_KEY`, `CORS_ALLOW_ORIGINS` 환경변수를 추가했다.
  - `/predict`, `/agent/send`, `/rag/search`, `/history`, 도메인 catalog API는 API key 보호를 적용할 수 있다.
- 입력 검증 강화
  - AI4I `type`을 `L/M/H`로 제한했다.
  - 온도, 회전수, 토크, 공구 마모 시간 범위를 Pydantic에서 검증한다.
  - message, inspection notes, session id, top_k 길이와 범위를 제한했다.
- 오류 응답 정리
  - 내부 예외 메시지를 그대로 노출하지 않도록 공통 예외 핸들러를 추가했다.
  - 모델/RAG/안전 검증 오류를 도메인 예외로 분리했다.
- RAG 개선
  - 단순 TF-IDF에 가까운 점수 계산을 BM25 형태로 개선했다.
  - 검색 결과가 없을 때 점수 0 문서를 억지로 반환하지 않도록 바꿨다.
- 예측 모델 보완
  - 학습 시 train/test split 기반 성능 지표를 `ai_server/storage/models/ai4i_metrics.json`에 저장한다.
  - 입력값이 학습 데이터 분포 외곽이면 `input_warnings`로 경고한다.
- Artifact Quality Gate / Answer Text Review 추가
  - Prediction, Evidence, SafetyArtifact는 생성 직후 quality gate를 통과해야 한다.
  - 최종 답변 문장은 Answer Text Review에서 debug leak, safety 누락, 금지 표현을 검증한다.
  - rewrite/RAG rerun/safety rerun은 `RuntimeArtifact`의 per-action retry budget으로 제한한다.
- LLM 토큰/비용 측정 추가
  - OpenAI 응답의 `usage`에서 input/output/cached token을 읽어 `llm_usage`에 기록한다.
  - 내부 모델 가격표 기준으로 요청별 예상 비용을 계산한다.
  - OpenTelemetry는 토큰을 직접 계산하지 않고 관측/수집용으로 사용한다.
  - LLM 호출 단위 span에는 `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.total_tokens`, `gen_ai.usage.estimated_cost_usd`, latency를 기록한다.
  - Agent 실행 단위 span에는 LLM 호출 수, re-plan 횟수, 총 토큰, 예상 비용을 집계해서 기록한다.
  - Streamlit과 History에서 호출 수, re-plan 횟수, input/output/total tokens, 예상 비용을 확인할 수 있다.
- 이력 저장 개선
  - JSONL append 방식 대신 SQLite 저장소를 사용한다.
  - 기본 DB 경로는 `ai_server/storage/history/agent_runs.sqlite3`이다.
- Streamlit 테스트 UI 추가
  - `streamlit_app.py`에서 Agent, Prediction, RAG, History를 테스트할 수 있다.
- 최소 회귀 테스트 추가
  - `ai_server/tests/test_rag_and_safety.py`를 추가했다.

## 로컬 실행 순서

### 1. 의존성 설치

```bash
cd ai_server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 환경변수 준비

프로젝트 루트 또는 `ai_server` 안에서 `.env.example`을 복사한다.

```bash
cp .env.example .env
```

로컬 테스트에서도 OpenAI 설정이 필요하다.

```env
API_AUTH_ENABLED=false
LLM_PROVIDER=openai
LLM_MODEL=gpt-5.4-mini
OPENAI_API_KEY=your_openai_api_key_here
```

외부에 노출하거나 팀원이 접근하는 환경에서는 반드시 바꿔야 한다.

```env
API_AUTH_ENABLED=true
API_KEY=충분히_긴_랜덤_키
CORS_ALLOW_ORIGINS=http://localhost:8501,https://your-ui-domain.example
```

### 3. 데이터와 모델 준비

```bash
cd ai_server
python scripts/train_ai4i_model.py
python scripts/ingest_docs.py --sample-only
```

또는 프로젝트 루트에서:

```bash
make prepare
```

### 4. FastAPI 서버 실행

```bash
cd ai_server
uvicorn app.main:app --reload --port 8000
```

또는 프로젝트 루트에서:

```bash
make run-ai
```

확인:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

### 5. Streamlit UI 실행

프로젝트 루트에서 실행한다.

```bash
streamlit run streamlit_app.py
```

또는:

```bash
make run-ui
```

브라우저에서 다음 주소를 연다.

```text
http://localhost:8501
```

`API_AUTH_ENABLED=true`로 실행했다면 Streamlit 왼쪽 사이드바의 `API key`에 `.env`의 `API_KEY`를 입력한다.

## 테스트 실행

```bash
cd ai_server
pytest
```

## Streamlit에서 확인할 것

- Agent 탭
  - 기본 질문으로 실행했을 때 `안전 확인`, `권장 조치`, `주의 사항`이 포함되는지 본다.
  - 보고서 형식 요청은 별도 체크박스 없이 일반 답변 본문으로 정리되는지 본다.
  - 위험 입력을 넣었을 때 warnings에 학습 분포 경고가 나오는지 본다.
  - 근거가 부족한 질문을 넣었을 때 진행 과정에 `Supervisor Re-plan`이 표시되는지 본다.
  - LLM 사용 시 상단 metric에서 `LLM calls`, `Re-plans`, input/output/total tokens, 예상 비용이 표시되는지 본다.
- Prediction 탭
  - 정상 범위 입력이 예측되는지 본다.
  - 극단값을 넣었을 때 validation 또는 `input_warnings`가 나오는지 본다.
- RAG 탭
  - 관련 query는 문서가 나오고, 무관한 query는 빈 결과가 나오는지 본다.
- History 탭
  - Agent 실행 후 이력이 SQLite에 저장되고 조회되는지 본다.

## 사용자가 추가로 해야 할 작업

### 1. 운영 보안 설정

- 운영/공유 환경에서는 `API_AUTH_ENABLED=true`를 반드시 켠다.
- `API_KEY`는 랜덤하고 긴 값으로 설정한다.
- `CORS_ALLOW_ORIGINS`는 실제 UI 도메인만 허용한다.
- `/history` 응답에 저장되는 현장 메모나 공정 데이터가 민감하다면 마스킹 정책을 추가해야 한다.

### 2. 실제 문서 인덱스 구축

- 현재 샘플 문서만으로는 제조 현장 근거로 부족하다.
- 설비 매뉴얼, 사내 안전 절차, 점검표, 작업표준서, 변경 이력을 별도 metadata와 함께 ingest해야 한다.
- 문서 metadata에는 최소한 `source`, `document_title`, `doc_type`, `equipment_type`, `section`, `version`, `effective_date`를 넣는 것이 좋다.

### 3. 모델 검증 강화

- AI4I 모델은 공개 데이터 기반 데모 모델이다.
- 실제 설비 데이터로 재학습하거나, 최소한 현장 데이터로 검증해야 한다.
- false negative가 위험한 업무라면 recall 중심으로 threshold를 재설정해야 한다.
- `ai_server/storage/models/ai4i_metrics.json`의 성능 지표를 검토하고 모델 카드로 정리해야 한다.

### 4. 안전 정책 확정

- LOTO, 전기 격리, 회전부 방호, 고온, 비상 대응 기준은 회사/현장 기준에 맞게 확정해야 한다.
- `ai_server/domain/safety_gate_matrix.yaml`의 문구를 안전관리자 검토 후 수정해야 한다.
- 안전 관련 답변에서 반드시 들어가야 하는 문장과 절대 나오면 안 되는 문장을 golden test로 추가해야 한다.

### 5. 테스트 확대

- 현재 추가된 테스트는 시작점일 뿐이다.
- API endpoint 테스트, 모델 threshold 테스트, domain catalog test, history store test를 더 추가해야 한다.
- 중요한 제조 질의는 `data/golden`에 케이스로 저장하고 CI에서 회귀 평가해야 한다.

### 6. 배포 전 점검

- `.env`가 Git에 포함되지 않는지 확인한다.
- SQLite 대신 PostgreSQL을 사용할지 결정한다.
- request id, latency, RAG hit rate, LLM error rate, safety validation failure rate를 로그/모니터링에 추가한다.
- Docker compose의 `backend_spring` 서비스는 현재 이 저장소에 소스가 없으므로 실제 배포 전 compose 구성을 재검토해야 한다.
