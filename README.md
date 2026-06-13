# Manufacturing AI Agent

AI4I 공정 데이터 예측, OSHA/Haas/KOSHA 문서 RAG, 제조 도메인 YAML 정책,
Artifact 품질 검증, 안전 게이트 검증, 답변 문장 검증을 결합한 제조 특화 AI Agent 서버입니다.

이 프로젝트는 단순 RAG 챗봇이 아니라, 제조 현장에서 필요한 판단 흐름을
분리해 다룹니다.

```text
AI4I process data
  -> prediction tool

OSHA / Haas / KOSHA documents
  -> RAG Evidence SubAgent

Domain YAML
  -> equipment taxonomy
  -> failure mode catalog
  -> safety gate matrix
  -> action policy

RootManufacturingGraph
  -> LangGraph top-level orchestration
```

Agent는 설비를 직접 제어하지 않습니다. 예측, 문서 근거 검색, 점검 절차,
안전 확인, 조치 권고를 보조하며 실제 작업 판단은 담당자와 안전관리자
확인을 전제로 합니다.

이 프로젝트는 자율형 멀티 에이전트가 아닙니다.
제조 안전 도메인에 맞춰 Prediction, Adaptive RAG Evidence, Safety Contract,
Answer Text Review를 Artifact contract와 LangGraph conditional edge로 제어하는
bounded agentic workflow입니다.

---

## 1. Portfolio Summary

한 줄 소개:

```text
제조 안전 도메인에 맞춰 Prediction, Adaptive RAG Evidence, Safety Contract,
Answer Text Review를 Artifact contract와 LangGraph conditional edge로 제어하는
bounded agentic workflow입니다.
```

강조할 점:

- FastAPI 기반 제품 API: `POST /agent/send`
- LangGraph `RootManufacturingGraph(StateGraph)` v3 artifact-only runtime
- bounded workflow component:
  - `ContextSubAgent`
  - `PlanningSubAgent`
  - `RagEvidenceSubAgent` = Adaptive RAG 실행 단위
  - `SafetyContractSubAgent`
  - Artifact Quality Gates
  - `AnswerTextReview`
  - `MemorySubAgent`
- AI4I prediction과 RAG Evidence layer 분리
- `RequestArtifact`, `ContextArtifact`, `PlanningArtifact`, `PredictionArtifact`, `EvidenceArtifact`, `SafetyArtifact`, `AnswerDraft`, `ValidationReport`, `ResponseArtifact`, `RuntimeArtifact` 기반 내부 contract
- RAG corpus는 AI4I CSV가 아니라 OSHA/Haas/KOSHA 문서
- Chroma collection: `manufacturing_rag`
- 현재 corpus 상태: `rag_chunks.jsonl 727` / `Chroma 727`
- safety gate 기반 검증 및 unsafe response 차단
- Artifact Quality Gate와 Answer Text Review 기반 bounded validation
- user/session 기반 LangGraph checkpoint memory
- LLM usage/token/cost 추적
- Streamlit 기반 테스트 UI

포트폴리오 검토자는 아래 문서부터 보면 됩니다.

1. `docs/PORTFOLIO_REVIEW_GUIDE.md`
2. `docs/PORTFOLIO_ROADMAP.md`
3. `docs/DEMO_SCRIPT.md`
4. `docs/LANGGRAPH_FINAL_ARCHITECTURE.md`
5. `docs/rag_evidence_orchestration.md`
6. `docs/archive/TROUBLESHOOTING_AND_ARCHITECTURE_EVOLUTION_2026-06-13.md`

---

## 2. Current Runtime Architecture

```text
POST /agent/send
  -> RootManufacturingGraph(StateGraph)
      -> request_context
      -> planning_router
      -> conditional:
           prediction_node -> prediction_quality_gate -> planning_router
           rag_evidence_subagent -> evidence_quality_gate -> planning_router
           safety_contract_subagent -> safety_contract_gate -> planning_router
           fast_answer -> output_policy_gate -> response_packager
           clarification_response -> response_packager
           answer_compose
      -> answer_compose
      -> answer_text_review
      -> conditional:
           pass -> response_packager
           rewrite_only -> invalidate_rewrite -> answer_rewrite -> answer_text_review
           rerun_rag -> invalidate_rag_downstream -> rag_evidence_subagent -> planning_router
           rerun_safety -> invalidate_safety_downstream -> safety_contract_subagent -> planning_router
           block -> safe_block_response -> response_packager
           max_retry_exceeded -> safe_review_fallback -> response_packager
      -> memory_writer
      -> audit_persistence
```

`/rag/search`는 agent 내부 답변 경로가 아니라 API/debug seam입니다.

```text
POST /rag/search
  -> RagService
  -> ChromaRetriever
  -> Chroma manufacturing_rag
```

---

## 3. Main Capabilities

| Capability | Implementation | Why it matters |
| --- | --- | --- |
| AI4I prediction | `PredictionService` | 공정 feature 기반 고장 가능성 예측 |
| AI4I feature audit | `ContextSubAgent` | feature가 부족하면 예측하지 않고 clarification |
| Adaptive RAG Evidence | `RagEvidenceSubAgent`, `RagQueryPlanner`, `RagFanoutPolicy` | safety/prediction/troubleshooting 질문별 retrieval profile, fan-out, filtering, grading, citation |
| Safety gates | `safety_gate_matrix.yaml`, `SafetyContractSubAgent` | EvidenceArtifact를 소비해 LOTO, 회전부 방호, 전기 격리 등 필수 확인을 자연어 contract로 변환 |
| Safety validation | `SafetyValidationService` | 필수 안전 내용 누락 또는 금지 표현 차단 |
| Domain modeling | `ai_server/domain/*.yaml` | 설비/고장모드/조치/문서 정책을 코드 밖에서 관리 |
| Memory | LangGraph checkpoint + `MemorySubAgent` | user/session별 follow-up context 유지 |
| Observability | `llm_usage`, trace, history | token/cost/route/debug 정보를 내부 기록으로 보존 |

---

## 4. AI4I And RAG Boundary

AI4I CSV/process data는 prediction input입니다. Vector DB에 넣지 않습니다.

```text
AI4I Type, Air temperature, Process temperature, Rotational speed, Torque, Tool wear
  -> PredictionService
  -> failure probability / failure mode scores
```

RAG corpus는 외부 제조 문서입니다.

```text
OSHA
Haas
KOSHA
  -> rag_chunks.jsonl
  -> Chroma manufacturing_rag
  -> RAG Evidence SubAgent
```

AI4I prediction이 없는 RAG-only safety 질문에는 AI4I 예측 문구, 고장 확률,
TWF/OSF/HDF/PWF 확률을 출력하지 않습니다.

---

## 5. Setup

Python 앱, scripts, notebook은 모두 같은 가상환경과 env 파일을 공유합니다.

- 가상환경: `ai_server/.venv`
- env 파일: `ai_server/.env`
- env 예시: `ai_server/.env.example`
- 의존성: `ai_server/requirements.txt`

```bash
cd ai_server
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`ai_server/.env`에 API key와 runtime 값을 넣습니다. 실제 key는 commit하지 않습니다.

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-5.4-mini
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://api.openai.com/v1
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=manufacturing-agent-eval
RAG_EMBEDDING_MODEL=text-embedding-3-small
CHROMA_COLLECTION=manufacturing_rag
CHROMA_PERSIST_DIR=ai_server/data/vector_db/chroma
```

기존에 repo root의 `.env`를 쓰고 있었다면 `ai_server/.env`로 옮겨서 하나만 유지합니다.

AI4I model bundle이 없는 환경에서는 명시적으로 학습 스크립트를 실행해야 합니다.
runtime에서 자동 train하지 않습니다.

```bash
cd ai_server
.venv/bin/python scripts/train_ai4i_model.py
```

Chroma vector DB는 git ignored입니다. 새 환경에서는 기존 JSONL chunks로
재색인합니다.

```bash
cd ai_server
.venv/bin/python scripts/index_rag_chunks_chroma.py --reset
```

자세한 절차는 `docs/RAG_INDEX_RUNBOOK.md`를 참고하세요.

---

## 6. Run

FastAPI:

```bash
cd ai_server
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Streamlit UI:

```bash
cd ..
ai_server/.venv/bin/python -m streamlit run streamlit_app.py \
  --server.address 127.0.0.1 \
  --server.port 8501
```

URLs:

```text
FastAPI:   http://127.0.0.1:8000
API docs:  http://127.0.0.1:8000/docs
Streamlit: http://127.0.0.1:8501
```

---

## 7. Main APIs

Health:

```bash
curl http://127.0.0.1:8000/health
```

Agent:

```bash
curl -X POST http://127.0.0.1:8000/agent/send \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo-user",
    "session_id": "demo-session",
    "message": "AI4I 데이터가 Type=M, Air temperature=300.2K, Process temperature=309.0K, Rotational speed=1480rpm, Torque=34Nm, Tool wear=235min일 때 공구 마모 고장 가능성을 예측하고, 공구 교체 전 확인해야 할 항목을 알려줘.",
    "top_k": 5,
    "mode": "auto",
    "llm_model": "gpt-5.4-mini"
  }'
```

RAG debug:

```bash
curl -X POST http://127.0.0.1:8000/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query":"Haas spindle load tool wear troubleshooting torque", "top_k":5}'
```

---

## 8. Demo Questions

AI4I + RAG:

```text
AI4I 데이터가 Type=M, Air temperature=300.2K, Process temperature=309.0K,
Rotational speed=1480rpm, Torque=34Nm, Tool wear=235min일 때
공구 마모 고장 가능성을 예측하고, 공구 교체 전 확인해야 할 항목을 알려줘.
```

RAG-only safety:

```text
드릴기 작업 전에 공작물 고정 상태, 방호덮개, 비상정지장치를 어떻게 확인해야 하는지 알려줘.
```

Troubleshooting RAG:

```text
Haas 밀 장비에서 스핀들 이상음, 진동, 경보가 발생했을 때 우선 확인해야 할 항목을 알려줘.
```

AI4I clarification:

```text
AI4I Type=M, Torque=34Nm일 때 공구 마모 고장 가능성을 예측해줘.
```

---

## 9. Tests

```bash
cd ai_server
.venv/bin/python -m pytest
```

RAG evaluation notebook:

```bash
cd ai_server
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name manufacturing-ai-server --display-name "manufacturing-ai-server"
```

Then open `ai_server/notebooks/01_rag_eval_lab.ipynb`, select the `manufacturing-ai-server` kernel, and run it with working directory set to `ai_server`. The notebook reads `ai_server/.env` and writes outputs to `ai_server/eval/results/`.

Current verification snapshot:

```text
124 passed
RAG eval notebook executes end-to-end without cell errors
```

Portfolio-level RAG evaluation details are summarized in `docs/PORTFOLIO_ROADMAP.md`.

---

## 10. What Is Intentionally Not Included

- 설비 직접 제어
- safety/legal guarantee
- Streamlit upload/vectorize UI
- automatic Chroma repair
- corpus versioning
- runtime auto-training
- AI4I CSV rows in vector DB
- separate report-generation route

---

## 11. Documentation Map

Portfolio:

- `docs/PORTFOLIO_REVIEW_GUIDE.md`
- `docs/PORTFOLIO_ROADMAP.md`
- `docs/DEMO_SCRIPT.md`

Architecture:

- `docs/LANGGRAPH_FINAL_ARCHITECTURE.md`
- `docs/architecture.md`
- `docs/rag_evidence_orchestration.md`
- `docs/CURRENT_BACKEND_ARCHITECTURE_AUDIT.md`

Operations:

- `docs/RAG_INDEX_RUNBOOK.md`
- `docs/API_EXAMPLES.md`
- `docs/RUN_AND_NEXT_WORK.md`

Historical troubleshooting:

- `docs/archive/TROUBLESHOOTING_AND_ARCHITECTURE_EVOLUTION_2026-06-13.md`
