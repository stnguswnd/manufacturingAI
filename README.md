# Manufacturing Domain FastAPI Agent

공개 제조 데이터와 공개 문서를 결합한 **제조 특화 FastAPI Agent 서버**입니다.

이 버전은 단순 RAG 챗봇이 아니라, 제조업 특화 설계를 코드에 반영했습니다.

```text
AI4I 공정 데이터 예측
+ Haas / OSHA / KOSHA 공개 문서 RAG 검색
+ 설비 계층 모델
+ 고장모드 카탈로그
+ 위험도 산정
+ 안전 게이트
+ 조치 계획
+ 점검/정비 보고서 초안
+ 제조 특화 LLM 평가/Audit
```

> 이 시스템은 설비를 직접 제어하지 않습니다. 예측, 문서 검색, 설명, 안전 확인, 보고서 작성 보조만 수행합니다.

---

## 1. 왜 “제조 특화”인가?

일반 RAG Agent는 문서를 찾아 답변합니다. 제조 Agent는 그보다 더 많은 구조가 필요합니다.

| 제조 특화 요소 | 구현 위치 | 설명 |
|---|---|---|
| 설비 계층 | `ai_server/domain/equipment_taxonomy.yaml` | CNC → Spindle / Tool Changer / Coolant System 같은 구조 |
| 고장모드 | `ai_server/domain/failure_mode_catalog.yaml` | TWF/HDF/PWF/OSF/RNF와 원인·점검·안전 게이트 연결 |
| 안전 게이트 | `ai_server/domain/safety_gate_matrix.yaml` | LOTO, 회전부 방호, 전기 격리, 비상대응 확인 |
| 조치 카탈로그 | `ai_server/domain/action_catalog.yaml` | 고장모드별 점검 순서, LOTO 필요 여부, 자격자 필요 여부 |
| 보고서 구조 | `ai_server/domain/report_templates.yaml` | 점검/정비 보고서 섹션 정의 |
| 문서 검색 정책 | `ai_server/domain/document_policy.yaml` | 안전/정비/문서화 상황별 검색 범위 |
| 제조 Supervisor | `ai_server/app/services/supervisor_service.py` | 제조 업무 흐름에 따라 route 생성 |
| 도메인 서비스 | `ai_server/app/services/domain_service.py` | 설비, 공정조건, 고장모드, 위험도, 안전게이트, 조치계획 생성 |

---

## 2. 전체 실행 흐름

예시 질문:

```text
토크가 높고 공구 마모가 큰데 어떤 점검과 안전 절차를 확인해야 해?
```

실행 흐름:

```text
1. Manufacturing Supervisor
   - 복합 요청으로 분류
   - 예측 + 정비 + 안전 + 보고서 필요 여부 판단

2. Asset Context Agent
   - 설비 유형: CNC
   - 관련 하위 모듈: Spindle, Tool Changer, Coolant System 등 추정

3. Prediction Tool
   - AI4I 기반 불량/고장모드 예측

4. Process Condition Agent
   - torque_high, tool_wear_high 같은 제조 조건 태그 생성

5. Failure Mode Agent
   - OSF, TWF, HDF 등 고장모드 후보 생성

6. Risk & Priority Agent
   - 품질/설비/안전/생산/문서 근거 위험도 산정

7. Procedure Retrieval Agent
   - Haas/OSHA/KOSHA 문서 검색 query 구성

8. Safety Gate Agent
   - LOTO 확인
   - 회전부 방호장치 확인
   - 전기 격리/고온/비상대응 필요 여부 판단

9. Action Planner Agent
   - 점검 조치 순서 생성

10. Explanation Agent
   - 예측 결과 + 문서 근거 + 안전 게이트 + 조치 계획으로 답변 생성

11. Report Agent
   - 점검/정비 보고서 초안 생성

12. Evaluation / Audit Agent
   - 금지 표현, 안전 게이트 누락, route 적절성 점검
```

---

## 3. 폴더 구조

```text
manufacturing_ai_agent_domain_fastapi/
  ai_server/
    app/
      agent/graph.py              # 제조 특화 Agent 실행 그래프
      services/
        supervisor_service.py     # Manufacturing Supervisor
        domain_service.py         # 제조 도메인 규칙/카탈로그 서비스
        prediction_service.py     # AI4I 예측 Tool
        rag_service.py            # 경량 RAG 검색
        report_service.py         # 답변/보고서 생성
        llm_service.py            # 외부 LLM 어댑터
        evaluation_service.py     # 제조 특화 평가
      schemas.py
      main.py
    domain/
      equipment_taxonomy.yaml
      failure_mode_catalog.yaml
      safety_gate_matrix.yaml
      action_catalog.yaml
      report_templates.yaml
      document_policy.yaml
    scripts/
      train_ai4i_model.py
      ingest_docs.py
      bootstrap_sample_docs.py
    storage/
  data/
    ai4i/
    processed_docs/
  docs/
  .env.example
```

---

## 3.1 포트폴리오/확장 문서

| 문서 | 용도 |
|---|---|
| `docs/PORTFOLIO_ROADMAP.md` | 현재 구현, 설계 고려사항, 보완 과제, 확장 로드맵 |
| `docs/FEATURE_REGISTRY.md` | 기능 추가 시 상태/파일/API/테스트/데모 방법 누적 |
| `docs/ARCHITECTURE_DECISIONS.md` | 주요 설계 판단 기록 |
| `docs/QUALITY_CHECKLIST.md` | 기능 추가 전후 검증 체크리스트 |
| `docs/DEMO_SCRIPT.md` | 포트폴리오 시연 순서 |
| `docs/USER_CONTEXT_ENGINEERING_PLAN.md` | 유저 생성/삭제와 유저별 context engineering 기획 |

---

## 4. 외부 LLM 설정

이 프로젝트는 OpenAI LLM 사용을 전제로 실행됩니다. API Key가 없으면 Agent 실행은 `llm_unavailable` 오류로 실패합니다.

```bash
cp .env.example .env
```

공식 OpenAI 사용 예시:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-5.4-mini
OPENAI_API_KEY=your_openai_api_key_here
LLM_ENABLE_STRUCTURED_OUTPUT=true
AGENT_SUPERVISOR_LLM_REFINEMENT=true
```

OpenAI-compatible provider 사용 예시:

```env
LLM_PROVIDER=openai_compatible
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=dummy_or_provider_key
LLM_MODEL=your-model-name
```

---

## 4.1 LangGraph Checkpoint / Session Memory

Local/MVP 환경에서는 SQLite persistent checkpointer를 사용합니다.

```text
기본 경로: ai_server/storage/checkpoints/langgraph_checkpoints.sqlite3
thread_id: user_id:session_id
```

같은 `user_id + session_id`로 들어온 요청은 서버 재시작 후에도 short-term conversation state를 복원할 수 있습니다. 반대로 다른 `user_id`가 같은 `session_id`를 쓰거나, 같은 사용자가 다른 `session_id`를 쓰면 대화 초점과 이전 공정 데이터가 섞이지 않도록 `thread_id`를 분리합니다.

운영 환경에서 동시 요청이 많아지면 SQLite 대신 Postgres 또는 Redis 기반 checkpointer로 전환하고, lock과 connection pooling 정책을 별도로 설계해야 합니다.

---

## 5. 빠른 실행

```bash
cd manufacturing_ai_agent_domain_fastapi/ai_server
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/train_ai4i_model.py
python scripts/ingest_docs.py --sample-only

uvicorn app.main:app --reload --port 8000
```

API 문서:

```text
http://localhost:8000/docs
```

---

## 5.1 RAG 문서 수집/정제 파이프라인

AI4I CSV는 예측 모델, fixture, `process_data` 예시 용도입니다. AI4I CSV row를 Vector DB에 넣지 않습니다. Vector DB에는 AI4I 예측 결과를 정비 절차, 안전 확인, 보고서 문구로 바꿔줄 OSHA/Haas/KOSHA 문서만 넣습니다.

환경변수:

```bash
cp .env.example .env
```

`.env`에 KOSHA 공공데이터 API 키를 넣습니다.

```env
KOSHA_API_KEY=
KOSHA_CALL_API_ID=1050
```

설치:

```bash
pip install -r ai_server/requirements.txt
```

실행 순서:

```bash
python ai_server/scripts/download_static_rag_sources.py
python ai_server/scripts/download_kosha_sources.py --num-rows 20 --pages 1
python ai_server/scripts/build_rag_documents.py
python ai_server/scripts/build_rag_chunks.py
python ai_server/scripts/inspect_rag_corpus.py
```

secondary keywords까지 포함:

```bash
python ai_server/scripts/download_kosha_sources.py --include-secondary --num-rows 20 --pages 1
```

더 많이 수집:

```bash
python ai_server/scripts/download_kosha_sources.py --include-secondary --num-rows 50 --pages 3
```

생성 산출물:

```text
ai_server/data/processed/kosha_download_index.json
ai_server/data/processed/kosha_download_index.jsonl
ai_server/data/processed/rag_documents.jsonl
ai_server/data/processed/rag_chunks.jsonl
ai_server/data/processed/rag_corpus_report.md
```

주의사항:

```text
- AI4I CSV는 모델 예측/fixture/process_data 용도이며 Vector DB에 넣지 않는다.
- OSHA/Haas/KOSHA 문서는 RAG 근거용이다.
- KOSHA는 넓게 수집할 수 있지만, 검색 노이즈를 막기 위해 project_priority와 retrieval_scope metadata를 반드시 사용한다.
- KOSHA의 일부 HWP 파일은 자동 텍스트 추출이 실패할 수 있다.
- 텍스트 추출 실패 문서는 metadata-only로 남기고 기본 chunking 대상에서 제외한다.
- 실제 답변에서 KOSHA GUIDE는 국내 기술지침/참고자료 성격으로 사용하고, 법적 보증처럼 표현하지 않는다.
```

Optional Chroma indexing:

```bash
python ai_server/scripts/index_rag_chunks_chroma.py
```

Chroma는 선택 사항입니다. 기본 완료 기준은 `rag_documents.jsonl`, `rag_chunks.jsonl`, `rag_corpus_report.md` 생성입니다.

---

## 6. 주요 API

### 6.1 Health

```bash
curl http://localhost:8000/health
```

### 6.2 도메인 설정 확인

```bash
curl http://localhost:8000/domain/catalog
```

### 6.3 실행 계획 미리보기

```bash
curl -X POST http://localhost:8000/agent/plan \
  -H "Content-Type: application/json" \
  -d '{
    "message": "토크가 높고 공구 마모가 큰데 어떤 점검과 안전 절차를 확인해야 해?",
    "process_data": {
      "type": "L",
      "air_temperature_k": 302.1,
      "process_temperature_k": 311.3,
      "rotational_speed_rpm": 1380,
      "torque_nm": 58.2,
      "tool_wear_min": 210
    },
    "generate_report": true,
    "llm_model": "gpt-5.4-mini"
  }'
```

### 6.4 권장 Agent API: `/agent/send`

```bash
curl -X POST http://localhost:8000/agent/send \
  -H "Content-Type: application/json" \
  -d '{
    "message": "토크가 높고 공구 마모가 큰데 어떤 점검과 안전 절차를 확인해야 해?",
    "session_id": "demo-session-001",
    "process_data": {
      "type": "L",
      "air_temperature_k": 302.1,
      "process_temperature_k": 311.3,
      "rotational_speed_rpm": 1380,
      "torque_nm": 58.2,
      "tool_wear_min": 210
    },
    "inspection_notes": "작업자가 공구 마모 증가와 냉각 성능 저하를 의심함",
    "generate_report": true,
    "mode": "auto",
    "llm_model": "gpt-5.4-mini"
  }'
```

응답에는 다음이 포함됩니다.

```text
route
plan
prediction
manufacturing_context.asset_context
manufacturing_context.process_conditions
manufacturing_context.failure_modes
manufacturing_context.risk_assessment
manufacturing_context.safety_gates
manufacturing_context.action_plan
retrieved_documents
answer
report
warnings
trace
```

---

## 7. 제조 특화 응답에서 확인할 것

응답 JSON에서 아래가 나오면 제조 특화 설계가 작동한 것입니다.

```json
{
  "manufacturing_context": {
    "asset_context": {
      "equipment_type": "CNC",
      "inferred_subsystems": ["Spindle", "Tool Changer"]
    },
    "process_conditions": [
      {"tag": "torque_high"},
      {"tag": "tool_wear_high"}
    ],
    "failure_modes": [
      {"code": "OSF", "name_ko": "과부하 고장"}
    ],
    "safety_gates": [
      {"gate_id": "loto_if_physical_maintenance"},
      {"gate_id": "rotating_parts_guard_check"}
    ],
    "action_plan": [
      {"action_id": "inspect_tool_wear", "requires_loto": true}
    ]
  }
}
```

---

## 8. LLM/Agent 평가

평가는 단순 “답변이 자연스러운가”가 아니라 제조 업무 기준을 봅니다.

| 평가 항목 | 설명 |
|---|---|
| Route Correctness | 제조 업무에 맞는 Agent들이 실행되었는가 |
| Failure Mode Correctness | 고장모드 설명이 맞는가 |
| Safety Gate Compliance | 필요한 안전 게이트를 누락하지 않았는가 |
| Evidence Traceability | 예측 근거와 문서 근거가 분리되어 명확한가 |
| Action Feasibility | 현장에서 실제 점검 가능한 조치인가 |
| Scope Control | 설비 제어, 안전 보증, 법적 판단을 하지 않았는가 |
| Report Completeness | 보고서에 입력 데이터, 근거, 조치, 안전 확인이 포함되었는가 |

평가 API:

```bash
curl -X POST http://localhost:8000/evaluation/score \
  -H "Content-Type: application/json" \
  -d '{
    "agent_answer": "...",
    "route": ["Asset Context Agent", "Safety Gate Agent", "Action Planner Agent"],
    "expected_contract": {
      "expected_route": ["Asset Context Agent", "Safety Gate Agent", "Action Planner Agent"],
      "required_safety_gates": ["loto_if_physical_maintenance", "rotating_parts_guard_check"],
      "must_include": ["높은 토크", "공구 마모", "LOTO"],
      "forbidden": ["설비를 자동으로 정지했다고 말하기"]
    }
  }'
```

---

## 9. 현재 한계

```text
1. RAG는 MVP 안정성을 위해 경량 lexical 검색을 사용한다.
   실제 서비스에서는 Chroma/pgvector로 교체하는 것이 좋다.

2. 샘플 문서는 데모용 요약 문서다.
   실제 Haas/OSHA/KOSHA 문서를 수집하려면 collect/extract/ingest 스크립트를 사용해야 한다.

3. 제조 도메인 규칙은 범용 예시다.
   실제 공장에서는 설비별 매뉴얼, 점검표, 사내 안전 절차로 교체해야 한다.

4. AI는 설비 제어를 하지 않는다.
   정비 실행, 안전 보증, 법적 판단은 담당자가 최종 확인해야 한다.
```

### Future Work

이번 MVP 안정화 단계에서는 아래 작업을 의도적으로 구현하지 않고 확장 과제로 남깁니다.

```text
1. RAG 고도화
   - rag_query_planner
   - document_retriever
   - evidence_filter
   - evidence_grader
   - citation_builder
   - retrieval_replan

2. Safety 고도화
   - gate_builder
   - constraint_injector
   - action_validator
   - answer_validator
   - report_validator
   - escalation_decision

3. 운영 checkpointer 전환
   - SQLite에서 Postgres/Redis checkpointer로 전환
   - 동시 요청, lock, connection pooling 고려
```

---

## 10. 이번 버전의 핵심 가치

기존 MVP:

```text
Prediction Tool + RAG + Safety Agent + Report Agent
```

이번 버전:

```text
Manufacturing Supervisor
 ├─ Asset Context Agent
 ├─ Process Condition Agent
 ├─ Failure Mode Agent
 ├─ Risk & Priority Agent
 ├─ Procedure Retrieval Agent
 ├─ Safety Gate Agent
 ├─ Action Planner Agent
 ├─ Report Agent
 └─ Evaluation / Audit Agent
```

즉, 단순 제조 문서 챗봇이 아니라 **제조 현장 업무 절차를 반영한 Agent 시스템**입니다.
