# Demo Script

포트폴리오 시연용 스크립트입니다. 현재 제품 endpoint는 `POST /agent/send`이고, `/rag/search`는 Chroma 검색 확인용 debug seam입니다.

## 1. 서버 실행

FastAPI:

```bash
cd ai_server
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Streamlit:

```bash
ai_server/.venv/bin/python -m streamlit run streamlit_app.py \
  --server.address 127.0.0.1 \
  --server.port 8501
```

브라우저:

```text
FastAPI docs: http://127.0.0.1:8000/docs
Streamlit:    http://127.0.0.1:8501
```

## 2. 상태 확인

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/llm/models
```

설명 포인트:

- LLM provider/model 상태
- 비활성화된 고비용 모델 정책
- Chroma/RAG 준비 상태
- domain catalog loading 상태

## 3. Demo 1: AI4I Prediction + RAG

질문:

```text
AI4I 데이터가 Type=M, Air temperature=300.2K, Process temperature=309.0K,
Rotational speed=1480rpm, Torque=34Nm, Tool wear=235min일 때
공구 마모 고장 가능성을 예측하고, 공구 교체 전 확인해야 할 항목을 알려줘.
```

기대 설명:

- 6개 AI4I feature가 모두 파싱되어 `prediction_called=true`
- `prediction_plus_rag` profile 적용
- AI4I prediction은 고장 가능성/고장모드 점수에만 사용
- RAG는 OSHA/Haas/KOSHA 문서 근거로 점검/안전 절차 설명
- 사용자 답변에는 token/cost/run id/debug metadata가 노출되지 않음

## 4. Demo 2: RAG-only Safety

질문:

```text
드릴기 작업 전에 공작물 고정 상태, 방호덮개, 비상정지장치를 어떻게 확인해야 하는지 알려줘.
```

기대 설명:

- AI4I feature가 없으므로 `prediction_called=false`
- 예측 확률, TWF/OSF/HDF/PWF 확률이 출력되지 않음
- `rag_only_safety` profile 적용
- 장비명과 title/document_title metadata를 이용해 드릴기 관련 안전 문서 우선
- safety gate id는 metadata/debug에만 남고 사용자 답변에는 노출되지 않음

## 5. Demo 3: AI4I Clarification

질문:

```text
AI4I Type=M, Torque=34Nm일 때 공구 마모 고장 가능성을 예측해줘.
```

기대 설명:

- 예측 의도는 있지만 필수 feature가 부족함
- `prediction_called=false`
- `prediction_skip_reason=missing_ai4i_features`
- RAG-only 답변으로 우회하지 않고 누락 feature만 요청

## 6. Demo 4: Haas Troubleshooting RAG

질문:

```text
Haas 밀 장비에서 스핀들 이상음, 진동, 경보가 발생했을 때 우선 확인해야 할 항목을 알려줘.
```

기대 설명:

- `troubleshooting_rag` profile 적용
- Haas troubleshooting 문서와 관련 안전 문서를 citation으로 사용
- 물리 점검/커버 개방/회전부 접근이 필요한 경우 safety 절차가 조건부로 표시

## 7. 진행 Trace 설명

Streamlit 진행 trace에서는 현재 구조를 아래처럼 설명합니다.

```text
ContextSubAgent
planning_router
prediction_node
prediction_quality_gate
rag_evidence_subagent
evidence_quality_gate
safety_contract_subagent
safety_contract_gate
answer_compose
answer_text_review
response_packager
memory_writer
audit_persistence
```

`generate_report` 체크박스는 제거되었습니다. 내부 history/debug에는 run metadata가 저장되지만 사용자-facing answer에는 긴 보고서나 debug 정보가 붙지 않습니다.

## 8. RAG Debug 확인

```bash
curl -X POST http://127.0.0.1:8000/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query":"Haas spindle load tool wear troubleshooting torque", "top_k":5}'
```

설명 포인트:

- `/rag/search`는 Root graph 답변 경로가 아니라 검색 상태 확인용 API
- Chroma collection: `manufacturing_rag`
- 기대 corpus count: 727

## 9. 마무리 설명

강조할 점:

- Agent는 설비 직접 제어를 하지 않음
- AI4I prediction과 RAG Evidence를 분리
- 안전 판단은 LLM만이 아니라 YAML/metadata 기반 gate와 validator로 보강
- 실패를 silent fallback으로 숨기지 않고 trace/warning/error로 드러냄
- 운영 관점에서 token/cost/history/debug를 내부 기록으로 보존
