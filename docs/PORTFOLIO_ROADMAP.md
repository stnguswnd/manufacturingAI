# Portfolio Roadmap

작성일: 2026-06-13

이 문서는 포트폴리오 검토자가 프로젝트의 현재 완성도, 설계 의도, 남은 개선 방향을 빠르게 이해하도록 정리한 문서입니다. 과거 구현 이력은 `docs/archive/`를 참고하고, 현재 운영 구조는 이 문서를 기준으로 봅니다.

## 1. 프로젝트 한 줄 소개

AI4I 제조 공정 예측, OSHA/Haas/KOSHA 문서 기반 RAG, YAML 안전 게이트, LangGraph SubAgent orchestration, token/cost 관측을 결합한 제조 특화 AI Agent 서버입니다.

## 2. 현재 제품 구조

```text
POST /agent/send
  -> RootManufacturingGraph(StateGraph)
      -> ContextSubAgent
      -> PlanningSubAgent
      -> manufacturing_analysis
      -> RagEvidenceSubAgent
      -> SafetySubAgent
      -> response_synthesis
      -> response_packager
      -> MemorySubAgent
      -> audit_persistence
```

`/rag/search`는 Root graph 답변 경로가 아니라 Chroma 검색을 확인하기 위한 API/debug seam입니다.

## 3. 포트폴리오에서 강조할 구현 포인트

| 영역 | 현재 구현 | 강조 포인트 |
| --- | --- | --- |
| FastAPI 제품 API | `/agent/send`, `/agent/send/stream`, `/rag/search` | 실제 서버 API와 Streamlit 데모 UI가 분리됨 |
| LangGraph orchestration | Root graph + 5개 SubAgent | 큰 graph를 책임별 StateGraph로 분리 |
| AI4I prediction | 6개 필수 feature가 완전할 때만 예측 | 불완전하면 clarification으로 종료하고 RAG를 실행하지 않음 |
| RAG Evidence | Chroma `manufacturing_rag`, 727 vectors | AI4I CSV가 아니라 OSHA/Haas/KOSHA 문서만 corpus로 사용 |
| Adaptive RAG | prediction_plus_rag, rag_only_safety, troubleshooting_rag, concept_explanation | 질문 유형별 retrieval profile과 evidence selection 분리 |
| Safety gate | `safety_gate_matrix.yaml` + `SafetySubAgent` | LOTO, 회전부 방호, 정비 자격 등 deterministic policy 적용 |
| 안전 검증 | Safety validator + replan/차단 | 필수 안전 내용 누락 또는 금지 표현을 최종 응답 전에 차단 |
| Context/Memory | ContextSubAgent, MemorySubAgent, checkpoint/history | user/session별 follow-up context 유지 |
| Observability | llm usage, trace, run history | token, cost, route, citation, warnings를 내부 기록으로 보존 |
| Streamlit UI | Agent 실행, 진행 trace, RAG 확인 | 데모와 디버깅에 필요한 화면 제공 |

## 4. 중요한 설계 경계

### AI4I와 RAG 분리

AI4I 데이터는 예측 입력입니다. Vector DB에 넣지 않습니다.

```text
AI4I feature
  -> PredictionService

OSHA / Haas / KOSHA documents
  -> rag_chunks.jsonl
  -> Chroma
  -> RagEvidenceSubAgent
```

### 사용자 답변과 debug 정보 분리

사용자-facing answer에는 `run_id`, model, token, cost, calls, raw score, chunk id, safety gate id를 출력하지 않습니다. 이 정보는 response debug, trace, history, Streamlit 상세 패널에서만 확인합니다.

### 보고서 옵션 제거

`generate_report`는 사용자/API/UI 입력에서 제거했습니다. 내부 실행 기록은 계속 저장하지만, 사용자는 항상 일반 답변을 받습니다. “보고서 형식으로 정리해줘”는 별도 report mode가 아니라 Markdown 답변 스타일로 처리합니다.

## 5. 현재 검증 상태

```text
Full test snapshot: 93 passed
RAG corpus: rag_chunks.jsonl 727
Chroma collection: manufacturing_rag 727 vectors
RAG evaluation notebook: executed end-to-end, 15 code cells, 0 cell errors
```

Chroma vector DB는 git ignored입니다. 새 환경에서는 `docs/RAG_INDEX_RUNBOOK.md` 절차로 `rag_chunks.jsonl`에서 재색인합니다.

### 5.1 RAG 평가 방법

RAG/Agent 답변 품질은 `ai_server/notebooks/01_rag_eval_lab.ipynb`에서 golden dataset 기반으로 확인합니다.

평가 입력:

- `ai_server/eval/golden_rag_cases.jsonl`
- 현재 5개 case: 제조 개념 질문, LOTO/정비 안전 질문, CNC 칩 배출 개념 질문, 애매한 운전 지속 질문, 현장 로그 부족 질문
- 각 case는 `expected_path`, `expected_intent`, `retrieval_policy`, `citation_policy`, `expected_source_ids`, `expected_doc_ids`, `expected_keywords`, `forbidden_keywords`를 가집니다.

평가 단계:

1. 실제 `RootManufacturingGraph`를 실행해 case별 raw output을 생성합니다.
2. deterministic checks로 retrieval/citation/route/intent/keyword/error를 평가합니다.
3. Ragas로 retrieved context 기반 metric을 계산합니다.
   - 사용 metric: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`
   - `retrieved_contexts`가 없는 case는 Ragas에서 제외합니다.
4. custom LLM judge로 답변 직접성, 근거성, unsafe instruction, citation issue, 장황함을 평가합니다.
5. 결과를 `merged_eval_results_v1.csv`와 `failure_cases_v1.csv`로 합칩니다.

재현 명령:

```bash
MPLBACKEND=Agg ai_server/.venv/bin/jupyter nbconvert \
  --to notebook \
  --execute ai_server/notebooks/01_rag_eval_lab.ipynb \
  --output 01_rag_eval_lab.executed.ipynb \
  --output-dir /tmp
```

Jupyter kernel은 로컬 포트를 열기 때문에 제한된 sandbox에서는 `PermissionError: Operation not permitted`가 날 수 있습니다. 일반 터미널/Jupyter 환경에서는 같은 notebook이 끝까지 실행됩니다.

### 5.2 최근 RAG 평가 결과 스냅샷

마지막 확인 시각 기준 결과 파일:

- `ai_server/eval/results/rag_raw_outputs_v1.jsonl`
- `ai_server/eval/results/deterministic_check_results_v1.csv`
- `ai_server/eval/results/ragas_scores_v1.csv`
- `ai_server/eval/results/custom_llm_judge_scores_v1.csv`
- `ai_server/eval/results/merged_eval_results_v1.csv`
- `ai_server/eval/results/failure_cases_v1.csv`

주요 결과:

| Case | 목적 | 주요 결과 |
| --- | --- | --- |
| `rag-001` | 토크 상승과 공구 마모 개념 질문 | custom judge `1`, `irrelevant_answer`. 일반 답변 경로가 질문 핵심을 직접 답하지 못함. |
| `rag-002` | 공구 교체 전 LOTO/정비 안전 절차 | retrieval/citation/expected source recall 통과. custom judge `5`, pass. Ragas `context_precision=1.0`, `context_recall=0.5`. |
| `rag-003` | CNC 칩 배출 불량 영향 | custom judge `1`, `irrelevant_answer`. 일반 답변 경로가 질문 핵심을 직접 답하지 못함. |
| `rag-004` | 애매한 “계속 돌려도 돼?” 질문 | clarification route와 retrieval forbidden 정책은 통과. 기대 keyword 일부 누락. |
| `rag-005` | 현장 로그 부족 상태에서 원인 확정 요청 | 확정 거부 방향은 맞지만 Ragas `context_precision=0.0`, `context_recall=0.0`. 이 유형은 Ragas보다 deterministic/custom judge 중심으로 해석해야 함. |

현재 평가로 확인된 점:

- 안전 절차형 RAG case(`rag-002`)는 expected KOSHA source와 citation integrity를 잘 잡습니다.
- 일반 개념 질문(`rag-001`, `rag-003`)은 fast/general answer path가 질문별 핵심 개념을 충분히 반영하지 못하는 회귀가 드러났습니다.
- 부족한 현장 데이터 또는 애매한 질문(`rag-004`, `rag-005`)은 Ragas 점수보다 deterministic policy와 custom judge가 더 적합합니다.
- Ragas는 retrieved context가 있는 case만 평가하므로, retrieval-free clarification/개념 답변은 별도 deterministic/judge 기준으로 봐야 합니다.

### 5.3 남은 평가 보완

- golden dataset을 5개에서 30개 이상으로 확장합니다.
- `expected_source_ids`와 `expected_doc_ids`를 안전/정비/RAG 필수 case마다 더 채웁니다.
- `rag-001`, `rag-003`의 general answer formatter를 보완해 질문 핵심어를 직접 답하도록 고정합니다.
- `CUSTOM_JUDGE_LIMIT=all`로 전체 case judge를 주기적으로 실행합니다.
- Ragas `faithfulness`가 NaN으로 남는 case는 Ragas LLM/metric 실행 로그를 별도로 남겨 원인을 추적합니다.

## 6. 데모 시나리오

1. AI4I + RAG
   - 6개 feature를 모두 포함한 질문
   - prediction_called=true
   - prediction_plus_rag profile
   - 공구 마모/TWF 해석 + 안전 확인 + citation

2. RAG-only safety
   - 드릴기/공구 교체/방호덮개/비상정지 질문
   - prediction_called=false
   - AI4I 확률 문구 없음
   - rag_only_safety profile
   - 장비명 title/metadata supplement로 관련 KOSHA 문서 우선

3. AI4I clarification
   - Type, Torque만 주고 예측 요청
   - missing_ai4i_features
   - RAG 실행 없이 누락 feature만 재요청

4. Haas troubleshooting
   - 스핀들 이상음, 진동, 경보 질문
   - troubleshooting_rag profile
   - Haas troubleshooting + 필요한 안전 문서 citation

## 7. 남은 개선 우선순위

| Priority | 항목 | 이유 | 완료 기준 |
| --- | --- | --- | --- |
| P0 | Golden evaluation set | RAG/안전/예측 품질 회귀 방지 | 대표 질의 30개, 기대 route/citation/safety 기준 |
| P0 | 답변 품질 평가 자동화 | LLM 답변은 테스트만으로 품질 보장이 어려움 | unsupported claim, citation coverage, safety omission 측정 |
| P1 | 운영 모니터링 | token/cost/latency/error를 장기 추적해야 함 | OTLP exporter 또는 dashboard 연결 |
| P1 | History schema versioning | 응답 구조 변경 시 과거 이력 해석 필요 | `schema_version`, migration policy |
| P1 | Corpus metadata 강화 | 설비/업종/문서 적용 범위를 더 정교하게 필터링 | source version, equipment scope, effective date |
| P2 | Admin/debug UI | 운영자가 corpus 상태와 run trace를 쉽게 확인 | 별도 admin panel 또는 read-only dashboard |
| P2 | Corpus versioning | 현재는 runbook 기반 재색인 | corpus manifest, vector count audit, release note |
| P2 | 배포 패키징 | 포트폴리오 재현성 강화 | Docker compose, healthcheck, env template |

## 8. 포트폴리오 참고 문서

검토자는 아래 순서로 보면 됩니다.

1. `README.md`
2. `docs/PORTFOLIO_REVIEW_GUIDE.md`
3. `docs/DEMO_SCRIPT.md`
4. `docs/LANGGRAPH_FINAL_ARCHITECTURE.md`
5. `docs/rag_evidence_orchestration.md`
6. `docs/archive/TROUBLESHOOTING_AND_ARCHITECTURE_EVOLUTION_2026-06-13.md`

`docs/archive/` 문서는 현재 운영 매뉴얼이 아니라 문제 해결과 구조 진화 기록입니다.
