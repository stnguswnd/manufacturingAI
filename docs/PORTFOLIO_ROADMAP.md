# Portfolio Roadmap

작성일: 2026-06-12

## 1. 프로젝트 한 줄 소개

제조 공정 데이터, 도메인 규칙, RAG 문서 검색, 안전 게이트, Supervisor re-plan, OpenAI LLM 응답 생성, 토큰/비용 관측을 결합한 제조 특화 AI Agent 서버와 Streamlit 테스트 UI.

## 2. 현재 구현된 핵심 기능

| 영역 | 구현 내용 | 포트폴리오에서 강조할 점 |
|---|---|---|
| Agent Orchestration | Manufacturing Supervisor가 요청 의도와 실행 route를 계획 | 단순 챗봇이 아니라 제조 업무 단계 기반 Agent 구조 |
| Re-plan Loop | RAG 근거 부족, LLM 구조화 출력 실패, 안전 검증 실패 시 재계획 | Agent가 실패를 감지하고 제한된 횟수 안에서 회복 시도 |
| Prediction Tool | AI4I 기반 불량/고장모드 예측 | ML 예측 결과를 LLM 답변 context로 연결 |
| Domain Modeling | 설비 taxonomy, 고장모드, 안전 게이트, 조치 카탈로그 YAML화 | 제조 지식을 코드에 하드코딩하지 않고 설정화 |
| RAG Retrieval | 샘플 매뉴얼/안전 문서 검색 | 답변 근거와 citation 구조 확보 |
| Safety Gate | LOTO, 회전부 방호, 전기 격리 등 필수 확인사항 검증 | 제조/안전 도메인에서 중요한 guardrail 구현 |
| LLM Integration | OpenAI Responses API 기반 JSON structured output | 생성형 답변을 스키마로 통제 |
| Model Policy | 모델 선택 UI, 고비용 모델 비활성화 | 비용 폭주 방지와 사용자 선택권 균형 |
| Usage/Cost Tracking | OpenAI usage 기반 토큰/비용 계산 | 실제 운영 비용을 UI와 history에 노출 |
| Observability | OpenTelemetry span attribute 기록 | 운영 모니터링 확장 가능성 |
| History | SQLite 기반 실행 이력 저장 | 실행 결과, usage, trace 누적 |
| User Context | 유저별 profile, session summary, recent runs, long-term memory 구성 | 단순 history injection이 아닌 계층형 context engineering |
| UI | Streamlit Agent/Prediction/RAG/History 테스트 화면 | 데모 가능한 end-to-end 제품 형태 |

## 3. 설계에서 고려한 점

### 3.1 LLM-only 실행 정책

- `use_llm=false`, template provider, mock smoke path를 제거했다.
- Agent 실행은 OpenAI 또는 OpenAI-compatible LLM이 설정되어야 가능하다.
- API key가 없으면 조용히 템플릿으로 답하지 않고 `llm_unavailable`로 실패한다.
- 포트폴리오 관점에서는 “데모용 fake output”이 아니라 실제 LLM 운영 흐름을 보여줄 수 있다.

### 3.2 비용 통제

- 모델별 가격표를 내부 config로 관리한다.
- OpenAI 응답의 `usage`에서 input/output/cached tokens를 읽어 비용을 계산한다.
- Streamlit에 LLM calls, Re-plans, input/output/total tokens, estimated cost를 표시한다.
- 고비용 모델은 API와 UI 양쪽에서 선택 불가 상태로 둔다.

### 3.3 안전성과 책임 경계

- Agent는 설비를 직접 제어하지 않는다.
- 답변은 정비 담당자/안전관리자 검토를 전제로 한다.
- 안전 게이트 필수 문구가 누락되거나 금지 표현이 나오면 차단한다.
- LLM 실패를 무조건 fallback 답변으로 숨기지 않고 실패 상태를 드러낸다.

### 3.4 확장성

- 제조 도메인 지식은 YAML 카탈로그로 분리했다.
- 신규 설비/고장모드/안전 절차는 코드 수정 없이 domain YAML과 문서 ingest로 확장 가능하게 설계했다.
- `PredictionService`, `RagService`, `DomainKnowledgeService`, `SupervisorService`, `LLMService`는 역할별로 분리했다.

## 4. 현재 아쉬운 점과 보완 우선순위

| 우선순위 | 보완 항목 | 이유 | 완료 기준 |
|---|---|---|---|
| P0 | 실제 사내/현장 문서 ingest | 샘플 문서만으로는 제조 현장 신뢰도가 낮음 | 설비 매뉴얼, SOP, 점검표, 안전절차 PDF/HTML ingest |
| P0 | Golden test set 구축 | Agent 품질을 기능 추가 때마다 유지해야 함 | 대표 질의 30개 이상, 기대 route/필수 안전문구/citation 기준 정의 |
| P0 | API 통합 테스트 | 현재 단위 테스트가 적음 | `/agent/send`, `/llm/models`, `/history`, error case 테스트 |
| P1 | OpenTelemetry exporter 설정 | 현재 span attribute 기록만 있고 전송 설정은 없음 | OTLP exporter로 Jaeger/Tempo/Datadog 중 하나에 전송 |
| P1 | History schema 버전 관리 | 응답 구조가 바뀌면 과거 이력 해석이 어려움 | `schema_version`, migration, history filter 추가 |
| P1 | RAG 품질 평가 | 검색 결과가 실제로 답변에 충분한지 측정 필요 | hit rate, citation coverage, unsupported answer rate 기록 |
| P1 | 모델/비용 budget guard | 사용량이 누적될 때 예산 초과 방지 필요 | 일/월 budget, 요청당 max cost, 사용자 확인 UI |
| P1 | 비동기 job 처리 | 긴 LLM/RAG 실행이 API request lifecycle에 묶임 | run id 기반 job queue와 status endpoint |
| P2 | Dependency Injection 정리 | 전역 singleton 서비스는 테스트/운영 교체가 불편함 | FastAPI dependency provider와 service factory 도입 |
| P2 | 문서 metadata 강화 | 최신성/버전/적용 설비 판단이 필요함 | source, version, effective_date, equipment_type 필터 적용 |
| P2 | UI 개선 | 현재는 테스트 UI이며 포트폴리오 데모 완성도는 더 높일 수 있음 | 실행 trace timeline, 비용 chart, history 비교 화면 |
| P2 | Docker/배포 정리 | 재현 가능한 배포가 포트폴리오 설득력을 높임 | backend+ui docker compose, healthcheck, env template |

## 5. 기능 확장 누적 구조

새 기능을 추가할 때마다 아래 5개 문서를 같이 업데이트한다.

```text
docs/
  PORTFOLIO_ROADMAP.md          # 포트폴리오 관점의 현재 상태, 보완점, 확장 계획
  FEATURE_REGISTRY.md           # 기능별 상태/소유 파일/테스트/데모 방법
  ARCHITECTURE_DECISIONS.md     # 왜 그렇게 설계했는지 ADR 형식으로 기록
  QUALITY_CHECKLIST.md          # 기능 추가 전후 검증 체크리스트
  DEMO_SCRIPT.md                # 면접/포트폴리오 시연 순서와 예시 질의
```

## 6. Feature Registry 템플릿

기능을 추가할 때마다 `FEATURE_REGISTRY.md`에 아래 형식으로 누적한다.

```markdown
## F-000: 기능명

- Status: planned | in_progress | done | deprecated
- User value:
- Main files:
- API/UI entry:
- Data dependency:
- Cost impact:
- Safety impact:
- Observability:
- Tests:
- Demo steps:
- Follow-up:
```

예시:

```markdown
## F-001: LLM Token/Cost Meter

- Status: done
- User value: 요청별 토큰 사용량과 예상 비용을 Streamlit과 history에서 확인
- Main files: `llm_service.py`, `observability_service.py`, `graph.py`, `streamlit_app.py`
- API/UI entry: `/agent/send`, Streamlit Agent result metrics
- Data dependency: OpenAI response `usage`
- Cost impact: 모델별 가격표 기반 estimated cost 표시
- Safety impact: 없음
- Observability: `gen_ai.usage.*`, `gen_ai.request.model`
- Tests: compileall, pytest, `/llm/models` 확인
- Demo steps: Agent 실행 후 Estimated cost metric 확인
- Follow-up: 월별 budget guard 추가
```

## 7. Architecture Decision Record 템플릿

설계 판단이 필요한 변경은 `ARCHITECTURE_DECISIONS.md`에 기록한다.

```markdown
## ADR-000: 제목

- Date:
- Status: proposed | accepted | superseded
- Context:
- Decision:
- Alternatives:
- Consequences:
- Related features:
```

초기 ADR 후보:

| ADR | 내용 |
|---|---|
| ADR-001 | LLM-only runtime을 채택하고 template/mock 경로를 제거 |
| ADR-002 | OpenTelemetry는 관측용, 비용 계산은 OpenAI usage 기반으로 직접 수행 |
| ADR-003 | 제조 도메인 지식은 YAML catalog로 관리 |
| ADR-004 | 안전 검증 실패는 조용한 fallback이 아니라 re-plan 후 차단 |
| ADR-005 | 고비용 모델은 config 정책으로 UI/API에서 선택 제한 |

## 8. Quality Checklist 템플릿

기능 추가 PR 또는 작업 단위마다 확인한다.

```markdown
## 기능명

- [ ] API schema가 기존 응답과 충돌하지 않는다.
- [ ] Streamlit에서 실행 가능하다.
- [ ] LLM 호출 횟수와 비용 증가가 의도된 수준이다.
- [ ] Safety gate 검증을 우회하지 않는다.
- [ ] History에 필요한 metadata가 저장된다.
- [ ] OpenTelemetry attribute가 필요한 만큼 기록된다.
- [ ] pytest 또는 golden test가 추가됐다.
- [ ] README/PORTFOLIO_ROADMAP/FEATURE_REGISTRY가 갱신됐다.
```

## 9. Demo Script 초안

포트폴리오 시연은 아래 흐름으로 구성한다.

1. Health 확인
   - `/health`에서 LLM provider, model, domain catalog 로딩 상태 확인
2. 모델 정책 확인
   - `/llm/models` 또는 Streamlit 모델 선택창에서 `gpt-5.5` 비활성 확인
3. Agent 실행
   - 질문: `토크가 높고 공구 마모가 큰데 어떤 점검과 안전 절차를 확인해야 해?`
   - 공정 데이터 포함
   - 보고서 생성 ON
4. 진행 trace 설명
   - Supervisor
   - Prediction
   - RAG
   - Safety Gate
   - Explanation
   - Report
   - Evaluation/Audit
5. 비용/관측 설명
   - LLM calls
   - Re-plans
   - Input/output tokens
   - Estimated cost
6. History 확인
   - 실행 결과와 usage가 SQLite history에 저장되는지 확인
7. 안전성 설명
   - 설비 제어를 하지 않음
   - 담당자 검토 필요
   - LOTO/회전부/전기 격리 등 필수 safety gate

## 10. 다음 확장 추천 순서

### Phase 1: 포트폴리오 완성도

- `FEATURE_REGISTRY.md`, `ARCHITECTURE_DECISIONS.md`, `QUALITY_CHECKLIST.md`, `DEMO_SCRIPT.md` 생성
- Streamlit 화면에 실행 timeline과 비용 카드 개선
- README에 데모 스크린샷/시연 GIF 추가

### Phase 2: 품질 검증

- golden dataset 구축
- API integration test 추가
- RAG citation coverage 평가
- safety gate regression test 추가

### Phase 3: 운영성

- OTLP exporter 설정
- request id와 structured logging 추가
- budget guard 추가
- Docker compose 정리

### Phase 4: 실제 제조 도메인 확장

- 실제 설비 매뉴얼 ingest
- 사내 안전 절차 metadata 추가
- 설비별 taxonomy 확장
- 현장 데이터 기반 prediction model 재학습

## 11. 포트폴리오 문장 예시

```text
제조 공정 예측 모델과 문서 RAG, 안전 게이트, Supervisor re-plan 구조를 결합한 제조 특화 AI Agent를 구현했습니다. 단순 챗봇이 아니라 설비/고장모드/위험도/안전 절차/조치 계획을 분리해 reasoning context를 구성하고, OpenAI structured output으로 답변을 생성합니다. 또한 응답 usage 기반 토큰/비용 계산과 OpenTelemetry 관측 포인트를 추가해 실제 운영 비용과 실행 흐름을 UI에서 확인할 수 있게 했습니다.
```
