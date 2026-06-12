# Manufacturing AI Agent Troubleshooting Record - 2026-06-13

이 문서는 제조 AI Agent를 고도화하면서 발견한 구조적 문제와 해결 과정을 정리한 트러블슈팅 기록이다. 단순 버그 수정 기록이 아니라, 기존 구조가 왜 불안정했는지, 어떤 구조로 바꿨는지, 그 결과 어떤 문제가 해결됐는지를 성과 중심으로 남긴다.

## 요약

오늘 작업의 핵심은 제조 AI Agent를 "키워드와 임시 상태에 의존하는 챗봇"에서 "Context Engineering, structured routing, formatter contract, safety policy, checkpoint v2를 가진 LangGraph 기반 Agent"로 정리한 것이다.

주요 성과:

- 단순 개념 질문이 heavy manufacturing format으로 오염되는 문제를 해결했다.
- "이것", "이걸", "왜?", "방금 권장조치" 같은 후속 질문을 구조적으로 처리하도록 바꿨다.
- `last_focus`, `last_answer_claims`, top-level `recommended_actions` 같은 레거시 state를 제거하고 `AnswerMemory`로 통합했다.
- intent classifier에 전체 대화/RAG/safety context가 섞이지 않도록 node별 `ContextPack`을 도입했다.
- formatter가 질문 문자열을 다시 해석하지 않고, 이미 결정된 `selected_path`와 `answer_type`만 렌더링하도록 분리했다.
- checkpoint 저장 state를 v2 primitive-only 정책으로 정리했다.
- RAG, safety, heavy planning, citation, recommendation 책임을 모듈 단위로 분리했다.
- compatibility facade/wrapper/private dependency를 제거하고 새 구조만 남겼다.

최종 검증:

```bash
cd ai_server
../.venv312/bin/python -m pytest tests/test_context_engineering.py -q
# 35 passed

../.venv312/bin/python -m pytest tests/test_intent_gateway.py -q
# 10 passed

../.venv312/bin/python -m pytest -q
# 62 passed
```

## 1. 단순 개념 질문이 heavy 분석 포맷으로 나가던 문제

| 항목 | 기존 구조 | 문제 | 변경 구조 | 해결된 성과 |
| --- | --- | --- | --- | --- |
| 개념 질문 라우팅 | `토크란?`, `마모가 뭐야?` 같은 질문도 supervisor/heavy path로 떨어질 수 있었음 | 답변에 `판정`, `위험도`, `안전 확인`, `권장 조치`가 붙어 사용자가 질문한 범위를 벗어남 | `fast_concept_answer`, `general_lightweight_answer`, `heavy_analysis_answer`를 분리 | 개념 질문은 가볍게 답하고, 현재 설비 판단은 데이터가 있을 때만 수행 |
| formatter 역할 | formatter가 질문 문자열을 보고 다시 의도를 추론 | `selected_path`가 맞아도 최종 답변 포맷이 섞일 수 있음 | `FormatterRegistry` 도입. formatter는 이미 결정된 `answer_type`만 렌더링 | fast answer에 heavy report format이 섞이지 않음 |
| glossary 응답 | 일부 용어만 Python dict/휴리스틱으로 처리 | glossary hit가 없으면 heavy fallback으로 갈 수 있음 | glossary fast path + structured intent classifier + hard gate validator 조합 | No-LLM fast path와 LLM route가 역할을 나눔 |

대표 문제:

```text
User: 토크란?

Assistant:
판정
주요 근거
위험도
안전 확인
권장 조치
...
```

해결 후 기대 동작:

```text
토크는 물체를 회전시키는 힘의 효과를 뜻합니다.
제조 설비에서는 모터, 스핀들, 축, 공구처럼 회전하는 부품에 걸리는 부하를 이해할 때 사용합니다.
현재 설비가 위험한지는 실제 토크 값, 회전수, 공구 마모, 온도 같은 공정 데이터가 있어야 판단할 수 있습니다.
```

## 2. "이것/이걸/그거" 후속 질문을 못 잡던 문제

| 항목 | 기존 구조 | 문제 | 변경 구조 | 해결된 성과 |
| --- | --- | --- | --- | --- |
| 후속 질문 해석 | 직전 run/history나 문자열 치환에 의존 | "이것"이 직전 핵심 주제인지, 문서인지, process_data인지 불안정 | `ContextResolver`가 `ContextResolution`을 생성 | 후속 질문 여부와 대상이 구조화됨 |
| 대화 초점 관리 | `last_focus`, `recent_entities` 등 임시 필드 증가 | 케이스별 state가 늘어나 유지보수 어려움 | `AnswerMemory.focus`, `claims`, `recommended_actions`로 통합 | 다음 턴에서 참조 가능한 기억이 한 곳에 모임 |
| session 격리 | session_id만 쓰거나 history 검색에 의존 | 다른 user/session context 오염 가능 | `thread_id = user_id:session_id` 정책 | user/session별 short-term memory 분리 |

대표 문제:

```text
User: 토크란?
User: 그렇다면 이걸 볼 때 주의해서 봐야하는 것은?
```

기존에는 `resolved=false`, `resolved_target=null`이 되거나 clarification으로 빠질 수 있었다.

변경 후 흐름:

```text
ContextResolver
  -> is_followup=True
  -> followup_type=previous_concept
  -> followup_target=토크
  -> standalone_query=토크를 볼 때 주의해서 봐야 하는 것은?
```

성과:

- "이걸"을 LLM에게 추측시키지 않고 state 기반으로 먼저 해소한다.
- 후속 질문이 아니면 과거 memory가 신규 질문을 오염시키지 않는다.

## 3. "왜?", "방금", "순서대로" 질문이 fallback으로 깨지던 문제

| 항목 | 기존 구조 | 문제 | 변경 구조 | 해결된 성과 |
| --- | --- | --- | --- | --- |
| 이유 질문 | `왜/이유` 키워드 휴리스틱 | 표현이 조금만 바뀌어도 깨짐 | `previous_answer_reason` followup type | 직전 답변 claim에 대한 rationale 질문으로 처리 |
| 권장조치 recap | public context나 general lightweight로 우회 | "방금 권장조치 순서대로"가 일반 설명으로 처리될 수 있음 | `recommended_action_recap` answer type + dedicated formatter | 권장조치 목록을 우선순위/번호로 안정적으로 출력 |
| 특정 항목 질문 | `그중 2번`을 문자열로 처리 | 해당 action의 rationale/safety_note를 못 씀 | `previous_recommended_action_item`, `followup_item_index` | 특정 action의 이유를 구조화 정보로 답변 |

대표 문제:

```text
User: 토크가 높고 공구 마모가 큰데 어떤 점검과 안전 절차를 확인해야 해?
Assistant: 권장 조치 1, 2, 3...

User: 너가 방금 준 권장조치 내용 중에서 가장 중요한 순서대로 나열해봐
```

기존에는 intent classifier schema 오류나 fallback clarification이 사용자에게 노출될 수 있었다.

변경 후 흐름:

```text
ContextResolver
  -> followup_type=previous_recommended_actions

ContextPackBuilder
  -> formatter_context.answer_type=recommended_action_recap
  -> recommended_actions 포함

FormatterRegistry
  -> RecommendedActionFormatter
```

성과:

- 권장조치 follow-up은 더 이상 `general_lightweight_answer`로 우회하지 않는다.
- `RecommendedAction` 구조의 `title`, `rationale`, `safety_note`, `priority`를 활용한다.

## 4. "움직이는 도표들" 오해석 문제

| 항목 | 기존 구조 | 문제 | 변경 구조 | 해결된 성과 |
| --- | --- | --- | --- | --- |
| 직전 답변 claim 참조 | 직전 답변의 핵심 claim이 저장되지 않음 | "도표"를 일반 chart/animation으로 해석 | `AnswerMemory.claims`, `key_points`, `short_summary` 저장 | 직전 답변의 "함께 움직이는 지표" claim을 참조 가능 |
| chart/general 구분 | `도표`, `보여줘` 키워드로 chart 판단 | "예시 보여줘"도 chart로 오분류 가능 | structured output intent classifier + policy validator | chart guidance와 rationale follow-up 분리 |

대표 문제:

```text
User: 마모가 뭐야?
Assistant: 공구 마모는 여러 지표와 함께 봐야 합니다.

User: 왜? 움직이는 도표들을 그렇게 많이 봐? 이유 알려줘
```

기존에는 "움직이는 도표"를 일반 그래프 질문으로 보고 heavy/general answer가 섞일 수 있었다.

변경 후:

```text
selected_path=general_lightweight_answer
answer_type=rationale
resolved_reference=previous_answer_claim
```

성과:

- 직전 답변의 claim을 근거로 "여러 지표를 같이 보는 이유"를 설명한다.
- chart 요청과 rationale 요청을 분리한다.

## 5. 내부 debug/error 정보가 사용자 답변에 노출되던 문제

| 항목 | 기존 구조 | 문제 | 변경 구조 | 해결된 성과 |
| --- | --- | --- | --- | --- |
| fallback reason | raw exception을 답변에 포함할 수 있었음 | `BadRequestError`, `invalid_json_schema`, `resolved=false` 등이 사용자에게 노출 | `FallbackReason` 도입: `public_reason`과 `internal_reason` 분리 | 사용자 답변에는 안전한 public reason만 출력 |
| public answer context | 전체 state를 formatter/LLM에 넘기는 경향 | debug state가 답변에 섞임 | `ContextPackBuilder`와 formatter context 분리 | final answer와 trace/debug state 분리 |
| sanitize 방식 | 출력 후 문자열 제거 | 근본 해결이 아님 | 애초에 public context만 전달 | sanitize는 최후의 방어선으로 축소 |

대표 문제:

```text
Classifier confidence below threshold: safe fallback clarification:
ValueError: BadRequestError: invalid_json_schema ...
```

성과:

- raw exception은 내부 디버깅에만 남고, 사용자에게는 "판단에 필요한 맥락이 부족합니다" 같은 public reason만 표시된다.

## 6. OpenAI structured output schema 오류

| 항목 | 기존 구조 | 문제 | 변경 구조 | 해결된 성과 |
| --- | --- | --- | --- | --- |
| structured output schema | 서비스별 schema 변환 | OpenAI strict schema에서 `additionalProperties: false` 누락 오류 | 공용 strict schema helper | nested object, anyOf, $defs까지 재사용 가능 |
| intent classifier 실패 | schema 오류가 route 실패로 연결 | 후속 질문이 clarification으로 깨짐 | schema validation + safe fallback + policy validator | classifier 장애가 사용자-facing raw error로 번지지 않음 |

대표 오류:

```text
Invalid schema for response_format 'intent_classifier_output':
'additionalProperties' is required to be supplied and to be false.
```

성과:

- OpenAI strict structured output 요구사항을 공용 helper로 처리한다.
- IntentClassifierService 안에 박혀 있던 schema 변환 책임을 분리했다.

## 7. Context Engineering v2 clean-slate 전환

| 항목 | 기존 구조 | 문제 | 변경 구조 | 해결된 성과 |
| --- | --- | --- | --- | --- |
| state schema | `last_focus`, `last_answer_claims`, `last_answer_key_phrases`, top-level `recommended_actions` 혼재 | 필드가 계속 늘고 의미가 겹침 | `state_schema_version=2`, `AnswerMemory`, `ContextResolution`, `ContextPacks` | 단일 진실 공급원 확립 |
| checkpoint 호환 | v1 state와 호환하려는 adapter 필요 | 레거시 의존이 계속 살아남음 | 기존 checkpoint 보존 포기, v2만 사용 | 구조 단순화, migration 부담 제거 |
| checkpoint content | Pydantic model/object가 들어갈 가능성 | 저장/복원 불안정 | primitive-only sanitize | checkpoint 안정성 향상 |

최종 v2 memory 허용:

- `last_answer_memory`
- `recent_turns`
- `rolling_summary`

제거한 legacy top-level memory:

- `last_focus`
- `last_answer_claims`
- `last_answer_key_phrases`
- top-level `recommended_actions`

성과:

- "이전 대화 전체를 프롬프트에 넣기"가 아니라 필요한 context만 pack으로 전달한다.
- 신규 질문은 과거 heavy path에 오염되지 않는다.
- 후속 질문은 `last_answer_memory`를 사용해 명시적으로 해소한다.

## 8. ContextPackBuilder로 node별 context contract 분리

| 항목 | 기존 구조 | 문제 | 변경 구조 | 해결된 성과 |
| --- | --- | --- | --- | --- |
| classifier input | 전체 user_context, recent_runs, RAG 등이 섞일 수 있음 | intent classifier가 과거 context/RAG에 오염 | `classifier_context` 최소화 | 라우팅 안정성 향상 |
| answer input | path별 필요한 context 구분 약함 | fast answer에 safety/heavy context가 섞임 | `answer_context`, `formatter_context`, `safety_context` 분리 | selected_path별 포맷 누수 방지 |
| memory writer input | 답변 텍스트에서 임의 추출 | focus/action/claim 품질 불안정 | `memory_writer_context` 분리 | 구조화된 AnswerMemory 생성 |

classifier에 넣지 않는 것:

- retrieved docs 원문
- full conversation history
- safety manual 원문
- raw exception
- root graph raw state

성과:

- 라우터는 라우팅에 필요한 작은 context만 본다.
- RAG/안전/응답 생성은 각자 필요한 contract만 받는다.

## 9. Hard gate registry화

| 항목 | 기존 구조 | 문제 | 변경 구조 | 해결된 성과 |
| --- | --- | --- | --- | --- |
| keyword list | `IntentGatewayService` 안에 계속 증가 | 유지보수 어려움, selected_path 직접 결정 위험 | `GateRegistry`, `GateResult` | final gate와 candidate signal 분리 |
| safety/scope guard | 일반 intent 분류와 섞임 | 위험 요청이 lightweight로 갈 수 있음 | control/scope/process hard gate | 안전/스코프 관련 오분류 방지 |
| follow-up signal | 키워드가 route를 직접 결정 | "방금", "왜" 같은 표현이 만능 분기화 | follow-up은 ContextResolver에서 처리 | hard gate의 책임 축소 |

GateResult 필드:

- `matched`
- `gate_name`
- `selected_path`
- `answer_type`
- `reason`
- `confidence`
- `is_final`
- `category`

성과:

- 안전/기계제어/scope guard는 final gate 가능.
- follow-up signal은 selected_path를 직접 결정하지 않음.

## 10. Heavy manufacturing path 분리

| 항목 | 기존 구조 | 문제 | 변경 구조 | 해결된 성과 |
| --- | --- | --- | --- | --- |
| planning | `SupervisorService` keyword/private method 의존 | private method coupling, 테스트 어려움 | `DiagnosticPlanner`, `DiagnosticPlanToAgentPlanTranslator`, `PlanRefiner` | 계획 생성 contract 분리 |
| diagnostic state | `last_diagnostic_plan` instance state | 요청 간 상태 오염 가능 | `PlanningResult` 반환 | request-scoped state 제거 |
| RAG facade | `RagEvidencePlanner` compatibility facade | 새 구조와 구 구조가 공존 | facade 삭제, 5개 모듈 직접 사용 | RAG path 명확화 |
| graph helper wrapper | `_make_rag_query`, `_warnings`, `_llm_payload` 등 | wrapper가 실제 로직 위치를 숨김 | 전용 모듈 직접 호출 | root/graph orchestration 중심화 |

새 heavy path 구성:

- `DiagnosticPlanner`
- `DiagnosticFallbackPolicy`
- `DiagnosticPlanToAgentPlanTranslator`
- `PlanRefiner`
- `RagQueryPlanner`
- `Retriever`
- `EvidenceFilter`
- `EvidenceGrader`
- `CitationBuilder`
- `SafetyGateBuilder`
- `RecommendationBuilder`
- `StructuredAnswerPayloadBuilder`

성과:

- `DiagnosticPlanner`가 더 이상 `SupervisorService._intent`, `_layers`, `_llm_refine`에 의존하지 않는다.
- `RagEvidencePlanner`와 `make_query` compatibility method가 완전히 제거됐다.
- 요청별 planning snapshot은 instance field가 아니라 `PlanningResult`로 전달된다.

## 11. RAG 데이터 수집/정제 파이프라인

| 항목 | 기존 구조 | 문제 | 변경 구조 | 해결된 성과 |
| --- | --- | --- | --- | --- |
| RAG source | AI4I CSV와 RAG 문서 역할이 혼재될 수 있음 | 예측 데이터와 문서 근거가 섞임 | AI4I CSV는 prediction, OSHA/Haas/KOSHA는 RAG source로 분리 | Vector DB 역할 명확화 |
| source manifest | URL/metadata가 코드에 흩어질 수 있음 | 재현성 부족 | `rag_source_manifest.yaml` | 수집 대상과 metadata 관리 |
| KOSHA 수집 | API/download/text extraction 없음 | 한국어 안전 문서 활용 어려움 | KOSHA API downloader, file downloader, extractor | 한국어 safety/report 근거 확보 기반 |
| chunk/report | corpus quality 확인 어려움 | 어떤 문서가 들어갔는지 알기 어려움 | `rag_documents.jsonl`, `rag_chunks.jsonl`, `rag_corpus_report.md` | corpus audit 가능 |

구성:

- `download_static_rag_sources.py`
- `download_kosha_sources.py`
- `build_rag_documents.py`
- `build_rag_chunks.py`
- `inspect_rag_corpus.py`
- `index_rag_chunks_chroma.py`

성과:

- AI4I row를 Vector DB에 넣지 않는 원칙을 명확히 했다.
- KOSHA low priority/restricted 문서를 버리지 않고 metadata로 통제한다.
- Chroma indexing은 optional script로 분리했다.

## 12. User-scoped Context Engineering

| 항목 | 기존 구조 | 문제 | 변경 구조 | 해결된 성과 |
| --- | --- | --- | --- | --- |
| user identity | session 중심 | 다른 사용자 context 오염 가능 | `user_id` 중심 users/sessions/memories/history | 사용자별 격리 |
| memory | 실행 이력 중심 | 장기 선호/반복 이슈 반영 어려움 | `user_memories`, `ContextService`, `MemoryService` | user-scoped long-term context 기반 |
| 삭제 정책 | 명확하지 않음 | 개인정보/이력 잔존 가능 | hard/soft delete 설계 | 삭제 정책 명확화 |
| UI | user 선택/관리 없음 | 사용자별 테스트 어려움 | Streamlit user select/create/delete/context tab | 사용자별 context 확인 가능 |

성과:

- 다른 user의 context가 follow-up 해석에 섞이지 않게 했다.
- user history/context를 API와 UI에서 확인 가능하게 했다.
- memory extraction은 초기 버전에서 rule-based로 제한해 비용과 위험을 통제했다.

## 13. Streamlit UI와 Usage/Cost 관측

| 항목 | 기존 구조 | 문제 | 변경 구조 | 해결된 성과 |
| --- | --- | --- | --- | --- |
| 실행 상태 | AI가 무엇을 하는지 보이지 않음 | 사용자 입장에서 대기 이유 불명확 | trace/progress UI | 어떤 노드가 실행 중인지 표시 |
| usage | 단건/누적 구분 없음 | 비용 추적 어려움 | 단일 응답 usage + Usage tab 누적값 | 요청별/누적 비용 확인 |
| 비용 | USD만 있거나 불명확 | 한국 사용자에게 비용 체감 어려움 | 환율 기반 KRW 추정 | 비용 가시성 향상 |
| 모델 선택 | 고정 모델 | 비용 통제 어려움 | 모델 선택 UI, 고가 모델 제한 | 운영 비용 통제 |
| chat history | history tab 중심 | 채팅 맥락 확인 불편 | chat tab/history 표시 | 대화형 테스트 개선 |

성과:

- 단일 AI 응답에는 그 요청의 usage만 표시.
- Usage tab에서는 누적 calls/tokens/cost를 확인.
- OpenAI usage field 기반 비용 계산과 OpenTelemetry span/metric 설계를 문서화했다.

## 14. Python 3.12와 실행 환경 정리

| 항목 | 기존 구조 | 문제 | 변경 구조 | 해결된 성과 |
| --- | --- | --- | --- | --- |
| Python version | 명확하지 않음 | 의존성/typing 혼선 | Python 3.12 기준 환경 정리 | 실행 환경 일관성 |
| ignored files | library/cache/db 파일이 git에 많이 잡힘 | 변경 추적 노이즈 | `.gitignore` 보강 | 실수 커밋 위험 감소 |
| smoke/mock | LLM false/mock 코드 잔존 | 실제 agent 검증 방해 | mock/smoke 제거 방향 | 실제 LLM 기반 검증 중심 |

성과:

- `.venv312` 기준 테스트 명령 정리.
- checkpoint/db/vector/raw/cache 파일 ignore 강화.

## 15. 대표 트러블슈팅 사건별 전후 비교

### 15.1 `토크란?`이 heavy format으로 답하던 문제

Before:

```text
토크란?
-> 판정 / 주요 근거 / 위험도 / 안전 확인 / 권장 조치
```

After:

```text
IntentGateway
  -> selected_path=fast_concept_answer
  -> answer_type=definition
FormatterRegistry
  -> FastConceptFormatter
```

성과:

- 단순 개념 질문의 token/cost 감소.
- 불필요한 safety/report formatting 제거.

### 15.2 `이것의 단점은?`이 대상을 못 찾던 문제

Before:

```text
resolved=false
prediction 값 없음
rag_contexts 없음
failure_modes 없음
```

After:

```text
ContextResolver
  -> previous_concept / previous_answer_reason
  -> standalone_query 생성
```

성과:

- follow-up을 raw history 검색이 아니라 structured memory로 처리.

### 15.3 `방금 권장조치 순서대로`가 clarification으로 빠지던 문제

Before:

```text
Classifier schema error
-> unsupported_or_clarification
-> raw exception 노출 가능
```

After:

```text
ContextResolution.followup_type=previous_recommended_actions
FormatterRegistry -> RecommendedActionFormatter
```

성과:

- 권장조치 recap이 독립 formatter로 처리됨.
- raw exception 노출 방지.

### 15.4 RAG facade와 wrapper가 남아 있던 문제

Before:

```text
RagEvidencePlanner
make_query(...)
ManufacturingAgentGraph._make_rag_query(...)
```

After:

```text
RagQueryPlanner.plan(...)
Retriever.retrieve(...)
EvidenceFilter.filter(...)
EvidenceGrader.grade(...)
CitationBuilder.build(...)
```

성과:

- compatibility facade 삭제.
- RAG path 책임이 명확해짐.

## 16. 현재 최종 구조

```text
User Request
  -> RootManufacturingGraph
  -> Checkpoint v2 thread lookup
  -> ContextCompressor
  -> ContextResolver
  -> ContextPackBuilder
  -> IntentGateway / structured classifier
  -> selected_path + answer_type
  -> answer node or heavy manufacturing path
  -> SafetyPolicy if required
  -> FormatterRegistry
  -> AnswerMemoryWriter
  -> checkpoint v2 save
```

Heavy manufacturing path:

```text
DiagnosticFallbackPolicy
  -> DiagnosticPlan
  -> DiagnosticPlanToAgentPlanTranslator
  -> AgentPlan
  -> optional PlanRefiner
  -> Manufacturing Analysis
  -> RAG modules
  -> SafetyGateBuilder
  -> RecommendationBuilder
  -> StructuredAnswerPayloadBuilder
  -> FormatterRegistry
```

## 17. Contract tests로 보장하는 것

| 보장 항목 | 테스트 방향 |
| --- | --- |
| classifier context에 retrieved docs 원문이 들어가지 않음 | context pack contract test |
| 신규 개념 질문이 이전 heavy memory에 오염되지 않음 | context resolver / intent gateway test |
| 권장조치 recap은 RecommendedActionFormatter로 감 | formatter registry test |
| `그중 2번`은 해당 action rationale을 사용 | recommended action item test |
| checkpoint state는 primitive-only | checkpoint state test |
| user/session memory가 섞이지 않음 | thread_id isolation test |
| `RagEvidencePlanner` 삭제 | removal contract test |
| `last_diagnostic_plan` 삭제 | diagnostic planner state test |
| Supervisor private method 의존 제거 | planner contract test |
| heavy module이 `current_turn` raw schema를 읽지 않음 | grep/removal contract |

## 18. 검증 기록

마지막 검증 결과:

```bash
../.venv312/bin/python -m py_compile ...
# pass

../.venv312/bin/python -m pytest tests/test_context_engineering.py -q
# 35 passed

../.venv312/bin/python -m pytest tests/test_intent_gateway.py -q
# 10 passed

../.venv312/bin/python -m pytest -q
# 62 passed
```

삭제 기준 grep 결과:

```text
RagEvidencePlanner: 0
rag_evidence_planner: 0
make_query: 0
last_diagnostic_plan: 0
self.supervisor._: 0
_llm_refine in heavy/root/graph/tests: 0
_intent in heavy/root/graph/tests: 0
_layers in heavy/root/graph/tests: 0
current_turn in heavy/tests: 0
compatibility wrapper wording: 0
```

## 19. 포트폴리오에 어필할 수 있는 성과 문장

- 제조 AI Agent의 대화 맥락 처리를 단순 history injection이 아니라 `ContextResolution`, `ContextPack`, `AnswerMemory` 기반 구조로 재설계했다.
- 단순 개념 질문과 공정 상태 판단 질문을 분리해 불필요한 예측/RAG/Safety 호출을 줄이고, 답변 포맷 오염을 방지했다.
- 사용자별 `thread_id=user_id:session_id`와 checkpoint v2 정책을 적용해 user/session context contamination을 방지했다.
- 권장조치, 후속 질문, 안전 절차, 보고서 생성 등 제조 업무 흐름을 selected_path/answer_type contract로 분리했다.
- RAG path를 query planning, retrieval, filtering, grading, citation building으로 나눠 근거 검색 품질을 검증 가능한 구조로 만들었다.
- OpenAI structured output schema 문제를 공용 strict schema helper로 해결해 intent classifier 안정성을 높였다.
- compatibility facade와 private method 의존을 제거하고, root graph를 orchestration 중심으로 정리했다.

## 20. 아직 남은 리스크와 다음 개선점

| 리스크 | 현재 상태 | 다음 개선 |
| --- | --- | --- |
| SupervisorService 내부 private helper | heavy/root/graph 의존은 제거됐지만 서비스 내부 구현은 남아 있음 | SupervisorService 자체도 public planning service로 재정리 |
| RAG subgraph | 모듈 분리는 완료, LangGraph `Send` 기반 병렬 subgraph는 아직 아님 | query별 parallel retrieval과 local replan 추가 |
| Safety subgraph | SafetyPolicy/Formatter 분리 완료 | action/answer/report validator를 더 엄격히 분리 |
| Chroma integration | ingestion/index script 준비 | 실제 PDF 업로드 UI와 Chroma retriever 연결 |
| cost/usage | usage field 기반 추정 구현 | 운영에서는 OpenAI dashboard/budget과 비교 검증 |
| AnswerMemory 품질 | rule/structured 기반 저장 | 장기적으로 memory extraction 평가셋 추가 |

## 21. 결론

오늘 작업의 핵심 성과는 "특정 문구를 if문으로 맞추는 챗봇"에서 벗어나, 제조 AI Agent가 질문 유형, 이전 답변 기억, 안전 정책, RAG 근거, formatter contract를 분리해서 다루도록 만든 것이다.

이제 시스템은 다음 기준을 만족한다.

- 신규 질문과 follow-up 질문을 구분한다.
- context를 node별로 제한해 prompt/context 오염을 줄인다.
- 단순 질문은 가볍게 처리하고, 제조 판단/정비/안전 질문만 heavy path로 보낸다.
- 답변 포맷은 selected_path/answer_type에 의해 결정된다.
- 권장조치와 후속 질문은 구조화된 `AnswerMemory`를 기반으로 처리된다.
- 레거시 wrapper/facade/private dependency는 제거되었다.
