# Architecture Decisions

## ADR-001: LLM-only Runtime

- Date: 2026-06-12
- Status: accepted
- Context: template/mock 경로와 `use_llm=false` 옵션이 실제 운영 경로와 테스트 경로를 혼재시켰다.
- Decision: Agent 실행은 OpenAI 또는 OpenAI-compatible LLM 설정이 있어야 가능하게 한다.
- Alternatives: API key가 없을 때 템플릿 답변으로 대체.
- Consequences: 포트폴리오 데모는 실제 LLM 비용이 발생하지만, 동작 경로가 명확해진다.
- Related features: F-001

## ADR-002: Usage-based Cost Calculation

- Date: 2026-06-12
- Status: accepted
- Context: 토큰 추정은 실제 과금과 오차가 크다.
- Decision: OpenAI 응답 `usage`에서 input/output/cached tokens를 읽고, 내부 가격표로 예상 비용을 계산한다.
- Alternatives: tokenizer 기반 사전 추정, OpenTelemetry metric만 사용.
- Consequences: 응답을 받은 뒤 비용을 정확히 집계할 수 있다. 사전 예산 차단은 별도 기능이 필요하다.
- Related features: F-003

## ADR-003: Domain Catalog as YAML

- Date: 2026-06-12
- Status: accepted
- Context: 설비, 고장모드, 안전 절차는 계속 바뀌며 코드보다 설정으로 관리하는 편이 낫다.
- Decision: equipment taxonomy, failure mode, safety gate, action catalog를 YAML로 분리한다.
- Alternatives: Python 코드에 하드코딩, DB 관리.
- Consequences: 포트폴리오 데모에서는 읽기 쉽고, 운영 확장 시 DB/관리 UI로 이전 가능하다.
- Related features: Domain Modeling

## ADR-004: Safety Failure Blocks Response

- Date: 2026-06-12
- Status: accepted
- Context: 안전 검증 실패를 조용히 대체 답변으로 숨기면 위험한 응답 품질 문제가 드러나지 않는다.
- Decision: 안전 검증 실패는 re-plan 후 재시도하고, 최종 실패 시 차단한다.
- Alternatives: 템플릿 안전 답변으로 대체.
- Consequences: 실패가 사용자에게 노출될 수 있지만, 안전 기준을 우선한다.
- Related features: F-002

## ADR-005: Expensive Model Disabled by Policy

- Date: 2026-06-12
- Status: accepted
- Context: 포트폴리오 데모 중 고비용 모델을 실수로 선택할 수 있다.
- Decision: 모델 catalog에 `selectable` 정책을 두고 UI/API 양쪽에서 제한한다.
- Alternatives: UI에서만 숨김, 사용자에게 경고만 표시.
- Consequences: 정책이 서버에서 강제되므로 UI 우회 요청도 막을 수 있다.
- Related features: F-004

## ADR-006: User Context Is Supporting Evidence Only

- Date: 2026-06-12
- Status: accepted
- Context: 유저별 과거 대화와 memory를 LLM에 넣으면 편의성은 좋아지지만, 과거 정보가 현재 센서값이나 안전 판단을 덮어쓸 위험이 있다.
- Decision: user context는 supporting evidence로만 사용하고, 현재 입력, 현재 RAG 문서, 현재 safety gate를 최우선으로 둔다.
- Alternatives: 최근 대화 전체를 prompt에 그대로 삽입.
- Consequences: context 품질은 더 안정적이지만, ContextBuilder와 MemoryService 구현이 필요하다.
- Related features: F-005
