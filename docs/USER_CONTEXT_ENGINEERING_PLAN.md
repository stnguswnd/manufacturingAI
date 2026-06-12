# User Context Engineering Plan

작성일: 2026-06-12

## 1. 목표

유저 생성/삭제 기능을 추가하고, 유저별로 이전 대화와 실행 결과를 활용해 Agent가 더 일관된 답변을 만들 수 있도록 컨텍스트 엔지니어링 구조를 설계한다.

현재 시스템에는 `session_id`와 SQLite 실행 이력은 있지만, 다음이 부족하다.

- 유저 단위 identity
- 유저별 선호 설정
- 유저별 장기 기억
- 이전 실행 요약 기반 context injection
- 유저 삭제 시 관련 데이터 삭제 정책
- context가 LLM 비용과 안전성에 미치는 영향 통제

## 2. 핵심 요구사항

### 2.1 User Management

- 유저 생성
- 유저 조회
- 유저 목록 조회
- 유저 수정
- 유저 삭제
- 유저 삭제 시 관련 session, memory, run history 삭제 또는 익명화

### 2.2 User-scoped Context

- 같은 유저의 이전 질의, 설비, 공정 데이터, 위험도, 안전 게이트, 보고서 이력을 참조한다.
- 이전 대화 전체를 무작정 넣지 않고, 요약/선별/최근성 기준으로 context를 구성한다.
- 유저별 업무 범위, 자주 보는 설비, 선호 언어/보고서 형식 같은 profile을 유지한다.

### 2.3 Context Engineering

- LLM prompt에 들어갈 context를 계층화한다.
- 비용 폭주를 막기 위해 context budget을 둔다.
- 안전 관련 정보는 항상 최신 안전 게이트와 현재 입력을 우선한다.
- 과거 정보가 현재 공정 데이터보다 우선하지 않도록 우선순위를 명시한다.

## 3. 권장 UX 흐름

### 3.1 Streamlit

1. Sidebar에 User 영역 추가
   - 유저 선택
   - 새 유저 생성
   - 유저 삭제
   - 유저 상세 보기
2. Agent 탭
   - 선택된 유저 기준으로 Agent 실행
   - 실행 결과는 `user_id`, `session_id`와 함께 저장
3. History 탭
   - 전체 history 대신 선택 유저 history 우선 표시
   - 유저별 token/cost 누적 표시
4. Context 탭 추가
   - 유저 profile
   - 최근 대화 요약
   - 자주 등장한 설비/고장모드/안전 게이트
   - 장기 memory 목록

### 3.2 API

```text
POST   /users
GET    /users
GET    /users/{user_id}
PATCH  /users/{user_id}
DELETE /users/{user_id}

GET    /users/{user_id}/context
POST   /users/{user_id}/context/rebuild
GET    /users/{user_id}/history

POST   /agent/send
POST   /agent/send/stream
```

`/agent/send` 요청에는 `user_id`를 추가한다.

```json
{
  "user_id": "user_abc",
  "session_id": "session_001",
  "message": "지난번처럼 CNC 스핀들 쪽 점검 기준으로 보고서 만들어줘",
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
}
```

## 4. 데이터 모델 설계

현재 `agent_runs` 테이블에 `session_id`만 있다. 아래 테이블을 추가한다.

### 4.1 users

```sql
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    role TEXT,
    department TEXT,
    preferred_language TEXT DEFAULT 'ko',
    report_style TEXT DEFAULT 'standard',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
);
```

### 4.2 user_sessions

```sql
CREATE TABLE user_sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(user_id)
);
```

### 4.3 user_memories

```sql
CREATE TABLE user_memories (
    memory_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    content_json TEXT NOT NULL,
    source_run_id TEXT,
    confidence REAL DEFAULT 1.0,
    importance INTEGER DEFAULT 3,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT,
    FOREIGN KEY(user_id) REFERENCES users(user_id)
);
```

`memory_type` 예시:

- `profile`
- `equipment_preference`
- `recent_summary`
- `recurring_failure_mode`
- `report_preference`
- `safety_note`

### 4.4 agent_runs 확장

```sql
ALTER TABLE agent_runs ADD COLUMN user_id TEXT;
CREATE INDEX idx_agent_runs_user_created ON agent_runs(user_id, created_at);
```

## 5. Context Layer 설계

Agent 실행 전 `ContextBuilder`를 추가한다.

```text
AgentSendRequest
→ UserService.validate_user()
→ ContextBuilder.build(user_id, session_id, current_request)
→ ManufacturingAgentGraph.run()
→ LLM payload에 user_context 포함
→ 실행 후 MemoryExtractor.update()
→ History 저장
```

### 5.1 ContextBuilder 출력

```json
{
  "user_profile": {
    "display_name": "Kim",
    "role": "maintenance_engineer",
    "preferred_language": "ko",
    "report_style": "standard"
  },
  "session_context": {
    "session_id": "session_001",
    "recent_turns_summary": "최근 CNC 스핀들 과부하와 공구 마모 이슈를 확인했다."
  },
  "long_term_memory": [
    {
      "type": "equipment_preference",
      "content": "CNC 설비 관련 질문이 많고 스핀들/냉각 계통을 자주 다룬다.",
      "importance": 4
    }
  ],
  "recent_runs": [
    {
      "run_id": "...",
      "question": "...",
      "risk_level": "high",
      "failure_modes": ["OSF", "TWF"],
      "safety_gates": ["loto_if_physical_maintenance"]
    }
  ],
  "context_policy": {
    "current_input_priority": "highest",
    "safety_policy_priority": "highest",
    "historical_context_priority": "supporting_only"
  }
}
```

### 5.2 LLM Prompt 우선순위

LLM system prompt에 아래 규칙을 추가한다.

```text
유저 과거 context는 참고 정보다.
현재 입력된 공정 데이터, 현재 검색된 문서, 현재 safety gate가 과거 context보다 우선한다.
과거 context에 근거해 현재 센서값, 현장 상태, 안전 상태를 단정하지 마라.
과거 context가 현재 질문과 직접 관련 없으면 답변에 사용하지 마라.
```

## 6. Context Selection 정책

전체 history를 넣으면 비용과 hallucination 위험이 커진다. 다음 기준으로 제한한다.

| Context 종류 | 선택 기준 | 기본 개수 |
|---|---|---|
| User profile | 항상 포함 | 1 |
| Session summary | 같은 session이면 포함 | 1 |
| Recent runs | 같은 user의 최신 실행 | 3 |
| Similar runs | 질문 키워드/설비/고장모드 유사도 | 3 |
| Long-term memory | importance 높은 순 | 5 |
| Safety memory | 현재 safety gate와 직접 관련될 때만 | 3 |

초기 구현은 SQLite keyword matching으로 시작하고, 나중에 embedding 기반 retrieval로 확장한다.

## 7. Memory Extraction 정책

Agent 실행 후 아래 정보를 자동 추출해 user memory로 저장한다.

### 7.1 저장할 수 있는 정보

- 자주 묻는 설비 유형
- 반복 등장하는 고장모드
- 자주 필요한 보고서 형식
- 사용자가 명시한 선호 언어/출력 방식
- 반복되는 안전 게이트

### 7.2 저장하지 말아야 할 정보

- API key
- 개인정보
- 민감한 현장 사고 정보 원문
- 근거 없는 추정
- 안전 상태 보증 표현

### 7.3 Memory write 방식

초기 버전은 rule-based extraction으로 충분하다.

```text
if asset_context.equipment_type:
  upsert equipment_preference

if failure_modes:
  increment recurring_failure_mode count

if generate_report:
  upsert report_preference

if safety_gates:
  upsert safety_note summary
```

LLM 기반 memory extraction은 나중에 추가한다. 이유는 비용과 개인정보 리스크 때문이다.

## 8. 삭제 정책

유저 삭제는 2가지 모드를 둔다.

### 8.1 Soft Delete

- `users.deleted_at`만 설정
- history와 memory는 유지하지만 UI/API에서 기본 제외
- 운영 감사나 복구가 필요할 때 유리

### 8.2 Hard Delete

- `users`, `user_sessions`, `user_memories`, 해당 user의 `agent_runs` 삭제
- 포트폴리오/로컬 데모에서는 hard delete가 이해하기 쉽다

초기 구현은 hard delete를 기본으로 하고, API 파라미터로 soft delete를 선택할 수 있게 한다.

```text
DELETE /users/{user_id}?mode=hard
DELETE /users/{user_id}?mode=soft
```

## 9. API 상세 기획

### 9.1 Create User

```http
POST /users
```

```json
{
  "display_name": "Maintenance Engineer A",
  "role": "maintenance_engineer",
  "department": "plant_1",
  "preferred_language": "ko",
  "report_style": "standard"
}
```

응답:

```json
{
  "user_id": "usr_...",
  "display_name": "Maintenance Engineer A",
  "created_at": "..."
}
```

### 9.2 Delete User

```http
DELETE /users/usr_xxx?mode=hard
```

응답:

```json
{
  "deleted": true,
  "mode": "hard",
  "deleted_counts": {
    "sessions": 2,
    "memories": 8,
    "runs": 15
  }
}
```

### 9.3 Get User Context

```http
GET /users/usr_xxx/context?session_id=session_001
```

응답:

```json
{
  "user_id": "usr_xxx",
  "context": {
    "profile": {},
    "session_summary": {},
    "long_term_memory": [],
    "recent_runs": []
  },
  "estimated_context_tokens": 1200
}
```

## 10. 코드 구조 제안

```text
ai_server/app/
  services/
    user_service.py
    context_service.py
    memory_service.py
  storage/
    sqlite_store.py
  schemas.py
  main.py
```

### 10.1 UserService

책임:

- create/list/get/update/delete user
- user_id 검증
- user 삭제 cascade 처리

### 10.2 ContextService

책임:

- user profile 조회
- recent runs 조회
- long-term memory 조회
- context budget 적용
- LLM payload에 넣을 `user_context` 생성

### 10.3 MemoryService

책임:

- Agent 실행 결과에서 memory 후보 추출
- user memory upsert
- session summary 갱신

### 10.4 Storage

현재 `JsonLineStore`는 이름과 역할이 맞지 않으므로 다음 단계에서 `SQLiteStore` 또는 `AgentRunStore`로 이름 변경을 권장한다.

## 11. Streamlit UI 기획

### 11.1 Sidebar

- User selectbox
- `Create user` button
- `Delete user` button
- User metadata editor

### 11.2 Agent Tab

- 선택된 user 표시
- `session_id` 자동 생성 또는 선택
- Agent 실행 시 `user_id`, `session_id` 전송
- 답변 하단에 “사용된 user context” expander 표시

### 11.3 Context Tab

- Profile
- Session summary
- Long-term memories
- Recent runs
- Context rebuild button

### 11.4 History Tab

- 선택 user 기준 history filter
- user별 누적 LLM calls/tokens/cost 표시

## 12. 보안/개인정보 고려사항

- API key는 절대 memory에 저장하지 않는다.
- user memory에는 민감정보 원문 대신 요약만 저장한다.
- hard delete 시 user 관련 모든 run/context/memory를 삭제한다.
- context payload를 Raw response에 그대로 노출할지 여부는 설정으로 제어한다.
- 운영 환경에서는 user API도 `require_api_key` 적용 대상이다.

## 13. 관측성 설계

OpenTelemetry span에 아래 attribute를 추가한다.

```text
app.user_id_hash
app.session_id
app.context.recent_runs_count
app.context.memories_count
app.context.estimated_tokens
app.memory.updated_count
```

주의:

- raw user_id 대신 hash를 기록한다.
- memory content 원문은 span attribute에 기록하지 않는다.

## 14. 구현 단계

### Phase 1: User CRUD

- schemas 추가
- users/user_sessions/user_memories table 추가
- UserService 추가
- `/users` API 추가
- Streamlit user select/create/delete 추가

완료 기준:

- UI에서 user 생성/삭제 가능
- `/agent/send`가 `user_id`를 받을 수 있음
- history에 `user_id` 저장

### Phase 2: User History Context

- ContextService 추가
- 같은 user의 recent runs를 context로 구성
- LLM payload에 `user_context` 추가
- Streamlit에서 사용된 context 표시

완료 기준:

- “지난번처럼” 같은 요청에서 같은 user의 최근 실행 요약을 참고
- 다른 user의 history가 섞이지 않음

### Phase 3: Long-term Memory

- MemoryService 추가
- 실행 후 equipment/failure/safety/report preference memory upsert
- `/users/{user_id}/context` API 추가
- Context tab 추가

완료 기준:

- 유저별 자주 묻는 설비/고장모드가 memory로 누적
- context rebuild 가능

### Phase 4: Context Quality Control

- context budget
- memory importance/expiration
- user별 누적 cost
- context 사용 여부 trace 기록
- golden tests 추가

완료 기준:

- context가 너무 길면 자동 축약
- token/cost 증가가 UI에 표시
- context injection 회귀 테스트 통과

## 15. 테스트 계획

| 테스트 | 검증 내용 |
|---|---|
| User CRUD test | 생성/조회/삭제 정상 동작 |
| Delete cascade test | hard delete 시 sessions/memories/runs 삭제 |
| User isolation test | A 유저 context가 B 유저 응답에 섞이지 않음 |
| Context budget test | 제한 개수/토큰 이상 context 제외 |
| Memory extraction test | 실행 결과에서 설비/고장모드 memory 저장 |
| Agent integration test | user_id 포함 Agent 실행 후 history 저장 |
| Safety priority test | 과거 context보다 현재 safety gate가 우선 |

## 16. 포트폴리오에서 강조할 문장

```text
유저별 장기 컨텍스트와 최근 실행 이력을 분리해 관리하고, Agent 실행 시 현재 입력·현재 검색 문서·현재 안전 게이트를 최우선으로 두는 context engineering 구조를 설계했습니다. 단순 대화 history 주입이 아니라 profile, session summary, recent runs, long-term memory를 계층화하고 context budget과 삭제 정책을 함께 고려했습니다.
```

