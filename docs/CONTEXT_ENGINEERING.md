# Context Engineering

작성일: 2026-06-12

## 왜 단순 history injection이 위험한가

이전 대화 전체를 LLM prompt에 그대로 넣으면 토큰 비용이 급격히 증가하고, 현재 센서값·현재 문서 근거·현재 안전 게이트보다 과거 정보가 더 강하게 작동할 수 있다. 제조 안전 도메인에서는 과거 memory가 현재 안전 판단을 덮어쓰는 것이 특히 위험하다.

따라서 이 프로젝트는 전체 history를 그대로 넣지 않고 유저별 context를 계층화한다.

## Context 계층

```text
user profile
session summary
recent runs
similar runs
long-term memory
context policy
```

각 Agent 실행은 `user_id`를 기준으로 context를 만든다. 다른 유저의 history나 memory는 조회하지 않는다.

## 우선순위 정책

LLM prompt에는 다음 규칙이 들어간다.

```text
유저 과거 context는 참고 정보다.
현재 입력된 공정 데이터, 현재 검색된 문서, 현재 safety gate가 과거 context보다 우선한다.
과거 context에 근거해 현재 센서값, 현장 상태, 안전 상태를 단정하지 마라.
과거 context가 현재 질문과 직접 관련 없으면 답변에 사용하지 마라.
```

## Context budget

기본값:

```text
MAX_CONTEXT_TOKENS=2000
MAX_RECENT_RUNS=3
MAX_SIMILAR_RUNS=3
MAX_LONG_TERM_MEMORIES=5
```

정확한 tokenizer 대신 초기 버전에서는 `len(text) // 4` rough estimate를 사용한다.

budget 초과 시 축소 순서:

1. similar runs 제거
2. recent runs 개수 축소
3. long-term memory importance 낮은 것 제거
4. session summary 축약
5. 그래도 초과하면 profile + policy만 유지

## Memory extraction

초기 버전은 LLM 기반 extraction을 쓰지 않는다. 실행 결과에서 rule-based로 아래 memory를 갱신한다.

- `equipment_preference`
- `recurring_failure_mode`
- `report_preference`
- `safety_note`
- `recent_summary`

같은 `user_id + memory_type + memory_key` 조합은 새로 insert하지 않고 upsert한다.

## 삭제 정책

유저 삭제는 두 가지 모드를 지원한다.

```text
DELETE /users/{user_id}?mode=hard
DELETE /users/{user_id}?mode=soft
```

hard delete는 users, user_sessions, user_memories, agent_runs를 삭제한다. soft delete는 `users.deleted_at`만 설정한다.

## 구현 파일

```text
ai_server/app/storage/sqlite_store.py
ai_server/app/services/user_service.py
ai_server/app/services/context_service.py
ai_server/app/services/memory_service.py
ai_server/app/main.py
ai_server/app/agent/graph.py
streamlit_app.py
```

## 포트폴리오 문장

유저별 장기 컨텍스트와 최근 실행 이력을 분리해 관리하고, Agent 실행 시 현재 입력·현재 검색 문서·현재 안전 게이트를 최우선으로 두는 context engineering 구조를 설계했습니다. 단순 대화 history 주입이 아니라 profile, session summary, recent runs, long-term memory를 계층화하고 context budget과 삭제 정책을 함께 고려했습니다.

