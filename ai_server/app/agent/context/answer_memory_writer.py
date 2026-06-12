from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.schemas import AgentResponse
from app.agent.context.schemas import AnswerMemory, RecommendedAction


class AnswerMemoryWriter:
    def build(self, *, state: dict[str, Any], response: AgentResponse) -> AnswerMemory:
        gateway = state.get('intent_gateway') or {}
        formatter_context = state.get('formatter_context') or {}
        structured_payload = state.get('structured_answer_payload') or {}
        concept_payload = structured_payload.get('concept') or formatter_context.get('concept_payload') or {}
        context_resolution = state.get('context_resolution') or {}
        selected_path = state.get('selected_path') or gateway.get('selected_path') or 'unknown'
        answer_type = gateway.get('answer_type') or formatter_context.get('answer_type') or 'unknown'
        actions = self.normalize_recommended_actions(structured_payload.get('recommended_actions') or formatter_context.get('recommended_actions') or [])
        if not actions and response.answer and selected_path not in {'fast_concept_answer', 'general_lightweight_answer', 'meta_feedback', 'unsupported_or_clarification'}:
            actions = self.normalize_recommended_actions(self._extract_action_like_lines(response.answer))
        source_refs = [str(item.get('source') or item.get('title') or item) for item in (response.citations or [])]
        focus = (
            ((gateway.get('resolved_reference') or {}).get('normalized') if isinstance(gateway.get('resolved_reference'), dict) else None)
            or concept_payload.get('term')
            or context_resolution.get('followup_target')
        )
        mentioned_entities = self._merge_focus_entity(focus, self._mentioned_entities(response.answer))
        short_summary = self._summary(response.answer)
        claims = self._claims(
            selected_path=selected_path,
            answer_type=answer_type,
            focus=focus or (mentioned_entities[0] if mentioned_entities else None),
            actions=actions,
            concept_payload=concept_payload,
            answer=response.answer,
        )
        return AnswerMemory(
            selected_path=selected_path,
            answer_type=answer_type,
            user_intent=gateway.get('turn_type') or gateway.get('reason') or selected_path,
            short_summary=short_summary,
            focus=(focus or (mentioned_entities[0] if mentioned_entities else None)),
            key_points=self._key_points(response.answer),
            claims=claims,
            recommended_actions=actions[:10],
            decisions=self._decisions(response.answer, selected_path=selected_path, answer_type=answer_type),
            code_changes=[],
            mentioned_entities=mentioned_entities,
            unresolved_questions=self._unresolved_questions(response.answer),
            source_refs=source_refs,
            safety_level=self._safety_level(response.answer),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def key_phrases_and_claims(self, memory: AnswerMemory | dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
        data = memory.model_dump() if isinstance(memory, AnswerMemory) else memory
        key_phrases = list(dict.fromkeys([
            data.get('short_summary') or '',
            *data.get('key_points', [])[:6],
            *[AnswerMemoryWriter._action_title(item) for item in data.get('recommended_actions', [])[:6]],
            *data.get('mentioned_entities', [])[:6],
        ]))
        key_phrases = [item for item in key_phrases if item]
        claims: list[dict[str, Any]] = []
        if data.get('claims'):
            for claim in data.get('claims') or []:
                claims.append({
                    'claim': str(claim),
                    'target': data.get('focus'),
                    'domain_target': data.get('focus'),
                    'answer_type': data.get('answer_type'),
                    'reason_type': 'answer_memory_claim',
                })
        elif data.get('recommended_actions'):
            claims.append({
                'claim': '직전 답변에는 점검과 안전 절차에 대한 권장조치가 포함되어 있다',
                'target': '권장조치',
                'domain_target': '점검 및 안전 절차',
                'answer_type': data.get('answer_type'),
                'reason_type': 'maintenance_action_prioritization',
            })
        return key_phrases, claims

    @staticmethod
    def _summary(answer: str) -> str:
        text = re.sub(r'\s+', ' ', answer or '').strip()
        if not text:
            return ''
        return text[:220]

    @staticmethod
    def _key_points(answer: str) -> list[str]:
        points: list[str] = []
        for line in (answer or '').splitlines():
            clean = line.strip().lstrip('-').strip()
            if clean and len(clean) >= 8:
                points.append(clean)
            if len(points) >= 8:
                break
        return points

    @staticmethod
    def _extract_action_like_lines(answer: str) -> list[str]:
        actions: list[str] = []
        action_terms = ['확인', '점검', '요청', '기록', '차단', '보고']
        for line in (answer or '').splitlines():
            clean = line.strip().lstrip('-').strip()
            if clean and any(term in clean for term in action_terms):
                actions.append(clean)
            if len(actions) >= 10:
                break
        return actions

    @staticmethod
    def normalize_recommended_actions(items: list[Any]) -> list[RecommendedAction]:
        actions: list[RecommendedAction] = []
        for idx, item in enumerate(items, start=1):
            if isinstance(item, RecommendedAction):
                actions.append(item)
            elif isinstance(item, dict):
                title = str(item.get('title') or item.get('action') or item.get('text') or '').strip()
                if title:
                    actions.append(RecommendedAction(
                        id=str(item.get('id') or f'action_{idx}'),
                        title=title,
                        rationale=item.get('rationale'),
                        safety_note=item.get('safety_note'),
                        priority=item.get('priority') or idx,
                    ))
            else:
                title = str(item or '').strip()
                if title:
                    actions.append(RecommendedAction(id=f'action_{idx}', title=title, priority=idx))
        return actions

    @staticmethod
    def _decisions(answer: str, *, selected_path: str, answer_type: str) -> list[str]:
        decisions = [f'selected_path={selected_path}', f'answer_type={answer_type}']
        if '위험' in (answer or ''):
            decisions.append('risk_related_answer')
        return decisions

    @staticmethod
    def _mentioned_entities(answer: str) -> list[str]:
        candidates = ['토크', '공구 마모', '마모', '스핀들', '회전수', 'LOTO', 'OSF', 'TWF', 'HDF', 'PWF', '권장조치', '안전 절차']
        return [item for item in candidates if item.lower() in (answer or '').lower()]

    @staticmethod
    def _merge_focus_entity(focus: str | None, entities: list[str]) -> list[str]:
        if not focus:
            return entities
        return list(dict.fromkeys([focus, *entities]))

    @staticmethod
    def _action_title(item: Any) -> str:
        if isinstance(item, dict):
            return str(item.get('title') or '').strip()
        if isinstance(item, RecommendedAction):
            return item.title
        return str(item or '').strip()

    @staticmethod
    def _claims(
        *,
        selected_path: str,
        answer_type: str,
        focus: str | None,
        actions: list[str],
        concept_payload: dict[str, Any],
        answer: str,
    ) -> list[str]:
        claims: list[str] = []
        if actions:
            claims.append('직전 답변에는 점검과 안전 절차에 대한 권장조치가 포함되어 있다')
        if selected_path == 'fast_concept_answer' and focus:
            watch_points = concept_payload.get('watch_points') or []
            if answer_type in {'definition', 'watch_points'} and watch_points:
                claims.append(f'{focus}는 값 하나만 보지 말고 여러 지표와 함께 봐야 한다')
        return claims[:5]

    @staticmethod
    def _unresolved_questions(answer: str) -> list[str]:
        markers = ['추가 정보', '구체적으로 지정', '확인할 수 없습니다']
        if any(marker in (answer or '') for marker in markers):
            return ['사용자 추가 정보 필요']
        return []

    @staticmethod
    def _safety_level(answer: str) -> str | None:
        if '위험도' in (answer or '') or '안전' in (answer or ''):
            if '높음' in answer:
                return 'high'
            return 'mentioned'
        return None
