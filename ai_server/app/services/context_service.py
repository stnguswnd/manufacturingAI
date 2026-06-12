from __future__ import annotations

import re
from typing import Any

from app.config import MAX_CONTEXT_TOKENS, MAX_LONG_TERM_MEMORIES, MAX_RECENT_RUNS, MAX_SIMILAR_RUNS
from app.schemas import AgentRequest
from app.storage.sqlite_store import SQLiteStore


class ContextService:
    def __init__(self, store: SQLiteStore | None = None):
        self.store = store or SQLiteStore()

    def build(self, *, user_id: str, session_id: str | None, request: AgentRequest) -> dict[str, Any]:
        user = self.store.get_user(user_id)
        if not user:
            return {}
        memories = self.store.list_memories(user_id, limit=50)
        recent_runs = self._recent_runs(user_id, session_id)
        similar_runs = self._similar_runs(user_id, request.question, recent_runs)
        session_summary = self._session_summary(memories, session_id)
        context = {
            'user_profile': {
                'display_name': user.get('display_name'),
                'role': user.get('role'),
                'department': user.get('department'),
                'preferred_language': user.get('preferred_language') or 'ko',
                'report_style': user.get('report_style') or 'standard',
            },
            'session_context': {
                'session_id': session_id,
                'recent_turns_summary': session_summary,
            },
            'long_term_memory': self._memory_items(memories),
            'recent_runs': recent_runs,
            'similar_runs': similar_runs,
            'context_policy': {
                'current_input_priority': 'highest',
                'safety_policy_priority': 'highest',
                'historical_context_priority': 'supporting_only',
            },
        }
        return self._apply_budget(context)

    def rebuild(self, user_id: str) -> dict[str, Any]:
        deleted = self.store.delete_memories(user_id)
        return {'user_id': user_id, 'deleted_memories': deleted, 'rebuilt': False}

    @staticmethod
    def metadata(context: dict[str, Any]) -> dict[str, Any]:
        return {
            'user_id': None,
            'recent_runs_count': len(context.get('recent_runs') or []),
            'similar_runs_count': len(context.get('similar_runs') or []),
            'memories_count': len(context.get('long_term_memory') or []),
            'estimated_context_tokens': context.get('estimated_context_tokens', 0),
            'context_policy': context.get('context_policy') or {},
        }

    def _similar_runs(self, user_id: str, question: str, recent_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        recent_ids = {item.get('run_id') for item in recent_runs}
        terms = self._terms(question)
        if not terms:
            return []
        candidates = self._summarize_runs(self.store.list(limit=50, user_id=user_id))
        scored: list[tuple[int, dict[str, Any]]] = []
        for item in candidates:
            if item.get('run_id') in recent_ids:
                continue
            blob = ' '.join([item.get('question') or '', ' '.join(item.get('failure_modes') or []), ' '.join(item.get('safety_gates') or [])]).lower()
            score = sum(1 for term in terms if term in blob)
            if score:
                scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:MAX_SIMILAR_RUNS]]

    def _recent_runs(self, user_id: str, session_id: str | None) -> list[dict[str, Any]]:
        candidates = self._summarize_runs(self.store.list(limit=50, user_id=user_id))
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        if session_id:
            for item in candidates:
                run_id = item.get('run_id')
                if item.get('session_id') == session_id and run_id not in seen:
                    selected.append(item)
                    seen.add(run_id)
                if len(selected) >= MAX_RECENT_RUNS:
                    return selected
        for item in candidates:
            run_id = item.get('run_id')
            if run_id not in seen:
                selected.append(item)
                seen.add(run_id)
            if len(selected) >= MAX_RECENT_RUNS:
                break
        return selected

    @staticmethod
    def _summarize_runs(runs: list[dict]) -> list[dict[str, Any]]:
        summarized = []
        for row in runs:
            response = row.get('response') or {}
            request = row.get('request') or {}
            mfg = response.get('manufacturing_context') or {}
            failures = [f.get('code') for f in (mfg.get('failure_modes') or []) if isinstance(f, dict)]
            gates = [g.get('gate_id') for g in (mfg.get('safety_gates') or []) if isinstance(g, dict)]
            conditions = [
                {
                    'tag': c.get('tag'),
                    'label_ko': c.get('label_ko'),
                    'severity': c.get('severity'),
                    'value': c.get('value'),
                }
                for c in (mfg.get('process_conditions') or [])
                if isinstance(c, dict)
            ]
            asset = mfg.get('asset_context') or {}
            summarized.append({
                'run_id': row.get('run_id'),
                'session_id': row.get('session_id'),
                'question': request.get('question') or request.get('message') or '',
                'answer_preview': str(response.get('answer') or '')[:500],
                'has_process_data': bool(request.get('process_data')),
                'has_prediction': bool(response.get('prediction')),
                'equipment_type': asset.get('equipment_type') if isinstance(asset, dict) else None,
                'process_conditions': conditions,
                'risk_level': row.get('risk_level'),
                'failure_modes': [x for x in failures if x],
                'safety_gates': [x for x in gates if x],
            })
        return summarized

    @staticmethod
    def _memory_items(memories: list[dict]) -> list[dict[str, Any]]:
        items = []
        for memory in memories:
            if len(items) >= MAX_LONG_TERM_MEMORIES:
                break
            content = memory.get('content') or {}
            items.append({
                'type': memory.get('memory_type'),
                'key': memory.get('memory_key'),
                'content': content.get('summary') or content.get('value') or content,
                'importance': memory.get('importance'),
                'count': content.get('count'),
            })
        return items

    @staticmethod
    def _session_summary(memories: list[dict], session_id: str | None) -> str:
        if not session_id:
            return ''
        for memory in memories:
            if memory.get('memory_type') == 'recent_summary' and memory.get('memory_key') == session_id:
                content = memory.get('content') or {}
                return str(content.get('summary') or '')
        return ''

    def _apply_budget(self, context: dict[str, Any]) -> dict[str, Any]:
        context['estimated_context_tokens'] = self._estimate_tokens(context)
        while context['estimated_context_tokens'] > MAX_CONTEXT_TOKENS:
            if context.get('similar_runs'):
                context['similar_runs'].pop()
            elif len(context.get('recent_runs') or []) > 1:
                context['recent_runs'].pop()
            elif context.get('long_term_memory'):
                context['long_term_memory'].pop()
            elif context.get('session_context', {}).get('recent_turns_summary'):
                context['session_context']['recent_turns_summary'] = context['session_context']['recent_turns_summary'][:400]
            else:
                context['recent_runs'] = []
                context['similar_runs'] = []
                context['long_term_memory'] = []
                context['session_context']['recent_turns_summary'] = ''
                break
            context['estimated_context_tokens'] = self._estimate_tokens(context)
        return context

    @staticmethod
    def _estimate_tokens(value: Any) -> int:
        return max(len(str(value)) // 4, 0)

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {t.lower() for t in re.findall(r'[가-힣A-Za-z0-9_+#.-]+', text or '') if len(t) >= 2}
