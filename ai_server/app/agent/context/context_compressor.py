from __future__ import annotations

from typing import Any

from app.agent.context.schemas import CompressedContext


class ContextCompressor:
    def __init__(self, *, max_recent_turns: int = 5, max_summary_chars: int = 1200):
        self.max_recent_turns = max_recent_turns
        self.max_summary_chars = max_summary_chars

    def compress(self, *, messages: list[Any], previous_rolling_summary: str | None = None) -> CompressedContext:
        serial = [self._message_to_dict(message) for message in messages]
        max_recent_messages = max(self.max_recent_turns * 2, 2)
        recent = serial[-max_recent_messages:]
        older = serial[:-max_recent_messages]
        rolling_summary = previous_rolling_summary or ''
        if older:
            older_summary = ' '.join(f'{item["role"]}: {item["content"][:120]}' for item in older)
            rolling_summary = self._trim(' '.join(part for part in [rolling_summary, older_summary] if part).strip())
        return CompressedContext(rolling_summary=rolling_summary, recent_turns=recent, recent_turn_count=len(recent), compressed_message_count=len(older), max_recent_turns=self.max_recent_turns)

    def _trim(self, text: str) -> str:
        if len(text) <= self.max_summary_chars:
            return text
        return text[-self.max_summary_chars:]

    @staticmethod
    def _message_to_dict(message: Any) -> dict[str, str]:
        if isinstance(message, dict):
            return {'role': str(message.get('role') or ''), 'content': str(message.get('content') or '')}
        role = getattr(message, 'type', '') or message.__class__.__name__
        content = str(getattr(message, 'content', '') or '')
        return {'role': role, 'content': content}
