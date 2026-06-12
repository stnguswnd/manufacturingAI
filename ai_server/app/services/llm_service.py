from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from app.config import (
    LLM_ENABLE_STRUCTURED_OUTPUT,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_MODEL_CATALOG,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_SELECTABLE_MODELS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_ORG_ID,
    OPENAI_PROJECT_ID,
    USD_KRW_EXCHANGE_RATE,
)
from app.errors import LLMUnavailableError, ModelSelectionError
from app.schemas import LLMUsageRecord
from app.services.observability_service import record_llm_usage_span


class LLMService:
    """Thin external LLM adapter.

    Design choice:
    - For official OpenAI, use the Responses API because it supports modern
      agent workflows, function/tool calling, and structured JSON outputs.
    - For OpenAI-compatible gateways, use Chat Completions because many local
      or third-party providers expose that interface first.
    - Missing provider/key is treated as configuration error instead of silently
      switching to template output.
    """

    def __init__(self) -> None:
        self.provider = LLM_PROVIDER
        self.model = LLM_MODEL
        self.last_error: str | None = None

    @property
    def enabled(self) -> bool:
        if self.provider not in {'openai', 'openai_responses', 'openai_compatible', 'compatible'}:
            return False
        if self.provider in {'openai', 'openai_responses'} and not OPENAI_API_KEY:
            return False
        if self.provider in {'openai_compatible', 'compatible'} and not OPENAI_API_KEY:
            return False
        return True

    def generate_json(
        self,
        *,
        schema_name: str,
        schema: dict[str, Any],
        system_prompt: str,
        payload: dict[str, Any],
        model: str | None = None,
        operation: str | None = None,
        usage_callback: Callable[[LLMUsageRecord], None] | None = None,
    ) -> dict[str, Any] | None:
        self.last_error = None
        if not self.enabled:
            raise LLMUnavailableError(f'LLM provider/key is not configured: provider={self.provider or "empty"}')
        selected_model = self._resolve_model(model)
        try:
            if self.provider in {'openai', 'openai_responses'}:
                return self._responses_json(schema_name=schema_name, schema=schema, system_prompt=system_prompt, payload=payload, model=selected_model, operation=operation or schema_name, usage_callback=usage_callback)
            if self.provider in {'openai_compatible', 'compatible'}:
                return self._chat_json(schema_name=schema_name, schema=schema, system_prompt=system_prompt, payload=payload, model=selected_model, operation=operation or schema_name, usage_callback=usage_callback)
            raise LLMUnavailableError(f'Unsupported LLM_PROVIDER={self.provider}')
        except ModelSelectionError:
            raise
        except LLMUnavailableError:
            raise
        except Exception as exc:
            self.last_error = f'{type(exc).__name__}: {exc}'
            return None

    def _resolve_model(self, model: str | None = None) -> str:
        selected = (model or self.model).strip()
        if selected not in LLM_SELECTABLE_MODELS:
            info = LLM_MODEL_CATALOG.get(selected)
            if info and info.get('tier') == 'expensive':
                raise ModelSelectionError(f'{selected} is marked expensive and is disabled')
            raise ModelSelectionError(f'{selected} is not in selectable model list')
        return selected

    def _client(self):
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError('openai package is not installed. Run: pip install -r requirements.txt') from exc
        kwargs: dict[str, Any] = {'api_key': OPENAI_API_KEY, 'timeout': LLM_TIMEOUT_SECONDS}
        if OPENAI_BASE_URL:
            kwargs['base_url'] = OPENAI_BASE_URL
        if OPENAI_ORG_ID:
            kwargs['organization'] = OPENAI_ORG_ID
        if OPENAI_PROJECT_ID:
            kwargs['project'] = OPENAI_PROJECT_ID
        return OpenAI(**kwargs)

    def _responses_json(self, *, schema_name: str, schema: dict[str, Any], system_prompt: str, payload: dict[str, Any], model: str, operation: str, usage_callback: Callable[[LLMUsageRecord], None] | None) -> dict[str, Any] | None:
        client = self._client()
        user_input = json.dumps(payload, ensure_ascii=False, indent=2)
        kwargs: dict[str, Any] = {
            'model': model,
            'input': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_input},
            ],
        }
        if LLM_TEMPERATURE is not None:
            kwargs['temperature'] = LLM_TEMPERATURE
        # Responses API structured output shape.
        if LLM_ENABLE_STRUCTURED_OUTPUT:
            kwargs['text'] = {
                'format': {
                    'type': 'json_schema',
                    'name': schema_name,
                    'schema': schema,
                    'strict': True,
                }
            }
        if LLM_MAX_OUTPUT_TOKENS > 0:
            kwargs['max_output_tokens'] = LLM_MAX_OUTPUT_TOKENS
        started = time.perf_counter()
        try:
            response = client.responses.create(**kwargs)
        except Exception as exc:
            if 'temperature' in str(exc).lower() and 'temperature' in kwargs:
                kwargs.pop('temperature', None)
                response = client.responses.create(**kwargs)
            else:
                raise
        self._record_usage(response=response, model=model, operation=operation, started=started, usage_callback=usage_callback)
        text = getattr(response, 'output_text', None) or self._collect_response_text(response)
        return self._parse_json(text)

    def _chat_json(self, *, schema_name: str, schema: dict[str, Any], system_prompt: str, payload: dict[str, Any], model: str, operation: str, usage_callback: Callable[[LLMUsageRecord], None] | None) -> dict[str, Any] | None:
        client = self._client()
        prompt = (
            f'{system_prompt}\n\n'
            f'반드시 아래 JSON Schema에 맞는 JSON 객체만 출력하세요. Schema name: {schema_name}\n'
            f'{json.dumps(schema, ensure_ascii=False)}\n\n'
            f'입력:\n{json.dumps(payload, ensure_ascii=False, indent=2)}'
        )
        kwargs: dict[str, Any] = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'response_format': {'type': 'json_object'},
            'timeout': LLM_TIMEOUT_SECONDS,
        }
        if LLM_TEMPERATURE is not None:
            kwargs['temperature'] = LLM_TEMPERATURE
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:
            if 'temperature' in str(exc).lower() and 'temperature' in kwargs:
                kwargs.pop('temperature', None)
                response = client.chat.completions.create(**kwargs)
            else:
                raise
        self._record_usage(response=response, model=model, operation=operation, started=started, usage_callback=usage_callback)
        text = response.choices[0].message.content or ''
        return self._parse_json(text)

    def _record_usage(self, *, response: Any, model: str, operation: str, started: float, usage_callback: Callable[[LLMUsageRecord], None] | None) -> None:
        record = self._usage_record(response=response, model=model, operation=operation, latency_ms=(time.perf_counter() - started) * 1000)
        if not record:
            return
        record_llm_usage_span(record)
        if usage_callback:
            usage_callback(record)

    def _usage_record(self, *, response: Any, model: str, operation: str, latency_ms: float) -> LLMUsageRecord | None:
        usage = getattr(response, 'usage', None)
        if usage is None and isinstance(response, dict):
            usage = response.get('usage')
        if usage is None:
            return None

        input_tokens = self._usage_int(usage, 'input_tokens', 'prompt_tokens')
        output_tokens = self._usage_int(usage, 'output_tokens', 'completion_tokens')
        total_tokens = self._usage_int(usage, 'total_tokens') or (input_tokens + output_tokens)
        cached_input_tokens = self._cached_input_tokens(usage)
        cost = self.estimate_cost_usd(model=model, input_tokens=input_tokens, output_tokens=output_tokens, cached_input_tokens=cached_input_tokens)
        cost_krw = self.estimate_cost_krw(cost)
        return LLMUsageRecord(
            provider=self.provider,
            model=model,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
            estimated_cost_krw=cost_krw,
            usd_krw_exchange_rate=USD_KRW_EXCHANGE_RATE,
            latency_ms=round(latency_ms, 2),
        )

    @staticmethod
    def _usage_int(usage: Any, *names: str) -> int:
        for name in names:
            value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return 0
        return 0

    @staticmethod
    def _cached_input_tokens(usage: Any) -> int:
        for details_name in ['input_tokens_details', 'prompt_tokens_details']:
            details = usage.get(details_name) if isinstance(usage, dict) else getattr(usage, details_name, None)
            if details is None:
                continue
            value = details.get('cached_tokens') if isinstance(details, dict) else getattr(details, 'cached_tokens', None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return 0
        return 0

    @staticmethod
    def estimate_cost_usd(*, model: str, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> float:
        price = LLM_MODEL_CATALOG.get(model) or {}
        input_rate = float(price.get('input_per_1m') or 0.0)
        cached_rate = float(price.get('cached_input_per_1m') or input_rate)
        output_rate = float(price.get('output_per_1m') or 0.0)
        cached = min(max(cached_input_tokens, 0), max(input_tokens, 0))
        uncached = max(input_tokens - cached, 0)
        cost = (uncached * input_rate + cached * cached_rate + max(output_tokens, 0) * output_rate) / 1_000_000
        return round(cost, 8)

    @staticmethod
    def estimate_cost_krw(cost_usd: float) -> float:
        return round(float(cost_usd or 0.0) * USD_KRW_EXCHANGE_RATE, 2)

    @staticmethod
    def _collect_response_text(response: Any) -> str:
        parts: list[str] = []
        for item in getattr(response, 'output', []) or []:
            for content in getattr(item, 'content', []) or []:
                value = getattr(content, 'text', None)
                if value:
                    parts.append(value)
        return '\n'.join(parts)

    @staticmethod
    def _parse_json(text: str | None) -> dict[str, Any] | None:
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Be permissive for OpenAI-compatible providers that sometimes wrap JSON.
            match = re.search(r'\{.*\}', text, re.S)
            if not match:
                raise
            return json.loads(match.group(0))


PLAN_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'additionalProperties': False,
    'properties': {
        'intent': {'type': 'string', 'enum': ['prediction','knowledge_qa','safety_ops','documentation','hybrid','general']},
        'confidence': {'type': 'number'},
        'prediction_required': {'type': 'boolean'},
        'rag_required': {'type': 'boolean'},
        'safety_required': {'type': 'boolean'},
        'report_required': {'type': 'boolean'},
        'rag_query': {'type': 'string'},
        'rationale': {'type': 'string'},
    },
    'required': ['intent','confidence','prediction_required','rag_required','safety_required','report_required','rag_query','rationale'],
}

ANSWER_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'additionalProperties': False,
    'properties': {
        'answer': {'type': 'string'},
        'safety_guidance': {'type': ['string','null']},
        'recommended_actions': {'type': 'array', 'items': {'type': 'string'}},
        'report': {'type': ['string','null']},
        'warnings': {'type': 'array', 'items': {'type': 'string'}},
    },
    'required': ['answer','safety_guidance','recommended_actions','report','warnings'],
}
