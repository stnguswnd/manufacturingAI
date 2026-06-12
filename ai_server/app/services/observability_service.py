from __future__ import annotations

import hashlib

from app.schemas import LLMUsageRecord, LLMUsageSummary


def record_llm_usage_span(record: LLMUsageRecord) -> None:
    """Record LLM usage on an OpenTelemetry span when OTel is installed.

    Without an SDK/exporter this is a no-op, which keeps local development and
    tests dependency-light while still exposing standard-ish attributes when an
    exporter is configured later.
    """
    try:
        from opentelemetry import trace
    except Exception:
        return

    try:
        tracer = trace.get_tracer('manufacturing-ai-agent')
        with tracer.start_as_current_span('llm.call') as span:
            span.set_attribute('gen_ai.system', record.provider)
            span.set_attribute('gen_ai.request.model', record.model)
            span.set_attribute('gen_ai.operation.name', record.operation)
            span.set_attribute('gen_ai.usage.input_tokens', record.input_tokens)
            span.set_attribute('gen_ai.usage.output_tokens', record.output_tokens)
            span.set_attribute('gen_ai.usage.cached_input_tokens', record.cached_input_tokens)
            span.set_attribute('gen_ai.usage.total_tokens', record.total_tokens)
            span.set_attribute('gen_ai.usage.estimated_cost_usd', record.estimated_cost_usd)
            span.set_attribute('gen_ai.usage.estimated_cost_krw', record.estimated_cost_krw)
            span.set_attribute('gen_ai.usage.usd_krw_exchange_rate', record.usd_krw_exchange_rate)
            span.set_attribute('gen_ai.latency_ms', record.latency_ms)
    except Exception:
        return


def record_agent_run_span(
    *,
    run_id: str,
    route: list[str],
    llm_provider: str,
    llm_model: str,
    llm_used: bool,
    usage: LLMUsageSummary,
    context_used: dict | None = None,
) -> None:
    """Record aggregate agent-run usage on an OpenTelemetry span."""
    try:
        from opentelemetry import trace
    except Exception:
        return

    try:
        tracer = trace.get_tracer('manufacturing-ai-agent')
        with tracer.start_as_current_span('agent.run') as span:
            span.set_attribute('agent.run_id', run_id)
            span.set_attribute('agent.route', ','.join(route))
            span.set_attribute('agent.llm_used', llm_used)
            span.set_attribute('agent.replan_count', usage.replan_count)
            if context_used:
                user_id = context_used.get('actual_user_id') or context_used.get('user_id')
                if user_id:
                    span.set_attribute('app.user_id_hash', hashlib.sha256(str(user_id).encode()).hexdigest()[:16])
                if context_used.get('session_id'):
                    span.set_attribute('app.session_id', str(context_used.get('session_id')))
                span.set_attribute('app.context.recent_runs_count', int(context_used.get('recent_runs_count') or 0))
                span.set_attribute('app.context.memories_count', int(context_used.get('memories_count') or 0))
                span.set_attribute('app.context.estimated_tokens', int(context_used.get('estimated_context_tokens') or 0))
            span.set_attribute('gen_ai.system', llm_provider)
            span.set_attribute('gen_ai.request.model', llm_model)
            span.set_attribute('gen_ai.usage.llm_calls', usage.calls)
            span.set_attribute('gen_ai.usage.input_tokens', usage.input_tokens)
            span.set_attribute('gen_ai.usage.output_tokens', usage.output_tokens)
            span.set_attribute('gen_ai.usage.cached_input_tokens', usage.cached_input_tokens)
            span.set_attribute('gen_ai.usage.total_tokens', usage.total_tokens)
            span.set_attribute('gen_ai.usage.estimated_cost_usd', usage.estimated_cost_usd)
            span.set_attribute('gen_ai.usage.estimated_cost_krw', usage.estimated_cost_krw)
            span.set_attribute('gen_ai.usage.usd_krw_exchange_rate', usage.usd_krw_exchange_rate)
    except Exception:
        return
