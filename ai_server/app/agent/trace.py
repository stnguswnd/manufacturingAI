from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.schemas import AgentTraceStep


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def trace_step(
    *,
    node_id: str,
    node_name: str,
    node_type: str,
    layer: str,
    status: str,
    input_summary: str = '',
    output_summary: str = '',
    error: str | None = None,
    replan_reason: str | None = None,
) -> dict[str, Any]:
    now = utc_iso()
    return {
        'node_id': node_id,
        'node_name': node_name,
        'node_type': node_type,
        'layer': layer,
        'status': status,
        'started_at': now,
        'ended_at': now,
        'latency_ms': 0,
        'input_summary': input_summary,
        'output_summary': output_summary,
        'error': error,
        'replan_reason': replan_reason,
    }


def append_trace(state: dict[str, Any], step: dict[str, Any]) -> None:
    state.setdefault('trace', []).append(step)


def to_agent_trace_steps(trace: list[dict[str, Any]]) -> list[AgentTraceStep]:
    steps: list[AgentTraceStep] = []
    for item in trace:
        node_id = item.get('node_id') or item.get('node_name') or 'node'
        detail_parts = [
            f'status={item.get("status")}',
            item.get('output_summary') or item.get('input_summary') or '',
        ]
        if item.get('error'):
            detail_parts.append(f'error={item.get("error")}')
        steps.append(AgentTraceStep(step=str(node_id), detail=' | '.join(part for part in detail_parts if part)))
    return steps

