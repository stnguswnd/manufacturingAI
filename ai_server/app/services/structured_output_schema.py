from __future__ import annotations

from typing import Any


def to_openai_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a Pydantic JSON schema into OpenAI strict structured-output form."""
    return _strict(schema)


def _strict(value: Any) -> Any:
    if isinstance(value, list):
        return [_strict(item) for item in value]
    if not isinstance(value, dict):
        return value

    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        if key == 'default':
            continue
        cleaned[key] = _strict(item)

    if cleaned.get('type') == 'object' or 'properties' in cleaned:
        cleaned['additionalProperties'] = False
        properties = cleaned.get('properties')
        if isinstance(properties, dict):
            cleaned['required'] = list(properties.keys())
    return cleaned
