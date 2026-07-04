"""JSON-schema sanitizer for structured outputs.

The API's structured-output grammar supports a subset of JSON Schema — no
numerical bounds, no string length constraints, and every object must set
additionalProperties: false. We send a sanitized (whitelisted) schema on the
wire and enforce the full Pydantic constraints client-side, retrying with
error feedback when validation fails.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

_ALLOWED_KEYS = frozenset(
    {
        "type",
        "properties",
        "required",
        "items",
        "enum",
        "const",
        "anyOf",
        "allOf",
        "$ref",
        "$defs",
        "description",
        "title",
        "additionalProperties",
    }
)


# Keys whose VALUES are name->schema maps: the names themselves are arbitrary
# (property names, model names) and must never be filtered by the whitelist.
_MAP_KEYS = frozenset({"properties", "$defs"})


class UnsupportedSchemaError(TypeError):
    """A draft schema uses a construct the structured-output grammar rejects."""


def _sanitize(node: Any) -> Any:
    if isinstance(node, dict):
        cleaned: dict[str, Any] = {}
        for key, value in node.items():
            if key in _MAP_KEYS and isinstance(value, dict):
                cleaned[key] = {name: _sanitize(sub) for name, sub in value.items()}
            elif key in _ALLOWED_KEYS:
                cleaned[key] = _sanitize(value)
        if cleaned.get("type") == "object":
            if "properties" not in cleaned:
                # The grammar requires additionalProperties:false on EVERY
                # object, so an open map (dict[str, Any]) can only ever emit
                # an empty object. Fail at build time, not at the API.
                raise UnsupportedSchemaError(
                    "draft schemas may not use open dict fields (object without "
                    "'properties'); use typed optional fields or a JSON-encoded "
                    "string field instead"
                )
            cleaned["additionalProperties"] = False
            # Strict structured-output modes (Anthropic AND OpenAI/Azure) both
            # require EVERY property key to appear in "required", regardless
            # of whether the Pydantic field has a default — optionality is
            # expressed by the property's own type union, not by omission
            # from "required". Pydantic's model_json_schema() only lists
            # fields with no default, so we widen it here rather than at the
            # Pydantic model (which would break Python-side defaults/validation).
            cleaned["required"] = list(cleaned["properties"])
        return cleaned
    if isinstance(node, list):
        return [_sanitize(item) for item in node]
    return node


def output_schema_for(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Wire-safe JSON schema for a Pydantic model."""
    return _sanitize(model_cls.model_json_schema())
