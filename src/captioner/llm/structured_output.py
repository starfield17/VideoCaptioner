"""Strict JSON-schema response formatting and validation."""

import json
from typing import Literal

from pydantic import BaseModel, ValidationError

from captioner.llm.errors import StructuredOutputError


def response_format_for[ModelT: BaseModel](
    model_type: type[ModelT],
    mode: Literal["json_schema", "json_object"] = "json_schema",
) -> dict[str, object]:
    """Build the provider-native strict JSON Schema response format."""

    if mode == "json_object":
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model_type.__name__.lower(),
            "strict": True,
            "schema": model_type.model_json_schema(),
        },
    }


def parse_strict_json[ModelT: BaseModel](raw: str, model_type: type[ModelT]) -> ModelT:
    """Parse JSON and then validate it with the exact Pydantic model."""

    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise StructuredOutputError("LLM response was not valid JSON") from exc
    try:
        return model_type.model_validate(payload)
    except (ValidationError, TypeError, ValueError) as exc:
        raise StructuredOutputError(
            f"LLM response did not match {model_type.__name__}"
        ) from exc


__all__ = ["parse_strict_json", "response_format_for"]
