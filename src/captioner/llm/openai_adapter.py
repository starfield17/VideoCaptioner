"""OpenAI-compatible Chat Completions adapter with strict response models."""

import os
import random
import time
from collections.abc import Callable, Mapping
from typing import Protocol, cast

from pydantic import BaseModel

from captioner.llm.client import ThreadLocalClient
from captioner.llm.config import LlmOptions
from captioner.llm.errors import LlmRetryableError, StructuredOutputError
from captioner.llm.models import (
    BoundarySelection,
    LlmTextItem,
    LlmToken,
    TextUpdateBatch,
)
from captioner.llm.prompts import (
    CORRECTION_SYSTEM,
    REPAIR_SYSTEM,
    SEGMENTATION_SYSTEM,
    TRANSLATION_SYSTEM,
)
from captioner.llm.retry import RetryPolicy
from captioner.llm.structured_output import parse_strict_json, response_format_for
from captioner.shared.errors import (
    LlmAuthenticationError,
    LlmPermanentError,
)


class _Completions(Protocol):
    def create(self, **kwargs: object) -> object:
        """Create one synchronous chat completion."""
        ...


class _Chat(Protocol):
    @property
    def completions(self) -> _Completions:
        """Return the chat completions resource."""
        ...


class _Client(Protocol):
    @property
    def chat(self) -> _Chat:
        """Return the chat resource."""
        ...


class OpenAICompatibleLlm:
    """Hide SDK, credentials, retries, and thread-local clients from subtitles."""

    def __init__(
        self,
        config: LlmOptions,
        client_factory: Callable[[], _Client] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        random_source: Callable[[], float] = random.random,
    ) -> None:
        self._config = config
        self._client_factory = client_factory or self._build_client
        self._clients = ThreadLocalClient(self._client_factory)
        self._retry = RetryPolicy(
            max_attempts=config.max_attempts,
            base_delay_seconds=config.backoff_base_seconds,
            max_delay_seconds=config.backoff_max_seconds,
            sleeper=sleeper,
            random_source=random_source,
        )

    def choose_boundaries(self, tokens: tuple[LlmToken, ...]) -> BoundarySelection:
        payload = {
            "tokens": [token.model_dump(mode="json") for token in tokens],
            "constraints": {
                "max_duration_ms": 7_000,
                "max_chars_cjk": 24,
                "max_words_latin": 14,
            },
        }
        return self._complete(
            SEGMENTATION_SYSTEM,
            payload,
            BoundarySelection,
        )

    def correct(self, items: tuple[LlmTextItem, ...]) -> TextUpdateBatch:
        return self._complete(
            CORRECTION_SYSTEM,
            {"items": [item.model_dump(mode="json") for item in items]},
            TextUpdateBatch,
        )

    def translate(
        self, items: tuple[LlmTextItem, ...], target_language: str
    ) -> TextUpdateBatch:
        return self._complete(
            TRANSLATION_SYSTEM,
            {
                "target_language": target_language,
                "items": [item.model_dump(mode="json") for item in items],
            },
            TextUpdateBatch,
        )

    def repair(
        self, items: tuple[LlmTextItem, ...], target_language: str
    ) -> TextUpdateBatch:
        return self._complete(
            REPAIR_SYSTEM,
            {
                "target_language": target_language,
                "items": [item.model_dump(mode="json") for item in items],
            },
            TextUpdateBatch,
        )

    def _complete[ModelT: BaseModel](
        self,
        system_prompt: str,
        payload: Mapping[str, object],
        response_type: type[ModelT],
    ) -> ModelT:
        messages = (
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _json_payload(payload)},
        )
        response = self._request(messages, response_type)
        raw_content = _response_content(response)
        try:
            return parse_strict_json(raw_content, response_type)
        except StructuredOutputError as first_error:
            feedback_messages = (
                *messages,
                {
                    "role": "user",
                    "content": (
                        "Return only valid JSON matching the requested schema. "
                        "Do not add markdown, commentary, timing, or extra fields."
                    ),
                },
            )
            feedback_response = self._request(feedback_messages, response_type)
            feedback_content = _response_content(feedback_response)
            try:
                return parse_strict_json(feedback_content, response_type)
            except StructuredOutputError as second_error:
                raise second_error from first_error

    def _request[ModelT: BaseModel](
        self,
        messages: tuple[dict[str, str], ...],
        response_type: type[ModelT],
    ) -> object:
        def operation() -> object:
            client = self._clients.get()
            return client.chat.completions.create(
                model=self._config.model,
                messages=list(messages),
                temperature=0,
                response_format=response_format_for(response_type),
            )

        try:
            return self._retry.run(operation, _is_retryable)
        except (LlmAuthenticationError, LlmPermanentError):
            raise
        except Exception as exc:
            if _is_retryable(exc):
                raise LlmRetryableError(
                    f"LLM request exhausted retries: {exc}"
                ) from exc
            raise LlmPermanentError(f"LLM request failed: {exc}") from exc

    def _build_client(self) -> _Client:
        api_key = os.getenv(self._config.api_key_env)
        if not api_key:
            raise LlmAuthenticationError(
                f"LLM API key environment variable is missing: "
                f"{self._config.api_key_env}"
            )
        from openai import OpenAI

        base_url = os.getenv(self._config.base_url_env)
        if base_url:
            return cast(
                _Client,
                OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=self._config.timeout_seconds,
                    max_retries=0,
                ),
            )
        return cast(
            _Client,
            OpenAI(
                api_key=api_key,
                timeout=self._config.timeout_seconds,
                max_retries=0,
            ),
        )


def _json_payload(payload: Mapping[str, object]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _response_content(response: object) -> str:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, (list, tuple)) or not choices:
        raise LlmPermanentError("LLM response contained no choices")
    choice_values = cast(list[object] | tuple[object, ...], choices)
    message = getattr(choice_values[0], "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content:
        raise LlmPermanentError("LLM response contained no text content")
    return content


def _is_retryable(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code == 429 or 500 <= status_code <= 599
    return isinstance(exc, (TimeoutError, ConnectionError, OSError))


__all__ = ["OpenAICompatibleLlm"]
