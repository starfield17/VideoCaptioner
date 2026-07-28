from dataclasses import dataclass
from typing import cast

import pytest

from captioner.llm.config import LlmOptions
from captioner.llm.errors import StructuredOutputError
from captioner.llm.models import LlmTextItem, LlmToken
from captioner.llm.openai_adapter import OpenAICompatibleLlm


@dataclass
class _Message:
    content: str | None


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list[_Choice]


class _RateLimitError(Exception):
    status_code = 429


class _ServerError(Exception):
    status_code = 503


class _FakeCompletions:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self._completions = completions

    @property
    def completions(self) -> _FakeCompletions:
        return self._completions


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self._chat = _FakeChat(completions)

    @property
    def chat(self) -> _FakeChat:
        return self._chat


def _config(**updates: object) -> LlmOptions:
    values: dict[str, object] = {
        "provider": "openai-compatible",
        "model": "test-model",
        "max_attempts": 3,
        "backoff_base_seconds": 0.2,
        "backoff_max_seconds": 1.0,
    }
    values.update(updates)
    return LlmOptions.model_validate(values)


def test_adapter_sends_strict_json_schema_and_parses_boundary_ids() -> None:
    completions = _FakeCompletions(
        [_Response([_Choice(_Message('{"break_after":["w1"]}'))])]
    )
    adapter = OpenAICompatibleLlm(
        _config(), client_factory=lambda: _FakeClient(completions)
    )

    result = adapter.choose_boundaries(
        (LlmToken(id="w1", text="hello", gap_after_ms=10),)
    )

    assert result.break_after == ("w1",)
    request = completions.calls[0]
    response_format = cast(dict[str, object], request["response_format"])
    json_schema = cast(dict[str, object], response_format["json_schema"])
    assert json_schema["strict"] is True
    assert "start_ms" not in str(json_schema)
    assert "end_ms" not in str(json_schema)


def test_adapter_supports_json_object_with_an_explicit_schema() -> None:
    completions = _FakeCompletions(
        [_Response([_Choice(_Message('{"break_after":["w1"]}'))])]
    )
    adapter = OpenAICompatibleLlm(
        _config(structured_output_mode="json_object"),
        client_factory=lambda: _FakeClient(completions),
    )

    result = adapter.choose_boundaries(
        (LlmToken(id="w1", text="hello", gap_after_ms=10),)
    )

    assert result.break_after == ("w1",)
    request = completions.calls[0]
    assert request["response_format"] == {"type": "json_object"}
    messages = cast(list[dict[str, str]], request["messages"])
    assert '"output_schema"' in messages[1]["content"]
    assert "start_ms" not in messages[1]["content"]
    assert "end_ms" not in messages[1]["content"]


def test_invalid_structured_output_is_rejected_without_repairing_json() -> None:
    completions = _FakeCompletions(
        [
            _Response([_Choice(_Message('{"break_after":["w1"],"text":"bad"}'))]),
            _Response([_Choice(_Message('{"break_after":["w1"],"text":"bad"}'))]),
        ]
    )
    adapter = OpenAICompatibleLlm(
        _config(), client_factory=lambda: _FakeClient(completions)
    )

    with pytest.raises(StructuredOutputError):
        adapter.choose_boundaries((LlmToken(id="w1", text="hello", gap_after_ms=10),))
    assert len(completions.calls) == 2


def test_empty_json_content_gets_one_format_feedback_retry() -> None:
    completions = _FakeCompletions(
        [
            _Response([_Choice(_Message(""))]),
            _Response([_Choice(_Message('{"break_after":["w1"]}'))]),
        ]
    )
    adapter = OpenAICompatibleLlm(
        _config(structured_output_mode="json_object"),
        client_factory=lambda: _FakeClient(completions),
    )

    result = adapter.choose_boundaries(
        (LlmToken(id="w1", text="hello", gap_after_ms=10),)
    )

    assert result.break_after == ("w1",)
    assert len(completions.calls) == 2


def test_semantic_id_mismatch_gets_bounded_feedback_retry() -> None:
    completions = _FakeCompletions(
        [
            _Response([_Choice(_Message('{"items":[{"id":"cue1","text":"A"}]}'))]),
            _Response(
                [
                    _Choice(
                        _Message(
                            '{"items":['
                            '{"id":"cue1","text":"A"},'
                            '{"id":"cue2","text":"B"}'
                            "]}"
                        )
                    )
                ]
            ),
        ]
    )
    adapter = OpenAICompatibleLlm(
        _config(structured_output_mode="json_object"),
        client_factory=lambda: _FakeClient(completions),
    )

    result = adapter.translate(
        (
            LlmTextItem(id="cue1", text="one"),
            LlmTextItem(id="cue2", text="two"),
        ),
        "zh-CN",
    )

    assert [item.id for item in result.items] == ["cue1", "cue2"]
    assert len(completions.calls) == 2
    messages = cast(list[dict[str, str]], completions.calls[1]["messages"])
    assert "contract_feedback" in messages[1]["content"]


def test_semantic_id_mismatch_stops_at_configured_limit() -> None:
    response = _Response([_Choice(_Message('{"items":[]}'))])
    completions = _FakeCompletions([response, response])
    adapter = OpenAICompatibleLlm(
        _config(max_attempts=2),
        client_factory=lambda: _FakeClient(completions),
    )

    with pytest.raises(StructuredOutputError, match="semantic retries"):
        adapter.correct((LlmTextItem(id="cue1", text="one"),))

    assert len(completions.calls) == 2


def test_rate_limit_retries_one_batch_in_place_with_bounded_backoff() -> None:
    completions = _FakeCompletions(
        [
            _RateLimitError(),
            _RateLimitError(),
            _Response([_Choice(_Message('{"break_after":["w1"]}'))]),
        ]
    )
    sleeps: list[float] = []
    adapter = OpenAICompatibleLlm(
        _config(),
        client_factory=lambda: _FakeClient(completions),
        sleeper=sleeps.append,
        random_source=lambda: 0.0,
    )

    result = adapter.choose_boundaries(
        (LlmToken(id="w1", text="hello", gap_after_ms=10),)
    )

    assert result.break_after == ("w1",)
    assert len(completions.calls) == 3
    assert sleeps == [0.1, 0.2]


def test_server_error_is_also_retryable() -> None:
    completions = _FakeCompletions(
        [
            _ServerError(),
            _Response([_Choice(_Message('{"break_after":["w1"]}'))]),
        ]
    )
    adapter = OpenAICompatibleLlm(
        _config(max_attempts=2),
        client_factory=lambda: _FakeClient(completions),
        sleeper=lambda _delay: None,
        random_source=lambda: 0.0,
    )

    result = adapter.choose_boundaries(
        (LlmToken(id="w1", text="hello", gap_after_ms=10),)
    )

    assert result.break_after == ("w1",)
    assert len(completions.calls) == 2
