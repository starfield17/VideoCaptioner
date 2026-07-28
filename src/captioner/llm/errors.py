"""Errors local to the LLM adapter and batch boundary."""

from captioner.shared.errors import CaptionerError


class LlmRetryableError(CaptionerError):
    """A bounded retry policy exhausted a transient LLM request."""


class StructuredOutputError(CaptionerError):
    """The provider response was not valid for the requested strict schema."""


__all__ = ["LlmRetryableError", "StructuredOutputError"]
