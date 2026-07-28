"""Errors local to the LLM adapter and batch boundary."""

from captioner.shared.errors import LlmRetryableError, StructuredOutputError

__all__ = ["LlmRetryableError", "StructuredOutputError"]
