"""Public LLM contract used by the subtitles module."""

from typing import Protocol

from captioner.llm.batching import split_batches
from captioner.llm.client import ThreadLocalClient
from captioner.llm.concurrency import BatchResult, ParallelLlmExecutor
from captioner.llm.config import LlmOptions
from captioner.llm.models import (
    BoundarySelection,
    LlmTextItem,
    LlmToken,
    TextUpdate,
    TextUpdateBatch,
)
from captioner.llm.openai_adapter import OpenAICompatibleLlm


class CloudLlm(Protocol):
    """Timing-free contract for subtitle LLM stages."""

    def choose_boundaries(self, tokens: tuple[LlmToken, ...]) -> BoundarySelection:
        """Choose boundary IDs without returning text or timestamps."""
        ...

    def correct(self, items: tuple[LlmTextItem, ...]) -> TextUpdateBatch:
        """Return corrected text keyed by the input IDs."""
        ...

    def translate(
        self, items: tuple[LlmTextItem, ...], target_language: str
    ) -> TextUpdateBatch:
        """Return translated text keyed by the input IDs."""
        ...

    def repair(
        self, items: tuple[LlmTextItem, ...], target_language: str
    ) -> TextUpdateBatch:
        """Return repaired text keyed by the input IDs."""
        ...


__all__ = [
    "BoundarySelection",
    "BatchResult",
    "CloudLlm",
    "LlmOptions",
    "LlmTextItem",
    "LlmToken",
    "TextUpdate",
    "TextUpdateBatch",
    "OpenAICompatibleLlm",
    "ParallelLlmExecutor",
    "split_batches",
    "ThreadLocalClient",
]
