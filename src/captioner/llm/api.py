"""Public LLM contract used by the subtitles module."""

from typing import Protocol

from captioner.llm.batching import split_batches
from captioner.llm.client import ThreadLocalClient
from captioner.llm.concurrency import BatchResult, ParallelLlmExecutor
from captioner.llm.config import LlmOptions
from captioner.llm.models import (
    BoundarySelection,
    ContentContext,
    LlmGlossaryEntry,
    LlmStageContext,
    LlmTextBatch,
    LlmTextItem,
    LlmToken,
    TextUpdate,
    TextUpdateBatch,
)
from captioner.llm.openai_adapter import OpenAICompatibleLlm


class CloudLlm(Protocol):
    """Timing-free contract for subtitle LLM stages."""

    def analyze_context(self, text: str) -> ContentContext:
        """Analyze one bounded file transcript without modifying it."""
        ...

    def choose_boundaries(
        self,
        tokens: tuple[LlmToken, ...],
        *,
        context: LlmStageContext | None = None,
    ) -> BoundarySelection:
        """Choose boundary IDs without returning text or timestamps."""
        ...

    def correct(
        self,
        items: tuple[LlmTextItem, ...],
        *,
        context: LlmStageContext | None = None,
    ) -> TextUpdateBatch:
        """Return corrected text keyed by the input IDs."""
        ...

    def translate(
        self,
        items: tuple[LlmTextItem, ...],
        target_language: str,
        *,
        context: LlmStageContext | None = None,
    ) -> TextUpdateBatch:
        """Return translated text keyed by the input IDs."""
        ...

    def repair(
        self,
        items: tuple[LlmTextItem, ...],
        target_language: str,
        *,
        context: LlmStageContext | None = None,
    ) -> TextUpdateBatch:
        """Return repaired text keyed by the input IDs."""
        ...


__all__ = [
    "BoundarySelection",
    "BatchResult",
    "CloudLlm",
    "ContentContext",
    "LlmGlossaryEntry",
    "LlmOptions",
    "LlmStageContext",
    "LlmTextBatch",
    "LlmTextItem",
    "LlmToken",
    "TextUpdate",
    "TextUpdateBatch",
    "OpenAICompatibleLlm",
    "ParallelLlmExecutor",
    "split_batches",
    "ThreadLocalClient",
]
