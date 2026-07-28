"""Deterministic local LLM test double."""

from typing import Literal

from captioner.llm.models import (
    BoundarySelection,
    ContentContext,
    LlmStageContext,
    LlmTextItem,
    LlmToken,
    TextUpdate,
    TextUpdateBatch,
)
from captioner.shared.errors import LlmPermanentError

FakeStage = Literal["context", "segmentation", "correction", "translation", "repair"]


class FakeLlm:
    """Perform predictable text-only transformations without network access."""

    def __init__(self, fail_stage: FakeStage | None = None) -> None:
        self._fail_stage = fail_stage

    def analyze_context(self, text: str) -> ContentContext:
        self._raise_if_configured("context")
        summary = text.strip()[:120]
        return ContentContext(summary=summary, domain="test", tone="neutral")

    def choose_boundaries(
        self,
        tokens: tuple[LlmToken, ...],
        *,
        context: LlmStageContext | None = None,
    ) -> BoundarySelection:
        del context
        self._raise_if_configured("segmentation")
        if not tokens:
            return BoundarySelection(break_after=())
        boundary_ids = [
            token.id
            for token in tokens
            if token.text.rstrip().endswith(("。", "！", "？", ".", "!", "?"))
        ]
        if tokens[-1].id not in boundary_ids:
            boundary_ids.append(tokens[-1].id)
        return BoundarySelection(break_after=tuple(boundary_ids))

    def correct(
        self,
        items: tuple[LlmTextItem, ...],
        *,
        context: LlmStageContext | None = None,
    ) -> TextUpdateBatch:
        del context
        self._raise_if_configured("correction")
        return TextUpdateBatch(
            items=tuple(
                TextUpdate(id=item.id, text=self._correct_text(item.text))
                for item in items
            )
        )

    def translate(
        self,
        items: tuple[LlmTextItem, ...],
        target_language: str,
        *,
        context: LlmStageContext | None = None,
    ) -> TextUpdateBatch:
        del context
        self._raise_if_configured("translation")
        return TextUpdateBatch(
            items=tuple(
                TextUpdate(
                    id=item.id,
                    text=f"[{target_language}] {item.text.strip()}",
                )
                for item in items
            )
        )

    def repair(
        self,
        items: tuple[LlmTextItem, ...],
        target_language: str,
        *,
        context: LlmStageContext | None = None,
    ) -> TextUpdateBatch:
        del context
        self._raise_if_configured("repair")
        return TextUpdateBatch(
            items=tuple(
                TextUpdate(
                    id=item.id,
                    text=f"[{target_language}] {item.text.strip()}",
                )
                for item in items
            )
        )

    def _raise_if_configured(self, stage: FakeStage) -> None:
        if self._fail_stage == stage:
            raise LlmPermanentError(f"Fake LLM failure in {stage}")

    @staticmethod
    def _correct_text(text: str) -> str:
        return text.strip().replace("错误", "正确")


__all__ = ["FakeLlm", "FakeStage"]
