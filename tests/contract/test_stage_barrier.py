import threading
import time
from pathlib import Path

from captioner.llm.api import (
    BoundarySelection,
    ContentContext,
    LlmStageContext,
    LlmTextItem,
    LlmToken,
    TextUpdate,
)
from captioner.llm.models import TextUpdateBatch
from captioner.media.api import FakeMediaService
from captioner.subtitles.api import SubtitleService
from captioner.transcription.api import FakeTranscriptionService
from captioner.workflow.api import PipelineOptions, PipelineServices, run_files

ROOT = Path(__file__).resolve().parents[2]


class _BarrierLlm:
    def __init__(self) -> None:
        self.events: list[str] = []
        self._lock = threading.Lock()

    def analyze_context(self, text: str) -> ContentContext:
        return ContentContext(summary=text)

    def choose_boundaries(
        self,
        tokens: tuple[LlmToken, ...],
        *,
        context: LlmStageContext | None = None,
    ) -> BoundarySelection:
        del context
        self._record("segment_start")
        time.sleep(0.002)
        self._record("segment_end")
        return BoundarySelection(break_after=(tokens[-1].id,))

    def correct(
        self,
        items: tuple[LlmTextItem, ...],
        *,
        context: LlmStageContext | None = None,
    ) -> TextUpdateBatch:
        del context
        self._record("correction_start")
        time.sleep(0.002)
        self._record("correction_end")
        return TextUpdateBatch(
            items=tuple(
                TextUpdate(id=item.id, text=f"corrected:{item.text}") for item in items
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
        self._record("translation_start")
        time.sleep(0.002)
        self._record("translation_end")
        return TextUpdateBatch(
            items=tuple(
                TextUpdate(id=item.id, text=f"{target_language}:{item.text}")
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
        self._record("repair_start")
        self._record("repair_end")
        return TextUpdateBatch(
            items=tuple(
                TextUpdate(id=item.id, text=f"{target_language}:repaired")
                for item in items
            )
        )

    def _record(self, event: str) -> None:
        with self._lock:
            self.events.append(event)


def test_pipeline_has_barriers_between_llm_stages(tmp_path: Path) -> None:
    options = PipelineOptions.model_validate(
        {
            "run": {"keep_workdir": True},
            "segmentation": {"batch_tokens": 2, "parallelism": 4},
            "correction": {
                "batch_size": 1,
                "parallelism": 4,
                "max_change_ratio": 1.0,
            },
            "translation": {"batch_size": 1, "parallelism": 4},
            "output": {"formats": ["json"]},
        }
    )
    llm = _BarrierLlm()
    result = run_files(
        (ROOT / "tests/fixtures/fake_input.json",),
        options,
        PipelineServices(
            media=FakeMediaService(),
            transcription=FakeTranscriptionService(),
            subtitles=SubtitleService(llm),
        ),
        tmp_path,
    )

    assert not result.failed
    correction_start = llm.events.index("correction_start")
    translation_start = llm.events.index("translation_start")
    assert (
        max(index for index, event in enumerate(llm.events) if event == "segment_end")
        < correction_start
    )
    assert (
        max(
            index for index, event in enumerate(llm.events) if event == "correction_end"
        )
        < translation_start
    )
