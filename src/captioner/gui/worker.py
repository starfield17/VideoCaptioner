"""Run one synchronous Workflow operation in a Qt worker thread."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PySide6.QtCore import QObject, Signal, Slot

from captioner.workflow.api import (
    ApplicationCommand,
    AsrProfile,
    CancellationToken,
    ExecutionContext,
    PipelineOptions,
    ProgressEvent,
    download_model,
    execute_doctor,
    execute_refine,
    execute_run,
    execute_transcribe,
    plan_operation,
)

JobKind = Literal["scan", "run", "transcribe", "refine", "doctor", "download"]


@dataclass(frozen=True)
class JobSpec:
    kind: JobKind
    options: PipelineOptions
    input_path: Path | None = None
    output_dir: Path | None = None
    source_language: str = "und"
    input_bilingual: bool = False
    provider: str | None = None
    profile: AsrProfile | None = None
    load_model: bool = False
    model_key: str | None = None
    scan_command: ApplicationCommand = "run"


class OperationWorker(QObject):
    progress = Signal(object)
    succeeded = Signal(object)
    failed = Signal(str, str)
    finished = Signal()

    def __init__(self, spec: JobSpec, cancellation: CancellationToken) -> None:
        super().__init__()
        self._spec = spec
        self._cancellation = cancellation

    @Slot()
    def run(self) -> None:
        context = ExecutionContext.create(
            cancellation=self._cancellation,
            observer=self._progress,
        )
        try:
            result = self._execute(context)
            self.succeeded.emit(result)
        except Exception as exc:
            self.failed.emit(type(exc).__name__, str(exc))
        finally:
            self.finished.emit()

    def _execute(self, context: ExecutionContext) -> object:
        spec = self._spec
        if spec.kind == "doctor":
            return execute_doctor(
                spec.options,
                provider=spec.provider,
                profile=spec.profile,
                load_model=spec.load_model,
                output_dir=spec.output_dir,
                context=context,
            )
        if spec.kind == "download":
            if spec.model_key is None:
                raise ValueError("download job requires model_key")
            return download_model(spec.options, spec.model_key, context=context)
        if spec.kind == "scan":
            if spec.input_path is None or spec.output_dir is None:
                raise ValueError("scan job requires input and output")
            context.checkpoint()
            return plan_operation(
                spec.scan_command,
                spec.input_path,
                spec.output_dir,
                spec.options,
                context,
            )
        if spec.input_path is None or spec.output_dir is None:
            raise ValueError(f"{spec.kind} job requires input and output")
        if spec.kind == "transcribe":
            return execute_transcribe(
                spec.input_path,
                spec.options,
                spec.output_dir,
                context,
            )
        if spec.kind == "refine":
            return execute_refine(
                spec.input_path,
                spec.options,
                spec.output_dir,
                context,
                source_language=spec.source_language,
                input_bilingual=spec.input_bilingual,
            )
        return execute_run(
            spec.input_path,
            spec.options,
            spec.output_dir,
            context,
        )

    def _progress(self, event: ProgressEvent) -> None:
        self.progress.emit(event)


__all__ = ["JobKind", "JobSpec", "OperationWorker"]
