"""High-level application operations shared by CLI and GUI adapters."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from captioner.models import (
    MODEL_CATALOG,
    ModelStore,
    download_with_dependencies,
)
from captioner.shared.app_paths import ApplicationPaths, application_paths
from captioner.shared.errors import ConfigurationError
from captioner.shared.logging import LogLevel, configure_logging, log_extra
from captioner.workflow.doctor import DoctorReport, run_doctor
from captioner.workflow.model_preparation import prepare_asr_model
from captioner.workflow.models import RunResult, TranscriptionRunResult
from captioner.workflow.options import (
    AsrProfile,
    PipelineOptions,
    load_options,
    with_asr_profile,
    with_keep_workdir,
)
from captioner.workflow.pipeline import (
    build_services,
    discover_inputs,
    run_files,
    transcribe_files,
)
from captioner.workflow.progress import (
    ExecutionContext,
    ProgressEvent,
    ProgressKind,
    ProgressStage,
    execution_context,
)
from captioner.workflow.refine import RefineResult, refine_srt

ApplicationCommand = Literal["run", "transcribe", "refine"]


@dataclass(frozen=True)
class ApplicationPlan:
    command: ApplicationCommand
    inputs: tuple[Path, ...]
    output_dir: Path
    provider: str
    output_formats: tuple[str, ...]


@dataclass(frozen=True)
class ModelStatus:
    key: str
    provider: str
    repository: str
    dependencies: tuple[str, ...]
    note: str
    path: Path | None

    @property
    def downloaded(self) -> bool:
        return self.path is not None


def resolve_options(
    config_path: Path | None = None,
    *,
    profile: AsrProfile | None = None,
    keep_workdir: bool | None = None,
) -> PipelineOptions:
    """Load options and apply the small adapter-level override set."""

    options = load_options(config_path)
    if profile is not None:
        options = with_asr_profile(options, profile)
    if keep_workdir is not None:
        options = with_keep_workdir(options, keep_workdir)
    return options


def plan_operation(
    command: ApplicationCommand,
    input_path: Path,
    output_dir: Path,
    options: PipelineOptions,
) -> ApplicationPlan:
    """Validate an operation without downloads, API calls, or filesystem writes."""

    if command == "refine":
        if not input_path.is_file() or input_path.suffix.lower() not in {
            ".srt",
            ".json",
        }:
            raise ConfigurationError("refine accepts an SRT or subtitle JSON file")
        inputs = (input_path,)
    else:
        inputs = discover_inputs(input_path, provider=options.asr.provider)
    return ApplicationPlan(
        command=command,
        inputs=inputs,
        output_dir=output_dir,
        provider=options.asr.provider,
        output_formats=tuple(item.value for item in options.output.formats),
    )


def execute_run(
    input_path: Path,
    options: PipelineOptions,
    output_dir: Path,
    context: ExecutionContext | None = None,
    *,
    level_override: LogLevel | None = None,
) -> RunResult:
    selected = execution_context(context)
    inputs = discover_inputs(input_path, provider=options.asr.provider)
    _configure_run_logging(options, level_override)
    prepared = _prepare_options(options, selected)
    selected.checkpoint()
    return run_files(
        inputs,
        prepared,
        build_services(prepared, selected),
        output_dir,
        selected,
    )


def execute_transcribe(
    input_path: Path,
    options: PipelineOptions,
    output_dir: Path,
    context: ExecutionContext | None = None,
    *,
    level_override: LogLevel | None = None,
) -> TranscriptionRunResult:
    selected = execution_context(context)
    inputs = discover_inputs(input_path, provider=options.asr.provider)
    _configure_run_logging(options, level_override)
    prepared = _prepare_options(options, selected)
    selected.checkpoint()
    return transcribe_files(
        inputs,
        prepared,
        build_services(prepared, selected),
        output_dir,
        selected,
    )


def execute_refine(
    input_path: Path,
    options: PipelineOptions,
    output_dir: Path,
    context: ExecutionContext | None = None,
    *,
    source_language: str = "und",
    input_bilingual: bool = False,
    level_override: LogLevel | None = None,
) -> RefineResult:
    selected = execution_context(context)
    _configure_run_logging(options, level_override)
    return refine_srt(
        input_path,
        options,
        output_dir,
        source_language=source_language,
        input_bilingual=input_bilingual,
        context=selected,
    )


def execute_doctor(
    options: PipelineOptions,
    *,
    provider: str | None = None,
    profile: AsrProfile | None = None,
    load_model: bool = False,
    output_dir: Path | None = None,
    context: ExecutionContext | None = None,
    level_override: LogLevel | None = None,
) -> DoctorReport:
    selected = execution_context(context)
    _configure_run_logging(options, level_override)
    checked_options = _doctor_options(options, provider, profile)
    if load_model and checked_options.asr.provider != "fake":
        checked_options = _prepare_options(checked_options, selected)
    selected.checkpoint()
    return run_doctor(
        checked_options,
        provider=checked_options.asr.provider,
        load_model=load_model,
        output_dir=output_dir,
    )


def get_application_paths() -> ApplicationPaths:
    return application_paths()


def list_models(options: PipelineOptions) -> tuple[ModelStatus, ...]:
    store = _model_store(options)
    return tuple(
        ModelStatus(
            key=item.key,
            provider=item.provider,
            repository=item.repository,
            dependencies=item.dependencies,
            note=item.note,
            path=store.path(item.key),
        )
        for item in MODEL_CATALOG
    )


def download_model(
    options: PipelineOptions,
    key: str,
    *,
    revision: str | None = None,
    context: ExecutionContext | None = None,
) -> dict[str, Path]:
    selected = execution_context(context)
    selected.checkpoint()
    selected.emit(
        ProgressEvent(
            ProgressKind.STAGE_STARTED,
            stage=ProgressStage.MODEL_DOWNLOAD,
            message=key,
        )
    )
    paths = download_with_dependencies(_model_store(options), key, revision)
    selected.checkpoint()
    selected.emit(
        ProgressEvent(
            ProgressKind.STAGE_COMPLETED,
            stage=ProgressStage.MODEL_DOWNLOAD,
            message=key,
        )
    )
    return paths


def _prepare_options(
    options: PipelineOptions,
    context: ExecutionContext,
) -> PipelineOptions:
    if options.asr.provider == "fake":
        return options
    context.checkpoint()
    context.emit(
        ProgressEvent(
            ProgressKind.STAGE_STARTED,
            stage=ProgressStage.MODEL_DOWNLOAD,
            message=options.asr.provider,
        )
    )
    prepared = prepare_asr_model(options)
    context.checkpoint()
    context.emit(
        ProgressEvent(
            ProgressKind.STAGE_COMPLETED,
            stage=ProgressStage.MODEL_DOWNLOAD,
            message=options.asr.provider,
        )
    )
    return prepared


def _model_store(options: PipelineOptions) -> ModelStore:
    return ModelStore(
        options.models.cache_dir,
        endpoint=options.models.endpoint,
        offline=options.models.offline,
    )


def _doctor_options(
    options: PipelineOptions,
    provider: str | None,
    profile: AsrProfile | None,
) -> PipelineOptions:
    selected = with_asr_profile(options, profile) if profile is not None else options
    if provider is None or provider == selected.asr.provider:
        return selected
    if profile is not None:
        raise ConfigurationError("--provider conflicts with --asr-profile")
    defaults: dict[str, AsrProfile] = {
        "fake": "fake",
        "faster-whisper": "faster-whisper-turbo",
        "qwen3-asr": "qwen3-1.7b",
        "nemo-asr": "nemo-parakeet-v3",
    }
    try:
        return with_asr_profile(options, defaults[provider])
    except KeyError as exc:
        raise ConfigurationError(f"unsupported provider: {provider}") from exc


def _configure_run_logging(
    options: PipelineOptions,
    level_override: LogLevel | None,
) -> None:
    secret = options.llm.api_key
    secrets = () if secret is None else (secret.get_secret_value(),)
    path = configure_logging(
        options.logging,
        secrets=secrets,
        level_override=level_override,
    )
    logging.getLogger("captioner").info(
        "run started",
        extra=log_extra(
            stage="startup",
            provider=options.asr.provider,
            log_path=str(path) if path else None,
        ),
    )


__all__ = [
    "ApplicationCommand",
    "ApplicationPlan",
    "ModelStatus",
    "download_model",
    "execute_doctor",
    "execute_refine",
    "execute_run",
    "execute_transcribe",
    "get_application_paths",
    "list_models",
    "plan_operation",
    "resolve_options",
]
