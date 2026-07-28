"""Minimal synchronous CLI."""

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from captioner.cli import refine_command
from captioner.models import MODEL_CATALOG, ModelStore, download_with_dependencies
from captioner.shared.app_paths import application_paths
from captioner.shared.errors import (
    ConfigurationError,
    ExportError,
    LlmAuthenticationError,
    LlmPermanentError,
    LlmRetryableError,
    ProviderUnavailableError,
    StructuredOutputError,
    SubtitleValidationError,
)
from captioner.shared.logging import LogLevel, configure_logging, log_extra
from captioner.workflow.api import (
    AsrProfile,
    build_services,
    discover_inputs,
    load_options,
    prepare_asr_model,
    run_doctor,
    run_files,
    transcribe_files,
    with_asr_profile,
    with_keep_workdir,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the documented process exit code."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "config":
            return _config(arguments)
        if arguments.command == "models":
            return _models(arguments)
        if arguments.command == "doctor":
            options = _effective_options(arguments)
            _start_logging(options, arguments.log_level)
            return _doctor(
                options,
                arguments.provider,
                arguments.load_model,
                arguments.output_dir,
            )
        if arguments.command == "transcribe":
            return _transcribe(arguments)
        if arguments.command == "refine":
            return refine_command.run(arguments)
        return _run(arguments)
    except ConfigurationError as exc:
        print(f"configuration/input error: {exc}", file=sys.stderr)
        return 2
    except ProviderUnavailableError as exc:
        print(f"ASR provider unavailable: {exc}", file=sys.stderr)
        return 3
    except (
        LlmAuthenticationError,
        LlmPermanentError,
        LlmRetryableError,
        StructuredOutputError,
    ) as exc:
        print(f"LLM provider error: {exc}", file=sys.stderr)
        return 4
    except ExportError as exc:
        print(f"output error: {exc}", file=sys.stderr)
        return 5
    except SubtitleValidationError as exc:
        print(f"subtitle validation error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="captioner")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="check the selected environment")
    doctor.add_argument("--config", type=Path)
    _add_profile_and_logging_arguments(doctor)
    doctor.add_argument(
        "--provider",
        choices=("fake", "faster-whisper", "qwen3-asr", "nemo-asr"),
    )
    doctor.add_argument("--load-model", action="store_true")
    doctor.add_argument("--output-dir", type=Path, default=Path.cwd())

    run = commands.add_parser("run", help="run the selected provider pipeline")
    _add_pipeline_arguments(run)
    transcribe = commands.add_parser(
        "transcribe", help="write Transcript JSON from the selected provider"
    )
    _add_pipeline_arguments(transcribe)
    refine = commands.add_parser(
        "refine", help="correct and translate an existing SRT subtitle file"
    )
    refine_command.add_arguments(refine)

    config = commands.add_parser("config", help="inspect or initialize configuration")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    config_commands.add_parser("path", help="print platform-native application paths")
    show = config_commands.add_parser("show", help="print effective masked settings")
    show.add_argument("--config", type=Path)
    initialize = config_commands.add_parser("init", help="create the default config")
    initialize.add_argument("--path", type=Path)
    initialize.add_argument("--force", action="store_true")

    models = commands.add_parser("models", help="inspect or download ASR models")
    model_commands = models.add_subparsers(dest="models_command", required=True)
    model_commands.add_parser("path", help="print the application model directory")
    list_command = model_commands.add_parser("list", help="list curated models")
    list_command.add_argument("--provider")
    download = model_commands.add_parser("download", help="download a curated model")
    download.add_argument("model")
    download.add_argument("--revision")
    download.add_argument("--endpoint", default="https://huggingface.co")
    download.add_argument("--cache-dir", type=Path)
    return parser


def _add_pipeline_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("input", type=Path)
    command.add_argument("--config", type=Path)
    command.add_argument("--output-dir", type=Path, default=Path.cwd())
    command.add_argument("--keep-workdir", action="store_true", default=None)
    command.add_argument("--dry-run", action="store_true")
    _add_profile_and_logging_arguments(command)


def _add_profile_and_logging_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--asr-profile",
        choices=(
            "fake",
            "faster-whisper-turbo",
            "faster-whisper-small",
            "faster-whisper-large-v2",
            "faster-whisper-large-v3",
            "qwen3-0.6b",
            "qwen3-1.7b",
            "nemo-parakeet-v3",
            "nemo-parakeet-110m-en",
        ),
    )
    command.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "ALL", "OFF"),
    )


def _doctor(
    options: object,
    provider: str | None,
    load_model: bool,
    output_dir: Path,
) -> int:
    from captioner.workflow.options import PipelineOptions

    assert isinstance(options, PipelineOptions)
    if load_model and options.asr.provider != "fake":
        options = prepare_asr_model(options)
    report = run_doctor(
        options,
        provider=provider,
        load_model=load_model,
        output_dir=output_dir,
    )
    print(
        json.dumps(
            {"ok": report.ok, "checks": report.checks, "details": report.details}
        )
    )
    return 0 if report.ok else 1


def _transcribe(arguments: argparse.Namespace) -> int:
    options = _effective_options(arguments)
    _start_logging(options, arguments.log_level)
    if arguments.keep_workdir is not None:
        options = with_keep_workdir(options, arguments.keep_workdir)
    inputs = discover_inputs(arguments.input, provider=options.asr.provider)
    if arguments.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "command": "transcribe",
                    "provider": options.asr.provider,
                    "inputs": [str(path) for path in inputs],
                    "output_dir": str(arguments.output_dir),
                },
                ensure_ascii=False,
            )
        )
        return 0

    options = prepare_asr_model(options)
    result = transcribe_files(
        inputs,
        options,
        build_services(options),
        arguments.output_dir,
    )
    payload = {
        "succeeded": [
            {
                "input": str(item.input_path),
                "output": str(item.output_path),
                "warnings": list(item.warnings),
            }
            for item in result.succeeded
        ],
        "failed": [
            {
                "input": str(item.input_path),
                "error_type": item.error_type,
                "message": item.message,
            }
            for item in result.failed
        ],
        "workdir": str(result.workdir) if result.workdir else None,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if not result.failed else 1


def _run(arguments: argparse.Namespace) -> int:
    options = _effective_options(arguments)
    _start_logging(options, arguments.log_level)
    if arguments.keep_workdir is not None:
        options = with_keep_workdir(options, arguments.keep_workdir)
    inputs = discover_inputs(arguments.input, provider=options.asr.provider)
    if arguments.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "provider": options.asr.provider,
                    "inputs": [str(path) for path in inputs],
                    "output_dir": str(arguments.output_dir),
                },
                ensure_ascii=False,
            )
        )
        return 0

    options = prepare_asr_model(options)
    result = run_files(
        inputs,
        options,
        build_services(options),
        arguments.output_dir,
    )
    payload = {
        "succeeded": [
            {
                "input": str(item.input_path),
                "outputs": [str(path) for path in item.output_paths],
                "warnings": list(item.warnings),
            }
            for item in result.succeeded
        ],
        "failed": [
            {
                "input": str(item.input_path),
                "error_type": item.error_type,
                "message": item.message,
            }
            for item in result.failed
        ],
        "workdir": str(result.workdir) if result.workdir else None,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if not result.failed else 1


def _effective_options(arguments: argparse.Namespace):
    options = load_options(arguments.config)
    profile: AsrProfile | None = arguments.asr_profile
    if profile is not None:
        options = with_asr_profile(options, profile)
    return options


def _start_logging(options: object, level: LogLevel | None) -> None:
    from captioner.workflow.options import PipelineOptions

    assert isinstance(options, PipelineOptions)
    secret = options.llm.api_key
    secrets = () if secret is None else (secret.get_secret_value(),)
    path = configure_logging(options.logging, secrets=secrets, level_override=level)
    logging.getLogger("captioner").info(
        "run started",
        extra=log_extra(
            stage="startup",
            provider=options.asr.provider,
            log_path=str(path) if path else None,
        ),
    )


def _config(arguments: argparse.Namespace) -> int:
    paths = application_paths()
    if arguments.config_command == "path":
        print(
            json.dumps(
                {
                    "config": str(paths.config_file),
                    "models": str(paths.model_dir),
                    "logs": str(paths.log_dir),
                },
                ensure_ascii=False,
            )
        )
        return 0
    if arguments.config_command == "show":
        selected = arguments.config
        source = selected or (
            paths.config_file if paths.config_file.is_file() else None
        )
        options = load_options(selected)
        print(
            json.dumps(
                {
                    "source": str(source) if source else "built-in defaults",
                    "settings": options.model_dump(mode="json"),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    target = arguments.path or paths.config_file
    if target.exists() and not arguments.force:
        raise ConfigurationError(f"configuration already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_DEFAULT_CONFIG, encoding="utf-8")
    print(json.dumps({"created": str(target)}, ensure_ascii=False))
    return 0


def _models(arguments: argparse.Namespace) -> int:
    paths = application_paths()
    if arguments.models_command == "path":
        print(str(paths.model_dir))
        return 0
    if arguments.models_command == "list":
        store = ModelStore()
        payload: list[dict[str, object]] = []
        for model in MODEL_CATALOG:
            if arguments.provider and model.provider != arguments.provider:
                continue
            local = store.path(model.key)
            payload.append(
                {
                    "key": model.key,
                    "provider": model.provider,
                    "repository": model.repository,
                    "downloaded": local is not None,
                    "path": str(local) if local else None,
                    "dependencies": model.dependencies,
                    "note": model.note,
                }
            )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    store = ModelStore(arguments.cache_dir, endpoint=arguments.endpoint)
    downloaded = download_with_dependencies(store, arguments.model, arguments.revision)
    print(
        json.dumps(
            {"downloaded": {key: str(value) for key, value in downloaded.items()}},
            ensure_ascii=False,
        )
    )
    return 0


_DEFAULT_CONFIG = """# VideoCaptioner user configuration
[asr]
provider = "faster-whisper"
language = "auto"

[asr.faster_whisper]
model = "turbo"
device = "auto"
compute_type = "auto-int8"
beam_size = 5

[models]
endpoint = "https://huggingface.co"
offline = false

[logging]
level = "INFO"
console = true
file = true
max_bytes = 10485760
backup_count = 5

[segmentation]
batch_tokens = 800
overlap_tokens = 0
parallelism = 4

[correction]
batch_size = 30
parallelism = 8

[translation]
batch_size = 30
parallelism = 16

[repair]
batch_size = 20
parallelism = 8
"""


if __name__ == "__main__":
    raise SystemExit(main())
