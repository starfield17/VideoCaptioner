"""Minimal synchronous CLI."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from captioner.cli import refine_command
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
from captioner.workflow.api import (
    build_services,
    discover_inputs,
    load_options,
    run_doctor,
    run_files,
    transcribe_files,
    with_keep_workdir,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the documented process exit code."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "doctor":
            return _doctor(
                arguments.config,
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
    return parser


def _add_pipeline_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("input", type=Path)
    command.add_argument("--config", type=Path)
    command.add_argument("--output-dir", type=Path, default=Path.cwd())
    command.add_argument("--keep-workdir", action="store_true", default=None)
    command.add_argument("--dry-run", action="store_true")


def _doctor(
    config_path: Path | None,
    provider: str | None,
    load_model: bool,
    output_dir: Path,
) -> int:
    options = load_options(config_path)
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
    options = load_options(arguments.config)
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
    options = load_options(arguments.config)
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


if __name__ == "__main__":
    raise SystemExit(main())
