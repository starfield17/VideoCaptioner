"""Thin synchronous command-line adapter."""

import argparse
import signal
import sys
from collections.abc import Callable, Sequence
from importlib.metadata import version
from types import FrameType

from captioner.cli import (
    config_command,
    doctor_command,
    models_command,
    pipeline_command,
    refine_command,
    runtimes_command,
)
from captioner.cli.progress import render_progress
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
    CancellationToken,
    ExecutionContext,
    OperationCancelled,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the documented process exit code."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    cancellation = CancellationToken()
    context = ExecutionContext.create(
        cancellation=cancellation,
        observer=render_progress,
    )
    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _interrupt_handler(cancellation))
    try:
        return _dispatch(arguments, context)
    except OperationCancelled as exc:
        print(str(exc), file=sys.stderr)
        return 130
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
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def _dispatch(arguments: argparse.Namespace, context: ExecutionContext) -> int:
    if arguments.command == "config":
        return config_command.run(arguments)
    if arguments.command == "models":
        return models_command.run(arguments, context)
    if arguments.command == "runtimes":
        return runtimes_command.run(arguments, context)
    if arguments.command == "doctor":
        return doctor_command.run(arguments, context)
    if arguments.command == "transcribe":
        return pipeline_command.run(arguments, "transcribe", context)
    if arguments.command == "refine":
        return refine_command.run(arguments, context)
    return pipeline_command.run(arguments, "run", context)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="captioner")
    parser.add_argument("--version", action="version", version=version("captioner"))
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="check the selected environment")
    doctor_command.add_arguments(doctor)
    run = commands.add_parser("run", help="run the selected provider pipeline")
    pipeline_command.add_arguments(run)
    transcribe = commands.add_parser(
        "transcribe", help="write Transcript JSON from the selected provider"
    )
    pipeline_command.add_arguments(transcribe)
    refine = commands.add_parser(
        "refine", help="correct and translate an existing subtitle file"
    )
    refine_command.add_arguments(refine)
    config = commands.add_parser("config", help="inspect or initialize configuration")
    config_command.add_arguments(config)
    models = commands.add_parser("models", help="inspect or download ASR models")
    models_command.add_arguments(models)
    runtimes = commands.add_parser("runtimes", help="inspect or manage ASR runtimes")
    runtimes_command.add_arguments(runtimes)
    return parser


def _interrupt_handler(
    cancellation: CancellationToken,
) -> Callable[[int, FrameType | None], None]:
    def handle(_signal: int, _frame: FrameType | None) -> None:
        if cancellation.cancelled:
            raise KeyboardInterrupt
        cancellation.cancel()
        print(
            "cancellation requested; press Ctrl+C again to interrupt", file=sys.stderr
        )

    return handle


if __name__ == "__main__":
    raise SystemExit(main())
