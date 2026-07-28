"""Run and transcribe CLI adapters."""

import argparse
import json
from pathlib import Path
from typing import cast

from captioner.workflow.api import (
    ASR_PROFILES,
    ApplicationCommand,
    AsrProfile,
    ExecutionContext,
    LogLevel,
    execute_run,
    execute_transcribe,
    plan_operation,
    resolve_options,
)


def add_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("input", type=Path)
    command.add_argument("--config", type=Path)
    command.add_argument("--output-dir", type=Path, default=Path.cwd())
    command.add_argument("--keep-workdir", action="store_true", default=None)
    command.add_argument("--dry-run", action="store_true")
    command.add_argument("--asr-profile", choices=ASR_PROFILES)
    command.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "ALL", "OFF"),
    )


def run(
    arguments: argparse.Namespace,
    command: str,
    context: ExecutionContext,
) -> int:
    profile = cast(AsrProfile | None, arguments.asr_profile)
    level = cast(LogLevel | None, arguments.log_level)
    options = resolve_options(
        arguments.config,
        profile=profile,
        keep_workdir=arguments.keep_workdir,
    )
    if arguments.dry_run:
        plan = plan_operation(
            cast(ApplicationCommand, command),
            arguments.input,
            arguments.output_dir,
            options,
        )
        _print(
            command,
            {
                "dry_run": True,
                "provider": plan.provider,
                "inputs": [str(path) for path in plan.inputs],
                "input_root": (
                    str(plan.input_root) if plan.input_root is not None else None
                ),
                "output_dir": str(plan.output_dir),
                "output_formats": plan.output_formats,
            },
        )
        return 0
    if command == "transcribe":
        result = execute_transcribe(
            arguments.input,
            options,
            arguments.output_dir,
            context,
            level_override=level,
        )
        _print(
            command,
            {
                "succeeded": [
                    {
                        "input": str(item.input_path),
                        "output": str(item.output_path),
                        "warnings": item.warnings,
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
            },
        )
        return 0 if not result.failed else 1
    result = execute_run(
        arguments.input,
        options,
        arguments.output_dir,
        context,
        level_override=level,
    )
    _print(
        command,
        {
            "succeeded": [
                {
                    "input": str(item.input_path),
                    "outputs": [str(path) for path in item.output_paths],
                    "warnings": item.warnings,
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
        },
    )
    return 0 if not result.failed else 1


def _print(command: str, values: dict[str, object]) -> None:
    print(
        json.dumps(
            {"schema_version": "cli.v1", "command": command, **values},
            ensure_ascii=False,
        )
    )


__all__ = ["add_arguments", "run"]
