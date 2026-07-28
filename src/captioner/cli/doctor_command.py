"""Doctor CLI adapter."""

import argparse
import json
from pathlib import Path
from typing import cast

from captioner.workflow.api import (
    ASR_PROFILES,
    AsrProfile,
    ExecutionContext,
    LogLevel,
    execute_doctor,
    resolve_options,
)


def add_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--config", type=Path)
    command.add_argument("--asr-profile", choices=ASR_PROFILES)
    command.add_argument(
        "--provider",
        choices=("fake", "faster-whisper", "qwen3-asr", "nemo-asr"),
    )
    command.add_argument("--load-model", action="store_true")
    command.add_argument("--output-dir", type=Path, default=Path.cwd())
    command.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "ALL", "OFF"),
    )


def run(arguments: argparse.Namespace, context: ExecutionContext) -> int:
    options = resolve_options(arguments.config)
    report = execute_doctor(
        options,
        provider=arguments.provider,
        profile=cast(AsrProfile | None, arguments.asr_profile),
        load_model=arguments.load_model,
        output_dir=arguments.output_dir,
        context=context,
        level_override=cast(LogLevel | None, arguments.log_level),
    )
    print(
        json.dumps(
            {
                "schema_version": "cli.v1",
                "command": "doctor",
                "ok": report.ok,
                "checks": report.checks,
                "details": report.details,
            },
            ensure_ascii=False,
        )
    )
    return 0 if report.ok else 1


__all__ = ["add_arguments", "run"]
