"""Thin CLI adapter for the synchronous existing-subtitle refine workflow."""

import argparse
import json
from pathlib import Path
from typing import cast

from captioner.workflow.api import (
    ExecutionContext,
    LogLevel,
    execute_refine,
    plan_operation,
    resolve_options,
)


def add_arguments(command: argparse.ArgumentParser) -> None:
    """Register the refine command's explicit input and output arguments."""

    command.add_argument("input", type=Path)
    command.add_argument("--config", type=Path)
    command.add_argument("--output-dir", type=Path, default=Path.cwd())
    command.add_argument("--dry-run", action="store_true")
    command.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "ALL", "OFF"),
    )
    command.add_argument("--source-language", default="und")
    command.add_argument(
        "--bilingual-input",
        action="store_true",
        help="interpret the first two text lines of each cue as source/translation",
    )


def run(arguments: argparse.Namespace, context: ExecutionContext) -> int:
    """Execute refine and print a machine-readable result summary."""

    options = resolve_options(arguments.config)
    if arguments.dry_run:
        plan = plan_operation(
            "refine",
            arguments.input,
            arguments.output_dir,
            options,
        )
        print(
            json.dumps(
                {
                    "schema_version": "cli.v1",
                    "command": "refine",
                    "dry_run": True,
                    "inputs": [str(path) for path in plan.inputs],
                    "output_dir": str(plan.output_dir),
                    "output_formats": plan.output_formats,
                },
                ensure_ascii=False,
            )
        )
        return 0
    result = execute_refine(
        arguments.input,
        options,
        arguments.output_dir,
        context,
        source_language=arguments.source_language,
        input_bilingual=arguments.bilingual_input,
        level_override=cast(LogLevel | None, arguments.log_level),
    )
    print(
        json.dumps(
            {
                "schema_version": "cli.v1",
                "command": "refine",
                "input": str(result.input_path),
                "outputs": [str(path) for path in result.output_paths],
                "quality_issues": [
                    issue.model_dump(mode="json")
                    for issue in result.quality_report.issues
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


__all__ = ["add_arguments", "run"]
