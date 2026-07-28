"""Thin CLI adapter for the synchronous existing-subtitle refine workflow."""

import argparse
import json
from pathlib import Path

from captioner.workflow.api import load_options, refine_srt


def add_arguments(command: argparse.ArgumentParser) -> None:
    """Register the refine command's explicit input and output arguments."""

    command.add_argument("input", type=Path)
    command.add_argument("--config", type=Path)
    command.add_argument("--output-dir", type=Path, default=Path.cwd())
    command.add_argument("--source-language", default="und")
    command.add_argument(
        "--bilingual-input",
        action="store_true",
        help="interpret the first two text lines of each cue as source/translation",
    )


def run(arguments: argparse.Namespace) -> int:
    """Execute refine and print a machine-readable result summary."""

    options = load_options(arguments.config)
    result = refine_srt(
        arguments.input,
        options,
        arguments.output_dir,
        source_language=arguments.source_language,
        input_bilingual=arguments.bilingual_input,
    )
    print(
        json.dumps(
            {
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
