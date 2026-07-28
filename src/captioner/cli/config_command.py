"""Configuration CLI adapter."""

import argparse
import json
from pathlib import Path

from captioner.workflow.api import (
    PipelineOptions,
    get_application_paths,
    load_options,
    save_options,
)


def add_arguments(command: argparse.ArgumentParser) -> None:
    commands = command.add_subparsers(dest="config_command", required=True)
    commands.add_parser("path", help="print platform-native application paths")
    show = commands.add_parser("show", help="print effective masked settings")
    show.add_argument("--config", type=Path)
    initialize = commands.add_parser("init", help="create the default config")
    initialize.add_argument("--path", type=Path)
    initialize.add_argument("--force", action="store_true")


def run(arguments: argparse.Namespace) -> int:
    paths = get_application_paths()
    if arguments.config_command == "path":
        _print(
            {
                "config": str(paths.config_file),
                "models": str(paths.model_dir),
                "logs": str(paths.log_dir),
            }
        )
        return 0
    if arguments.config_command == "show":
        selected = arguments.config
        source = selected or (
            paths.config_file if paths.config_file.is_file() else None
        )
        options = load_options(selected)
        _print(
            {
                "source": str(source) if source else "built-in defaults",
                "settings": options.model_dump(mode="json"),
            }
        )
        return 0
    created = save_options(
        PipelineOptions(),
        arguments.path,
        overwrite=arguments.force,
    )
    _print({"created": str(created)})
    return 0


def _print(values: dict[str, object]) -> None:
    print(
        json.dumps(
            {"schema_version": "cli.v1", "command": "config", **values},
            indent=2,
            ensure_ascii=False,
        )
    )


__all__ = ["add_arguments", "run"]
