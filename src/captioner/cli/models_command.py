"""Model catalog CLI adapter."""

import argparse
import json
from pathlib import Path

from captioner.workflow.api import (
    ExecutionContext,
    download_model,
    get_application_paths,
    list_models,
    resolve_options,
)


def add_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--config", type=Path)
    commands = command.add_subparsers(dest="models_command", required=True)
    commands.add_parser("path", help="print the configured model directory")
    list_command = commands.add_parser("list", help="list curated models")
    list_command.add_argument("--provider")
    download = commands.add_parser("download", help="download a curated model")
    download.add_argument("model")
    download.add_argument("--revision")
    download.add_argument("--endpoint")
    download.add_argument("--cache-dir", type=Path)


def run(arguments: argparse.Namespace, context: ExecutionContext) -> int:
    options = resolve_options(arguments.config)
    updates: dict[str, object] = {}
    if getattr(arguments, "endpoint", None):
        updates["endpoint"] = arguments.endpoint
    if getattr(arguments, "cache_dir", None):
        updates["cache_dir"] = arguments.cache_dir
    if updates:
        options = options.model_copy(
            update={"models": options.models.model_copy(update=updates)}
        )
    if arguments.models_command == "path":
        path = options.models.cache_dir or get_application_paths().model_dir
        _print({"path": str(path)})
        return 0
    if arguments.models_command == "list":
        values = [
            {
                "key": item.key,
                "provider": item.provider,
                "repository": item.repository,
                "downloaded": item.downloaded,
                "path": str(item.path) if item.path else None,
                "dependencies": item.dependencies,
                "note": item.note,
            }
            for item in list_models(options)
            if not arguments.provider or item.provider == arguments.provider
        ]
        _print({"models": values})
        return 0
    paths = download_model(
        options,
        arguments.model,
        revision=arguments.revision,
        context=context,
    )
    _print({"downloaded": {key: str(path) for key, path in paths.items()}})
    return 0


def _print(values: dict[str, object]) -> None:
    print(
        json.dumps(
            {"schema_version": "cli.v1", "command": "models", **values},
            indent=2,
            ensure_ascii=False,
        )
    )


__all__ = ["add_arguments", "run"]
