"""Managed ASR runtime CLI adapter."""

import argparse
import json

from captioner.workflow.api import (
    ExecutionContext,
    RuntimeStatus,
    get_application_paths,
    install_runtime,
    list_runtimes,
    remove_runtime,
)

_PROVIDERS = ("faster-whisper", "qwen3-asr", "nemo-asr")


def add_arguments(command: argparse.ArgumentParser) -> None:
    commands = command.add_subparsers(dest="runtimes_command", required=True)
    commands.add_parser("path", help="print the managed runtime directory")
    commands.add_parser("list", help="list provider runtime status")
    install = commands.add_parser("install", help="install a provider runtime")
    install.add_argument("provider", choices=_PROVIDERS)
    repair = commands.add_parser("repair", help="replace a provider runtime")
    repair.add_argument("provider", choices=_PROVIDERS)
    remove = commands.add_parser("remove", help="remove a provider runtime")
    remove.add_argument("provider", choices=_PROVIDERS)


def run(arguments: argparse.Namespace, context: ExecutionContext) -> int:
    if arguments.runtimes_command == "path":
        _print({"path": str(get_application_paths().runtime_dir)})
        return 0
    if arguments.runtimes_command == "list":
        _print({"runtimes": [_status(item) for item in list_runtimes()]})
        return 0
    if arguments.runtimes_command == "remove":
        result = remove_runtime(arguments.provider)
    else:
        result = install_runtime(
            arguments.provider,
            repair=arguments.runtimes_command == "repair",
            context=context,
        )
    _print({"runtime": _status(result)})
    return 0


def _status(status: RuntimeStatus) -> dict[str, object]:
    descriptor = status.descriptor
    return {
        "provider": descriptor.provider,
        "platform": f"{descriptor.os}-{descriptor.host_arch}",
        "process_arch": descriptor.process_arch,
        "accelerator": descriptor.accelerator,
        "stability": descriptor.stability.value,
        "recipe_sha256": descriptor.recipe_sha256,
        "available": descriptor.available,
        "installed": status.installed,
        "path": str(status.path) if status.path else None,
        "detail": status.detail,
    }


def _print(values: dict[str, object]) -> None:
    print(
        json.dumps(
            {"schema_version": "cli.v1", "command": "runtimes", **values},
            indent=2,
            ensure_ascii=False,
        )
    )


__all__ = ["add_arguments", "run"]
