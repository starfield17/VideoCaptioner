"""Plain stderr progress rendering for synchronous CLI commands."""

import sys

from captioner.workflow.api import ProgressEvent, ProgressKind


def render_progress(event: ProgressEvent) -> None:
    """Render stable human diagnostics without polluting stdout."""

    if event.kind not in {
        ProgressKind.STAGE_STARTED,
        ProgressKind.STAGE_COMPLETED,
        ProgressKind.FILE_FAILED,
        ProgressKind.CANCELLED,
    }:
        return
    position = ""
    if event.file_index is not None and event.file_count is not None:
        position = f"[{event.file_index}/{event.file_count}]"
    stage = f"[{event.stage.value}]" if event.stage is not None else ""
    message = event.message or event.kind.value
    print(f"{position}{stage} {message}".strip(), file=sys.stderr)


__all__ = ["render_progress"]
