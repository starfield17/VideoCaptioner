"""Protocol constants available to both the core client and workers."""

from typing import Final

PROTOCOL_VERSION: Final = "asr-worker.v1"
SUPPORTED_COMMANDS: Final[tuple[str, ...]] = (
    "hello",
    "load",
    "transcribe",
    "shutdown",
)
