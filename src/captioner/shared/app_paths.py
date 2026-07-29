"""Cross-platform per-user paths and bundled executable discovery."""

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from platformdirs import PlatformDirs

APP_NAME = "VideoCaptioner"


@dataclass(frozen=True)
class ApplicationPaths:
    """Resolved paths without creating directories as a read side effect."""

    config_file: Path
    model_dir: Path
    log_dir: Path
    runtime_dir: Path

    @classmethod
    def resolve(cls, *, dirs: PlatformDirs | None = None) -> "ApplicationPaths":
        selected = dirs or PlatformDirs(appname=APP_NAME, appauthor=False)
        return cls(
            config_file=Path(selected.user_config_dir) / "config.toml",
            model_dir=Path(selected.user_cache_dir) / "models",
            log_dir=Path(selected.user_log_dir),
            runtime_dir=Path(selected.user_data_dir) / "runtimes",
        )


def application_paths() -> ApplicationPaths:
    """Return platform-native application paths."""

    return ApplicationPaths.resolve()


def bundled_executable(name: str) -> str | None:
    """Find an executable shipped beside a frozen application, then use PATH."""

    executable_name = f"{name}.exe" if sys.platform == "win32" else name
    roots = (
        Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)),
        Path(sys.executable).parent,
    )
    for root in roots:
        for relative in (
            Path("runtime") / executable_name,
            Path("_internal") / "runtime" / executable_name,
        ):
            candidate = root / relative
            if candidate.is_file():
                return str(candidate)
    return shutil.which(name)


__all__ = [
    "APP_NAME",
    "ApplicationPaths",
    "application_paths",
    "bundled_executable",
]
