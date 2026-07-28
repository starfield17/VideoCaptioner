"""Cross-platform per-user paths for configuration, models, and logs."""

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

    @classmethod
    def resolve(cls, *, dirs: PlatformDirs | None = None) -> "ApplicationPaths":
        selected = dirs or PlatformDirs(appname=APP_NAME, appauthor=False)
        return cls(
            config_file=Path(selected.user_config_dir) / "config.toml",
            model_dir=Path(selected.user_cache_dir) / "models",
            log_dir=Path(selected.user_log_dir),
        )


def application_paths() -> ApplicationPaths:
    """Return platform-native application paths."""

    return ApplicationPaths.resolve()


__all__ = ["APP_NAME", "ApplicationPaths", "application_paths"]
