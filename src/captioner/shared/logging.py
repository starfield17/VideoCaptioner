"""Structured application logging with cross-platform rotation and redaction."""

import json
import logging
import sys
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from captioner.shared.app_paths import application_paths

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "ALL", "OFF"]
_STANDARD_FIELDS = frozenset(logging.makeLogRecord({}).__dict__)
_run_id = "-"


class LoggingOptions(BaseModel):
    """User-facing logging policy."""

    model_config = ConfigDict(extra="forbid")

    level: LogLevel = "INFO"
    console: bool = True
    file: bool = True
    max_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    backup_count: int = Field(default=5, ge=0, le=100)
    directory: Path | None = None


class _RedactionFilter(logging.Filter):
    def __init__(self, secrets: tuple[str, ...]) -> None:
        super().__init__()
        self._secrets = tuple(value for value in secrets if value)

    def filter(self, record: logging.LogRecord) -> bool:
        rendered = record.getMessage()
        for secret in self._secrets:
            rendered = rendered.replace(secret, "**********")
        record.msg = rendered
        record.args = ()
        return True


class _JsonFormatter(logging.Formatter):
    def __init__(self, secrets: tuple[str, ...]) -> None:
        super().__init__()
        self._secrets = secrets

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "run_id": getattr(record, "run_id", _run_id),
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_FIELDS and key not in payload and key != "message":
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return _redact(
            json.dumps(payload, ensure_ascii=False, default=str),
            self._secrets,
        )


class _ConsoleFormatter(logging.Formatter):
    def __init__(self, secrets: tuple[str, ...]) -> None:
        super().__init__()
        self._secrets = secrets

    def format(self, record: logging.LogRecord) -> str:
        stage = getattr(record, "stage", None)
        prefix = f"[{record.levelname.lower()}]"
        if stage:
            prefix += f"[{stage}]"
        return _redact(f"{prefix} {record.getMessage()}", self._secrets)


def configure_logging(
    options: LoggingOptions,
    *,
    secrets: tuple[str, ...] = (),
    level_override: LogLevel | None = None,
) -> Path | None:
    """Configure application handlers and return the rotating log path."""

    global _run_id
    _run_id = uuid4().hex
    selected = level_override or options.level
    level = _logging_level(selected)
    logger = logging.getLogger("captioner")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(level if level is not None else logging.CRITICAL + 1)
    redaction = _RedactionFilter(secrets)
    if level is None:
        return None
    if options.console:
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(level)
        console.setFormatter(_ConsoleFormatter(secrets))
        console.addFilter(redaction)
        logger.addHandler(console)
    log_path: Path | None = None
    if options.file:
        directory = options.directory or application_paths().log_dir
        directory.mkdir(parents=True, exist_ok=True)
        log_path = directory / "captioner.log"
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=options.max_bytes,
            backupCount=options.backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(_JsonFormatter(secrets))
        file_handler.addFilter(redaction)
        logger.addHandler(file_handler)
    return log_path


def log_extra(**values: object) -> dict[str, object]:
    """Attach the current run identifier to a structured event."""

    return {"run_id": _run_id, **values}


def _logging_level(level: LogLevel) -> int | None:
    if level == "OFF":
        return None
    if level == "ALL":
        return logging.DEBUG
    return getattr(logging, level)


def _redact(value: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        if secret:
            value = value.replace(secret, "**********")
    return value


__all__ = ["LoggingOptions", "LogLevel", "configure_logging", "log_extra"]
