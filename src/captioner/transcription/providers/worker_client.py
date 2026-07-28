"""Blocking NDJSON Worker clients shared by ASR providers."""

import json
import os
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import cast

from captioner.shared.errors import (
    CaptionerError,
    ProviderUnavailableError,
    TranscriptionError,
)
from captioner.shared.worker_protocol import PROTOCOL_VERSION
from captioner.transcription.capabilities import AsrCapabilities
from captioner.transcription.models import TranscriptDocument
from captioner.transcription.requests import TranscriptionRequest


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TranscriptionError(f"{name} must be a JSON object")
    return cast(dict[str, object], value)


def _repository_root() -> Path:
    source_file = Path(__file__).resolve()
    for candidate in (source_file, *source_file.parents):
        if (candidate / "src").is_dir() and (candidate / "workers").is_dir():
            return candidate
    return Path.cwd()


class NdjsonWorkerClient:
    """Run one blocking Worker process and exchange one JSON line per request."""

    def __init__(self, command: tuple[str, ...], expected_provider_id: str) -> None:
        self._command = command
        self._expected_provider_id = expected_provider_id
        self._process: subprocess.Popen[str] | None = None
        self.capabilities: AsrCapabilities | None = None

    def probe(self) -> AsrCapabilities:
        """Start the Worker, perform hello, and shut it down without loading a model."""

        if self._process is not None:
            raise ProviderUnavailableError("ASR Worker client is already started")
        self._launch()
        try:
            return self._hello()
        finally:
            self.shutdown()

    def start(self, load_payload: dict[str, object]) -> AsrCapabilities:
        """Perform hello and exactly one load for this Worker session."""

        if self._process is not None:
            raise ProviderUnavailableError("ASR Worker client is already started")
        self._launch()
        try:
            capabilities = self._hello()
            self._send("load", load_payload)
            self.capabilities = capabilities
            return capabilities
        except (CaptionerError, ValueError, TypeError) as exc:
            self.shutdown()
            if isinstance(exc, ProviderUnavailableError):
                raise
            raise ProviderUnavailableError(f"ASR Worker startup failed: {exc}") from exc

    def transcribe(
        self, request: TranscriptionRequest, artifact_dir: Path
    ) -> TranscriptDocument:
        result = self._send(
            "transcribe",
            {
                "audio_path": str(request.audio_path),
                "language": request.language,
                "initial_prompt": request.initial_prompt,
                "timestamps": request.timestamps.value,
                "artifact_dir": str(artifact_dir),
            },
        )
        artifact_value = result.get("artifact_path")
        if not isinstance(artifact_value, str):
            raise TranscriptionError("ASR Worker response omitted artifact_path")
        artifact_path = Path(artifact_value)
        try:
            raw_document = json.loads(artifact_path.read_text(encoding="utf-8"))
            return TranscriptDocument.model_validate(raw_document)
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            raise TranscriptionError(
                f"ASR Worker produced an invalid transcript artifact: {artifact_path}"
            ) from exc

    def shutdown(self) -> None:
        """Request Worker shutdown and release the subprocess."""

        if self._process is None:
            return
        try:
            if self._process.poll() is None:
                with suppress(CaptionerError):
                    self._send("shutdown", {})
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.terminate()
                    self._process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            if self._process.poll() is None:
                self._process.terminate()
        finally:
            self._process = None
            self.capabilities = None

    def _launch(self) -> None:
        repository_root = _repository_root()
        source_root = repository_root / "src"
        environment = os.environ.copy()
        current_pythonpath = environment.get("PYTHONPATH", "")
        pythonpath_parts = [str(source_root), str(repository_root)]
        if current_pythonpath:
            pythonpath_parts.append(current_pythonpath)
        environment["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
        try:
            self._process = subprocess.Popen(
                list(self._command),
                cwd=repository_root,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise ProviderUnavailableError(
                f"could not start {self._expected_provider_id} Worker: {exc}"
            ) from exc

    def _hello(self) -> AsrCapabilities:
        hello = self._send("hello", {})
        if hello.get("protocol_version") != PROTOCOL_VERSION:
            raise ProviderUnavailableError("ASR Worker protocol version mismatch")
        if hello.get("provider_id") != self._expected_provider_id:
            raise ProviderUnavailableError("ASR Worker provider ID mismatch")
        return AsrCapabilities.model_validate(hello.get("capabilities"))

    def _send(self, command: str, payload: dict[str, object]) -> dict[str, object]:
        process = self._process
        if process is None or process.poll() is not None:
            raise ProviderUnavailableError("ASR Worker is not running")
        if process.stdin is None or process.stdout is None:
            raise ProviderUnavailableError("ASR Worker pipes are unavailable")
        request = json.dumps(
            {"command": command, "payload": payload}, ensure_ascii=False
        )
        try:
            process.stdin.write(request + "\n")
            process.stdin.flush()
            response_line = process.stdout.readline()
        except OSError as exc:
            raise TranscriptionError(
                f"could not communicate with ASR Worker: {exc}"
            ) from exc
        if not response_line:
            detail = "ASR Worker exited without a response"
            if process.poll() is not None and process.stderr is not None:
                stderr = process.stderr.read().strip()
                if stderr:
                    detail = f"{detail}: {stderr[-1_000:]}"
            raise TranscriptionError(detail)
        try:
            response = _object(json.loads(response_line), "Worker response")
        except (json.JSONDecodeError, TypeError) as exc:
            raise TranscriptionError("ASR Worker returned invalid JSON") from exc
        if response.get("ok") is not True:
            error = _object(response.get("error"), "Worker error")
            message = error.get("message", "unknown ASR Worker error")
            raise TranscriptionError(str(message))
        return _object(response.get("result"), "Worker result")


class FakeWorkerClient:
    """Compatibility wrapper for the Phase 0 Fake Worker."""

    def __init__(self, client: NdjsonWorkerClient | None = None) -> None:
        self._client = client

    @property
    def capabilities(self) -> AsrCapabilities | None:
        return self._client.capabilities if self._client is not None else None

    def start(self, model_name: str = "fake-v1") -> AsrCapabilities:
        client = self._get_client()
        return client.start({"model_name": model_name})

    def transcribe(
        self, request: TranscriptionRequest, artifact_dir: Path
    ) -> TranscriptDocument:
        return self._get_client().transcribe(request, artifact_dir)

    def shutdown(self) -> None:
        if self._client is not None:
            self._client.shutdown()

    def close(self) -> None:
        self.shutdown()

    def _get_client(self) -> NdjsonWorkerClient:
        if self._client is None:
            self._client = NdjsonWorkerClient(
                command=(sys.executable, "-m", "workers.fake"),
                expected_provider_id="fake",
            )
        return self._client
