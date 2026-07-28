"""Fake ASR worker implementation and its NDJSON command loop."""

import json
import sys
from pathlib import Path
from typing import cast

from captioner.shared.errors import TranscriptionError
from captioner.transcription.models import (
    TimedWord,
    TimingOrigin,
    TranscriptDocument,
    TranscriptSegment,
)
from workers.common.protocol import PROTOCOL_VERSION, SUPPORTED_COMMANDS


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TranscriptionError(f"{name} must be a JSON object")
    return cast(dict[str, object], value)


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TranscriptionError(f"{name} must be a JSON array")
    return cast(list[object], value)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TranscriptionError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TranscriptionError(f"{name} must be an integer")
    return value


def _confidence(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TranscriptionError(f"{name} must be a number")
    return float(value)


class FakeAsrWorker:
    """Stateful worker with the four commands defined by the manual."""

    provider_id = "fake"
    worker_version = "fake-0.1"

    def __init__(self) -> None:
        self._loaded = False
        self._shutdown = False

    def hello(self) -> dict[str, object]:
        self._ensure_running()
        return {
            "protocol_version": PROTOCOL_VERSION,
            "provider_id": self.provider_id,
            "capabilities": {
                "native_word_timestamps": True,
                "forced_alignment": False,
                "language_detection": False,
                "initial_prompt": True,
                "internal_vad": False,
                "supported_languages": ["en", "zh"],
            },
            "worker_version": self.worker_version,
        }

    def load(self, payload: dict[str, object]) -> dict[str, object]:
        self._ensure_running()
        if self._loaded:
            raise TranscriptionError("Fake ASR model is already loaded")
        model_name = payload.get("model_name", "fake-v1")
        _string(model_name, "model_name")
        self._loaded = True
        return {"loaded": True, "model_name": model_name}

    def transcribe(self, payload: dict[str, object]) -> dict[str, object]:
        self._ensure_running()
        if not self._loaded:
            raise TranscriptionError("Fake ASR model must be loaded before transcribe")
        audio_path = Path(_string(payload.get("audio_path"), "audio_path"))
        artifact_dir = Path(_string(payload.get("artifact_dir"), "artifact_dir"))
        document = self._map_fixture(audio_path)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "transcript.raw.json"
        temporary_path = artifact_dir / "transcript.raw.json.tmp"
        temporary_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        temporary_path.replace(artifact_path)
        return {
            "artifact_path": str(artifact_path),
            "summary": {
                "language": document.language,
                "segments": len(document.segments),
                "words": len(document.words),
                "timing_origin": document.timing_origin.value,
            },
        }

    def shutdown(self) -> dict[str, object]:
        self._ensure_running()
        self._shutdown = True
        return {"shutdown": True}

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown

    def handle(self, command: str, payload: dict[str, object]) -> dict[str, object]:
        if command not in SUPPORTED_COMMANDS:
            raise TranscriptionError(f"unsupported worker command: {command}")
        if command == "hello":
            return self.hello()
        if command == "load":
            return self.load(payload)
        if command == "transcribe":
            return self.transcribe(payload)
        return self.shutdown()

    def _ensure_running(self) -> None:
        if self._shutdown:
            raise TranscriptionError("Fake ASR worker is shut down")

    @staticmethod
    def _map_fixture(audio_path: Path) -> TranscriptDocument:
        if not audio_path.is_file():
            raise TranscriptionError(f"prepared fixture does not exist: {audio_path}")
        try:
            raw_value = json.loads(audio_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TranscriptionError(f"invalid fake fixture: {audio_path}") from exc
        raw = _object(raw_value, "fixture")
        language = _string(raw.get("language"), "language")
        model_name = raw.get("model_name", "fake-v1")
        _string(model_name, "model_name")
        raw_segments = _list(raw.get("segments"), "segments")

        words: list[TimedWord] = []
        segments: list[TranscriptSegment] = []
        for segment_index, raw_segment_value in enumerate(raw_segments, start=1):
            raw_segment = _object(raw_segment_value, f"segments[{segment_index}]")
            segment_id = _string(
                raw_segment.get("id", f"seg{segment_index:06d}"),
                f"segments[{segment_index}].id",
            )
            segment_text = _string(
                raw_segment.get("text"), f"segments[{segment_index}].text"
            )
            segment_start = _integer(
                raw_segment.get("start_ms"),
                f"segments[{segment_index}].start_ms",
            )
            segment_end = _integer(
                raw_segment.get("end_ms"),
                f"segments[{segment_index}].end_ms",
            )
            raw_words = raw_segment.get("words")
            word_ids: list[str] = []
            if raw_words is not None:
                for word_index, raw_word_value in enumerate(
                    _list(raw_words, f"segments[{segment_index}].words"),
                    start=1,
                ):
                    raw_word = _object(
                        raw_word_value,
                        f"segments[{segment_index}].words[{word_index}]",
                    )
                    word_id = _string(
                        raw_word.get("id", f"w{len(words) + 1:06d}"),
                        "word.id",
                    )
                    word = TimedWord(
                        id=word_id,
                        text=_string(raw_word.get("text"), "word.text"),
                        start_ms=_integer(raw_word.get("start_ms"), "word.start_ms"),
                        end_ms=_integer(raw_word.get("end_ms"), "word.end_ms"),
                        confidence=_confidence(
                            raw_word.get("confidence"), "word.confidence"
                        ),
                    )
                    words.append(word)
                    word_ids.append(word.id)
            segments.append(
                TranscriptSegment(
                    id=segment_id,
                    text=segment_text,
                    start_ms=segment_start,
                    end_ms=segment_end,
                    word_ids=tuple(word_ids),
                )
            )

        if not segments:
            raise TranscriptionError("fake fixture must contain at least one segment")
        text_value = raw.get("text")
        text = (
            _string(text_value, "text")
            if text_value is not None
            else "".join(segment.text for segment in segments)
        )
        origin = TimingOrigin.ASR_NATIVE if words else TimingOrigin.SEGMENT_NATIVE
        return TranscriptDocument(
            language=language,
            text=text,
            timing_origin=origin,
            words=tuple(words),
            segments=tuple(segments),
            provider="fake",
            model_name=cast(str, model_name),
        )


def _response(result: dict[str, object]) -> str:
    return json.dumps({"ok": True, "result": result}, ensure_ascii=False)


def run_worker() -> None:
    """Read one JSON command per line and write one response per line."""

    worker = FakeAsrWorker()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            raw_request = json.loads(line)
            request = _object(raw_request, "request")
            command = _string(request.get("command"), "command")
            payload_value = request.get("payload", {})
            payload = _object(payload_value, "payload")
            output = _response(worker.handle(command, payload))
        except (TranscriptionError, ValueError, TypeError, json.JSONDecodeError) as exc:
            output = json.dumps(
                {
                    "ok": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
                ensure_ascii=False,
            )
        print(output, flush=True)
        if worker.is_shutdown:
            break
