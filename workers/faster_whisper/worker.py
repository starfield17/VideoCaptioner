"""Faster Whisper Python API Worker implementation."""

import importlib
import importlib.util
import json
import math
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol, cast

from captioner.shared.errors import (
    CaptionerError,
    ProviderUnavailableError,
    TranscriptionError,
)
from captioner.shared.worker_protocol import PROTOCOL_VERSION, SUPPORTED_COMMANDS
from captioner.transcription.models import (
    TimedWord,
    TimingOrigin,
    TranscriptDocument,
    TranscriptSegment,
)
from captioner.transcription.providers.faster_whisper import FasterWhisperConfig


class _WhisperModel(Protocol):
    def transcribe(
        self, audio: str, **kwargs: object
    ) -> tuple[Iterable[object], object]:
        """Run the Faster Whisper Python API."""
        ...


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TranscriptionError(f"{name} must be a JSON object")
    return cast(dict[str, object], value)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TranscriptionError(f"{name} must be a non-empty string")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TranscriptionError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise TranscriptionError(f"{name} must be finite")
    return number


def _seconds_to_ms(value: object, name: str) -> int:
    seconds = _number(value, name)
    if seconds < 0:
        raise TranscriptionError(f"{name} cannot be negative")
    return int(round(seconds * 1_000))


def _iterable(value: object, name: str) -> Iterable[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TranscriptionError(f"{name} must be an iterable")
    return cast(Iterable[object], value)


class FasterWhisperWorker:
    """Stateful single-model Worker for one serial CLI run."""

    provider_id = "faster-whisper"
    worker_version = "faster-whisper-0.1"

    def __init__(self) -> None:
        self._model: _WhisperModel | None = None
        self._config: FasterWhisperConfig | None = None
        self._shutdown = False

    def hello(self) -> dict[str, object]:
        self._ensure_running()
        self._ensure_dependency()
        return {
            "protocol_version": PROTOCOL_VERSION,
            "provider_id": self.provider_id,
            "capabilities": {
                "native_word_timestamps": True,
                "forced_alignment": False,
                "language_detection": True,
                "initial_prompt": True,
                "internal_vad": True,
                "supported_languages": None,
            },
            "worker_version": self.worker_version,
        }

    def load(self, payload: dict[str, object]) -> dict[str, object]:
        self._ensure_running()
        if self._model is not None:
            raise ProviderUnavailableError("Faster Whisper model is already loaded")
        try:
            config = FasterWhisperConfig.model_validate(payload.get("config"))
            module = importlib.import_module("faster_whisper")
            constructor_value = getattr(module, "WhisperModel", None)
            if not callable(constructor_value):
                raise ProviderUnavailableError(
                    "faster_whisper.WhisperModel is unavailable"
                )
            constructor = cast(Callable[..., _WhisperModel], constructor_value)
            self._model = constructor(
                config.model,
                device=config.device,
                compute_type=config.compute_type,
            )
            self._config = config
        except (
            CaptionerError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            self.shutdown()
            if isinstance(exc, ProviderUnavailableError):
                raise
            raise ProviderUnavailableError(
                f"could not load Faster Whisper model: {exc}"
            ) from exc
        return {"loaded": True, "model_name": config.model}

    def transcribe(self, payload: dict[str, object]) -> dict[str, object]:
        self._ensure_running()
        model = self._model
        config = self._config
        if model is None or config is None:
            raise TranscriptionError("Faster Whisper model must be loaded first")
        audio_path = Path(_string(payload.get("audio_path"), "audio_path"))
        artifact_dir = Path(_string(payload.get("artifact_dir"), "artifact_dir"))
        if not audio_path.is_file():
            raise TranscriptionError(f"prepared audio does not exist: {audio_path}")
        timestamps = _string(payload.get("timestamps"), "timestamps")
        if timestamps == "disabled":
            raise TranscriptionError(
                "Faster Whisper Phase 1 requires native word timestamps"
            )
        language_value = payload.get("language")
        language = (
            None if language_value is None else _string(language_value, "language")
        )
        prompt_value = payload.get("initial_prompt")
        initial_prompt = (
            None if prompt_value is None else _string(prompt_value, "initial_prompt")
        )
        try:
            raw_segments, info = model.transcribe(
                str(audio_path),
                language=language,
                initial_prompt=initial_prompt,
                beam_size=config.beam_size,
                word_timestamps=True,
                vad_filter=config.vad.enabled,
                vad_parameters=config.vad.as_transcribe_parameters(),
            )
            detected_language = _string(
                getattr(info, "language", None), "transcription_info.language"
            )
            document = map_faster_whisper_segments(
                raw_segments,
                language=detected_language,
                model_name=config.model,
            )
        except (CaptionerError, OSError, RuntimeError, TypeError, ValueError) as exc:
            if isinstance(exc, TranscriptionError):
                raise
            raise TranscriptionError(
                f"Faster Whisper transcription failed: {exc}"
            ) from exc

        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "transcript.raw.json"
        temporary_path = artifact_dir / "transcript.raw.json.tmp"
        temporary_path.write_text(
            document.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
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
        self._model = None
        self._config = None
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
            raise ProviderUnavailableError("Faster Whisper Worker is shut down")

    @staticmethod
    def _ensure_dependency() -> None:
        if importlib.util.find_spec("faster_whisper") is None:
            raise ProviderUnavailableError(
                "faster-whisper is not installed in the ASR environment"
            )


def map_faster_whisper_segments(
    raw_segments: Iterable[object],
    language: str,
    model_name: str,
) -> TranscriptDocument:
    """Map real provider word objects without inventing finer timestamps."""

    words: list[TimedWord] = []
    segments: list[TranscriptSegment] = []
    segment_texts: list[str] = []
    for segment_index, raw_segment in enumerate(raw_segments, start=1):
        segment_text = _string(getattr(raw_segment, "text", None), "segment.text")
        segment_start = _seconds_to_ms(
            getattr(raw_segment, "start", None), "segment.start"
        )
        segment_end = _seconds_to_ms(getattr(raw_segment, "end", None), "segment.end")
        if segment_start >= segment_end:
            raise TranscriptionError("Faster Whisper returned an invalid segment range")
        raw_words = getattr(raw_segment, "words", None)
        if raw_words is None:
            raise TranscriptionError(
                "Faster Whisper returned no native words; refusing to fabricate timing"
            )
        word_ids: list[str] = []
        for raw_word in _iterable(raw_words, "segment.words"):
            word_index = len(words) + 1
            word_text = _string(getattr(raw_word, "word", None), "word.word")
            word_start = _seconds_to_ms(getattr(raw_word, "start", None), "word.start")
            word_end = _seconds_to_ms(getattr(raw_word, "end", None), "word.end")
            if word_start >= word_end:
                raise TranscriptionError(
                    "Faster Whisper returned an invalid word range"
                )
            word_id = f"w{word_index:06d}"
            words.append(
                TimedWord(
                    id=word_id,
                    text=word_text,
                    start_ms=word_start,
                    end_ms=word_end,
                    confidence=_number(
                        getattr(raw_word, "probability", None), "word.probability"
                    ),
                )
            )
            word_ids.append(word_id)
        if not word_ids:
            raise TranscriptionError(
                "Faster Whisper returned a segment without native words"
            )
        segments.append(
            TranscriptSegment(
                id=f"seg{segment_index:06d}",
                text=segment_text,
                start_ms=segment_start,
                end_ms=segment_end,
                word_ids=tuple(word_ids),
            )
        )
        segment_texts.append(segment_text)
    if not segments:
        raise TranscriptionError("Faster Whisper returned no speech segments")
    return TranscriptDocument(
        language=language,
        text="".join(segment_texts),
        timing_origin=TimingOrigin.ASR_NATIVE,
        words=tuple(words),
        segments=tuple(segments),
        provider="faster-whisper",
        model_name=model_name,
    )


def _response(result: dict[str, object]) -> str:
    return json.dumps({"ok": True, "result": result}, ensure_ascii=False)


def run_worker() -> None:
    """Read one JSON command per line and write one response per line."""

    worker = FasterWhisperWorker()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            raw_request = json.loads(line)
            request = _object(raw_request, "request")
            command = _string(request.get("command"), "command")
            payload = _object(request.get("payload", {}), "payload")
            output = _response(worker.handle(command, payload))
        except (CaptionerError, ValueError, TypeError, json.JSONDecodeError) as exc:
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


__all__ = ["FasterWhisperWorker", "map_faster_whisper_segments", "run_worker"]
