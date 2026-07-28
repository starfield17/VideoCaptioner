"""Qwen3 ASR and optional Forced Aligner Worker implementation."""

import importlib
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
from captioner.transcription.providers.qwen3 import Qwen3Config
from captioner.transcription.requests import TimestampRequirement


class _Qwen3Model(Protocol):
    def transcribe(self, audio: str, **kwargs: object) -> object:
        """Run the Qwen3 Python API."""
        ...


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TranscriptionError(f"{name} must be a JSON object")
    return cast(dict[str, object], value)


def _value(value: object, name: str) -> object:
    if isinstance(value, dict):
        mapping = cast(dict[str, object], value)
        return mapping.get(name)
    return getattr(value, name, None)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
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


def _qwen_language(value: object) -> str | None:
    if value is None:
        return None
    language = _string(value, "language").strip()
    if language.lower() == "auto":
        return None
    language_names = {
        "ar": "Arabic",
        "cs": "Czech",
        "da": "Danish",
        "de": "German",
        "el": "Greek",
        "en": "English",
        "es": "Spanish",
        "fa": "Persian",
        "fi": "Finnish",
        "fil": "Filipino",
        "fr": "French",
        "hi": "Hindi",
        "hu": "Hungarian",
        "id": "Indonesian",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "ms": "Malay",
        "nl": "Dutch",
        "pl": "Polish",
        "pt": "Portuguese",
        "ro": "Romanian",
        "ru": "Russian",
        "sv": "Swedish",
        "th": "Thai",
        "tr": "Turkish",
        "vi": "Vietnamese",
        "yue": "Cantonese",
        "zh": "Chinese",
    }
    return language_names.get(
        language.lower(), language[:1].upper() + language[1:].lower()
    )


def _result_items(raw_timestamps: object) -> Iterable[object]:
    items = _value(raw_timestamps, "items")
    return _iterable(
        raw_timestamps if items is None else items,
        "result.time_stamps.items",
    )


def _join_aligned_text(parts: Iterable[str]) -> str:
    values = tuple(parts)
    if any(any(_is_cjk(character) for character in value) for value in values):
        return "".join(values)
    return " ".join(values)


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3000 <= codepoint <= 0x30FF
        or 0x3400 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def _aligned_items(raw_timestamps: object) -> tuple[tuple[str, int, int], ...]:
    """Keep provider ranges and merge zero-span text into a timed neighbor."""

    aligned: list[tuple[str, int, int]] = []
    pending_text: list[str] = []
    for raw_item in _result_items(raw_timestamps):
        item_text = _string(_value(raw_item, "text"), "time_stamps.text")
        start_seconds = _number(
            _value(raw_item, "start_time"), "time_stamps.start_time"
        )
        end_seconds = _number(_value(raw_item, "end_time"), "time_stamps.end_time")
        if start_seconds < 0 or end_seconds < 0:
            raise TranscriptionError("Qwen3 Forced Aligner returned a negative range")
        if start_seconds > end_seconds:
            raise TranscriptionError("Qwen3 Forced Aligner returned a reversed range")
        if start_seconds == end_seconds:
            pending_text.append(item_text)
            continue
        start_ms = _seconds_to_ms(start_seconds, "time_stamps.start_time")
        end_ms = _seconds_to_ms(end_seconds, "time_stamps.end_time")
        if start_ms >= end_ms:
            raise TranscriptionError(
                "Qwen3 Forced Aligner returned a sub-millisecond range"
            )
        text = _join_aligned_text((*pending_text, item_text))
        pending_text.clear()
        aligned.append((text, start_ms, end_ms))
    if pending_text:
        if not aligned:
            raise TranscriptionError(
                "Qwen3 Forced Aligner returned no positive-duration items"
            )
        text, start_ms, end_ms = aligned[-1]
        aligned[-1] = (
            _join_aligned_text((text, *pending_text)),
            start_ms,
            end_ms,
        )
    return tuple(aligned)


def _raw_results(value: object) -> tuple[object, ...]:
    if isinstance(value, dict) or hasattr(value, "text"):
        return (cast(object, value),)
    return tuple(_iterable(value, "Qwen3 transcription results"))


def _native_segment_values(raw_result: object) -> Iterable[object]:
    nested = _value(raw_result, "segments")
    if nested is not None:
        return _iterable(nested, "result.segments")
    if (
        _value(raw_result, "start_time") is not None
        and _value(raw_result, "end_time") is not None
    ):
        return (raw_result,)
    raise TranscriptionError(
        "Qwen3 returned no Forced Aligner timestamps or native segment timestamps; "
        "refusing to fabricate Word timing"
    )


def _native_segment_range(raw_segment: object) -> tuple[int, int]:
    start_value = _value(raw_segment, "start_time")
    if start_value is None:
        start_value = _value(raw_segment, "start")
    end_value = _value(raw_segment, "end_time")
    if end_value is None:
        end_value = _value(raw_segment, "end")
    start_ms = _seconds_to_ms(start_value, "segment.start_time")
    end_ms = _seconds_to_ms(end_value, "segment.end_time")
    if start_ms >= end_ms:
        raise TranscriptionError("Qwen3 returned an invalid native segment range")
    return start_ms, end_ms


def map_qwen3_results(
    raw_results: object,
    *,
    language: str | None,
    model_name: str,
) -> TranscriptDocument:
    """Map Qwen3 results using only ASR or Forced Aligner timing fields."""

    results = _raw_results(raw_results)
    if not results:
        raise TranscriptionError("Qwen3 returned no transcription results")

    words: list[TimedWord] = []
    word_segments: list[TranscriptSegment] = []
    native_segments: list[TranscriptSegment] = []
    result_texts: list[str] = []
    detected_language = ""

    for result_index, raw_result in enumerate(results, start=1):
        result_text = _string(_value(raw_result, "text"), "result.text")
        result_texts.append(result_text)
        result_language = _value(raw_result, "language")
        if isinstance(result_language, str) and result_language.strip():
            detected_language = result_language.strip()

        raw_timestamps = _value(raw_result, "time_stamps")
        if raw_timestamps is not None:
            word_ids: list[str] = []
            for item_text, start_ms, end_ms in _aligned_items(raw_timestamps):
                word_id = f"w{len(words) + 1:06d}"
                words.append(
                    TimedWord(
                        id=word_id,
                        text=item_text,
                        start_ms=start_ms,
                        end_ms=end_ms,
                    )
                )
                word_ids.append(word_id)
            if not word_ids:
                raise TranscriptionError(
                    "Qwen3 Forced Aligner returned no aligned items"
                )
            word_segments.append(
                TranscriptSegment(
                    id=f"seg{result_index:06d}",
                    text=result_text,
                    start_ms=words[-len(word_ids)].start_ms,
                    end_ms=words[-1].end_ms,
                    word_ids=tuple(word_ids),
                )
            )
            continue

        for native_index, raw_segment in enumerate(
            _native_segment_values(raw_result), start=1
        ):
            segment_text_value = _value(raw_segment, "text")
            if segment_text_value is None and native_index == 1:
                segment_text_value = result_text
            segment_text = _string(segment_text_value, "segment.text")
            start_ms, end_ms = _native_segment_range(raw_segment)
            native_segments.append(
                TranscriptSegment(
                    id=f"seg{result_index:06d}-{native_index:03d}",
                    text=segment_text,
                    start_ms=start_ms,
                    end_ms=end_ms,
                )
            )

    if words and native_segments:
        raise TranscriptionError(
            "Qwen3 returned mixed Forced Aligner and native segment timing"
        )
    if words:
        timing_origin = TimingOrigin.FORCED_ALIGNMENT
        segments = word_segments
    else:
        timing_origin = TimingOrigin.SEGMENT_NATIVE
        segments = native_segments
    if not segments:
        raise TranscriptionError("Qwen3 returned no timed segments")

    document_language = detected_language or (language or "")
    if not document_language:
        raise TranscriptionError("Qwen3 returned no language")
    return TranscriptDocument(
        language=document_language,
        text="".join(result_texts),
        timing_origin=timing_origin,
        words=tuple(words),
        segments=tuple(segments),
        provider="qwen3-asr",
        model_name=model_name,
    )


class Qwen3Worker:
    """Stateful single-model Worker for one serial CLI run."""

    provider_id = "qwen3-asr"
    worker_version = "qwen3-asr-0.1"

    def __init__(
        self,
        model_factory: Callable[[Qwen3Config], _Qwen3Model] | None = None,
    ) -> None:
        self._model: _Qwen3Model | None = None
        self._config: Qwen3Config | None = None
        self._model_factory = model_factory
        self._shutdown = False

    def hello(self) -> dict[str, object]:
        self._ensure_running()
        if self._model_factory is None:
            self._ensure_dependency()
        return {
            "protocol_version": PROTOCOL_VERSION,
            "provider_id": self.provider_id,
            "capabilities": {
                "native_word_timestamps": False,
                "forced_alignment": True,
                "language_detection": True,
                "initial_prompt": True,
                "internal_vad": False,
                "supported_languages": None,
            },
            "worker_version": self.worker_version,
        }

    def load(self, payload: dict[str, object]) -> dict[str, object]:
        self._ensure_running()
        if self._model is not None:
            raise ProviderUnavailableError("Qwen3 ASR model is already loaded")
        config = Qwen3Config.model_validate(payload.get("config"))
        try:
            factory = self._model_factory or _default_model_factory
            self._model = factory(config)
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
                f"could not load Qwen3 ASR model: {exc}"
            ) from exc
        return {"loaded": True, "model_name": config.model}

    def transcribe(self, payload: dict[str, object]) -> dict[str, object]:
        self._ensure_running()
        model = self._model
        config = self._config
        if model is None or config is None:
            raise TranscriptionError("Qwen3 ASR model must be loaded first")
        audio_path = Path(_string(payload.get("audio_path"), "audio_path"))
        artifact_dir = Path(_string(payload.get("artifact_dir"), "artifact_dir"))
        if not audio_path.is_file():
            raise TranscriptionError(f"prepared audio does not exist: {audio_path}")
        timestamps = _string(payload.get("timestamps"), "timestamps")
        if timestamps not in {
            requirement.value for requirement in TimestampRequirement
        }:
            raise TranscriptionError(f"unsupported timestamp requirement: {timestamps}")
        if timestamps == TimestampRequirement.REQUIRED.value and (
            config.forced_aligner_model is None
        ):
            raise TranscriptionError(
                "Qwen3 requires a forced aligner for required word timestamps"
            )
        language = _qwen_language(payload.get("language"))
        prompt_value = payload.get("initial_prompt")
        initial_prompt = (
            "" if prompt_value is None else _string(prompt_value, "initial_prompt")
        )
        return_time_stamps = (
            timestamps != TimestampRequirement.DISABLED.value
            and config.forced_aligner_model is not None
        )
        try:
            raw_results = model.transcribe(
                str(audio_path),
                context=initial_prompt,
                language=language,
                return_time_stamps=return_time_stamps,
            )
            document = map_qwen3_results(
                raw_results,
                language=language,
                model_name=config.model,
            )
        except (CaptionerError, OSError, RuntimeError, TypeError, ValueError) as exc:
            if isinstance(exc, TranscriptionError):
                raise
            raise TranscriptionError(f"Qwen3 transcription failed: {exc}") from exc

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
            raise ProviderUnavailableError("Qwen3 ASR Worker is shut down")

    @staticmethod
    def _ensure_dependency() -> None:
        try:
            importlib.import_module("qwen_asr")
        except Exception as exc:
            raise ProviderUnavailableError(
                f"qwen-asr import failed under Python {sys.version.split()[0]}: {exc}"
            ) from exc


def _default_model_factory(config: Qwen3Config) -> _Qwen3Model:
    try:
        qwen_module = importlib.import_module("qwen_asr")
        torch_module = importlib.import_module("torch")
    except Exception as exc:
        raise ProviderUnavailableError(
            f"Qwen3 runtime imports failed under Python {sys.version.split()[0]}: {exc}"
        ) from exc
    constructor = getattr(qwen_module, "Qwen3ASRModel", None)
    if not callable(constructor):
        raise ProviderUnavailableError("qwen_asr.Qwen3ASRModel is unavailable")
    dtype = getattr(torch_module, config.dtype, None)
    if dtype is None:
        raise ProviderUnavailableError(
            f"torch does not provide configured dtype: {config.dtype}"
        )
    parameters: dict[str, object] = {
        "dtype": dtype,
        "device_map": config.device,
    }
    aligner = (
        str(config.forced_aligner_path)
        if config.forced_aligner_path is not None
        else config.forced_aligner_model
    )
    if aligner is not None:
        parameters["forced_aligner"] = aligner
        parameters["forced_aligner_kwargs"] = {
            "dtype": dtype,
            "device_map": config.device,
        }
    from_pretrained = getattr(constructor, "from_pretrained", None)
    if not callable(from_pretrained):
        raise ProviderUnavailableError(
            "qwen_asr.Qwen3ASRModel.from_pretrained is unavailable"
        )
    factory = cast(Callable[..., _Qwen3Model], from_pretrained)
    model = str(config.model_path) if config.model_path is not None else config.model
    return factory(model, **parameters)


def _response(result: dict[str, object]) -> str:
    return json.dumps({"ok": True, "result": result}, ensure_ascii=False)


def run_worker() -> None:
    """Read one JSON command per line and write one response per line."""

    worker = Qwen3Worker()
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


__all__ = ["Qwen3Worker", "map_qwen3_results", "run_worker"]
