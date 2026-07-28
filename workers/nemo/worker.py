"""NVIDIA NeMo Parakeet Worker with native word timestamp mapping."""

import importlib
import json
import math
import sys
from collections.abc import Callable, Iterable
from contextlib import redirect_stdout
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
from captioner.transcription.providers.nemo import NemoConfig
from captioner.transcription.requests import TimestampRequirement


class _NemoModel(Protocol):
    def to(self, device: str) -> object:
        """Move the model to its configured device."""
        ...

    def eval(self) -> object:
        """Switch the model to inference mode."""
        ...

    def transcribe(self, **kwargs: object) -> object:
        """Run the NeMo Python transcription API."""
        ...


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TranscriptionError(f"{name} must be a JSON object")
    return cast(dict[str, object], value)


def _value(value: object, name: str) -> object:
    if isinstance(value, dict):
        return cast(dict[str, object], value).get(name)
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


def _hypotheses(value: object) -> tuple[object, ...]:
    candidate = value
    if isinstance(candidate, tuple):
        tuple_value = cast(tuple[object, ...], candidate)
        if len(tuple_value) == 2:
            candidate = tuple_value[0]
        else:
            candidate = cast(object, tuple_value)
    values = tuple(_iterable(candidate, "NeMo transcription results"))
    if len(values) != 1:
        raise TranscriptionError("NeMo must return exactly one hypothesis per request")
    return values


def _stamp_text(value: object, *names: str) -> str:
    for name in names:
        candidate = _value(value, name)
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    raise TranscriptionError(f"NeMo timestamp omitted text field: {'/'.join(names)}")


def _aggregate_segment(text: str, words: tuple[TimedWord, ...]) -> TranscriptSegment:
    return TranscriptSegment(
        id="seg000001",
        text=text,
        start_ms=words[0].start_ms,
        end_ms=words[-1].end_ms,
        word_ids=tuple(word.id for word in words),
    )


def _native_segments(
    raw_segments: object,
    words: tuple[TimedWord, ...],
) -> tuple[TranscriptSegment, ...] | None:
    if raw_segments is None:
        return None
    remaining = list(words)
    segments: list[TranscriptSegment] = []
    previous_end = -1
    for raw_segment in _iterable(raw_segments, "hypothesis.timestamp.segment"):
        text = _stamp_text(raw_segment, "segment", "text")
        start_ms = _seconds_to_ms(_value(raw_segment, "start"), "segment.start")
        end_ms = _seconds_to_ms(_value(raw_segment, "end"), "segment.end")
        if start_ms >= end_ms or start_ms < previous_end:
            return None
        selected = [
            word
            for word in remaining
            if start_ms <= (word.start_ms + word.end_ms) // 2 <= end_ms
        ]
        if not selected:
            return None
        segment = TranscriptSegment(
            id=f"seg{len(segments) + 1:06d}",
            text=text,
            start_ms=min(start_ms, selected[0].start_ms),
            end_ms=max(end_ms, selected[-1].end_ms),
            word_ids=tuple(word.id for word in selected),
        )
        segments.append(segment)
        selected_ids = {word.id for word in selected}
        remaining = [word for word in remaining if word.id not in selected_ids]
        previous_end = segment.end_ms
    if remaining or not segments:
        return None
    return tuple(segments)


def map_nemo_hypotheses(
    raw_results: object,
    *,
    language: str | None,
    model_name: str,
) -> TranscriptDocument:
    """Map only native NeMo timestamp fields into the stable transcript model."""

    hypothesis = _hypotheses(raw_results)[0]
    text = _string(_value(hypothesis, "text"), "hypothesis.text")
    timestamp = _object(_value(hypothesis, "timestamp"), "hypothesis.timestamp")
    words: list[TimedWord] = []
    for raw_word in _iterable(timestamp.get("word"), "hypothesis.timestamp.word"):
        start_ms = _seconds_to_ms(_value(raw_word, "start"), "word.start")
        end_ms = _seconds_to_ms(_value(raw_word, "end"), "word.end")
        if start_ms >= end_ms:
            raise TranscriptionError("NeMo returned an invalid native word range")
        words.append(
            TimedWord(
                id=f"w{len(words) + 1:06d}",
                text=_stamp_text(raw_word, "word", "char", "text"),
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )
    if not words:
        raise TranscriptionError(
            "NeMo returned no native words; refusing to fabricate timing"
        )
    word_values = tuple(words)
    segments = _native_segments(timestamp.get("segment"), word_values)
    if segments is None:
        segments = (_aggregate_segment(text, word_values),)
    detected = _value(hypothesis, "language")
    if not isinstance(detected, str) or not detected.strip():
        detected = _value(hypothesis, "lang")
    document_language = (
        detected.strip()
        if isinstance(detected, str) and detected.strip()
        else language or "und"
    )
    return TranscriptDocument(
        language=document_language,
        text=text,
        timing_origin=TimingOrigin.ASR_NATIVE,
        words=word_values,
        segments=segments,
        provider="nemo-asr",
        model_name=model_name,
    )


class NemoWorker:
    """Stateful single-model Worker for one serial CLI run."""

    provider_id = "nemo-asr"
    worker_version = "nemo-asr-0.1"

    def __init__(
        self,
        model_factory: Callable[[NemoConfig], _NemoModel] | None = None,
    ) -> None:
        self._model: _NemoModel | None = None
        self._config: NemoConfig | None = None
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
                "native_word_timestamps": True,
                "forced_alignment": False,
                "language_detection": True,
                "initial_prompt": False,
                "internal_vad": False,
                "supported_languages": (
                    "bg",
                    "cs",
                    "da",
                    "de",
                    "el",
                    "en",
                    "es",
                    "et",
                    "fi",
                    "fr",
                    "hr",
                    "hu",
                    "it",
                    "lt",
                    "lv",
                    "mt",
                    "nl",
                    "pl",
                    "pt",
                    "ro",
                    "ru",
                    "sk",
                    "sl",
                    "sv",
                    "uk",
                ),
            },
            "worker_version": self.worker_version,
        }

    def load(self, payload: dict[str, object]) -> dict[str, object]:
        self._ensure_running()
        if self._model is not None:
            raise ProviderUnavailableError("NeMo ASR model is already loaded")
        config = NemoConfig.model_validate(payload.get("config"))
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
                f"could not load NeMo ASR model: {exc}"
            ) from exc
        return {"loaded": True, "model_name": config.model}

    def transcribe(self, payload: dict[str, object]) -> dict[str, object]:
        self._ensure_running()
        model = self._model
        config = self._config
        if model is None or config is None:
            raise TranscriptionError("NeMo ASR model must be loaded first")
        audio_path = Path(_string(payload.get("audio_path"), "audio_path"))
        artifact_dir = Path(_string(payload.get("artifact_dir"), "artifact_dir"))
        if not audio_path.is_file():
            raise TranscriptionError(f"prepared audio does not exist: {audio_path}")
        timestamps = _string(payload.get("timestamps"), "timestamps")
        if timestamps == TimestampRequirement.DISABLED.value:
            raise TranscriptionError("NeMo Provider requires native word timestamps")
        if timestamps not in {
            requirement.value for requirement in TimestampRequirement
        }:
            raise TranscriptionError(f"unsupported timestamp requirement: {timestamps}")
        prompt = payload.get("initial_prompt")
        if isinstance(prompt, str) and prompt.strip():
            raise TranscriptionError("NeMo Parakeet does not support initial_prompt")
        language_value = payload.get("language")
        language = (
            None if language_value is None else _string(language_value, "language")
        )
        try:
            raw_results = model.transcribe(
                audio=[str(audio_path)],
                batch_size=config.batch_size,
                timestamps=True,
                verbose=False,
            )
            document = map_nemo_hypotheses(
                raw_results,
                language=language,
                model_name=config.model,
            )
        except (CaptionerError, OSError, RuntimeError, TypeError, ValueError) as exc:
            if isinstance(exc, TranscriptionError):
                raise
            raise TranscriptionError(f"NeMo transcription failed: {exc}") from exc

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
            raise ProviderUnavailableError("NeMo ASR Worker is shut down")

    @staticmethod
    def _ensure_dependency() -> None:
        try:
            importlib.import_module("nemo.collections.asr")
        except Exception as exc:
            raise ProviderUnavailableError(
                f"nemo-toolkit import failed under Python "
                f"{sys.version.split()[0]}: {exc}"
            ) from exc


def _default_model_factory(config: NemoConfig) -> _NemoModel:
    try:
        asr_module = importlib.import_module("nemo.collections.asr")
    except Exception as exc:
        raise ProviderUnavailableError(
            f"NeMo runtime import failed under Python {sys.version.split()[0]}: {exc}"
        ) from exc
    models = getattr(asr_module, "models", None)
    constructor = getattr(models, "ASRModel", None)
    if config.model_path is not None:
        restore_from = getattr(constructor, "restore_from", None)
        if not callable(restore_from):
            raise ProviderUnavailableError(
                "nemo.collections.asr.models.ASRModel.restore_from is unavailable"
            )
        matches = tuple(config.model_path.glob("*.nemo"))
        if len(matches) != 1:
            raise ProviderUnavailableError(
                "NeMo model directory must contain one .nemo file"
            )
        restore = cast(Callable[..., _NemoModel], restore_from)
        model = restore(restore_path=str(matches[0]))
    else:
        from_pretrained = getattr(constructor, "from_pretrained", None)
        if not callable(from_pretrained):
            raise ProviderUnavailableError(
                "nemo.collections.asr.models.ASRModel.from_pretrained is unavailable"
            )
        factory = cast(Callable[..., _NemoModel], from_pretrained)
        model = factory(model_name=config.model)
    device = "cuda" if config.device == "auto" else config.device
    try:
        model.to(device)
    except (OSError, RuntimeError) as exc:
        message = str(exc).lower()
        fallback_markers = (".so", "cuda", "cudnn", "cublas", "out of memory")
        if config.device != "auto" or not any(
            marker in message for marker in fallback_markers
        ):
            raise
        model.to("cpu")
    model.eval()
    return model


def _response(result: dict[str, object]) -> str:
    return json.dumps({"ok": True, "result": result}, ensure_ascii=False)


def run_worker() -> None:
    """Read one JSON command per line and write one response per line."""

    worker = NemoWorker()
    protocol_stdout = sys.stdout
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            raw_request = json.loads(line)
            request = _object(raw_request, "request")
            command = _string(request.get("command"), "command")
            payload = _object(request.get("payload", {}), "payload")
            with redirect_stdout(sys.stderr):
                output = _response(worker.handle(command, payload))
        except (CaptionerError, ValueError, TypeError, json.JSONDecodeError) as exc:
            output = json.dumps(
                {
                    "ok": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
                ensure_ascii=False,
            )
        print(output, file=protocol_stdout, flush=True)
        if worker.is_shutdown:
            break


__all__ = ["NemoWorker", "map_nemo_hypotheses", "run_worker"]
