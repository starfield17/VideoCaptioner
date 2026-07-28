import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from workers.faster_whisper import worker as faster_whisper_worker

from captioner.shared.errors import TranscriptionError
from captioner.transcription.models import TimingOrigin, TranscriptDocument
from captioner.transcription.providers.faster_whisper import (
    FasterWhisperConfig,
    FasterWhisperVadConfig,
)
from captioner.transcription.providers.registry import PROVIDERS


def test_faster_whisper_capability_is_registered() -> None:
    capabilities = PROVIDERS["faster-whisper"].capabilities
    assert capabilities.native_word_timestamps
    assert capabilities.internal_vad
    assert capabilities.language_detection


@dataclass(frozen=True)
class _RawWord:
    word: str
    start: float
    end: float
    probability: float


@dataclass(frozen=True)
class _RawSegment:
    text: str
    start: float
    end: float
    words: tuple[_RawWord, ...] | None


@dataclass(frozen=True)
class _TranscriptionInfo:
    language: str


def test_native_word_mapping_preserves_provider_timestamps() -> None:
    raw_segment = _RawSegment(
        text="a deliberately uneven phrase",
        start=1.234,
        end=4.567,
        words=(
            _RawWord("a", 1.234, 1.801, 0.91),
            _RawWord("deliberately", 4.111, 4.567, 0.82),
        ),
    )

    document = faster_whisper_worker.map_faster_whisper_segments(
        (raw_segment,), language="en", model_name="tiny"
    )

    assert document.timing_origin is TimingOrigin.ASR_NATIVE
    assert document.provider == "faster-whisper"
    assert document.model_name == "tiny"
    assert [word.start_ms for word in document.words] == [1_234, 4_111]
    assert [word.end_ms for word in document.words] == [1_801, 4_567]
    assert [word.confidence for word in document.words] == [0.91, 0.82]
    assert document.segments[0].word_ids == ("w000001", "w000002")


def test_mapping_rejects_missing_native_words() -> None:
    with pytest.raises(TranscriptionError, match="no native words"):
        faster_whisper_worker.map_faster_whisper_segments(
            (
                _RawSegment(
                    text="no word timing",
                    start=0.0,
                    end=1.0,
                    words=None,
                ),
            ),
            language="en",
            model_name="tiny",
        )


def test_worker_forwards_native_timestamps_and_silero_vad(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    created_models: list[_RecordingModel] = []

    def construct(
        model_name: str, *, device: str, compute_type: str
    ) -> _RecordingModel:
        model = _RecordingModel(model_name, device, compute_type)
        created_models.append(model)
        return model

    def import_module(_name: str) -> object:
        return SimpleNamespace(WhisperModel=construct)

    monkeypatch.setattr(faster_whisper_worker.importlib, "import_module", import_module)
    worker = faster_whisper_worker.FasterWhisperWorker()
    config = FasterWhisperConfig(
        model="tiny",
        device="cpu",
        compute_type="int8",
        beam_size=7,
        vad=FasterWhisperVadConfig(
            threshold=0.63,
            min_speech_duration_ms=300,
            min_silence_duration_ms=600,
            speech_pad_ms=150,
        ),
    )
    audio_path = tmp_path / "prepared.wav"
    audio_path.write_bytes(b"test audio")

    worker.load({"config": config.model_dump(mode="json")})
    result = worker.transcribe(
        {
            "audio_path": str(audio_path),
            "artifact_dir": str(tmp_path / "artifacts"),
            "language": "en",
            "initial_prompt": "captioning vocabulary",
            "timestamps": "required",
        }
    )

    assert len(created_models) == 1
    model = created_models[0]
    assert model.audio == str(audio_path)
    assert model.kwargs is not None
    assert model.kwargs["language"] == "en"
    assert model.kwargs["initial_prompt"] == "captioning vocabulary"
    assert model.kwargs["beam_size"] == 7
    assert model.kwargs["word_timestamps"] is True
    assert model.kwargs["vad_filter"] is True
    assert model.kwargs["vad_parameters"] == {
        "threshold": 0.63,
        "min_speech_duration_ms": 300,
        "min_silence_duration_ms": 600,
        "speech_pad_ms": 150,
    }
    artifact_value = result["artifact_path"]
    assert isinstance(artifact_value, str)
    document = TranscriptDocument.model_validate(
        json.loads(Path(artifact_value).read_text(encoding="utf-8"))
    )
    assert document.timing_origin is TimingOrigin.ASR_NATIVE
    worker.shutdown()


class _RecordingModel:
    def __init__(self, model_name: str, device: str, compute_type: str) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.audio: str | None = None
        self.kwargs: dict[str, object] | None = None

    def transcribe(
        self, audio: str, **kwargs: object
    ) -> tuple[Iterable[object], object]:
        self.audio = audio
        self.kwargs = kwargs
        return (
            iter(
                (
                    _RawSegment(
                        text="a deliberately uneven phrase",
                        start=1.234,
                        end=4.567,
                        words=(
                            _RawWord("a", 1.234, 1.801, 0.91),
                            _RawWord("deliberately", 4.111, 4.567, 0.82),
                        ),
                    ),
                )
            ),
            _TranscriptionInfo(language="en"),
        )
