import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from workers.qwen3 import worker as qwen3_worker

from captioner.shared.errors import ProviderUnavailableError, TranscriptionError
from captioner.transcription.capabilities import AsrCapabilities
from captioner.transcription.models import TimingOrigin, TranscriptDocument
from captioner.transcription.providers.qwen3 import (
    Qwen3Config,
    Qwen3TranscriptionService,
)
from captioner.transcription.providers.registry import PROVIDERS
from captioner.transcription.requests import TimestampRequirement, TranscriptionRequest


def _fixture(name: str) -> object:
    path = Path(__file__).parents[1] / "fixtures" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_qwen3_capability_is_registered_for_forced_alignment() -> None:
    capabilities = PROVIDERS["qwen3-asr"].capabilities
    assert not capabilities.native_word_timestamps
    assert capabilities.forced_alignment
    assert capabilities.language_detection


def test_forced_aligner_fixture_preserves_real_item_ranges() -> None:
    document = qwen3_worker.map_qwen3_results(
        _fixture("qwen3_aligned.json"), language=None, model_name="Qwen/Qwen3-ASR-1.7B"
    )

    assert isinstance(document, TranscriptDocument)
    assert document.timing_origin is TimingOrigin.FORCED_ALIGNMENT
    assert document.provider == "qwen3-asr"
    assert [word.start_ms for word in document.words] == [125, 720]
    assert [word.end_ms for word in document.words] == [610, 1_305]
    assert document.segments[0].word_ids == ("w000001", "w000002")


def test_no_aligner_does_not_fabricate_word_timing() -> None:
    with pytest.raises(TranscriptionError, match="refusing to fabricate Word timing"):
        qwen3_worker.map_qwen3_results(
            _fixture("qwen3_no_aligner.json"),
            language=None,
            model_name="Qwen/Qwen3-ASR-1.7B",
        )


def test_native_segment_fixture_has_no_words() -> None:
    document = qwen3_worker.map_qwen3_results(
        [
            {
                "language": "English",
                "text": "a segment",
                "segments": [{"text": "a segment", "start_time": 0.0, "end_time": 1.2}],
            }
        ],
        language=None,
        model_name="Qwen/Qwen3-ASR-1.7B",
    )

    assert document.timing_origin is TimingOrigin.SEGMENT_NATIVE
    assert document.words == ()
    assert document.segments[0].word_ids == ()


@dataclass
class _RawAlignedItem:
    text: str
    start_time: float
    end_time: float


@dataclass
class _RawResult:
    language: str
    text: str
    time_stamps: tuple[_RawAlignedItem, ...]


class _RecordingQwenModel:
    def __init__(self) -> None:
        self.audio: str | None = None
        self.kwargs: dict[str, object] | None = None

    def transcribe(self, audio: str, **kwargs: object) -> object:
        self.audio = audio
        self.kwargs = kwargs
        return [
            _RawResult(
                language="Chinese",
                text="今天下雨",
                time_stamps=(
                    _RawAlignedItem("今天", 0.125, 0.610),
                    _RawAlignedItem("下雨", 0.720, 1.305),
                ),
            )
        ]


def test_worker_contract_forwards_context_and_alignment_request(
    tmp_path: Path,
) -> None:
    model = _RecordingQwenModel()

    def factory(_config: Qwen3Config) -> _RecordingQwenModel:
        return model

    worker = qwen3_worker.Qwen3Worker(model_factory=factory)
    hello = worker.hello()
    assert hello["provider_id"] == "qwen3-asr"
    assert hello["capabilities"] == {
        "native_word_timestamps": False,
        "forced_alignment": True,
        "language_detection": True,
        "initial_prompt": True,
        "internal_vad": False,
        "supported_languages": None,
    }

    worker.load(
        {
            "config": Qwen3Config(
                model="fixture-qwen",
                device="cpu",
                dtype="float32",
                forced_aligner_model="fixture-aligner",
            ).model_dump(mode="json")
        }
    )
    audio_path = tmp_path / "prepared.wav"
    audio_path.write_bytes(b"fixture audio")
    result = worker.transcribe(
        {
            "audio_path": str(audio_path),
            "artifact_dir": str(tmp_path / "artifacts"),
            "language": "zh",
            "initial_prompt": "captioning vocabulary",
            "timestamps": "required",
        }
    )

    assert model.audio == str(audio_path)
    assert model.kwargs == {
        "context": "captioning vocabulary",
        "language": "Chinese",
        "return_time_stamps": True,
    }
    artifact_value = result["artifact_path"]
    assert isinstance(artifact_value, str)
    document = TranscriptDocument.model_validate(
        json.loads(Path(artifact_value).read_text(encoding="utf-8"))
    )
    assert document.timing_origin is TimingOrigin.FORCED_ALIGNMENT
    assert document.words
    assert worker.shutdown() == {"shutdown": True}


def test_default_factory_forwards_forced_aligner_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, object] = {}

    class Constructor:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs: object) -> _RecordingQwenModel:
            created["model_name"] = model_name
            created["kwargs"] = kwargs
            return _RecordingQwenModel()

    def import_module(name: str) -> object:
        if name == "qwen_asr":
            return SimpleNamespace(Qwen3ASRModel=Constructor)
        if name == "torch":
            return SimpleNamespace(bfloat16="fixture-bfloat16")
        raise AssertionError(f"unexpected module import: {name}")

    monkeypatch.setattr(qwen3_worker.importlib, "import_module", import_module)
    worker = qwen3_worker.Qwen3Worker()
    worker.load(
        {
            "config": Qwen3Config(
                model="fixture-asr",
                device="cuda:0",
                dtype="bfloat16",
                forced_aligner_model="fixture-aligner",
            ).model_dump(mode="json")
        }
    )

    assert created == {
        "model_name": "fixture-asr",
        "kwargs": {
            "dtype": "fixture-bfloat16",
            "device_map": "cuda:0",
            "forced_aligner": "fixture-aligner",
            "forced_aligner_kwargs": {
                "dtype": "fixture-bfloat16",
                "device_map": "cuda:0",
            },
        },
    }
    worker.shutdown()


class _StartMustNotBeCalled:
    def start(self, config: Qwen3Config) -> AsrCapabilities:
        del config
        raise AssertionError("the provider must fail before Worker start")

    def transcribe(
        self, request: TranscriptionRequest, artifact_dir: Path
    ) -> TranscriptDocument:
        del request, artifact_dir
        raise AssertionError("transcribe must not be called")

    def shutdown(self) -> None:
        raise AssertionError("shutdown must not be called")


def test_required_words_fail_before_model_start_without_aligner() -> None:
    service = Qwen3TranscriptionService(
        Qwen3Config(forced_aligner_model=None),
        timestamps=TimestampRequirement.REQUIRED,
        client=_StartMustNotBeCalled(),
    )

    with pytest.raises(ProviderUnavailableError, match="forced_aligner_model"):
        service.start()
