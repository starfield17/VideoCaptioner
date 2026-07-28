import json
import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
import workers.nemo.worker as nemo_worker_module
from workers.nemo.worker import NemoWorker, map_nemo_hypotheses, run_worker

from captioner.shared.errors import ProviderUnavailableError, TranscriptionError
from captioner.transcription.models import TimingOrigin, TranscriptDocument
from captioner.transcription.providers.nemo import NemoConfig
from captioner.transcription.providers.registry import PROVIDERS


def _hypothesis() -> SimpleNamespace:
    return SimpleNamespace(
        text="A deliberately uneven phrase.",
        language="en",
        timestamp={
            "word": (
                {"word": "A", "start": 1.234, "end": 1.801},
                {
                    "word": "deliberately",
                    "start": 2.111,
                    "end": 3.456,
                },
                {"word": "uneven", "start": 3.5, "end": 4.567},
                {"word": "phrase.", "start": 4.6, "end": 5.1},
            ),
            "segment": (
                {
                    "segment": "A deliberately uneven phrase.",
                    "start": 1.2,
                    "end": 5.2,
                },
            ),
        },
    )


def test_nemo_capability_is_registered() -> None:
    capabilities = PROVIDERS["nemo-asr"].capabilities

    assert capabilities.native_word_timestamps
    assert not capabilities.forced_alignment
    assert capabilities.language_detection
    assert not capabilities.initial_prompt
    assert "en" in (capabilities.supported_languages or ())


def test_native_timestamp_mapping_preserves_provider_words() -> None:
    document = map_nemo_hypotheses(
        [_hypothesis()],
        language=None,
        model_name="nvidia/parakeet-tdt-0.6b-v3",
    )

    assert document.timing_origin is TimingOrigin.ASR_NATIVE
    assert document.provider == "nemo-asr"
    assert document.language == "en"
    assert [word.start_ms for word in document.words] == [
        1_234,
        2_111,
        3_500,
        4_600,
    ]
    assert [word.end_ms for word in document.words] == [
        1_801,
        3_456,
        4_567,
        5_100,
    ]
    assert document.segments[0].word_ids == (
        "w000001",
        "w000002",
        "w000003",
        "w000004",
    )


def test_mapping_rejects_missing_native_words() -> None:
    hypothesis = _hypothesis()
    hypothesis.timestamp = {"word": (), "segment": ()}

    with pytest.raises(TranscriptionError, match="no native words"):
        map_nemo_hypotheses(
            [hypothesis],
            language="en",
            model_name="test",
        )


def test_mapping_aggregates_only_native_word_bounds_when_segments_are_missing() -> None:
    hypothesis = _hypothesis()
    hypothesis.timestamp = {"word": hypothesis.timestamp["word"]}

    document = map_nemo_hypotheses(
        [hypothesis],
        language="en",
        model_name="test",
    )

    assert document.segments[0].start_ms == document.words[0].start_ms
    assert document.segments[0].end_ms == document.words[-1].end_ms


class _RecordingModel:
    def __init__(self) -> None:
        self.device: str | None = None
        self.evaluated = False
        self.kwargs: dict[str, object] | None = None

    def to(self, device: str) -> object:
        self.device = device
        return self

    def eval(self) -> object:
        self.evaluated = True
        return self

    def transcribe(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        return [_hypothesis()]


def test_worker_loads_once_and_forwards_timestamp_request(tmp_path: Path) -> None:
    created: list[_RecordingModel] = []

    def factory(_config: NemoConfig) -> _RecordingModel:
        model = _RecordingModel()
        created.append(model)
        return model

    worker = NemoWorker(model_factory=factory)
    audio_path = tmp_path / "prepared.wav"
    audio_path.write_bytes(b"test audio")
    config = NemoConfig(model="test-model", device="cpu", batch_size=2)

    worker.load({"config": config.model_dump(mode="json")})
    with pytest.raises(ProviderUnavailableError, match="already loaded"):
        worker.load({"config": config.model_dump(mode="json")})
    result = worker.transcribe(
        {
            "audio_path": str(audio_path),
            "artifact_dir": str(tmp_path / "artifacts"),
            "language": "en",
            "initial_prompt": None,
            "timestamps": "required",
        }
    )

    assert len(created) == 1
    assert created[0].kwargs == {
        "audio": [str(audio_path)],
        "batch_size": 2,
        "timestamps": True,
        "verbose": False,
    }
    artifact_value = result["artifact_path"]
    assert isinstance(artifact_value, str)
    document = TranscriptDocument.model_validate_json(
        Path(artifact_value).read_text("utf-8")
    )
    assert document.timing_origin is TimingOrigin.ASR_NATIVE
    worker.shutdown()


def test_worker_rejects_initial_prompt(tmp_path: Path) -> None:
    worker = NemoWorker(model_factory=lambda _config: _RecordingModel())
    worker.load({"config": NemoConfig(model="test").model_dump(mode="json")})
    audio_path = tmp_path / "prepared.wav"
    audio_path.write_bytes(b"test audio")

    with pytest.raises(TranscriptionError, match="does not support initial_prompt"):
        worker.transcribe(
            {
                "audio_path": str(audio_path),
                "artifact_dir": str(tmp_path),
                "language": "en",
                "initial_prompt": "bias phrase",
                "timestamps": "required",
            }
        )


def test_worker_artifact_is_canonical_json(tmp_path: Path) -> None:
    worker = NemoWorker(model_factory=lambda _config: _RecordingModel())
    worker.load({"config": NemoConfig(model="test").model_dump(mode="json")})
    audio_path = tmp_path / "prepared.wav"
    audio_path.write_bytes(b"test audio")

    result = worker.transcribe(
        {
            "audio_path": str(audio_path),
            "artifact_dir": str(tmp_path / "artifacts"),
            "language": None,
            "initial_prompt": None,
            "timestamps": "preferred",
        }
    )

    artifact = Path(str(result["artifact_path"]))
    payload = json.loads(artifact.read_text("utf-8"))
    assert payload["schema_version"] == "transcript.v1"
    assert payload["provider"] == "nemo-asr"


def test_worker_keeps_framework_stdout_out_of_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NoisyWorker:
        is_shutdown = True

        def handle(
            self, _command: str, _payload: dict[str, object]
        ) -> dict[str, object]:
            print("framework log")
            return {"protocol": "clean"}

    protocol_output = StringIO()
    framework_output = StringIO()
    monkeypatch.setattr(nemo_worker_module, "NemoWorker", _NoisyWorker)
    monkeypatch.setattr(sys, "stdin", StringIO('{"command":"hello","payload":{}}\n'))
    monkeypatch.setattr(sys, "stdout", protocol_output)
    monkeypatch.setattr(sys, "stderr", framework_output)

    run_worker()

    assert json.loads(protocol_output.getvalue()) == {
        "ok": True,
        "result": {"protocol": "clean"},
    }
    assert framework_output.getvalue() == "framework log\n"
