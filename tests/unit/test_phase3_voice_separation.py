import sys
from pathlib import Path

from captioner.llm.fake import FakeLlm
from captioner.media.api import FakeMediaService
from captioner.media.models import AudioAsset
from captioner.media.voice_separation import (
    CommandVoiceSeparator,
    VoiceSeparationError,
)
from captioner.subtitles.api import SubtitleService
from captioner.transcription.api import (
    AsrCapabilities,
    FakeTranscriptionService,
    TranscriptDocument,
    TranscriptionRequest,
)
from captioner.workflow.api import PipelineOptions, PipelineServices, run_files

ROOT = Path(__file__).resolve().parents[2]


class _FailingSeparator:
    def separate(self, audio: AudioAsset, output_path: Path) -> AudioAsset:
        del audio, output_path
        raise VoiceSeparationError("synthetic separation failure")


class _SuccessfulSeparator:
    def separate(self, audio: AudioAsset, output_path: Path) -> AudioAsset:
        output_path.write_bytes(audio.path.read_bytes())
        return audio.model_copy(update={"path": output_path})


class _RecordingTranscription:
    def __init__(self) -> None:
        self._inner = FakeTranscriptionService()
        self.audio_paths: list[Path] = []

    def start(self, model_name: str = "fake-v1") -> AsrCapabilities:
        return self._inner.start(model_name)

    def transcribe(
        self, request: TranscriptionRequest, artifact_dir: Path
    ) -> TranscriptDocument:
        self.audio_paths.append(request.audio_path)
        return self._inner.transcribe(request, artifact_dir)

    def shutdown(self) -> None:
        self._inner.shutdown()


def _services(
    transcription: _RecordingTranscription,
    separator: _FailingSeparator | _SuccessfulSeparator,
) -> PipelineServices:
    return PipelineServices(
        media=FakeMediaService(),
        transcription=transcription,
        subtitles=SubtitleService(FakeLlm()),
        voice_separator=separator,
    )


def test_optional_voice_separation_falls_back_to_original_audio(tmp_path: Path) -> None:
    transcription = _RecordingTranscription()
    options = PipelineOptions.model_validate(
        {
            "run": {"keep_workdir": True},
            "audio": {"voice_separation": {"enabled": True, "required": False}},
        }
    )

    result = run_files(
        (ROOT / "tests/fixtures/fake_input.json",),
        options,
        _services(transcription, _FailingSeparator()),
        tmp_path,
    )

    assert not result.failed
    assert result.succeeded[0].warnings == (
        "voice_separation_fallback:VoiceSeparationError",
    )
    assert transcription.audio_paths[0].name == "prepared.fake.json"


def test_required_voice_separation_fails_the_current_file(tmp_path: Path) -> None:
    transcription = _RecordingTranscription()
    options = PipelineOptions.model_validate(
        {
            "run": {"keep_workdir": True},
            "audio": {"voice_separation": {"enabled": True, "required": True}},
        }
    )

    result = run_files(
        (ROOT / "tests/fixtures/fake_input.json",),
        options,
        _services(transcription, _FailingSeparator()),
        tmp_path,
    )

    assert not result.succeeded
    assert result.failed[0].error_type == "VoiceSeparationError"
    assert transcription.audio_paths == []


def test_successful_voice_separation_is_the_audio_sent_to_asr(tmp_path: Path) -> None:
    transcription = _RecordingTranscription()
    options = PipelineOptions.model_validate(
        {
            "run": {"keep_workdir": True},
            "audio": {"voice_separation": {"enabled": True}},
        }
    )

    result = run_files(
        (ROOT / "tests/fixtures/fake_input.json",),
        options,
        _services(transcription, _SuccessfulSeparator()),
        tmp_path,
    )

    assert not result.failed
    assert transcription.audio_paths[0].name == "vocals.wav"


def test_command_voice_separator_writes_a_new_asset(tmp_path: Path) -> None:
    source = tmp_path / "prepared.wav"
    source.write_bytes(b"audio")
    output = tmp_path / "vocals.wav"
    separator = CommandVoiceSeparator(
        command=(
            sys.executable,
            "-c",
            "import shutil, sys; shutil.copyfile(sys.argv[1], sys.argv[2])",
        )
    )

    asset = separator.separate(
        AudioAsset(
            source_path=source,
            path=source,
            sample_rate=16_000,
            channels=1,
        ),
        output,
    )

    assert asset.path == output
    assert output.read_bytes() == b"audio"
