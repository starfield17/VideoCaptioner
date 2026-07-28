import json
import shutil
from pathlib import Path

from captioner.llm.fake import FakeLlm
from captioner.media.api import FakeMediaService
from captioner.shared.errors import TranscriptionError
from captioner.subtitles.api import SubtitleService
from captioner.transcription.api import (
    AsrCapabilities,
    FakeTranscriptionService,
    FasterWhisperConfig,
    FasterWhisperTranscriptionService,
    TranscriptDocument,
    TranscriptionRequest,
)
from captioner.workflow.api import (
    PipelineOptions,
    PipelineServices,
    discover_inputs,
    run_files,
    transcribe_files,
)
from captioner.workflow.options import (
    CorrectionOptions,
    FasterWhisperAsrOptions,
    RunOptions,
    TranslationOptions,
)

ROOT = Path(__file__).resolve().parents[2]


def test_directory_discovery_accepts_media_and_sorts_inputs(tmp_path: Path) -> None:
    video_b = tmp_path / "b.mp4"
    video_a = tmp_path / "a.mov"
    ignored = tmp_path / "notes.txt"
    for path in (video_b, video_a, ignored):
        path.write_bytes(b"")

    assert discover_inputs(tmp_path, provider="faster-whisper") == (
        video_a,
        video_b,
    )


def test_transcribe_files_writes_transcript_json(tmp_path: Path) -> None:
    input_path = ROOT / "tests/fixtures/fake_input.json"
    options = PipelineOptions(run=RunOptions(keep_workdir=True))
    services = PipelineServices(
        media=FakeMediaService(),
        transcription=FakeTranscriptionService(),
        subtitles=SubtitleService(FakeLlm()),
    )

    result = transcribe_files((input_path,), options, services, tmp_path / "out")

    assert not result.failed
    assert len(result.succeeded) == 1
    output_path = result.succeeded[0].output_path
    assert output_path.name == "fake_input.transcript.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "transcript.v1"
    assert payload["timing_origin"] == "asr_native"
    assert payload["words"]


def test_serial_run_loads_once_and_continues_after_one_file_failure(
    tmp_path: Path,
) -> None:
    first_input = tmp_path / "first.json"
    second_input = tmp_path / "second.json"
    fixture = ROOT / "tests/fixtures/fake_input.json"
    shutil.copyfile(fixture, first_input)
    shutil.copyfile(fixture, second_input)

    asr_options = FasterWhisperAsrOptions(
        provider="faster-whisper",
        faster_whisper=FasterWhisperConfig(
            model="tiny", device="cpu", compute_type="int8"
        ),
    )
    options = PipelineOptions(
        asr=asr_options,
        correction=CorrectionOptions(enabled=False),
        translation=TranslationOptions(enabled=False),
        run=RunOptions(keep_workdir=True),
    )
    worker = _RecordingWorker()
    services = PipelineServices(
        media=FakeMediaService(),
        transcription=FasterWhisperTranscriptionService(
            asr_options.faster_whisper, worker
        ),
        subtitles=SubtitleService(FakeLlm()),
    )

    result = run_files((first_input, second_input), options, services, tmp_path / "out")

    assert [failure.input_path for failure in result.failed] == [first_input]
    assert [success.input_path for success in result.succeeded] == [second_input]
    assert worker.start_calls == 1
    assert worker.transcribe_calls == 2
    assert worker.shutdown_calls == 1
    assert (tmp_path / "out" / "second.srt").is_file()
    assert (tmp_path / "out" / "second.subtitle.json").is_file()


class _RecordingWorker:
    def __init__(self) -> None:
        self.start_calls = 0
        self.transcribe_calls = 0
        self.shutdown_calls = 0

    def start(self, config: FasterWhisperConfig) -> AsrCapabilities:
        assert config.model == "tiny"
        self.start_calls += 1
        return AsrCapabilities(
            native_word_timestamps=True,
            forced_alignment=False,
            language_detection=True,
            initial_prompt=True,
            internal_vad=True,
            supported_languages=None,
        )

    def transcribe(
        self, request: TranscriptionRequest, artifact_dir: Path
    ) -> TranscriptDocument:
        del request, artifact_dir
        self.transcribe_calls += 1
        if self.transcribe_calls == 1:
            raise TranscriptionError("synthetic first-file failure")
        return TranscriptDocument.model_validate(
            {
                "language": "en",
                "text": "ready.",
                "timing_origin": "asr_native",
                "words": [
                    {
                        "id": "w000001",
                        "text": "ready.",
                        "start_ms": 0,
                        "end_ms": 400,
                    }
                ],
                "segments": [
                    {
                        "id": "seg000001",
                        "text": "ready.",
                        "start_ms": 0,
                        "end_ms": 400,
                        "word_ids": ["w000001"],
                    }
                ],
                "provider": "faster-whisper",
                "model_name": "tiny",
            }
        )

    def shutdown(self) -> None:
        self.shutdown_calls += 1
