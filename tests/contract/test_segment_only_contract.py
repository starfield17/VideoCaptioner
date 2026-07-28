import shutil
from pathlib import Path

from captioner.transcription.models import TimingOrigin
from captioner.transcription.providers.worker_client import FakeWorkerClient
from captioner.transcription.requests import TranscriptionRequest

ROOT = Path(__file__).resolve().parents[2]


def test_segment_only_fixture_never_creates_timed_words(tmp_path: Path) -> None:
    prepared_path = tmp_path / "prepared.fake.json"
    shutil.copyfile(ROOT / "tests/fixtures/segment_only.json", prepared_path)
    client = FakeWorkerClient()
    client.start()
    document = client.transcribe(
        TranscriptionRequest(audio_path=prepared_path), tmp_path
    )
    client.shutdown()
    assert document.timing_origin is TimingOrigin.SEGMENT_NATIVE
    assert document.words == ()
    assert document.segments[0].word_ids == ()
