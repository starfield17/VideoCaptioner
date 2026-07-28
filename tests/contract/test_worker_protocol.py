import json
import shutil
from pathlib import Path

from captioner.transcription.models import TimingOrigin
from captioner.transcription.providers.worker_client import FakeWorkerClient
from captioner.transcription.requests import TranscriptionRequest

ROOT = Path(__file__).resolve().parents[2]


def test_fake_worker_lifecycle_and_artifact_contract(tmp_path: Path) -> None:
    fixture = ROOT / "tests" / "fixtures" / "fake_input.json"
    prepared_path = tmp_path / "prepared.fake.json"
    shutil.copyfile(fixture, prepared_path)
    client = FakeWorkerClient()

    capabilities = client.start()
    assert capabilities.native_word_timestamps
    document = client.transcribe(
        TranscriptionRequest(audio_path=prepared_path), tmp_path
    )
    assert document.timing_origin is TimingOrigin.ASR_NATIVE
    artifact_path = tmp_path / "transcript.raw.json"
    assert artifact_path.is_file()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "transcript.v1"
    client.shutdown()
