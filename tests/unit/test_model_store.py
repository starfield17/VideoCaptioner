from pathlib import Path

import pytest
from pytest import MonkeyPatch

from captioner.models import ModelStore
from captioner.shared.errors import ConfigurationError


class _Info:
    sha = "resolved-revision"


class _Api:
    calls = 0

    def __init__(self, *, endpoint: str) -> None:
        assert endpoint == "https://mirror.example"

    def model_info(self, repository: str, revision: str | None = None) -> _Info:
        del revision
        type(self).calls += 1
        assert repository == "Systran/faster-whisper-small"
        return _Info()


def _snapshot_download(**values: object) -> str:
    assert values["endpoint"] == "https://mirror.example"
    local_dir = Path(str(values["local_dir"]))
    local_dir.mkdir(parents=True)
    (local_dir / "model.bin").write_bytes(b"model")
    return str(local_dir)


def test_model_store_rejects_insecure_remote_endpoint(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        ModelStore(tmp_path, endpoint="http://mirror.example")


def test_model_store_downloads_atomically_and_reuses_current(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _Api.calls = 0
    monkeypatch.setattr("captioner.models.HfApi", _Api)
    monkeypatch.setattr("captioner.models.snapshot_download", _snapshot_download)
    store = ModelStore(tmp_path, endpoint="https://mirror.example")

    first = store.download("faster-whisper-small")
    second = store.download("faster-whisper-small")

    assert first == second
    assert (first / "model.bin").read_bytes() == b"model"
    assert (first / "captioner-model.json").is_file()
    assert not tuple((tmp_path / "faster-whisper-small").glob(".download-*"))
    assert _Api.calls == 1
