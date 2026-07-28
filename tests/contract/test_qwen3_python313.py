import sys
from pathlib import Path

import pytest
from workers.qwen3 import worker as qwen3_worker

from captioner.shared.errors import ProviderUnavailableError


def test_qwen3_environment_declares_project_python_version() -> None:
    environment = (Path(__file__).parents[2] / "conda" / "asr-qwen3.yml").read_text(
        encoding="utf-8"
    )
    assert "python=3.13" in environment
    assert "qwen-asr" in environment
    assert sys.version_info[:2] == (3, 13)


def test_qwen3_import_failure_keeps_python_reason_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(_name: str) -> object:
        raise ImportError("simulated package incompatibility")

    monkeypatch.setattr(qwen3_worker.importlib, "import_module", fail_import)

    with pytest.raises(
        ProviderUnavailableError,
        match=(
            r"qwen-asr import failed under Python 3\.13(?:\.\d+)?: "
            r"simulated package incompatibility"
        ),
    ):
        qwen3_worker.Qwen3Worker().hello()
