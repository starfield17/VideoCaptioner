import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from captioner.shared.errors import OperationCancelled, RuntimeInstallationError
from captioner.shared.runtimes import (
    RuntimePlatform,
    RuntimeStability,
    RuntimeStore,
    detect_runtime_platform,
    runtime_descriptor,
)


def test_platform_normalization_and_windows_arm_worker_emulation() -> None:
    selected = detect_runtime_platform(system="Windows", machine="ARM64")
    faster = runtime_descriptor("faster-whisper", runtime_platform=selected)

    assert selected.key == "windows-aarch64"
    assert selected.conda_platform == "win-arm64"
    assert faster.process_arch == "x86_64"
    assert faster.conda_platform == "win-64"
    assert faster.stability is RuntimeStability.STABLE


def test_nemo_is_disabled_with_reason_outside_linux() -> None:
    selected = detect_runtime_platform(system="Darwin", machine="arm64")
    descriptor = runtime_descriptor("nemo-asr", runtime_platform=selected)

    assert not descriptor.available
    assert descriptor.stability is RuntimeStability.UNAVAILABLE
    assert "macOS" in descriptor.reason


def test_runtime_install_updates_pointer_only_after_probe(tmp_path: Path) -> None:
    micromamba = tmp_path / "micromamba"
    micromamba.touch()
    payload = tmp_path / "captioner.whl"
    payload.touch()
    calls: list[tuple[str, ...]] = []

    def runner(
        command: Sequence[str],
        input_text: str | None,
        cwd: Path | None,
    ) -> subprocess.CompletedProcess[str]:
        del cwd
        values = tuple(command)
        calls.append(values)
        if "create" in values:
            prefix = Path(values[values.index("--prefix") + 1])
            (prefix / "bin").mkdir(parents=True)
            (prefix / "bin/python").touch()
        if input_text is not None:
            output = (
                json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "provider_id": "faster-whisper",
                            "protocol_version": "asr-worker.v1",
                        },
                    }
                )
                + "\n"
            )
        else:
            output = ""
        return subprocess.CompletedProcess(values, 0, output, "")

    store = RuntimeStore(
        tmp_path / "runtimes",
        micromamba=micromamba,
        project_payload=payload,
        runner=runner,
        runtime_platform=RuntimePlatform("linux", "x86_64", "linux-64"),
    )
    status = store.install("faster-whisper")

    assert status.installed
    assert status.path is not None
    assert store.status("faster-whisper").path == status.path
    manifest = json.loads(
        (status.path / "captioner-runtime.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "captioner-runtime.v1"
    assert manifest["recipe_sha256"] == status.descriptor.recipe_sha256
    assert any("pip" in call for call in calls)
    assert any(str(status.path / "env/bin/python") in call for call in calls)
    assert store.command("faster-whisper") is not None


def test_windows_runtime_command_uses_prefix_python(tmp_path: Path) -> None:
    platform = RuntimePlatform("windows", "x86_64", "win-64")
    descriptor = runtime_descriptor("faster-whisper", runtime_platform=platform)
    micromamba = tmp_path / "micromamba.exe"
    micromamba.touch()
    target = tmp_path / "faster-whisper/windows-x86_64/1-fixture"
    (target / "env").mkdir(parents=True)
    (target / "env/python.exe").touch()
    (target / "captioner-runtime.json").write_text(
        json.dumps(
            {
                "schema_version": "captioner-runtime.v1",
                "provider": descriptor.provider,
                "protocol_version": descriptor.protocol_version,
                "recipe_sha256": descriptor.recipe_sha256,
            }
        ),
        encoding="utf-8",
    )
    (target.parent / "current.json").write_text(
        '{"path":"1-fixture"}', encoding="utf-8"
    )
    store = RuntimeStore(
        tmp_path,
        micromamba=micromamba,
        runtime_platform=platform,
    )

    command = store.command("faster-whisper")

    assert command == (
        str(target / "env/python.exe"),
        "-m",
        descriptor.worker_module,
    )


def test_failed_probe_keeps_existing_runtime_and_removes_staging(
    tmp_path: Path,
) -> None:
    micromamba = tmp_path / "micromamba"
    micromamba.touch()
    payload = tmp_path / "captioner.whl"
    payload.touch()
    successful = True

    def runner(
        command: Sequence[str],
        input_text: str | None,
        cwd: Path | None,
    ) -> subprocess.CompletedProcess[str]:
        del cwd
        values = tuple(command)
        if "create" in values:
            prefix = Path(values[values.index("--prefix") + 1])
            (prefix / "bin").mkdir(parents=True)
            (prefix / "bin/python").touch()
        if input_text is not None and successful:
            output = (
                '{"ok":true,"result":{"provider_id":"faster-whisper",'
                '"protocol_version":"asr-worker.v1"}}\n'
            )
            return subprocess.CompletedProcess(values, 0, output, "")
        if input_text is not None:
            return subprocess.CompletedProcess(values, 1, "", "broken probe")
        return subprocess.CompletedProcess(values, 0, "", "")

    store = RuntimeStore(
        tmp_path / "runtimes",
        micromamba=micromamba,
        project_payload=payload,
        runner=runner,
        runtime_platform=RuntimePlatform("linux", "x86_64", "linux-64"),
    )
    first = store.install("faster-whisper")
    successful = False

    with pytest.raises(RuntimeInstallationError, match="probe failed"):
        store.install("faster-whisper", repair=True)

    assert store.status("faster-whisper").path == first.path
    provider_root = tmp_path / "runtimes/faster-whisper/linux-x86_64"
    assert len(tuple(provider_root.glob("1-*"))) == 1


def test_remove_deletes_only_selected_runtime(tmp_path: Path) -> None:
    platform = RuntimePlatform("linux", "x86_64", "linux-64")
    descriptor = runtime_descriptor("faster-whisper", runtime_platform=platform)
    target = tmp_path / "faster-whisper/linux-x86_64/1-fixture"
    (target / "env/bin").mkdir(parents=True)
    (target / "env/bin/python").touch()
    (target / "captioner-runtime.json").write_text(
        json.dumps(
            {
                "schema_version": "captioner-runtime.v1",
                "provider": descriptor.provider,
                "protocol_version": descriptor.protocol_version,
                "recipe_sha256": descriptor.recipe_sha256,
            }
        ),
        encoding="utf-8",
    )
    (target.parent / "current.json").write_text(
        '{"path":"1-fixture"}', encoding="utf-8"
    )
    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("keep", encoding="utf-8")
    store = RuntimeStore(tmp_path, runtime_platform=platform)

    status = store.remove("faster-whisper")

    assert not status.installed
    assert not target.exists()
    assert unrelated.is_file()


def test_cancelled_install_cleans_lock_and_preserves_error_type(
    tmp_path: Path,
) -> None:
    micromamba = tmp_path / "micromamba"
    micromamba.touch()
    payload = tmp_path / "captioner.whl"
    payload.touch()
    store = RuntimeStore(
        tmp_path / "runtimes",
        micromamba=micromamba,
        project_payload=payload,
        runtime_platform=RuntimePlatform("linux", "x86_64", "linux-64"),
    )

    def cancel() -> None:
        raise OperationCancelled("cancelled in test")

    with pytest.raises(OperationCancelled, match="cancelled in test"):
        store.install("faster-whisper", checkpoint=cancel)

    provider_root = tmp_path / "runtimes/faster-whisper/linux-x86_64"
    assert not (provider_root / ".install.lock").exists()
    assert not tuple(provider_root.glob("1-*"))
