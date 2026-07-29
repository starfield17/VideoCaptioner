"""Application-owned ASR runtime environments and Worker command resolution."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast
from uuid import uuid4

from captioner.shared.app_paths import application_paths, bundled_executable
from captioner.shared.errors import (
    ConfigurationError,
    OperationCancelled,
    ProviderUnavailableError,
    RuntimeInstallationError,
)
from captioner.shared.worker_protocol import PROTOCOL_VERSION

RUNTIME_SCHEMA_VERSION = "captioner-runtime.v1"
RUNTIME_VERSION = "1"
RUNTIME_PROVIDERS = ("faster-whisper", "qwen3-asr", "nemo-asr")


class RuntimeStability(StrEnum):
    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class RuntimePlatform:
    os: str
    host_arch: str
    conda_platform: str

    @property
    def key(self) -> str:
        return f"{self.os}-{self.host_arch}"


@dataclass(frozen=True)
class RuntimeDescriptor:
    provider: str
    runtime_version: str
    protocol_version: str
    os: str
    host_arch: str
    process_arch: str
    conda_platform: str
    accelerator: str
    stability: RuntimeStability
    python_version: str
    conda_packages: tuple[str, ...]
    pip_packages: tuple[str, ...]
    worker_module: str
    available: bool
    reason: str

    @property
    def key(self) -> str:
        return f"{self.provider}-{self.os}-{self.host_arch}"

    @property
    def recipe_sha256(self) -> str:
        """Identify the declared package recipe stored in runtime manifests."""

        values = {
            "provider": self.provider,
            "runtime_version": self.runtime_version,
            "protocol_version": self.protocol_version,
            "conda_platform": self.conda_platform,
            "python_version": self.python_version,
            "conda_packages": self.conda_packages,
            "pip_packages": self.pip_packages,
            "worker_module": self.worker_module,
        }
        encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RuntimeStatus:
    descriptor: RuntimeDescriptor
    path: Path | None
    detail: str

    @property
    def installed(self) -> bool:
        return self.path is not None


CommandRunner = Callable[
    [Sequence[str], str | None, Path | None],
    subprocess.CompletedProcess[str],
]
Checkpoint = Callable[[], None]


def detect_runtime_platform(
    *, system: str | None = None, machine: str | None = None
) -> RuntimePlatform:
    """Normalize host names to the release/runtime platform vocabulary."""

    selected_system = (system or platform.system()).lower()
    selected_machine = (machine or platform.machine()).lower()
    architecture = {
        "amd64": "x86_64",
        "x86_64": "x86_64",
        "arm64": "aarch64",
        "aarch64": "aarch64",
    }.get(selected_machine)
    os_name = {
        "linux": "linux",
        "darwin": "macos",
        "windows": "windows",
    }.get(selected_system)
    if architecture is None or os_name is None:
        raise ConfigurationError(
            f"unsupported runtime platform: {selected_system}/{selected_machine}"
        )
    conda_platform = {
        ("linux", "x86_64"): "linux-64",
        ("linux", "aarch64"): "linux-aarch64",
        ("macos", "x86_64"): "osx-64",
        ("macos", "aarch64"): "osx-arm64",
        ("windows", "x86_64"): "win-64",
        ("windows", "aarch64"): "win-arm64",
    }[(os_name, architecture)]
    return RuntimePlatform(os_name, architecture, conda_platform)


def runtime_descriptor(
    provider: str,
    *,
    runtime_platform: RuntimePlatform | None = None,
) -> RuntimeDescriptor:
    """Return the deterministic support declaration for one provider."""

    if provider not in RUNTIME_PROVIDERS:
        raise ConfigurationError(f"managed runtime is unavailable for {provider}")
    selected = runtime_platform or detect_runtime_platform()
    process_arch = selected.host_arch
    conda_platform = selected.conda_platform
    stability = RuntimeStability.EXPERIMENTAL
    available = True
    reason = "Runtime recipe is experimental on this platform."
    accelerator = "cpu"

    if provider == "faster-whisper":
        stability = RuntimeStability.STABLE
        reason = "Portable CPU runtime."
        if selected.os in {"linux", "windows"} and selected.host_arch == "x86_64":
            accelerator = "cpu; optional host CUDA"
        if selected.os == "windows" and selected.host_arch == "aarch64":
            process_arch = "x86_64"
            conda_platform = "win-64"
            reason = "x64 Worker runs through Windows emulation."
        packages = ("faster-whisper==1.2.1",)
        module = "workers.faster_whisper"
    elif provider == "qwen3-asr":
        reason = "Best-effort Transformers runtime; model execution is experimental."
        if selected.os in {"linux", "windows"} and selected.host_arch == "x86_64":
            accelerator = "cpu; optional host CUDA"
        elif selected.os == "macos" and selected.host_arch == "aarch64":
            accelerator = "cpu; experimental MPS"
        if selected.os == "windows" and selected.host_arch == "aarch64":
            process_arch = "x86_64"
            conda_platform = "win-64"
            reason = "Experimental x64 Worker runs through Windows emulation."
        packages = ("torch>=2.6", "qwen-asr==0.0.6")
        module = "workers.qwen3"
    else:
        packages = ("nemo_toolkit[asr]==2.7.3",)
        module = "workers.nemo"
        if selected.os != "linux":
            available = False
            stability = RuntimeStability.UNAVAILABLE
            reason = "NeMo local ASR has no supported macOS or Windows recipe."
        else:
            accelerator = "cpu; optional host CUDA"
            reason = "Linux-only experimental NeMo runtime."

    common = (
        "pydantic>=2,<3",
        "huggingface-hub>=0.34,<2",
        "platformdirs>=4,<5",
        "tomli-w>=1,<2",
    )
    return RuntimeDescriptor(
        provider=provider,
        runtime_version=RUNTIME_VERSION,
        protocol_version=PROTOCOL_VERSION,
        os=selected.os,
        host_arch=selected.host_arch,
        process_arch=process_arch,
        conda_platform=conda_platform,
        accelerator=accelerator,
        stability=stability,
        python_version="3.13",
        conda_packages=("python=3.13", "pip"),
        pip_packages=(*common, *packages),
        worker_module=module,
        available=available,
        reason=reason,
    )


class RuntimeStore:
    """Install versioned micromamba prefixes and atomically select one."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        micromamba: Path | None = None,
        project_payload: Path | None = None,
        runner: CommandRunner | None = None,
        runtime_platform: RuntimePlatform | None = None,
    ) -> None:
        self.root = root or application_paths().runtime_dir
        self._micromamba = micromamba
        self._project_payload = project_payload
        self._runner = runner
        self._platform = runtime_platform or detect_runtime_platform()

    def list(self) -> tuple[RuntimeStatus, ...]:
        return tuple(self.status(provider) for provider in RUNTIME_PROVIDERS)

    def status(self, provider: str) -> RuntimeStatus:
        descriptor = runtime_descriptor(provider, runtime_platform=self._platform)
        if not descriptor.available:
            return RuntimeStatus(descriptor, None, descriptor.reason)
        pointer = self._provider_root(descriptor) / "current.json"
        try:
            raw = cast(
                dict[str, object],
                json.loads(pointer.read_text(encoding="utf-8")),
            )
            relative_path = str(raw["path"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return RuntimeStatus(descriptor, None, "Not installed.")
        target = self._provider_root(descriptor) / relative_path
        if not self._valid(target, descriptor):
            return RuntimeStatus(descriptor, None, "Installed runtime is incomplete.")
        return RuntimeStatus(descriptor, target, "Installed and ready.")

    def command(self, provider: str) -> tuple[str, ...] | None:
        status = self.status(provider)
        if status.path is None:
            return None
        micromamba = self._find_micromamba()
        return (
            *self._python_command(
                micromamba,
                status.path / "env",
                status.descriptor,
            ),
            "-m",
            status.descriptor.worker_module,
        )

    def install(
        self,
        provider: str,
        *,
        repair: bool = False,
        checkpoint: Checkpoint | None = None,
    ) -> RuntimeStatus:
        descriptor = runtime_descriptor(provider, runtime_platform=self._platform)
        if not descriptor.available:
            raise RuntimeInstallationError(descriptor.reason)
        current = self.status(provider)
        if current.installed and not repair:
            return current
        selected_checkpoint = checkpoint or (lambda: None)
        micromamba = self._find_micromamba()
        payload = self._find_project_payload()
        provider_root = self._provider_root(descriptor)
        provider_root.mkdir(parents=True, exist_ok=True)
        lock_path = provider_root / ".install.lock"
        self._acquire_lock(lock_path)
        target = provider_root / (f"{descriptor.runtime_version}-{uuid4().hex[:12]}")
        env_path = target / "env"
        old_path = current.path
        try:
            selected_checkpoint()
            target.mkdir()
            create = (
                str(micromamba),
                "create",
                "--yes",
                "--root-prefix",
                str(self.root / "_micromamba"),
                "--prefix",
                str(env_path),
                "--platform",
                descriptor.conda_platform,
                "--channel",
                "conda-forge",
                *descriptor.conda_packages,
            )
            self._checked(
                create,
                cwd=provider_root,
                checkpoint=selected_checkpoint,
            )
            selected_checkpoint()
            install = (
                *self._python_command(micromamba, env_path, descriptor),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                *descriptor.pip_packages,
                str(payload),
            )
            self._checked(
                install,
                cwd=provider_root,
                checkpoint=selected_checkpoint,
            )
            selected_checkpoint()
            self._probe(
                micromamba,
                env_path,
                descriptor,
                provider_root,
                selected_checkpoint,
            )
            manifest = {
                "schema_version": RUNTIME_SCHEMA_VERSION,
                **asdict(descriptor),
            }
            manifest["stability"] = descriptor.stability.value
            manifest["recipe_sha256"] = descriptor.recipe_sha256
            (target / "captioner-runtime.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self._write_pointer(provider_root, target.name)
            if repair and old_path is not None and old_path != target:
                shutil.rmtree(old_path, ignore_errors=True)
            return RuntimeStatus(descriptor, target, "Installed and ready.")
        except RuntimeInstallationError:
            shutil.rmtree(target, ignore_errors=True)
            raise
        except OperationCancelled:
            shutil.rmtree(target, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(target, ignore_errors=True)
            raise RuntimeInstallationError(
                f"could not install {provider} runtime: {exc}"
            ) from exc
        finally:
            lock_path.unlink(missing_ok=True)

    def remove(self, provider: str) -> RuntimeStatus:
        status = self.status(provider)
        provider_root = self._provider_root(status.descriptor)
        pointer = provider_root / "current.json"
        if status.path is not None:
            shutil.rmtree(status.path, ignore_errors=True)
        pointer.unlink(missing_ok=True)
        return self.status(provider)

    def _provider_root(self, descriptor: RuntimeDescriptor) -> Path:
        return (
            self.root
            / descriptor.provider
            / (f"{descriptor.os}-{descriptor.host_arch}")
        )

    def _find_micromamba(self) -> Path:
        if self._micromamba is not None and self._micromamba.is_file():
            return self._micromamba
        discovered = bundled_executable("micromamba")
        if discovered is None:
            raise RuntimeInstallationError(
                "micromamba is not bundled and was not found on PATH"
            )
        return Path(discovered)

    def _find_project_payload(self) -> Path:
        if self._project_payload is not None and self._project_payload.exists():
            return self._project_payload
        roots = (
            Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "runtime",
            Path(sys.executable).parent / "runtime",
        )
        for root in roots:
            wheels = tuple(root.glob("captioner-*.whl"))
            if len(wheels) == 1:
                return wheels[0]
        source_file = Path(__file__).resolve()
        for candidate in source_file.parents:
            if (candidate / "pyproject.toml").is_file() and (
                candidate / "workers"
            ).is_dir():
                return candidate
        raise RuntimeInstallationError("captioner Worker payload was not found")

    def _checked(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        checkpoint: Checkpoint,
    ) -> None:
        completed = self._execute(command, None, cwd, checkpoint)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeInstallationError(
                f"runtime command failed ({command[1]}): {detail[-2000:]}"
            )

    def _probe(
        self,
        micromamba: Path,
        env_path: Path,
        descriptor: RuntimeDescriptor,
        cwd: Path,
        checkpoint: Checkpoint,
    ) -> None:
        command = (
            *self._python_command(micromamba, env_path, descriptor),
            "-m",
            descriptor.worker_module,
        )
        input_text = (
            json.dumps({"command": "hello", "payload": {}})
            + "\n"
            + json.dumps({"command": "shutdown", "payload": {}})
            + "\n"
        )
        completed = self._execute(command, input_text, cwd, checkpoint)
        if completed.returncode != 0:
            raise RuntimeInstallationError(
                f"Worker probe failed: {(completed.stderr or '').strip()[-2000:]}"
            )
        try:
            response = json.loads(completed.stdout.splitlines()[0])
            result = response["result"]
        except (IndexError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeInstallationError(
                "Worker probe returned invalid JSON"
            ) from exc
        if (
            response.get("ok") is not True
            or result.get("provider_id") != descriptor.provider
            or result.get("protocol_version") != PROTOCOL_VERSION
        ):
            raise RuntimeInstallationError("Worker probe identity mismatch")

    @staticmethod
    def _valid(path: Path, descriptor: RuntimeDescriptor) -> bool:
        manifest_path = path / "captioner-runtime.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        python_path = RuntimeStore._python_executable(path / "env", descriptor)
        return bool(
            python_path.is_file()
            and manifest.get("schema_version") == RUNTIME_SCHEMA_VERSION
            and manifest.get("provider") == descriptor.provider
            and manifest.get("protocol_version") == PROTOCOL_VERSION
            and manifest.get("recipe_sha256") == descriptor.recipe_sha256
        )

    @staticmethod
    def _python_executable(
        env_path: Path,
        descriptor: RuntimeDescriptor,
    ) -> Path:
        if descriptor.conda_platform.startswith("win-"):
            return env_path / "python.exe"
        return env_path / "bin" / "python"

    def _python_command(
        self,
        micromamba: Path,
        env_path: Path,
        descriptor: RuntimeDescriptor,
    ) -> tuple[str, ...]:
        python = self._python_executable(env_path, descriptor)
        if descriptor.conda_platform.startswith("win-"):
            return (str(python),)
        return (
            str(micromamba),
            "run",
            "--root-prefix",
            str(self.root / "_micromamba"),
            "--prefix",
            str(env_path),
            str(python),
        )

    @staticmethod
    def _write_pointer(parent: Path, target_name: str) -> None:
        temporary = parent / ".current.json.tmp"
        temporary.write_text(
            json.dumps({"path": target_name}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(parent / "current.json")

    @staticmethod
    def _acquire_lock(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeInstallationError(
                "another runtime installation is already running"
            ) from exc
        else:
            os.close(descriptor)

    def _execute(
        self,
        command: Sequence[str],
        input_text: str | None,
        cwd: Path | None,
        checkpoint: Checkpoint,
    ) -> subprocess.CompletedProcess[str]:
        if self._runner is not None:
            return self._runner(command, input_text, cwd)
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        with (
            tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout,
            tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr,
        ):
            process = subprocess.Popen(
                tuple(command),
                cwd=cwd,
                env=environment,
                stdin=subprocess.PIPE if input_text is not None else None,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
            if input_text is not None and process.stdin is not None:
                process.stdin.write(input_text)
                process.stdin.close()
            try:
                while process.poll() is None:
                    checkpoint()
                    try:
                        process.wait(timeout=0.25)
                    except subprocess.TimeoutExpired:
                        continue
            except OperationCancelled:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                raise
            stdout.seek(0)
            stderr.seek(0)
            return subprocess.CompletedProcess(
                tuple(command),
                process.returncode,
                stdout.read(),
                stderr.read(),
            )


def resolve_worker_command(
    provider: str,
    *,
    environment_name: str,
    worker_module: str,
) -> tuple[str, ...]:
    """Prefer a managed runtime, retaining Conda only for source development."""

    managed = RuntimeStore().command(provider)
    if managed is not None:
        return managed
    conda = shutil.which("conda")
    if conda is not None:
        return (
            conda,
            "run",
            "--no-capture-output",
            "-n",
            environment_name,
            "python",
            "-m",
            worker_module,
        )
    status = RuntimeStore().status(provider)
    if not status.descriptor.available:
        raise ProviderUnavailableError(status.descriptor.reason)
    raise ProviderUnavailableError(
        f"{provider} runtime is not installed; run "
        f"'captioner runtimes install {provider}'"
    )


__all__ = [
    "RUNTIME_PROVIDERS",
    "RUNTIME_SCHEMA_VERSION",
    "RuntimeDescriptor",
    "RuntimePlatform",
    "RuntimeStability",
    "RuntimeStatus",
    "RuntimeStore",
    "detect_runtime_platform",
    "resolve_worker_command",
    "runtime_descriptor",
]
