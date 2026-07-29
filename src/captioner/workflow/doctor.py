"""Provider environment diagnostics."""

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from captioner.shared.app_paths import bundled_executable
from captioner.shared.errors import CaptionerError
from captioner.shared.runtimes import RuntimeStore
from captioner.transcription.api import (
    FakeTranscriptionService,
    FasterWhisperConfig,
    FasterWhisperTranscriptionService,
    FasterWhisperWorkerClient,
    NemoConfig,
    NemoTranscriptionService,
    NemoWorkerClient,
    Qwen3Config,
    Qwen3TranscriptionService,
    Qwen3WorkerClient,
)
from captioner.transcription.requests import TimestampRequirement
from captioner.workflow.options import (
    FasterWhisperAsrOptions,
    NemoAsrOptions,
    PipelineOptions,
    Qwen3AsrOptions,
)
from captioner.workflow.pipeline import build_subtitle_service


@dataclass(frozen=True)
class DoctorReport:
    """Machine-readable doctor result."""

    checks: dict[str, bool]
    details: dict[str, str]

    @property
    def ok(self) -> bool:
        return all(self.checks.values())


def run_doctor(
    options: PipelineOptions,
    provider: str | None = None,
    load_model: bool = False,
    output_dir: Path | None = None,
) -> DoctorReport:
    """Check the selected provider without downloading a model by default."""

    selected_provider = provider or options.asr.provider
    ffmpeg = bundled_executable("ffmpeg")

    checks = {
        "python_3_13": sys.version_info[:2] == (3, 13),
        "ffmpeg": ffmpeg is not None,
        "output_directory": False,
        "gpu_cuda": True,
        "configuration": True,
        "provider_environment": False,
    }
    details = {
        "python_3_13": sys.version.split()[0],
        "conda": shutil.which("conda") or "not found; managed runtimes are supported",
        "ffmpeg": ffmpeg or "not found",
        "configuration": "valid",
        "provider": selected_provider,
    }
    if selected_provider != "fake":
        runtime = RuntimeStore().status(selected_provider)
        details["managed_runtime"] = (
            str(runtime.path) if runtime.path is not None else runtime.detail
        )
    selected_output = output_dir or Path.cwd()
    try:
        selected_output.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=selected_output):
            pass
        checks["output_directory"] = True
        details["output_directory"] = f"writable: {selected_output}"
    except OSError as exc:
        details["output_directory"] = str(exc)

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        details["gpu_cuda"] = "nvidia-smi not found; CPU execution remains available"
    else:
        result = subprocess.run(
            (nvidia_smi, "--query-gpu=name,driver_version", "--format=csv,noheader"),
            check=False,
            capture_output=True,
            text=True,
        )
        checks["gpu_cuda"] = result.returncode == 0
        details["gpu_cuda"] = (
            result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
        )

    if options.audio.voice_separation.enabled:
        command = os.getenv(options.audio.voice_separation.command_env, "")
        checks["voice_separation_command"] = bool(command)
        details["voice_separation_command"] = (
            "configured" if command else "not configured"
        )

    if options.llm.provider == "openai-compatible":
        checks["llm_api"] = False
        try:
            build_subtitle_service(options).analyze_context(
                "Doctor connectivity probe."
            )
            checks["llm_api"] = True
            details["llm_api"] = "structured context probe succeeded"
        except CaptionerError as exc:
            details["llm_api"] = f"{type(exc).__name__}: {exc}"

    if selected_provider == "fake":
        worker = FakeTranscriptionService()
        try:
            capabilities = worker.start()
            checks["provider_environment"] = True
            details["provider_environment"] = (
                "provider=fake, "
                f"native_word_timestamps={capabilities.native_word_timestamps}"
            )
        except CaptionerError as exc:
            details["provider_environment"] = str(exc)
        finally:
            worker.shutdown()
    elif selected_provider == "faster-whisper":
        config = (
            options.asr.faster_whisper
            if isinstance(options.asr, FasterWhisperAsrOptions)
            else FasterWhisperConfig()
        )
        client = FasterWhisperWorkerClient()
        try:
            capabilities = client.probe()
            checks["provider_environment"] = True
            details["provider_environment"] = (
                "provider=faster-whisper, "
                f"native_word_timestamps={capabilities.native_word_timestamps}"
            )
        except CaptionerError as exc:
            details["provider_environment"] = str(exc)
        if load_model and checks["provider_environment"]:
            checks["model_load"] = False
            load_client = FasterWhisperWorkerClient()
            service = FasterWhisperTranscriptionService(config, load_client)
            try:
                service.start()
                checks["model_load"] = True
                details["model_load"] = f"loaded={config.model}"
            except CaptionerError as exc:
                details["model_load"] = str(exc)
            finally:
                service.shutdown()
    elif selected_provider == "qwen3-asr":
        config = (
            options.asr.qwen3
            if isinstance(options.asr, Qwen3AsrOptions)
            else Qwen3Config()
        )
        client = Qwen3WorkerClient()
        try:
            capabilities = client.probe()
            checks["provider_environment"] = True
            details["provider_environment"] = (
                "provider=qwen3-asr, "
                f"forced_alignment={capabilities.forced_alignment}, "
                f"native_word_timestamps={capabilities.native_word_timestamps}"
            )
        except CaptionerError as exc:
            details["provider_environment"] = str(exc)
        if load_model and checks["provider_environment"]:
            checks["model_load"] = False
            load_client = Qwen3WorkerClient()
            timestamps = (
                options.asr.timestamps
                if isinstance(options.asr, Qwen3AsrOptions)
                else TimestampRequirement.REQUIRED
            )
            service = Qwen3TranscriptionService(
                config,
                timestamps=timestamps,
                client=load_client,
            )
            try:
                service.start()
                checks["model_load"] = True
                details["model_load"] = f"loaded={config.model}"
            except CaptionerError as exc:
                details["model_load"] = str(exc)
            finally:
                service.shutdown()
    elif selected_provider == "nemo-asr":
        config = (
            options.asr.nemo
            if isinstance(options.asr, NemoAsrOptions)
            else NemoConfig()
        )
        client = NemoWorkerClient()
        try:
            capabilities = client.probe()
            checks["provider_environment"] = True
            details["provider_environment"] = (
                "provider=nemo-asr, "
                f"native_word_timestamps={capabilities.native_word_timestamps}, "
                f"language_detection={capabilities.language_detection}"
            )
        except CaptionerError as exc:
            details["provider_environment"] = str(exc)
        if load_model and checks["provider_environment"]:
            checks["model_load"] = False
            load_client = NemoWorkerClient()
            service = NemoTranscriptionService(config, load_client)
            try:
                service.start()
                checks["model_load"] = True
                details["model_load"] = f"loaded={config.model}"
            except CaptionerError as exc:
                details["model_load"] = str(exc)
            finally:
                service.shutdown()
    else:
        details["provider_environment"] = f"unsupported provider: {selected_provider}"
    return DoctorReport(checks=checks, details=details)
