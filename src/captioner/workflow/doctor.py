"""Provider environment diagnostics."""

import shutil
import sys
from dataclasses import dataclass

from captioner.shared.errors import CaptionerError
from captioner.transcription.api import (
    FakeTranscriptionService,
    FasterWhisperConfig,
    FasterWhisperTranscriptionService,
    FasterWhisperWorkerClient,
    Qwen3Config,
    Qwen3TranscriptionService,
    Qwen3WorkerClient,
)
from captioner.transcription.requests import TimestampRequirement
from captioner.workflow.options import (
    FasterWhisperAsrOptions,
    PipelineOptions,
    Qwen3AsrOptions,
)


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
) -> DoctorReport:
    """Check the selected provider without downloading a model by default."""

    selected_provider = provider or options.asr.provider

    checks = {
        "python_3_13": sys.version_info[:2] == (3, 13),
        "conda": shutil.which("conda") is not None,
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "configuration": True,
        "provider_environment": False,
    }
    details = {
        "python_3_13": sys.version.split()[0],
        "conda": shutil.which("conda") or "not found",
        "ffmpeg": shutil.which("ffmpeg") or "not found",
        "configuration": "valid",
        "provider": selected_provider,
    }

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
    else:
        details["provider_environment"] = f"unsupported provider: {selected_provider}"
    return DoctorReport(checks=checks, details=details)
