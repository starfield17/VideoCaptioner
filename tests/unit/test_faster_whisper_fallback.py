from pathlib import Path

import pytest

from captioner.shared.errors import TranscriptionError
from captioner.transcription.capabilities import AsrCapabilities
from captioner.transcription.models import TimingOrigin, TranscriptDocument
from captioner.transcription.providers.faster_whisper import FasterWhisperConfig
from captioner.transcription.requests import TranscriptionRequest
from captioner.transcription.service import FasterWhisperTranscriptionService


class _LazyCudaFailureClient:
    def __init__(self) -> None:
        self.configs: list[FasterWhisperConfig] = []
        self.transcribe_calls = 0

    def start(self, config: FasterWhisperConfig) -> AsrCapabilities:
        self.configs.append(config)
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
            raise TranscriptionError("Library libcublas.so.12 cannot be loaded")
        return TranscriptDocument(
            provider="faster-whisper",
            model_name="turbo",
            language="en",
            text="test",
            timing_origin=TimingOrigin.ASR_NATIVE,
            segments=(),
            words=(),
        )

    def shutdown(self) -> None:
        pass


def test_auto_device_retries_lazy_cuda_failure_once_on_cpu(tmp_path: Path) -> None:
    client = _LazyCudaFailureClient()
    service = FasterWhisperTranscriptionService(
        FasterWhisperConfig(),
        client=client,
    )
    request = TranscriptionRequest(audio_path=tmp_path / "audio.wav")

    service.start()
    result = service.transcribe(request, tmp_path)

    assert result.provider == "faster-whisper"
    assert [config.device for config in client.configs] == ["auto", "cpu"]
    assert client.configs[-1].compute_type == "int8"
    assert client.transcribe_calls == 2


def test_explicit_cuda_does_not_fallback(tmp_path: Path) -> None:
    client = _LazyCudaFailureClient()
    service = FasterWhisperTranscriptionService(
        FasterWhisperConfig(device="cuda", compute_type="float16"),
        client=client,
    )
    service.start()

    with pytest.raises(TranscriptionError, match="libcublas"):
        service.transcribe(
            TranscriptionRequest(audio_path=tmp_path / "audio.wav"),
            tmp_path,
        )

    assert len(client.configs) == 1
