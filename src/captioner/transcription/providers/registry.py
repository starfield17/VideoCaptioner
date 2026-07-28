"""Static provider registry; dynamic plugin discovery is intentionally absent."""

from dataclasses import dataclass

from captioner.transcription.capabilities import AsrCapabilities


@dataclass(frozen=True)
class WorkerSpec:
    """Minimal static description of a worker provider."""

    provider_id: str
    capabilities: AsrCapabilities


PROVIDERS: dict[str, WorkerSpec] = {
    "fake": WorkerSpec(
        provider_id="fake",
        capabilities=AsrCapabilities(
            native_word_timestamps=True,
            forced_alignment=False,
            language_detection=False,
            initial_prompt=True,
            internal_vad=False,
            supported_languages=("en", "zh"),
        ),
    ),
    "faster-whisper": WorkerSpec(
        provider_id="faster-whisper",
        capabilities=AsrCapabilities(
            native_word_timestamps=True,
            forced_alignment=False,
            language_detection=True,
            initial_prompt=True,
            internal_vad=True,
            supported_languages=None,
        ),
    ),
    "qwen3-asr": WorkerSpec(
        provider_id="qwen3-asr",
        capabilities=AsrCapabilities(
            native_word_timestamps=False,
            forced_alignment=True,
            language_detection=True,
            initial_prompt=True,
            internal_vad=False,
            supported_languages=None,
        ),
    ),
    "nemo-asr": WorkerSpec(
        provider_id="nemo-asr",
        capabilities=AsrCapabilities(
            native_word_timestamps=True,
            forced_alignment=False,
            language_detection=True,
            initial_prompt=False,
            internal_vad=False,
            supported_languages=(
                "bg",
                "cs",
                "da",
                "de",
                "el",
                "en",
                "es",
                "et",
                "fi",
                "fr",
                "hr",
                "hu",
                "it",
                "lt",
                "lv",
                "mt",
                "nl",
                "pl",
                "pt",
                "ro",
                "ru",
                "sk",
                "sl",
                "sv",
                "uk",
            ),
        ),
    ),
}
