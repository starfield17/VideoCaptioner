import wave
from pathlib import Path

from captioner.media.api import FfmpegMediaService


def test_ffmpeg_extracts_and_normalizes_audio(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    with wave.open(str(source_path), "wb") as source:
        source.setnchannels(2)
        source.setsampwidth(2)
        source.setframerate(8_000)
        source.writeframes(bytes(8_000 * 2 * 2 // 10))

    asset = FfmpegMediaService(sample_rate=16_000, channels=1).prepare_audio(
        source_path, tmp_path / "prepared"
    )

    assert asset.path.is_file()
    assert asset.sample_rate == 16_000
    assert asset.channels == 1
    with wave.open(str(asset.path), "rb") as prepared:
        assert prepared.getnchannels() == 1
        assert prepared.getsampwidth() == 2
        assert prepared.getframerate() == 16_000
