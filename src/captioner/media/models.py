"""Media boundary models."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class AudioAsset(BaseModel):
    """Prepared audio-like input passed to the ASR boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_path: Path
    path: Path
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
