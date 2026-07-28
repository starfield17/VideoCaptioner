"""Resolve configured ASR models into the application-owned cache."""

from pathlib import Path

from captioner.models import ModelStore, download_with_dependencies
from captioner.workflow.options import (
    FasterWhisperAsrOptions,
    NemoAsrOptions,
    PipelineOptions,
    Qwen3AsrOptions,
)


def prepare_asr_model(options: PipelineOptions) -> PipelineOptions:
    """Download the selected model and return options pointing at local files."""

    root = options.models.cache_dir
    store = ModelStore(
        root,
        endpoint=options.models.endpoint,
        offline=options.models.offline,
    )
    if isinstance(options.asr, FasterWhisperAsrOptions):
        config = options.asr.faster_whisper
        if config.model_path is not None:
            return options
        legacy_path = Path(config.model)
        if legacy_path.is_dir():
            asr = options.asr.model_copy(
                update={
                    "faster_whisper": config.model_copy(
                        update={"model_path": legacy_path}
                    )
                }
            )
            return options.model_copy(update={"asr": asr})
        model_name = config.model.rsplit("/", maxsplit=1)[-1]
        model_name = model_name.removeprefix("faster-whisper-")
        key = f"faster-whisper-{model_name}"
        paths = download_with_dependencies(store, key)
        asr = options.asr.model_copy(
            update={
                "faster_whisper": config.model_copy(update={"model_path": paths[key]})
            }
        )
    elif isinstance(options.asr, Qwen3AsrOptions):
        config = options.asr.qwen3
        if config.model_path is not None:
            return options
        key = "qwen3-0.6b" if "0.6B" in config.model else "qwen3-1.7b"
        paths = download_with_dependencies(store, key)
        asr = options.asr.model_copy(
            update={
                "qwen3": config.model_copy(
                    update={
                        "model_path": paths[key],
                        "forced_aligner_path": paths.get("qwen3-forced-aligner-0.6b"),
                    }
                )
            }
        )
    elif isinstance(options.asr, NemoAsrOptions):
        config = options.asr.nemo
        if config.model_path is not None:
            return options
        key = "nemo-parakeet-110m-en" if "110m" in config.model else "nemo-parakeet-v3"
        paths = download_with_dependencies(store, key)
        asr = options.asr.model_copy(
            update={"nemo": config.model_copy(update={"model_path": paths[key]})}
        )
    else:
        return options
    return options.model_copy(update={"asr": asr})


__all__ = ["prepare_asr_model"]
