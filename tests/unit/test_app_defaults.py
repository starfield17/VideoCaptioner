import json
import logging
from pathlib import Path
from typing import cast

from platformdirs import PlatformDirs

from captioner.cli.main import main
from captioner.shared.app_paths import ApplicationPaths
from captioner.shared.logging import LoggingOptions, configure_logging, log_extra
from captioner.workflow.options import (
    FasterWhisperAsrOptions,
    PipelineOptions,
    with_asr_profile,
)


class _Directories:
    user_config_dir = "/test/config"
    user_cache_dir = "/test/cache"
    user_log_dir = "/test/log"
    user_data_dir = "/test/data"


def test_application_paths_keep_models_out_of_shared_sdk_cache() -> None:
    paths = ApplicationPaths.resolve(dirs=cast(PlatformDirs, _Directories()))

    assert paths.config_file == Path("/test/config/config.toml")
    assert paths.model_dir == Path("/test/cache/models")
    assert paths.log_dir == Path("/test/log")
    assert paths.runtime_dir == Path("/test/data/runtimes")


def test_built_in_default_is_quantized_turbo_on_auto_device() -> None:
    options = PipelineOptions()

    assert isinstance(options.asr, FasterWhisperAsrOptions)
    assert options.asr.language == "auto"
    assert options.asr.faster_whisper.model == "turbo"
    assert options.asr.faster_whisper.device == "auto"
    assert options.asr.faster_whisper.compute_type == "auto-int8"
    assert options.translation.batch_size == 30
    assert options.translation.parallelism == 16


def test_asr_profile_changes_only_asr_settings() -> None:
    original = PipelineOptions()
    selected = with_asr_profile(original, "nemo-parakeet-110m-en")

    assert selected.asr.provider == "nemo-asr"
    assert selected.asr.language == "en"
    assert selected.translation == original.translation


def test_config_init_round_trips_and_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"

    assert main(("config", "init", "--path", str(path))) == 0
    assert main(("config", "init", "--path", str(path))) == 2
    text = path.read_text(encoding="utf-8")
    assert 'model = "turbo"' in text


def test_json_log_levels_rotation_and_secret_redaction(tmp_path: Path) -> None:
    secret = "local-test-secret"
    log_path = configure_logging(
        LoggingOptions(
            level="ALL",
            console=False,
            directory=tmp_path,
            max_bytes=300,
            backup_count=2,
        ),
        secrets=(secret,),
    )
    logger = logging.getLogger("captioner.test")
    for index in range(20):
        logger.warning(
            "event %s secret=%s",
            index,
            secret,
            extra=log_extra(stage="test", index=index, credential=secret),
        )

    assert log_path is not None
    assert log_path == tmp_path / "captioner.log"
    files = tuple(tmp_path.glob("captioner.log*"))
    assert len(files) == 3
    combined = "".join(path.read_text(encoding="utf-8") for path in files)
    assert secret not in combined
    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["level"] == "WARNING"
    assert record["stage"] == "test"
