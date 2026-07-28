"""Curated ASR model catalog and application-owned downloads."""

# pyright: reportUnknownVariableType=false

import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from captioner.shared.app_paths import application_paths
from captioner.shared.errors import ConfigurationError


@dataclass(frozen=True)
class ModelDescriptor:
    key: str
    provider: str
    repository: str
    note: str
    required_files: tuple[str, ...]
    dependencies: tuple[str, ...] = ()


MODEL_CATALOG = (
    ModelDescriptor(
        "faster-whisper-turbo",
        "faster-whisper",
        "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "Default: SDK-native INT8; faster than large-v3 with small accuracy loss.",
        ("model.bin",),
    ),
    ModelDescriptor(
        "faster-whisper-small",
        "faster-whisper",
        "Systran/faster-whisper-small",
        "Smaller opt-in multilingual model; SDK-native INT8.",
        ("model.bin",),
    ),
    ModelDescriptor(
        "faster-whisper-large-v2",
        "faster-whisper",
        "Systran/faster-whisper-large-v2",
        "Full-size comparison model; not selected by default.",
        ("model.bin",),
    ),
    ModelDescriptor(
        "faster-whisper-large-v3",
        "faster-whisper",
        "Systran/faster-whisper-large-v3",
        "Full-size comparison model; not selected by default.",
        ("model.bin",),
    ),
    ModelDescriptor(
        "qwen3-0.6b",
        "qwen3-asr",
        "Qwen/Qwen3-ASR-0.6B",
        "Official smaller model; published average WER is worse than 1.7B.",
        ("config.json", "*.safetensors"),
        ("qwen3-forced-aligner-0.6b",),
    ),
    ModelDescriptor(
        "qwen3-1.7b",
        "qwen3-asr",
        "Qwen/Qwen3-ASR-1.7B",
        "Official full-precision model; no official quantized checkpoint is available.",
        ("config.json", "*.safetensors"),
        ("qwen3-forced-aligner-0.6b",),
    ),
    ModelDescriptor(
        "qwen3-forced-aligner-0.6b",
        "qwen3-asr",
        "Qwen/Qwen3-ForcedAligner-0.6B",
        "Timestamp dependency for Qwen3 ASR.",
        ("config.json", "*.safetensors"),
    ),
    ModelDescriptor(
        "nemo-parakeet-v3",
        "nemo-asr",
        "nvidia/parakeet-tdt-0.6b-v3",
        "Official multilingual checkpoint; no official quantized release.",
        ("*.nemo",),
    ),
    ModelDescriptor(
        "nemo-parakeet-110m-en",
        "nemo-asr",
        "nvidia/parakeet-tdt_ctc-110m",
        "Smaller English model; published WER exceeds the default threshold.",
        ("*.nemo",),
    ),
)
_BY_KEY = {item.key: item for item in MODEL_CATALOG}


class ModelStore:
    """Download curated snapshots into an isolated application cache."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        endpoint: str = "https://huggingface.co",
        offline: bool = False,
    ) -> None:
        self.root = root or application_paths().model_dir
        self.endpoint = endpoint.rstrip("/")
        self.offline = offline
        if not self.endpoint.startswith(
            ("https://", "http://localhost", "http://127.0.0.1")
        ):
            raise ConfigurationError("model endpoint must use HTTPS or localhost HTTP")

    def path(self, key: str) -> Path | None:
        descriptor = model_descriptor(key)
        pointer = self.root / descriptor.key / "current.json"
        try:
            data = json.loads(pointer.read_text(encoding="utf-8"))
            relative = data["path"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            return None
        selected = self.root / descriptor.key / str(relative)
        return selected if self._valid(selected, descriptor) else None

    def download(self, key: str, revision: str | None = None) -> Path:
        descriptor = model_descriptor(key)
        existing = self.path(key)
        if existing is not None and revision is None:
            return existing
        if self.offline:
            if existing is not None:
                return existing
            raise ConfigurationError(f"model is unavailable in offline mode: {key}")

        self.root.mkdir(parents=True, exist_ok=True)
        api = HfApi(endpoint=self.endpoint)
        try:
            info = api.model_info(descriptor.repository, revision=revision)
            resolved_revision = str(info.sha)
        except Exception as exc:
            raise ConfigurationError(
                f"could not resolve model {key} from {self.endpoint}: {exc}"
            ) from exc
        target_parent = self.root / descriptor.key
        target = target_parent / resolved_revision
        if self._valid(target, descriptor):
            self._write_pointer(target_parent, resolved_revision)
            return target

        target_parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".download-", dir=target_parent))
        try:
            files = staging / "files"
            snapshot_download(
                repo_id=descriptor.repository,
                revision=resolved_revision,
                endpoint=self.endpoint,
                local_dir=files,
            )
            if not self._valid(files, descriptor):
                raise ConfigurationError(
                    f"downloaded model is missing required files: {key}"
                )
            manifest = {
                **asdict(descriptor),
                "revision": resolved_revision,
                "endpoint": self.endpoint,
            }
            (files / "captioner-model.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            files.replace(target)
            self._write_pointer(target_parent, resolved_revision)
            return target
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ConfigurationError(f"could not download model {key}: {exc}") from exc
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    @staticmethod
    def _valid(path: Path, descriptor: ModelDescriptor) -> bool:
        return path.is_dir() and all(
            any(candidate.is_file() for candidate in path.glob(pattern))
            for pattern in descriptor.required_files
        )

    @staticmethod
    def _write_pointer(parent: Path, revision: str) -> None:
        temporary = parent / ".current.json.tmp"
        temporary.write_text(
            json.dumps({"path": revision}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(parent / "current.json")


def model_descriptor(key: str) -> ModelDescriptor:
    try:
        return _BY_KEY[key]
    except KeyError as exc:
        choices = ", ".join(sorted(_BY_KEY))
        raise ConfigurationError(
            f"unknown model {key!r}; choose one of: {choices}"
        ) from exc


def download_with_dependencies(
    store: ModelStore, key: str, revision: str | None = None
) -> dict[str, Path]:
    descriptor = model_descriptor(key)
    paths = {
        dependency: store.download(dependency) for dependency in descriptor.dependencies
    }
    paths[key] = store.download(key, revision=revision)
    return paths


__all__ = [
    "MODEL_CATALOG",
    "ModelDescriptor",
    "ModelStore",
    "download_with_dependencies",
    "model_descriptor",
]
