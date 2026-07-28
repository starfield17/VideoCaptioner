import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src"
WORKER_ROOT = ROOT / "workers"


def _python_files() -> tuple[Path, ...]:
    return tuple(sorted((*SOURCE_ROOT.rglob("*.py"), *WORKER_ROOT.rglob("*.py"))))


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            values.append(node.module)
    return tuple(values)


def test_public_api_modules_exist() -> None:
    for module_path in (
        SOURCE_ROOT / "captioner/media/api.py",
        SOURCE_ROOT / "captioner/transcription/api.py",
        SOURCE_ROOT / "captioner/subtitles/api.py",
        SOURCE_ROOT / "captioner/llm/api.py",
        SOURCE_ROOT / "captioner/workflow/api.py",
    ):
        assert module_path.is_file(), module_path


def test_forbidden_runtime_features_are_absent() -> None:
    files = _python_files()
    source = "\n".join(path.read_text(encoding="utf-8") for path in files)
    concurrency_path = SOURCE_ROOT / "captioner/llm/concurrency.py"
    executor_name = "ThreadPool" + "Executor"
    for path in files:
        path_source = path.read_text(encoding="utf-8")
        if executor_name in path_source:
            assert path == concurrency_path
    assert "asyncio" not in source
    assert not re.search(r"\basync\s+def\b", source)
    assert "sqlite" not in source.lower()
    assert "sqlalchemy" not in source.lower()


def test_import_direction_is_explicit() -> None:
    violations: list[str] = []
    for path in _python_files():
        imports = _imports(path)
        if path.is_relative_to(SOURCE_ROOT / "captioner/shared"):
            forbidden = (
                "captioner.media",
                "captioner.transcription",
                "captioner.subtitles",
                "captioner.llm",
                "captioner.workflow",
            )
            violations.extend(
                f"{path}: {name}" for name in imports if name.startswith(forbidden)
            )
        elif path.is_relative_to(SOURCE_ROOT / "captioner/media"):
            forbidden = (
                "captioner.transcription",
                "captioner.subtitles",
                "captioner.llm",
                "captioner.workflow",
            )
            violations.extend(
                f"{path}: {name}" for name in imports if name.startswith(forbidden)
            )
        elif path.is_relative_to(SOURCE_ROOT / "captioner/transcription"):
            forbidden = ("captioner.media", "captioner.subtitles", "captioner.workflow")
            violations.extend(
                f"{path}: {name}" for name in imports if name.startswith(forbidden)
            )
        elif path.is_relative_to(SOURCE_ROOT / "captioner/subtitles"):
            forbidden = ("captioner.media", "captioner.workflow")
            violations.extend(
                f"{path}: {name}" for name in imports if name.startswith(forbidden)
            )
            violations.extend(
                f"{path}: {name}"
                for name in imports
                if name.startswith("captioner.transcription")
                and name != "captioner.transcription.api"
            )
            violations.extend(
                f"{path}: {name}"
                for name in imports
                if name.startswith("captioner.llm") and name != "captioner.llm.api"
            )
        elif path.is_relative_to(SOURCE_ROOT / "captioner/llm"):
            forbidden = (
                "captioner.media",
                "captioner.transcription",
                "captioner.subtitles",
                "captioner.workflow",
            )
            violations.extend(
                f"{path}: {name}" for name in imports if name.startswith(forbidden)
            )
        elif path.is_relative_to(SOURCE_ROOT / "captioner/cli"):
            forbidden = (
                "captioner.media",
                "captioner.transcription",
                "captioner.subtitles",
                "captioner.llm",
            )
            violations.extend(
                f"{path}: {name}" for name in imports if name.startswith(forbidden)
            )
            allowed_captioner = (
                "captioner.cli",
                "captioner.shared",
                "captioner.workflow",
            )
            violations.extend(
                f"{path}: {name}"
                for name in imports
                if name.startswith("captioner.")
                and not name.startswith(allowed_captioner)
            )
        elif path.is_relative_to(WORKER_ROOT):
            violations.extend(
                f"{path}: {name}"
                for name in imports
                if name.startswith("captioner.workflow")
            )
    assert not violations, "\n".join(violations)


def test_llm_structured_outputs_have_no_timing_fields() -> None:
    from captioner.llm import models

    for model in (models.BoundarySelection, models.TextUpdate, models.TextUpdateBatch):
        fields = set(model.model_fields)
        assert not fields.intersection({"start_ms", "end_ms", "timestamp", "time"})


def test_domain_models_do_not_read_environment_or_provider_state() -> None:
    for path in (
        SOURCE_ROOT / "captioner/transcription/models.py",
        SOURCE_ROOT / "captioner/subtitles/models.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "os.environ" not in source
        assert "os.getenv" not in source
