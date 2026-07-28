"""Run one real provider over local media and retain a sanitized audit record."""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

from captioner.workflow.api import (
    FasterWhisperAsrOptions,
    NemoAsrOptions,
    PipelineOptions,
    Qwen3AsrOptions,
    build_services,
    discover_inputs,
    load_options,
    prepare_asr_model,
    run_files,
    with_asr_profile,
)
from captioner.workflow.models import ProcessingResult


def main() -> int:
    arguments = _parser().parse_args()
    options = load_options(arguments.config)
    if arguments.asr_profile is not None:
        options = with_asr_profile(options, arguments.asr_profile)
    if options.asr.provider == "fake":
        raise SystemExit("real E2E requires a real ASR provider")
    if options.llm.provider != "openai-compatible":
        raise SystemExit("real E2E requires the OpenAI-compatible LLM adapter")

    record_dir = arguments.record_dir.resolve()
    _prepare_record(record_dir)
    shutil.copy2(
        arguments.config,
        record_dir / "configs" / arguments.config.name,
    )
    if arguments.refresh_existing:
        return _refresh_existing(options, record_dir)
    inputs = discover_inputs(arguments.input_dir, provider=options.asr.provider)
    options = prepare_asr_model(options)
    options = options.model_copy(
        update={
            "run": options.run.model_copy(
                update={"continue_on_error": False, "keep_workdir": True}
            )
        }
    )

    first_options = _cpu_options(options) if arguments.force_cpu else options
    first_name = "cpu" if arguments.force_cpu else "gpu"
    first_attempt = _run_attempt(first_options, inputs, record_dir, first_name)
    attempts = [first_attempt]
    if (
        not arguments.force_cpu
        and not first_attempt["ok"]
        and _is_cuda_failure(str(first_attempt["error"]))
    ):
        attempts.append(_run_attempt(_cpu_options(options), inputs, record_dir, "cpu"))

    provider_summary: dict[str, object] = {
        "provider": options.asr.provider,
        "inputs": [str(path.resolve()) for path in inputs],
        "attempts": attempts,
        "ok": bool(attempts[-1]["ok"]),
    }
    _update_summary(record_dir, provider_summary)
    _write_readme(record_dir)
    _assert_secret_absent(options, record_dir)
    _write_checksums(record_dir)
    return 0 if provider_summary["ok"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument(
        "--asr-profile",
        choices=(
            "faster-whisper-turbo",
            "faster-whisper-small",
            "faster-whisper-large-v2",
            "faster-whisper-large-v3",
            "qwen3-0.6b",
            "qwen3-1.7b",
            "nemo-parakeet-v3",
            "nemo-parakeet-110m-en",
        ),
    )
    parser.add_argument("--refresh-existing", action="store_true")
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="run the selected ASR provider on CPU without a GPU attempt",
    )
    return parser


def _run_attempt(
    options: PipelineOptions,
    inputs: tuple[Path, ...],
    record_dir: Path,
    attempt: str,
) -> dict[str, object]:
    provider = options.asr.provider
    started = time.monotonic()
    output_dir = record_dir / "outputs" / provider / attempt
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object]
    try:
        result = run_files(inputs, options, build_services(options), output_dir)
        artifact_dir = _copy_artifacts(
            result.workdir,
            record_dir / "artifacts" / provider / attempt,
        )
        failures = [
            {
                "input": str(failure.input_path),
                "error_type": failure.error_type,
                "message": failure.message,
            }
            for failure in result.failed
        ]
        files, violations = _file_summaries(
            result.succeeded,
            artifact_dir,
            provider,
        )
        payload = {
            "attempt": attempt,
            "device": _device(options),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "ok": not failures and not violations and len(files) == len(inputs),
            "error": "; ".join(
                [
                    *(failure["message"] for failure in failures),
                    *violations,
                ]
            ),
            "failures": failures,
            "files": files,
            "artifacts": str(artifact_dir) if artifact_dir else None,
        }
    except Exception as exc:
        payload = {
            "attempt": attempt,
            "device": _device(options),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "failures": [],
            "files": [],
            "artifacts": None,
        }
    _write_attempt_log(
        record_dir / "logs" / f"{provider}-{attempt}.log",
        payload,
        options,
    )
    return cast(dict[str, object], _sanitized(payload, options))


def _file_summaries(
    succeeded: tuple[ProcessingResult, ...],
    artifact_dir: Path | None,
    provider: str,
) -> tuple[list[dict[str, object]], list[str]]:
    files: list[dict[str, object]] = []
    violations: list[str] = []
    expected_origin = "forced_alignment" if provider == "qwen3-asr" else "asr_native"
    for item in succeeded:
        subtitle = item.subtitle
        input_path = item.input_path
        cues = subtitle.cues
        cue_warnings = [warning for cue in cues for warning in cue.warnings]
        missing_correction = [cue.id for cue in cues if cue.corrected_text is None]
        missing_translation = [cue.id for cue in cues if cue.translated_text is None]
        failed_warnings = [
            warning
            for warning in cue_warnings
            if warning.startswith(
                ("segmentation_fallback", "correction_failed", "translation_failed")
            )
        ]
        transcript = _transcript_summary(artifact_dir, input_path.stem)
        if transcript.get("timing_origin") != expected_origin:
            violations.append(f"{input_path.name}: unexpected timing origin")
        if not transcript.get("words"):
            violations.append(f"{input_path.name}: transcript contains no words")
        if missing_correction:
            violations.append(f"{input_path.name}: missing corrected text")
        if missing_translation:
            violations.append(f"{input_path.name}: missing translated text")
        if failed_warnings:
            violations.append(f"{input_path.name}: LLM stage fallback occurred")
        files.append(
            {
                "input": str(input_path.resolve()),
                "cue_count": len(cues),
                "corrected_count": len(cues) - len(missing_correction),
                "translated_count": len(cues) - len(missing_translation),
                "target_language": subtitle.target_language,
                "quality_issue_count": len(item.quality_report.issues),
                "warnings": [*item.warnings, *cue_warnings],
                "outputs": [str(path) for path in item.output_paths],
                "transcript": transcript,
            }
        )
    return files, violations


def _transcript_summary(
    artifact_dir: Path | None,
    input_stem: str,
) -> dict[str, object]:
    if artifact_dir is None:
        return {}
    safe_stem = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in input_stem
    ).strip("_")
    candidates = tuple(artifact_dir.glob(f"*-{safe_stem}/transcript.raw.json"))
    if len(candidates) != 1:
        return {}
    raw = json.loads(candidates[0].read_text(encoding="utf-8"))
    return {
        "provider": raw.get("provider"),
        "model_name": raw.get("model_name"),
        "language": raw.get("language"),
        "timing_origin": raw.get("timing_origin"),
        "segments": len(raw.get("segments", [])),
        "words": len(raw.get("words", [])),
    }


def _copy_artifacts(source: Path | None, destination: Path) -> Path | None:
    if source is None or not source.is_dir():
        return None
    shutil.copytree(source, destination, dirs_exist_ok=True)
    return destination


def _cpu_options(options: PipelineOptions) -> PipelineOptions:
    if isinstance(options.asr, FasterWhisperAsrOptions):
        provider_config = options.asr.faster_whisper.model_copy(
            update={"device": "cpu", "compute_type": "int8"}
        )
        asr = options.asr.model_copy(update={"faster_whisper": provider_config})
    elif isinstance(options.asr, Qwen3AsrOptions):
        provider_config = options.asr.qwen3.model_copy(
            update={"device": "cpu", "dtype": "float32"}
        )
        asr = options.asr.model_copy(update={"qwen3": provider_config})
    elif isinstance(options.asr, NemoAsrOptions):
        provider_config = options.asr.nemo.model_copy(update={"device": "cpu"})
        asr = options.asr.model_copy(update={"nemo": provider_config})
    else:
        raise ValueError("CPU fallback requires a real ASR provider")
    return options.model_copy(update={"asr": asr})


def _device(options: PipelineOptions) -> str:
    if isinstance(options.asr, FasterWhisperAsrOptions):
        return options.asr.faster_whisper.device
    if isinstance(options.asr, Qwen3AsrOptions):
        return options.asr.qwen3.device
    if isinstance(options.asr, NemoAsrOptions):
        return options.asr.nemo.device
    return "none"


def _is_cuda_failure(message: str) -> bool:
    lowered = message.lower()
    indicators = (
        ".so",
        "cannot open shared object file",
        "shared librar",
        "undefined symbol",
        "cuda",
        "cudnn",
        "cublas",
        "out of memory",
    )
    return any(indicator in lowered for indicator in indicators)


def _prepare_record(record_dir: Path) -> None:
    for name in ("configs", "environment", "logs", "outputs", "artifacts"):
        (record_dir / name).mkdir(parents=True, exist_ok=True)
    environment_dir = record_dir / "environment"
    if not (environment_dir / "system.txt").exists():
        gpu = _command_output(
            (
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            )
        )
        (environment_dir / "system.txt").write_text(
            f"python={sys.version}\nplatform={platform.platform()}\n"
            f"ffmpeg={shutil.which('ffmpeg')}\ngpu={gpu}\n",
            encoding="utf-8",
        )
        environments = {
            "lab-packages.txt": "Lab",
            "faster-packages.txt": "captioner-asr-faster-whisper",
            "qwen-packages.txt": "captioner-asr-qwen3",
            "nemo-packages.txt": "captioner-asr-nemo",
        }
        for filename, environment in environments.items():
            (environment_dir / filename).write_text(
                _command_output(("conda", "list", "-n", environment)),
                encoding="utf-8",
            )
        (record_dir / "git-revision.txt").write_text(
            _command_output(("git", "rev-parse", "HEAD")) + "\n",
            encoding="utf-8",
        )


def _command_output(command: tuple[str, ...]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"
    return (completed.stdout or completed.stderr).strip()


def _write_attempt_log(
    path: Path,
    payload: dict[str, object],
    options: PipelineOptions,
) -> None:
    path.write_text(
        json.dumps(_sanitized(payload, options), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sanitized(value: object, options: PipelineOptions) -> object:
    rendered = json.dumps(value, ensure_ascii=False)
    key = options.llm.api_key
    if key is not None:
        rendered = rendered.replace(key.get_secret_value(), "**********")
    return json.loads(rendered)


def _update_summary(record_dir: Path, provider_summary: dict[str, object]) -> None:
    path = record_dir / "summary.json"
    payload: dict[str, object] = {"runs": []}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    runs = cast(list[dict[str, object]], payload.setdefault("runs", []))
    runs[:] = [
        run for run in runs if run.get("provider") != provider_summary["provider"]
    ]
    runs.append(provider_summary)
    payload["ok"] = all(bool(run.get("ok")) for run in runs)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_readme(record_dir: Path) -> None:
    summary = json.loads((record_dir / "summary.json").read_text(encoding="utf-8"))
    lines = [
        "# VideoCaptioner Real E2E Record",
        "",
        "Credentials are stored only in ignored local TOML files under `configs/`.",
        "Logs and artifacts were scanned to ensure the credential is absent.",
        "",
        "## Runs",
        "",
    ]
    for run in summary["runs"]:
        attempts = ", ".join(
            f"{attempt['device']}={'PASS' if attempt['ok'] else 'FAIL'}"
            for attempt in run["attempts"]
        )
        lines.append(
            f"- {run['provider']}: {'PASS' if run['ok'] else 'FAIL'} ({attempts})"
        )
    lines.extend(
        [
            "",
            "Inspect `summary.json`, `logs/`, `artifacts/`, and `outputs/` "
            "for detailed evidence.",
            "",
        ]
    )
    (record_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _refresh_existing(options: PipelineOptions, record_dir: Path) -> int:
    summary_path = record_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    provider_runs = [
        run for run in summary["runs"] if run.get("provider") == options.asr.provider
    ]
    if len(provider_runs) != 1:
        raise RuntimeError("record does not contain exactly one matching provider")
    run = provider_runs[0]
    expected_origin = (
        "forced_alignment" if options.asr.provider == "qwen3-asr" else "asr_native"
    )
    for attempt in run["attempts"]:
        files = attempt.get("files", [])
        artifact_value = attempt.get("artifacts")
        if not files or not isinstance(artifact_value, str):
            continue
        artifact_dir = Path(artifact_value)
        violations: list[str] = []
        for file_summary in files:
            input_path = Path(file_summary["input"])
            transcript = _transcript_summary(artifact_dir, input_path.stem)
            file_summary["transcript"] = transcript
            if transcript.get("timing_origin") != expected_origin:
                violations.append(f"{input_path.name}: unexpected timing origin")
            if not transcript.get("words"):
                violations.append(f"{input_path.name}: transcript contains no words")
        prior_error = str(attempt.get("error", ""))
        non_audit_errors = [
            part
            for part in prior_error.split("; ")
            if "timing origin" not in part
            and "transcript contains no words" not in part
        ]
        errors = [*non_audit_errors, *violations]
        attempt["error"] = "; ".join(errors)
        attempt["ok"] = not errors and not attempt.get("failures")
        _write_attempt_log(
            record_dir / "logs" / f"{options.asr.provider}-{attempt['attempt']}.log",
            attempt,
            options,
        )
    run["ok"] = bool(run["attempts"][-1]["ok"])
    summary["ok"] = all(bool(value.get("ok")) for value in summary["runs"])
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_readme(record_dir)
    _assert_secret_absent(options, record_dir)
    _write_checksums(record_dir)
    return 0 if run["ok"] else 1


def _assert_secret_absent(options: PipelineOptions, record_dir: Path) -> None:
    key = options.llm.api_key
    if key is None:
        return
    secret = key.get_secret_value().encode()
    leaks: list[str] = []
    for path in record_dir.rglob("*"):
        if not path.is_file() or path.is_relative_to(record_dir / "configs"):
            continue
        if secret in path.read_bytes():
            leaks.append(str(path.relative_to(record_dir)))
    if leaks:
        raise RuntimeError(f"credential leaked outside configs/: {leaks}")


def _write_checksums(record_dir: Path) -> None:
    lines: list[str] = []
    for path in sorted(record_dir.rglob("*")):
        if (
            not path.is_file()
            or path.name == "checksums.sha256"
            or path.is_relative_to(record_dir / "configs")
        ):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(record_dir)}")
    (record_dir / "checksums.sha256").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
