from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication
from pytest import CaptureFixture
from pytestqt.qtbot import QtBot

from captioner.cli.main import main as cli_main
from captioner.gui.main import select_ui_font
from captioner.gui.main_window import MainWindow
from captioner.gui.worker import JobSpec, OperationWorker
from captioner.workflow.api import (
    CancellationToken,
    PipelineOptions,
    RunResult,
    save_options,
)

FIXTURE = Path("tests/fixtures/fake_input.json")


def _options() -> PipelineOptions:
    return PipelineOptions.model_validate(
        {
            "asr": {"provider": "fake"},
            "logging": {"file": False, "console": False},
        }
    )


def test_ui_font_returns_available_family(qapp: QApplication) -> None:
    del qapp
    family = select_ui_font().family()
    installed = QFontDatabase.families()

    assert family
    if installed:
        assert family in installed


def test_window_has_four_pages_and_switches_language(
    qtbot: QtBot,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    settings = QSettings(
        str(tmp_path / "gui-test.ini"),
        QSettings.Format.IniFormat,
    )
    settings.setValue("language", "zh_CN")
    window = MainWindow(qapp, settings=settings)
    qtbot.addWidget(window)

    assert window.navigation.count() == 4
    assert window.pages.count() == 4
    assert window.windowTitle() == "视频字幕工作台"

    window.language.setCurrentIndex(window.language.findData("en"))

    assert window.windowTitle() == "VideoCaptioner"
    assert window.navigation.item(3).text() == "Diagnostics"


def test_worker_emits_result_and_progress(tmp_path: Path) -> None:
    worker = OperationWorker(
        JobSpec(
            kind="run",
            options=_options(),
            input_path=FIXTURE,
            output_dir=tmp_path,
        ),
        CancellationToken(),
    )
    results: list[object] = []
    events: list[object] = []
    failures: list[tuple[str, str]] = []

    def record_failure(kind: str, message: str) -> None:
        failures.append((kind, message))

    worker.succeeded.connect(results.append)
    worker.progress.connect(events.append)
    worker.failed.connect(record_failure)

    worker.run()

    assert not failures
    assert len(results) == 1
    assert isinstance(results[0], RunResult)
    assert events
    assert (tmp_path / "fake_input.subtitle.json").is_file()


def test_worker_reports_cooperative_cancellation(tmp_path: Path) -> None:
    cancellation = CancellationToken()
    cancellation.cancel()
    worker = OperationWorker(
        JobSpec(
            kind="run",
            options=_options(),
            input_path=FIXTURE,
            output_dir=tmp_path,
        ),
        cancellation,
    )
    failures: list[tuple[str, str]] = []

    def record_failure(kind: str, message: str) -> None:
        failures.append((kind, message))

    worker.failed.connect(record_failure)

    worker.run()

    assert failures
    assert failures[0][0] == "OperationCancelled"
    assert not (tmp_path / "fake_input.subtitle.json").exists()


def test_gui_worker_and_cli_produce_the_same_subtitle_json(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    options = _options()
    config = save_options(options, tmp_path / "config.toml")
    gui_output = tmp_path / "gui"
    cli_output = tmp_path / "cli"
    worker = OperationWorker(
        JobSpec(
            kind="run",
            options=options,
            input_path=FIXTURE,
            output_dir=gui_output,
        ),
        CancellationToken(),
    )
    failures: list[str] = []

    def record_failure(kind: str, message: str) -> None:
        failures.append(f"{kind}: {message}")

    worker.failed.connect(record_failure)

    worker.run()
    exit_code = cli_main(
        (
            "run",
            str(FIXTURE),
            "--config",
            str(config),
            "--output-dir",
            str(cli_output),
        )
    )
    capsys.readouterr()

    assert not failures
    assert exit_code == 0
    assert (gui_output / "fake_input.subtitle.json").read_text(encoding="utf-8") == (
        cli_output / "fake_input.subtitle.json"
    ).read_text(encoding="utf-8")
