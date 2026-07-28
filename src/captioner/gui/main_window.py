"""Main PySide6 window for the caption production console."""

from pathlib import Path
from typing import cast

from PySide6.QtCore import QSettings, QThread, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from captioner.gui.i18n import CatalogTranslator, tr
from captioner.gui.pages import DiagnosticsPage, ModelsPage, RunPage, SettingsPage
from captioner.gui.worker import JobKind, JobSpec, OperationWorker
from captioner.workflow.api import (
    AsrProfile,
    CancellationToken,
    DoctorReport,
    PipelineOptions,
    ProgressEvent,
    RefineResult,
    RunResult,
    TranscriptionRunResult,
    get_application_paths,
    load_options,
    save_options,
)


class MainWindow(QMainWindow):
    """One-window, one-active-job GUI adapter."""

    def __init__(
        self,
        application: QApplication,
        *,
        options: PipelineOptions | None = None,
        settings: QSettings | None = None,
    ) -> None:
        super().__init__()
        self._application = application
        self._settings = (
            settings
            if settings is not None
            else QSettings("VideoCaptioner", "VideoCaptioner")
        )
        self._translator: CatalogTranslator | None = None
        self._options = options or load_options()
        self._thread: QThread | None = None
        self._worker: OperationWorker | None = None
        self._cancellation: CancellationToken | None = None
        self._busy = False
        self._close_requested = False
        self._build()
        self._set_language(str(self._settings.value("language", "zh_CN")))
        self._load_options()

    def _build(self) -> None:
        self.setMinimumSize(1040, 700)
        self.resize(1240, 800)
        central = QWidget()
        shell = QHBoxLayout(central)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        rail = QFrame()
        rail.setObjectName("navRail")
        rail.setFixedWidth(225)
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(18, 24, 12, 18)
        self.brand = QLabel()
        self.brand.setObjectName("brand")
        self.subtitle = QLabel()
        self.subtitle.setObjectName("subtitle")
        self.subtitle.setWordWrap(True)
        rail_layout.addWidget(self.brand)
        rail_layout.addWidget(self.subtitle)
        rail_layout.addSpacing(26)
        self.navigation = QListWidget()
        self.navigation.setObjectName("navigation")
        for _ in range(4):
            self.navigation.addItem(QListWidgetItem())
        self.navigation.setCurrentRow(0)
        rail_layout.addWidget(self.navigation, 1)
        self.language_label = QLabel()
        self.language_label.setObjectName("muted")
        self.language = QComboBox()
        self.language.addItem("简体中文", "zh_CN")
        self.language.addItem("English", "en")
        rail_layout.addWidget(self.language_label)
        rail_layout.addWidget(self.language)
        shell.addWidget(rail)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 20, 24, 20)
        header = QHBoxLayout()
        self.page_title = QLabel()
        self.page_title.setObjectName("brand")
        self.status = QLabel()
        self.status.setObjectName("muted")
        header.addWidget(self.page_title)
        header.addStretch()
        header.addWidget(self.status)
        content_layout.addLayout(header)
        self.pages = QStackedWidget()
        self.run_page = RunPage()
        self.settings_page = SettingsPage()
        self.models_page = ModelsPage()
        self.diagnostics_page = DiagnosticsPage()
        for page in (
            self.run_page,
            self.settings_page,
            self.models_page,
            self.diagnostics_page,
        ):
            self.pages.addWidget(page)
        content_layout.addWidget(self.pages, 1)
        shell.addWidget(content, 1)
        self.setCentralWidget(central)

        self.navigation.currentRowChanged.connect(self._page_changed)
        self.language.currentIndexChanged.connect(self._language_changed)
        self.run_page.start_button.clicked.connect(self._start_pipeline)
        self.run_page.cancel_button.clicked.connect(self._cancel)
        self.settings_page.save_button.clicked.connect(self._save_settings)
        self.models_page.refresh_button.clicked.connect(self._refresh_models)
        self.models_page.download_button.clicked.connect(self._download_model)
        self.diagnostics_page.run_button.clicked.connect(self._run_doctor)

    def _load_options(self) -> None:
        self.run_page.load_options(self._options)
        self.settings_page.load_options(self._options)
        self.models_page.refresh(self._options)

    def _current_options(self) -> PipelineOptions:
        return self.run_page.apply_options(
            self.settings_page.apply_options(self._options)
        )

    def _start_pipeline(self) -> None:
        try:
            options = self._current_options()
            input_path = Path(self.run_page.input_path.text().strip())
            output_dir = Path(self.run_page.output_path.text().strip())
            if not self.run_page.input_path.text().strip():
                raise ValueError(tr("Select input"))
            if not self.run_page.output_path.text().strip():
                raise ValueError(tr("Select output folder"))
        except Exception as exc:
            QMessageBox.warning(self, tr("Invalid settings"), str(exc))
            return
        self.run_page.clear_results()
        profile = (
            None
            if self.run_page.job_kind == "refine"
            else cast(AsrProfile, self.run_page.profile.currentData())
        )
        self._start_job(
            JobSpec(
                kind=cast(JobKind, self.run_page.job_kind),
                options=options,
                input_path=input_path,
                output_dir=output_dir,
                source_language=self.run_page.source.text().strip() or "und",
                input_bilingual=self.run_page.input_bilingual.isChecked(),
                profile=profile,
            )
        )

    def _run_doctor(self) -> None:
        try:
            options = self._current_options()
        except Exception as exc:
            QMessageBox.warning(self, tr("Invalid settings"), str(exc))
            return
        self._start_job(
            JobSpec(
                kind="doctor",
                options=options,
                provider=self.diagnostics_page.provider.currentText(),
                load_model=self.diagnostics_page.load_model.isChecked(),
                output_dir=Path.cwd(),
            )
        )

    def _download_model(self) -> None:
        key = self.models_page.selected_key()
        if key is None:
            QMessageBox.information(self, tr("Models"), tr("Select a model first."))
            return
        try:
            options = self._current_options()
        except Exception as exc:
            QMessageBox.warning(self, tr("Invalid settings"), str(exc))
            return
        self._start_job(JobSpec(kind="download", options=options, model_key=key))

    def _start_job(self, spec: JobSpec) -> None:
        if self._busy:
            return
        self._busy = True
        self._options = spec.options
        self._cancellation = CancellationToken()
        self._thread = QThread(self)
        self._worker = OperationWorker(spec, self._cancellation)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.succeeded.connect(self._on_success)
        self._worker.failed.connect(self._on_failure)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._job_finished)
        self.run_page.set_busy(True)
        self.status.setText(tr("Running"))
        self._thread.start()

    def _cancel(self) -> None:
        if self._cancellation is not None:
            self._cancellation.cancel()
            self.status.setText(tr("Cancelling"))

    def _on_progress(self, event: object) -> None:
        if isinstance(event, ProgressEvent):
            self.run_page.add_event(event)

    def _on_success(self, result: object) -> None:
        if isinstance(result, DoctorReport):
            self.diagnostics_page.show_report(result)
            self.status.setText(tr("Doctor complete"))
            return
        if isinstance(result, dict):
            self.models_page.refresh(self._options)
            self.status.setText(tr("Download complete"))
            return
        if isinstance(result, RunResult):
            for item in result.succeeded:
                self.run_page.add_result(
                    item.input_path,
                    tr("Completed"),
                    item.output_paths,
                )
            for item in result.failed:
                self.run_page.add_result(item.input_path, tr("Failed"), ())
        elif isinstance(result, TranscriptionRunResult):
            for item in result.succeeded:
                self.run_page.add_result(
                    item.input_path,
                    tr("Completed"),
                    (item.output_path,),
                )
            for item in result.failed:
                self.run_page.add_result(item.input_path, tr("Failed"), ())
        elif isinstance(result, RefineResult):
            self.run_page.add_result(
                result.input_path,
                tr("Completed"),
                result.output_paths,
            )
        self.status.setText(tr("Completed"))

    def _on_failure(self, error_type: str, message: str) -> None:
        if error_type == "OperationCancelled":
            self.status.setText(tr("Cancelling"))
            return
        self.status.setText(tr("Failed"))
        QMessageBox.critical(
            self,
            tr("Operation failed"),
            f"{error_type}: {message}",
        )

    def _job_finished(self) -> None:
        self._busy = False
        self._thread = None
        self._worker = None
        self._cancellation = None
        self.run_page.set_busy(False)
        if self.status.text() in {tr("Running"), tr("Cancelling")}:
            self.status.setText(tr("Ready"))
        if self._close_requested:
            QTimer.singleShot(0, self.close)

    def _save_settings(self) -> None:
        try:
            self._options = self._current_options()
            path = save_options(
                self._options,
                get_application_paths().config_file,
                overwrite=True,
            )
        except Exception as exc:
            QMessageBox.warning(self, tr("Invalid settings"), str(exc))
            return
        self.settings_page.config_path.setText(str(path))
        self.settings_page.saved_label.setText(tr("Settings saved"))
        QTimer.singleShot(3000, lambda: self.settings_page.saved_label.setText(""))
        self.models_page.refresh(self._options)

    def _refresh_models(self) -> None:
        try:
            self._options = self._current_options()
            self.models_page.refresh(self._options)
        except Exception as exc:
            QMessageBox.warning(self, tr("Invalid settings"), str(exc))

    def _page_changed(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        self.page_title.setText(self.navigation.item(index).text())

    def _language_changed(self) -> None:
        language = str(self.language.currentData())
        if language:
            self._set_language(language)

    def _set_language(self, language: str) -> None:
        if self._translator is not None:
            self._application.removeTranslator(self._translator)
        self._translator = CatalogTranslator(language)
        self._application.installTranslator(self._translator)
        self._settings.setValue("language", language)
        index = self.language.findData(language)
        if index >= 0:
            self.language.blockSignals(True)
            self.language.setCurrentIndex(index)
            self.language.blockSignals(False)
        self.retranslate()

    def retranslate(self) -> None:
        self.setWindowTitle(tr("VideoCaptioner"))
        self.brand.setText(tr("VideoCaptioner"))
        self.subtitle.setText(tr("Caption production console"))
        labels = ("Run", "Settings", "Models", "Diagnostics")
        for index, value in enumerate(labels):
            self.navigation.item(index).setText(tr(value))
        self.language_label.setText(tr("Language"))
        self.language.setItemText(0, tr("Simplified Chinese"))
        self.language.setItemText(1, tr("English"))
        self.run_page.retranslate()
        self.settings_page.retranslate()
        self.models_page.retranslate()
        self.diagnostics_page.retranslate()
        self._page_changed(self.navigation.currentRow())
        if not self._busy:
            self.status.setText(tr("Ready"))

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._busy:
            event.accept()
            return
        answer = QMessageBox.question(
            self,
            tr("Confirm close"),
            tr("A task is still running. Request cancellation and close afterwards?"),
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._close_requested = True
            self._cancel()
        event.ignore()


__all__ = ["MainWindow"]
