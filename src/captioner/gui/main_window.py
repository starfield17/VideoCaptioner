"""Main PySide6 window for the caption production console."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

from PySide6.QtCore import QElapsedTimer, QSettings, Qt, QThread, QTimer
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QDragEnterEvent,
    QDropEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStyle,
    QToolBar,
    QWidget,
)

from captioner.gui.dialogs import (
    ActivityLogDialog,
    CompletionDialog,
    DoctorDialog,
    ModelsDialog,
    RunConfirmationDialog,
    ScanPreviewDialog,
    SettingsDialog,
)
from captioner.gui.i18n import CatalogTranslator, tr
from captioner.gui.pages import RunPage
from captioner.gui.styles import ThemeName, style_for
from captioner.gui.worker import JobKind, JobSpec, OperationWorker
from captioner.workflow.api import (
    ApplicationCommand,
    ApplicationPlan,
    AsrProfile,
    CancellationToken,
    DoctorReport,
    PipelineOptions,
    ProgressEvent,
    RefineResult,
    RunResult,
    RuntimeStatus,
    TranscriptionRunResult,
    get_application_paths,
    get_runtime_status,
    load_options,
    plan_operation,
    save_options,
)

_MEDIA_SUFFIXES = {
    ".aac",
    ".avi",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".wav",
    ".webm",
}


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
        self._language = str(self._settings.value("language", "en"))
        self._theme = self._normalized_theme(
            str(self._settings.value("theme", "light"))
        )
        self._options = options or load_options()
        self._thread: QThread | None = None
        self._worker: OperationWorker | None = None
        self._cancellation: CancellationToken | None = None
        self._active_spec: JobSpec | None = None
        self._busy = False
        self._close_requested = False
        self._scan_intent: str | None = None
        self._cached_plan: ApplicationPlan | None = None
        self._after_job: Callable[[], None] | None = None
        self._elapsed = QElapsedTimer()
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        self._build()
        self._apply_theme(self._theme, persist=False)
        self._set_language(self._language, persist=False)
        self.run_page.load_options(self._options)

    def _build(self) -> None:
        self.setAcceptDrops(True)
        self.setMinimumSize(1080, 720)
        self.resize(1320, 860)
        root = QWidget()
        root.setObjectName("appRoot")
        self.run_page = RunPage()
        root_layout = self._zero_layout(root)
        root_layout.addWidget(self.run_page)
        self.setCentralWidget(root)

        self.toolbar = QToolBar()
        self.toolbar.setMovable(False)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(self.toolbar)
        self.add_file_action = self._action(
            QStyle.StandardPixmap.SP_FileIcon, self._select_file
        )
        self.add_folder_action = self._action(
            QStyle.StandardPixmap.SP_DirOpenIcon, self._select_folder
        )
        self.scan_action = self._action(
            QStyle.StandardPixmap.SP_BrowserReload, self._scan_preview
        )
        self.start_action = self._action(
            QStyle.StandardPixmap.SP_MediaPlay, self._start_pipeline
        )
        self.cancel_action = self._action(
            QStyle.StandardPixmap.SP_BrowserStop, self._cancel
        )
        for action in (
            self.add_file_action,
            self.add_folder_action,
            self.scan_action,
            self.start_action,
            self.cancel_action,
        ):
            self.toolbar.addAction(action)
        self.toolbar.addSeparator()
        self.activity_action = self._action(
            QStyle.StandardPixmap.SP_FileDialogDetailedView,
            self._show_activity,
        )
        self.models_action = self._action(
            QStyle.StandardPixmap.SP_DriveHDIcon, self._show_models
        )
        self.doctor_action = self._action(
            QStyle.StandardPixmap.SP_DialogApplyButton, self._show_doctor
        )
        self.settings_action = self._action(
            QStyle.StandardPixmap.SP_FileDialogContentsView,
            self._show_settings,
        )
        for action in (
            self.activity_action,
            self.models_action,
            self.doctor_action,
            self.settings_action,
        ):
            self.toolbar.addAction(action)

        self.status = QLabel()
        self.status.setObjectName("muted")
        self.statusBar().addWidget(self.status, 1)

        self.activity_dialog = ActivityLogDialog(self)
        self.models_dialog = ModelsDialog(self)
        self.doctor_dialog = DoctorDialog(self)
        self.models_dialog.refresh_button.clicked.connect(self._refresh_models)
        self.models_dialog.download_button.clicked.connect(self._download_model)
        self.models_dialog.install_runtime_button.clicked.connect(self._install_runtime)
        self.models_dialog.repair_runtime_button.clicked.connect(
            lambda: self._install_runtime(repair=True)
        )
        self.models_dialog.remove_runtime_button.clicked.connect(self._remove_runtime)
        self.doctor_dialog.run_button.clicked.connect(self._run_doctor)
        self.run_page.input_path.textChanged.connect(self._invalidate_plan)
        self.run_page.output_button.clicked.connect(self._select_output)
        self.run_page.mode.currentIndexChanged.connect(self._invalidate_plan)
        self.run_page.profile.currentIndexChanged.connect(self._invalidate_plan)
        self.cancel_action.setEnabled(False)

    @staticmethod
    def _zero_layout(parent: QWidget):
        from PySide6.QtWidgets import QVBoxLayout

        layout = QVBoxLayout(parent)
        layout.setContentsMargins(0, 0, 0, 0)
        return layout

    def _action(
        self,
        icon: QStyle.StandardPixmap,
        callback: Callable[[], None],
    ) -> QAction:
        action = QAction(self.style().standardIcon(icon), "", self)
        action.triggered.connect(callback)
        return action

    def _current_options(self) -> PipelineOptions:
        return self.run_page.apply_options(self._options)

    def _validated_paths(self) -> tuple[Path, Path]:
        input_text = self.run_page.input_path.text().strip()
        output_text = self.run_page.output_path.text().strip()
        if not input_text:
            raise ValueError(tr("Select input"))
        if not output_text:
            raise ValueError(tr("Select output folder"))
        return Path(input_text), Path(output_text)

    def _start_pipeline(self) -> None:
        try:
            options = self._current_options()
            input_path, output_dir = self._validated_paths()
            command = cast(ApplicationCommand, self.run_page.job_kind)
            if input_path.is_dir() and command != "refine":
                if not self._plan_matches(
                    self._cached_plan,
                    input_path,
                    output_dir,
                    command,
                    options.asr.provider,
                ):
                    self._begin_scan("start")
                    return
                plan = cast(ApplicationPlan, self._cached_plan)
            else:
                plan = plan_operation(
                    command,
                    input_path,
                    output_dir,
                    options,
                )
        except Exception as exc:
            QMessageBox.warning(self, tr("Invalid settings"), str(exc))
            return
        self._confirm_and_launch(plan, options)

    def _begin_scan(self, intent: str) -> None:
        try:
            options = self._current_options()
            input_path, output_dir = self._validated_paths()
            if self.run_page.job_kind == "refine":
                raise ValueError(tr("Folder scanning is unavailable in refine mode."))
        except Exception as exc:
            QMessageBox.warning(self, tr("Invalid settings"), str(exc))
            return
        self._scan_intent = intent
        self._start_job(
            JobSpec(
                kind="scan",
                options=options,
                input_path=input_path,
                output_dir=output_dir,
                scan_command=cast(ApplicationCommand, self.run_page.job_kind),
            )
        )

    def _scan_preview(self) -> None:
        self._begin_scan("preview")

    def _confirm_and_launch(
        self, plan: ApplicationPlan, options: PipelineOptions
    ) -> None:
        if plan.provider != "fake":
            runtime = get_runtime_status(plan.provider)
            if not runtime.installed:
                if not runtime.descriptor.available:
                    QMessageBox.warning(
                        self,
                        tr("Runtime unavailable"),
                        runtime.detail,
                    )
                    return
                answer = QMessageBox.question(
                    self,
                    tr("Install runtime"),
                    (
                        f"{plan.provider} {tr('runtime is not installed.')} "
                        f"[{runtime.descriptor.stability.value}]\n\n"
                        f"{runtime.descriptor.reason}\n\n"
                        f"{tr('Install it now?')}"
                    ),
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
                self._after_job = lambda: self._confirm_and_launch(plan, options)
                self._start_job(
                    JobSpec(
                        kind="runtime_install",
                        options=options,
                        runtime_provider=plan.provider,
                    )
                )
                return
        if self._setting_bool("confirm_before_run", True):
            stages = tuple(
                name
                for name, enabled in (
                    ("Correction", options.correction.enabled),
                    ("Translation", options.translation.enabled),
                    ("Repair", options.repair.enabled),
                )
                if enabled
            )
            dialog = RunConfirmationDialog(
                plan=plan,
                profile=(
                    self.run_page.profile.currentText()
                    if plan.command != "refine"
                    else "—"
                ),
                stages=stages,
                formats=plan.output_formats,
                parent=self,
            )
            if dialog.exec() != dialog.DialogCode.Accepted:
                return
            if dialog.dont_ask.isChecked():
                self._settings.setValue("confirm_before_run", False)
        profile = (
            None
            if plan.command == "refine"
            else cast(AsrProfile, self.run_page.profile.currentData())
        )
        self.run_page.prepare_run(len(plan.inputs))
        self.activity_dialog.add_entry(
            f"{tr('Starting')} {plan.command}: {len(plan.inputs)} {tr('files')}"
        )
        self._start_job(
            JobSpec(
                kind=cast(JobKind, plan.command),
                options=options,
                input_path=(
                    plan.input_root if plan.input_root is not None else plan.inputs[0]
                ),
                output_dir=plan.output_dir,
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
                provider=self.doctor_dialog.page.provider.currentText(),
                load_model=self.doctor_dialog.page.load_model.isChecked(),
                output_dir=Path.cwd(),
            )
        )

    def _download_model(self) -> None:
        key = self.models_dialog.page.selected_key()
        if key is None:
            QMessageBox.information(self, tr("Models"), tr("Select a model first."))
            return
        answer = QMessageBox.question(
            self,
            tr("Download model"),
            f"{tr('Download model')} “{key}”?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            options = self._current_options()
        except Exception as exc:
            QMessageBox.warning(self, tr("Invalid settings"), str(exc))
            return
        self._start_job(JobSpec(kind="download", options=options, model_key=key))

    def _install_runtime(self, *, repair: bool = False) -> None:
        runtime = self.models_dialog.page.selected_runtime()
        if runtime is None:
            QMessageBox.information(self, tr("Runtimes"), tr("Select a runtime first."))
            return
        if not runtime.descriptor.available:
            QMessageBox.warning(
                self, tr("Runtime unavailable"), runtime.descriptor.reason
            )
            return
        action = tr("Repair runtime") if repair else tr("Install runtime")
        answer = QMessageBox.question(
            self,
            action,
            (
                f"{action}: {runtime.descriptor.provider}?\n\n"
                f"[{runtime.descriptor.stability.value}] "
                f"{runtime.descriptor.reason}"
            ),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._start_job(
            JobSpec(
                kind="runtime_repair" if repair else "runtime_install",
                options=self._current_options(),
                runtime_provider=runtime.descriptor.provider,
            )
        )

    def _remove_runtime(self) -> None:
        runtime = self.models_dialog.page.selected_runtime()
        if runtime is None or not runtime.installed:
            QMessageBox.information(
                self, tr("Runtimes"), tr("Select an installed runtime first.")
            )
            return
        answer = QMessageBox.warning(
            self,
            tr("Remove runtime"),
            f"{tr('Remove runtime')} {runtime.descriptor.provider}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._start_job(
            JobSpec(
                kind="runtime_remove",
                options=self._current_options(),
                runtime_provider=runtime.descriptor.provider,
            )
        )

    def _start_job(self, spec: JobSpec) -> None:
        if self._busy:
            return
        self._busy = True
        self._active_spec = spec
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
        self._set_busy(True)
        self._elapsed.start()
        self._elapsed_timer.start()
        self.status.setText(tr("Scanning") if spec.kind == "scan" else tr("Running"))
        self._thread.start()

    def _cancel(self) -> None:
        if self._cancellation is not None:
            self._cancellation.cancel()
            self.status.setText(tr("Cancelling"))

    def _on_progress(self, event: object) -> None:
        if isinstance(event, ProgressEvent):
            line = self.run_page.add_event(event)
            self.activity_dialog.add_entry(line)

    def _on_success(self, result: object) -> None:
        kind = self._active_spec.kind if self._active_spec is not None else None
        if kind == "scan" and isinstance(result, ApplicationPlan):
            preview = ScanPreviewDialog(result, self)
            accepted = preview.exec() == preview.DialogCode.Accepted
            if accepted:
                self._cached_plan = result
                self.status.setText(
                    f"{tr('Ready')} · {len(result.inputs)} {tr('files')}"
                )
                if self._scan_intent == "start":
                    options = self._options
                    self._after_job = lambda: self._confirm_and_launch(result, options)
            else:
                self.status.setText(tr("Ready"))
            return
        if isinstance(result, DoctorReport):
            self.doctor_dialog.page.show_report(result)
            self.status.setText(tr("Doctor complete"))
            QMessageBox.information(
                self.doctor_dialog,
                tr("Diagnostics"),
                tr("Diagnostics complete."),
            )
            return
        if isinstance(result, RuntimeStatus):
            self.models_dialog.page.refresh(self._options)
            self.status.setText(tr("Runtime operation complete"))
            QMessageBox.information(
                self.models_dialog,
                tr("Runtimes"),
                f"{result.descriptor.provider}: {result.detail}",
            )
            return
        if isinstance(result, dict):
            self.models_dialog.page.refresh(self._options)
            self.status.setText(tr("Download complete"))
            QMessageBox.information(
                self.models_dialog,
                tr("Models"),
                tr("Download complete"),
            )
            return

        succeeded = 0
        failed = 0
        failure_details: list[str] = []
        output_dir = (
            self._active_spec.output_dir
            if self._active_spec and self._active_spec.output_dir
            else Path.cwd()
        )
        if isinstance(result, RunResult):
            for item in result.succeeded:
                succeeded += 1
                self.run_page.add_result(
                    item.input_path, tr("Completed"), item.output_paths
                )
            for item in result.failed:
                failed += 1
                detail = f"{item.error_type}: {item.message}"
                failure_details.append(f"{item.input_path}: {detail}")
                self.run_page.add_result(
                    item.input_path, tr("Failed"), (), detail=detail
                )
        elif isinstance(result, TranscriptionRunResult):
            for item in result.succeeded:
                succeeded += 1
                self.run_page.add_result(
                    item.input_path, tr("Completed"), (item.output_path,)
                )
            for item in result.failed:
                failed += 1
                detail = f"{item.error_type}: {item.message}"
                failure_details.append(f"{item.input_path}: {detail}")
                self.run_page.add_result(
                    item.input_path, tr("Failed"), (), detail=detail
                )
        elif isinstance(result, RefineResult):
            succeeded = 1
            self.run_page.add_result(
                result.input_path, tr("Completed"), result.output_paths
            )
        self.status.setText(tr("Completed") if failed == 0 else tr("Failed"))
        CompletionDialog(
            succeeded=succeeded,
            failed=failed,
            output_dir=output_dir,
            failures=tuple(failure_details),
            parent=self,
        ).exec()

    def _on_failure(self, error_type: str, message: str) -> None:
        self.activity_dialog.add_entry(
            f"{error_type}: {message}",
            error=True,
        )
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
        self._active_spec = None
        self._scan_intent = None
        self._elapsed_timer.stop()
        self._set_busy(False)
        if self.status.text() in {tr("Running"), tr("Scanning"), tr("Cancelling")}:
            self.status.setText(tr("Ready"))
        after_job = self._after_job
        self._after_job = None
        if after_job is not None:
            QTimer.singleShot(0, after_job)
        if self._close_requested:
            QTimer.singleShot(0, self.close)

    def _set_busy(self, busy: bool) -> None:
        for action in (
            self.add_file_action,
            self.add_folder_action,
            self.scan_action,
            self.start_action,
            self.models_action,
            self.doctor_action,
            self.settings_action,
        ):
            action.setEnabled(not busy)
        self.cancel_action.setEnabled(busy)
        self.run_page.input_path.setEnabled(not busy)
        self.run_page.output_path.setEnabled(not busy)
        self.run_page.mode.setEnabled(not busy)
        self.run_page.output_button.setEnabled(not busy)
        self.models_dialog.refresh_button.setEnabled(not busy)
        self.models_dialog.download_button.setEnabled(not busy)
        self.models_dialog.install_runtime_button.setEnabled(not busy)
        self.models_dialog.repair_runtime_button.setEnabled(not busy)
        self.models_dialog.remove_runtime_button.setEnabled(not busy)
        self.doctor_dialog.run_button.setEnabled(not busy)

    def _update_elapsed(self) -> None:
        if self._elapsed.isValid():
            self.run_page.set_elapsed(self._elapsed.elapsed() // 1000)

    def _show_settings(self) -> None:
        dialog = SettingsDialog(
            self._options,
            language=self._language,
            theme=self._theme,
            confirm_run=self._setting_bool("confirm_before_run", True),
            parent=self,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        try:
            options = dialog.page.apply_options(self._options)
            save_options(
                options,
                get_application_paths().config_file,
                overwrite=True,
            )
        except Exception as exc:
            QMessageBox.warning(self, tr("Invalid settings"), str(exc))
            return
        self._options = options
        self.run_page.load_options(options)
        language = str(dialog.page.language.currentData())
        theme = self._normalized_theme(str(dialog.page.theme.currentData()))
        self._settings.setValue(
            "confirm_before_run", dialog.page.confirm_run.isChecked()
        )
        self._apply_theme(theme)
        self._set_language(language)
        QMessageBox.information(self, tr("Settings"), tr("Settings saved"))

    def _show_models(self) -> None:
        self._refresh_models()
        self.models_dialog.show()
        self.models_dialog.raise_()
        self.models_dialog.activateWindow()

    def _show_doctor(self) -> None:
        self.doctor_dialog.show()
        self.doctor_dialog.raise_()
        self.doctor_dialog.activateWindow()

    def _show_activity(self) -> None:
        self.activity_dialog.show()
        self.activity_dialog.raise_()
        self.activity_dialog.activateWindow()

    def _refresh_models(self) -> None:
        try:
            self._options = self._current_options()
            self.models_dialog.page.refresh(self._options)
        except Exception as exc:
            QMessageBox.warning(self, tr("Invalid settings"), str(exc))

    def _select_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("Select input"),
            "",
            tr("Media and subtitle files")
            + " (*.mp4 *.mkv *.mov *.avi *.webm *.mp3 *.wav *.flac *.m4a "
            "*.aac *.ogg *.srt *.json);;" + tr("All files (*)"),
        )
        if path:
            self._accept_input(Path(path), scan_folder=False)

    def _select_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, tr("Select input"))
        if path:
            self._accept_input(Path(path), scan_folder=True)

    def _select_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            tr("Select output folder"),
            self.run_page.output_path.text(),
        )
        if path:
            self.run_page.output_path.setText(path)

    def _accept_input(self, path: Path, *, scan_folder: bool) -> None:
        if self._busy:
            QMessageBox.warning(
                self, tr("Task running"), tr("Wait for the current task to finish.")
            )
            return
        if (
            path.is_dir()
            and self.run_page.job_kind == "refine"
            or (
                path.is_file()
                and self.run_page.job_kind == "refine"
                and path.suffix.lower() not in {".srt", ".json"}
            )
        ):
            self.run_page.set_mode("run")
        self.run_page.input_path.setText(str(path))
        if scan_folder and path.is_dir():
            self._begin_scan("preview")

    def _invalidate_plan(self) -> None:
        self._cached_plan = None

    @staticmethod
    def _plan_matches(
        plan: ApplicationPlan | None,
        input_path: Path,
        output_dir: Path,
        command: str,
        provider: str,
    ) -> bool:
        return bool(
            plan is not None
            and plan.input_root == input_path
            and plan.output_dir == output_dir
            and plan.command == command
            and plan.provider == provider
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if len(urls) != 1 or not urls[0].isLocalFile():
            QMessageBox.warning(
                self,
                tr("Unsupported drop"),
                tr("Drop exactly one local media file or folder."),
            )
            return
        path = Path(urls[0].toLocalFile())
        if not path.exists():
            QMessageBox.warning(
                self, tr("Unsupported drop"), tr("The dropped path does not exist.")
            )
            return
        if path.is_file():
            allowed = set(_MEDIA_SUFFIXES)
            if self._options.asr.provider == "fake":
                allowed.add(".json")
            if self.run_page.job_kind == "refine":
                allowed.update({".srt", ".json"})
            if path.suffix.lower() not in allowed:
                QMessageBox.warning(
                    self,
                    tr("Unsupported drop"),
                    tr("The dropped file type is not supported."),
                )
                return
        event.acceptProposedAction()
        self._accept_input(path, scan_folder=path.is_dir())

    def _set_language(self, language: str, *, persist: bool = True) -> None:
        selected = language if language in {"en", "zh_CN"} else "en"
        if self._translator is not None:
            self._application.removeTranslator(self._translator)
        self._translator = CatalogTranslator(selected)
        self._application.installTranslator(self._translator)
        self._language = selected
        if persist:
            self._settings.setValue("language", selected)
        self.retranslate()

    def _apply_theme(self, theme: ThemeName, *, persist: bool = True) -> None:
        self._theme = theme
        self._application.setStyleSheet(style_for(theme))
        if persist:
            self._settings.setValue("theme", theme)

    def retranslate(self) -> None:
        self.setWindowTitle(tr("VideoCaptioner"))
        for action, text in (
            (self.add_file_action, "Add file"),
            (self.add_folder_action, "Add folder"),
            (self.scan_action, "Scan"),
            (self.start_action, "Start"),
            (self.cancel_action, "Cancel"),
            (self.activity_action, "Activity log"),
            (self.models_action, "Models"),
            (self.doctor_action, "Doctor"),
            (self.settings_action, "Settings"),
        ):
            action.setText(tr(text))
        self.run_page.retranslate()
        self.activity_dialog.retranslate()
        self.models_dialog.retranslate()
        self.doctor_dialog.retranslate()
        if not self._busy:
            self.status.setText(tr("Ready"))

    def _setting_bool(self, key: str, default: bool) -> bool:
        value = self._settings.value(key, default)
        if isinstance(value, bool):
            return value
        return str(value).lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _normalized_theme(value: str) -> ThemeName:
        return cast(ThemeName, value if value in {"light", "dark"} else "light")

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
