"""Four focused pages used by the desktop production console."""

from pathlib import Path
from typing import cast

from pydantic import SecretStr
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from captioner.gui.i18n import tr
from captioner.workflow.api import (
    ASR_PROFILES,
    AsrProfile,
    DoctorReport,
    ModelStatus,
    OutputFormat,
    PipelineOptions,
    ProgressEvent,
    get_application_paths,
    list_models,
    with_asr_profile,
)


def _button(text: str, *, name: str = "") -> QPushButton:
    button = QPushButton(tr(text))
    if name:
        button.setObjectName(name)
    return button


class RunPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._build()
        self.retranslate()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        form_box = QGroupBox()
        grid = QGridLayout(form_box)
        self.mode_label = QLabel()
        self.mode = QComboBox()
        self.mode.addItem("", "run")
        self.mode.addItem("", "transcribe")
        self.mode.addItem("", "refine")
        grid.addWidget(self.mode_label, 0, 0)
        grid.addWidget(self.mode, 0, 1, 1, 3)

        self.input_label = QLabel()
        self.input_path = QLineEdit()
        self.file_button = _button("Browse")
        self.folder_button = _button("Browse")
        grid.addWidget(self.input_label, 1, 0)
        grid.addWidget(self.input_path, 1, 1)
        grid.addWidget(self.file_button, 1, 2)
        grid.addWidget(self.folder_button, 1, 3)

        self.output_label = QLabel()
        self.output_path = QLineEdit(str(Path.cwd()))
        self.output_button = _button("Browse")
        grid.addWidget(self.output_label, 2, 0)
        grid.addWidget(self.output_path, 2, 1, 1, 2)
        grid.addWidget(self.output_button, 2, 3)

        self.profile_label = QLabel()
        self.profile = QComboBox()
        for value in ASR_PROFILES:
            self.profile.addItem(value, value)
        self.profile.setCurrentText("faster-whisper-turbo")
        self.source_label = QLabel()
        self.source = QLineEdit("auto")
        self.target_label = QLabel()
        self.target = QLineEdit("en")
        grid.addWidget(self.profile_label, 3, 0)
        grid.addWidget(self.profile, 3, 1)
        grid.addWidget(self.source_label, 3, 2)
        grid.addWidget(self.source, 3, 3)
        grid.addWidget(self.target_label, 4, 0)
        grid.addWidget(self.target, 4, 1)

        self.correction = QCheckBox()
        self.translation = QCheckBox()
        self.repair = QCheckBox()
        self.bilingual = QCheckBox()
        self.input_bilingual = QCheckBox()
        for control in (
            self.correction,
            self.translation,
            self.repair,
            self.bilingual,
        ):
            control.setChecked(True)
        toggles = QHBoxLayout()
        for control in (
            self.correction,
            self.translation,
            self.repair,
            self.bilingual,
            self.input_bilingual,
        ):
            toggles.addWidget(control)
        toggles.addStretch()
        grid.addLayout(toggles, 5, 1, 1, 3)

        self.srt = QCheckBox("SRT")
        self.vtt = QCheckBox("VTT")
        self.json = QCheckBox("JSON")
        self.srt.setChecked(True)
        self.json.setChecked(True)
        formats = QHBoxLayout()
        for control in (self.srt, self.vtt, self.json):
            formats.addWidget(control)
        formats.addStretch()
        grid.addLayout(formats, 6, 1, 1, 3)
        layout.addWidget(form_box)

        action_row = QHBoxLayout()
        self.start_button = _button("Start", name="primary")
        self.cancel_button = _button("Cancel", name="danger")
        self.cancel_button.setEnabled(False)
        self.stage_caption = QLabel()
        self.stage_value = QLabel("-")
        self.stage_value.setObjectName("muted")
        action_row.addWidget(self.start_button)
        action_row.addWidget(self.cancel_button)
        action_row.addSpacing(18)
        action_row.addWidget(self.stage_caption)
        action_row.addWidget(self.stage_value, 1)
        layout.addLayout(action_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)

        lower = QHBoxLayout()
        self.activity_box = QGroupBox()
        activity_layout = QVBoxLayout(self.activity_box)
        self.activity = QPlainTextEdit()
        self.activity.setReadOnly(True)
        self.activity.setMaximumBlockCount(500)
        activity_layout.addWidget(self.activity)
        lower.addWidget(self.activity_box, 2)

        self.results_box = QGroupBox()
        result_layout = QVBoxLayout(self.results_box)
        self.results = QTableWidget(0, 3)
        self.results.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.results.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.results.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.results.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.open_result_button = _button("Open")
        result_layout.addWidget(self.results)
        result_layout.addWidget(self.open_result_button)
        lower.addWidget(self.results_box, 3)
        layout.addLayout(lower, 1)

        self.file_button.clicked.connect(self._select_file)
        self.folder_button.clicked.connect(self._select_folder)
        self.output_button.clicked.connect(self._select_output)
        self.open_result_button.clicked.connect(self._open_selected)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self._mode_changed()

    def retranslate(self) -> None:
        self.mode_label.setText(tr("Mode"))
        self.mode.setItemText(0, tr("Full pipeline"))
        self.mode.setItemText(1, tr("Transcribe only"))
        self.mode.setItemText(2, tr("Refine subtitles"))
        self.input_label.setText(tr("Input"))
        self.file_button.setText(tr("Browse file"))
        self.folder_button.setText(tr("Browse folder"))
        self.output_label.setText(tr("Output folder"))
        self.output_button.setText(tr("Browse"))
        self.profile_label.setText(tr("ASR profile"))
        self.source_label.setText(tr("Source language"))
        self.target_label.setText(tr("Target language"))
        self.correction.setText(tr("Correction"))
        self.translation.setText(tr("Translation"))
        self.repair.setText(tr("Repair"))
        self.bilingual.setText(tr("Bilingual"))
        self.input_bilingual.setText(tr("Bilingual input"))
        self.start_button.setText(tr("Start"))
        self.cancel_button.setText(tr("Cancel"))
        self.stage_caption.setText(f"{tr('Stage')}:")
        self.activity_box.setTitle(tr("Activity"))
        self.results_box.setTitle(tr("Results"))
        self.results.setHorizontalHeaderLabels(
            (tr("File"), tr("Status"), tr("Outputs"))
        )
        self.open_result_button.setText(tr("Open"))

    def apply_options(self, options: PipelineOptions) -> PipelineOptions:
        selected = with_asr_profile(
            options,
            cast(AsrProfile, self.profile.currentData()),
        )
        selected = selected.model_copy(
            update={
                "asr": selected.asr.model_copy(
                    update={"language": self.source.text().strip() or "auto"}
                ),
                "correction": selected.correction.model_copy(
                    update={"enabled": self.correction.isChecked()}
                ),
                "translation": selected.translation.model_copy(
                    update={
                        "enabled": self.translation.isChecked(),
                        "target_language": self.target.text().strip() or "en",
                    }
                ),
                "repair": selected.repair.model_copy(
                    update={"enabled": self.repair.isChecked()}
                ),
                "output": selected.output.model_copy(
                    update={
                        "formats": self._formats(),
                        "bilingual": self.bilingual.isChecked(),
                    }
                ),
            }
        )
        return PipelineOptions.model_validate(selected.model_dump(mode="python"))

    def load_options(self, options: PipelineOptions) -> None:
        self.source.setText(options.asr.language)
        self.target.setText(options.translation.target_language)
        self.correction.setChecked(options.correction.enabled)
        self.translation.setChecked(options.translation.enabled)
        self.repair.setChecked(options.repair.enabled)
        self.bilingual.setChecked(options.output.bilingual)
        values = {item.value for item in options.output.formats}
        self.srt.setChecked("srt" in values)
        self.vtt.setChecked("vtt" in values)
        self.json.setChecked("json" in values)

    def set_busy(self, busy: bool) -> None:
        self.start_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.progress.setRange(0, 0 if busy else 1)
        if not busy:
            self.progress.setValue(1)

    def add_event(self, event: ProgressEvent) -> None:
        position = ""
        if event.file_index and event.file_count:
            position = f"{event.file_index}/{event.file_count} "
        stage = event.stage.value if event.stage else event.kind.value
        self.stage_value.setText(f"{position}{stage}")
        self.activity.appendPlainText(
            f"{position}{stage}: {event.kind.value}"
            + (f" — {event.message}" if event.message else "")
        )

    def clear_results(self) -> None:
        self.results.setRowCount(0)
        self.activity.clear()

    def add_result(self, path: Path, status: str, outputs: tuple[Path, ...]) -> None:
        row = self.results.rowCount()
        self.results.insertRow(row)
        self.results.setItem(row, 0, QTableWidgetItem(path.name))
        self.results.setItem(row, 1, QTableWidgetItem(status))
        output_text = "\n".join(str(item) for item in outputs)
        item = QTableWidgetItem(output_text)
        item.setData(Qt.ItemDataRole.UserRole, str(outputs[0]) if outputs else "")
        self.results.setItem(row, 2, item)

    @property
    def job_kind(self) -> str:
        return str(self.mode.currentData())

    def _formats(self) -> tuple[OutputFormat, ...]:
        formats: list[OutputFormat] = []
        if self.srt.isChecked():
            formats.append(OutputFormat.SRT)
        if self.vtt.isChecked():
            formats.append(OutputFormat.VTT)
        if self.json.isChecked():
            formats.append(OutputFormat.JSON)
        return tuple(formats)

    def _select_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("Select input"))
        if path:
            self.input_path.setText(path)

    def _select_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, tr("Select input"))
        if path:
            self.input_path.setText(path)

    def _select_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, tr("Select output folder"))
        if path:
            self.output_path.setText(path)

    def _open_selected(self) -> None:
        row = self.results.currentRow()
        if row < 0:
            return
        item = self.results.item(row, 2)
        value = item.data(Qt.ItemDataRole.UserRole) if item else ""
        if value:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(value)))

    def _mode_changed(self) -> None:
        refine = self.job_kind == "refine"
        self.profile.setEnabled(not refine)
        self.folder_button.setEnabled(not refine)
        self.input_bilingual.setVisible(refine)


class SettingsPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._build()
        self.retranslate()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        columns = QHBoxLayout()
        self.llm_box = QGroupBox()
        llm_form = QFormLayout(self.llm_box)
        self.llm_provider = QComboBox()
        self.llm_provider.addItems(("fake", "openai-compatible"))
        self.base_url = QLineEdit()
        self.model_id = QLineEdit()
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.structured = QComboBox()
        self.structured.addItems(("json_schema", "json_object"))
        self.timeout = QDoubleSpinBox()
        self.timeout.setRange(1, 3600)
        self.attempts = QSpinBox()
        self.attempts.setRange(1, 10)
        self.llm_labels = [QLabel() for _ in range(7)]
        for label, control in zip(
            self.llm_labels,
            (
                self.llm_provider,
                self.base_url,
                self.model_id,
                self.api_key,
                self.structured,
                self.timeout,
                self.attempts,
            ),
            strict=True,
        ):
            llm_form.addRow(label, control)
        self.key_note = QLabel()
        self.key_note.setWordWrap(True)
        self.key_note.setObjectName("muted")
        llm_form.addRow("", self.key_note)
        columns.addWidget(self.llm_box)

        self.pipeline_box = QGroupBox()
        pipeline_form = QFormLayout(self.pipeline_box)
        self.correction_batch = self._spin(1, 1000)
        self.correction_workers = self._spin(1, 100)
        self.translation_batch = self._spin(1, 1000)
        self.translation_workers = self._spin(1, 100)
        self.repair_batch = self._spin(1, 1000)
        self.repair_workers = self._spin(1, 100)
        self.pipeline_labels = [QLabel() for _ in range(6)]
        for label, control in zip(
            self.pipeline_labels,
            (
                self.correction_batch,
                self.correction_workers,
                self.translation_batch,
                self.translation_workers,
                self.repair_batch,
                self.repair_workers,
            ),
            strict=True,
        ):
            pipeline_form.addRow(label, control)
        columns.addWidget(self.pipeline_box)

        self.runtime_box = QGroupBox()
        runtime_form = QFormLayout(self.runtime_box)
        self.endpoint = QLineEdit()
        self.offline = QCheckBox()
        self.log_level = QComboBox()
        self.log_level.addItems(("INFO", "DEBUG", "WARNING", "ERROR", "ALL", "OFF"))
        self.runtime_labels = [QLabel() for _ in range(3)]
        runtime_form.addRow(self.runtime_labels[0], self.endpoint)
        runtime_form.addRow(self.runtime_labels[1], self.offline)
        runtime_form.addRow(self.runtime_labels[2], self.log_level)
        columns.addWidget(self.runtime_box)
        layout.addLayout(columns)

        row = QHBoxLayout()
        self.config_path = QLabel()
        self.config_path.setObjectName("muted")
        self.save_button = _button("Save settings", name="primary")
        self.saved_label = QLabel()
        self.saved_label.setObjectName("muted")
        row.addWidget(self.config_path, 1)
        row.addWidget(self.saved_label)
        row.addWidget(self.save_button)
        layout.addLayout(row)
        layout.addStretch()

    @staticmethod
    def _spin(minimum: int, maximum: int) -> QSpinBox:
        value = QSpinBox()
        value.setRange(minimum, maximum)
        return value

    def retranslate(self) -> None:
        self.llm_box.setTitle("LLM")
        for label, text in zip(
            self.llm_labels,
            (
                "LLM provider",
                "Base URL",
                "Model ID",
                "API key",
                "Structured output",
                "Timeout (seconds)",
                "Attempts",
            ),
            strict=True,
        ):
            label.setText(tr(text))
        self.key_note.setText(
            tr("The API key is stored as plaintext in the local config file.")
        )
        self.pipeline_box.setTitle(tr("Configuration"))
        stage_names = (
            "Correction",
            "Correction",
            "Translation",
            "Translation",
            "Repair",
            "Repair",
        )
        values = (
            "Batch size",
            "Workers",
            "Batch size",
            "Workers",
            "Batch size",
            "Workers",
        )
        for label, stage, value in zip(
            self.pipeline_labels, stage_names, values, strict=True
        ):
            label.setText(f"{tr(stage)} · {tr(value)}")
        self.runtime_box.setTitle(tr("Models"))
        self.runtime_labels[0].setText(tr("Model endpoint"))
        self.runtime_labels[1].setText(tr("Offline mode"))
        self.runtime_labels[2].setText(tr("Log level"))
        self.save_button.setText(tr("Save settings"))

    def load_options(self, options: PipelineOptions) -> None:
        self.llm_provider.setCurrentText(options.llm.provider)
        self.base_url.setText(options.llm.base_url or "")
        self.model_id.setText(options.llm.model)
        self.api_key.setText(
            options.llm.api_key.get_secret_value() if options.llm.api_key else ""
        )
        self.structured.setCurrentText(options.llm.structured_output_mode)
        self.timeout.setValue(options.llm.timeout_seconds)
        self.attempts.setValue(options.llm.max_attempts)
        self.correction_batch.setValue(options.correction.batch_size)
        self.correction_workers.setValue(options.correction.parallelism)
        self.translation_batch.setValue(options.translation.batch_size)
        self.translation_workers.setValue(options.translation.parallelism)
        self.repair_batch.setValue(options.repair.batch_size)
        self.repair_workers.setValue(options.repair.parallelism)
        self.endpoint.setText(options.models.endpoint)
        self.offline.setChecked(options.models.offline)
        self.log_level.setCurrentText(options.logging.level)
        self.config_path.setText(str(get_application_paths().config_file))

    def apply_options(self, options: PipelineOptions) -> PipelineOptions:
        key = self.api_key.text().strip()
        updated = options.model_copy(
            update={
                "llm": options.llm.model_copy(
                    update={
                        "provider": self.llm_provider.currentText(),
                        "base_url": self.base_url.text().strip() or None,
                        "model": self.model_id.text().strip(),
                        "api_key": SecretStr(key) if key else None,
                        "structured_output_mode": self.structured.currentText(),
                        "timeout_seconds": self.timeout.value(),
                        "max_attempts": self.attempts.value(),
                    }
                ),
                "correction": options.correction.model_copy(
                    update={
                        "batch_size": self.correction_batch.value(),
                        "parallelism": self.correction_workers.value(),
                    }
                ),
                "translation": options.translation.model_copy(
                    update={
                        "batch_size": self.translation_batch.value(),
                        "parallelism": self.translation_workers.value(),
                    }
                ),
                "repair": options.repair.model_copy(
                    update={
                        "batch_size": self.repair_batch.value(),
                        "parallelism": self.repair_workers.value(),
                    }
                ),
                "models": options.models.model_copy(
                    update={
                        "endpoint": self.endpoint.text().strip(),
                        "offline": self.offline.isChecked(),
                    }
                ),
                "logging": options.logging.model_copy(
                    update={"level": self.log_level.currentText()}
                ),
            }
        )
        return PipelineOptions.model_validate(updated.model_dump(mode="python"))


class ModelsPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.path_caption = QLabel()
        self.path_value = QLabel()
        self.path_value.setObjectName("muted")
        self.refresh_button = _button("Refresh")
        self.download_button = _button("Download selected", name="primary")
        top.addWidget(self.path_caption)
        top.addWidget(self.path_value, 1)
        top.addWidget(self.refresh_button)
        top.addWidget(self.download_button)
        layout.addLayout(top)
        self.table = QTableWidget(0, 5)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.Stretch
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)
        self.retranslate()

    def retranslate(self) -> None:
        self.path_caption.setText(f"{tr('Model directory')}:")
        self.refresh_button.setText(tr("Refresh"))
        self.download_button.setText(tr("Download selected"))
        self.table.setHorizontalHeaderLabels(
            (tr("Key"), tr("Provider"), tr("Installed"), tr("Location"), tr("Notes"))
        )

    def refresh(self, options: PipelineOptions) -> None:
        models = list_models(options)
        path = options.models.cache_dir or get_application_paths().model_dir
        self.path_value.setText(str(path))
        self.table.setRowCount(0)
        for model in models:
            self._add(model)

    def selected_key(self) -> str | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        return str(item.data(Qt.ItemDataRole.UserRole)) if item else None

    def _add(self, model: ModelStatus) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        key = QTableWidgetItem(model.key)
        key.setData(Qt.ItemDataRole.UserRole, model.key)
        self.table.setItem(row, 0, key)
        self.table.setItem(row, 1, QTableWidgetItem(model.provider))
        self.table.setItem(
            row, 2, QTableWidgetItem(tr("Yes") if model.downloaded else tr("No"))
        )
        self.table.setItem(
            row, 3, QTableWidgetItem(str(model.path) if model.path else "")
        )
        self.table.setItem(row, 4, QTableWidgetItem(model.note))


class DiagnosticsPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.provider_label = QLabel()
        self.provider = QComboBox()
        self.provider.addItems(("fake", "faster-whisper", "qwen3-asr", "nemo-asr"))
        self.load_model = QCheckBox()
        self.run_button = _button("Run doctor", name="primary")
        self.open_logs_button = _button("Open log folder")
        top.addWidget(self.provider_label)
        top.addWidget(self.provider)
        top.addWidget(self.load_model)
        top.addStretch()
        top.addWidget(self.open_logs_button)
        top.addWidget(self.run_button)
        layout.addLayout(top)
        self.table = QTableWidget(0, 3)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.table)
        self.open_logs_button.clicked.connect(self._open_logs)
        self.retranslate()

    def retranslate(self) -> None:
        self.provider_label.setText(f"{tr('Provider')}:")
        self.load_model.setText(tr("Load model"))
        self.run_button.setText(tr("Run doctor"))
        self.open_logs_button.setText(tr("Open log folder"))
        self.table.setHorizontalHeaderLabels((tr("Check"), tr("Status"), tr("Detail")))

    def show_report(self, report: DoctorReport) -> None:
        self.table.setRowCount(0)
        for key, passed in report.checks.items():
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(key))
            self.table.setItem(
                row, 1, QTableWidgetItem(tr("Yes") if passed else tr("No"))
            )
            self.table.setItem(row, 2, QTableWidgetItem(report.details.get(key, "")))

    def _open_logs(self) -> None:
        QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(get_application_paths().log_dir))
        )


__all__ = ["DiagnosticsPage", "ModelsPage", "RunPage", "SettingsPage"]
