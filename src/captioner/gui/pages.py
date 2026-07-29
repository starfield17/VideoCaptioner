"""Primary production surface and reusable dialog panels."""

from pathlib import Path
from typing import cast

from pydantic import SecretStr
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
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
    ProgressKind,
    ProgressStage,
    RuntimeStatus,
    get_application_paths,
    list_models,
    list_runtimes,
    with_asr_profile,
)

_RIBBON_STAGES = (
    ("MEDIA", ProgressStage.MEDIA),
    ("ASR", ProgressStage.TRANSCRIPTION),
    ("CONTEXT", ProgressStage.CONTEXT_ANALYSIS),
    ("SEGMENT", ProgressStage.SEGMENTATION),
    ("CORRECT", ProgressStage.CORRECTION),
    ("TRANSLATE", ProgressStage.TRANSLATION),
    ("QC", ProgressStage.QUALITY),
    ("EXPORT", ProgressStage.EXPORT),
)


class RunPage(QWidget):
    """The always-visible source, pipeline, and job-monitoring surface."""

    def __init__(self) -> None:
        super().__init__()
        self._completed = 0
        self._failed = 0
        self._build()
        self.retranslate()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(13)

        heading = QHBoxLayout()
        heading_text = QVBoxLayout()
        self.eyebrow = QLabel()
        self.eyebrow.setObjectName("eyebrow")
        self.title = QLabel()
        self.title.setObjectName("brand")
        self.description = QLabel()
        self.description.setObjectName("muted")
        heading_text.addWidget(self.eyebrow)
        heading_text.addWidget(self.title)
        heading_text.addWidget(self.description)
        heading.addLayout(heading_text)
        heading.addStretch()
        self.drop_hint = QLabel()
        self.drop_hint.setObjectName("dropHint")
        self.drop_hint.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.drop_hint.setWordWrap(True)
        heading.addWidget(self.drop_hint, 1)
        layout.addLayout(heading)

        self.source_box = QGroupBox()
        source_grid = QGridLayout(self.source_box)
        self.mode_label = QLabel()
        self.mode = QComboBox()
        self.mode.addItem("", "run")
        self.mode.addItem("", "transcribe")
        self.mode.addItem("", "refine")
        self.input_label = QLabel()
        self.input_path = QLineEdit()
        self.input_path.setAcceptDrops(False)
        self.output_label = QLabel()
        self.output_path = QLineEdit(str(Path.cwd()))
        self.output_path.setAcceptDrops(False)
        self.output_button = QPushButton()
        self.profile_label = QLabel()
        self.profile = QComboBox()
        for value in ASR_PROFILES:
            self.profile.addItem(value, value)
        self.profile.setCurrentText("faster-whisper-turbo")
        source_grid.addWidget(self.mode_label, 0, 0)
        source_grid.addWidget(self.mode, 0, 1)
        source_grid.addWidget(self.profile_label, 0, 2)
        source_grid.addWidget(self.profile, 0, 3)
        source_grid.addWidget(self.input_label, 1, 0)
        source_grid.addWidget(self.input_path, 1, 1, 1, 4)
        source_grid.addWidget(self.output_label, 2, 0)
        source_grid.addWidget(self.output_path, 2, 1, 1, 3)
        source_grid.addWidget(self.output_button, 2, 4)
        source_grid.setColumnStretch(1, 2)
        source_grid.setColumnStretch(3, 2)
        layout.addWidget(self.source_box)

        self.tabs = QTabWidget()
        self.asr_tab = QWidget()
        asr_form = QFormLayout(self.asr_tab)
        self.source = QLineEdit("auto")
        self.target = QLineEdit("en")
        self.input_bilingual = QCheckBox()
        self.source_label = QLabel()
        self.target_label = QLabel()
        asr_form.addRow(self.source_label, self.source)
        asr_form.addRow(self.target_label, self.target)
        asr_form.addRow("", self.input_bilingual)

        self.pipeline_tab = QWidget()
        pipeline_layout = QVBoxLayout(self.pipeline_tab)
        self.correction = QCheckBox()
        self.translation = QCheckBox()
        self.repair = QCheckBox()
        for control in (self.correction, self.translation, self.repair):
            control.setChecked(True)
            pipeline_layout.addWidget(control)
        pipeline_layout.addStretch()

        self.output_tab = QWidget()
        output_layout = QVBoxLayout(self.output_tab)
        self.bilingual = QCheckBox()
        self.bilingual.setChecked(True)
        self.srt = QCheckBox("SRT")
        self.vtt = QCheckBox("VTT")
        self.json = QCheckBox("JSON")
        self.srt.setChecked(True)
        self.json.setChecked(True)
        for control in (self.bilingual, self.srt, self.vtt, self.json):
            output_layout.addWidget(control)
        output_layout.addStretch()

        self.summary_tab = QWidget()
        summary_layout = QVBoxLayout(self.summary_tab)
        self.quick_summary = QLabel()
        self.quick_summary.setWordWrap(True)
        self.quick_summary.setObjectName("muted")
        summary_layout.addWidget(self.quick_summary)
        summary_layout.addStretch()
        for tab in (
            self.asr_tab,
            self.pipeline_tab,
            self.output_tab,
            self.summary_tab,
        ):
            self.tabs.addTab(tab, "")
        layout.addWidget(self.tabs)

        self.jobs_box = QGroupBox()
        jobs_layout = QVBoxLayout(self.jobs_box)
        metrics = QHBoxLayout()
        self.metric_values: dict[str, QLabel] = {}
        self.metric_captions: dict[str, QLabel] = {}
        for key in ("total", "completed", "failed", "current", "elapsed"):
            card = QFrame()
            card.setObjectName("metricCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(11, 7, 11, 7)
            caption = QLabel()
            caption.setObjectName("metricCaption")
            value = QLabel("0" if key != "current" else "—")
            value.setObjectName("metricValue")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            card_layout.addWidget(caption)
            card_layout.addWidget(value)
            metrics.addWidget(card, 1)
            self.metric_captions[key] = caption
            self.metric_values[key] = value
        jobs_layout.addLayout(metrics)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        jobs_layout.addWidget(self.progress)

        ribbon = QFrame()
        ribbon.setObjectName("stageRibbon")
        ribbon_layout = QHBoxLayout(ribbon)
        ribbon_layout.setContentsMargins(8, 6, 8, 6)
        self.stage_chips: dict[ProgressStage, QLabel] = {}
        for text, stage in _RIBBON_STAGES:
            chip = QLabel(text)
            chip.setObjectName("stageChip")
            chip.setProperty("state", "")
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ribbon_layout.addWidget(chip)
            self.stage_chips[stage] = chip
        jobs_layout.addWidget(ribbon)

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
        self.results.setAlternatingRowColors(True)
        self.results.doubleClicked.connect(self.open_selected_result)
        jobs_layout.addWidget(self.results, 1)
        layout.addWidget(self.jobs_box, 1)

        self.mode.currentIndexChanged.connect(self._mode_changed)
        for control in (
            self.profile,
            self.source,
            self.target,
            self.correction,
            self.translation,
            self.repair,
            self.bilingual,
            self.srt,
            self.vtt,
            self.json,
        ):
            signal = (
                control.currentIndexChanged
                if isinstance(control, QComboBox)
                else (
                    control.textChanged
                    if isinstance(control, QLineEdit)
                    else control.toggled
                )
            )
            signal.connect(self._update_summary)
        self._mode_changed()

    def retranslate(self) -> None:
        self.eyebrow.setText(tr("CAPTION PRODUCTION"))
        self.title.setText(tr("Production workspace"))
        self.description.setText(
            tr("Transcribe, improve, translate, and export in one controlled run.")
        )
        self.drop_hint.setText(
            tr("Drop one media file or folder anywhere in this window.")
        )
        self.source_box.setTitle(tr("Source setup"))
        self.mode_label.setText(tr("Mode"))
        self.mode.setItemText(0, tr("Full pipeline"))
        self.mode.setItemText(1, tr("Transcribe only"))
        self.mode.setItemText(2, tr("Refine subtitles"))
        self.input_label.setText(tr("Input"))
        self.input_path.setPlaceholderText(tr("Choose or drop a media file or folder"))
        self.output_label.setText(tr("Output folder"))
        self.output_button.setText(tr("Browse"))
        self.profile_label.setText(tr("ASR profile"))
        self.source_label.setText(tr("Source language"))
        self.target_label.setText(tr("Target language"))
        self.input_bilingual.setText(tr("Bilingual input"))
        self.correction.setText(tr("Correction"))
        self.translation.setText(tr("Translation"))
        self.repair.setText(tr("Repair"))
        self.bilingual.setText(tr("Bilingual output"))
        tab_titles = (
            tr("ASR & languages").replace("&", "&&"),
            tr("Pipeline stages"),
            tr("Output formats"),
            tr("Run summary"),
        )
        for index, title in enumerate(tab_titles):
            self.tabs.setTabText(index, title)
        self.jobs_box.setTitle(tr("Jobs"))
        for key, text in (
            ("total", "Total"),
            ("completed", "Completed"),
            ("failed", "Failed"),
            ("current", "Current"),
            ("elapsed", "Elapsed"),
        ):
            self.metric_captions[key].setText(tr(text))
        self.results.setHorizontalHeaderLabels(
            (tr("File"), tr("Status"), tr("Outputs"))
        )
        self._update_summary()

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
                        "formats": self.formats(),
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
        self._update_summary()

    def prepare_run(self, total: int) -> None:
        self.results.setRowCount(0)
        self._completed = 0
        self._failed = 0
        self.metric_values["total"].setText(str(total))
        self.metric_values["completed"].setText("0")
        self.metric_values["failed"].setText("0")
        self.metric_values["current"].setText("—")
        self.metric_values["elapsed"].setText("00:00")
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(0)
        for chip in self.stage_chips.values():
            self._set_chip_state(chip, "")

    def set_elapsed(self, seconds: int) -> None:
        minutes, remainder = divmod(seconds, 60)
        self.metric_values["elapsed"].setText(f"{minutes:02d}:{remainder:02d}")

    def add_event(self, event: ProgressEvent) -> str:
        position = ""
        if event.file_index and event.file_count:
            position = f"{event.file_index}/{event.file_count} "
        stage = event.stage.value if event.stage else event.kind.value
        if event.input_path is not None:
            self.metric_values["current"].setText(event.input_path.name)
        if event.kind in {ProgressKind.FILE_COMPLETED, ProgressKind.FILE_FAILED}:
            self._completed += event.kind is ProgressKind.FILE_COMPLETED
            self._failed += event.kind is ProgressKind.FILE_FAILED
            self.metric_values["completed"].setText(str(self._completed))
            self.metric_values["failed"].setText(str(self._failed))
            self.progress.setValue(self._completed + self._failed)
        event_stage = event.stage
        if event_stage is not None and event_stage in self.stage_chips:
            state = "done" if event.kind is ProgressKind.STAGE_COMPLETED else "active"
            self._set_chip_state(self.stage_chips[event_stage], state)
        return f"{position}{stage}: {event.kind.value}" + (
            f" — {event.message}" if event.message else ""
        )

    def add_result(
        self,
        path: Path,
        status: str,
        outputs: tuple[Path, ...],
        *,
        detail: str = "",
    ) -> None:
        row = self.results.rowCount()
        self.results.insertRow(row)
        file_item = QTableWidgetItem(path.name)
        file_item.setToolTip(str(path))
        self.results.setItem(row, 0, file_item)
        status_item = QTableWidgetItem(status)
        status_item.setToolTip(detail)
        self.results.setItem(row, 1, status_item)
        output_text = "\n".join(str(item) for item in outputs)
        output_item = QTableWidgetItem(output_text)
        output_item.setData(
            Qt.ItemDataRole.UserRole, str(outputs[0]) if outputs else ""
        )
        self.results.setItem(row, 2, output_item)

    def open_selected_result(self) -> None:
        row = self.results.currentRow()
        if row < 0:
            return
        item = self.results.item(row, 2)
        value = item.data(Qt.ItemDataRole.UserRole) if item else ""
        if value:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(value)))

    @property
    def job_kind(self) -> str:
        return str(self.mode.currentData())

    def formats(self) -> tuple[OutputFormat, ...]:
        formats: list[OutputFormat] = []
        if self.srt.isChecked():
            formats.append(OutputFormat.SRT)
        if self.vtt.isChecked():
            formats.append(OutputFormat.VTT)
        if self.json.isChecked():
            formats.append(OutputFormat.JSON)
        return tuple(formats)

    def set_mode(self, mode: str) -> None:
        index = self.mode.findData(mode)
        if index >= 0:
            self.mode.setCurrentIndex(index)

    def _mode_changed(self) -> None:
        refine = self.job_kind == "refine"
        self.profile.setEnabled(not refine)
        self.input_bilingual.setVisible(refine)
        self._update_summary()

    def _update_summary(self) -> None:
        stages = [
            tr(name)
            for name, enabled in (
                ("Correction", self.correction.isChecked()),
                ("Translation", self.translation.isChecked()),
                ("Repair", self.repair.isChecked()),
            )
            if enabled
        ]
        formats = ", ".join(item.value.upper() for item in self.formats()) or "—"
        self.quick_summary.setText(
            f"{tr('Profile')}: {self.profile.currentText()}\n"
            f"{tr('Stages')}: {', '.join(stages) or '—'}\n"
            f"{tr('Formats')}: {formats}"
        )

    @staticmethod
    def _set_chip_state(chip: QLabel, state: str) -> None:
        chip.setProperty("state", state)
        chip.style().unpolish(chip)
        chip.style().polish(chip)


class SettingsPage(QWidget):
    """Editable canonical configuration plus GUI preferences."""

    def __init__(self) -> None:
        super().__init__()
        self._build()
        self.retranslate()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        general = QWidget()
        general_form = QFormLayout(general)
        self.language = QComboBox()
        self.language.addItem("English", "en")
        self.language.addItem("简体中文", "zh_CN")
        self.theme = QComboBox()
        self.theme.addItem("Light", "light")
        self.theme.addItem("Dark", "dark")
        self.confirm_run = QCheckBox()
        self.general_labels = [QLabel() for _ in range(3)]
        general_form.addRow(self.general_labels[0], self.language)
        general_form.addRow(self.general_labels[1], self.theme)
        general_form.addRow(self.general_labels[2], self.confirm_run)
        self.tabs.addTab(general, "")

        llm = QWidget()
        llm_form = QFormLayout(llm)
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
        self.tabs.addTab(llm, "LLM")

        pipeline = QWidget()
        pipeline_form = QFormLayout(pipeline)
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
        self.tabs.addTab(pipeline, "")

        runtime = QWidget()
        runtime_form = QFormLayout(runtime)
        self.endpoint = QLineEdit()
        self.offline = QCheckBox()
        self.log_level = QComboBox()
        self.log_level.addItems(("INFO", "DEBUG", "WARNING", "ERROR", "ALL", "OFF"))
        self.runtime_labels = [QLabel() for _ in range(3)]
        runtime_form.addRow(self.runtime_labels[0], self.endpoint)
        runtime_form.addRow(self.runtime_labels[1], self.offline)
        runtime_form.addRow(self.runtime_labels[2], self.log_level)
        self.tabs.addTab(runtime, "")

        self.config_path = QLabel()
        self.config_path.setObjectName("muted")
        self.config_path.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.config_path)

    @staticmethod
    def _spin(minimum: int, maximum: int) -> QSpinBox:
        value = QSpinBox()
        value.setRange(minimum, maximum)
        return value

    def retranslate(self) -> None:
        self.tabs.setTabText(0, tr("General"))
        self.tabs.setTabText(1, "LLM")
        self.tabs.setTabText(2, tr("Pipeline"))
        self.tabs.setTabText(3, tr("Runtime"))
        for label, text in zip(
            self.general_labels,
            ("Language", "Theme", "Confirm before run"),
            strict=True,
        ):
            label.setText(tr(text))
        self.language.setItemText(0, tr("English"))
        self.language.setItemText(1, tr("Simplified Chinese"))
        self.theme.setItemText(0, tr("Light"))
        self.theme.setItemText(1, tr("Dark"))
        self.confirm_run.setText(tr("Show a summary before starting"))
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
        for label, stage, value in zip(
            self.pipeline_labels,
            (
                "Correction",
                "Correction",
                "Translation",
                "Translation",
                "Repair",
                "Repair",
            ),
            (
                "Batch size",
                "Workers",
                "Batch size",
                "Workers",
                "Batch size",
                "Workers",
            ),
            strict=True,
        ):
            label.setText(f"{tr(stage)} · {tr(value)}")
        self.runtime_labels[0].setText(tr("Model endpoint"))
        self.runtime_labels[1].setText(tr("Offline mode"))
        self.runtime_labels[2].setText(tr("Log level"))

    def load_options(
        self,
        options: PipelineOptions,
        *,
        language: str,
        theme: str,
        confirm_run: bool,
    ) -> None:
        self.language.setCurrentIndex(max(0, self.language.findData(language)))
        self.theme.setCurrentIndex(max(0, self.theme.findData(theme)))
        self.confirm_run.setChecked(confirm_run)
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
        self.tabs = QTabWidget()
        models_panel = QWidget()
        models_layout = QVBoxLayout(models_panel)
        top = QHBoxLayout()
        self.path_caption = QLabel()
        self.path_value = QLabel()
        self.path_value.setObjectName("muted")
        top.addWidget(self.path_caption)
        top.addWidget(self.path_value, 1)
        models_layout.addLayout(top)
        self.table = QTableWidget(0, 5)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.Stretch
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        models_layout.addWidget(self.table)
        self.tabs.addTab(models_panel, "")

        runtimes_panel = QWidget()
        runtimes_layout = QVBoxLayout(runtimes_panel)
        runtime_top = QHBoxLayout()
        self.runtime_path_caption = QLabel()
        self.runtime_path_value = QLabel()
        self.runtime_path_value.setObjectName("muted")
        runtime_top.addWidget(self.runtime_path_caption)
        runtime_top.addWidget(self.runtime_path_value, 1)
        runtimes_layout.addLayout(runtime_top)
        self.runtime_table = QTableWidget(0, 6)
        self.runtime_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.runtime_table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.Stretch
        )
        self.runtime_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        runtimes_layout.addWidget(self.runtime_table)
        self.tabs.addTab(runtimes_panel, "")
        layout.addWidget(self.tabs)
        self.retranslate()

    def retranslate(self) -> None:
        self.path_caption.setText(f"{tr('Model directory')}:")
        self.table.setHorizontalHeaderLabels(
            (tr("Key"), tr("Provider"), tr("Installed"), tr("Location"), tr("Notes"))
        )
        self.runtime_path_caption.setText(f"{tr('Runtime directory')}:")
        self.runtime_table.setHorizontalHeaderLabels(
            (
                tr("Provider"),
                tr("Status"),
                tr("Stability"),
                tr("Process architecture"),
                tr("Location"),
                tr("Detail"),
            )
        )
        self.tabs.setTabText(0, tr("Models"))
        self.tabs.setTabText(1, tr("Runtimes"))

    def refresh(self, options: PipelineOptions) -> None:
        models = list_models(options)
        path = options.models.cache_dir or get_application_paths().model_dir
        self.path_value.setText(str(path))
        self.table.setRowCount(0)
        for model in models:
            self._add(model)
        self.runtime_path_value.setText(str(get_application_paths().runtime_dir))
        self.runtime_table.setRowCount(0)
        for runtime in list_runtimes():
            self._add_runtime(runtime)

    def selected_key(self) -> str | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        return str(item.data(Qt.ItemDataRole.UserRole)) if item else None

    def selected_runtime(self) -> RuntimeStatus | None:
        row = self.runtime_table.currentRow()
        if row < 0:
            return None
        item = self.runtime_table.item(row, 0)
        value = item.data(Qt.ItemDataRole.UserRole) if item else None
        return value if isinstance(value, RuntimeStatus) else None

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

    def _add_runtime(self, runtime: RuntimeStatus) -> None:
        row = self.runtime_table.rowCount()
        self.runtime_table.insertRow(row)
        provider = QTableWidgetItem(runtime.descriptor.provider)
        provider.setData(Qt.ItemDataRole.UserRole, runtime)
        self.runtime_table.setItem(row, 0, provider)
        status = (
            tr("Installed")
            if runtime.installed
            else tr("Available")
            if runtime.descriptor.available
            else tr("Unavailable")
        )
        self.runtime_table.setItem(row, 1, QTableWidgetItem(status))
        self.runtime_table.setItem(
            row, 2, QTableWidgetItem(runtime.descriptor.stability.value)
        )
        self.runtime_table.setItem(
            row, 3, QTableWidgetItem(runtime.descriptor.process_arch)
        )
        self.runtime_table.setItem(
            row, 4, QTableWidgetItem(str(runtime.path) if runtime.path else "")
        )
        self.runtime_table.setItem(row, 5, QTableWidgetItem(runtime.detail))


class DiagnosticsPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.provider = QComboBox()
        self.provider.addItems(("fake", "faster-whisper", "qwen3-asr", "nemo-asr"))
        self.load_model = QCheckBox()
        self.provider_label = QLabel()
        form.addRow(self.provider_label, self.provider)
        form.addRow("", self.load_model)
        layout.addLayout(form)
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
        self.retranslate()

    def retranslate(self) -> None:
        self.provider_label.setText(tr("Provider"))
        self.load_model.setText(tr("Load model"))
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


__all__ = ["DiagnosticsPage", "ModelsPage", "RunPage", "SettingsPage"]
