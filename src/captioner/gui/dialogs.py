"""Focused modal and utility windows used by the desktop adapter."""

from collections import Counter
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from captioner.gui.i18n import tr
from captioner.gui.pages import DiagnosticsPage, ModelsPage, SettingsPage
from captioner.workflow.api import (
    ApplicationPlan,
    PipelineOptions,
    get_application_paths,
)


class ScanPreviewDialog(QDialog):
    def __init__(self, plan: ApplicationPlan, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(660, 440)
        layout = QVBoxLayout(self)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setObjectName("muted")
        layout.addWidget(self.summary)
        self.items = QListWidget()
        layout.addWidget(self.items, 1)
        buttons = QDialogButtonBox()
        self.use_button = buttons.addButton(
            tr("Use this input"), QDialogButtonBox.ButtonRole.AcceptRole
        )
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.set_plan(plan)
        self.retranslate()

    def set_plan(self, plan: ApplicationPlan) -> None:
        root = plan.input_root or plan.inputs[0].parent
        counts = Counter(path.suffix.lower() or "(none)" for path in plan.inputs)
        extensions = ", ".join(
            f"{suffix}: {count}" for suffix, count in sorted(counts.items())
        )
        self.summary.setText(
            f"{tr('Root')}: {root}\n"
            f"{tr('Supported files')}: {len(plan.inputs)}\n"
            f"{tr('Extensions')}: {extensions}"
        )
        self.items.clear()
        for path in plan.inputs[:100]:
            try:
                text = str(path.relative_to(root))
            except ValueError:
                text = str(path)
            self.items.addItem(text)
        if len(plan.inputs) > 100:
            self.items.addItem(tr("… and more"))

    def retranslate(self) -> None:
        self.setWindowTitle(tr("Scan preview"))
        self.use_button.setText(tr("Use this input"))


class RunConfirmationDialog(QDialog):
    def __init__(
        self,
        *,
        plan: ApplicationPlan,
        profile: str,
        stages: tuple[str, ...],
        formats: tuple[str, ...],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumWidth(540)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        values = (
            str(len(plan.inputs)),
            tr(plan.command.replace("_", " ").title()),
            profile or "—",
            ", ".join(stages) or "—",
            ", ".join(formats) or "—",
            str(plan.output_dir),
        )
        for label, value in zip(
            ("Files", "Mode", "Profile", "Stages", "Formats", "Output folder"),
            values,
            strict=True,
        ):
            value_label = QLabel(value)
            value_label.setWordWrap(True)
            value_label.setTextInteractionFlags(
                value_label.textInteractionFlags()
                | value_label.textInteractionFlags().TextSelectableByMouse
            )
            form.addRow(f"{tr(label)}:", value_label)
        layout.addLayout(form)
        self.dont_ask = QCheckBox(tr("Don't ask again"))
        layout.addWidget(self.dont_ask)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        self.start_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.start_button.setObjectName("primary")
        self.start_button.setText(tr("Start"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setWindowTitle(tr("Confirm run"))


class CompletionDialog(QDialog):
    def __init__(
        self,
        *,
        succeeded: int,
        failed: int,
        output_dir: Path,
        failures: tuple[str, ...] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)
        headline = QLabel(
            tr("Run completed") if failed == 0 else tr("Run completed with failures")
        )
        headline.setObjectName("brand")
        layout.addWidget(headline)
        summary = QLabel(
            f"{tr('Completed')}: {succeeded}    {tr('Failed')}: {failed}\n"
            f"{tr('Output folder')}: {output_dir}"
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)
        if failures:
            details = QPlainTextEdit("\n".join(failures))
            details.setReadOnly(True)
            details.setMinimumHeight(110)
            details.setVisible(False)
            layout.addWidget(details)
        else:
            details = None
        actions = QHBoxLayout()
        open_output = QPushButton(tr("Open output"))
        open_output.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_dir)))
        )
        actions.addWidget(open_output)
        if details is not None:
            view_failures = QPushButton(tr("View failures"))
            view_failures.clicked.connect(
                lambda: details.setVisible(not details.isVisible())
            )
            actions.addWidget(view_failures)
        actions.addStretch()
        close_button = QPushButton(tr("Close"))
        close_button.setObjectName("primary")
        close_button.clicked.connect(self.accept)
        actions.addWidget(close_button)
        layout.addLayout(actions)
        self.setWindowTitle(tr("Run summary"))


class ActivityLogDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.resize(760, 480)
        self._entries: list[tuple[str, str]] = []
        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        self.filter = QComboBox()
        self.filter.addItem("", "all")
        self.filter.addItem("", "process")
        self.filter.addItem("", "error")
        self.export_button = QPushButton()
        self.clear_button = QPushButton()
        toolbar.addWidget(self.filter)
        toolbar.addStretch()
        toolbar.addWidget(self.export_button)
        toolbar.addWidget(self.clear_button)
        layout.addLayout(toolbar)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(2000)
        layout.addWidget(self.text)
        self.filter.currentIndexChanged.connect(self._render)
        self.export_button.clicked.connect(self._export)
        self.clear_button.clicked.connect(self.clear)
        self.retranslate()

    def add_entry(self, message: str, *, error: bool = False) -> None:
        self._entries.append(("error" if error else "process", message))
        self._render()

    def clear(self) -> None:
        self._entries.clear()
        self._render()

    def retranslate(self) -> None:
        self.setWindowTitle(tr("Activity log"))
        self.filter.setItemText(0, tr("All"))
        self.filter.setItemText(1, tr("Process"))
        self.filter.setItemText(2, tr("Error"))
        self.export_button.setText(tr("Export"))
        self.clear_button.setText(tr("Clear"))

    def _render(self) -> None:
        selected = str(self.filter.currentData())
        self.text.setPlainText(
            "\n".join(
                message
                for kind, message in self._entries
                if selected == "all" or selected == kind
            )
        )

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            tr("Export activity log"),
            "captioner-activity.log",
            tr("Log files (*.log);;All files (*)"),
        )
        if not path:
            return
        try:
            Path(path).write_text(
                "\n".join(message for _kind, message in self._entries),
                encoding="utf-8",
            )
        except OSError as exc:
            QMessageBox.critical(self, tr("Export failed"), str(exc))


class SettingsDialog(QDialog):
    def __init__(
        self,
        options: PipelineOptions,
        *,
        language: str,
        theme: str,
        confirm_run: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.resize(700, 570)
        layout = QVBoxLayout(self)
        self.page = SettingsPage()
        self.page.load_options(
            options,
            language=language,
            theme=theme,
            confirm_run=confirm_run,
        )
        layout.addWidget(self.page)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Save
        )
        save_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        save_button.setObjectName("primary")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setWindowTitle(tr("Settings"))


class ModelsDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.resize(880, 520)
        layout = QVBoxLayout(self)
        self.page = ModelsPage()
        layout.addWidget(self.page)
        actions = QHBoxLayout()
        self.refresh_button = QPushButton(tr("Refresh"))
        self.download_button = QPushButton(tr("Download selected"))
        self.download_button.setObjectName("primary")
        actions.addStretch()
        actions.addWidget(self.refresh_button)
        actions.addWidget(self.download_button)
        layout.addLayout(actions)
        self.setWindowTitle(tr("Models"))

    def retranslate(self) -> None:
        self.setWindowTitle(tr("Models"))
        self.refresh_button.setText(tr("Refresh"))
        self.download_button.setText(tr("Download selected"))
        self.page.retranslate()


class DoctorDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.resize(760, 500)
        layout = QVBoxLayout(self)
        self.page = DiagnosticsPage()
        layout.addWidget(self.page)
        actions = QHBoxLayout()
        self.open_logs_button = QPushButton(tr("Open log folder"))
        self.run_button = QPushButton(tr("Run doctor"))
        self.run_button.setObjectName("primary")
        self.open_logs_button.clicked.connect(
            lambda: QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(get_application_paths().log_dir))
            )
        )
        actions.addStretch()
        actions.addWidget(self.open_logs_button)
        actions.addWidget(self.run_button)
        layout.addLayout(actions)
        self.setWindowTitle(tr("Diagnostics"))

    def retranslate(self) -> None:
        self.setWindowTitle(tr("Diagnostics"))
        self.open_logs_button.setText(tr("Open log folder"))
        self.run_button.setText(tr("Run doctor"))
        self.page.retranslate()


__all__ = [
    "ActivityLogDialog",
    "CompletionDialog",
    "DoctorDialog",
    "ModelsDialog",
    "RunConfirmationDialog",
    "ScanPreviewDialog",
    "SettingsDialog",
]
