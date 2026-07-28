"""Light and dark visual themes for the desktop production console."""

from typing import Literal

ThemeName = Literal["light", "dark"]

_TOKENS = {
    "light": {
        "canvas": "#F3F5F8",
        "surface": "#FFFFFF",
        "surface_alt": "#F8FAFC",
        "text": "#0F172A",
        "muted": "#64748B",
        "border": "#D8E0EA",
        "primary": "#2563EB",
        "primary_hover": "#1D4ED8",
        "accent": "#D97706",
        "danger": "#B42318",
        "disabled": "#A3AFBF",
        "selection": "#DBEAFE",
    },
    "dark": {
        "canvas": "#101722",
        "surface": "#172232",
        "surface_alt": "#1D2A3B",
        "text": "#E6EDF5",
        "muted": "#96A8BA",
        "border": "#34465A",
        "primary": "#60A5FA",
        "primary_hover": "#93C5FD",
        "accent": "#F3B562",
        "danger": "#FDA29B",
        "disabled": "#617489",
        "selection": "#244A73",
    },
}


def style_for(theme: ThemeName | str) -> str:
    """Return the complete application stylesheet for a supported theme."""

    selected = theme if theme in _TOKENS else "light"
    colors = _TOKENS[selected]
    return f"""
QMainWindow, QDialog, QWidget#appRoot {{
    background: {colors["canvas"]};
    color: {colors["text"]};
    font-size: 13px;
}}
QWidget {{
    color: {colors["text"]};
}}
QToolBar {{
    background: {colors["surface"]};
    border: 0;
    border-bottom: 1px solid {colors["border"]};
    spacing: 4px;
    padding: 7px 10px;
}}
QToolBar QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    min-width: 68px;
    min-height: 38px;
    padding: 3px 8px;
}}
QToolBar QToolButton:hover {{
    background: {colors["surface_alt"]};
    border-color: {colors["border"]};
}}
QToolBar QToolButton:checked {{
    background: {colors["selection"]};
    color: {colors["primary"]};
}}
QLabel#brand {{
    color: {colors["text"]};
    font-size: 22px;
    font-weight: 700;
}}
QLabel#eyebrow {{
    color: {colors["primary"]};
    font-size: 11px;
    font-weight: 700;
}}
QLabel#muted, QLabel#dropHint, QLabel#metricCaption {{
    color: {colors["muted"]};
}}
QLabel#metricValue {{
    font-size: 18px;
    font-weight: 700;
}}
QFrame#metricCard, QFrame#stageRibbon {{
    background: {colors["surface_alt"]};
    border: 1px solid {colors["border"]};
    border-radius: 7px;
}}
QLabel#stageChip {{
    background: transparent;
    color: {colors["muted"]};
    border: 1px solid {colors["border"]};
    border-radius: 10px;
    padding: 3px 8px;
    font-size: 10px;
    font-weight: 600;
}}
QLabel#stageChip[state="active"] {{
    background: {colors["selection"]};
    border-color: {colors["primary"]};
    color: {colors["primary"]};
}}
QLabel#stageChip[state="done"] {{
    background: {colors["primary"]};
    border-color: {colors["primary"]};
    color: {colors["surface"]};
}}
QPushButton {{
    background: {colors["surface"]};
    border: 1px solid {colors["border"]};
    border-radius: 6px;
    min-height: 30px;
    padding: 2px 13px;
}}
QPushButton:hover {{
    background: {colors["surface_alt"]};
    border-color: {colors["primary"]};
}}
QPushButton:focus {{
    border: 2px solid {colors["accent"]};
}}
QPushButton#primary {{
    background: {colors["primary"]};
    border-color: {colors["primary"]};
    color: {colors["surface"]};
    font-weight: 700;
}}
QPushButton#primary:hover {{
    background: {colors["primary_hover"]};
}}
QPushButton#danger {{
    color: {colors["danger"]};
    border-color: {colors["danger"]};
}}
QPushButton:disabled {{
    color: {colors["disabled"]};
    background: {colors["surface_alt"]};
}}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit,
QTableWidget, QListWidget {{
    background: {colors["surface"]};
    border: 1px solid {colors["border"]};
    border-radius: 5px;
    min-height: 29px;
    selection-background-color: {colors["selection"]};
    selection-color: {colors["text"]};
}}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    padding: 2px 7px;
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus,
QDoubleSpinBox:focus, QPlainTextEdit:focus {{
    border-color: {colors["primary"]};
}}
QGroupBox {{
    background: {colors["surface"]};
    border: 1px solid {colors["border"]};
    border-radius: 8px;
    margin-top: 12px;
    padding: 16px 12px 12px 12px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}}
QTabWidget::pane {{
    background: {colors["surface"]};
    border: 1px solid {colors["border"]};
    border-radius: 7px;
}}
QTabBar::tab {{
    background: transparent;
    color: {colors["muted"]};
    border-bottom: 2px solid transparent;
    padding: 8px 16px;
}}
QTabBar::tab:selected {{
    color: {colors["primary"]};
    border-bottom-color: {colors["primary"]};
}}
QHeaderView::section {{
    background: {colors["surface_alt"]};
    color: {colors["muted"]};
    border: 0;
    border-bottom: 1px solid {colors["border"]};
    padding: 7px;
    font-weight: 600;
}}
QTableWidget {{
    gridline-color: {colors["border"]};
}}
QProgressBar {{
    border: 1px solid {colors["border"]};
    border-radius: 4px;
    background: {colors["surface_alt"]};
    min-height: 9px;
    max-height: 9px;
    text-align: center;
}}
QProgressBar::chunk {{
    background: {colors["primary"]};
    border-radius: 3px;
}}
QCheckBox {{
    spacing: 7px;
}}
QStatusBar {{
    background: {colors["surface"]};
    border-top: 1px solid {colors["border"]};
    color: {colors["muted"]};
}}
QScrollBar:vertical {{
    background: {colors["canvas"]};
    width: 10px;
}}
QScrollBar::handle:vertical {{
    background: {colors["border"]};
    border-radius: 5px;
}}
"""


APP_STYLE = style_for("light")

__all__ = ["APP_STYLE", "ThemeName", "style_for"]
