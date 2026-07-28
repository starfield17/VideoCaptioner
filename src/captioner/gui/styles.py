"""Visual identity for the caption production console."""

APP_STYLE = """
QMainWindow, QWidget {
    background: #101a24;
    color: #e7eef3;
    font-size: 13px;
}
QFrame#navRail {
    background: #0b141d;
    border-right: 1px solid #263746;
}
QLabel#brand {
    color: #f3b562;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 1px;
}
QLabel#subtitle, QLabel#muted {
    color: #8fa6b6;
}
QPushButton {
    background: #213443;
    border: 1px solid #345064;
    border-radius: 6px;
    min-height: 30px;
    padding: 2px 13px;
}
QPushButton:hover { background: #294355; border-color: #35b6a8; }
QPushButton:focus { border: 2px solid #f3b562; }
QPushButton#primary {
    background: #35b6a8;
    border-color: #35b6a8;
    color: #081510;
    font-weight: 700;
}
QPushButton#danger { color: #ffb4a9; border-color: #8d4b49; }
QPushButton:disabled { color: #607383; background: #172431; }
QListWidget#navigation {
    background: transparent;
    border: 0;
    outline: 0;
}
QListWidget#navigation::item {
    border-left: 3px solid transparent;
    padding: 12px 14px;
    margin: 2px 0;
}
QListWidget#navigation::item:selected {
    background: #172b37;
    border-left-color: #35b6a8;
    color: #ffffff;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTableWidget {
    background: #172431;
    border: 1px solid #2c4354;
    border-radius: 5px;
    min-height: 29px;
    selection-background-color: #287f79;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { padding: 2px 7px; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #35b6a8;
}
QGroupBox {
    border: 1px solid #263b4b;
    border-radius: 8px;
    margin-top: 12px;
    padding: 16px 12px 12px 12px;
    font-weight: 600;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
QHeaderView::section {
    background: #1d303e;
    color: #a9bdc9;
    border: 0;
    border-bottom: 1px solid #345064;
    padding: 7px;
}
QTableWidget { gridline-color: #263b4b; }
QProgressBar {
    border: 1px solid #2c4354;
    border-radius: 4px;
    background: #172431;
    height: 8px;
    text-align: center;
}
QProgressBar::chunk { background: #35b6a8; border-radius: 3px; }
QCheckBox { spacing: 7px; }
QCheckBox::indicator { width: 16px; height: 16px; }
QCheckBox::indicator:checked { background: #35b6a8; border: 1px solid #65d4c8; }
QScrollBar:vertical { background: #101a24; width: 10px; }
QScrollBar::handle:vertical { background: #345064; border-radius: 5px; }
"""

__all__ = ["APP_STYLE"]
