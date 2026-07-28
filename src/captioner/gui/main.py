"""Desktop application entry point."""

import sys
from collections.abc import Sequence
from typing import cast

from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from captioner.gui.main_window import MainWindow


def select_ui_font() -> QFont:
    """Choose an installed cross-platform UI font with CJK preferences."""

    installed = set(QFontDatabase.families())
    for family in (
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Microsoft YaHei UI",
        "PingFang SC",
        "Segoe UI",
        "Helvetica Neue",
        "DejaVu Sans",
    ):
        if family in installed:
            return QFont(family, 10)
    return QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)


def main(argv: Sequence[str] | None = None) -> int:
    application = cast(QApplication | None, QCoreApplication.instance())
    owns_application = application is None
    if application is None:
        application = QApplication(list(argv) if argv is not None else sys.argv)
    application.setApplicationName("VideoCaptioner")
    application.setOrganizationName("VideoCaptioner")
    application.setFont(select_ui_font())
    window = MainWindow(application)
    window.show()
    return application.exec() if owns_application else 0


if __name__ == "__main__":
    raise SystemExit(main())
