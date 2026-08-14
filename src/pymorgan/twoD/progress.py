"""Progress tracking utility for GUI dialogs and terminal output."""

from __future__ import annotations

from ..log import get_logger

logger = get_logger(__name__)


class ProgressTracker:
    def __init__(self, total: int, title: str = "Progress", label: str = "Please wait...", is_gui: bool | None = None):
        self.total = total
        self.current = 0
        self.dialog = None
        self.is_gui = False

        if is_gui is False:
            self.is_gui = False
        else:
            try:
                from PyQt6.QtWidgets import QApplication
                app = QApplication.instance()
                if app is not None:
                    import os
                    # Avoid instantiating GUI widgets if running offscreen
                    if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                        self.is_gui = True
            except ImportError:
                self.is_gui = False

        if self.is_gui:
            try:
                from PyQt6.QtCore import Qt
                from PyQt6.QtWidgets import QApplication, QProgressDialog
                # Get the active window as parent to centre the dialog
                parent = QApplication.activeWindow()
                self.dialog = QProgressDialog(label, "Cancel", 0, total, parent)
                self.dialog.setWindowTitle(title)
                self.dialog.setWindowModality(Qt.WindowModality.WindowModal)
                self.dialog.setMinimumDuration(0)
                self.dialog.setValue(0)
                self.dialog.show()
                QApplication.processEvents()
            except Exception:
                self.is_gui = False
                self.dialog = None

        if not self.is_gui:
            self.title = title
            self.label = label
            self._print_terminal(0)

    def update(self, current: int, label: str | None = None):
        self.current = current
        if self.is_gui and self.dialog is not None:
            try:
                from PyQt6.QtWidgets import QApplication
                if label is not None:
                    self.dialog.setLabelText(label)
                self.dialog.setValue(self.current)
                QApplication.processEvents()
            except Exception:
                logger.debug("Progress-dialog update failed; continuing without it.", exc_info=True)
        else:
            self._print_terminal(self.current, label)

    def update_split(self, current: int, total: int, phase: str):
        if total <= 0:
            return

        self.total = 100
        if self.is_gui and self.dialog is not None:
            try:
                self.dialog.setMaximum(100)
            except Exception:
                logger.debug("Could not set the progress-dialog maximum.", exc_info=True)

        if phase == "loading":
            scaled_value = int(50 * current / total)
            label = f"Loading ({current} of {total})"
        else:
            scaled_value = int(50 + 50 * current / total)
            label = f"Processing ({current} of {total})"

        self.current = scaled_value

        if self.is_gui and self.dialog is not None:
            try:
                from PyQt6.QtWidgets import QApplication
                self.dialog.setLabelText(label)
                self.dialog.setValue(scaled_value)
                QApplication.processEvents()
            except Exception:
                logger.debug("Progress-dialog update failed; continuing without it.", exc_info=True)
        else:
            self._print_terminal(scaled_value, label)

    def set_total(self, total: int):
        self.total = total
        if self.is_gui and self.dialog is not None:
            try:
                self.dialog.setMaximum(total)
                self.dialog.setValue(0)
            except Exception:
                logger.debug("Could not reset the progress dialog.", exc_info=True)
        else:
            self._print_terminal(0)

    def _print_terminal(self, current: int, label: str | None = None):
        if self.total <= 0:
            return
        lbl = label if label is not None else self.label
        percent = f"{100 * (current / float(self.total)):.1f}"
        length = 30
        filled_length = int(length * current // self.total)
        bar = "█" * filled_length + '-' * (length - filled_length)
        print(f"\r{lbl} |{bar}| {percent}%", end="")
        if current >= self.total:
            print()

    def close(self):
        if self.is_gui and self.dialog is not None:
            try:
                self.dialog.close()
                self.dialog.deleteLater()
            except Exception:
                logger.debug("Closing the progress dialog failed.", exc_info=True)
            self.dialog = None
