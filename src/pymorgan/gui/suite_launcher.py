"""Unified Suite Launcher for PyMORGAN and PyRATE-TA.

Provides a clean, platform-independent launcher window that lets users launch
PyMORGAN, PyRATE-TA, or both simultaneously, detects package installation and
version, verifies font configuration, and suppresses command-line console windows.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from pymorgan.gui.console import hide_console, set_windows_app_id
from pymorgan.log import get_logger

logger = get_logger(__name__)


def is_pyrate_installed() -> bool:
    """Return True if PyRATE-TA is installed and importable."""
    return importlib.util.find_spec("pyrate_ta") is not None


def get_version(package_name: str) -> str:
    """Safely query installed package version."""
    try:
        return importlib.metadata.version(package_name)
    except Exception:
        try:
            mod = importlib.import_module(package_name)
            return getattr(mod, "__version__", "unknown")
        except Exception:
            return "not found"


def launch_process(module_name: str, args: list[str] | None = None) -> subprocess.Popen:
    """Launch a module in a detached background process with no console window."""
    cmd = [sys.executable, "-m", module_name]
    if args:
        cmd.extend(args)

    kwargs: dict = {}
    if sys.platform == "win32":
        # DETACHED_PROCESS (0x00000008) and CREATE_NEW_PROCESS_GROUP (0x00000200)
        # ensure child process is completely independent and has no console.
        kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
        kwargs["close_fds"] = True
    else:
        kwargs["start_new_session"] = True

    logger.info("Spawning detached process: %s", " ".join(cmd))
    return subprocess.Popen(cmd, **kwargs)


class SuiteLauncherDialog(QDialog):
    """Modern card-based launcher for PyMORGAN & PyRATE-TA."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PyMORGAN Suite \u2014 Spectroscopy & Kinetic Analysis")
        self.setFixedSize(680, 480)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(16)

        # Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)

        title_lbl = QLabel("Ultrafast Spectroscopy Suite")
        title_font = QFont()
        title_font.setPointSize(15)
        title_font.setBold(True)
        title_lbl.setFont(title_font)
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle_lbl = QLabel("Integrated 1D/2D Transient Absorption Processing & Kinetic Modeling")
        sub_font = QFont()
        sub_font.setPointSize(9)
        subtitle_lbl.setFont(sub_font)
        subtitle_lbl.setStyleSheet("color: #64748b;")
        subtitle_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        header_layout.addWidget(title_lbl)
        header_layout.addWidget(subtitle_lbl)
        main_layout.addLayout(header_layout)

        # App Cards Container
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(16)

        # Card 1: PyMORGAN
        self.card_pymorgan = self._build_app_card(
            name="PyMORGAN",
            version=get_version("pymorgan"),
            desc="Transient absorption & 2D-IR data processing, chirp correction, SVD, anisotropy, and stitching.",
            icon_name="pirate-hat.png",
            launch_action=self._launch_pymorgan,
            btn_text="Launch PyMORGAN",
            accent_color="#2563eb",
            is_installed=True,
        )
        cards_layout.addWidget(self.card_pymorgan)

        # Card 2: PyRATE-TA
        pyrate_present = is_pyrate_installed()
        pyrate_ver = get_version("pyrate_ta") if pyrate_present else "Not Installed"
        self.card_pyrate = self._build_app_card(
            name="PyRATE-TA",
            version=pyrate_ver,
            desc="Kinetic analysis of time-resolved spectroscopy: multi-exponential fitting, global and target analysis.",
            icon_name="pirate_ship.png",
            launch_action=self._launch_pyrate,
            btn_text="Launch PyRATE-TA" if pyrate_present else "Install PyRATE-TA",
            accent_color="#0d9488",
            is_installed=pyrate_present,
        )
        cards_layout.addWidget(self.card_pyrate)

        main_layout.addLayout(cards_layout)

        # Bottom Controls
        bottom_layout = QHBoxLayout()

        self.chk_close_on_launch = QCheckBox("Close launcher after starting application")
        self.chk_close_on_launch.setChecked(True)
        bottom_layout.addWidget(self.chk_close_on_launch)

        bottom_layout.addStretch()

        btn_both = QPushButton("Launch Both Side-by-Side")
        btn_both.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px 12px; border: 1px solid #cbd5e1; border-radius: 4px; background: #f8fafc; }"
            "QPushButton:hover { background: #e2e8f0; }"
        )
        btn_both.clicked.connect(self._launch_both)
        btn_both.setEnabled(pyrate_present)
        bottom_layout.addWidget(btn_both)

        btn_settings = QPushButton("Settings...")
        btn_settings.setStyleSheet(
            "QPushButton { padding: 6px 10px; border: 1px solid #cbd5e1; border-radius: 4px; background: #f8fafc; }"
            "QPushButton:hover { background: #e2e8f0; }"
        )
        btn_settings.clicked.connect(self._launch_settings)
        bottom_layout.addWidget(btn_settings)

        main_layout.addLayout(bottom_layout)

    def _build_app_card(
        self,
        name: str,
        version: str,
        desc: str,
        icon_name: str,
        launch_action,
        btn_text: str,
        accent_color: str,
        is_installed: bool,
    ) -> QFrame:
        """Create a styled interactive card for an application."""
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setStyleSheet(
            "QFrame { background: white; border: 1px solid #e2e8f0; border-radius: 8px; }"
        )
        vbox = QVBoxLayout(card)
        vbox.setContentsMargins(16, 16, 16, 16)
        vbox.setSpacing(10)

        # Top row: icon + title + version
        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        icon_lbl = QLabel()
        icon_path = Path(__file__).parent / "icons" / icon_name
        # Fallback to sibling PyRATE-TA icon if pirate_ship.png is queried
        if not icon_path.exists() and icon_name == "pirate_ship.png":
            alt_path = Path(__file__).resolve().parents[4] / "PyRATE-TA" / "src" / "pyrate_ta" / "gui" / "icons" / "pirate_ship.png"
            if alt_path.exists():
                icon_path = alt_path

        if icon_path.exists():
            pix = QPixmap(str(icon_path)).scaled(
                44, 44, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            icon_lbl.setPixmap(pix)
        else:
            icon_lbl.setText("APP")
            icon_lbl.setStyleSheet("font-weight: bold; font-size: 14px; color: #64748b;")

        top_row.addWidget(icon_lbl)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        lbl_name = QLabel(name)
        f_name = QFont()
        f_name.setPointSize(12)
        f_name.setBold(True)
        lbl_name.setFont(f_name)

        status_text = f"v{version}" if is_installed else "Not Installed"
        lbl_ver = QLabel(status_text)
        ver_color = "#15803d" if is_installed else "#b91c1c"
        lbl_ver.setStyleSheet(f"font-size: 9pt; color: {ver_color}; font-weight: 500;")

        title_box.addWidget(lbl_name)
        title_box.addWidget(lbl_ver)
        top_row.addLayout(title_box)
        top_row.addStretch()

        vbox.addLayout(top_row)

        # Description
        lbl_desc = QLabel(desc)
        lbl_desc.setWordWrap(True)
        lbl_desc.setStyleSheet("color: #475569; font-size: 9pt; min-height: 85px;")
        lbl_desc.setAlignment(Qt.AlignmentFlag.AlignTop)
        vbox.addWidget(lbl_desc)

        # Button
        btn = QPushButton(btn_text)
        if is_installed:
            btn.setStyleSheet(
                f"QPushButton {{ background-color: {accent_color}; color: white; font-weight: bold; padding: 10px; border-radius: 5px; font-size: 10pt; }}"
                f"QPushButton:hover {{ opacity: 0.9; }}"
            )
            btn.clicked.connect(launch_action)
        else:
            btn.setStyleSheet(
                "QPushButton { background-color: #f1f5f9; color: #475569; font-weight: bold; padding: 10px; border: 1px solid #cbd5e1; border-radius: 5px; font-size: 9.5pt; }"
            )
            btn.clicked.connect(self._show_pyrate_install_instructions)

        vbox.addWidget(btn)
        return card

    def _launch_pymorgan(self):
        """Launch PyMORGAN."""
        try:
            launch_process("pymorgan")
            if self.chk_close_on_launch.isChecked():
                self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "Launch Error", f"Could not launch PyMORGAN:\n{exc}")

    def _launch_pyrate(self):
        """Launch PyRATE-TA."""
        if not is_pyrate_installed():
            self._show_pyrate_install_instructions()
            return
        try:
            launch_process("pyrate_ta.gui.app")
            if self.chk_close_on_launch.isChecked():
                self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "Launch Error", f"Could not launch PyRATE-TA:\n{exc}")

    def _launch_both(self):
        """Launch both PyMORGAN and PyRATE-TA concurrently."""
        try:
            launch_process("pymorgan")
            launch_process("pyrate_ta.gui.app")
            if self.chk_close_on_launch.isChecked():
                self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "Launch Error", f"Could not launch applications:\n{exc}")

    def _launch_settings(self):
        """Launch the PyMORGAN settings configuration window."""
        try:
            launch_process("pymorgan.gui.settings_app")
        except Exception as exc:
            QMessageBox.critical(self, "Settings Error", f"Could not launch settings:\n{exc}")

    def _show_pyrate_install_instructions(self):
        """Show dialog explaining how to install PyRATE-TA."""
        msg = (
            "PyRATE-TA is the companion kinetic analysis package for PyMORGAN.\n\n"
            "To install PyRATE-TA in your environment, run:\n"
            "    uv pip install pyrate-ta\n\n"
            "Or clone from GitHub:\n"
            "    https://github.com/RJFernandezTeran/PyRATE-TA"
        )
        QMessageBox.information(self, "Install PyRATE-TA", msg)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the PyMORGAN Suite Launcher."""
    parser = argparse.ArgumentParser(description="PyMORGAN Suite Launcher")
    parser.add_argument("--pymorgan", action="store_true", help="Launch PyMORGAN directly")
    parser.add_argument("--pyrate", action="store_true", help="Launch PyRATE-TA directly")
    args, unknown = parser.parse_known_args(argv)

    # Fast dispatch if flag is provided
    if args.pymorgan:
        from pymorgan.launcher import main as morgan_main

        return morgan_main(unknown)

    if args.pyrate:
        if is_pyrate_installed():
            launch_process("pyrate_ta.gui.app", unknown)
            return 0
        print("[PyMORGAN Suite] PyRATE-TA is not installed.")
        return 1

    # Hide host console window if running on Windows
    set_windows_app_id("PyMORGAN.SuiteLauncher")
    hide_console()

    # Ensure preferred fonts are installed and registered with Matplotlib early
    try:
        from pymorgan.fonts import ensure_fonts_installed

        ensure_fonts_installed(quiet=True)
    except Exception:
        pass

    app = QApplication.instance() or QApplication(sys.argv)
    icon_path = Path(__file__).parent / "icons" / "pirate-hat.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    dialog = SuiteLauncherDialog()
    dialog.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
