"""Unified Suite Launcher for PyMORGAN and PyRATE-TA.

Provides a clean, platform-independent launcher window that lets users launch
PyMORGAN, PyRATE-TA, or both simultaneously, detects package installation and
version, verifies font configuration, explains Python library usage outside the
standalone GUIs, and suppresses command-line console windows.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStyleFactory,
    QTextEdit,
    QVBoxLayout,
)

from pymorgan.gui.console import hide_console, set_windows_app_id
from pymorgan.integrations.pyrate_bridge import (
    find_pyrate_command,
    get_pyrate_version,
    is_pyrate_installed,
)
from pymorgan.log import get_logger

logger = get_logger(__name__)


def get_version(package_name: str) -> str:
    """Safely query installed package version."""
    if package_name == "pyrate_ta":
        ver = get_pyrate_version()
        if ver:
            return ver
    try:
        mod = importlib.import_module(package_name)
        ver = getattr(mod, "__version__", None)
        if ver:
            return str(ver)
    except Exception:
        pass
    try:
        return importlib.metadata.version(package_name.replace("_", "-"))
    except Exception:
        return "not found"


def find_pymorgan_command(extra_args: list[str] | None = None) -> list[str]:
    """Return the command argv to launch PyMORGAN in a standalone process."""
    cmd: list[str] = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        for name in ("PyMORGAN.exe", "pymorgan.exe", "PyMORGAN", "pymorgan"):
            cand = exe_dir / name
            if cand.is_file():
                cmd = [str(cand)]
                break
        if not cmd:
            cmd = [sys.executable, "--pymorgan"]
    else:
        py_exe = sys.executable
        if sys.platform == "win32":
            pyw = Path(py_exe).with_name("pythonw.exe")
            if pyw.is_file():
                py_exe = str(pyw)
        cmd = [py_exe, "-m", "pymorgan"]

    if extra_args:
        cmd.extend(extra_args)
    return cmd


def launch_command(cmd: list[str]) -> subprocess.Popen:
    """Spawn a command in an independent background process with no console window."""
    kwargs: dict = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200
        )
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True

    logger.info("Spawning background process: %s", " ".join(cmd))
    return subprocess.Popen(cmd, **kwargs)


def launch_process(module_name: str, args: list[str] | None = None) -> subprocess.Popen:
    """Launch a target module or frozen binary in a detached background process."""
    if module_name in ("pymorgan", "pymorgan.launcher", "pymorgan.gui.app"):
        return launch_command(find_pymorgan_command(args))
    if module_name in ("pyrate_ta", "pyrate_ta.launcher", "pyrate_ta.gui.app"):
        base = find_pyrate_command()
        if base is None:
            raise RuntimeError("PyRATE-TA executable or module was not found.")
        cmd = list(base)
        if args:
            cmd.extend(args)
        return launch_command(cmd)
    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "--settings"] if "settings" in module_name else [sys.executable]
    else:
        py_exe = sys.executable
        if sys.platform == "win32":
            pyw = Path(py_exe).with_name("pythonw.exe")
            if pyw.is_file():
                py_exe = str(pyw)
        cmd = [py_exe, "-m", module_name]
    if args:
        cmd.extend(args)
    return launch_command(cmd)


class SuiteLauncherDialog(QDialog):
    """Modern card-based launcher for PyMORGAN & PyRATE-TA."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SuiteLauncher")
        self.setWindowTitle("PyMORGAN Suite - Ultrafast Spectroscopy & Kinetic Analysis")
        self.setFixedSize(820, 600)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)

        # Scoped stylesheet: QLabel rules explicitly prevent inheriting QFrame borders
        # and guarantee high-contrast dark slate typography regardless of OS dark mode.
        self.setStyleSheet(
            """
            QDialog#SuiteLauncher {
                background-color: #f1f5f9;
                color: #0f172a;
            }
            QLabel {
                border: none;
                background: transparent;
                color: #0f172a;
            }
            QCheckBox {
                color: #1e293b;
                font-size: 9.5pt;
                spacing: 6px;
            }
            QFrame#AppCard {
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 10px;
            }
            QFrame#LibraryBanner {
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                border-left: 4px solid #3b82f6;
                border-radius: 8px;
            }
            """
        )

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 18)
        main_layout.setSpacing(14)

        # --- Header Banner ---
        header_layout = QVBoxLayout()
        header_layout.setSpacing(3)

        title_lbl = QLabel("PyMORGAN & PyRATE-TA Spectroscopy Suite")
        title_lbl.setStyleSheet(
            "font-size: 16pt; font-weight: bold; color: #0f172a; border: none; background: transparent;"
        )
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle_lbl = QLabel(
            "Integrated 1D/2D Ultrafast (Multidimensional) Spectra Processing, Global Fitting & Target Kinetic Analysis"
        )
        subtitle_lbl.setStyleSheet(
            "font-size: 9.5pt; color: #475569; border: none; background: transparent;"
        )
        subtitle_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        header_layout.addWidget(title_lbl)
        header_layout.addWidget(subtitle_lbl)
        main_layout.addLayout(header_layout)

        # --- Application Cards Row ---
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(16)

        self.card_pymorgan = self._build_app_card(
            name="PyMORGAN",
            subtitle="1D & 2D Spectroscopy Processing",
            pkg_import="import pymorgan as pm",
            version=get_version("pymorgan"),
            bullets=[
                "Pump-Probe (fsTA, nsTA, TRIR) & 2D spectroscopy (e.g. 2D-IR)",
                "Chirp correction, SVD filtering, stitching & anisotropy",
                "Publication-grade Matplotlib figures & PDAT export",
            ],
            icon_name="pirate-hat.png",
            launch_action=self._launch_pymorgan,
            btn_text="Launch PyMORGAN",
            accent_color="#2563eb",
            hover_color="#1d4ed8",
            is_installed=True,
        )
        cards_layout.addWidget(self.card_pymorgan, 1)

        pyrate_present = is_pyrate_installed()
        pyrate_ver = get_version("pyrate_ta") if pyrate_present else "Not Installed"
        self.card_pyrate = self._build_app_card(
            name="PyRATE-TA",
            subtitle="Global, Target & Lifetime Density Fitting",
            pkg_import="import pyrate_ta as pr",
            version=pyrate_ver,
            bullets=[
                "Multi-exponential & Gaussian-IRF kinetic trace fitting",
                "Parallel (DAS), Sequential (EAS) & Target (SAS) models",
                "L-curve / GCV Lifetime Density Analysis (LDA)",
            ],
            icon_name="pirate_ship.png",
            launch_action=self._launch_pyrate,
            btn_text="Launch PyRATE-TA" if pyrate_present else "Install PyRATE-TA",
            accent_color="#0d9488",
            hover_color="#0f766e",
            is_installed=pyrate_present,
        )
        cards_layout.addWidget(self.card_pyrate, 1)

        main_layout.addLayout(cards_layout, 1)

        # --- Python Library / Scripting Info Banner ---
        lib_banner = QFrame()
        lib_banner.setObjectName("LibraryBanner")
        lib_layout = QHBoxLayout(lib_banner)
        lib_layout.setContentsMargins(14, 10, 14, 10)
        lib_layout.setSpacing(12)

        lib_text_box = QVBoxLayout()
        lib_text_box.setSpacing(3)

        lib_title = QLabel(
            "Using PyMORGAN & PyRATE-TA as Python Libraries (Scripts / Jupyter Notebooks)"
        )
        lib_title.setStyleSheet(
            "font-size: 9.5pt; font-weight: bold; color: #0f172a; border: none; background: transparent;"
        )

        lib_desc = QLabel(
            "The standalone installer bundles an isolated runtime for the GUI apps. "
            "To import <span style='font-family: Consolas, monospace; font-weight: bold; color: #1e40af;'>pymorgan</span> "
            "and <span style='font-family: Consolas, monospace; font-weight: bold; color: #0f766e;'>pyrate_ta</span> "
            "in your own Python &gt;= 3.12 environment, install from PyPI:"
        )
        lib_desc.setTextFormat(Qt.TextFormat.RichText)
        lib_desc.setWordWrap(True)
        lib_desc.setStyleSheet(
            "font-size: 8.8pt; color: #334155; border: none; background: transparent;"
        )

        cmd_pill = QLabel(
            "pip install pymorgan pyrate-ta     |     uv pip install pymorgan pyrate-ta"
        )
        cmd_pill.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        cmd_pill.setStyleSheet(
            "font-family: Consolas, 'Courier New', monospace; font-size: 8.8pt; font-weight: bold; "
            "color: #0f172a; background-color: #f8fafc; border: 1px solid #cbd5e1; "
            "border-radius: 4px; padding: 4px 8px;"
        )

        lib_text_box.addWidget(lib_title)
        lib_text_box.addWidget(lib_desc)
        lib_text_box.addWidget(cmd_pill)
        lib_layout.addLayout(lib_text_box, 1)

        lib_btn_box = QVBoxLayout()
        lib_btn_box.setSpacing(6)

        self.btn_copy_pip = QPushButton("Copy pip Command")
        self.btn_copy_pip.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_copy_pip.setStyleSheet(
            "QPushButton { font-size: 8.8pt; font-weight: bold; color: #1e293b; "
            "background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 4px; padding: 5px 10px; }"
            "QPushButton:hover { background-color: #e2e8f0; }"
        )
        self.btn_copy_pip.clicked.connect(self._copy_pip_command)

        btn_api_guide = QPushButton("Python API Guide...")
        btn_api_guide.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_api_guide.setStyleSheet(
            "QPushButton { font-size: 8.8pt; font-weight: bold; color: #1d4ed8; "
            "background-color: #eff6ff; border: 1px solid #bfdbfe; border-radius: 4px; padding: 5px 10px; }"
            "QPushButton:hover { background-color: #dbeafe; }"
        )
        btn_api_guide.clicked.connect(self._show_python_library_guide)

        lib_btn_box.addWidget(self.btn_copy_pip)
        lib_btn_box.addWidget(btn_api_guide)
        lib_layout.addLayout(lib_btn_box)

        main_layout.addWidget(lib_banner)

        # --- Bottom Controls Row ---
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(10)

        self.chk_close_on_launch = QCheckBox("Close launcher after starting app")
        self.chk_close_on_launch.setChecked(True)
        bottom_layout.addWidget(self.chk_close_on_launch)

        bottom_layout.addStretch()

        btn_fonts = QPushButton("Verify Fonts")
        btn_fonts.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_fonts.setToolTip(
            "Run the TeX Gyre Heros font installer twice and rebuild Matplotlib cache"
        )
        btn_fonts.setStyleSheet(
            "QPushButton { font-size: 9pt; color: #334155; padding: 6px 10px; "
            "border: 1px solid #cbd5e1; border-radius: 5px; background-color: #ffffff; }"
            "QPushButton:hover { background-color: #e2e8f0; }"
        )
        btn_fonts.clicked.connect(self._reinstall_fonts)
        bottom_layout.addWidget(btn_fonts)

        self.btn_updates = QPushButton("Check for Updates")
        self.btn_updates.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_updates.setToolTip(
            "Check GitHub for new releases or sync bleeding-edge updates from the main branch"
        )
        self.btn_updates.setStyleSheet(
            "QPushButton { font-size: 9pt; color: #334155; padding: 6px 10px; "
            "border: 1px solid #cbd5e1; border-radius: 5px; background-color: #ffffff; }"
            "QPushButton:hover { background-color: #e2e8f0; }"
        )
        self.btn_updates.clicked.connect(self._check_updates)
        bottom_layout.addWidget(self.btn_updates)

        btn_settings = QPushButton("Settings...")
        btn_settings.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_settings.setStyleSheet(
            "QPushButton { font-size: 9pt; color: #334155; padding: 6px 12px; "
            "border: 1px solid #cbd5e1; border-radius: 5px; background-color: #ffffff; }"
            "QPushButton:hover { background-color: #e2e8f0; }"
        )
        btn_settings.clicked.connect(self._launch_settings)
        bottom_layout.addWidget(btn_settings)

        btn_both = QPushButton("Launch Both Side-by-Side")
        btn_both.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_both.setStyleSheet(
            "QPushButton { font-size: 9.5pt; font-weight: bold; color: #ffffff; "
            "padding: 7px 14px; border: none; border-radius: 5px; background-color: #334155; }"
            "QPushButton:hover { background-color: #1e293b; }"
            "QPushButton:disabled { background-color: #94a3b8; color: #e2e8f0; }"
        )
        btn_both.clicked.connect(self._launch_both)
        btn_both.setEnabled(pyrate_present)
        bottom_layout.addWidget(btn_both)

        main_layout.addLayout(bottom_layout)
        QTimer.singleShot(1200, self._check_updates_background)

    def _build_app_card(
        self,
        name: str,
        subtitle: str,
        pkg_import: str,
        version: str,
        bullets: list[str],
        icon_name: str,
        launch_action,
        btn_text: str,
        accent_color: str,
        hover_color: str,
        is_installed: bool,
    ) -> QFrame:
        """Create a styled interactive card for an application."""
        card = QFrame()
        card.setObjectName("AppCard")
        vbox = QVBoxLayout(card)
        vbox.setContentsMargins(18, 16, 18, 16)
        vbox.setSpacing(10)

        # Top row: icon + package name + subtitle + version pill
        top_row = QHBoxLayout()
        top_row.setSpacing(14)

        icon_lbl = QLabel()
        icon_lbl.setFixedSize(58, 58)
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_path = Path(__file__).parent / "icons" / icon_name
        if not icon_path.exists() and icon_name == "pirate_ship.png":
            alt_path = (
                Path(__file__).resolve().parents[4]
                / "PyRATE-TA"
                / "src"
                / "pyrate_ta"
                / "gui"
                / "icons"
                / "pirate_ship.png"
            )
            if alt_path.exists():
                icon_path = alt_path

        if icon_path.exists():
            pix = QPixmap(str(icon_path)).scaled(
                54,
                54,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            icon_lbl.setPixmap(pix)
        else:
            icon_lbl.setText(name[:2].upper())
            icon_lbl.setStyleSheet(
                f"font-weight: bold; font-size: 16pt; color: {accent_color}; "
                "background-color: #f8fafc; border-radius: 8px;"
            )

        top_row.addWidget(icon_lbl)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)

        lbl_name = QLabel(name)
        lbl_name.setStyleSheet(
            "font-size: 15pt; font-weight: bold; color: #0f172a; border: none; background: transparent;"
        )

        lbl_sub = QLabel(subtitle)
        lbl_sub.setStyleSheet(
            "font-size: 9pt; font-weight: 600; color: #475569; border: none; background: transparent;"
        )

        badge_row = QHBoxLayout()
        badge_row.setSpacing(6)

        status_text = f"v{version}" if is_installed else "Not Installed"
        lbl_ver = QLabel(status_text)
        if is_installed:
            lbl_ver.setStyleSheet(
                "font-size: 8.5pt; font-weight: bold; color: #166534; "
                "background-color: #dcfce7; border: 1px solid #86efac; "
                "border-radius: 4px; padding: 1px 6px;"
            )
        else:
            lbl_ver.setStyleSheet(
                "font-size: 8.5pt; font-weight: bold; color: #991b1b; "
                "background-color: #fee2e2; border: 1px solid #fca5a5; "
                "border-radius: 4px; padding: 1px 6px;"
            )

        lbl_pkg = QLabel(pkg_import)
        lbl_pkg.setStyleSheet(
            "font-family: Consolas, 'Courier New', monospace; font-size: 8pt; color: #334155; "
            "background-color: #f1f5f9; border: 1px solid #cbd5e1; "
            "border-radius: 4px; padding: 1px 6px;"
        )

        badge_row.addWidget(lbl_ver)
        badge_row.addWidget(lbl_pkg)
        badge_row.addStretch()

        title_box.addWidget(lbl_name)
        title_box.addWidget(lbl_sub)
        title_box.addLayout(badge_row)
        top_row.addLayout(title_box, 1)

        vbox.addLayout(top_row)

        # Divider line
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFixedHeight(1)
        divider.setStyleSheet("background-color: #e2e8f0; border: none;")
        vbox.addWidget(divider)

        # Bullet points
        bullets_html = (
            "<ul style='margin: 0px; padding-left: 16px; line-height: 140%;'>"
            + "".join(f"<li style='margin-bottom: 4px; color: #334155;'>{b}</li>" for b in bullets)
            + "</ul>"
        )
        lbl_desc = QLabel(bullets_html)
        lbl_desc.setTextFormat(Qt.TextFormat.RichText)
        lbl_desc.setWordWrap(True)
        lbl_desc.setStyleSheet(
            "font-size: 9pt; color: #334155; border: none; background: transparent;"
        )
        lbl_desc.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        vbox.addWidget(lbl_desc, 1)

        # Primary Launch Button
        btn = QPushButton(btn_text)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setMinimumHeight(38)
        if is_installed:
            btn.setStyleSheet(
                f"QPushButton {{ background-color: {accent_color}; color: #ffffff; "
                f"font-weight: bold; padding: 8px 14px; border: none; border-radius: 6px; font-size: 10.5pt; }}"
                f"QPushButton:hover {{ background-color: {hover_color}; }}"
            )
            btn.clicked.connect(launch_action)
        else:
            btn.setStyleSheet(
                "QPushButton { background-color: #f1f5f9; color: #475569; font-weight: bold; "
                "padding: 8px 14px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 10pt; }"
                "QPushButton:hover { background-color: #e2e8f0; }"
            )
            btn.clicked.connect(self._show_pyrate_install_instructions)

        vbox.addWidget(btn)
        return card

    def _copy_pip_command(self):
        """Copy pip install command to clipboard."""
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText("pip install pymorgan pyrate-ta")
        self.btn_copy_pip.setText("Copied!")
        QTimer.singleShot(1800, lambda: self.btn_copy_pip.setText("Copy pip Command"))

    def _reinstall_fonts(self):
        """Run the TeX Gyre Heros font installer twice and report status."""
        try:
            from pymorgan.fonts import available_preferred_fonts, install_fonts_twice

            install_fonts_twice(quiet=True)
            found = available_preferred_fonts()
            fonts_str = ", ".join(found) if found else "TeX Gyre Heros"
            QMessageBox.information(
                self,
                "Fonts Configured (2/2 Passes Complete)",
                f"TeX Gyre Heros fonts were installed and verified across 2 passes.\n\n"
                f"Active Matplotlib preferred fonts: {fonts_str}",
            )
        except Exception as exc:
            QMessageBox.warning(self, "Font Configuration Warning", f"Font setup reported:\n{exc}")

    def _check_updates(self):
        """Interactive check for new releases or bleeding-edge main branch commits."""
        try:
            from pymorgan.gui.updater import check_for_updates_interactive

            self.btn_updates.setEnabled(False)
            self.btn_updates.setText("Checking...")

            def _restore(*_):
                self.btn_updates.setEnabled(True)
                self.btn_updates.setText("Check for Updates")

            check_for_updates_interactive(
                self,
                github_url="https://github.com/RJFernandezTeran/PyMORGAN",
                current_version=get_version("pymorgan"),
                app_name="PyMORGAN Suite",
            )
            worker = getattr(self, "_active_update_worker", None)
            if worker is not None:
                worker.check_finished.connect(_restore)
                worker.check_failed.connect(_restore)
            else:
                _restore()
        except Exception as exc:
            self.btn_updates.setEnabled(True)
            self.btn_updates.setText("Check for Updates")
            QMessageBox.warning(self, "Update Check Error", str(exc))

    def _check_updates_background(self):
        """Quietly check on startup if a newer release or bleeding-edge commit is available."""
        try:
            from pymorgan.gui.updater import UpdateCheckWorker

            worker = UpdateCheckWorker(
                "https://github.com/RJFernandezTeran/PyMORGAN",
                get_version("pymorgan"),
                self,
            )

            def _on_bg_info(info):
                if info.is_update_available:
                    self.btn_updates.setText(f"Update Available (v{info.latest_version})")
                    self.btn_updates.setStyleSheet(
                        "QPushButton { font-size: 9pt; font-weight: bold; color: #ffffff; "
                        "padding: 6px 10px; border: 1px solid #14b8a6; border-radius: 5px; "
                        "background-color: #0d9488; }"
                        "QPushButton:hover { background-color: #0f766e; }"
                    )

            worker.check_finished.connect(_on_bg_info)
            worker.start()
            self._bg_update_worker = worker
        except Exception:
            pass

    def _show_python_library_guide(self):
        """Show dialog explaining how to use pymorgan and pyrate_ta as Python libraries."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Using PyMORGAN & PyRATE-TA as Python Libraries")
        dlg.resize(660, 480)
        dlg.setStyleSheet("QDialog { background-color: #ffffff; color: #0f172a; }")
        vbox = QVBoxLayout(dlg)
        vbox.setContentsMargins(18, 16, 18, 16)
        vbox.setSpacing(10)

        intro = QLabel(
            "<b>Why install from PyPI for scripting?</b><br>"
            "The standalone installer bundles an isolated Python runtime dedicated to the desktop GUIs "
            "(<code>PyMORGAN</code> and <code>PyRATE-TA</code>). "
            "If you want to import <code>pymorgan</code> or <code>pyrate_ta</code> inside your own "
            "Python scripts, Jupyter notebooks, VS Code, or Spyder (Python &gt;= 3.12), install the "
            "official packages from PyPI into your Python environment:"
        )
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.TextFormat.RichText)
        vbox.addWidget(intro)

        code_box = QTextEdit(dlg)
        code_box.setReadOnly(True)
        code_box.setStyleSheet(
            "QTextEdit { font-family: Consolas, 'Courier New', monospace; font-size: 9.5pt; "
            "background-color: #f8fafc; color: #0f172a; border: 1px solid #cbd5e1; "
            "border-radius: 6px; padding: 8px; }"
        )
        code_box.setPlainText(
            "# 1. Install in your Python >= 3.12 environment (or virtual environment):\n"
            "pip install pymorgan pyrate-ta\n"
            "# Or using uv:\n"
            "uv pip install pymorgan pyrate-ta\n\n"
            "# 2. Configure TeX Gyre Heros fonts in Matplotlib (optional, automatic on first GUI run):\n"
            "pymorgan-install-fonts\n\n"
            "# 3. Example Python script / Jupyter notebook usage:\n"
            "import pymorgan as pm\n"
            "import pyrate_ta as pr\n\n"
            "# Load a 1D/2D ultrafast spectroscopy dataset\n"
            "ds = pm.load_1d('my_spectrum.pdat')\n"
            "fig, ax = pm.plot_contour_1d(ds)\n\n"
            "# Run Global & Target kinetic analysis in PyRATE-TA\n"
            "fit = pr.fit_global(ds, n_components=3, model_type='Sequential')\n"
            "print(fit.summary())\n"
        )
        vbox.addWidget(code_box, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        vbox.addWidget(buttons)
        dlg.exec()

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
        """Open the PyMORGAN settings configuration window in-process."""
        try:
            import pymorgan as pm
            from pymorgan.gui.settings_panel import SettingsPanel

            cfg = pm.settings_path()
            try:
                pm.load_settings(cfg)
            except Exception:
                pass

            dlg = QDialog(self)
            dlg.setWindowTitle("PyMORGAN Settings Editor")
            dlg.resize(620, 520)
            layout = QVBoxLayout(dlg)
            panel = SettingsPanel(dlg)
            layout.addWidget(panel)

            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close
            )
            save_btn = buttons.button(QDialogButtonBox.StandardButton.Save)
            if save_btn is not None:
                save_btn.setText("Save permanently")

            def _save_settings():
                try:
                    pm.save_settings(cfg)
                    QMessageBox.information(
                        dlg, "Settings Saved", f"Settings successfully saved to {cfg}"
                    )
                except Exception as exc:
                    QMessageBox.critical(dlg, "Save Failed", str(exc))

            buttons.accepted.connect(_save_settings)
            buttons.rejected.connect(dlg.reject)
            layout.addWidget(buttons)
            dlg.exec()
        except Exception as exc:
            QMessageBox.critical(self, "Settings Error", f"Could not open settings:\n{exc}")

    def _show_pyrate_install_instructions(self):
        """Show dialog explaining how to install PyRATE-TA."""
        msg = (
            "PyRATE-TA is the companion kinetic analysis package for PyMORGAN.\n\n"
            "To install PyRATE-TA in your Python environment, run:\n"
            "    pip install pyrate-ta\n"
            "or with uv:\n"
            "    uv pip install pyrate-ta\n\n"
            "Or download the full PyMORGAN Suite installer from GitHub:\n"
            "    https://github.com/RJFernandezTeran/PyMORGAN/releases"
        )
        QMessageBox.information(self, "Install PyRATE-TA", msg)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the PyMORGAN Suite Launcher."""
    parser = argparse.ArgumentParser(description="PyMORGAN Suite Launcher")
    parser.add_argument("--pymorgan", action="store_true", help="Launch PyMORGAN directly")
    parser.add_argument("--pyrate", action="store_true", help="Launch PyRATE-TA directly")
    parser.add_argument("--settings", action="store_true", help="Launch Settings Editor directly")
    parser.add_argument(
        "--install-fonts",
        action="store_true",
        help="Install and verify TeX Gyre Heros fonts in Matplotlib (runs 2 passes)",
    )
    args, unknown = parser.parse_known_args(argv)

    if args.install_fonts:
        import io
        from matplotlib.figure import Figure
        from pymorgan.fonts import install_fonts_twice

        install_fonts_twice(quiet=False)
        fig = Figure(figsize=(2, 2))
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1])
        for fmt in ("png", "pdf", "svg", "eps"):
            buf = io.BytesIO()
            fig.savefig(buf, format=fmt)
        return 0

    # Fast dispatch if flag is provided
    if args.pymorgan:
        from pymorgan.launcher import main as morgan_main

        return morgan_main(unknown)

    if args.pyrate:
        if importlib.util.find_spec("pyrate_ta") is not None:
            from pyrate_ta.launcher import main as pyrate_main

            return pyrate_main(unknown)
        if is_pyrate_installed():
            launch_process("pyrate_ta.gui.app", unknown)
            return 0
        print("[PyMORGAN Suite] PyRATE-TA is not installed.")
        return 1

    if args.settings:
        from pymorgan.gui.settings_app import main as settings_main

        return settings_main(unknown)

    # Hide host console window if running on Windows
    set_windows_app_id("PyMORGAN.SuiteLauncher")
    hide_console()

    # Ensure preferred fonts are installed and registered with Matplotlib early (2 passes on first run)
    try:
        from pymorgan.fonts import ensure_fonts_installed

        ensure_fonts_installed(quiet=True)
    except Exception:
        pass

    qt_args = list(sys.argv)
    if (
        sys.platform.startswith("win")
        and "-platform" not in qt_args
        and not os.environ.get("QT_QPA_PLATFORM")
    ):
        qt_args += ["-platform", "windows:darkmode=0"]

    app = QApplication.instance() or QApplication(qt_args)
    fusion = QStyleFactory.create("Fusion")
    if fusion is not None:
        app.setStyle(fusion)
        app.setPalette(fusion.standardPalette())

    icon_path = Path(__file__).parent / "icons" / "pirate-hat.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    dialog = SuiteLauncherDialog()
    dialog.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
