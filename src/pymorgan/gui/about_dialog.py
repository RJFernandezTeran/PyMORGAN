"""Reusable Modern Theme-Aware About Dialog for PyQt6 / PySide6 applications."""

from __future__ import annotations

import os

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def _is_dark() -> bool:
    try:
        from pymorgan.gui.theme import is_dark_palette

        return bool(is_dark_palette())
    except Exception:
        return False


class ModernAboutDialog(QDialog):
    """Theme-aware About dialog displaying application branding, version, and author metadata."""

    def __init__(
        self,
        parent: QWidget | None = None,
        title: str = "About Application",
        app_name: str = "Application Name",
        version: str = "1.0.0",
        subtitle: str = "Modern Desktop Application Framework",
        description: str | None = None,
        author: str = "Dr. Ricardo J. Fernández-Terán",
        department: str = "Department of Physical Chemistry",
        institution: str = "University of Geneva, Switzerland",
        contact_email: str = "Ricardo.FernandezTeran@unige.ch",
        website_url: str | None = None,
        github_url: str | None = None,
        license_name: str = "BSD 3-Clause License",
        banner_path: str | None = None,
        icon_path: str | None = None,
        manual_pdf_path: str | None = None,
        docs_html_path: str | None = None,
        docs_url: str | None = None,
        ai_credit: str | None = "Developed with AI assistance from <b>Google Antigravity</b>.",
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ModernAboutDialog")
        self._dark = _is_dark()

        self._app_name = app_name
        self._version = version
        self._subtitle = subtitle
        self._description = description or "High-performance modular analytical application."
        self._author = author
        self._department = department
        self._institution = institution
        self._contact_email = contact_email
        self._website_url = website_url
        self._github_url = github_url
        self._license_name = license_name
        self._banner_path = banner_path
        self._icon_path = icon_path
        self._manual_pdf_path = manual_pdf_path
        self._docs_html_path = docs_html_path
        self._docs_url = docs_url
        self._ai_credit = ai_credit

        self.setWindowTitle(title)
        self.setFixedSize(580, 540)
        self.setModal(True)

        if self._icon_path and os.path.exists(self._icon_path):
            self.setWindowIcon(QIcon(self._icon_path))

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        if self._dark:
            self.setStyleSheet("""
                QDialog#ModernAboutDialog {
                    background-color: #181c24;
                    color: #f1f5f9;
                    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Arial, sans-serif;
                }
                QDialog#ModernAboutDialog QFrame#headerFrame {
                    background-color: #222834;
                    border: 1px solid #3b82f6;
                    border-radius: 10px;
                    padding: 12px;
                }
                QDialog#ModernAboutDialog QLabel {
                    color: #f1f5f9;
                    background: transparent;
                    border: none;
                }
                QDialog#ModernAboutDialog QLabel#lblTitle {
                    color: #60a5fa;
                    font-size: 20px;
                    font-weight: bold;
                }
                QDialog#ModernAboutDialog QLabel#lblSubtitle {
                    color: #94a3b8;
                    font-size: 12.5px;
                    font-style: italic;
                }
                QDialog#ModernAboutDialog QLabel#lblDetails {
                    color: #e2e8f0;
                    font-size: 12px;
                    line-height: 1.4;
                }
                QDialog#ModernAboutDialog QPushButton {
                    background-color: #222834;
                    color: #e2e8f0;
                    border: 1px solid #475569;
                    border-radius: 5px;
                    padding: 7px 16px;
                    font-weight: bold;
                    font-size: 12px;
                }
                QDialog#ModernAboutDialog QPushButton:hover {
                    background-color: #2563eb;
                    color: #ffffff;
                }
                QDialog#ModernAboutDialog QPushButton#btnManual {
                    background-color: #2563eb;
                    color: #ffffff;
                    border: 1px solid #60a5fa;
                }
                QDialog#ModernAboutDialog QPushButton#btnManual:hover {
                    background-color: #1d4ed8;
                }
            """)
        else:
            self.setStyleSheet("""
                QDialog#ModernAboutDialog {
                    background-color: #f8fafc;
                    color: #0f172a;
                    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Arial, sans-serif;
                }
                QDialog#ModernAboutDialog QFrame#headerFrame {
                    background-color: #ffffff;
                    border: 1px solid #93c5fd;
                    border-left: 4px solid #2563eb;
                    border-radius: 10px;
                    padding: 12px;
                }
                QDialog#ModernAboutDialog QLabel {
                    color: #0f172a;
                    background: transparent;
                    border: none;
                }
                QDialog#ModernAboutDialog QLabel#lblTitle {
                    color: #1d4ed8;
                    font-size: 20px;
                    font-weight: bold;
                }
                QDialog#ModernAboutDialog QLabel#lblSubtitle {
                    color: #475569;
                    font-size: 12.5px;
                    font-style: italic;
                }
                QDialog#ModernAboutDialog QLabel#lblDetails {
                    color: #1e293b;
                    font-size: 12px;
                    line-height: 1.4;
                }
                QDialog#ModernAboutDialog QPushButton {
                    background-color: #ffffff;
                    color: #1e293b;
                    border: 1px solid #cbd5e1;
                    border-radius: 5px;
                    padding: 7px 16px;
                    font-weight: bold;
                    font-size: 12px;
                }
                QDialog#ModernAboutDialog QPushButton:hover {
                    background-color: #e2e8f0;
                    color: #0f172a;
                }
                QDialog#ModernAboutDialog QPushButton#btnManual {
                    background-color: #2563eb;
                    color: #ffffff;
                    border: 1px solid #1d4ed8;
                }
                QDialog#ModernAboutDialog QPushButton#btnManual:hover {
                    background-color: #1d4ed8;
                }
            """)

        # ------------------------------------------------------------------ #
        # 1. Header Frame (Banner + Title/Subtitle)                           #
        # ------------------------------------------------------------------ #
        header_frame = QFrame()
        header_frame.setObjectName("headerFrame")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(10, 10, 10, 10)
        header_layout.setSpacing(16)

        if self._banner_path and os.path.exists(self._banner_path):
            lbl_banner = QLabel()
            lbl_banner.setObjectName("lblBanner")
            pixmap = QPixmap(self._banner_path)
            scaled_pixmap = pixmap.scaled(
                105,
                105,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            lbl_banner.setPixmap(scaled_pixmap)
            header_layout.addWidget(lbl_banner)

        header_text_layout = QVBoxLayout()
        header_text_layout.setSpacing(4)

        lbl_title = QLabel(f"{self._app_name} v{self._version}")
        lbl_title.setObjectName("lblTitle")

        lbl_subtitle = QLabel(self._subtitle)
        lbl_subtitle.setObjectName("lblSubtitle")
        lbl_subtitle.setWordWrap(True)

        header_text_layout.addWidget(lbl_title)
        header_text_layout.addWidget(lbl_subtitle)
        header_text_layout.addStretch()

        header_layout.addLayout(header_text_layout)
        layout.addWidget(header_frame)

        # ------------------------------------------------------------------ #
        # 2. Metadata Body & Description                                      #
        # ------------------------------------------------------------------ #
        link_color = "#60a5fa" if self._dark else "#2563eb"
        hr_color = "#334155" if self._dark else "#cbd5e1"
        desc_color = "#cbd5e1" if self._dark else "#334155"
        credit_color = "#94a3b8" if self._dark else "#64748b"
        badge_left_bg = "#222834" if self._dark else "#f1f5f9"
        badge_left_fg = "#cbd5e1" if self._dark else "#334155"

        github_line = ""
        github_badge = ""
        if self._github_url:
            repo_slug = (
                self._github_url.rstrip("/").split("github.com/")[-1]
                if "github.com/" in self._github_url
                else self._github_url
            )
            github_line = f'<b>GitHub:</b> <a href="{self._github_url}" style="color: {link_color}; text-decoration: none;">{self._github_url}</a><br>'
            github_badge = f"""
        <table cellpadding="2" cellspacing="0" style="margin-top: 4px; margin-bottom: 2px;">
            <tr>
                <td bgcolor="{badge_left_bg}" style="border: 1px solid {hr_color}; border-right: none; padding: 2px 6px;">
                    <font color="{badge_left_fg}" size="2"><b>&nbsp;GitHub&nbsp;</b></font>
                </td>
                <td bgcolor="#2563eb" style="border: 1px solid #3b82f6; padding: 2px 6px;">
                    <a href="{self._github_url}" style="color: #ffffff; text-decoration: none;"><font color="#ffffff" size="2"><b>&nbsp;{repo_slug}&nbsp;</b></font></a>
                </td>
            </tr>
        </table>
        """

        website_line = (
            f'<b>Website:</b> <a href="{self._website_url}" style="color: {link_color}; text-decoration: none;">{self._website_url}</a><br>'
            if self._website_url
            else ""
        )

        body_html = f"""
        <p style="margin-top:0px; margin-bottom: 8px;">
            <b>Author:</b> {self._author}<br>
            <b>Department:</b> {self._department}<br>
            <b>Institution:</b> {self._institution}<br>
            <b>Contact:</b> <a href="mailto:{self._contact_email}" style="color: {link_color}; text-decoration: none;">{self._contact_email}</a><br>
            {website_line}{github_line}<b>License:</b> {self._license_name}
        </p>
        {github_badge}
        <hr style="border: 0; border-top: 1px solid {hr_color}; margin: 8px 0;">

        <p style="color: {desc_color}; font-size: 11.5px; margin-top: 6px; margin-bottom: 6px;">
            {self._description}
        </p>
        """

        if self._ai_credit:
            body_html += f"""
            <p style="color: {credit_color}; font-size: 10.5px; margin-top: 6px; margin-bottom: 0px;">
                <em>{self._ai_credit}</em>
            </p>
            """

        lbl_details = QLabel(body_html)
        lbl_details.setObjectName("lblDetails")
        lbl_details.setOpenExternalLinks(True)
        lbl_details.setWordWrap(True)
        layout.addWidget(lbl_details)

        layout.addStretch()

        # ------------------------------------------------------------------ #
        # 3. Action Buttons                                                   #
        # ------------------------------------------------------------------ #
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        if self._manual_pdf_path:
            btn_manual = QPushButton("Open User Manual (PDF)")
            btn_manual.setObjectName("btnManual")
            btn_manual.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_manual.clicked.connect(self._on_open_manual)
            btn_layout.addWidget(btn_manual)

        if self._docs_html_path or self._docs_url:
            btn_docs = QPushButton("Web Docs")
            btn_docs.setObjectName("btnDocs")
            btn_docs.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_docs.setToolTip("Open interactive Sphinx HTML documentation")
            btn_docs.clicked.connect(self._on_open_docs)
            btn_layout.addWidget(btn_docs)

        if self._github_url:
            btn_github = QPushButton("GitHub")
            btn_github.setObjectName("btnGithub")
            btn_github.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_github.clicked.connect(self._on_open_github)
            btn_layout.addWidget(btn_github)

            btn_updates = QPushButton("Check for Updates")
            btn_updates.setObjectName("btnUpdates")
            btn_updates.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_updates.clicked.connect(self._on_check_updates)
            btn_layout.addWidget(btn_updates)

            btn_py_env = QPushButton("Setup in Python...")
            btn_py_env.setObjectName("btnPyEnv")
            btn_py_env.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_py_env.setToolTip(
                "Configure Conda, Homebrew, or system Python so you can 'import pymorgan' and 'import pyrate_ta'"
            )
            btn_py_env.clicked.connect(self._on_setup_python_env)
            btn_layout.addWidget(btn_py_env)

        btn_layout.addStretch()

        btn_close = QPushButton("Close")
        btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)

    def _on_check_updates(self) -> None:
        """Trigger an online update check for the latest release on GitHub."""
        if not self._github_url:
            return
        from pymorgan.gui.updater import check_for_updates_interactive

        btn = self.findChild(QPushButton, "btnUpdates")
        if btn is not None:
            btn.setEnabled(False)
            btn.setText("Checking...")

        def restore_btn(*_):
            if btn is not None:
                btn.setEnabled(True)
                btn.setText("Check for Updates")

        try:
            check_for_updates_interactive(
                self,
                github_url=self._github_url,
                current_version=self._version,
                app_name=self._app_name,
            )
            worker = getattr(self, "_active_update_worker", None)
            if worker is not None:
                worker.check_finished.connect(restore_btn)
                worker.check_failed.connect(restore_btn)
            else:
                restore_btn()
        except Exception as exc:
            restore_btn()
            QMessageBox.warning(self, "Update Check Error", str(exc))

    def _on_setup_python_env(self) -> None:
        """Open the Local Python Environment Setup wizard."""
        try:
            from pymorgan.gui.python_env_setup import open_local_python_setup_dialog

            open_local_python_setup_dialog(self)
        except Exception as exc:
            QMessageBox.warning(self, "Python Setup Error", str(exc))

    def _on_open_github(self) -> None:
        """Open the project GitHub repository in the default web browser."""
        if self._github_url:
            QDesktopServices.openUrl(QUrl(self._github_url))

    def _on_open_docs(self) -> None:
        """Open the local Sphinx HTML documentation or fallback online URL."""
        if self._docs_html_path and os.path.isfile(self._docs_html_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(self._docs_html_path)))
        elif self._docs_url:
            QDesktopServices.openUrl(QUrl(self._docs_url))

    def _on_open_manual(self) -> None:
        """Locate and open the user manual PDF using the default desktop PDF reader."""
        if not self._manual_pdf_path:
            return

        candidates = [
            self._manual_pdf_path,
            os.path.abspath(self._manual_pdf_path),
            os.path.join(os.path.dirname(__file__), self._manual_pdf_path),
            os.path.join(os.getcwd(), self._manual_pdf_path),
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", self._manual_pdf_path)
            ),
        ]

        found_path: str | None = None
        for p in candidates:
            if os.path.exists(p) and os.path.isfile(p):
                found_path = os.path.abspath(p)
                break

        if found_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(found_path))
        else:
            QMessageBox.warning(
                self,
                "Manual Not Found",
                f"Could not locate documentation at:\n'{self._manual_pdf_path}'.\n\nPlease verify that the PDF is compiled/present.",
            )
