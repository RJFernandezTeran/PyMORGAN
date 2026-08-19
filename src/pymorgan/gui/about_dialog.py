"""Reusable Modern Dark-Themed About Dialog for PyQt6 / PySide6 applications."""

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


class ModernAboutDialog(QDialog):
    """
    Modern dark-themed About dialog displaying application branding,
    version, author metadata, description, and documentation triggers.
    """

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
        github_url: str | None = None,
        license_name: str = "BSD 3-Clause License",
        banner_path: str | None = None,
        icon_path: str | None = None,
        manual_pdf_path: str | None = None,
        ai_credit: str | None = "Developed with AI assistance from <b>Google Antigravity</b>.",
    ) -> None:
        super().__init__(parent)

        self._app_name = app_name
        self._version = version
        self._subtitle = subtitle
        self._description = description or "High-performance modular analytical application."
        self._author = author
        self._department = department
        self._institution = institution
        self._contact_email = contact_email
        self._github_url = github_url
        self._license_name = license_name
        self._banner_path = banner_path
        self._icon_path = icon_path
        self._manual_pdf_path = manual_pdf_path
        self._ai_credit = ai_credit

        self.setWindowTitle(title)
        self.setFixedSize(580, 520)
        self.setModal(True)

        if self._icon_path and os.path.exists(self._icon_path):
            self.setWindowIcon(QIcon(self._icon_path))

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        # Apply cohesive dark palette stylesheet
        self.setStyleSheet("""
            QDialog {
                background-color: #0f172a;
                color: #e2e8f0;
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Arial, sans-serif;
            }
            QFrame#headerFrame {
                background-color: #1e293b;
                border: 1px solid #3b82f6;
                border-radius: 10px;
                padding: 12px;
            }
            QLabel#lblBanner {
                border-radius: 8px;
            }
            QLabel#lblTitle {
                color: #60a5fa;
                font-size: 20px;
                font-weight: bold;
            }
            QLabel#lblSubtitle {
                color: #94a3b8;
                font-size: 12.5px;
                font-style: italic;
            }
            QLabel#lblDetails {
                color: #cbd5e1;
                font-size: 12px;
                line-height: 1.4;
            }
            QPushButton {
                background-color: #1e293b;
                color: #93c5fd;
                border: 1px solid #3b82f6;
                border-radius: 5px;
                padding: 7px 18px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #2563eb;
                color: #ffffff;
            }
            QPushButton#btnManual {
                background-color: #2563eb;
                color: #ffffff;
                border: 1px solid #60a5fa;
            }
            QPushButton#btnManual:hover {
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

        # Optional Banner Graphic
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

        # Title + Version + Subtitle
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
        github_line = ""
        github_badge = ""
        if self._github_url:
            repo_slug = (
                self._github_url.rstrip("/").split("github.com/")[-1]
                if "github.com/" in self._github_url
                else self._github_url
            )
            github_line = f'<b>GitHub:</b> <a href="{self._github_url}" style="color: #60a5fa; text-decoration: none;">{self._github_url}</a><br>'
            github_badge = f"""
        <table cellpadding="2" cellspacing="0" style="margin-top: 4px; margin-bottom: 2px;">
            <tr>
                <td bgcolor="#1e293b" style="border: 1px solid #334155; border-right: none; padding: 2px 6px;">
                    <font color="#94a3b8" size="2"><b>&nbsp;GitHub&nbsp;</b></font>
                </td>
                <td bgcolor="#2563eb" style="border: 1px solid #3b82f6; padding: 2px 6px;">
                    <a href="{self._github_url}" style="color: #ffffff; text-decoration: none;"><font color="#ffffff" size="2"><b>&nbsp;{repo_slug}&nbsp;</b></font></a>
                </td>
            </tr>
        </table>
        """

        body_html = f"""
        <p style="margin-top:0px; margin-bottom: 8px;">
            <b>Author:</b> {self._author}<br>
            <b>Department:</b> {self._department}<br>
            <b>Institution:</b> {self._institution}<br>
            <b>Contact:</b> <a href="mailto:{self._contact_email}" style="color: #60a5fa; text-decoration: none;">{self._contact_email}</a><br>
            {github_line}<b>License:</b> {self._license_name}
        </p>
        {github_badge}
        <hr style="border: 0; border-top: 1px solid #334155; margin: 8px 0;">

        <p style="color: #94a3b8; font-size: 11px; margin-top: 6px; margin-bottom: 6px;">
            {self._description}
        </p>
        """

        if self._ai_credit:
            body_html += f"""
            <p style="color: #64748b; font-size: 10.5px; margin-top: 6px; margin-bottom: 0px;">
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
            btn_manual = QPushButton("📖 Open User Manual (PDF)")
            btn_manual.setObjectName("btnManual")
            btn_manual.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_manual.clicked.connect(self._on_open_manual)
            btn_layout.addWidget(btn_manual)

        if self._github_url:
            btn_github = QPushButton("🌐 GitHub")
            btn_github.setObjectName("btnGithub")
            btn_github.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_github.clicked.connect(self._on_open_github)
            btn_layout.addWidget(btn_github)

        btn_layout.addStretch()

        btn_close = QPushButton("Close")
        btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)

    def _on_open_github(self) -> None:
        """Open the project GitHub repository in the default web browser."""
        if self._github_url:
            QDesktopServices.openUrl(QUrl(self._github_url))

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
