"""GitHub Releases updater for PyMORGAN and PyRATE-TA desktop applications.

Provides non-blocking background update checking, version comparison via
``packaging.version``, release notes display, and optional direct installer download.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version
from PyQt6.QtCore import QCoreApplication, QSettings, Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


@dataclass
class UpdateInfo:
    """Information about a release fetched from GitHub."""

    is_update_available: bool
    current_version: str
    latest_version: str
    release_name: str
    release_notes: str
    html_url: str
    published_at: str
    installer_asset: dict[str, Any] | None = None


def normalize_version(ver_str: str) -> str:
    """Strip leading 'v' or 'V' and whitespace from version string."""
    return ver_str.strip().lstrip("vV")


def is_newer_version(latest_str: str, current_str: str) -> bool:
    """Compare two version strings, returning True if latest > current."""
    l_clean = normalize_version(latest_str)
    c_clean = normalize_version(current_str)
    try:
        return Version(l_clean) > Version(c_clean)
    except InvalidVersion:
        # Fallback to string / tuple comparison if custom version schema is used
        try:
            l_parts = [int(p) for p in l_clean.split(".") if p.isdigit()]
            c_parts = [int(p) for p in c_clean.split(".") if p.isdigit()]
            return l_parts > c_parts
        except Exception:
            return l_clean != c_clean


def fetch_latest_release(
    repo_url: str,
    current_version: str,
    timeout: float = 6.0,
) -> UpdateInfo:
    """Query GitHub Releases API for the repository's latest release.

    Parameters
    ----------
    repo_url : str
        GitHub repository URL, e.g. "https://github.com/RJFernandezTeran/PyMORGAN".
    current_version : str
        Currently running version, e.g. "1.0.0".
    timeout : float, default 6.0
        HTTP request timeout in seconds.

    Returns
    -------
    UpdateInfo
    """
    clean_url = repo_url.rstrip("/")
    if "github.com/" in clean_url:
        repo_slug = clean_url.split("github.com/")[-1]
    else:
        repo_slug = clean_url

    api_url = f"https://api.github.com/repos/{repo_slug}/releases/latest"
    headers = {
        "User-Agent": "PyMORGAN-PyRATE-TA-Updater",
        "Accept": "application/vnd.github.v3+json",
    }
    req = urllib.request.Request(api_url, headers=headers)

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"GitHub API returned HTTP {resp.status}")
        data = json.loads(resp.read().decode("utf-8"))

    tag_name = data.get("tag_name", "")
    latest_ver = normalize_version(tag_name)
    release_name = data.get("name") or tag_name
    release_notes = data.get("body") or "No release notes provided."
    html_url = data.get("html_url") or repo_url
    published_at = data.get("published_at", "")[:10]

    # Find installer asset (.exe / Setup / .msi) if available
    assets = data.get("assets", [])
    installer_asset = None
    for asset in assets:
        name_lower = asset.get("name", "").lower()
        if name_lower.endswith(".exe") or "setup" in name_lower or name_lower.endswith(".msi"):
            installer_asset = asset
            break

    update_avail = is_newer_version(latest_ver, current_version)

    return UpdateInfo(
        is_update_available=update_avail,
        current_version=current_version,
        latest_version=latest_ver,
        release_name=release_name,
        release_notes=release_notes,
        html_url=html_url,
        published_at=published_at,
        installer_asset=installer_asset,
    )


class UpdateCheckWorker(QThread):
    """Background worker to check for updates without freezing the GUI."""

    check_finished = pyqtSignal(object)  # UpdateInfo
    check_failed = pyqtSignal(str)

    def __init__(self, repo_url: str, current_version: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.repo_url = repo_url
        self.current_version = current_version

    def run(self):
        try:
            info = fetch_latest_release(self.repo_url, self.current_version)
            self.check_finished.emit(info)
        except urllib.error.URLError as err:
            self.check_failed.emit(f"Network connection error: {err.reason}")
        except Exception as exc:
            self.check_failed.emit(str(exc))


class DownloadWorker(QThread):
    """Background worker to download an installer asset."""

    progress = pyqtSignal(int, int)  # bytes_downloaded, total_bytes
    finished = pyqtSignal(str)       # local_file_path
    failed = pyqtSignal(str)

    def __init__(self, download_url: str, file_name: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.download_url = download_url
        self.file_name = file_name
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            dest_path = Path(tempfile.gettempdir()) / self.file_name
            headers = {"User-Agent": "PyMORGAN-PyRATE-TA-Updater"}
            req = urllib.request.Request(self.download_url, headers=headers)

            with urllib.request.urlopen(req, timeout=15) as resp:
                total_size = int(resp.headers.get("content-length", 0))
                downloaded = 0
                chunk_size = 64 * 1024

                with open(dest_path, "wb") as f:
                    while not self._is_cancelled:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        self.progress.emit(downloaded, total_size)

            if self._is_cancelled:
                if dest_path.exists():
                    dest_path.unlink(missing_ok=True)
                return

            self.finished.emit(str(dest_path))
        except Exception as exc:
            self.failed.emit(str(exc))


class UpdateDialog(QDialog):
    """Dialog displaying new version details, changelog, and download triggers."""

    def __init__(
        self,
        info: UpdateInfo,
        app_name: str,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.info = info
        self.app_name = app_name
        self._download_worker: DownloadWorker | None = None

        self.setWindowTitle(f"Update Available - {self.app_name}")
        self.setFixedSize(540, 480)
        self.setModal(True)

        self.setStyleSheet("""
            QDialog {
                background-color: #0f172a;
                color: #e2e8f0;
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Arial, sans-serif;
            }
            QFrame#headerBox {
                background-color: #1e293b;
                border: 1px solid #3b82f6;
                border-radius: 8px;
                padding: 12px;
            }
            QLabel {
                color: #e2e8f0;
            }
            QLabel#lblHeadline {
                color: #60a5fa;
                font-size: 16px;
                font-weight: bold;
            }
            QLabel#lblNotes {
                color: #f1f5f9;
                font-size: 13px;
                font-weight: bold;
            }
            QLabel#lblVersionBadge {
                color: #94a3b8;
                font-size: 12.5px;
            }
            QTextBrowser {
                background-color: #1e293b;
                color: #f1f5f9;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 10px;
                font-size: 12.5px;
            }
            QPushButton {
                background-color: #1e293b;
                color: #93c5fd;
                border: 1px solid #3b82f6;
                border-radius: 5px;
                padding: 7px 16px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #2563eb;
                color: #ffffff;
            }
            QPushButton#btnPrimary {
                background-color: #2563eb;
                color: #ffffff;
                border: 1px solid #60a5fa;
            }
            QPushButton#btnPrimary:hover {
                background-color: #1d4ed8;
            }
            QProgressBar {
                border: 1px solid #334155;
                border-radius: 4px;
                background-color: #1e293b;
                text-align: center;
                color: #ffffff;
                height: 18px;
            }
            QProgressBar::chunk {
                background-color: #2563eb;
                border-radius: 3px;
            }
        """)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # Header Box
        header = QFrame()
        header.setObjectName("headerBox")
        h_layout = QVBoxLayout(header)
        h_layout.setSpacing(6)

        lbl_head = QLabel(f"🎉 A new version of {self.app_name} is available!")
        lbl_head.setObjectName("lblHeadline")
        h_layout.addWidget(lbl_head)

        badge_text = (
            f"<b>Current Version:</b> v{self.info.current_version} &nbsp; ➔ &nbsp; "
            f"<b>Latest Version:</b> <span style='color: #4ade80;'>v{self.info.latest_version}</span>"
        )
        if self.info.published_at:
            badge_text += f" &nbsp;({self.info.published_at})"
        lbl_badge = QLabel(badge_text)
        lbl_badge.setObjectName("lblVersionBadge")
        h_layout.addWidget(lbl_badge)

        layout.addWidget(header)

        # Release Notes Title & Box
        lbl_notes = QLabel(f"<b>What's New in {self.info.release_name}:</b>")
        lbl_notes.setObjectName("lblNotes")
        layout.addWidget(lbl_notes)

        self.text_notes = QTextBrowser()
        self.text_notes.setOpenExternalLinks(True)
        self.text_notes.document().setDefaultStyleSheet("""
            body, p, li, span, div { color: #f1f5f9; font-size: 12px; line-height: 1.45; }
            h1, h2, h3, h4 { color: #60a5fa; font-weight: bold; margin-top: 6px; margin-bottom: 4px; }
            code, pre { background-color: #0f172a; color: #38bdf8; font-family: monospace; border-radius: 3px; padding: 2px 4px; }
            a { color: #38bdf8; text-decoration: underline; }
            ul, ol { margin-left: 14px; margin-top: 2px; margin-bottom: 4px; }
        """)
        # Render markdown/plain text
        self.text_notes.setMarkdown(self.info.release_notes)
        layout.addWidget(self.text_notes)

        # Download progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Button row
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        if self.info.installer_asset:
            asset_name = self.info.installer_asset.get("name", "Setup.exe")
            self.btn_download = QPushButton(f"⬇️ Download & Run {asset_name}")
            self.btn_download.setObjectName("btnPrimary")
            self.btn_download.setCursor(Qt.CursorShape.PointingHandCursor)
            self.btn_download.clicked.connect(self._on_download_installer)
            btn_layout.addWidget(self.btn_download)

        self.btn_github = QPushButton("🌐 View on GitHub")
        self.btn_github.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_github.clicked.connect(self._on_open_github)
        btn_layout.addWidget(self.btn_github)

        btn_layout.addStretch()

        self.btn_close = QPushButton("Later")
        self.btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_close.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_close)

        layout.addLayout(btn_layout)

    def _on_open_github(self):
        QDesktopServices.openUrl(QUrl(self.info.html_url))

    def _on_download_installer(self):
        if not self.info.installer_asset:
            return

        download_url = self.info.installer_asset.get("browser_download_url")
        file_name = self.info.installer_asset.get("name", "Setup.exe")
        if not download_url:
            self._on_open_github()
            return

        self.btn_download.setEnabled(False)
        self.btn_download.setText("⏳ Downloading...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self._download_worker = DownloadWorker(download_url, file_name, self)
        self._download_worker.progress.connect(self._on_download_progress)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.failed.connect(self._on_download_failed)
        self._download_worker.start()

    def _on_download_progress(self, downloaded: int, total: int):
        if total > 0:
            pct = int((downloaded / total) * 100)
            self.progress_bar.setValue(pct)
            mb_down = downloaded / (1024 * 1024)
            mb_tot = total / (1024 * 1024)
            self.progress_bar.setFormat(f"%p% ({mb_down:.1f} / {mb_tot:.1f} MB)")
        else:
            self.progress_bar.setFormat(f"{downloaded / (1024 * 1024):.1f} MB downloaded")

    def _on_download_finished(self, local_path: str):
        self.progress_bar.setValue(100)
        self.btn_download.setText("✅ Download Complete")

        mb = QMessageBox(self)
        mb.setIcon(QMessageBox.Icon.Question)
        mb.setWindowTitle("Update Ready to Install")
        mb.setText(
            f"The update has been downloaded to:\n{local_path}\n\n"
            f"Would you like to close {self.app_name} and run the installer now?"
        )
        mb.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        mb.setDefaultButton(QMessageBox.StandardButton.Yes)
        _style_message_box(mb)
        reply = mb.exec()

        if reply == QMessageBox.StandardButton.Yes:
            # Launch installer
            if os.name == "nt":
                os.startfile(local_path)  # noqa: S606
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(local_path))
            QCoreApplication.quit()
        else:
            self.accept()

    def _on_download_failed(self, error_msg: str):
        self.btn_download.setEnabled(True)
        self.btn_download.setText("⬇️ Retry Download")
        self.progress_bar.setVisible(False)
        mb = QMessageBox(self)
        mb.setIcon(QMessageBox.Icon.Warning)
        mb.setWindowTitle("Download Failed")
        mb.setText(
            f"Could not download the update:\n{error_msg}\n\n"
            "You can download it manually from the GitHub release page."
        )
        _style_message_box(mb)
        mb.exec()


def _style_message_box(mb: QMessageBox) -> QMessageBox:
    """Ensure message boxes displayed over dark dialogs have crisp high-contrast text."""
    mb.setStyleSheet("""
        QMessageBox {
            background-color: #0f172a;
        }
        QLabel {
            color: #f8fafc;
            font-size: 12.5px;
            min-width: 290px;
        }
        QPushButton {
            background-color: #1e293b;
            color: #93c5fd;
            border: 1px solid #3b82f6;
            border-radius: 5px;
            padding: 6px 18px;
            font-weight: bold;
            font-size: 12px;
            min-width: 65px;
        }
        QPushButton:hover {
            background-color: #2563eb;
            color: #ffffff;
        }
    """)
    return mb


def check_for_updates_interactive(
    parent: QWidget,
    github_url: str,
    current_version: str,
    app_name: str,
    status_callback: Any = None,
) -> None:
    """Check for updates in the background and present the result cleanly.

    Parameters
    ----------
    parent : QWidget
        Parent widget for modal dialogs.
    github_url : str
        GitHub project URL.
    current_version : str
        Current app version string.
    app_name : str
        Application name for display.
    status_callback : callable, optional
        Callback accepting a string message (e.g. status bar).
    """
    if status_callback:
        status_callback(f"Checking for updates for {app_name}...")

    worker = UpdateCheckWorker(github_url, current_version, parent)

    def on_success(info: UpdateInfo):
        if status_callback:
            status_callback("")
        if info.is_update_available:
            dlg = UpdateDialog(info, app_name, parent)
            dlg.exec()
        else:
            mb = QMessageBox(parent)
            mb.setIcon(QMessageBox.Icon.Information)
            mb.setWindowTitle("Up to Date")
            mb.setText(
                f"You are using the latest version of {app_name} (v{current_version}).\n\n"
                "No updates are currently available."
            )
            _style_message_box(mb)
            mb.exec()

    def on_error(err_msg: str):
        if status_callback:
            status_callback("")
        mb = QMessageBox(parent)
        mb.setIcon(QMessageBox.Icon.Warning)
        mb.setWindowTitle("Update Check Failed")
        mb.setText(
            f"Unable to check for updates for {app_name}:\n{err_msg}\n\n"
            "Please check your internet connection or visit the GitHub repository."
        )
        _style_message_box(mb)
        mb.exec()

    worker.check_finished.connect(on_success)
    worker.check_failed.connect(on_error)
    worker.start()
    # Keep reference on parent to avoid garbage collection before thread finishes
    parent._active_update_worker = worker


def check_for_updates_background(
    window: QMainWindow,
    github_url: str,
    current_version: str,
    app_name: str,
    force: bool = False,
) -> None:
    """Run a quiet, passive update check in the background.

    Checks at most once every 24 hours (unless force=True). If a newer version
    is found, adds a non-intrusive notification pill to the main window's status bar.
    """
    settings = QSettings("RJFernandezTeran", app_name)
    now = time.time()
    last_check = float(settings.value("last_update_check", 0) or 0)
    if not force and (now - last_check < 86400):
        return

    settings.setValue("last_update_check", now)

    worker = UpdateCheckWorker(github_url, current_version, window)

    def on_success(info: UpdateInfo):
        if not info.is_update_available or not hasattr(window, "statusBar"):
            return

        sb = window.statusBar()
        if sb is None:
            return

        # Check if an update button is already present
        existing_btn = sb.findChild(QPushButton, "btnStatusUpdateNotification")
        if existing_btn is not None:
            return

        btn = QPushButton(f"🎉 {app_name} v{info.latest_version} Available")
        btn.setObjectName("btnStatusUpdateNotification")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip("A new release is available! Click to view release notes and download.")
        btn.setStyleSheet("""
            QPushButton#btnStatusUpdateNotification {
                background-color: #2563eb;
                color: #ffffff;
                font-weight: bold;
                font-size: 11px;
                border-radius: 4px;
                padding: 3px 10px;
                border: 1px solid #60a5fa;
            }
            QPushButton#btnStatusUpdateNotification:hover {
                background-color: #1d4ed8;
            }
        """)

        def on_click():
            dlg = UpdateDialog(info, app_name, window)
            dlg.exec()

        btn.clicked.connect(on_click)
        sb.addPermanentWidget(btn)

    worker.check_finished.connect(on_success)
    # Background check ignores network errors silently
    worker.start()
    window._bg_update_worker = worker
