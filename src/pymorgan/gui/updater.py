"""GitHub Releases and Bleeding-Edge (main branch) updater for PyMORGAN and PyRATE-TA.

Provides non-blocking background update checking against both official GitHub
Releases and the repository ``main`` branch``__about__.py`` ("Bleeding Edge"),
version comparison via ``packaging.version``, one-click bleeding-edge source
code syncing, and direct platform installer download.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
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

BLEEDING_EDGE_DIR = Path.home() / ".config" / "pymorgan" / "bleeding_edge_src"


@dataclass
class UpdateInfo:
    """Information about official release and bleeding-edge main branch on GitHub."""

    is_update_available: bool
    current_version: str
    latest_version: str
    release_name: str
    release_notes: str
    html_url: str
    published_at: str
    installer_asset: dict[str, Any] | None = None
    release_version: str = ""
    bleeding_edge_version: str = ""
    bleeding_edge_commit: str = ""
    bleeding_edge_date: str = ""
    bleeding_edge_message: str = ""
    has_release_update: bool = False
    has_bleeding_edge_update: bool = False


def normalize_version(ver_str: str) -> str:
    """Strip leading 'v' or 'V' and whitespace from version string."""
    return ver_str.strip().lstrip("vV")


def is_newer_version(latest_str: str, current_str: str) -> bool:
    """Compare two version strings, returning True if latest > current."""
    if not latest_str or not current_str:
        return False
    l_clean = normalize_version(latest_str)
    c_clean = normalize_version(current_str)
    try:
        return Version(l_clean) > Version(c_clean)
    except InvalidVersion:
        try:
            l_parts = [int(p) for p in l_clean.split(".") if p.isdigit()]
            c_parts = [int(p) for p in c_clean.split(".") if p.isdigit()]
            return l_parts > c_parts
        except Exception:
            return l_clean != c_clean


def _select_platform_asset(assets: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the best release installer asset for the current operating system."""
    if not assets:
        return None
    if sys.platform == "win32":
        for a in assets:
            n = a.get("name", "").lower()
            if n.endswith(".exe") or "setup" in n or n.endswith(".msi"):
                return a
        for a in assets:
            n = a.get("name", "").lower()
            if "windows" in n and n.endswith(".zip"):
                return a
    elif sys.platform == "darwin":
        for a in assets:
            n = a.get("name", "").lower()
            if n.endswith(".dmg"):
                return a
        for a in assets:
            n = a.get("name", "").lower()
            if "macos" in n and n.endswith(".zip"):
                return a
    else:
        for a in assets:
            n = a.get("name", "").lower()
            if "linux" in n and (n.endswith(".tar.gz") or n.endswith(".AppImage")):
                return a
    return None


def _fetch_bleeding_edge_metadata(repo_slug: str, timeout: float = 5.0) -> tuple[str, str, str, str]:
    """Fetch __version__ and latest commit info from the repository's main branch."""
    pkg_dir = "pyrate_ta" if "pyrate" in repo_slug.lower() else "pymorgan"
    raw_url = f"https://raw.githubusercontent.com/{repo_slug}/main/src/{pkg_dir}/__about__.py"
    headers = {"User-Agent": "PyMORGAN-PyRATE-TA-Updater"}

    be_version = ""
    try:
        req = urllib.request.Request(raw_url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
        if m:
            be_version = normalize_version(m.group(1))
    except Exception:
        pass

    commit_sha = ""
    commit_date = ""
    commit_msg = ""
    try:
        commit_url = f"https://api.github.com/repos/{repo_slug}/commits/main"
        req_c = urllib.request.Request(
            commit_url,
            headers={
                "User-Agent": "PyMORGAN-PyRATE-TA-Updater",
                "Accept": "application/vnd.github.v3+json",
            },
        )
        with urllib.request.urlopen(req_c, timeout=timeout) as resp_c:
            c_data = json.loads(resp_c.read().decode("utf-8", errors="replace"))
        commit_sha = str(c_data.get("sha", ""))[:7]
        commit_obj = c_data.get("commit", {}) or {}
        author_obj = commit_obj.get("author", {}) or {}
        commit_date = str(author_obj.get("date", ""))[:10]
        raw_msg = str(commit_obj.get("message", "")).strip()
        commit_msg = raw_msg.splitlines()[0] if raw_msg else ""
    except Exception:
        pass

    return be_version, commit_sha, commit_date, commit_msg


def fetch_latest_release(
    repo_url: str,
    current_version: str,
    timeout: float = 6.0,
) -> UpdateInfo:
    """Query both GitHub Releases and the main branch (__about__.py) for updates."""
    clean_url = repo_url.rstrip("/")
    if "github.com/" in clean_url:
        repo_slug = clean_url.split("github.com/")[-1]
    else:
        repo_slug = clean_url

    headers = {
        "User-Agent": "PyMORGAN-PyRATE-TA-Updater",
        "Accept": "application/vnd.github.v3+json",
    }

    release_ver = ""
    release_name = ""
    release_notes = ""
    html_url = f"https://github.com/{repo_slug}"
    published_at = ""
    installer_asset = None

    # 1. Query latest formal GitHub Release (tolerating 404 if no release is published yet)
    api_url = f"https://api.github.com/repos/{repo_slug}/releases/latest"
    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                tag_name = data.get("tag_name", "")
                release_ver = normalize_version(tag_name)
                release_name = data.get("name") or tag_name
                release_notes = data.get("body") or ""
                html_url = data.get("html_url") or html_url
                published_at = data.get("published_at", "")[:10]
                installer_asset = _select_platform_asset(data.get("assets", []))
    except urllib.error.HTTPError as http_err:
        if http_err.code != 404:
            raise
    except Exception:
        pass

    # 2. Query main branch ("Bleeding Edge") __about__.py and latest commit
    be_ver, be_sha, be_date, be_msg = _fetch_bleeding_edge_metadata(repo_slug, timeout=timeout)

    if not release_ver and not be_ver:
        raise RuntimeError(f"Could not retrieve release or main branch version from {repo_slug}.")

    has_rel_update = is_newer_version(release_ver, current_version) if release_ver else False
    has_be_update = is_newer_version(be_ver, current_version) if be_ver else False
    update_avail = has_rel_update or has_be_update

    # Determine highest available version string
    if be_ver and release_ver:
        latest_ver = be_ver if is_newer_version(be_ver, release_ver) else release_ver
    else:
        latest_ver = be_ver or release_ver or current_version

    if not release_name:
        release_name = f"v{latest_ver} (main)"

    notes_parts: list[str] = []
    if be_ver:
        commit_info = f" (`{be_sha}` on {be_date})" if be_sha else ""
        msg_info = f"\n- **Latest commit on `main`:** {be_msg}" if be_msg else ""
        notes_parts.append(
            f"### Bleeding-Edge (`main` branch): `v{be_ver}`{commit_info}{msg_info}\n"
            "You can sync the pure-Python code directly from the `main` branch using "
            "**Sync Bleeding-Edge (`main`)** below without waiting for a full installer build."
        )
    if release_notes:
        notes_parts.append(f"### Official Release Notes ({release_name})\n{release_notes}")
    elif not notes_parts:
        notes_parts.append("No release notes provided.")

    return UpdateInfo(
        is_update_available=update_avail,
        current_version=normalize_version(current_version),
        latest_version=latest_ver,
        release_name=release_name,
        release_notes="\n\n---\n\n".join(notes_parts),
        html_url=html_url,
        published_at=published_at or be_date,
        installer_asset=installer_asset,
        release_version=release_ver,
        bleeding_edge_version=be_ver,
        bleeding_edge_commit=be_sha,
        bleeding_edge_date=be_date,
        bleeding_edge_message=be_msg,
        has_release_update=has_rel_update,
        has_bleeding_edge_update=has_be_update,
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


class BleedingEdgeSyncWorker(QThread):
    """Background worker that downloads and syncs PyMORGAN and PyRATE-TA from GitHub main."""

    progress = pyqtSignal(int, str)  # percentage, status_message
    finished = pyqtSignal(str, str)  # pymorgan_ver, pyrate_ver
    failed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

    def _download_and_extract_pkg(
        self, repo_slug: str, pkg_name: str, target_root: Path, pct_start: int, pct_end: int
    ) -> str:
        """Download main.zip from GitHub and extract src/<pkg_name> into target_root/<pkg_name>."""
        zip_url = f"https://github.com/{repo_slug}/archive/refs/heads/main.zip"
        self.progress.emit(pct_start, f"Downloading {repo_slug} (main)...")
        req = urllib.request.Request(
            zip_url, headers={"User-Agent": "PyMORGAN-PyRATE-TA-Updater"}
        )
        tmp_zip = Path(tempfile.gettempdir()) / f"{pkg_name}_main_sync.zip"
        with urllib.request.urlopen(req, timeout=20) as resp, open(tmp_zip, "wb") as out_f:
            shutil.copyfileobj(resp, out_f)

        self.progress.emit((pct_start + pct_end) // 2, f"Extracting {pkg_name}...")
        tmp_extract = Path(tempfile.gettempdir()) / f"{pkg_name}_main_extract"
        if tmp_extract.exists():
            shutil.rmtree(tmp_extract, ignore_errors=True)

        with zipfile.ZipFile(tmp_zip, "r") as zf:
            zf.extractall(tmp_extract)
        tmp_zip.unlink(missing_ok=True)

        # Locate src/<pkg_name> inside extracted archive
        found_src: Path | None = None
        for cand in tmp_extract.glob(f"*/src/{pkg_name}"):
            if cand.is_dir():
                found_src = cand
                break
        if found_src is None:
            raise RuntimeError(f"Could not find src/{pkg_name} in {repo_slug} archive.")

        dest_pkg = target_root / pkg_name
        if dest_pkg.exists():
            shutil.rmtree(dest_pkg)
        target_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(found_src, dest_pkg)

        about_file = dest_pkg / "__about__.py"
        synced_ver = "unknown"
        if about_file.is_file():
            m = re.search(
                r'__version__\s*=\s*["\']([^"\']+)["\']',
                about_file.read_text(encoding="utf-8", errors="replace"),
            )
            if m:
                synced_ver = m.group(1)

        shutil.rmtree(tmp_extract, ignore_errors=True)
        self.progress.emit(pct_end, f"Synced {pkg_name} v{synced_ver}")
        return synced_ver

    def _resolve_writable_target_for_pkg(self, pkg_name: str) -> Path | None:
        """If running in non-frozen Python and site-packages/src is writable, return its parent."""
        if getattr(sys, "frozen", False):
            return None
        try:
            mod = __import__(pkg_name)
            pkg_dir = Path(mod.__file__).resolve().parent
            # Avoid overwriting a git working tree with uncommitted changes
            repo_root = pkg_dir.parent.parent
            if (repo_root / ".git").exists():
                res = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if res.returncode == 0 and not res.stdout.strip():
                    # Clean git repo: try git pull --ff-only first
                    pull = subprocess.run(
                        ["git", "pull", "--ff-only"],
                        cwd=repo_root,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if pull.returncode == 0:
                        return pkg_dir.parent
            if os.access(pkg_dir.parent, os.W_OK):
                return pkg_dir.parent
        except Exception:
            pass
        return None

    def run(self):
        try:
            import pymorgan as pm

            bundled_ver = getattr(pm, "__version__", "0.0.0")

            # Determine target root(s)
            pm_target = self._resolve_writable_target_for_pkg("pymorgan") or BLEEDING_EDGE_DIR
            pr_target = self._resolve_writable_target_for_pkg("pyrate_ta") or BLEEDING_EDGE_DIR

            pm_ver = self._download_and_extract_pkg(
                "RJFernandezTeran/PyMORGAN", "pymorgan", pm_target, 10, 55
            )
            pr_ver = self._download_and_extract_pkg(
                "RJFernandezTeran/PyRATE-TA", "pyrate_ta", pr_target, 55, 95
            )

            # Always also populate BLEEDING_EDGE_DIR if frozen or if overlay is used
            if pm_target == BLEEDING_EDGE_DIR or pr_target == BLEEDING_EDGE_DIR or getattr(sys, "frozen", False):
                BLEEDING_EDGE_DIR.mkdir(parents=True, exist_ok=True)
                meta = {
                    "pymorgan_version": pm_ver,
                    "pyrate_ta_version": pr_ver,
                    "bundled_version_at_sync": bundled_ver,
                    "synced_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                (BLEEDING_EDGE_DIR / "sync_meta.json").write_text(
                    json.dumps(meta, indent=2), encoding="utf-8"
                )

            self.progress.emit(100, "Bleeding-edge sync complete!")
            self.finished.emit(pm_ver, pr_ver)
        except Exception as exc:
            self.failed.emit(str(exc))


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
    """Dialog displaying release & bleeding-edge version details, changelog, and sync/download actions."""

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
        self._sync_worker: BleedingEdgeSyncWorker | None = None

        title_prefix = "Update Available" if info.is_update_available else "Update & Sync Center"
        self.setWindowTitle(f"{title_prefix} - {self.app_name}")
        self.setFixedSize(620, 520)
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
                border: none;
                background: transparent;
            }
            QLabel#lblHeadline {
                color: #60a5fa;
                font-size: 15px;
                font-weight: bold;
            }
            QLabel#lblNotes {
                color: #f1f5f9;
                font-size: 13px;
                font-weight: bold;
            }
            QLabel#lblVersionBadge {
                color: #cbd5e1;
                font-size: 12px;
            }
            QTextBrowser {
                background-color: #1e293b;
                color: #f1f5f9;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 10px;
                font-size: 12px;
            }
            QPushButton {
                background-color: #1e293b;
                color: #93c5fd;
                border: 1px solid #3b82f6;
                border-radius: 5px;
                padding: 7px 14px;
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
            QPushButton#btnBleedingEdge {
                background-color: #0d9488;
                color: #ffffff;
                border: 1px solid #2dd4bf;
            }
            QPushButton#btnBleedingEdge:hover {
                background-color: #0f766e;
            }
            QProgressBar {
                border: 1px solid #334155;
                border-radius: 4px;
                background-color: #1e293b;
                text-align: center;
                color: #ffffff;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #0d9488;
                border-radius: 3px;
            }
        """)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        # Header Box
        header = QFrame()
        header.setObjectName("headerBox")
        h_layout = QVBoxLayout(header)
        h_layout.setSpacing(6)

        if self.info.is_update_available:
            headline = f"A newer version of {self.app_name} is available!"
        else:
            headline = f"{self.app_name} is up to date (v{self.info.current_version})"

        lbl_head = QLabel(headline)
        lbl_head.setObjectName("lblHeadline")
        h_layout.addWidget(lbl_head)

        rel_str = f"v{self.info.release_version}" if self.info.release_version else "N/A"
        be_str = f"v{self.info.bleeding_edge_version}" if self.info.bleeding_edge_version else "N/A"
        if self.info.bleeding_edge_commit:
            be_str += f" ({self.info.bleeding_edge_commit})"

        badge_text = (
            f"<b>Installed:</b> v{self.info.current_version} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Latest Release:</b> <span style='color: #38bdf8;'>{rel_str}</span> &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Bleeding Edge (main):</b> <span style='color: #4ade80;'>{be_str}</span>"
        )
        lbl_badge = QLabel(badge_text)
        lbl_badge.setObjectName("lblVersionBadge")
        lbl_badge.setWordWrap(True)
        h_layout.addWidget(lbl_badge)

        layout.addWidget(header)

        # Release Notes Title & Box
        lbl_notes = QLabel("<b>Channel Details & Release Notes:</b>")
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
        self.text_notes.setMarkdown(self.info.release_notes)
        layout.addWidget(self.text_notes, 1)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Button row
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        sync_label = (
            f"Sync Bleeding-Edge (v{self.info.bleeding_edge_version})"
            if self.info.has_bleeding_edge_update
            else "Sync Bleeding-Edge (main)"
        )
        self.btn_sync_be = QPushButton(sync_label)
        self.btn_sync_be.setObjectName("btnBleedingEdge")
        self.btn_sync_be.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_sync_be.setToolTip(
            "Download and sync the latest pure-Python code directly from the GitHub main branch "
            "for both PyMORGAN and PyRATE-TA"
        )
        self.btn_sync_be.clicked.connect(self._on_sync_bleeding_edge)
        btn_layout.addWidget(self.btn_sync_be)

        if self.info.installer_asset:
            asset_name = self.info.installer_asset.get("name", "Installer")
            self.btn_download = QPushButton(f"Download Installer ({asset_name})")
            self.btn_download.setObjectName("btnPrimary")
            self.btn_download.setCursor(Qt.CursorShape.PointingHandCursor)
            self.btn_download.clicked.connect(self._on_download_installer)
            btn_layout.addWidget(self.btn_download)

        self.btn_github = QPushButton("GitHub")
        self.btn_github.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_github.clicked.connect(self._on_open_github)
        btn_layout.addWidget(self.btn_github)

        btn_layout.addStretch()

        self.btn_close = QPushButton("Close")
        self.btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_close.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_close)

        layout.addLayout(btn_layout)

    def _on_open_github(self):
        QDesktopServices.openUrl(QUrl(self.info.html_url))

    def _on_sync_bleeding_edge(self):
        self.btn_sync_be.setEnabled(False)
        self.btn_sync_be.setText("Syncing from main...")
        if hasattr(self, "btn_download"):
            self.btn_download.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self._sync_worker = BleedingEdgeSyncWorker(self)
        self._sync_worker.progress.connect(self._on_sync_progress)
        self._sync_worker.finished.connect(self._on_sync_finished)
        self._sync_worker.failed.connect(self._on_sync_failed)
        self._sync_worker.start()

    def _on_sync_progress(self, pct: int, status_msg: str):
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"{pct}% - {status_msg}")

    def _on_sync_finished(self, pm_ver: str, pr_ver: str):
        self.progress_bar.setValue(100)
        self.btn_sync_be.setText("Synced!")

        mb = QMessageBox(self)
        mb.setIcon(QMessageBox.Icon.Information)
        mb.setWindowTitle("Bleeding-Edge Sync Complete")
        mb.setText(
            f"Successfully synced the latest code from GitHub (main):\n\n"
            f"  * PyMORGAN: v{pm_ver}\n"
            f"  * PyRATE-TA: v{pr_ver}\n\n"
            f"Restart {self.app_name} now to load the updated code?"
        )
        mb.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        mb.setDefaultButton(QMessageBox.StandardButton.Yes)
        _style_message_box(mb)
        reply = mb.exec()

        if reply == QMessageBox.StandardButton.Yes:
            try:
                if getattr(sys, "frozen", False):
                    subprocess.Popen([sys.executable, *sys.argv[1:]])
                else:
                    subprocess.Popen([sys.executable, *sys.argv])
            except Exception:
                pass
            QCoreApplication.quit()
        else:
            self.accept()

    def _on_sync_failed(self, err_msg: str):
        self.btn_sync_be.setEnabled(True)
        self.btn_sync_be.setText("Retry Bleeding-Edge Sync")
        if hasattr(self, "btn_download"):
            self.btn_download.setEnabled(True)
        self.progress_bar.setVisible(False)
        mb = QMessageBox(self)
        mb.setIcon(QMessageBox.Icon.Warning)
        mb.setWindowTitle("Bleeding-Edge Sync Failed")
        mb.setText(f"Could not sync from GitHub main branch:\n{err_msg}")
        _style_message_box(mb)
        mb.exec()

    def _on_download_installer(self):
        if not self.info.installer_asset:
            return

        download_url = self.info.installer_asset.get("browser_download_url")
        file_name = self.info.installer_asset.get("name", "Setup.exe")
        if not download_url:
            self._on_open_github()
            return

        self.btn_download.setEnabled(False)
        self.btn_download.setText("Downloading...")
        self.btn_sync_be.setEnabled(False)
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
        self.btn_download.setText("Download Complete")

        mb = QMessageBox(self)
        mb.setIcon(QMessageBox.Icon.Question)
        mb.setWindowTitle("Update Ready to Install")
        mb.setText(
            f"The update has been downloaded to:\n{local_path}\n\n"
            f"Would you like to close {self.app_name} and open the installer now?"
        )
        mb.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        mb.setDefaultButton(QMessageBox.StandardButton.Yes)
        _style_message_box(mb)
        reply = mb.exec()

        if reply == QMessageBox.StandardButton.Yes:
            if os.name == "nt":
                os.startfile(local_path)  # noqa: S606
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(local_path))
            QCoreApplication.quit()
        else:
            self.accept()

    def _on_download_failed(self, error_msg: str):
        self.btn_download.setEnabled(True)
        self.btn_download.setText("Retry Download")
        self.btn_sync_be.setEnabled(True)
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
            border: none;
            background: transparent;
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
    """Check for updates (both official releases and bleeding-edge main) and open UpdateDialog."""
    if status_callback:
        status_callback(f"Checking for updates for {app_name}...")

    worker = UpdateCheckWorker(github_url, current_version, parent)

    def on_success(info: UpdateInfo):
        if status_callback:
            status_callback("")
        dlg = UpdateDialog(info, app_name, parent)
        dlg.exec()

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
    parent._active_update_worker = worker


def check_for_updates_background(
    window: QMainWindow,
    github_url: str,
    current_version: str,
    app_name: str,
    force: bool = False,
) -> None:
    """Run a quiet, passive update check in the background (checks both release and main)."""
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

        existing_btn = sb.findChild(QPushButton, "btnStatusUpdateNotification")
        if existing_btn is not None:
            return

        channel_tag = " (main)" if info.has_bleeding_edge_update and not info.has_release_update else ""
        btn = QPushButton(f"Update Available: {app_name} v{info.latest_version}{channel_tag}")
        btn.setObjectName("btnStatusUpdateNotification")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip("A newer version is available on GitHub! Click to sync or download.")
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
    worker.start()
    window._bg_update_worker = worker
