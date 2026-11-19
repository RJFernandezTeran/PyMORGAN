"""GitHub Releases and Bleeding-Edge (main branch) updater for PyMORGAN and PyRATE-TA.

Provides non-blocking background update checking against both official GitHub
Releases and the repository ``main`` branch ``__about__.py`` ("Bleeding Edge"),
version comparison via ``packaging.version``, one-click bleeding-edge source
code syncing, and direct platform installer download.
"""

from __future__ import annotations

import html
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import defusedxml.ElementTree as DET
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
UPDATER_CACHE_FILE = Path.home() / ".config" / "pymorgan" / "updater_cache.json"
BACKGROUND_CHECK_COOLDOWN_S = 900.0
_ATOM_NS = "{http://www.w3.org/2005/Atom}"


def _is_headless_test_env() -> bool:
    """Return True when running inside pytest or an offscreen Qt test harness."""
    return (
        "PYTEST_CURRENT_TEST" in os.environ
        or "pytest" in sys.modules
        or os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen"
    )


def _github_headers(*, accept_json: bool = False) -> dict[str, str]:
    """Build HTTP headers for GitHub requests, attaching an env token if available."""
    headers = {"User-Agent": "PyMORGAN-PyRATE-TA-Updater"}
    if accept_json:
        headers["Accept"] = "application/vnd.github.v3+json"
    if not _is_headless_test_env():
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
        if token.strip():
            headers["Authorization"] = f"Bearer {token.strip()}"
    return headers


def _load_updater_cache() -> dict[str, Any]:
    """Load persistent ETag/cooldown cache (disabled in headless pytest runs)."""
    if _is_headless_test_env() or not UPDATER_CACHE_FILE.is_file():
        return {}
    try:
        data = json.loads(UPDATER_CACHE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_updater_cache(cache: dict[str, Any]) -> None:
    """Persist ETag/cooldown cache without interrupting updater flows on I/O errors."""
    if _is_headless_test_env():
        return
    try:
        UPDATER_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        UPDATER_CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass


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


def _atom_find(parent: Any, tag: str) -> Any:
    """Find child element matching ``tag`` with or without Atom XML namespace."""
    el = parent.find(f"{_ATOM_NS}{tag}")
    return el if el is not None else parent.find(tag)


def _atom_findall(parent: Any, tag: str) -> list[Any]:
    """Find child elements matching ``tag`` with or without Atom XML namespace."""
    els = parent.findall(f"{_ATOM_NS}{tag}")
    return els if els else parent.findall(tag)


def _atom_html_to_markdown(raw_html: str) -> str:
    """Convert GitHub Atom HTML release notes to simple Markdown for UpdateDialog."""
    if not raw_html or not raw_html.strip():
        return ""
    txt = raw_html
    txt = re.sub(r"<h[1-3][^>]*>(.*?)</h[1-3]>", r"\n### \1\n", txt, flags=re.I | re.S)
    txt = re.sub(r"<li[^>]*>(.*?)</li>", r"\n- \1", txt, flags=re.I | re.S)
    txt = re.sub(r"<(?:code|tt)[^>]*>(.*?)</(?:code|tt)>", r"`\1`", txt, flags=re.I | re.S)
    txt = re.sub(r"<(?:strong|b)[^>]*>(.*?)</(?:strong|b)>", r"**\1**", txt, flags=re.I | re.S)
    txt = re.sub(
        r"<a[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
        r"[\2](\1)",
        txt,
        flags=re.I | re.S,
    )
    txt = re.sub(r"<br\s*/?>|</(?:p|div|pre|ul|ol)>", "\n", txt, flags=re.I)
    txt = re.sub(r"<[^>]+>", "", txt)
    txt = html.unescape(txt)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in txt.splitlines()]
    return "\n".join(ln for ln in lines if ln).strip()


def _fetch_commits_atom(repo_slug: str, timeout: float = 5.0) -> tuple[str, str, str]:
    """Fetch latest commit SHA, date, and title from GitHub's zero-quota Atom feed."""
    atom_url = f"https://github.com/{repo_slug}/commits/main.atom"
    req = urllib.request.Request(atom_url, headers={"User-Agent": "PyMORGAN-PyRATE-TA-Updater"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        xml_bytes = resp.read()
    root = DET.fromstring(xml_bytes)
    entry = _atom_find(root, "entry")
    if entry is None:
        return "", "", ""

    commit_sha = ""
    for link_el in _atom_findall(entry, "link"):
        href = link_el.attrib.get("href", "")
        if "/commit/" in href:
            commit_sha = href.rsplit("/commit/", 1)[-1].strip()[:7]
            break
    if not commit_sha:
        id_el = _atom_find(entry, "id")
        if id_el is not None and id_el.text and "/" in id_el.text:
            commit_sha = id_el.text.rsplit("/", 1)[-1].strip()[:7]

    upd_el = _atom_find(entry, "updated")
    commit_date = (upd_el.text or "").strip()[:10] if upd_el is not None else ""

    commit_msg = ""
    content_el = _atom_find(entry, "content")
    if content_el is not None and content_el.text:
        plain = html.unescape(re.sub(r"<[^>]+>", "", content_el.text)).strip()
        if plain:
            commit_msg = plain.splitlines()[0].strip()
    if not commit_msg:
        title_el = _atom_find(entry, "title")
        if title_el is not None and title_el.text:
            commit_msg = title_el.text.strip().splitlines()[0].strip()

    return commit_sha, commit_date, commit_msg


def _fetch_bleeding_edge_metadata(
    repo_slug: str, timeout: float = 5.0
) -> tuple[str, str, str, str]:
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
        commit_sha, commit_date, commit_msg = _fetch_commits_atom(repo_slug, timeout=timeout)
    except Exception:
        pass

    if not commit_sha:
        try:
            commit_url = f"https://api.github.com/repos/{repo_slug}/commits/main"
            req_c = urllib.request.Request(
                commit_url,
                headers=_github_headers(accept_json=True),
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


def _fallback_platform_asset(
    repo_slug: str,
    tag_name: str,
    release_ver: str,
    timeout: float = 3.5,
) -> dict[str, Any] | None:
    """Probe predictable GitHub release asset URLs when API metadata is unavailable."""
    if not tag_name or not release_ver or "pymorgan" not in repo_slug.lower():
        return None
    if sys.platform == "win32":
        candidates = (
            f"PyMORGAN_Suite_Setup_v{release_ver}.exe",
            f"PyMORGAN_Suite_v{release_ver}_Windows_x64_Portable.zip",
        )
    elif sys.platform == "darwin":
        candidates = (f"PyMORGAN_Suite_v{release_ver}_macOS.dmg",)
    else:
        candidates = (f"PyMORGAN_Suite_v{release_ver}_Linux_x86_64.tar.gz",)

    probe_timeout = min(timeout, 3.5)
    for name in candidates:
        url = f"https://github.com/{repo_slug}/releases/download/{tag_name}/{name}"
        try:
            req = urllib.request.Request(
                url,
                method="HEAD",
                headers={"User-Agent": "PyMORGAN-PyRATE-TA-Updater"},
            )
            with urllib.request.urlopen(req, timeout=probe_timeout) as resp:
                if getattr(resp, "status", 200) < 400:
                    return {"name": name, "browser_download_url": url}
        except Exception:
            continue
    return None


def _fetch_release_via_web(
    repo_slug: str, timeout: float = 6.0
) -> tuple[str, str, str, str, str, dict[str, Any] | None]:
    """Fallback release lookup via GitHub Atom feed or web redirect (zero API quota)."""
    release_ver = ""
    release_name = ""
    release_notes = ""
    html_url = f"https://github.com/{repo_slug}"
    published_at = ""
    tag_name = ""

    atom_url = f"https://github.com/{repo_slug}/releases.atom"
    try:
        req = urllib.request.Request(atom_url, headers={"User-Agent": "PyMORGAN-PyRATE-TA-Updater"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            xml_bytes = resp.read()
        root = DET.fromstring(xml_bytes)
        entry = _atom_find(root, "entry")
        if entry is not None:
            for link_el in _atom_findall(entry, "link"):
                href = link_el.attrib.get("href", "")
                if "/releases/tag/" in href:
                    html_url = href
                    tag_name = href.rsplit("/releases/tag/", 1)[-1].strip()
                    break
            if not tag_name:
                id_el = _atom_find(entry, "id")
                if id_el is not None and id_el.text and "/" in id_el.text:
                    tag_name = id_el.text.rsplit("/", 1)[-1].strip()
            title_el = _atom_find(entry, "title")
            title_txt = (title_el.text or "").strip() if title_el is not None else ""
            if not tag_name:
                tag_name = title_txt
            release_ver = normalize_version(tag_name)
            release_name = title_txt or tag_name
            upd_el = _atom_find(entry, "updated")
            published_at = (upd_el.text or "").strip()[:10] if upd_el is not None else ""
            content_el = _atom_find(entry, "content")
            if content_el is not None and content_el.text:
                release_notes = _atom_html_to_markdown(content_el.text)
    except Exception:
        pass

    if not release_ver:
        latest_url = f"https://github.com/{repo_slug}/releases/latest"
        try:
            req = urllib.request.Request(
                latest_url, headers={"User-Agent": "PyMORGAN-PyRATE-TA-Updater"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                final_url = str(resp.geturl())
            if "/releases/tag/" in final_url:
                html_url = final_url
                tag_name = final_url.rsplit("/releases/tag/", 1)[-1].split("?")[0].strip()
                release_ver = normalize_version(tag_name)
                release_name = tag_name
        except Exception:
            pass

    installer_asset = _fallback_platform_asset(repo_slug, tag_name, release_ver, timeout=timeout)
    return release_ver, release_name, release_notes, html_url, published_at, installer_asset


def _recompute_update_flags(info: UpdateInfo, current_version: str) -> UpdateInfo:
    """Re-evaluate version comparison flags on an UpdateInfo instance for ``current_version``."""
    cur_norm = normalize_version(current_version)
    has_rel = is_newer_version(info.release_version, cur_norm) if info.release_version else False
    has_be = (
        is_newer_version(info.bleeding_edge_version, cur_norm)
        if info.bleeding_edge_version
        else False
    )
    if info.bleeding_edge_version and info.release_version:
        latest = (
            info.bleeding_edge_version
            if is_newer_version(info.bleeding_edge_version, info.release_version)
            else info.release_version
        )
    else:
        latest = info.bleeding_edge_version or info.release_version or cur_norm
    return UpdateInfo(
        is_update_available=has_rel or has_be,
        current_version=cur_norm,
        latest_version=latest,
        release_name=info.release_name or f"v{latest}",
        release_notes=info.release_notes,
        html_url=info.html_url,
        published_at=info.published_at or info.bleeding_edge_date,
        installer_asset=info.installer_asset,
        release_version=info.release_version,
        bleeding_edge_version=info.bleeding_edge_version,
        bleeding_edge_commit=info.bleeding_edge_commit,
        bleeding_edge_date=info.bleeding_edge_date,
        bleeding_edge_message=info.bleeding_edge_message,
        has_release_update=has_rel,
        has_bleeding_edge_update=has_be,
    )


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

    cache = _load_updater_cache()
    repo_cache = cache.get(repo_slug, {})
    if not isinstance(repo_cache, dict):
        repo_cache = {}

    headers = _github_headers(accept_json=True)
    cached_etag = repo_cache.get("etag_release")
    if isinstance(cached_etag, str) and cached_etag:
        headers["If-None-Match"] = cached_etag

    release_ver = ""
    release_name = ""
    release_notes = ""
    html_url = f"https://github.com/{repo_slug}"
    published_at = ""
    installer_asset = None

    def _populate_from_release_json(data: dict[str, Any]) -> None:
        nonlocal release_ver, release_name, release_notes, html_url, published_at, installer_asset
        tag_name = str(data.get("tag_name", "") or "")
        release_ver = normalize_version(tag_name)
        release_name = str(data.get("name") or tag_name)
        release_notes = str(data.get("body") or "")
        html_url = str(data.get("html_url") or html_url)
        published_at = str(data.get("published_at", "") or "")[:10]
        installer_asset = _select_platform_asset(data.get("assets", []))

    # 1. Query latest formal GitHub Release (with ETag cache and zero-quota Atom/web fallback)
    api_url = f"https://api.github.com/repos/{repo_slug}/releases/latest"
    need_web_fallback = False
    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, dict):
                    _populate_from_release_json(data)
                    resp_headers = getattr(resp, "headers", None)
                    new_etag = (
                        resp_headers.get("ETag") if hasattr(resp_headers, "get") else None
                    )
                    if isinstance(new_etag, str) and new_etag:
                        repo_cache["etag_release"] = new_etag
                        repo_cache["release_payload"] = data
                        cache[repo_slug] = repo_cache
                        _save_updater_cache(cache)
    except urllib.error.HTTPError as http_err:
        if http_err.code == 304 and isinstance(repo_cache.get("release_payload"), dict):
            _populate_from_release_json(repo_cache["release_payload"])
        elif http_err.code != 404:
            need_web_fallback = True
    except Exception:
        need_web_fallback = True

    if need_web_fallback and not release_ver:
        (
            release_ver,
            release_name,
            release_notes,
            fb_url,
            published_at,
            installer_asset,
        ) = _fetch_release_via_web(repo_slug, timeout=timeout)
        if fb_url:
            html_url = fb_url

    # 2. Query main branch ("Bleeding Edge") __about__.py and commits Atom feed
    be_ver, be_sha, be_date, be_msg = _fetch_bleeding_edge_metadata(repo_slug, timeout=timeout)

    if not release_ver and not be_ver:
        raise RuntimeError(f"Could not retrieve release or main branch version from {repo_slug}.")

    info = UpdateInfo(
        is_update_available=False,
        current_version=normalize_version(current_version),
        latest_version=normalize_version(current_version),
        release_name=release_name,
        release_notes=release_notes,
        html_url=html_url,
        published_at=published_at,
        installer_asset=installer_asset,
        release_version=release_ver,
        bleeding_edge_version=be_ver,
        bleeding_edge_commit=be_sha,
        bleeding_edge_date=be_date,
        bleeding_edge_message=be_msg,
    )
    return _recompute_update_flags(info, current_version)


def _is_dark_mode() -> bool:
    """Detect whether the host Qt application uses a dark palette."""
    try:
        from pymorgan.gui.theme import is_dark_palette

        return bool(is_dark_palette())
    except Exception:
        return False


def _markdown_to_styled_html(md_text: str, dark: bool) -> str:
    """Convert simple GitHub release-note markdown into high-contrast HTML without Qt's broken code spans."""
    if not md_text.strip():
        return ""
    fg = "#e2e8f0" if dark else "#1e293b"
    h_color = "#60a5fa" if dark else "#1d4ed8"
    code_bg = "#0f172a" if dark else "#f1f5f9"
    code_fg = "#38bdf8" if dark else "#0f172a"
    code_border = "#334155" if dark else "#cbd5e1"
    link_color = "#38bdf8" if dark else "#2563eb"

    lines_out: list[str] = []
    in_list = False

    def _inline(text: str) -> str:
        esc = html.escape(text)
        # Inline code `...`
        esc = re.sub(
            r"`([^`]+)`",
            rf'<span style="font-family: Consolas, monospace; background-color: {code_bg}; '
            rf'color: {code_fg}; border: 1px solid {code_border}; border-radius: 3px; padding: 1px 4px;">\1</span>',
            esc,
        )
        # Bold **...**
        esc = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", esc)
        # Links [label](url)
        esc = re.sub(
            r"\[([^\]]+)\]\((https?://[^\)]+)\)",
            rf'<a href="\2" style="color: {link_color}; text-decoration: underline;">\1</a>',
            esc,
        )
        return esc

    for raw_line in md_text.splitlines():
        line = raw_line.strip()
        if not line:
            if in_list:
                lines_out.append("</ul>")
                in_list = False
            continue
        if line.startswith("### ") or line.startswith("## ") or line.startswith("# "):
            if in_list:
                lines_out.append("</ul>")
                in_list = False
            heading = line.lstrip("#").strip()
            lines_out.append(
                f'<div style="color: {h_color}; font-weight: bold; font-size: 12.5px; '
                f'margin-top: 8px; margin-bottom: 4px;">{_inline(heading)}</div>'
            )
        elif line.startswith(("- ", "* ")):
            if not in_list:
                lines_out.append(
                    f'<ul style="margin-top: 2px; margin-bottom: 6px; padding-left: 18px; color: {fg};">'
                )
                in_list = True
            lines_out.append(f'<li style="margin-bottom: 3px;">{_inline(line[2:].strip())}</li>')
        else:
            if in_list:
                lines_out.append("</ul>")
                in_list = False
            lines_out.append(
                f'<p style="margin-top: 3px; margin-bottom: 5px; color: {fg};">{_inline(line)}</p>'
            )

    if in_list:
        lines_out.append("</ul>")
    return "\n".join(lines_out)


def _build_notes_html(info: UpdateInfo, dark: bool) -> str:
    """Build crisp, high-contrast HTML for the UpdateDialog release notes view."""
    fg = "#f1f5f9" if dark else "#0f172a"
    muted = "#94a3b8" if dark else "#475569"
    h_be = "#34d399" if dark else "#0f766e"
    h_rel = "#60a5fa" if dark else "#1d4ed8"
    code_bg = "#0f172a" if dark else "#f1f5f9"
    code_fg = "#38bdf8" if dark else "#0f172a"
    hr_color = "#334155" if dark else "#e2e8f0"

    sections: list[str] = []

    if info.bleeding_edge_version:
        commit_badge = ""
        if info.bleeding_edge_commit:
            date_str = f" on {html.escape(info.bleeding_edge_date)}" if info.bleeding_edge_date else ""
            commit_badge = (
                f' &nbsp;<span style="font-family: Consolas, monospace; background-color: {code_bg}; '
                f'color: {code_fg}; padding: 1px 5px; border-radius: 3px;">'
                f"{html.escape(info.bleeding_edge_commit)}</span>"
                f'<span style="color: {muted}; font-weight: normal;">{date_str}</span>'
            )
        msg_html = ""
        if info.bleeding_edge_message:
            msg_html = (
                f'<div style="margin-top: 4px; margin-bottom: 4px; color: {fg};">'
                f"<b>Latest commit on main:</b> {html.escape(info.bleeding_edge_message)}</div>"
            )
        sections.append(
            f'<div style="color: {h_be}; font-weight: bold; font-size: 13px; margin-bottom: 4px;">'
            f"Bleeding-Edge Channel (main): v{html.escape(info.bleeding_edge_version)}{commit_badge}</div>"
            f"{msg_html}"
            f'<div style="color: {muted}; font-size: 11.5px; margin-bottom: 6px;">'
            f"Syncs pure-Python code directly from the <b>main</b> branch for both PyMORGAN and PyRATE-TA.</div>"
        )

    if info.release_notes:
        rel_html = _markdown_to_styled_html(info.release_notes, dark)
        sections.append(
            f'<div style="color: {h_rel}; font-weight: bold; font-size: 13px; margin-bottom: 4px;">'
            f"Official Release ({html.escape(info.release_name)})</div>"
            f"{rel_html}"
        )
    elif info.release_version:
        sections.append(
            f'<div style="color: {h_rel}; font-weight: bold; font-size: 13px; margin-bottom: 4px;">'
            f"Official Release: v{html.escape(info.release_version)}</div>"
            f'<div style="color: {muted}; font-size: 11.5px;">'
            f"Full standalone installer bundle with pre-compiled dependencies.</div>"
        )

    if not sections:
        sections.append(f'<div style="color: {muted};">No release notes provided.</div>')

    divider = f'<hr style="border: 0; border-top: 1px solid {hr_color}; margin: 10px 0;">'
    body = divider.join(sections)
    return (
        f'<div style="font-family: \'Segoe UI\', -apple-system, Arial, sans-serif; '
        f'font-size: 12px; color: {fg}; line-height: 1.45;">{body}</div>'
    )


class UpdateCheckWorker(QThread):
    """Background worker to check for updates without freezing the GUI."""

    check_finished = pyqtSignal(object)  # UpdateInfo
    check_failed = pyqtSignal(str)

    def __init__(
        self,
        repo_url: str,
        current_version: str,
        parent: QWidget | None = None,
        *,
        use_bg_cooldown: bool = False,
    ):
        super().__init__(parent)
        self.repo_url = repo_url
        self.current_version = current_version
        self.use_bg_cooldown = use_bg_cooldown

    def run(self):
        try:
            clean_url = self.repo_url.rstrip("/")
            repo_slug = (
                clean_url.split("github.com/")[-1] if "github.com/" in clean_url else clean_url
            )
            headless = _is_headless_test_env()

            if self.use_bg_cooldown and not headless:
                cache = _load_updater_cache()
                repo_cache = cache.get(repo_slug, {})
                bg_entry = (
                    repo_cache.get("last_bg_check") if isinstance(repo_cache, dict) else None
                )
                if (
                    isinstance(bg_entry, dict)
                    and (time.time() - float(bg_entry.get("ts", 0.0))) < BACKGROUND_CHECK_COOLDOWN_S
                    and isinstance(bg_entry.get("info"), dict)
                ):
                    try:
                        cached_info = UpdateInfo(**bg_entry["info"])
                        self.check_finished.emit(
                            _recompute_update_flags(cached_info, self.current_version)
                        )
                        return
                    except Exception:
                        pass

            info = fetch_latest_release(self.repo_url, self.current_version)
            if not headless:
                cache = _load_updater_cache()
                repo_cache = cache.get(repo_slug, {})
                if not isinstance(repo_cache, dict):
                    repo_cache = {}
                repo_cache["last_bg_check"] = {"ts": time.time(), "info": asdict(info)}
                cache[repo_slug] = repo_cache
                _save_updater_cache(cache)

            self.check_finished.emit(info)
        except urllib.error.HTTPError as err:
            self.check_failed.emit(f"GitHub HTTP {err.code}: {err.reason}")
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
            meta_file = BLEEDING_EDGE_DIR / "sync_meta.json"
            if str(BLEEDING_EDGE_DIR) in sys.path and meta_file.is_file():
                try:
                    prev_meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    bundled_ver = prev_meta.get("bundled_version_at_sync") or bundled_ver
                except Exception:
                    pass

            # Determine target root(s)
            pm_target = self._resolve_writable_target_for_pkg("pymorgan") or BLEEDING_EDGE_DIR
            pr_target = self._resolve_writable_target_for_pkg("pyrate_ta") or BLEEDING_EDGE_DIR

            pm_ver = self._download_and_extract_pkg(
                "RJFernandezTeran/PyMORGAN", "pymorgan", pm_target, 10, 50
            )
            pr_ver = self._download_and_extract_pkg(
                "RJFernandezTeran/PyRATE-TA", "pyrate_ta", pr_target, 50, 90
            )
            self.progress.emit(92, "Verifying Matplotlib export backends...")
            try:
                from pymorgan.gui._mpl_export_shim import ensure_mpl_export_backends

                ensure_mpl_export_backends()
            except Exception:
                pass

            # Always also populate BLEEDING_EDGE_DIR if frozen or if overlay is used
            if (
                pm_target == BLEEDING_EDGE_DIR
                or pr_target == BLEEDING_EDGE_DIR
                or getattr(sys, "frozen", False)
            ):
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
    finished = pyqtSignal(str)  # local_file_path
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
        self.setObjectName("UpdateCenterDialog")
        self.info = info
        self.app_name = app_name
        self._dark = _is_dark_mode()
        self._download_worker: DownloadWorker | None = None
        self._sync_worker: BleedingEdgeSyncWorker | None = None

        title_prefix = "Update Available" if info.is_update_available else "Update & Sync Center"
        self.setWindowTitle(f"{title_prefix} - {self.app_name}")
        self.setFixedSize(620, 520)
        self.setModal(True)

        self._apply_theme_stylesheet()
        self._setup_ui()

    def _apply_theme_stylesheet(self) -> None:
        if self._dark:
            self.setStyleSheet("""
                QDialog#UpdateCenterDialog {
                    background-color: #181c24;
                    color: #f1f5f9;
                    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Arial, sans-serif;
                }
                QDialog#UpdateCenterDialog QFrame#headerBox {
                    background-color: #222834;
                    border: 1px solid #3b82f6;
                    border-radius: 8px;
                    padding: 12px;
                }
                QDialog#UpdateCenterDialog QLabel {
                    color: #f1f5f9;
                    border: none;
                    background: transparent;
                }
                QDialog#UpdateCenterDialog QLabel#lblHeadline {
                    color: #60a5fa;
                    font-size: 15px;
                    font-weight: bold;
                }
                QDialog#UpdateCenterDialog QLabel#lblNotes {
                    color: #f1f5f9;
                    font-size: 13px;
                    font-weight: bold;
                }
                QDialog#UpdateCenterDialog QLabel#lblVersionBadge {
                    color: #cbd5e1;
                    font-size: 12px;
                }
                QDialog#UpdateCenterDialog QTextBrowser {
                    background-color: #12161d;
                    color: #f1f5f9;
                    border: 1px solid #334155;
                    border-radius: 6px;
                    padding: 10px;
                    font-size: 12px;
                }
                QDialog#UpdateCenterDialog QPushButton {
                    background-color: #222834;
                    color: #e2e8f0;
                    border: 1px solid #475569;
                    border-radius: 6px;
                    padding: 6px 14px;
                    font-weight: bold;
                    font-size: 12px;
                }
                QDialog#UpdateCenterDialog QPushButton:hover {
                    background-color: #334155;
                    color: #ffffff;
                }
                QDialog#UpdateCenterDialog QPushButton#btnPrimary {
                    background-color: #2563eb;
                    color: #ffffff;
                    border: 1px solid #60a5fa;
                }
                QDialog#UpdateCenterDialog QPushButton#btnPrimary:hover {
                    background-color: #1d4ed8;
                }
                QDialog#UpdateCenterDialog QPushButton#btnBleedingEdge {
                    background-color: #0d9488;
                    color: #ffffff;
                    border: 1px solid #2dd4bf;
                }
                QDialog#UpdateCenterDialog QPushButton#btnBleedingEdge:hover {
                    background-color: #0f766e;
                }
                QDialog#UpdateCenterDialog QProgressBar {
                    border: 1px solid #334155;
                    border-radius: 4px;
                    background-color: #12161d;
                    text-align: center;
                    color: #ffffff;
                    height: 20px;
                }
                QDialog#UpdateCenterDialog QProgressBar::chunk {
                    background-color: #0d9488;
                    border-radius: 3px;
                }
            """)
        else:
            self.setStyleSheet("""
                QDialog#UpdateCenterDialog {
                    background-color: #f8fafc;
                    color: #0f172a;
                    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Arial, sans-serif;
                }
                QDialog#UpdateCenterDialog QFrame#headerBox {
                    background-color: #ffffff;
                    border: 1px solid #93c5fd;
                    border-left: 4px solid #2563eb;
                    border-radius: 8px;
                    padding: 12px;
                }
                QDialog#UpdateCenterDialog QLabel {
                    color: #0f172a;
                    border: none;
                    background: transparent;
                }
                QDialog#UpdateCenterDialog QLabel#lblHeadline {
                    color: #1d4ed8;
                    font-size: 15px;
                    font-weight: bold;
                }
                QDialog#UpdateCenterDialog QLabel#lblNotes {
                    color: #0f172a;
                    font-size: 13px;
                    font-weight: bold;
                }
                QDialog#UpdateCenterDialog QLabel#lblVersionBadge {
                    color: #334155;
                    font-size: 12px;
                }
                QDialog#UpdateCenterDialog QTextBrowser {
                    background-color: #ffffff;
                    color: #0f172a;
                    border: 1px solid #cbd5e1;
                    border-radius: 6px;
                    padding: 10px;
                    font-size: 12px;
                }
                QDialog#UpdateCenterDialog QPushButton {
                    background-color: #ffffff;
                    color: #1e293b;
                    border: 1px solid #cbd5e1;
                    border-radius: 6px;
                    padding: 6px 14px;
                    font-weight: bold;
                    font-size: 12px;
                }
                QDialog#UpdateCenterDialog QPushButton:hover {
                    background-color: #e2e8f0;
                    color: #0f172a;
                }
                QDialog#UpdateCenterDialog QPushButton#btnPrimary {
                    background-color: #2563eb;
                    color: #ffffff;
                    border: 1px solid #1d4ed8;
                }
                QDialog#UpdateCenterDialog QPushButton#btnPrimary:hover {
                    background-color: #1d4ed8;
                }
                QDialog#UpdateCenterDialog QPushButton#btnBleedingEdge {
                    background-color: #0d9488;
                    color: #ffffff;
                    border: 1px solid #0f766e;
                }
                QDialog#UpdateCenterDialog QPushButton#btnBleedingEdge:hover {
                    background-color: #0f766e;
                }
                QDialog#UpdateCenterDialog QProgressBar {
                    border: 1px solid #cbd5e1;
                    border-radius: 4px;
                    background-color: #e2e8f0;
                    text-align: center;
                    color: #0f172a;
                    font-weight: bold;
                    height: 20px;
                }
                QDialog#UpdateCenterDialog QProgressBar::chunk {
                    background-color: #0d9488;
                    border-radius: 3px;
                }
            """)

    def _default_sync_button_text(self) -> str:
        ver_sub = (
            f"main \u00b7 v{self.info.bleeding_edge_version}"
            if self.info.bleeding_edge_version
            else "main branch"
        )
        return f"Sync Bleeding-Edge\n{ver_sub}"

    def _default_download_button_text(self) -> str:
        ver_sub = (
            f"Release \u00b7 v{self.info.release_version}"
            if self.info.release_version
            else "Official Setup"
        )
        return f"Download Installer\n{ver_sub}"

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
        be_str = (
            f"v{self.info.bleeding_edge_version}" if self.info.bleeding_edge_version else "N/A"
        )
        if self.info.bleeding_edge_commit:
            be_str += f" ({self.info.bleeding_edge_commit})"

        rel_color = "#38bdf8" if self._dark else "#0284c7"
        be_color = "#4ade80" if self._dark else "#0d9488"

        badge_text = (
            f"<b>Installed:</b> v{self.info.current_version} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Latest Release:</b> <span style='color: {rel_color}; font-weight: bold;'>{rel_str}</span> &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Bleeding Edge (main):</b> <span style='color: {be_color}; font-weight: bold;'>{be_str}</span>"
        )
        lbl_badge = QLabel(badge_text)
        lbl_badge.setObjectName("lblVersionBadge")
        lbl_badge.setWordWrap(True)
        h_layout.addWidget(lbl_badge)

        layout.addWidget(header)

        # Release Notes Title & Box
        lbl_notes = QLabel("<b>Channel Details &amp; Release Notes:</b>")
        lbl_notes.setObjectName("lblNotes")
        layout.addWidget(lbl_notes)

        self.text_notes = QTextBrowser()
        self.text_notes.setOpenExternalLinks(True)
        self.text_notes.setHtml(_build_notes_html(self.info, self._dark))
        layout.addWidget(self.text_notes, 1)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Button row (2-line stacked action buttons)
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.btn_sync_be = QPushButton(self._default_sync_button_text())
        self.btn_sync_be.setObjectName("btnBleedingEdge")
        self.btn_sync_be.setMinimumHeight(44)
        self.btn_sync_be.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_sync_be.setToolTip(
            "Download and sync the latest pure-Python code directly from the GitHub main branch "
            "for both PyMORGAN and PyRATE-TA"
        )
        self.btn_sync_be.clicked.connect(self._on_sync_bleeding_edge)
        btn_layout.addWidget(self.btn_sync_be)

        if self.info.installer_asset:
            asset_name = self.info.installer_asset.get("name", "Installer")
            self.btn_download = QPushButton(self._default_download_button_text())
            self.btn_download.setObjectName("btnPrimary")
            self.btn_download.setMinimumHeight(44)
            self.btn_download.setCursor(Qt.CursorShape.PointingHandCursor)
            self.btn_download.setToolTip(f"Download official release installer: {asset_name}")
            self.btn_download.clicked.connect(self._on_download_installer)
            btn_layout.addWidget(self.btn_download)

        self.btn_github = QPushButton("GitHub")
        self.btn_github.setMinimumHeight(44)
        self.btn_github.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_github.setToolTip(f"Open {self.info.html_url} in your browser")
        self.btn_github.clicked.connect(self._on_open_github)
        btn_layout.addWidget(self.btn_github)

        btn_layout.addStretch()

        self.btn_close = QPushButton("Close")
        self.btn_close.setMinimumHeight(44)
        self.btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_close.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_close)

        layout.addLayout(btn_layout)

    def _on_open_github(self):
        QDesktopServices.openUrl(QUrl(self.info.html_url))

    def _on_sync_bleeding_edge(self):
        self.btn_sync_be.setEnabled(False)
        self.btn_sync_be.setText("Syncing...\nfrom main")
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
        self.btn_sync_be.setText(f"Synced!\nv{pm_ver}")

        mb = _create_message_box(
            self,
            icon=QMessageBox.Icon.Information,
            title="Bleeding-Edge Sync Complete",
            text=(
                f"Successfully synced the latest code from GitHub (main):\n\n"
                f"  \u2022 PyMORGAN: v{pm_ver}\n"
                f"  \u2022 PyRATE-TA: v{pr_ver}\n\n"
                f"Restart {self.app_name} now to load the updated code?"
            ),
            buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            default_button=QMessageBox.StandardButton.Yes,
        )
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
        self.btn_sync_be.setText("Retry Sync\nmain branch")
        if hasattr(self, "btn_download"):
            self.btn_download.setEnabled(True)
        self.progress_bar.setVisible(False)
        mb = _create_message_box(
            self,
            icon=QMessageBox.Icon.Warning,
            title="Bleeding-Edge Sync Failed",
            text=f"Could not sync from GitHub main branch:\n{err_msg}",
        )
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
        self.btn_download.setText("Downloading...\nPlease wait")
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
        self.btn_download.setText("Downloaded!\nReady to install")

        mb = _create_message_box(
            self,
            icon=QMessageBox.Icon.Question,
            title="Update Ready to Install",
            text=(
                f"The update has been downloaded to:\n{local_path}\n\n"
                f"Would you like to close {self.app_name} and open the installer now?"
            ),
            buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            default_button=QMessageBox.StandardButton.Yes,
        )
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
        self.btn_download.setText("Retry Download\nInstaller")
        self.btn_sync_be.setEnabled(True)
        self.progress_bar.setVisible(False)
        mb = _create_message_box(
            self,
            icon=QMessageBox.Icon.Warning,
            title="Download Failed",
            text=(
                f"Could not download the update:\n{error_msg}\n\n"
                "You can download it manually from the GitHub release page."
            ),
        )
        mb.exec()


def _create_message_box(
    parent: QWidget | None,
    *,
    icon: QMessageBox.Icon,
    title: str,
    text: str,
    buttons: QMessageBox.StandardButton = QMessageBox.StandardButton.Ok,
    default_button: QMessageBox.StandardButton | None = None,
) -> QMessageBox:
    """Create a theme-aware QMessageBox with guaranteed readable foreground/background contrast."""
    mb = QMessageBox(parent)
    mb.setObjectName("UpdateMessageBox")
    mb.setIcon(icon)
    mb.setWindowTitle(title)
    mb.setText(text)
    mb.setStandardButtons(buttons)
    if default_button is not None:
        mb.setDefaultButton(default_button)
    return _style_message_box(mb)


def _style_message_box(mb: QMessageBox) -> QMessageBox:
    """Style QMessageBox to match the active Light or Dark application palette with crisp contrast."""
    dark = _is_dark_mode()
    if dark:
        mb.setStyleSheet("""
            QMessageBox {
                background-color: #222834;
                color: #f8fafc;
            }
            QMessageBox QLabel {
                color: #f8fafc;
                font-size: 12.5px;
                min-width: 300px;
                border: none;
                background: transparent;
            }
            QMessageBox QPushButton {
                background-color: #2563eb;
                color: #ffffff;
                border: 1px solid #60a5fa;
                border-radius: 5px;
                padding: 6px 18px;
                font-weight: bold;
                font-size: 12px;
                min-width: 72px;
            }
            QMessageBox QPushButton:hover {
                background-color: #1d4ed8;
                color: #ffffff;
            }
        """)
    else:
        mb.setStyleSheet("""
            QMessageBox {
                background-color: #ffffff;
                color: #0f172a;
            }
            QMessageBox QLabel {
                color: #0f172a;
                font-size: 12.5px;
                min-width: 300px;
                border: none;
                background: transparent;
            }
            QMessageBox QPushButton {
                background-color: #f8fafc;
                color: #0f172a;
                border: 1px solid #94a3b8;
                border-radius: 5px;
                padding: 6px 18px;
                font-weight: bold;
                font-size: 12px;
                min-width: 72px;
            }
            QMessageBox QPushButton:hover {
                background-color: #2563eb;
                color: #ffffff;
                border-color: #1d4ed8;
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
        mb = _create_message_box(
            parent,
            icon=QMessageBox.Icon.Warning,
            title="Update Check Failed",
            text=(
                f"Unable to check for updates for {app_name}:\n{err_msg}\n\n"
                "Please check your internet connection or visit the GitHub repository."
            ),
        )
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
    show_popup: bool = True,
) -> None:
    """Check for updates on startup (both release and main) and notify the user with a popup."""
    if not force and _is_headless_test_env():
        return

    worker = UpdateCheckWorker(
        github_url,
        current_version,
        window,
        use_bg_cooldown=not force,
    )

    def on_success(info: UpdateInfo):
        if not info.is_update_available:
            return

        if hasattr(window, "statusBar"):
            sb = window.statusBar()
            if sb is not None and sb.findChild(QPushButton, "btnStatusUpdateNotification") is None:
                channel_tag = (
                    " (main)"
                    if info.has_bleeding_edge_update and not info.has_release_update
                    else ""
                )
                btn = QPushButton(
                    f"Update Available: {app_name} v{info.latest_version}{channel_tag}"
                )
                btn.setObjectName("btnStatusUpdateNotification")
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.setToolTip(
                    "A newer version is available on GitHub! Click to sync or download."
                )
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

        if show_popup and window.isVisible():
            shared_settings = QSettings("RJFernandezTeran", "PyMORGAN_Suite_Updater")
            now = time.time()
            last_popup_ts = float(shared_settings.value("last_popup_ts", 0) or 0)
            last_popup_ver = str(shared_settings.value("last_popup_ver", "") or "")
            # Avoid duplicate popup within 45s if user just dismissed it in Suite Launcher
            if not force and last_popup_ver == info.latest_version and (now - last_popup_ts < 45):
                return
            shared_settings.setValue("last_popup_ts", now)
            shared_settings.setValue("last_popup_ver", info.latest_version)
            dlg = UpdateDialog(info, app_name, window)
            dlg.exec()

    worker.check_finished.connect(on_success)
    worker.start()
    window._bg_update_worker = worker
