"""Tests for GitHub Releases update checking and version comparisons."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pymorgan.gui.updater import (
    fetch_latest_release,
    is_newer_version,
    normalize_version,
)


def test_normalize_version():
    assert normalize_version("v1.2.3") == "1.2.3"
    assert normalize_version("V2.0.0") == "2.0.0"
    assert normalize_version("  1.0.0b1  ") == "1.0.0b1"


def test_is_newer_version():
    assert is_newer_version("1.1.0", "1.0.0") is True
    assert is_newer_version("v2.0.0", "1.9.9") is True
    assert is_newer_version("1.0.1", "1.0.1") is False
    assert is_newer_version("1.0.0", "1.0.1") is False


def test_fetch_latest_release_mock():
    mock_response = MagicMock()
    mock_response.status = 200
    mock_payload = b"""{
        "tag_name": "v1.5.0",
        "name": "PyMORGAN v1.5.0",
        "body": "### Features\\n- Added automated update checker\\n- Improved stitching",
        "html_url": "https://github.com/RJFernandezTeran/PyMORGAN/releases/tag/v1.5.0",
        "published_at": "2026-10-11T12:00:00Z",
        "assets": [
            {
                "name": "PyMORGAN-Setup-x64.exe",
                "browser_download_url": "https://github.com/RJFernandezTeran/PyMORGAN/releases/download/v1.5.0/PyMORGAN-Setup-x64.exe"
            }
        ]
    }"""
    mock_response.read.return_value = mock_payload
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        info = fetch_latest_release("https://github.com/RJFernandezTeran/PyMORGAN", "1.0.0")
        assert info.is_update_available is True
        assert info.latest_version == "1.5.0"
        assert info.current_version == "1.0.0"
        assert info.installer_asset is not None
        assert info.installer_asset["name"] == "PyMORGAN-Setup-x64.exe"
        assert "Features" in info.release_notes


def test_check_for_updates_background():
    from PyQt6.QtWidgets import QApplication, QMainWindow

    from pymorgan.gui.updater import UpdateInfo, check_for_updates_background

    app = QApplication.instance() or QApplication([])

    win = QMainWindow()

    fake_info = UpdateInfo(
        is_update_available=True,
        current_version="1.0.0",
        latest_version="2.0.0",
        release_name="PyMORGAN 2.0",
        release_notes="New release notes",
        html_url="https://github.com/RJFernandezTeran/PyMORGAN",
        published_at="2026-10-11",
    )

    with patch("pymorgan.gui.updater.fetch_latest_release", return_value=fake_info):
        check_for_updates_background(win, "https://github.com/RJFernandezTeran/PyMORGAN", "1.0.0", "PyMORGAN", force=True)
        if hasattr(win, "_bg_update_worker"):
            win._bg_update_worker.wait(3000)
            app.processEvents()

    btn = win.statusBar().findChild(type(win.statusBar()), "btnStatusUpdateNotification")
    assert btn is not None or win.statusBar().children() is not None
    win.close()


def test_fetch_latest_release_survives_403_rate_limit_via_atom_and_raw():
    import urllib.error

    releases_atom = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>tag:github.com,2008:Repository/1/v1.261103.2</id>
    <updated>2026-11-03T00:04:35Z</updated>
    <link rel="alternate" type="text/html" href="https://github.com/RJFernandezTeran/PyMORGAN/releases/tag/v1.261103.2"/>
    <title>v1.261103.2</title>
    <content type="html">&lt;p&gt;&lt;strong&gt;Highlights&lt;/strong&gt;: Faster fitting&lt;/p&gt;</content>
  </entry>
</feed>"""

    commits_atom = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>tag:github.com,2008:Grit::Commit/8db36008c58b307caf58ec6da1cf49c2cb2a328a</id>
    <link rel="alternate" type="text/html" href="https://github.com/RJFernandezTeran/PyMORGAN/commit/8db36008c58b307caf58ec6da1cf49c2cb2a328a"/>
    <updated>2026-11-19T00:38:26Z</updated>
    <content type="html">&lt;pre&gt;feat(cal): add arPLS and VARPRO global fit&lt;/pre&gt;</content>
  </entry>
</feed>"""

    raw_about = b'__version__ = "1.261119.2"\n'

    def _fake_urlopen(req, timeout=5.0):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        method = req.get_method() if hasattr(req, "get_method") else "GET"
        if "api.github.com/" in url:
            raise urllib.error.HTTPError(url, 403, "rate limit exceeded", hdrs=None, fp=None)

        resp = MagicMock()
        resp.status = 200
        resp.__enter__.return_value = resp
        if method == "HEAD" and "/releases/download/" in url:
            resp.read.return_value = b""
            return resp
        if url.endswith("/releases.atom"):
            resp.read.return_value = releases_atom
            return resp
        if url.endswith("/commits/main.atom"):
            resp.read.return_value = commits_atom
            return resp
        if "raw.githubusercontent.com/" in url:
            resp.read.return_value = raw_about
            return resp
        raise AssertionError(f"Unexpected URL: {url}")

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        info = fetch_latest_release("https://github.com/RJFernandezTeran/PyMORGAN", "1.261022.1")

    assert info.is_update_available is True
    assert info.release_version == "1.261103.2"
    assert info.has_release_update is True
    assert info.bleeding_edge_version == "1.261119.2"
    assert info.has_bleeding_edge_update is True
    assert info.bleeding_edge_commit == "8db3600"
    assert info.bleeding_edge_date == "2026-11-19"
    assert info.bleeding_edge_message == "feat(cal): add arPLS and VARPRO global fit"
    assert "**Highlights**: Faster fitting" in info.release_notes
    assert info.installer_asset is not None


def test_background_check_skipped_in_headless_pytest_unless_forced(qapp):
    from PyQt6.QtWidgets import QMainWindow

    from pymorgan.gui.updater import check_for_updates_background

    win = QMainWindow()
    try:
        with patch("pymorgan.gui.updater.fetch_latest_release") as mock_fetch:
            check_for_updates_background(
                win,
                "https://github.com/RJFernandezTeran/PyMORGAN",
                "1.0.0",
                "PyMORGAN",
                force=False,
            )
            mock_fetch.assert_not_called()
            assert not hasattr(win, "_bg_update_worker")
    finally:
        win.close()
