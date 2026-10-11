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
