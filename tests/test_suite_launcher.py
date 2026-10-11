"""Unit tests for Suite Launcher, terminal hiding, and integration bridge."""

import os
from unittest.mock import MagicMock, patch

import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PyQt6.QtWidgets import QApplication

import pymorgan as pm
from pymorgan.gui.console import hide_console, set_windows_app_id
from pymorgan.gui.suite_launcher import SuiteLauncherDialog, get_version, is_pyrate_installed
from pymorgan.integrations.pyrate_bridge import (
    get_pyrate_version,
    launch_pyrate,
)
from pymorgan.integrations.pyrate_bridge import (
    is_pyrate_installed as bridge_is_pyrate_installed,
)


@pytest.fixture(scope="session")
def qapp():
    """Ensure QApplication instance for Qt tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_hide_console_and_set_windows_app_id():
    """Verify console hiding and app ID utilities run safely without error."""
    # Should be non-crashing on any platform
    res = hide_console()
    assert isinstance(res, bool)
    set_windows_app_id("PyMORGAN.Test")


def test_suite_launcher_detection_and_version():
    """Test package detection and version reporting."""
    v_morgan = get_version("pymorgan")
    assert v_morgan not in ("not found", "unknown")
    assert isinstance(is_pyrate_installed(), bool)


def test_suite_launcher_dialog_headless(qapp):
    """Test SuiteLauncherDialog instantiation and widget states."""
    dlg = SuiteLauncherDialog()
    assert "PyMORGAN Suite" in dlg.windowTitle()
    assert dlg.card_pymorgan is not None
    assert dlg.card_pyrate is not None
    assert dlg.chk_close_on_launch.isChecked()


def test_pyrate_bridge_detection():
    """Test PyRATE-TA integration bridge detection."""
    installed = bridge_is_pyrate_installed()
    assert isinstance(installed, bool)
    ver = get_pyrate_version()
    if installed:
        assert ver is not None
    else:
        assert ver is None


@patch("pymorgan.integrations.pyrate_bridge.is_pyrate_installed", return_value=True)
@patch("subprocess.Popen")
def test_pyrate_bridge_launch_with_dataset(mock_popen, mock_installed, tmp_path):
    """Test launching PyRATE-TA with a Dataset1D handoff."""
    mock_popen.return_value = MagicMock()

    # Create dummy 1D dataset
    t = [0.0, 1.0, 2.0]
    p = [500.0, 550.0]
    units = {"unitsT_lbl": "T", "unitsT_ltx": "ps", "unitsL_lbl": "L", "unitsL_ltx": "nm"}
    ds = pm.Dataset1D(Zavg_R=[[1, 2], [3, 4], [5, 6]], delays=t, probe=p, units=units)

    # Launch handoff
    _ = launch_pyrate(dataset_or_path=ds)
    assert mock_popen.called
    args, kwargs = mock_popen.call_args
    cmd = args[0]
    assert "pyrate_ta.gui.app" in cmd
    # Check that a temp handoff file was passed as argument
    assert any("for_kinetic_analysis.pdat" in arg for arg in cmd)
