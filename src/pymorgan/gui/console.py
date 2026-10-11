"""Console management utilities for GUI launchers."""

from __future__ import annotations

import sys

from pymorgan.log import get_logger

logger = get_logger(__name__)


def hide_console() -> bool:
    """Hide the host console/terminal window on Windows if one was allocated.

    Returns True if a console window was found and hidden, False otherwise.
    Safe to call on any platform (no-op on Linux / macOS).
    """
    if sys.platform != "win32":
        return False

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        user32 = ctypes.windll.user32

        hwnd = kernel32.GetConsoleWindow()
        if hwnd != 0:
            # SW_HIDE = 0
            user32.ShowWindow(hwnd, 0)
            logger.debug("Host console window (hwnd=%s) hidden.", hwnd)
            return True
    except Exception as exc:
        logger.debug("Failed to hide console window: %s", exc)

    return False


def set_windows_app_id(app_id: str = "PyMORGAN.GUI") -> None:
    """Set the Windows Application User Model ID so the taskbar displays the correct icon."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        except Exception as exc:
            logger.debug("Could not set Windows AppUserModelID: %s", exc)
