"""Application entry point for the PyMORGAN GUI."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from ..log import get_logger

logger = get_logger(__name__)

_START_TIME = time.perf_counter()

# Cache for the few settings read before the main window is built.
_STARTUP_SETTINGS: dict | None = None

# Pin matplotlib (and our widgets) to the PyQt6 Qt binding.
os.environ.setdefault("QT_API", "pyqt6")


def settings_path() -> Path:
    """Locate or determine ``settings.toml`` path."""
    from ..settings import settings_path as _get_settings_path

    return _get_settings_path()


def _startup_settings() -> dict:
    """Parse settings.toml with the stdlib TOML reader (cached).

    Only the handful of keys needed *before* the main window exists are read
    this way. Direct parsing keeps startup snappy; the full settings are loaded
    later by ``MainWindow._load_project_settings``.
    """
    global _STARTUP_SETTINGS
    if _STARTUP_SETTINGS is None:
        data: dict = {}
        try:
            from ..settings import ensure_settings_file

            cfg = ensure_settings_file()
            import tomllib

            data = tomllib.loads(cfg.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Could not read settings; using defaults.", exc_info=True)
        _STARTUP_SETTINGS = data
    return _STARTUP_SETTINGS


def _startup_value(key: str, default=None):
    """One start-up key, from the ``[gui]`` section or the file's top level.

    settings.toml is sectioned; the flat lookup is kept so an older file (or one
    a user has flattened) still starts the GUI with the right theme.
    """
    data = _startup_settings()
    section = data.get("gui")
    if isinstance(section, dict) and key in section:
        return section[key]
    return data.get(key, default)


def _read_gui_theme() -> str:
    """Return the raw ``gui_theme`` string from settings.toml ('Fusion' default)."""
    return str(_startup_value("gui_theme") or "Fusion")


def _read_splash_duration() -> float:
    """Return the GUI splash-screen duration in seconds (0 disables it)."""
    try:
        return float(_startup_value("splash_duration", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _resolve_theme(
    style_factory, theme: str | None = None, force_dark: bool = False
) -> tuple[str, bool]:
    """Resolve ``gui_theme`` into a ``(style_name, dark)`` pair.

    A ``"dark"`` token anywhere in the theme string (e.g. ``"Fusion Dark"`` or
    just ``"dark"``), or ``force_dark=True`` (the ``--dark`` command-line flag),
    requests a dark window palette; the token is stripped before the remaining
    name is matched, case-insensitively, against the styles available on this
    platform (``QStyleFactory.keys()``). A dark theme always uses Fusion, the
    only style whose palette can be overridden reliably across platforms.
    Unknown names fall back to Fusion.
    """
    import re

    raw = _read_gui_theme() if theme is None else str(theme)
    dark = force_dark or bool(re.search(r"(?i)\bdark\b", raw))
    base = re.sub(r"(?i)\bdark\b", "", raw).strip() or "Fusion"

    if dark:
        return "Fusion", True

    available = {k.lower(): k for k in style_factory.keys()}
    return available.get(base.lower(), "Fusion"), False


def main(argv: list[str] | None = None) -> int:
    """Create the QApplication, show the main window and run the event loop."""
    t_start = _START_TIME if "_START_TIME" in globals() else time.perf_counter()
    args = list(argv) if argv is not None else list(sys.argv)
    # ``--dark`` forces the dark theme regardless of settings.toml; strip it so
    # Qt does not treat it as an unknown command-line option.
    cli_dark = "--dark" in args
    if cli_dark:
        args = [a for a in args if a != "--dark"]
    # Ignore the OS dark theme on Windows.
    if sys.platform.startswith("win") and "-platform" not in args:
        args += ["-platform", "windows:darkmode=0"]
        # Distinct AppUserModelID so Windows shows our taskbar icon instead of
        # inheriting the host python.exe icon/grouping.
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("PyMORGAN.GUI")
        except Exception:
            logger.debug("Could not set the Windows AppUserModelID.", exc_info=True)

    from PyQt6.QtWidgets import QApplication, QStyleFactory

    app = QApplication.instance() or QApplication(args)
    from PyQt6.QtGui import QIcon

    _icon = Path(__file__).with_name("icons") / "pirate-hat.png"
    if _icon.exists():
        app.setWindowIcon(QIcon(str(_icon)))
    style_name, dark = _resolve_theme(QStyleFactory, force_dark=cli_dark)
    app.setStyle(style_name)
    # For Fusion, force the palette so the OS theme is ignored: a dark palette
    # when a dark theme was requested, otherwise the standard light one.
    if app.style().objectName().lower() == "fusion":
        if dark:
            from .theme import dark_palette

            app.setPalette(dark_palette())
        else:
            fusion = QStyleFactory.create("Fusion")
            if fusion is not None:
                app.setPalette(fusion.standardPalette())

    # Optional start-up splash screen (settings.toml: splash_duration, seconds;
    # 0 disables it). It is created *before* importing the main window, so the
    # user sees it while pymorgan/matplotlib are imported -- by far the slowest
    # part of start-up -- and it is replaced by the window afterwards.
    splash = None
    duration = _read_splash_duration()
    if duration > 0:
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QPixmap
        from PyQt6.QtWidgets import QSplashScreen

        splash_path = Path(__file__).with_name("icons") / "PyMORGAN.jpg"
        if splash_path.exists():
            pix = QPixmap(str(splash_path))
            if not pix.isNull():
                pix = pix.scaled(
                    300,
                    300,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                splash = QSplashScreen(pix)
                font = splash.font()
                font.setPointSize(10)
                font.setBold(True)
                splash.setFont(font)
                from ..__about__ import __version__

                splash.showMessage(
                    f"PyMORGAN - v{__version__}",
                    Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                    Qt.GlobalColor.black,
                )
                splash.show()
                app.processEvents()

    from .main_window import MainWindow

    window = MainWindow()
    if splash is not None:
        from PyQt6.QtCore import QTimer

        def _reveal() -> None:
            splash.finish(window)
            window.show()

        # The splash has already been up during the imports; only wait for the
        # remainder of the requested duration.
        remaining = max(0.0, duration - (time.perf_counter() - t_start))
        QTimer.singleShot(int(remaining * 1000), _reveal)
    else:
        window.show()

    elapsed = time.perf_counter() - t_start
    print(f"[PyMORGAN] GUI loaded in {elapsed:.2f} s")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

