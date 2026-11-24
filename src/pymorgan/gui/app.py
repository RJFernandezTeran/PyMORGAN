"""Application entry point for the PyMORGAN GUI."""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from pathlib import Path

from ..log import get_logger

logger = get_logger(__name__)

_IN_EXCEPTHOOK = False

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


_DIRECT_OPEN_SUFFIXES = (".pdat", ".pdatn", ".p2dat")
_QT_VALUE_FLAGS = frozenset(
    {
        "-platform",
        "-platformpluginpath",
        "-platformtheme",
        "-plugin",
        "-qmljsdebugger",
        "-qwindowgeometry",
        "-qwindowicon",
        "-qwindowtitle",
        "-session",
        "-display",
        "-geometry",
        "-style",
        "-stylesheet",
    }
)


def _extract_cli_targets(args: list[str]) -> list[str]:
    """Extract positional file/directory targets from raw CLI argv."""
    if not args:
        return []
    tokens = args if Path(args[0]).suffix.lower() in _DIRECT_OPEN_SUFFIXES else args[1:]
    targets: list[str] = []
    skip_next = False
    after_sep = False
    for tok in tokens:
        if after_sep:
            if tok:
                targets.append(tok)
            continue
        if skip_next:
            skip_next = False
            continue
        if tok == "--":
            after_sep = True
            continue
        if tok == "--dark":
            continue
        if tok in _QT_VALUE_FLAGS or any(tok.startswith(f"{f}=") for f in _QT_VALUE_FLAGS):
            skip_next = "=" not in tok
            continue
        if tok.startswith("-"):
            continue
        if tok:
            targets.append(tok)
    return targets


def _install_file_open_filter(app, window):
    """Install an application-level filter to open macOS/Qt QFileOpenEvent paths."""
    from PyQt6.QtCore import QEvent, QObject

    class _FileOpenFilter(QObject):
        def __init__(self, target_window):
            super().__init__(target_window)
            self._window = target_window

        def eventFilter(self, obj, event):
            if event is not None and event.type() == QEvent.Type.FileOpen:
                file_path = event.file() if hasattr(event, "file") else ""
                if not file_path and hasattr(event, "url"):
                    url = event.url()
                    if url is not None and url.isLocalFile():
                        file_path = url.toLocalFile()
                if file_path:
                    self._window.open_external_path(file_path)
                    return True
            return False

    filt = _FileOpenFilter(window)
    window._file_open_filter = filt
    app.installEventFilter(filt)
    return filt


def install_gui_excepthook(app_name: str = "PyMORGAN") -> None:
    """Install Qt-safe global exception hooks so ``pythonw`` builds report crashes.

    Unhandled exceptions are logged with full tracebacks and, when a graphical
    Qt session is active (outside headless/pytest runs), surfaced in a critical
    ``QMessageBox`` with expandable traceback details instead of terminating the
    GUI silently.
    """

    def _can_show_dialog() -> bool:
        if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
            return False
        if "PYTEST_CURRENT_TEST" in os.environ:
            return False
        try:
            from PyQt6.QtWidgets import QApplication

            return QApplication.instance() is not None
        except Exception:
            return False

    def _excepthook(exc_type, exc_value, exc_tb):
        global _IN_EXCEPTHOOK
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        if _IN_EXCEPTHOOK:
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return

        _IN_EXCEPTHOOK = True
        try:
            tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            logger.error(
                "Unhandled exception in %s",
                app_name,
                exc_info=(exc_type, exc_value, exc_tb),
            )
            if _can_show_dialog():
                try:
                    from PyQt6.QtWidgets import QApplication, QMessageBox

                    parent = QApplication.activeWindow()
                    box = QMessageBox(parent)
                    box.setIcon(QMessageBox.Icon.Critical)
                    box.setWindowTitle(f"Unexpected Error in {app_name}")
                    box.setText(
                        f"An unexpected error occurred in {app_name}:\n\n"
                        f"{exc_type.__name__}: {exc_value}"
                    )
                    box.setDetailedText(tb_text)
                    box.exec()
                    return
                except Exception:
                    logger.debug("Failed to display GUI error dialog", exc_info=True)
            if sys.stderr is not None:
                sys.stderr.write(tb_text)
                sys.stderr.flush()
        finally:
            _IN_EXCEPTHOOK = False

    def _thread_excepthook(args: threading.ExceptHookArgs):
        if issubclass(args.exc_type, SystemExit):
            return
        logger.error(
            "Unhandled exception in worker thread %s of %s",
            getattr(getattr(args, "thread", None), "name", "<unknown>"),
            app_name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook


def main(argv: list[str] | None = None) -> int:
    """Create the QApplication, show the main window and run the event loop."""
    t_start = _START_TIME if "_START_TIME" in globals() else time.perf_counter()
    raw_args = list(argv) if argv is not None else list(sys.argv)
    cli_targets = _extract_cli_targets(raw_args)
    # ``--dark`` forces the dark theme regardless of settings.toml; strip it so
    # Qt does not treat it as an unknown command-line option.
    cli_dark = "--dark" in raw_args
    args = [a for a in raw_args if a != "--dark"]
    if not args or Path(args[0]).suffix.lower() in _DIRECT_OPEN_SUFFIXES:
        prog_name = sys.argv[0] if sys.argv else "PyMORGAN"
        args = [prog_name, *args]
    # Ignore the OS dark theme on Windows unless a specific QPA platform (e.g. offscreen) is set.
    if (
        sys.platform.startswith("win")
        and "-platform" not in args
        and not os.environ.get("QT_QPA_PLATFORM")
    ):
        args += ["-platform", "windows:darkmode=0"]
        # Distinct AppUserModelID so Windows shows our taskbar icon instead of
        # inheriting the host python.exe icon/grouping.
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("PyMORGAN.GUI")
        except Exception:
            logger.debug("Could not set the Windows AppUserModelID.", exc_info=True)

    # Ensure preferred fonts are installed and registered with Matplotlib early
    try:
        from ..fonts import ensure_fonts_installed

        ensure_fonts_installed(quiet=True)
    except Exception:
        pass

    from PyQt6.QtWidgets import QApplication, QStyleFactory

    app = QApplication.instance() or QApplication(args)
    install_gui_excepthook("PyMORGAN")
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

    startup_target: Path | None = None
    for raw_target in cli_targets:
        candidate = Path(raw_target).expanduser()
        if candidate.exists():
            startup_target = candidate
            break

    from .main_window import MainWindow

    window = MainWindow(init_rootdir=(startup_target is None))
    _install_file_open_filter(app, window)

    if splash is not None:
        from PyQt6.QtCore import QTimer

        if startup_target is not None:
            try:
                window.open_external_path(startup_target)
            except Exception:
                logger.debug("Could not load initial path %s", startup_target, exc_info=True)

        def _reveal() -> None:
            splash.finish(window)
            window.show()

        # The splash has already been up during the imports; only wait for the
        # remainder of the requested duration.
        remaining = max(0.0, duration - (time.perf_counter() - t_start))
        QTimer.singleShot(int(remaining * 1000), _reveal)
    else:
        window.show()
        if startup_target is not None:
            try:
                window.open_external_path(startup_target)
            except Exception:
                logger.debug("Could not load initial path %s", startup_target, exc_info=True)

    elapsed = time.perf_counter() - t_start
    print(f"[PyMORGAN] GUI loaded in {elapsed:.2f} s", flush=True)
    from .console import hide_console

    hide_console()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
