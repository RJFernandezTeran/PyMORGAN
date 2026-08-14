"""Re-entrancy guard for long GUI operations.

Loading a 2-D dataset, fitting the chirp or running a Kubo fit can take many
seconds. Progress is kept alive by calling ``QApplication.processEvents()``,
which re-enters the Qt event loop -- so without a guard the user can click a
second button (or trigger a menu shortcut) *while the first operation is still
running*, and the two runs interleave on the same ``MainWindow`` state.

:func:`busy` closes that door by disabling the tab widget and the menu bar for
the duration of the operation, so no new slot can be triggered from the
re-entered event loop, and by showing a wait cursor. Modal progress dialogs are
top-level windows, so their *Cancel* button keeps working.

Usage -- as a decorator on the slot that owns the operation::

    @busy_guard("Loading 2D dataset...")
    def twoD_load_path(self, path, ...):
        ...

or inline, when only part of a method is slow::

    with busy(self, "Fitting chirp..."):
        ...

Nesting is safe: the widgets are re-enabled and the cursor restored only when
the outermost guard exits, so a guarded method may call another one.
"""

from __future__ import annotations

import functools
import inspect
from contextlib import contextmanager

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from ..log import get_logger

logger = get_logger(__name__)

# Widgets disabled while an operation runs (by objectName on the main window).
_INPUT_WIDGETS = ("MainTabs", "menubar")


@contextmanager
def busy(window, message: str | None = None):
    """Block GUI input on ``window`` while a long operation runs.

    Sets a wait cursor, disables the tab widget and menu bar, and optionally
    shows ``message`` in the status bar (restored on exit unless the operation
    set its own message). Safe without a running ``QApplication`` (headless
    tests) and safe to nest.
    """
    depth = getattr(window, "_busy_depth", 0)
    window._busy_depth = depth + 1
    outermost = depth == 0
    app = QApplication.instance()
    disabled = []

    if outermost:
        if hasattr(window, "setCursor"):
            try:
                window.setCursor(Qt.CursorShape.WaitCursor)
            except Exception:
                pass
        if app is not None:
            for name in _INPUT_WIDGETS:
                widget = getattr(window, name, None)
                if widget is not None and widget.isEnabled():
                    widget.setEnabled(False)
                    disabled.append(widget)
        if message:
            try:
                window.statusBar().showMessage(message)
            except Exception:
                logger.debug("Could not show the busy message in the status bar.", exc_info=True)
    try:
        yield
    finally:
        window._busy_depth = max(0, getattr(window, "_busy_depth", 1) - 1)
        if outermost:
            for widget in disabled:
                widget.setEnabled(True)
            if hasattr(window, "unsetCursor"):
                try:
                    window.unsetCursor()
                except Exception:
                    pass
            if message and app is not None:
                try:
                    bar = window.statusBar()
                    if bar.currentMessage() == message:  # the operation set no message of its own
                        bar.clearMessage()
                except Exception:
                    logger.debug("Could not clear the busy message.", exc_info=True)


def busy_guard(message: str | None = None):
    """Decorator applying :func:`busy` to a ``MainWindow`` method.

    Qt passes extra arguments to a slot when the signal provides them --
    ``clicked`` sends a ``checked`` bool, ``triggered`` likewise. PyQt only
    trims those for callables whose signature is narrow enough, and the wrapper
    below accepts ``*args``, so the extras would reach a slot defined as
    ``def _on_click(self)`` and raise ``TypeError``. They are therefore dropped
    here, exactly as an undecorated slot would have them dropped.
    """

    def decorator(func):
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        takes_varargs = any(p.kind is p.VAR_POSITIONAL for p in params)
        # positional parameters excluding ``self``
        max_args = (
            sum(1 for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)) - 1
        )

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not takes_varargs and len(args) > max_args:
                logger.debug(
                    "Dropping %d extra positional argument(s) passed to %s (Qt signal payload).",
                    len(args) - max_args,
                    func.__name__,
                )
                args = args[:max_args]
            with busy(self, message):
                return func(self, *args, **kwargs)

        wrapper.__signature__ = sig
        return wrapper

    return decorator


def is_busy(window) -> bool:
    """Whether a guarded operation is currently running on ``window``."""
    return bool(getattr(window, "_busy_depth", 0))
