"""Logging for PyMORGAN.

Every module obtains its logger with :func:`get_logger`, so all records travel
under the single ``pymorgan`` logger and can be silenced, redirected or raised
in verbosity by the host application::

    import logging
    logging.getLogger("pymorgan").setLevel(logging.DEBUG)   # full detail
    logging.getLogger("pymorgan").setLevel(logging.WARNING) # quiet

PyMORGAN is used both as a library and as an application (GUI, console
scripts), so :func:`configure_logging` attaches a plain console handler on
import *only* when nothing else has configured logging. A host that sets up its
own handlers -- on the root logger or on ``pymorgan`` -- keeps full control and
nothing is added.

Two conventions are used throughout the code base:

* ``logger.info`` for messages the user is meant to see (method citations, fit
  summaries) -- these replaced bare ``print`` calls;
* ``logger.debug(..., exc_info=True)`` inside ``except`` blocks whose failure is
  genuinely recoverable, so the traceback is retrievable at DEBUG level instead
  of being discarded.
"""

from __future__ import annotations

import logging

LOGGER_NAME = "pymorgan"

# Bare message: these records are read by scientists at a console, not parsed.
_FORMAT = "%(message)s"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the ``pymorgan`` logger, or a child of it.

    ``get_logger(__name__)`` from inside the package yields the dotted module
    logger (``pymorgan.twoD.load``); any other name is attached as a child.
    """
    if not name or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    if name.startswith(LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_logging(level: int | str = logging.INFO, *, force: bool = False) -> logging.Logger:
    """Give the ``pymorgan`` logger a console handler, unless one is set up.

    Called once on import so that informational output keeps reaching the
    console. If the root logger or the ``pymorgan`` logger already has handlers
    (i.e. the host application configured logging) nothing is changed, unless
    ``force=True``.
    """
    logger = logging.getLogger(LOGGER_NAME)
    configured = bool(logger.handlers) or bool(logging.getLogger().handlers)
    if configured and not force:
        return logger
    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
