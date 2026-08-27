"""Thin convenience wrappers around matplotlib's figure display.

Kept tiny and import-light: matplotlib is only imported when a function is
actually called.
"""

from __future__ import annotations


from collections.abc import Sequence

import matplotlib.axes
import matplotlib.text
import numpy as np


def show_plots(*, block: bool = True) -> None:
    """Display all open matplotlib figures.

    Equivalent to ``matplotlib.pyplot.show()``. With ``block=False`` the call
    returns immediately (useful inside an event loop / GUI), otherwise it blocks
    until the figure windows are closed.
    """
    import matplotlib.pyplot as plt

    plt.show(block=block)


def close_plots() -> None:
    """Close all open matplotlib figures (``plt.close('all')``)."""
    import matplotlib.pyplot as plt

    plt.close("all")


def add_subplot_labels(
    axes: Sequence[matplotlib.axes.Axes] | np.ndarray,
    *,
    fmt: str = "(a)",
    position: str = "top-left",
    inside: bool = True,
    pad: float | tuple[float, float] = 0.05,
    size: float | str | None = None,
    order: str = "row-first",
    bbox: dict | None = None,
    **kwargs,
) -> list[matplotlib.text.Text]:
    """Automatically add subplot labels like (a), (b), (c) to a sequence or grid of matplotlib axes.

    Parameters
    ----------
    axes : sequence of matplotlib.axes.Axes or ndarray of Axes
        Target axes to label. Can be a 1D sequence or a multidimensional array
        returned by ``plt.subplots(...)``.
    fmt : str, default "(a)"
        Label format template. Recognized format tokens:
        
        - ``"a"`` : lowercase letters ``(a)``, ``(b)``, ... ``(aa)``
        - ``"A"`` : uppercase letters ``(A)``, ``(B)``, ...
        - ``"1"`` : numbers ``1)``, ``2)``, ...
        - ``"I"`` : uppercase Roman numerals ``(I)``, ``(II)``, ...
        - ``"i"`` : lowercase Roman numerals ``(i)``, ``(ii)``, ...
        
        Delimiters in ``fmt`` are preserved (e.g. ``"a)"``, ``"A."``, ``"[1]"``).
    position : str, default "top-left"
        Position of the label relative to the axes boundary. Allowed values:
        ``"top-left"`` (or ``"tl"``), ``"top-right"`` (``"tr"``), ``"bottom-left"`` (``"bl"``),
        ``"bottom-right"`` (``"br"``), ``"top-centre"`` (``"tc"``), ``"bottom-centre"`` (``"bc"``).
    inside : bool, default True
        If True, place the label inside the axes frame using inward padding.
        If False, place the label outside the axes frame.
    pad : float or tuple of float, default 0.05
        Padding distance from axes edges in fractional axes units (0.0 to 1.0).
        Can be a single float or a ``(x_pad, y_pad)`` tuple.
    size : float or str, optional
        Font size of the labels. Defaults to ``plt.rcParams["axes.labelsize"]``
        (matching the current plot's axis label size setting).
    order : str, default "row-first"
        Order to traverse 2D arrays of axes:
        
        - ``"row-first"`` / ``"C"`` : row by row (top to bottom, left to right)
        - ``"column-first"`` / ``"F"`` : column by column (top to bottom, left to right)
    bbox : dict, optional
        Dictionary of bounding box properties passed to ``ax.text``
        (e.g., ``dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none")``).
    **kwargs
        Additional keyword arguments forwarded to ``matplotlib.axes.Axes.text``
        (e.g., ``fontweight="bold"``, ``color="black"``, ``zorder=10``).

    Returns
    -------
    list of matplotlib.text.Text
        List of created ``Text`` instances attached to each axis.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import pymorgan as pm
    >>> fig, axes = plt.subplots(2, 2)
    >>> pm.add_subplot_labels(axes, fmt="(a)", position="top-left")
    >>> pm.add_subplot_labels(axes, fmt="A)", position="top-right", inside=False)
    """
    import matplotlib.pyplot as plt

    # Flatten / sequence axes array according to order
    if isinstance(axes, np.ndarray):
        if order in ("column-first", "col-first", "F"):
            flat_axes = axes.flatten(order="F").tolist()
        else:
            flat_axes = axes.flatten(order="C").tolist()
    else:
        flat_axes = list(axes)

    # Determine size default
    if size is None:
        size = plt.rcParams.get("axes.labelsize", "medium")
        try:
            from matplotlib.font_manager import FontProperties
            size = FontProperties(size=size).get_size_in_points()
        except Exception:
            pass

    # Helper for formatting label index
    def _format_label(template: str, idx: int) -> str:
        if "a" in template:
            chars = []
            n = idx
            while True:
                chars.append(chr(ord("a") + (n % 26)))
                n = n // 26 - 1
                if n < 0:
                    break
            s = "".join(reversed(chars))
            return template.replace("a", s)
        elif "A" in template:
            chars = []
            n = idx
            while True:
                chars.append(chr(ord("A") + (n % 26)))
                n = n // 26 - 1
                if n < 0:
                    break
            s = "".join(reversed(chars))
            return template.replace("A", s)
        elif "1" in template:
            return template.replace("1", str(idx + 1))
        elif "I" in template:
            def _to_roman(num):
                val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
                syb = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
                roman_num = ""
                i = 0
                while num > 0:
                    for _ in range(num // val[i]):
                        roman_num += syb[i]
                        num -= val[i]
                    i += 1
                return roman_num
            return template.replace("I", _to_roman(idx + 1))
        elif "i" in template:
            def _to_roman_lower(num):
                val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
                syb = ["m", "cm", "d", "cd", "c", "xc", "l", "xl", "x", "ix", "v", "iv", "i"]
                roman_num = ""
                i = 0
                while num > 0:
                    for _ in range(num // val[i]):
                        roman_num += syb[i]
                        num -= val[i]
                    i += 1
                return roman_num
            return template.replace("i", _to_roman_lower(idx + 1))
        else:
            return f"{template}{idx + 1}"

    # Process pad (x_pad, y_pad)
    if isinstance(pad, (tuple, list)):
        x_pad, y_pad = float(pad[0]), float(pad[1])
    else:
        x_pad = y_pad = float(pad)

    # Position mapping to (x, y, ha, va) in axes coordinates transform
    pos_clean = position.lower().replace("_", "-")
    pos_map = {
        "top-left": ("top-left", "tl"),
        "top-right": ("top-right", "tr"),
        "bottom-left": ("bottom-left", "bl"),
        "bottom-right": ("bottom-right", "br"),
        "top-centre": ("top-centre", "top-center", "tc"),
        "bottom-centre": ("bottom-centre", "bottom-center", "bc"),
    }
    # Resolve aliases
    resolved_pos = "top-left"
    for key, aliases in pos_map.items():
        if pos_clean in aliases or pos_clean == key:
            resolved_pos = key
            break

    # Calculate coordinates and alignment
    if inside:
        match resolved_pos:
            case "top-left":
                x, y = x_pad, 1.0 - y_pad
                ha, va = "left", "top"
            case "top-right":
                x, y = 1.0 - x_pad, 1.0 - y_pad
                ha, va = "right", "top"
            case "bottom-left":
                x, y = x_pad, y_pad
                ha, va = "left", "bottom"
            case "bottom-right":
                x, y = 1.0 - x_pad, y_pad
                ha, va = "right", "bottom"
            case "top-centre":
                x, y = 0.5, 1.0 - y_pad
                ha, va = "center", "top"
            case "bottom-centre":
                x, y = 0.5, y_pad
                ha, va = "center", "bottom"
    else:  # outside
        match resolved_pos:
            case "top-left":
                x, y = -x_pad, 1.0 + y_pad
                ha, va = "right", "bottom"
            case "top-right":
                x, y = 1.0 + x_pad, 1.0 + y_pad
                ha, va = "left", "bottom"
            case "bottom-left":
                x, y = -x_pad, -y_pad
                ha, va = "right", "top"
            case "bottom-right":
                x, y = 1.0 + x_pad, -y_pad
                ha, va = "left", "top"
            case "top-centre":
                x, y = 0.5, 1.0 + y_pad
                ha, va = "center", "bottom"
            case "bottom-centre":
                x, y = 0.5, -y_pad
                ha, va = "center", "top"

    text_objs = []
    for idx, ax in enumerate(flat_axes):
        label_str = _format_label(fmt, idx)
        txt = ax.text(
            x,
            y,
            label_str,
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=size,
            bbox=bbox,
            **kwargs,
        )
        text_objs.append(txt)

    return text_objs


# Short alias.
show = show_plots


