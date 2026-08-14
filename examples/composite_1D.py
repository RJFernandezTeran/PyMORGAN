"""Composite (multi-panel) 1-D figures built from the PyMORGAN plotters.

Every 1-D plotter accepts an ``ax=`` argument, so a publication-style figure is
assembled by laying out a :class:`~matplotlib.gridspec.GridSpec` yourself and
handing each cell to the appropriate ``Dataset1D.plot_*`` method. The plotters
draw into the axis and never touch the figure geometry when an axis is supplied
(``_finalize_layout`` is a no-op in that case), so the layout below is the one
that survives.

Two figures are produced:

* **Figure 1** - three panels: contour map, kinetic traces, spectral traces.
* **Figure 2** - 2x2 panels: two datasets side by side, contour on top and the
  matching spectral traces below, with the probe (X) axis shared per column.

Run from anywhere once the package is installed (``uv pip install -e .``)::

    python examples/composite_1D.py

``REPO / "testData" / ...`` are placeholders; point ``NILERED_FILE`` and
``TRIR_FILE`` at your own data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pymorgan as pm

# Geometry of the divider colorbar that ``plot_contour`` appends (inches).
# Mirrors ``pymorgan.oneD.plot._CBAR_WIDTH_IN`` / ``_CBAR_PAD_IN``.
CBAR_WIDTH_IN = 0.15
CBAR_PAD_IN = 0.1

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO.parent / "testData"

# --------------------------------------------------------------------------- #
#                                  Settings                                    #
# --------------------------------------------------------------------------- #
settings_file = REPO / "settings.toml"
if settings_file.exists():
    pm.load_settings(settings_file)
pm.apply_style()

NILERED_FILE = DATA_DIR / "test_NileRed.pdat"
TRIR_FILE = DATA_DIR / "testTRIR.pdat"

# --------------------------------------------------------------------------- #
#                                    Data                                      #
# --------------------------------------------------------------------------- #
nilered = pm.load_1D(NILERED_FILE, data_type="PDAT")
trir = pm.load_1D(TRIR_FILE, data_type="PDAT")

# Background correction over the pre-zero window (do_correct=False stores the
# background without subtracting it; flip to True to apply).
nilered.background_correct(tmin=-2, tmax=-0.5, do_correct=False)
trir.background_correct(tmin=-20, tmax=-5, do_correct=False)

# Cuts to display. Delays are in the dataset's own time unit (ps for these
# files); probe positions are in the native probe unit (nm for NileRed,
# cm-1 for the TRIR set).
NILERED_DELAYS = [0.2, 0.5, 1, 2, 5, 10, 50, 200, 1000]
NILERED_PROBES = [480, 540, 610, 660]
TRIR_DELAYS = [0.25, 0.5, 1, 2, 5, 10, 50, 200, 1500]
TRIR_PROBES = [2132, 2218]


def _match_colorbar_width(ax):
    """Reserve, on ``ax``, the width that a contour panel gives its colorbar.

    ``plot_contour`` appends its colorbar with ``make_axes_locatable``, i.e. a
    fixed physical width taken out of the host axis. A panel drawn without a
    colorbar therefore ends up wider than the contour above it. Appending an
    invisible axis of identical geometry restores column alignment, which is
    what makes a shared X axis meaningful across the two rows.
    """
    spacer = make_axes_locatable(ax).append_axes("right", size=CBAR_WIDTH_IN, pad=CBAR_PAD_IN)
    spacer.set_axis_off()
    return spacer


def _inset_legend(ax, loc="upper right", fontsize=9, ncol=1):
    """Move the plotter's legend inside ``ax``.

    ``plot_spectra`` / ``plot_kinetics`` place a draggable legend *outside* the
    axis (``bbox_to_anchor=(1, 0.5)``), which is right for a single-panel figure
    but overlaps the neighbouring panel in a composite. Re-drawing the same
    handles with an inside ``loc`` replaces that legend and keeps every panel
    within its GridSpec cell.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    return ax.legend(
        handles,
        labels,
        loc=loc,
        ncol=ncol,
        fontsize=fontsize,
        handlelength=0.9,
        labelspacing=0.35,
        frameon=False,
    )


def _panel_tag(ax, text, *, dx=0.02, dy=0.97):
    """Put a bold ``(a)``-style tag in the upper-left corner of ``ax``."""
    ax.text(
        dx,
        dy,
        text,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
        zorder=10,
    )


# --------------------------------------------------------------------------- #
#           Figure 1 - contour + kinetics + spectra for one dataset           #
# --------------------------------------------------------------------------- #
# The contour panel is given the full top row; the two trace panels share the
# bottom row. Only the contour carries a colorbar, so its cell is not widened:
# the divider colorbar is appended *inside* the cell and tight_layout below
# accounts for it.
fig1 = plt.figure(figsize=(11, 8))
gs1 = fig1.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.32, wspace=0.28)

ax_contour = fig1.add_subplot(gs1[0, :])
ax_kin = fig1.add_subplot(gs1[1, 0])
ax_spec = fig1.add_subplot(gs1[1, 1])

# Contour: symlog delay axis, contour lines on, colour scale clipped to 30 % of
# the peak so the weaker long-time features stay visible.
trir.plot_contour(
    ax=ax_contour,
    ShowLines=True,
    Nskip=3,
    Zscale=30,
    Yscale="symlog",
)

# Kinetics: same probe positions that the contour panel spans.
trir.plot_kinetics(TRIR_PROBES, ax=ax_kin, plotStyle="-", lw=1.5)
_inset_legend(ax_kin)

# Spectra: light smoothing, rounded delay labels in the legend.
trir.plot_spectra(TRIR_DELAYS, ax=ax_spec, doSmooth=3, roundT=True)
_inset_legend(ax_spec, ncol=2)

for ax, tag in ((ax_contour, "(a)"), (ax_kin, "(b)"), (ax_spec, "(c)")):
    _panel_tag(ax, tag)

fig1.suptitle("TRIR - contour, kinetics and transient spectra", fontsize=15)
# tight_layout cannot see the divider colorbar (it warns about "Axes that are
# not compatible"), so it does not reserve room for the bar's tick labels and
# unit label. The ``rect`` keeps the panels clear of the suptitle; the explicit
# right margin afterwards is what actually makes room for the colorbar text.
fig1.tight_layout(rect=(0.0, 0.0, 0.93, 0.96))
fig1.subplots_adjust(right=0.88)
if getattr(fig1.canvas, "manager", None) is not None:
    fig1.canvas.manager.set_window_title("Composite 1D - three panels")


# --------------------------------------------------------------------------- #
#      Figure 2 - 2x2: two datasets, contour over spectra, shared X/column     #
# --------------------------------------------------------------------------- #
# ``sharex`` is set per column so each contour and the spectra below it use the
# same probe axis. The contour panels therefore hide their own X label and tick
# labels; the spectra panels underneath carry them. Because the columns hold
# different datasets (nm vs cm-1) the two columns are *not* shared with each
# other.
#
# ``wspace`` is generous because each contour panel carries its own colorbar,
# whose tick labels and unit label overhang the cell to the right.
fig2 = plt.figure(figsize=(13, 9))
gs2 = fig2.add_gridspec(2, 2, height_ratios=[1.0, 0.8], hspace=0.08, wspace=0.55)

datasets = (
    (nilered, NILERED_DELAYS, "Nile Red (TA)"),
    (trir, TRIR_DELAYS, "TRIR"),
)

for col, (dataset, delays, title) in enumerate(datasets):
    ax_top = fig2.add_subplot(gs2[0, col])
    ax_bot = fig2.add_subplot(gs2[1, col], sharex=ax_top)

    dataset.plot_contour(
        ax=ax_top,
        ShowLines=False,
        Zscale=30,
        Yscale="symlog",
        show_xlabel=False,  # the spectra panel below owns the probe label
    )
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_title(title, fontsize=14, pad=8)

    dataset.plot_spectra(delays, ax=ax_bot, doSmooth=3, roundT=True)
    _inset_legend(ax_bot, ncol=2)
    _match_colorbar_width(ax_bot)

    _panel_tag(ax_top, "(%s)" % "ac"[col])
    _panel_tag(ax_bot, "(%s)" % "bd"[col])

    # plot_spectra sets its own tight X limits; re-assert them on the shared
    # pair so the contour above lines up exactly with the spectra below.
    ax_top.set_xlim(ax_bot.get_xlim())

fig2.suptitle("Contour maps and transient spectra, shared probe axis per column", fontsize=15)
fig2.tight_layout(rect=(0.0, 0.0, 0.95, 0.96))
fig2.subplots_adjust(right=0.90)  # room for the right column's colorbar labels
if getattr(fig2.canvas, "manager", None) is not None:
    fig2.canvas.manager.set_window_title("Composite 1D - 2x2 two datasets")


pm.show()
