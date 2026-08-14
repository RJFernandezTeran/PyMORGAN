"""Composite 2-D figure: a t2 series of maps sharing one colorbar.

:func:`pymorgan.twoD.plot_map` accepts an ``ax=`` argument and returns a
:class:`~pymorgan.twoD.Map2DAxes` namedtuple ``(ax, top, cbar)``, so a row of
maps is assembled by handing each cell of a :class:`~matplotlib.gridspec.GridSpec`
to one ``plot_map`` call.

Two things make the colour scale comparable across panels:

* every panel is drawn with the *same* ``vmin``/``vmax``, taken from the largest
  absolute signal over the selected t2 maps (so panel-to-panel intensity
  differences are real, not a per-panel renormalisation);
* every panel is drawn with ``show_colorbar=False``, and a narrow fifth GridSpec
  column holds a single colorbar built from a matching
  :class:`~matplotlib.cm.ScalarMappable`.

The layout is therefore ``width_ratios=[1, 1, 1, 1, 0.12]`` - four square map
panels plus a thin colorbar strip.

Run from anywhere once the package is installed (``uv pip install -e .``)::

    python examples/composite_2D.py

``REPO / "testData" / ...`` is a placeholder; point ``DATA_FILE`` at your own
data and edit ``T2_DELAYS`` to delays your dataset actually contains.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import pymorgan as pm
from pymorgan import helpers as hlp

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO.parent / "testData"

# --------------------------------------------------------------------------- #
#                                  Settings                                    #
# --------------------------------------------------------------------------- #
settings_file = REPO / "settings.toml"
if settings_file.exists():
    pm.load_settings(settings_file)
pm.apply_style()

DATA_FILE = DATA_DIR / "test2DIR.p2dat"

# Requested population times (ps). ``plot_map`` snaps each to the nearest
# available t2, so these need not match a delay exactly.
T2_DELAYS = [0.2, 1.0, 5.0, 20.0]

# Fraction of the global peak used as the colour-scale limit. Below 100 the
# scale saturates, which brings out the weaker cross peaks.
ZSCALE = 50

# Number of filled contour levels. Must match the value handed to ``plot_map``
# so the standalone colorbar is discretised exactly like the panels.
NLEVELS = 30

# --------------------------------------------------------------------------- #
#                                    Data                                      #
# --------------------------------------------------------------------------- #
data = pm.load_2D(DATA_FILE, data_type="P2DAT")
print(data)

# Optional: subtract the last t2 map as a background.
# data.background_correct(reference=-1)

# --------------------------------------------------------------------------- #
#                       Common colour scale over the series                    #
# --------------------------------------------------------------------------- #
# Use the *processed* signal (``data.Z`` returns the corrected array when a
# background correction has been applied, the raw one otherwise) and take the
# peak over just the maps that will be shown.
indices = [data.map_index(t2) for t2 in T2_DELAYS]
peak = float(np.nanmax(np.abs(data.Z[:, :, indices])))
if not np.isfinite(peak) or peak == 0:
    peak = 1.0
VMAX = peak * ZSCALE / 100
VMIN = -VMAX

# Rebuild the colourmap exactly as ``plot_map`` does, so the shared colorbar and
# the panels use identical colours (including the zero-centred white band).
settings = pm.get_settings()
n_levels = NLEVELS + (NLEVELS % 2)
cmap, _ = hlp.CalcCMAP(settings.cmap, n_levels)
if settings.white_levels:
    cmap = hlp.zero_center_cmap(cmap, n_levels, int(settings.white_levels))

# --------------------------------------------------------------------------- #
#                                   Figure                                     #
# --------------------------------------------------------------------------- #
fig = plt.figure(figsize=(16, 4.6))
gs = fig.add_gridspec(
    1,
    len(T2_DELAYS) + 1,
    width_ratios=[1] * len(T2_DELAYS) + [0.12],
    wspace=0.22,
)

axes = []
for col, t2 in enumerate(T2_DELAYS):
    ax = fig.add_subplot(gs[0, col])
    out = data.plot_map(
        t2,
        ax=ax,
        ShowLines=True,
        Nskip=2,
        Nlevels=NLEVELS,
        diagonal=True,
        t2_label=True,
        vmin=VMIN,
        vmax=VMAX,
        show_colorbar=False,  # one shared bar instead of four
        show_ylabel=(col == 0),  # probe label only on the leftmost panel
        aspect="equal",
    )
    if col != 0:
        out.ax.tick_params(axis="y", labelleft=False)
    axes.append(out.ax)

# Repeating the pump label under every panel wastes space and the end ticks of
# adjacent panels collide. Take the label the plotter produced (so it follows
# Settings.freq_label / label_style / twoD_freq_unit and any pump/probe axis
# swap), move it to the figure, and drop the last tick of all but the last
# panel.
xlabel = axes[0].get_xlabel()
for col, ax in enumerate(axes):
    ax.set_xlabel("")
    if col != len(axes) - 1:
        xlim = ax.get_xlim()
        ax.set_xticks(ax.get_xticks()[:-1])
        ax.set_xlim(xlim)

# Narrow fifth column: the single colorbar for the whole row.
cax = fig.add_subplot(gs[0, -1])
mappable = ScalarMappable(norm=Normalize(vmin=VMIN, vmax=VMAX), cmap=cmap)
mappable.set_array([])
cbar = fig.colorbar(mappable, cax=cax)
# ``axis_units()`` reports the dataset's own units; only the signal (Z) unit is
# needed here, since the spectral labels come from the panels themselves.
units = data.axis_units()
label_style = getattr(settings.label_style, "value", str(settings.label_style))
cbar.set_label(units.z.label(label_style))
cbar.ax.tick_params(axis="y", direction="out")

fig.supxlabel(xlabel, fontsize=14, fontweight="bold")
fig.suptitle(
    "2D-IR maps versus population time (common colour scale, %g %% of peak)" % ZSCALE,
    fontsize=15,
)
# ``aspect="equal"`` fixes each panel's shape, so the row is positioned with
# subplots_adjust rather than tight_layout (which would fight the fixed aspect).
fig.subplots_adjust(left=0.075, right=0.92, bottom=0.20, top=0.84, wspace=0.22)
if getattr(fig.canvas, "manager", None) is not None:
    fig.canvas.manager.set_window_title("Composite 2D - t2 series with shared colorbar")

# The colorbar strip inherits the full GridSpec cell height, whereas the map
# panels are shrunk to their fixed aspect ratio. Matching the bar to a panel
# needs the *drawn* geometry, so force a draw before reading the positions.
fig.canvas.draw()
cax.set_position(
    [
        cax.get_position().x0,
        axes[-1].get_position().y0,
        cax.get_position().width,
        axes[-1].get_position().height,
    ]
)

pm.show()
