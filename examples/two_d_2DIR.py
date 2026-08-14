"""Unified 2-D (2D-IR) workflow: load -> process -> plot.

Run from anywhere once the package is installed (``uv pip install -e .``)::

    python examples/two_d_2DIR.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

import pymorgan as pm

REPO = Path(__file__).resolve().parent.parent

settings_file = REPO / "settings.toml"
if settings_file.exists():
    pm.load_settings(settings_file)
pm.apply_style()

DATA_FILE = REPO / "testData" / "test2DIR.p2dat"

# load -> process
data = pm.load_2D(DATA_FILE, data_type="P2DAT")
print(data)
# subtract the last t2 map as background (optional); omit for a passthrough
# data.background_correct(reference=-1)

# plot a single t2 map; pass top_spectrum=<Spectrum or {"X","Y"}> for an overlay
out = data.plot_map(0.5, ShowLines=True)
out.ax.figure.canvas.manager.set_window_title("2D-IR map")

plt.show()
