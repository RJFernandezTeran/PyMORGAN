"""Steady-state spectra: standalone, as a series, and as a transient overlay.

Run from anywhere once the package is installed (``uv pip install -e .``)::

    python examples/steady_state.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

import pymorgan as pm
from pymorgan import SpectrumSeries

REPO = Path(__file__).resolve().parent.parent
CAL = REPO / "testData" / "Cal"

pm.apply_style()

# --- A single absorption spectrum (delimiter and units auto-detected) ------- #
nir = pm.load_spectrum(CAL / "NIR-Holmium.csv", kind="absorption", label="NIR")
nir.plot()

# --- A normalised overlay of several spectra -------------------------------- #
series = SpectrumSeries.from_files(
    [CAL / "NIR-Holmium.csv", CAL / "UVVis-Holmium.csv"],
    kind="absorption",
    labels=["NIR", "UV-Vis"],
)
series.plot(normalise=True)

# --- Steady-state absorption overlaid on transient spectra ------------------ #
data = pm.load_1D(REPO / "testData" / "testTRIR.pdat").background_correct(-20, -5)
ftir = pm.load_spectrum(CAL / "FTIR-Dioxane_2cm-1.csv", kind="absorption")
data.plot_spectra([1, 5, 20], doSmooth=1, Abs=ftir.as_overlay_dict())

plt.show()
