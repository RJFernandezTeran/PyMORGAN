"""Unified 1-D pump-probe workflow.

This is the ``test_normal_PumpProbe.py`` barebones script (now archived under
``old_code/``) re-expressed through the unified :mod:`pymorgan` API. Loading,
background correction and the three standard plots are methods on a single
:class:`pymorgan.Dataset1D` object; the data-type is resolved via the loader
registry; and label style, colourmap, time-axis scale and ΔA-unit convention
come from the active :class:`pymorgan.Settings`.

Run from anywhere once the package is installed (``uv pip install -e .``)::

    python examples/pump_probe_1D.py

The ``REPO / "testData" / ...`` paths are placeholders; point ``DATA_FILE`` (and
the optional MESS folder near the bottom) at your own data.
"""

from pathlib import Path

import matplotlib.pyplot as plt

import pymorgan as pm

REPO = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
#                                  Settings                                    #
# --------------------------------------------------------------------------- #
# Load the project settings file (falls back to built-in defaults if absent),
# then apply the matplotlib style profile.
settings_file = REPO / "settings.toml"
if settings_file.exists():
    pm.load_settings(settings_file)
pm.apply_style()

DATA_TYPE = "PDAT"
DATA_FILE = REPO / "testData" / "testTRIR.pdat"

plot_contour = True
plot_spectra = False
plot_kinetics = False

# --------------------------------------------------------------------------- #
#                              Load and process                               #
# --------------------------------------------------------------------------- #
data = pm.load_1D(DATA_FILE, data_type=DATA_TYPE)
print(data)

# Background correction over the pre-zero window (set do_correct=True to apply).
data.background_correct(tmin=-20, tmax=-5, do_correct=False)

# --------------------------------------------------------------------------- #
#                                   Plots                                      #
# --------------------------------------------------------------------------- #
if plot_contour:
    # cmap_ID, Yscale and label_style default to the active settings; only the
    # per-figure choices are passed here.
    ax = data.plot_contour(ShowLines=0, Asinh=0, Zscale=20, Yscale="lin")
    ax.figure.canvas.manager.set_window_title("Contour Plot - Corrected Data")

if plot_spectra:
    delays_to_plot = [0.25, 0.5, 1, 2, 3, 4, 5, 10, 20, 50, 100, 500, 1500]
    # ``x_axis_unit`` and ``secondary_axis`` default to the active settings.
    # Pass them explicitly to convert the probe (X) axis to nm / cm-1 / eV / THz
    # and/or to add a top axis in the complementary unit, e.g.:
    #     data.plot_spectra(delays_to_plot, x_axis_unit="nm", secondary_axis=True)
    data.plot_spectra(delays_to_plot, doSmooth=1, normY=0, roundT=True)

if plot_kinetics:
    wavelengths_to_plot = [2132, 2218]
    _, t, Y = data.plot_kinetics(wavelengths_to_plot, plotStyle="-", lw=1.5)

# --------------------------------------------------------------------------- #
#                     Optional: MESS directory datasets                       #
# --------------------------------------------------------------------------- #
# MESS_TRIR (cm-1) and MESS_TRUVIS (nm) are *directory* datasets: a folder whose
# files are prefixed with the folder name. The same Dataset1D API applies; the
# format-specific extras are shown below. Set ``demo_mess = True`` and point
# ``MESS_FOLDER`` at a dataset folder to try it.
demo_mess = False
MESS_TYPE = "MESS_TRIR"
MESS_FOLDER = REPO / "testData" / "MESS_TRIR_example"

if demo_mess:
    from pymorgan.oneD.load import (
        describe_dataset,
        mess_calibration_status,
        mess_recalc_average,
    )

    # describe_dataset reports the selectable states (spectrometer windows `sp`
    # and slow-modulation polarisations `sm`) and the available anisotropy modes.
    print(describe_dataset(MESS_TYPE, MESS_FOLDER))

    # Pick a spectrometer window (sp) / slow-mod state (sm); anisotropy="NONE"
    # reads that state directly, other modes combine polarisation states. A
    # CalibratedProbe.csv (dataset folder, else parent) overrides the probe axis.
    mess = pm.load_1D(MESS_FOLDER, data_type=MESS_TYPE, spectrum=0, slowmod=0, anisotropy="NONE")
    print(mess, "|", mess_calibration_status(MESS_FOLDER))
    mess.plot_contour(Zscale=20)

    # Per-scan exploration needs the temp/ single-scan arrays (off by default).
    pm.update_settings(load_single_scans=True)
    mess = pm.load_1D(MESS_FOLDER, data_type=MESS_TYPE)
    if mess.has_single_scans:
        # One trace per scan, or per group of `binsize` scans (averaged).
        mess.plot_scan_kinetics([mess.probe.mean()], binsize=2)
        mess.plot_scan_spectra([1.0], binsize=2)

    # Noise-weighted recombination of a subset of scans (1-based selection);
    # combining every scan reproduces the on-disk averaged data.
    n_scans = int(mess.nscans)
    Zavg, Zstdv = mess_recalc_average(MESS_FOLDER, list(range(n_scans)), spectrum=0, slowmod=0)
    print("Recalculated noise-weighted average over", n_scans, "scans:", Zavg.shape)

plt.tight_layout()
plt.show()
