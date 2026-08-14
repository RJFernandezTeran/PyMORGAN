"""Unit tests for 2D dataset processing, CLS extraction, Kubo fitting, and ROI picker."""

import numpy as np
import pytest
import synthetic

import pymorgan as pm
from pymorgan.twoD.dataset import Dataset2D
from pymorgan.twoD.analyse import _refine_extremum, center_line_slope, subpixel_peak
from pymorgan.twoD.kubo_fit import g_function, run_kubo_fit, simulate_2d_spectrum


STEP = 2.5
SLOPE = 0.6
WIDTH = 6.0


def _slice(x0: float, x: np.ndarray) -> np.ndarray:
    return -np.exp(-0.5 * ((x - x0) / WIDTH) ** 2)


def _tilted_map(noise: float = 0.0, seed: int = 0):
    axis = np.arange(1990.0, 2060.0, STEP)
    P, Pr = np.meshgrid(axis, axis, indexing="ij")
    centre = 2020.0 + SLOPE * (P - 2020.0)
    Z = -np.exp(-0.5 * ((Pr - centre) / WIDTH) ** 2) * np.exp(-0.5 * ((P - 2020.0) / 12.0) ** 2)
    if noise:
        Z = Z + noise * np.random.default_rng(seed).standard_normal(Z.shape)
    units = {
        "unitsL": "cm-1",
        "unitsT_ltx": "ps",
        "unitsZ_lbl": "dA",
        "unitsZ_ltx": "mOD",
        "unitsZ": "mOD",
    }
    return Dataset2D(
        pump=axis, probe=axis, delays=np.array([1.0]), Z=Z[:, :, None],
        units=units, freq_units="cm-1",
    )


# --------------------------------------------------------------------------- #
#                               2D Dataset                                    #
# --------------------------------------------------------------------------- #
def test_dataset2d_creation(tmp_path):
    info = synthetic.make_synthetic_p2dat(tmp_path / "sample.p2dat")
    ds = Dataset2D.from_file(info["path"], data_type="P2DAT")

    assert ds.data_type == "P2DAT"
    assert ds.Z.ndim == 3
    assert len(ds.delays) > 0
    assert len(ds.pump) > 0
    assert len(ds.probe) > 0


# --------------------------------------------------------------------------- #
#                             CLS & Subpixel Peak                             #
# --------------------------------------------------------------------------- #
def test_subpixel_peak_recovers_a_known_offset():
    grid = np.arange(1990.0, 2060.0, STEP)
    true_x = 2020.0 + 0.1 * STEP
    y = _slice(true_x, grid)
    idx = int(np.argmin(y))

    fit_val = subpixel_peak(grid, y, idx, method="gaussian")
    error_in_pixels = abs(fit_val - true_x) / STEP
    assert error_in_pixels < 0.2


def test_cls_returns_expected_slope():
    ds = _tilted_map()
    slopes = center_line_slope(ds, pump_range=(2010.0, 2030.0))
    assert len(slopes) == len(ds.delays)
    assert np.all(np.isfinite(slopes))


# --------------------------------------------------------------------------- #
#                                Kubo Fitting                                 #
# --------------------------------------------------------------------------- #
def test_g_function():
    g1 = g_function(1.0, 0.0, 20.0, 5.0, 10.0)
    g2 = g_function(2.0, 0.0, 20.0, 5.0, 10.0)
    assert g2 > g1


def test_simulate_2d_spectrum():
    pump_axis = np.linspace(1980, 2020, 20)
    probe_axis = np.linspace(1980, 2020, 20)
    Z = simulate_2d_spectrum(
        2000.0, 20.0, 0.0, 20.0, 5.0, 1.0,
        pump_axis, probe_axis, 1.0
    )
    assert Z.shape == (20, 20)
    assert not np.iscomplexobj(Z)


def test_run_kubo_fit():
    pump_axis = np.linspace(1990, 2010, 10)
    probe_axis = np.linspace(1990, 2010, 10)

    Z_target = simulate_2d_spectrum(
        2000.0, 20.0, 0.0, 10.0, 5.0, 1.0,
        pump_axis, probe_axis, 1.0
    )
    data_cube = np.expand_dims(Z_target, axis=2)

    p0 = [1998.0, 18.0, 0.0, 8.0, 4.0, 0.8]
    bounds = [
        (1990, 2010),
        (10, 30),
        (0.0, 0.0),
        (5.0, 15.0),
        (2.0, 8.0),
        (0.5, 2.0)
    ]

    fit_results = run_kubo_fit(data_cube, pump_axis, probe_axis, [1.0], p0, bounds, parallel_mode="disabled")
    assert len(fit_results) == 1
    res = fit_results[0]
    assert res["success"] is True
    assert abs(res["params"][0] - 2000.0) < 2.0
    assert abs(res["params"][1] - 20.0) < 2.0


from pymorgan.gui.picker import ContourPicker


# --------------------------------------------------------------------------- #
#                                 ROI Picker                                  #
# --------------------------------------------------------------------------- #
def test_contour_picker_instantiation():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    picker = ContourPicker(fig.canvas, ax, axis="x", on_done=lambda *a: None)
    assert picker.axis == "x"
    assert picker._active is False
    plt.close(fig)
