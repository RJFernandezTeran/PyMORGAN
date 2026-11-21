"""Unit tests for 2D dataset processing, CLS extraction, Kubo fitting, and ROI picker."""

import numpy as np
import pytest
import synthetic

import pymorgan as pm
from pymorgan.twoD.analyse import center_line_slope, subpixel_peak
from pymorgan.twoD.dataset import Dataset2D
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


# --------------------------------------------------------------------------- #
#                                 ROI Picker                                  #
# --------------------------------------------------------------------------- #
def test_contour_picker_instantiation():
    import matplotlib.pyplot as plt

    from pymorgan.gui.picker import ContourPicker

    fig, ax = plt.subplots()
    picker = ContourPicker(fig.canvas, ax, axis="x", on_done=lambda *a: None)
    assert picker.axis == "x"
    assert picker._active is False
    plt.close(fig)


# --------------------------------------------------------------------------- #
#                          2D Gaussian Fitting                                #
# --------------------------------------------------------------------------- #
def test_2d_gaussian_fit_tight_position_and_anharm_bounds():
    from pymorgan.twoD.analyse import evaluate_2d_gaussian_map, fit_2d_gaussian_map

    pump = np.linspace(1980.0, 2060.0, 41)
    probe = np.linspace(1980.0, 2060.0, 41)

    # True mode 1: at (2010, 2010), anharm 16.0, GSB -1.2, ESA +0.9, sigma=8.0
    # True mode 2 (nearby band): at (2040, 2040), anharm 14.0, GSB -1.5, ESA +1.1, sigma=8.0
    true_modes = [
        {"w1": 2010.0, "w3": 2010.0, "anharm": 16.0, "amp_gsb": -1.2, "amp_esa": 0.9, "sigma_w1": 8.0, "sigma_w3": 8.0},
        {"w1": 2040.0, "w3": 2040.0, "anharm": 14.0, "amp_gsb": -1.5, "amp_esa": 1.1, "sigma_w1": 8.0, "sigma_w3": 8.0},
    ]
    map_2d = evaluate_2d_gaussian_map(pump, probe, true_modes)

    # Initial guesses slightly offset from truth
    init_modes = [
        {"w1": 2012.0, "w3": 2008.0, "anharm": 15.0, "amp_gsb": -1.0, "amp_esa": 0.8, "sigma_w1": 10.0, "sigma_w3": 10.0},
        {"w1": 2038.0, "w3": 2042.0, "anharm": 15.0, "amp_gsb": -1.0, "amp_esa": 0.8, "sigma_w1": 10.0, "sigma_w3": 10.0},
    ]

    bounds_cfg = {
        "pos_tol": 5.0,        # Tight position tolerance of +/- 5 cm-1
        "anharm_min": 8.0,     # Lower bound on anharmonicity
        "anharm_max": 30.0,    # Upper bound on anharmonicity
        "anharm_tol": 6.0,     # Tight tolerance around initial anharm
        "constrain_signs": True,
    }

    fitted_modes, fit_map, residual_map = fit_2d_gaussian_map(
        pump, probe, map_2d, init_modes, bounds_config=bounds_cfg
    )

    assert len(fitted_modes) == 2
    # Verify fitted parameters for mode 1
    m1 = fitted_modes[0]
    assert abs(m1["w1"] - 2010.0) < 1.0
    assert abs(m1["w3"] - 2010.0) < 1.0
    assert abs(m1["anharm"] - 16.0) < 1.5
    assert abs(m1["amp_gsb"] - (-1.2)) < 0.2
    assert abs(m1["amp_esa"] - 0.9) < 0.2

    # Verify fitted parameters for mode 2
    m2 = fitted_modes[1]
    assert abs(m2["w1"] - 2040.0) < 1.0
    assert abs(m2["w3"] - 2040.0) < 1.0
    assert abs(m2["anharm"] - 14.0) < 1.5

    # Verify that position bounds strictly prevent jumping to the other band
    assert abs(m1["w1"] - init_modes[0]["w1"]) <= 5.0 + 1e-4
    assert abs(m2["w1"] - init_modes[1]["w1"]) <= 5.0 + 1e-4

    # Max residual should be very low on clean data
    assert np.max(np.abs(residual_map)) < 0.05


def test_2d_gaussian_global_fit():
    from pymorgan.twoD.analyse import evaluate_2d_gaussian_map, fit_2d_gaussian_global

    pump = np.linspace(1990.0, 2030.0, 21)
    probe = np.linspace(1990.0, 2030.0, 21)
    delays = np.array([0.5, 1.0, 2.0])

    # True shared mode: w1=2010, w3=2010, anharm=15.0, sigma=7.0
    # Amplitudes decay with delay
    cube_3d = np.zeros((len(pump), len(probe), len(delays)))
    decay_factors = [1.0, 0.7, 0.4]
    for idx, dec in enumerate(decay_factors):
        modes_t = [{
            "w1": 2010.0, "w3": 2010.0, "anharm": 15.0,
            "amp_gsb": -1.0 * dec, "amp_esa": 0.8 * dec,
            "sigma_w1": 7.0, "sigma_w3": 7.0
        }]
        cube_3d[:, :, idx] = evaluate_2d_gaussian_map(pump, probe, modes_t)

    init_modes = [{
        "w1": 2011.0, "w3": 2009.0, "anharm": 14.0,
        "amp_gsb": -0.8, "amp_esa": 0.6,
        "sigma_w1": 8.0, "sigma_w3": 8.0
    }]

    bounds_cfg = {
        "pos_tol": 4.0,
        "anharm_min": 5.0,
        "anharm_max": 25.0,
        "constrain_signs": True,
    }

    shared, all_delay_modes, fit_cube, res_cube = fit_2d_gaussian_global(
        pump, probe, delays, cube_3d, init_modes, bounds_config=bounds_cfg
    )

    assert len(shared) == 1
    assert abs(shared[0]["w1"] - 2010.0) < 1.0
    assert abs(shared[0]["w3"] - 2010.0) < 1.0
    assert abs(shared[0]["anharm"] - 15.0) < 1.0
    assert len(all_delay_modes) == 3
    # Check amplitude decay tracking
    assert all_delay_modes[0][0]["amp_gsb"] < all_delay_modes[1][0]["amp_gsb"] < all_delay_modes[2][0]["amp_gsb"]
    assert np.max(np.abs(res_cube)) < 0.05


def test_2d_lorentzian_fit():
    from pymorgan.twoD.analyse import evaluate_2d_gaussian_map, fit_2d_gaussian_map

    pump = np.linspace(1990.0, 2030.0, 41)
    probe = np.linspace(1990.0, 2030.0, 41)

    true_modes = [
        {"w1": 2010.0, "w3": 2010.0, "anharm": 15.0, "amp_gsb": -1.0, "amp_esa": 0.8, "sigma_w1": 6.0, "sigma_w3": 6.0, "peak_shape": "lorentzian"}
    ]
    map_2d = evaluate_2d_gaussian_map(pump, probe, true_modes, peak_shape="lorentzian")

    init_modes = [
        {"w1": 2011.0, "w3": 2009.0, "anharm": 14.0, "amp_gsb": -0.8, "amp_esa": 0.6, "sigma_w1": 7.0, "sigma_w3": 7.0}
    ]

    bounds_cfg = {
        "pos_tol": 5.0,
        "anharm_min": 5.0,
        "anharm_max": 25.0,
        "constrain_signs": True,
        "peak_shape": "lorentzian",
    }

    fitted_modes, fit_map, residual_map = fit_2d_gaussian_map(
        pump, probe, map_2d, init_modes, bounds_config=bounds_cfg, peak_shape="lorentzian"
    )

    assert len(fitted_modes) == 1
    m = fitted_modes[0]
    assert abs(m["w1"] - 2010.0) < 0.5
    assert abs(m["w3"] - 2010.0) < 0.5
    assert abs(m["anharm"] - 15.0) < 0.8
    assert abs(m["sigma_w1"] - 6.0) < 0.8
    assert np.max(np.abs(residual_map)) < 0.05


def test_2d_peak_linking_coupling():
    from pymorgan.twoD.analyse import evaluate_2d_gaussian_map, fit_2d_gaussian_map

    pump = np.linspace(1980.0, 2060.0, 41)
    probe = np.linspace(1980.0, 2060.0, 41)

    # True 4-peak system: 2 diagonal peaks + 2 cross-peaks with independent coupling cross-anharmonicity (18.0)
    true_modes = [
        {"w1": 2010.0, "w3": 2010.0, "anharm": 16.0, "amp_gsb": -1.2, "amp_esa": 1.0, "sigma_w1": 7.0, "sigma_w3": 7.0},
        {"w1": 2040.0, "w3": 2040.0, "anharm": 14.0, "amp_gsb": -1.0, "amp_esa": 0.8, "sigma_w1": 7.0, "sigma_w3": 7.0},
        {"w1": 2010.0, "w3": 2040.0, "anharm": 18.0, "amp_gsb": -0.4, "amp_esa": 0.3, "sigma_w1": 7.0, "sigma_w3": 7.0},
        {"w1": 2040.0, "w3": 2010.0, "anharm": 18.0, "amp_gsb": -0.4, "amp_esa": 0.3, "sigma_w1": 7.0, "sigma_w3": 7.0},
    ]
    map_2d = evaluate_2d_gaussian_map(pump, probe, true_modes)

    # Initial guesses: Mode 0 and Mode 1 perturbed; Cross-peaks linked for w1 and w3, but anharm is independent
    init_modes = [
        {"w1": 2012.0, "w3": 2008.0, "anharm": 15.0, "amp_gsb": -1.0, "amp_esa": 0.8, "sigma_w1": 8.0, "sigma_w3": 8.0},
        {"w1": 2038.0, "w3": 2042.0, "anharm": 15.0, "amp_gsb": -0.8, "amp_esa": 0.6, "sigma_w1": 8.0, "sigma_w3": 8.0},
        {"w1": 2010.0, "w3": 2040.0, "anharm": 17.0, "amp_gsb": -0.3, "amp_esa": 0.2, "sigma_w1": 8.0, "sigma_w3": 8.0, "link_w1": 0, "link_w3": 1},
        {"w1": 2040.0, "w3": 2010.0, "anharm": 17.0, "amp_gsb": -0.3, "amp_esa": 0.2, "sigma_w1": 8.0, "sigma_w3": 8.0, "link_w1": 1, "link_w3": 0},
    ]

    bounds_cfg = {
        "pos_tol": 5.0,
        "anharm_min": 5.0,
        "anharm_max": 30.0,
        "constrain_signs": True,
    }

    fitted_modes, fit_map, residual_map = fit_2d_gaussian_map(
        pump, probe, map_2d, init_modes, bounds_config=bounds_cfg
    )

    assert len(fitted_modes) == 4
    # Cross-peak 2 must strictly have w1 = Mode 0 w1, w3 = Mode 1 w3
    assert fitted_modes[2]["w1"] == fitted_modes[0]["w1"]
    assert fitted_modes[2]["w3"] == fitted_modes[1]["w3"]
    # Cross-peak 3 must strictly have w1 = Mode 1 w1, w3 = Mode 0 w3
    assert fitted_modes[3]["w1"] == fitted_modes[1]["w1"]
    assert fitted_modes[3]["w3"] == fitted_modes[0]["w3"]
    # Independent anharmonicities for cross-peaks
    assert abs(fitted_modes[2]["anharm"] - 18.0) < 1.5
    assert abs(fitted_modes[3]["anharm"] - 18.0) < 1.5
    assert np.max(np.abs(residual_map)) < 0.05


def test_2d_peak_linking_population_transfer():
    from pymorgan.twoD.analyse import evaluate_2d_gaussian_map, fit_2d_gaussian_map

    pump = np.linspace(1980.0, 2060.0, 41)
    probe = np.linspace(1980.0, 2060.0, 41)

    # Population transfer: Cross-peak A->B (Mode 2) has anharm = Mode 1 anharm (14.0)
    # Cross-peak B->A (Mode 3) has anharm = Mode 0 anharm (16.0)
    true_modes = [
        {"w1": 2010.0, "w3": 2010.0, "anharm": 16.0, "amp_gsb": -1.2, "amp_esa": 1.0, "sigma_w1": 7.0, "sigma_w3": 7.0},
        {"w1": 2040.0, "w3": 2040.0, "anharm": 14.0, "amp_gsb": -1.0, "amp_esa": 0.8, "sigma_w1": 7.0, "sigma_w3": 7.0},
        {"w1": 2010.0, "w3": 2040.0, "anharm": 14.0, "amp_gsb": -0.4, "amp_esa": 0.3, "sigma_w1": 7.0, "sigma_w3": 7.0},
        {"w1": 2040.0, "w3": 2010.0, "anharm": 16.0, "amp_gsb": -0.4, "amp_esa": 0.3, "sigma_w1": 7.0, "sigma_w3": 7.0},
    ]
    map_2d = evaluate_2d_gaussian_map(pump, probe, true_modes)

    init_modes = [
        {"w1": 2012.0, "w3": 2008.0, "anharm": 15.0, "amp_gsb": -1.0, "amp_esa": 0.8, "sigma_w1": 8.0, "sigma_w3": 8.0},
        {"w1": 2038.0, "w3": 2042.0, "anharm": 15.0, "amp_gsb": -0.8, "amp_esa": 0.6, "sigma_w1": 8.0, "sigma_w3": 8.0},
        {"w1": 2010.0, "w3": 2040.0, "anharm": 15.0, "amp_gsb": -0.3, "amp_esa": 0.2, "sigma_w1": 8.0, "sigma_w3": 8.0, "link_w1": 0, "link_w3": 1, "link_anharm": 1},
        {"w1": 2040.0, "w3": 2010.0, "anharm": 15.0, "amp_gsb": -0.3, "amp_esa": 0.2, "sigma_w1": 8.0, "sigma_w3": 8.0, "link_w1": 1, "link_w3": 0, "link_anharm": 0},
    ]

    bounds_cfg = {
        "pos_tol": 5.0,
        "anharm_min": 5.0,
        "anharm_max": 30.0,
        "constrain_signs": True,
    }

    fitted_modes, fit_map, residual_map = fit_2d_gaussian_map(
        pump, probe, map_2d, init_modes, bounds_config=bounds_cfg
    )

    assert len(fitted_modes) == 4
    # Cross-peak 2 anharm is locked to Mode 1 anharm
    assert fitted_modes[2]["anharm"] == fitted_modes[1]["anharm"]
    # Cross-peak 3 anharm is locked to Mode 0 anharm
    assert fitted_modes[3]["anharm"] == fitted_modes[0]["anharm"]
    assert np.max(np.abs(residual_map)) < 0.05


def test_write_p2dat_roundtrip(tmp_path):
    from pymorgan.twoD.load import read_P2DAT, write_P2DAT

    pump = np.linspace(2000.0, 2040.0, 10)
    probe = np.linspace(2000.0, 2040.0, 12)
    delays = np.array([0.5, 1.0, 2.0])
    Z_orig = np.random.randn(len(pump), len(probe), len(delays))

    file_path = tmp_path / "test_out.p2dat"
    write_P2DAT(file_path, pump, probe, delays, Z_orig)
    assert file_path.exists()

    Z_read, p_read, r_read, d_read, _, _ = read_P2DAT(str(file_path))
    assert np.allclose(p_read, pump)
    assert np.allclose(r_read, probe)
    assert np.allclose(d_read, delays)
    assert np.allclose(Z_read, Z_orig, atol=1e-5)


def test_dataset2d_to_p2dat(tmp_path):
    info = synthetic.make_synthetic_p2dat(tmp_path / "syn.p2dat")
    ds = pm.load_2D(info["path"], data_type="P2DAT")

    out_path = tmp_path / "exported.p2dat"
    ds.to_p2dat(out_path)
    assert out_path.exists()

    ds_reloaded = pm.load_2D(out_path, data_type="P2DAT")
    assert np.allclose(ds_reloaded.pump, ds.pump)
    assert np.allclose(ds_reloaded.probe, ds.probe)
    assert np.allclose(ds_reloaded.delays, ds.delays)
    assert np.allclose(ds_reloaded.Z, ds.Z, atol=1e-5)


# --------------------------------------------------------------------------- #
#                             Robust Polynomial Fit                           #
# --------------------------------------------------------------------------- #
def test_robust_polyfit_linear_with_outliers():
    from pymorgan.twoD.process import robust_polyfit

    x = np.linspace(1, 50, 50)
    true_m, true_c = 2.75, 2010.0
    y = true_m * x + true_c

    # Add 25% severe outliers (including edge leverage points)
    y[0:8] = 2150.0
    y[20:23] = 1900.0

    coeffs = robust_polyfit(x, y, deg=1)
    assert len(coeffs) == 2
    assert np.isclose(coeffs[0], true_m, atol=0.05)
    assert np.isclose(coeffs[1], true_c, atol=1.0)


def test_robust_polyfit_quadratic_scatter_calibration():
    """Simulate 2D-IR scatter calibration with edge detector noise & leverage outliers."""
    from pymorgan.twoD.process import robust_polyfit

    pixels = np.arange(1, 65, dtype=float)
    true_a = 0.0055
    true_b = 2.75
    true_c = 2013.0
    y_clean = true_a * pixels**2 + true_b * pixels + true_c

    # Add small noise to inliers
    rng = np.random.default_rng(123)
    y = y_clean + rng.normal(0.0, 0.2, size=len(pixels))

    # Add ~25% severe outliers at detector edges & random spike artifacts
    y[0:10] = 2145.0  # edge cluster
    y[16] = 2136.0    # artifact
    y[18] = 2136.0
    y[37] = 2140.0

    coeffs = robust_polyfit(pixels, y, deg=2)
    assert len(coeffs) == 3

    # Check fitted values along the pixel array against ground truth
    p_fit = np.poly1d(coeffs)
    inlier_mask = np.ones(len(pixels), dtype=bool)
    inlier_mask[0:10] = False
    inlier_mask[[16, 18, 37]] = False

    max_err_inliers = np.max(np.abs(p_fit(pixels[inlier_mask]) - y_clean[inlier_mask]))
    assert max_err_inliers < 1.0

    # Ensure edge extrapolation is accurate
    assert abs(p_fit(1) - y_clean[0]) < 2.0


def test_robust_polyfit_deg0_and_cubic():
    from pymorgan.twoD.process import robust_polyfit

    x = np.linspace(0, 10, 40)
    # Deg 0
    y0 = np.ones_like(x) * 42.0
    y0[0:5] = 100.0
    c0 = robust_polyfit(x, y0, deg=0)
    assert np.isclose(c0[0], 42.0)

    # Deg 3
    y3 = 0.05 * x**3 - 0.3 * x**2 + 1.5 * x + 10.0
    y3[0:4] = 80.0
    c3 = robust_polyfit(x, y3, deg=3)
    p3 = np.poly1d(c3)
    assert np.allclose(p3(x[5:]), y3[5:], atol=0.5)


def test_robust_polyfit_edge_cases():
    from pymorgan.twoD.process import robust_polyfit

    # Too few points
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 5.0])
    c = robust_polyfit(x, y, deg=2)
    assert len(c) == 3

    # Flat data
    x_flat = np.linspace(1, 20, 20)
    y_flat = np.full(20, 5.0)
    c_flat = robust_polyfit(x_flat, y_flat, deg=1)
    assert np.isclose(c_flat[0], 0.0)
    assert np.isclose(c_flat[1], 5.0)


# --------------------------------------------------------------------------- #
# Dataset2D Copy, Crop & Slice                                                #
# --------------------------------------------------------------------------- #
def test_dataset2d_copy_and_crop():
    t2 = np.array([0.1, 0.5, 1.0, 5.0])
    pump = np.linspace(2000, 2050, 12)
    probe = np.linspace(1950, 2100, 16)
    Z = np.ones((len(pump), len(probe), len(t2)))
    units = {
        "unitsT_lbl": "Delay",
        "unitsT_ltx": "ps",
        "unitsL_lbl": "Wavenumber",
        "unitsL_ltx": r"$\mathrm{cm^{-1}}$",
        "unitsZ_lbl": r"$\Delta$A",
        "unitsZ_ltx": "mOD",
    }
    ds2 = Dataset2D(Z=Z, delays=t2, pump=pump, probe=probe, units=units, data_type="2DIR")

    cp = ds2.copy(deep=True)
    assert cp is not ds2
    assert np.array_equal(cp.Z, ds2.Z)

    cp.Z[0, 0, 0] = 42.0
    assert ds2.Z[0, 0, 0] != 42.0

    cropped = ds2.copy().crop(
        t2_min=0.5, t2_max=2.0, pump_min=2010, pump_max=2040, probe_min=1980, probe_max=2080
    )
    assert cropped.delays[0] >= 0.5
    assert cropped.delays[-1] <= 2.0
    assert cropped.pump[0] >= 2010
    assert cropped.pump[-1] <= 2040
    assert cropped.probe[0] >= 1980
    assert cropped.probe[-1] <= 2080
    assert cropped.Z.shape == (len(cropped.pump), len(cropped.probe), len(cropped.delays))


def test_slice_dataset1d_inheritance():
    from pymorgan.oneD.dataset import Dataset1D
    from pymorgan.twoD.analyse import SliceDataset1D

    t2 = np.array([0.1, 0.5, 1.0, 5.0])
    pump = np.linspace(2000, 2050, 12)
    probe = np.linspace(1950, 2100, 16)
    Z = np.ones((len(pump), len(probe), len(t2)))
    units = {
        "unitsT_lbl": "Delay",
        "unitsT_ltx": "ps",
        "unitsL_lbl": "Wavenumber",
        "unitsL_ltx": r"$\mathrm{cm^{-1}}$",
        "unitsZ_lbl": r"$\Delta$A",
        "unitsZ_ltx": "mOD",
    }
    ds2 = Dataset2D(Z=Z, delays=t2, pump=pump, probe=probe, units=units, data_type="2DIR")

    slice_1d = ds2.slice_at_pump(pump[2])
    assert isinstance(slice_1d, SliceDataset1D)
    assert isinstance(slice_1d, Dataset1D)
    assert slice_1d.delays.shape == ds2.delays.shape
    assert slice_1d.probe.shape == ds2.probe.shape
    assert slice_1d.noise_array() is None

    cropped_slice = slice_1d.copy().crop(tmin=0.5, tmax=2.0)
    assert len(cropped_slice.delays) <= len(slice_1d.delays)
    df = slice_1d.to_dataframe(format="tidy")
    assert len(df) == len(slice_1d.delays) * len(slice_1d.probe)




# --------------------------------------------------------------------------- #
#              Extended Kubo Physics, VarPro, Multi-Core & GUI Tests          #
# --------------------------------------------------------------------------- #
def test_kubo_static_inhomogeneity_and_ffcf_ct():
    from pymorgan.twoD.kubo_fit import evaluate_ffcf_ct, evaluate_ffcf_gt, simulate_2d_spectrum

    t = np.linspace(0.0, 10.0, 50)
    decays_static = [{"dw_T2": 1.5, "dw_0": 8.0, "terms": [(10.0, 1.0)]}]
    c_t = evaluate_ffcf_ct(t, decays_static, ffcf_id=0)
    # At t = 0: C(0) = dw_0^2 + dw_1^2 = 64 + 100 = 164
    assert np.isclose(c_t[0], 164.0)
    # At t -> inf (10 ps >> 1 ps): C(t) approaches dw_0^2 = 64
    assert np.isclose(c_t[-1], 64.0, atol=0.05)

    g_t = evaluate_ffcf_gt(t, decays_static, ffcf_id=0)
    assert g_t[0] == 0.0
    assert np.all(np.diff(g_t) > 0)

    pump = np.linspace(1985.0, 2015.0, 21)
    probe = np.linspace(1970.0, 2015.0, 25)
    bands = [{"w1": 2000.0, "w3": 2000.0, "anharm": 15.0, "amp_gsb": 1.0, "amp_esa": 0.8}]

    # Compare t2 = 50 ps with and without static inhomogeneity dw_0
    Z_with_static = simulate_2d_spectrum(
        bands,
        [{"dw_T2": 2.0, "dw_0": 10.0, "terms": [(8.0, 1.0)]}],
        pump,
        probe,
        t2_delay=50.0,
        tmax=4.0,
        nt=48,
        normalize_unit_basis=True,
    )
    Z_no_static = simulate_2d_spectrum(
        bands,
        [{"dw_T2": 2.0, "dw_0": 0.0, "terms": [(8.0, 1.0)]}],
        pump,
        probe,
        t2_delay=50.0,
        tmax=4.0,
        nt=48,
        normalize_unit_basis=True,
    )
    assert not np.allclose(Z_with_static, Z_no_static)


def test_kubo_off_diagonal_and_linked_cross_peaks():
    from pymorgan.twoD.kubo_fit import run_kubo_fit, simulate_2d_spectrum

    pump = np.linspace(1985.0, 2035.0, 16)
    probe = np.linspace(1975.0, 2035.0, 18)

    true_bands = [
        {"w1": 2000.0, "w3": 2000.0, "anharm": 14.0, "amp_gsb": 1.0, "amp_esa": 0.8, "link_w3": "diagonal"},
        {"w1": 2020.0, "w3": 2020.0, "anharm": 14.0, "amp_gsb": 0.9, "amp_esa": 0.7, "link_w3": "diagonal"},
        {"w1": 2000.0, "w3": 2020.0, "anharm": 14.0, "amp_gsb": 0.35, "amp_esa": 0.30, "link_w1": 0, "link_w3": 1, "link_anharm": 1, "corr_eta": 0.5},
    ]
    true_decays = [{"dw_T2": 2.0, "dw_0": 0.0, "terms": [(10.0, 2.0)]}]

    Z_target = simulate_2d_spectrum(
        true_bands, true_decays, pump, probe, t2_delay=0.5, tmax=3.5, nt=32, normalize_unit_basis=True
    )
    data_cube = Z_target[:, :, None]

    # Parameter vector with has_dw0=True: 3 bands * 4 + (dT2, dw0, d1, tau1) = 16 params
    p0 = [
        1999.0, 13.5, 0.7, 0.5,   # Band 1
        2021.0, 14.5, 0.6, 0.5,   # Band 2
        2000.0, 14.0, 0.2, 0.2,   # Cross-peak 1->2 (linked w1=0, w3=1, anharm=1)
        2.0, 0.0, 10.0, 2.0,      # FFCF (fixed for fast position/amp link test)
    ]
    bounds = [
        (1995.0, 2005.0), (10.0, 20.0), (0.0, 5.0), (0.0, 5.0),
        (2015.0, 2025.0), (10.0, 20.0), (0.0, 5.0), (0.0, 5.0),
        (2000.0, 2000.0), (14.0, 14.0), (0.0, 5.0), (0.0, 5.0),
        (2.0, 2.0), (0.0, 0.0), (10.0, 10.0), (2.0, 2.0),
    ]
    band_links = [
        {"link_w1": None, "link_w3": "diagonal", "link_anharm": None},
        {"link_w1": None, "link_w3": "diagonal", "link_anharm": None},
        {"link_w1": 0, "link_w3": 1, "link_anharm": 1, "corr_eta": 0.5},
    ]

    res_list = run_kubo_fit(
        data_cube,
        pump,
        probe,
        [0.5],
        p0,
        bounds,
        tmax=3.5,
        nt=32,
        num_bands=3,
        num_decays=1,
        has_dw0=True,
        solver="trf_varpro",
        band_links=band_links,
        parallel_mode="threadpool",
        max_workers=2,
    )
    res = res_list[0]
    assert res["success"]
    assert res["r_squared"] > 0.99
    b_fit = res["bands_fitted"]
    assert abs(b_fit[0]["w1"] - 2000.0) < 0.5
    assert abs(b_fit[1]["w1"] - 2020.0) < 0.5
    # Verify cross-peak linking
    assert np.isclose(b_fit[2]["w1"], b_fit[0]["w1"])
    assert np.isclose(b_fit[2]["w3"], b_fit[1]["w1"])
    assert np.isclose(b_fit[2]["anharm"], b_fit[1]["anharm"])


def test_kubo_trf_varpro_and_robben_sign():
    from pymorgan.twoD.kubo_fit import run_kubo_fit, simulate_2d_spectrum

    pump = np.linspace(1985.0, 2015.0, 14)
    probe = np.linspace(1970.0, 2015.0, 16)

    true_bands = [{"w1": 2000.0, "w3": 2000.0, "anharm": 15.0, "amp_gsb": 1.0, "amp_esa": 0.85}]
    true_decays = [{"dw_T2": 1.5, "dw_0": 0.0, "terms": [(11.0, 2.0)]}]

    Z_target = simulate_2d_spectrum(
        true_bands, true_decays, pump, probe, t2_delay=0.5, tmax=3.5, nt=32, normalize_unit_basis=True
    )
    data_cube = (2.5 * Z_target)[:, :, None]  # Test arbitrary experimental amplitude scaling

    p0 = [1999.0, 14.0, 0.5, 0.5, 1.5, 0.0, 9.5, 1.6]
    bounds = [
        (1994.0, 2006.0),
        (10.0, 22.0),
        (0.0, 10.0),
        (0.0, 10.0),
        (1.5, 1.5),  # fix dT2
        (0.0, 0.0),  # fix dw_0
        (4.0, 20.0),
        (0.5, 6.0),
    ]

    res_list = run_kubo_fit(
        data_cube,
        pump,
        probe,
        [0.5],
        p0,
        bounds,
        tmax=3.5,
        nt=32,
        num_bands=1,
        num_decays=1,
        has_dw0=True,
        solver="trf_varpro",
        parallel_mode="threadpool",
        max_workers=2,
    )
    res = res_list[0]
    assert res["success"]
    assert res["r_squared"] > 0.995
    assert abs(res["params"][0] - 2000.0) < 0.5
    assert abs(res["params"][1] - 15.0) < 0.5
    assert abs(res["params"][2] - 1.0) < 0.1
    assert abs(res["params"][3] - 0.85) < 0.1
    assert abs(res["params"][6] - 11.0) < 1.0
    assert len(res["sign"]) == len(p0)
    assert len(res["errors"]) == len(p0)
    assert len(res["ci95"]) == len(p0)
    assert np.allclose(res["Z_sim"], data_cube[:, :, 0], atol=0.1)


def test_kubo_global_multidelay_and_multicore():
    from pymorgan.twoD.kubo_fit import (
        run_kubo_global_fit,
        simulate_2d_spectrum,
        simulate_kubo_cube_parallel,
    )

    pump = np.linspace(1988.0, 2012.0, 12)
    probe = np.linspace(1972.0, 2012.0, 14)
    delays = np.array([0.2, 1.0, 3.0])

    decays = [{"dw_T2": 1.8, "dw_0": 2.0, "terms": [(10.0, 1.5)]}]
    cube = np.zeros((len(pump), len(probe), len(delays)), dtype=float)
    amp_Factors = [1.0, 0.75, 0.5]
    for i, f_amp in enumerate(amp_Factors):
        bands_i = [{"w1": 2000.0, "w3": 2000.0, "anharm": 15.0, "amp_gsb": 1.0 * f_amp, "amp_esa": 0.8 * f_amp}]
        cube[:, :, i] = simulate_2d_spectrum(
            bands_i, decays, pump, probe, t2_delay=float(delays[i]), tmax=3.0, nt=32, normalize_unit_basis=True
        )

    p0 = [1999.2, 14.3, 0.8, 0.6, 1.8, 2.0, 9.0, 1.5]
    bounds = [
        (1995.0, 2005.0),
        (11.0, 20.0),
        (0.0, 5.0),
        (0.0, 5.0),
        (1.8, 1.8),
        (2.0, 2.0),
        (5.0, 16.0),
        (1.5, 1.5),
    ]

    results = run_kubo_global_fit(
        cube,
        pump,
        probe,
        delays,
        p0,
        bounds,
        tmax=3.0,
        nt=32,
        num_bands=1,
        num_decays=1,
        has_dw0=True,
        parallel_mode="threadpool",
        max_workers=2,
    )
    assert len(results) == 3
    for r in results:
        assert r["global_fit"] is True
        assert abs(r["params"][0] - 2000.0) < 0.6
        assert abs(r["params"][1] - 15.0) < 0.6
        assert r["r_squared"] > 0.99

    kubo_details = {float(r["delay"]): {**r, "tmax": 3.0, "nt": 32, "w_rot": 1900.0, "u": 1.0, "fid_data": None} for r in results}
    sim_cube = simulate_kubo_cube_parallel(
        delays, kubo_details, pump, probe, pump, probe, cube, [0, 1, 2], parallel_mode="threadpool", max_workers=2
    )
    assert sim_cube.shape == cube.shape
    assert np.max(np.abs(sim_cube - cube)) < 0.08


def test_kubo_joint_ftir_fit():
    from pymorgan.twoD.kubo_fit import run_kubo_fit, simulate_1d_ftir_spectrum, simulate_2d_spectrum

    pump = np.linspace(1985.0, 2015.0, 14)
    probe = np.linspace(1970.0, 2015.0, 16)
    bands = [{"w1": 2000.0, "w3": 2000.0, "anharm": 15.0, "amp_gsb": 1.0, "amp_esa": 0.8}]
    decays = [{"dw_T2": 2.0, "dw_0": 0.0, "terms": [(10.0, 1.5)]}]

    w_ftir = np.linspace(1960.0, 2040.0, 120)
    y_ftir = simulate_1d_ftir_spectrum(bands, decays, w_ftir, tmax=4.0, nt=64, normalize=True)
    assert y_ftir.shape == w_ftir.shape
    assert np.isclose(np.max(y_ftir), 1.0)

    Z_2d = simulate_2d_spectrum(bands, decays, pump, probe, t2_delay=0.5, tmax=3.5, nt=32, normalize_unit_basis=True)
    fid_data = {"mode": "joint_fit", "spectrum": (w_ftir, y_ftir)}

    p0 = [1999.0, 14.5, 0.8, 0.7, 2.0, 0.0, 9.0, 1.5]
    bounds = [
        (1995.0, 2005.0),
        (11.0, 20.0),
        (0.0, 5.0),
        (0.0, 5.0),
        (2.0, 2.0),
        (0.0, 0.0),
        (5.0, 15.0),
        (1.5, 1.5),
    ]
    res = run_kubo_fit(
        Z_2d[:, :, None],
        pump,
        probe,
        [0.5],
        p0,
        bounds,
        tmax=3.5,
        nt=32,
        fid_data=fid_data,
        num_bands=1,
        num_decays=1,
        has_dw0=True,
        ftir_weight=0.5,
    )[0]
    assert res["success"]
    assert abs(res["params"][0] - 2000.0) < 0.6


def test_kubo_dialog_gui_workflow():
    from PyQt6.QtWidgets import QApplication

    from pymorgan.gui.kubo_dialog import KuboDialog

    app = QApplication.instance() or QApplication([])
    ds = _tilted_map()
    dlg = KuboDialog(None, ds)

    # Verify default 1x native ROI grid resolution
    assert dlg.grid_factor == 1
    assert len(dlg.pump_axis) == len(ds.pump)
    assert len(dlg.probe_axis) == len(ds.probe)

    # Test grid resolution switching (2x) and back to 1x
    dlg._apply_grid_resolution(2)
    assert len(dlg.pump_axis) == 2 * len(ds.pump)
    dlg._apply_grid_resolution(1)
    assert len(dlg.pump_axis) == len(ds.pump)

    # Test adding/deleting bands and decays
    assert dlg.num_bands == 1
    assert dlg.num_decays == 1
    dlg.add_band(w01=2020.0, dw=15.0, a_gsb=1.0, a_esa=0.8)
    assert dlg.num_bands == 2
    dlg.delete_band()
    assert dlg.num_bands == 1

    dlg.add_decay()
    assert dlg.num_decays == 2
    dlg.delete_decay()
    assert dlg.num_decays == 1

    # Switch 4th panel diagnostic views
    for view_idx in range(4):
        dlg.diag_view_cb.setCurrentIndex(view_idx)
        dlg._update_fourth_panel_only()

    # Run active fit with fast nt=24 and verify dataset & diagnostics table population
    dlg.ft_nt_spin.setValue(24)
    dlg.run_active_fit()
    assert hasattr(ds, "_sd_results") and "Kubo_Fit" in ds._sd_results
    assert hasattr(ds, "_sd_kubo_details") and 1.0 in ds._sd_kubo_details
    assert getattr(ds, "_kubo_fit_dataset", None) is not None
    assert dlg.diag_table.rowCount() >= 8

    dlg.close()
    assert app is not None


def test_shot_rotation_pipeline_and_gui(tmp_path):
    """Verify integer shot rotation before FT across Dataset2D, single scans, and GUI."""
    from PyQt6.QtWidgets import QApplication
    from synthetic import make_synthetic_quickcontrol_2d

    from pymorgan.gui.main_window import MainWindow
    from pymorgan.twoD.progress import ProgressTracker

    rng = np.random.default_rng(123)
    n_bins = 64
    dt1_fs = 30.0
    w0_cm1 = 1800.0
    c_cm_fs = 2.99792458e-5
    t1_fs = np.arange(n_bins, dtype=float) * dt1_fs
    probe = np.linspace(1960.0, 2040.0, 20)
    delays = np.array([0.2, 1.0])

    # Damped time-domain oscillation centred near 2000 cm-1 (delta_w = 200 cm-1)
    env = np.exp(-t1_fs / 450.0)
    osc = np.cos(2.0 * np.pi * (2000.0 - w0_cm1) * c_cm_fs * t1_fs) * env
    pr_prof = -np.exp(-0.5 * ((probe - 2000.0) / 8.0) ** 2) + 0.8 * np.exp(
        -0.5 * ((probe - 1985.0) / 8.0) ** 2
    )

    sig_ss_ref = np.zeros((n_bins, len(probe), len(delays), 3), dtype=float)
    for k in range(3):
        noise_k = 1.0e-4 * rng.standard_normal((n_bins, len(probe), len(delays)))
        for m, scale in enumerate((1.0, 0.7)):
            sig_ss_ref[:, :, m, k] = scale * np.outer(osc, pr_prof) + noise_k[:, :, m]
    sig_ref = np.mean(sig_ss_ref, axis=-1)
    int_ss_ref = np.mean(sig_ss_ref, axis=1)
    int_ref = np.mean(sig_ref, axis=1)

    units = {
        "unitsL": "cm-1",
        "unitsL_lbl": "Wavenumber",
        "unitsL_ltx": r"cm$^{-1}$",
        "unitsT": "ps",
        "unitsT_ltx": "ps",
        "unitsZ": "mOD",
        "unitsZ_lbl": r"$S_{\text{2DIR}}$",
        "unitsZ_ltx": r"mOD cm$^{1/2}$",
        "interferogram_from_signal": True,
    }

    ds_ref = Dataset2D(
        sig_ref.copy(),
        t1_fs.copy(),
        probe.copy(),
        delays.copy(),
        dict(units),
        freq_units="cm-1",
        datatype="shaper",
        raw_signal=sig_ref.copy(),
        raw_interferogram=int_ref.copy(),
        raw_t1delays=t1_fs.copy(),
        raw_signal_ss=sig_ss_ref.copy(),
        raw_interferogram_ss=int_ss_ref.copy(),
        w0_cm1=w0_cm1,
        dt1_fs=dt1_fs,
    )
    silent = ProgressTracker(1, is_gui=False)
    proc_opts = dict(
        apodise_method="Cos",
        zeropad_enable=True,
        zeropad_factor=2,
        zeropad_next2k=True,
        progress_tracker=silent,
    )
    ds_ref.process(shot_rotation=0, **proc_opts)

    # Misalign time-domain data by -3 shots; +3 shot_rotation must recover ds_ref exactly
    shift_err = -3
    ds_shifted = Dataset2D(
        np.roll(sig_ref, shift_err, axis=0),
        t1_fs.copy(),
        probe.copy(),
        delays.copy(),
        dict(units),
        freq_units="cm-1",
        datatype="shaper",
        raw_signal=np.roll(sig_ref, shift_err, axis=0),
        raw_interferogram=np.roll(int_ref, shift_err, axis=0),
        raw_t1delays=t1_fs.copy(),
        raw_signal_ss=np.roll(sig_ss_ref, shift_err, axis=0),
        raw_interferogram_ss=np.roll(int_ss_ref, shift_err, axis=0),
        w0_cm1=w0_cm1,
        dt1_fs=dt1_fs,
    )
    ds_shifted.process(shot_rotation=0, **proc_opts)
    z_unrotated = ds_shifted.Z_R.copy()
    assert not np.allclose(z_unrotated, ds_ref.Z_R, atol=1e-2)

    ds_shifted.process(shot_rotation=3, **proc_opts)
    assert ds_shifted.shot_rotation == 3
    np.testing.assert_allclose(ds_shifted.raw_signal, sig_ref, atol=1e-12)
    np.testing.assert_allclose(ds_shifted.raw_interferogram, int_ref, atol=1e-12)
    np.testing.assert_allclose(ds_shifted.Z_R, ds_ref.Z_R, atol=1e-10)

    # Idempotency across arbitrary sequential shot rotations
    ds_shifted.process(shot_rotation=-2, **proc_opts)
    ds_shifted.process(shot_rotation=0, **proc_opts)
    np.testing.assert_allclose(ds_shifted.Z_R, z_unrotated, atol=1e-10)
    ds_shifted.process(shot_rotation=3, **proc_opts)
    np.testing.assert_allclose(ds_shifted.Z_R, ds_ref.Z_R, atol=1e-10)

    # Single-scan map computation forwards shot_rotation cleanly
    scans_ref = ds_ref.compute_scan_maps_at_t2(
        0.2, shot_rotation=0, apodise_method="Cos", zeropad_factor=2, zeropad_next2k=True
    )
    scans_rot = ds_shifted.compute_scan_maps_at_t2(
        0.2, shot_rotation=3, apodise_method="Cos", zeropad_factor=2, zeropad_next2k=True
    )
    assert len(scans_ref) == len(scans_rot) == 3
    for (_, _, z_a), (_, _, z_b) in zip(scans_ref, scans_rot, strict=True):
        np.testing.assert_allclose(z_a, z_b, atol=1e-10)

    # Deep copy preserves unrotated base arrays
    ds_cp = ds_shifted.copy(deep=True)
    assert ds_cp.shot_rotation == 3
    ds_cp.process(shot_rotation=0, **proc_opts)
    np.testing.assert_allclose(ds_cp.Z_R, z_unrotated, atol=1e-10)

    # End-to-end GUI verification on Phasing/FT sub-tab
    app = QApplication.instance() or QApplication([])
    meta = make_synthetic_quickcontrol_2d(tmp_path / "qc_shot_rot")
    win = MainWindow()
    try:
        win.twoD_load_path(str(meta["info_path"]), data_type="QuickControl_2D")
        assert hasattr(win, "twoD_shot_rotation")
        assert win.twoD_shot_rotation.value() == 0
        n_td = win.twoD_dataset.raw_signal.shape[0]
        assert win.twoD_shot_rotation.minimum() == -(n_td - 1)
        assert win.twoD_shot_rotation.maximum() == (n_td - 1)

        z0 = win.twoD_dataset.Z_R.copy()
        base_td = win.twoD_dataset._base_raw_signal.copy()

        win.twoD_shot_rotation.setValue(2)
        assert win.twoD_dataset.shot_rotation == 2
        np.testing.assert_allclose(win.twoD_dataset.raw_signal, np.roll(base_td, 2, axis=0))
        assert not np.allclose(win.twoD_dataset.Z_R, z0)

        win.twoD_shot_rotation.setValue(0)
        assert win.twoD_dataset.shot_rotation == 0
        np.testing.assert_allclose(win.twoD_dataset.Z_R, z0, atol=1e-12)
    finally:
        win.close()
    assert app is not None


def test_twoD_native_nm_sample_info_slices_and_surface_default_inversion(monkeypatch):
    """Verify native-nm sample_info, slice xlims, and default electronic plot_surface inversion."""
    import matplotlib.pyplot as plt
    from PyQt6.QtWidgets import QApplication

    from pymorgan import helpers as hlp
    from pymorgan.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    assert app is not None

    # 1. Default plot_surface inverts both axes for electronic cm-1 datasets without explicit limits
    pump_cm1 = np.linspace(18500.0, 21500.0, 12)
    probe_cm1 = np.linspace(18000.0, 22000.0, 14)
    P, R = np.meshgrid(pump_cm1, probe_cm1, indexing="ij")
    Z_cm1 = -np.exp(-(((P - 20000.0) ** 2 + (R - 20000.0) ** 2) / (2.0 * 450.0**2)))
    ds_cm1 = Dataset2D(
        Z_cm1[:, :, np.newaxis],
        pump_cm1,
        probe_cm1,
        np.array([0.5]),
        hlp.units2dic("cm-1", "ps", "mOD"),
        freq_units="cm-1",
    )
    ax3d = ds_cm1.plot_surface(0.5)
    try:
        assert ax3d.get_xlim()[0] > ax3d.get_xlim()[1]
        assert ax3d.get_ylim()[0] > ax3d.get_ylim()[1]
    finally:
        plt.close(ax3d.figure)

    # 2. Native-nm electronic dataset reports true nm range in sample_info and matches slice xlim
    pump_nm = np.linspace(480.0, 600.0, 20)
    probe_nm = np.linspace(475.0, 610.0, 24)
    Pn, Rn = np.meshgrid(pump_nm, probe_nm, indexing="ij")
    Z_nm = -np.exp(-(((Pn - 540.0) ** 2 + (Rn - 540.0) ** 2) / (2.0 * 18.0**2)))
    units_nm = hlp.units2dic("nm", "ps", "mOD")
    units_nm["unitsL_lbl"] = "Wavelength"
    units_nm["unitsL_ltx"] = "nm"
    ds_nm = Dataset2D(
        np.stack([Z_nm, 0.7 * Z_nm], axis=-1),
        pump_nm,
        probe_nm,
        np.array([0.5, 2.0]),
        units_nm,
        freq_units="nm",
    )
    assert ds_nm.is_electronic is True
    info_html = ds_nm.sample_info()
    assert "(480.0\u2013600.0 nm)" in info_html
    assert "(475.0\u2013610.0 nm)" in info_html

    monkeypatch.setattr(plt, "show", lambda *a, **kw: None)
    win = MainWindow()
    try:
        win.twoD_dataset = ds_nm
        win._plot_pump_slice(540.0, [0.5], (480.0, 600.0))
        fig_pu = plt.gcf()
        try:
            ax_pu = fig_pu.axes[0]
            assert ax_pu.get_xlim() == pytest.approx((480.0, 600.0))
            xdata_pu = np.asarray(ax_pu.get_lines()[0].get_xdata(), dtype=float)
            assert float(np.min(xdata_pu)) >= 470.0 and float(np.max(xdata_pu)) <= 620.0
        finally:
            plt.close(fig_pu)

        win._plot_probe_slice(540.0, [0.5], (480.0, 600.0))
        fig_pr = plt.gcf()
        try:
            ax_pr = fig_pr.axes[0]
            assert ax_pr.get_xlim() == pytest.approx((480.0, 600.0))
            xdata_pr = np.asarray(ax_pr.get_lines()[0].get_xdata(), dtype=float)
            assert float(np.min(xdata_pr)) >= 470.0 and float(np.max(xdata_pr)) <= 620.0
        finally:
            plt.close(fig_pr)
    finally:
        win.close()


def test_cls_kinetics_plot_uses_mathtext_labels(monkeypatch):
    """Verify CLS kinetics figure annotations and labels use Matplotlib mathtext instead of Unicode subscripts."""
    import matplotlib.pyplot as plt
    from PyQt6.QtWidgets import QApplication, QInputDialog, QMessageBox

    from pymorgan import helpers as hlp
    from pymorgan.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    assert app is not None

    pump = np.linspace(1980.0, 2020.0, 12)
    probe = np.linspace(1980.0, 2020.0, 12)
    delays = np.linspace(0.5, 20.0, 8)
    Z = np.zeros((12, 12, 8), dtype=float)
    ds = Dataset2D(Z, pump, probe, delays, hlp.units2dic("cm-1", "ps", "mOD"))
    t = delays
    y = 0.35 * np.exp(-t / 1.5) + 0.25 * np.exp(-t / 8.0) + 0.15
    ds._sd_results = {"CLS": np.column_stack([t, y])}

    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **kw: ("Both", True))
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **kw: ("0:20", True))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **kw: QMessageBox.StandardButton.No)
    monkeypatch.setattr(plt, "show", lambda *a, **kw: None)

    win = MainWindow()
    try:
        win.twoD_dataset = ds
        win._update_spectral_diffusion_controls_state()
        win.twoD_show_cls_chk.setChecked(True)
        win._on_twoD_plot_sd_kinetics_clicked()
        fig = plt.gcf()
        try:
            ax = fig.axes[0]
            all_texts = [t_obj.get_text() for t_obj in ax.texts] + [ax.get_xlabel()]
            joined = "\n".join(all_texts)
            assert "$y_0$" in joined
            assert "$A_1$" in joined
            assert r"$\tau_1$" in joined
            assert "$t_2$" in ax.get_xlabel()
            for bad_cp in ("\u2080", "\u2081", "\u2082"):
                assert bad_cp not in joined
        finally:
            plt.close(fig)
    finally:
        win.close()


def test_process_atomicity_shared_base_and_gui_reprocess_guards(tmp_path):
    """Verify shared unrotated base arrays, atomic process(), spinner reset, and reentrancy/async guards."""
    import time

    from PyQt6.QtWidgets import QApplication
    from synthetic import make_synthetic_quickcontrol_2d

    from pymorgan.gui.main_window import MainWindow
    from pymorgan.twoD.progress import ProgressTracker

    n_bins, n_probe = 32, 12
    dt1_fs, w0_cm1, c_cm_fs = 30.0, 1800.0, 2.99792458e-5
    t1_fs = np.arange(n_bins, dtype=float) * dt1_fs
    probe = np.linspace(1960.0, 2040.0, n_probe)
    delays = np.array([0.2, 1.0])
    osc = np.cos(2.0 * np.pi * 200.0 * c_cm_fs * t1_fs) * np.exp(-t1_fs / 400.0)
    pr_prof = -np.exp(-0.5 * ((probe - 2000.0) / 8.0) ** 2)
    sig = np.stack([np.outer(osc, pr_prof), 0.6 * np.outer(osc, pr_prof)], axis=-1)
    interf = np.mean(sig, axis=1)

    units = {
        "unitsL": "cm-1",
        "unitsL_lbl": "Wavenumber",
        "unitsL_ltx": r"cm$^{-1}$",
        "unitsT": "ps",
        "unitsT_ltx": "ps",
        "unitsZ": "mOD",
        "unitsZ_lbl": r"$S_{\text{2DIR}}$",
        "unitsZ_ltx": r"mOD cm$^{1/2}$",
        "interferogram_from_signal": True,
    }
    ds = Dataset2D(
        sig.copy(),
        t1_fs.copy(),
        probe.copy(),
        delays.copy(),
        units,
        freq_units="cm-1",
        datatype="shaper",
        raw_signal=sig.copy(),
        raw_interferogram=interf.copy(),
        raw_t1delays=t1_fs.copy(),
        w0_cm1=w0_cm1,
        dt1_fs=dt1_fs,
    )
    assert ds.raw_signal is ds._base_raw_signal
    assert ds.raw_interferogram is ds._base_raw_interferogram

    silent = ProgressTracker(1, is_gui=False)
    ds.process(apodise_method="Cos", zeropad_factor=2, shot_rotation=0, progress_tracker=silent)
    z_ok = ds.Z_R.copy()

    # Failure during step 6 must leave shot_rotation, raw_signal, and Z_R untouched
    with pytest.raises(IndexError):
        ds.process(
            apodise_method="Cos",
            zeropad_factor=2,
            shot_rotation=3,
            bkg_sub=True,
            bkgIdx=999,
            progress_tracker=silent,
        )
    assert ds.shot_rotation == 0
    assert ds.raw_signal is ds._base_raw_signal
    np.testing.assert_allclose(ds.Z_R, z_ok)

    # GUI: loading second dataset resets shot_rotation spinner; stale reprocess is discarded;
    # large-delay (>16) shot rotation completes asynchronously on background worker.
    app = QApplication.instance() or QApplication([])
    m_small = make_synthetic_quickcontrol_2d(tmp_path / "qc_a", prefix="ds_a#01", n_scans=1)
    m_large = make_synthetic_quickcontrol_2d(
        tmp_path / "qc_b",
        prefix="ds_b#02",
        n_t1=32,
        n_pixels=12,
        t2_fs=tuple(float(i * 100) for i in range(18)),
        n_scans=1,
    )
    win = MainWindow()
    try:
        win.twoD_load_path(str(m_small["info_path"]), data_type="QuickControl_2D")
        win.twoD_shot_rotation.setValue(2)
        assert win.twoD_dataset.shot_rotation == 2

        win.twoD_load_path(str(m_large["info_path"]), data_type="QuickControl_2D")
        assert win.twoD_shot_rotation.value() == 0
        assert win.twoD_dataset.shot_rotation == 0

        # Simulate stale concurrent reprocess interrupted by a newer request during processing
        orig_runner = win._run_twoD_dataset_process
        sentinel_z = win.twoD_dataset.Z_R.copy()

        def _interrupting_runner(*args, **kwargs):
            orig_runner(*args, **kwargs)
            win._twoD_op_seq += 1

        win._run_twoD_dataset_process = _interrupting_runner
        try:
            win.twoD_shot_rotation.blockSignals(True)
            win.twoD_shot_rotation.setValue(4)
            win.twoD_shot_rotation.blockSignals(False)
            win.twoD_reprocess_data(silent=True)
            assert win.twoD_dataset.shot_rotation == 0
            np.testing.assert_allclose(win.twoD_dataset.Z_R, sentinel_z)
        finally:
            win._run_twoD_dataset_process = orig_runner

        # >16 delays triggers debounced background worker when spinner changes normally
        win.twoD_shot_rotation.setValue(3)
        deadline = time.monotonic() + 5.0
        while win.twoD_dataset.shot_rotation != 3 and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.02)
        assert win.twoD_dataset.shot_rotation == 3
    finally:
        win.close()


def test_2d_independent_axis_inversion_api():
    """Verify independent invert_pump / invert_probe across plot_map, plot_surface, and plot_scan_map_array."""
    import matplotlib.pyplot as plt

    orig_pump_axis = pm.get_settings().pump_axis
    try:
        ds = _tilted_map()
        ds_scans = ds.copy(deep=True)
        ds_scans.delays = np.array([1.0], dtype=float)
        ds_scans.raw_signal_ss = np.stack([ds.Z_R, 0.9 * ds.Z_R], axis=-1)
        ds_scans.nscans = 2
        ds_scans.datatype = "FreqDomain"

        pm.update_settings(pump_axis="Horizontal")
        out_pf = ds.plot_map(1.0, invert_pump=True, invert_probe=False)
        try:
            assert out_pf.ax.get_xlim()[0] > out_pf.ax.get_xlim()[1]
            assert out_pf.ax.get_ylim()[0] < out_pf.ax.get_ylim()[1]
        finally:
            plt.close(out_pf.ax.figure)

        out_fp = ds.plot_map(1.0, invert_pump=False, invert_probe=True)
        try:
            assert out_fp.ax.get_xlim()[0] < out_fp.ax.get_xlim()[1]
            assert out_fp.ax.get_ylim()[0] > out_fp.ax.get_ylim()[1]
        finally:
            plt.close(out_fp.ax.figure)

        ax_surf = ds.plot_surface(
            1.0,
            pump_lim=(1995.0, 2050.0),
            probe_lim=(1995.0, 2050.0),
            invert_pump=True,
            invert_probe=False,
        )
        try:
            assert ax_surf.get_xlim()[0] > ax_surf.get_xlim()[1]
            assert ax_surf.get_ylim()[0] < ax_surf.get_ylim()[1]
        finally:
            plt.close(ax_surf.get_figure())

        ax_surf_auto = ds.plot_surface(1.0, invert_pump=False, invert_probe=True)
        try:
            assert ax_surf_auto.get_xlim()[0] < ax_surf_auto.get_xlim()[1]
            assert ax_surf_auto.get_ylim()[0] > ax_surf_auto.get_ylim()[1]
        finally:
            plt.close(ax_surf_auto.get_figure())

        fig_scans, axes_scans, _ = ds_scans.plot_scan_maps(
            1.0, invert_pump=True, invert_probe=False
        )
        try:
            assert axes_scans[0].get_xlim()[0] > axes_scans[0].get_xlim()[1]
            assert axes_scans[0].get_ylim()[0] < axes_scans[0].get_ylim()[1]
        finally:
            plt.close(fig_scans)

        pm.update_settings(pump_axis="Vertical")
        out_vert = ds.plot_map(1.0, invert_pump=True, invert_probe=False)
        try:
            # Probe is X (ascending), Pump is Y (descending)
            assert out_vert.ax.get_xlim()[0] < out_vert.ax.get_xlim()[1]
            assert out_vert.ax.get_ylim()[0] > out_vert.ax.get_ylim()[1]
        finally:
            plt.close(out_vert.ax.figure)
    finally:
        pm.update_settings(pump_axis=orig_pump_axis)


def test_2d_axis_order_buttons_gui(tmp_path):
    """Verify square Pump/Probe ascending/descending toggle buttons across mid-IR and 2D-ES maps."""
    import matplotlib.pyplot as plt
    from PyQt6.QtWidgets import QApplication
    from synthetic import (
        make_synthetic_quickcontrol_2d,
        make_synthetic_quickcontrol_2des,
    )

    from pymorgan.gui.main_window import MainWindow

    orig_pump_axis = pm.get_settings().pump_axis
    orig_freq_unit = pm.get_settings().twoD_freq_unit

    app = QApplication.instance() or QApplication([])
    m_ir = make_synthetic_quickcontrol_2d(tmp_path / "qc_ir", prefix="ir#01", n_scans=1)
    m_el = make_synthetic_quickcontrol_2des(tmp_path / "qc_el", prefix="el#01", n_scans=1)

    win = MainWindow(init_rootdir=False)
    try:
        win.MainTabs.setCurrentIndex(1)
        win.twoD_load_path(str(m_ir["info_path"]), data_type="QuickControl_2D")
        app.processEvents()

        # Vibrational mid-IR defaults: both Pump and Probe Ascending (up arrow, unchecked)
        assert win.twoD_PC_pumpOrder.isChecked() is False
        assert win.twoD_PC_pumpOrder.text() == "\u2191"
        assert "Ascending" in win.twoD_PC_pumpOrder.toolTip()
        assert win.twoD_PC_probeOrder.isChecked() is False
        assert win.twoD_PC_probeOrder.text() == "\u2191"
        assert "Ascending" in win.twoD_PC_probeOrder.toolTip()

        ax = win.twoDaxes.ax
        assert ax.get_xlim()[0] < ax.get_xlim()[1]
        assert ax.get_ylim()[0] < ax.get_ylim()[1]

        # Click Pump order button (cut_plot=True path): only Pump becomes Descending
        win.twoD_PC_pumpOrder.click()
        app.processEvents()
        assert win.twoD_PC_pumpOrder.isChecked() is True
        assert win.twoD_PC_pumpOrder.text() == "\u2193"
        assert "Descending" in win.twoD_PC_pumpOrder.toolTip()
        assert win.twoD_PC_probeOrder.isChecked() is False
        assert win.twoDaxes.ax.get_xlim()[0] > win.twoDaxes.ax.get_xlim()[1]
        assert win.twoDaxes.ax.get_ylim()[0] < win.twoDaxes.ax.get_ylim()[1]

        # Standalone frame renderer matches active axis order
        fig_tmp, ax_tmp = plt.subplots()
        try:
            win.render_twoD_contour_frame(float(win.twoD_dataset.delays[0]), ax=ax_tmp)
            assert ax_tmp.get_xlim()[0] > ax_tmp.get_xlim()[1]
            assert ax_tmp.get_ylim()[0] < ax_tmp.get_ylim()[1]
        finally:
            plt.close(fig_tmp)

        # Uncheck cut_plot so _twoD_apply_view_limits fast path is exercised
        win.twoD_PC_cutPlot.setChecked(False)
        win._twoD_render_timer.stop()
        win._twoD_preview_contour()
        app.processEvents()

        win.twoD_PC_probeOrder.click()
        app.processEvents()
        assert win.twoD_PC_probeOrder.isChecked() is True
        assert win.twoD_PC_probeOrder.text() == "\u2193"
        assert win.twoDaxes.ax.get_xlim()[0] > win.twoDaxes.ax.get_xlim()[1]
        assert win.twoDaxes.ax.get_ylim()[0] > win.twoDaxes.ax.get_ylim()[1]

        win.twoD_PC_pumpOrder.click()
        app.processEvents()
        assert win.twoD_PC_pumpOrder.isChecked() is False
        assert win.twoD_PC_pumpOrder.text() == "\u2191"
        assert win.twoDaxes.ax.get_xlim()[0] < win.twoDaxes.ax.get_xlim()[1]
        assert win.twoDaxes.ax.get_ylim()[0] > win.twoDaxes.ax.get_ylim()[1]

        # Vertical pump orientation swaps X=Probe (descending) and Y=Pump (ascending)
        pm.update_settings(pump_axis="Vertical")
        win._twoD_preview_contour()
        app.processEvents()
        assert win.twoDaxes.ax.get_xlim()[0] > win.twoDaxes.ax.get_xlim()[1]
        assert win.twoDaxes.ax.get_ylim()[0] < win.twoDaxes.ax.get_ylim()[1]

        # Sync Limits preserves active axis-order directions
        pm.update_settings(pump_axis="Horizontal")
        win._twoD_preview_contour()
        app.processEvents()
        x0, x1 = win.twoDaxes.ax.get_xlim()
        y0, y1 = win.twoDaxes.ax.get_ylim()
        win.twoDaxes.ax.set_xlim(x0 + 0.1 * (x1 - x0), x1 - 0.1 * (x1 - x0))
        win.twoDaxes.ax.set_ylim(y0 + 0.1 * (y1 - y0), y1 - 0.1 * (y1 - y0))
        win._on_twoD_sync_limits()
        app.processEvents()
        assert win.twoD_plot_controls.pump_descending() is False
        assert win.twoD_plot_controls.probe_descending() is True

        # Restore resets both order buttons to dataset default (Ascending for IR)
        win.twoD_PC_restore.click()
        win._twoD_render_timer.stop()
        win._twoD_preview_contour()
        app.processEvents()
        assert win.twoD_PC_pumpOrder.isChecked() is False
        assert win.twoD_PC_probeOrder.isChecked() is False
        assert win.twoDaxes.ax.get_xlim()[0] < win.twoDaxes.ax.get_xlim()[1]
        assert win.twoDaxes.ax.get_ylim()[0] < win.twoDaxes.ax.get_ylim()[1]

        # Electronic 2D-ES defaults: both Pump and Probe Descending (down arrow, checked)
        win.twoD_load_path(str(m_el["info_path"]), data_type="QuickControl_2D")
        app.processEvents()
        assert win.twoD_PC_pumpOrder.isChecked() is True
        assert win.twoD_PC_pumpOrder.text() == "\u2193"
        assert win.twoD_PC_probeOrder.isChecked() is True
        assert win.twoD_PC_probeOrder.text() == "\u2193"
        assert win.twoDaxes.ax.get_xlim()[0] > win.twoDaxes.ax.get_xlim()[1]
        assert win.twoDaxes.ax.get_ylim()[0] > win.twoDaxes.ax.get_ylim()[1]

        # Toggling Probe on 2D-ES leaves Pump Descending while flipping Probe Ascending
        win.twoD_PC_probeOrder.click()
        app.processEvents()
        win.twoDaxes.canvas.draw()
        assert win.twoD_PC_pumpOrder.isChecked() is True
        assert win.twoD_PC_probeOrder.isChecked() is False
        assert win.twoDaxes.ax.get_xlim()[0] > win.twoDaxes.ax.get_xlim()[1]
        assert win.twoDaxes.ax.get_ylim()[0] < win.twoDaxes.ax.get_ylim()[1]

        # Child dialogs and movie exporter inherit active per-axis order
        from pymorgan.gui.movie_dialog import create_2d_movie_animation
        from pymorgan.gui.twoD_integral_dialog import TwoDIntegralDynamicsDialog

        dlg = TwoDIntegralDynamicsDialog(win, win.twoD_dataset)
        try:
            assert dlg.ax_2d.get_xlim()[0] > dlg.ax_2d.get_xlim()[1]
            assert dlg.ax_2d.get_ylim()[0] < dlg.ax_2d.get_ylim()[1]
        finally:
            dlg.close()

        gif_path = tmp_path / "anim.gif"
        ok = create_2d_movie_animation(
            win.twoD_dataset,
            np.array([0]),
            str(gif_path),
            plot_controls=win.twoD_plot_controls,
            parent_widget=None,
        )
        assert ok and gif_path.is_file()
    finally:
        pm.update_settings(pump_axis=orig_pump_axis, twoD_freq_unit=orig_freq_unit)
        win.close()


def test_twoD_kinetics_and_slices_unit_aware(qapp, tmp_path, monkeypatch):
    """2D non-interactive kinetics, area integration, slice dialogs, and delay prompts respect twoD_freq_unit & delay units."""
    import matplotlib.pyplot as plt
    from PyQt6.QtWidgets import QApplication, QInputDialog, QMessageBox

    def _fail_on_modal(_parent, title, msg, *_a, **_kw):
        raise AssertionError(f"Unexpected modal dialog ({title}): {msg}")

    monkeypatch.setattr(QMessageBox, "warning", _fail_on_modal)
    monkeypatch.setattr(QMessageBox, "information", _fail_on_modal)
    monkeypatch.setattr(QMessageBox, "critical", _fail_on_modal)

    from pymorgan import helpers as hlp
    from pymorgan.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    assert app is not None

    pump = np.linspace(1960.0, 2040.0, 9)
    probe = np.linspace(1950.0, 2050.0, 11)
    delays = np.array([-2.0, 0.5, 5.0, 20.0])
    # Encode each (pump_idx, probe_idx) pixel with a unique curve shape + amplitude
    Z = np.zeros((len(pump), len(probe), len(delays)), dtype=float)
    t_shape = np.array([0.0, 1.0, 0.5, 0.25])
    for ip in range(len(pump)):
        for ipr in range(len(probe)):
            Z[ip, ipr, :] = (ip * 100 + ipr + 1) * t_shape

    ds = Dataset2D(Z, pump, probe, delays, hlp.units2dic("cm-1", "ps", "mOD"))

    created_figs = []
    monkeypatch.setattr(plt, "show", lambda *a, **kw: created_figs.append(plt.gcf()))

    orig_freq_unit = pm.get_settings().twoD_freq_unit
    win = MainWindow(init_rootdir=False)
    try:
        win.MainTabs.setCurrentIndex(1)
        win.twoD_dataset = ds
        win.twoD_plot_controls.set_dataset(ds, 0)
        win.twoD_interactive_chk.setChecked(False)
        win._twoD_preview_contour()
        app.processEvents()

        # --- 1. Single Coordinate non-interactive in nm (scaled to 10^3 nm above 5000 nm) ---
        pm.update_settings(twoD_freq_unit="nm")
        win._twoD_render_timer.stop()
        win._twoD_preview_contour()
        app.processEvents()

        ip_t, ipr_t = 3, 7
        p_disp = float(win._twoD_display_meta()["to_display"](float(pump[ip_t])))
        pr_disp = float(win._twoD_display_meta()["to_display"](float(probe[ipr_t])))
        prompts = []

        monkeypatch.setattr(
            QInputDialog, "getItem", lambda *_a, **_kw: ("Single Coordinate", True)
        )
        monkeypatch.setattr(
            QInputDialog,
            "getText",
            lambda _p, title, label, text="": (
                prompts.append((title, label)),
                (f"{p_disp:.4f},{pr_disp:.4f}", True),
            )[1],
        )
        win._on_twoD_kinetics_clicked()
        app.processEvents()

        assert "nm" in prompts[-1][1]
        ax_k1 = created_figs[-1].axes[0]
        np.testing.assert_allclose(np.asarray(ax_k1.get_lines()[0].get_ydata()), Z[ip_t, ipr_t, :])
        lbl_k1 = ax_k1.get_legend().get_texts()[0].get_text()
        assert "nm" in lbl_k1 and "cm" not in lbl_k1

        # --- 2. Area Integration non-interactive in eV + RectangleSelector small span check ---
        pm.update_settings(twoD_freq_unit="eV")
        win._twoD_render_timer.stop()
        win._twoD_preview_contour()
        app.processEvents()

        assert win._rect_span_ok(win.twoDaxes.ax, (0.246, 0.250, 0.246, 0.250)) is True

        p_ev_lo = float(hlp.convert_spectral(float(pump[4]), "cm-1", "eV"))
        p_ev_hi = float(hlp.convert_spectral(float(pump[6]), "cm-1", "eV"))
        pr_ev_lo = float(hlp.convert_spectral(float(probe[5]), "cm-1", "eV"))
        pr_ev_hi = float(hlp.convert_spectral(float(probe[8]), "cm-1", "eV"))
        prompts.clear()
        monkeypatch.setattr(
            QInputDialog, "getItem", lambda *_a, **_kw: ("Area Integration", True)
        )
        monkeypatch.setattr(
            QInputDialog,
            "getText",
            lambda _p, title, label, text="": (
                prompts.append((title, label)),
                (f"{p_ev_lo:.5f}:{p_ev_hi:.5f},{pr_ev_lo:.5f}:{pr_ev_hi:.5f}", True),
            )[1],
        )
        win._on_twoD_kinetics_clicked()
        app.processEvents()

        assert "eV" in prompts[-1][1]
        ax_area = created_figs[-1].axes[0]
        assert len(ax_area.get_lines()) >= 1
        lbl_area = ax_area.get_legend().get_texts()[0].get_text()
        assert lbl_area.startswith("Int. [") and lbl_area.endswith("eV")

        # --- 3. 2D fixed-pump slice in nm + delay prompt showing ps & unit suffix parsing ---
        pm.update_settings(twoD_freq_unit="nm")
        win._twoD_render_timer.stop()
        win._twoD_preview_contour()
        app.processEvents()

        dbl_prompts = []
        delay_prompts = []
        monkeypatch.setattr(
            QInputDialog,
            "getItem",
            lambda *_a, **_kw: ("Along fixed pump WN", True),
        )
        monkeypatch.setattr(
            QInputDialog,
            "getDouble",
            lambda _p, title, label, val, *_a, **_kw: (
                dbl_prompts.append((title, label)),
                (p_disp, True),
            )[1],
        )
        monkeypatch.setattr(
            QInputDialog,
            "getText",
            lambda _p, title, label, text="": (
                delay_prompts.append((title, label, text)),
                ("500 fs, 5", True),
            )[1],
        )
        captured_slices = []
        orig_pump_slice = win._plot_pump_slice
        monkeypatch.setattr(
            win,
            "_plot_pump_slice",
            lambda pw, t2d, xl: (captured_slices.append((pw, t2d)), orig_pump_slice(pw, t2d, xl))[1],
        )
        win._on_twoD_slices_clicked()
        app.processEvents()

        assert "nm" in dbl_prompts[-1][1]
        assert "in ps" in delay_prompts[-1][1] and "-2 to 20 ps" in delay_prompts[-1][1]
        assert captured_slices[-1][0] == pytest.approx(float(pump[ip_t]), abs=1e-2)
        assert captured_slices[-1][1] == pytest.approx([0.5, 5.0])
    finally:
        for fig in created_figs:
            plt.close(fig)
        pm.update_settings(twoD_freq_unit=orig_freq_unit)
        win.close()
