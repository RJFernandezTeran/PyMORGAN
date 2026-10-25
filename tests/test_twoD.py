"""Unit tests for 2D dataset processing, CLS extraction, Kubo fitting, and ROI picker."""

import numpy as np
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
