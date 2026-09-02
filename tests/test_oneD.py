"""Unit tests for 1D dataset processing, background correction, chirp correction, shockwave subtraction, and steady-state loading."""

import copy
import numpy as np
import pytest
import synthetic

import pymorgan as pm
from pymorgan.oneD.dataset import Dataset1D
from pymorgan.steadyState.dataset import load_spectrum
from pymorgan.oneD.chirp import (
    ChirpFit,
    _bisquare_weights,
    _interpolate_nans,
    cauchy_t0,
    fit_cauchy_dispersion,
    load_chirp_fit,
    save_chirp_fit,
)


# --------------------------------------------------------------------------- #
#                           Background & Solvent                              #
# --------------------------------------------------------------------------- #
def test_background_subtracted(dataset):
    dataset.background_correct(tmin=-20, tmax=-1, do_correct=True)
    pre = dataset.delays < 0
    assert np.allclose(np.nanmean(dataset.Zavg_C[pre, :, 0]), 0.0, atol=1e-6)
    assert np.allclose(dataset.bkg_avg, 0.05, atol=1e-6)
    assert dataset.is_corrected


def test_passthrough_leaves_data(dataset):
    raw = dataset.Zavg_R.copy()
    dataset.background_correct(tmin=-20, tmax=-1, do_correct=False)
    assert np.array_equal(dataset.Zavg_C, raw)


def test_Z_property_prefers_corrected(dataset):
    assert np.array_equal(dataset.Z, dataset.Zavg_R)
    dataset.background_correct(tmin=-20, tmax=-1)
    assert np.array_equal(dataset.Z, dataset.Zavg_C)


def test_returns_self_for_chaining(dataset):
    assert dataset.background_correct(-20, -1) is dataset


def test_solvent_subtraction(dataset):
    solvent = copy.deepcopy(dataset)
    solvent.Zavg_R = dataset.Zavg_R * 0.5
    dataset.subtract_solvent(solvent, dt=0.0, scale=2.0)

    assert np.allclose(dataset.Z, 0.0, atol=1e-5)
    assert dataset._solvent_dataset is solvent
    assert dataset._solvent_dt == 0.0
    assert dataset._solvent_scale == 2.0


def test_fit_solvent_auto(dataset):
    solvent = copy.deepcopy(dataset)
    solvent.Zavg_R = dataset.Zavg_R * 0.8
    scale_opt, dt_opt = dataset.fit_solvent_auto(solvent)

    assert isinstance(dt_opt, float)
    assert isinstance(scale_opt, np.ndarray)
    assert scale_opt.shape == (dataset.probe.size,)
    assert np.allclose(np.nanmean(scale_opt), 1.25, atol=2e-1)


def test_fit_solvent_auto_with_nans(dataset):
    solvent = copy.deepcopy(dataset)
    solvent.Zavg_R = dataset.Zavg_R * 0.8
    dataset.Zavg_R[10:20, 5, 0] = np.nan
    solvent.Zavg_R[30:40, 10, 0] = np.nan

    scale_opt, dt_opt = dataset.fit_solvent_auto(solvent)
    assert isinstance(dt_opt, float)
    assert isinstance(scale_opt, np.ndarray)
    assert scale_opt.shape == (dataset.probe.size,)
    assert np.all(np.isfinite(scale_opt))
    assert np.isfinite(dt_opt)


def test_fit_solvent_auto_per_pixel_dt(dataset):
    solvent = copy.deepcopy(dataset)
    solvent.Zavg_R = dataset.Zavg_R * 0.8

    scale_opt, dt_opt = dataset.fit_solvent_auto(solvent, per_pixel_dt=True)
    assert isinstance(dt_opt, np.ndarray)
    assert dt_opt.shape == (dataset.probe.size,)
    assert isinstance(scale_opt, np.ndarray)
    assert scale_opt.shape == (dataset.probe.size,)
    assert np.all(np.isfinite(scale_opt))
    assert np.all(np.isfinite(dt_opt))
    assert np.allclose(np.nanmean(scale_opt), 1.25, atol=2e-1)


def test_fit_solvent_auto_with_mask(dataset):
    solvent = copy.deepcopy(dataset)
    solvent.Zavg_R = dataset.Zavg_R * 0.8

    dataset.mask_probe_regions([(500.0, 600.0)])
    scale_opt, dt_opt = dataset.fit_solvent_auto(solvent)

    assert isinstance(dt_opt, float)
    assert isinstance(scale_opt, np.ndarray)
    assert scale_opt.shape == (dataset.probe.size,)
    assert np.all(np.isfinite(scale_opt))
    assert np.isfinite(dt_opt)

    dataset.subtract_solvent(solvent, dt=dt_opt, scale=scale_opt)
    assert dataset.Zavg_C.shape == dataset.Z.shape


def test_multi_detector_mask_probe_regions():
    # 2 detectors, 50 pixels each, 10 delays
    delays = np.linspace(-1, 10, 10)
    probe = np.column_stack([np.linspace(1000, 1500, 50), np.linspace(2000, 2500, 50)])
    Zavg_R = np.ones((10, 50, 2))
    units = {"unitsL_lbl": "Wavenumber", "unitsL_ltx": r"$\mathrm{cm^{-1}}$"}

    ds = Dataset1D(Zavg_R, delays, probe, units)
    assert ds.n_detectors == 2

    # Mask range [1200, 1300] on detector 0
    ds.mask_probe_regions([(1200.0, 1300.0)], detector=0)

    # Shape must remain (10, 50, 2)
    assert ds.Zavg_R.shape == (10, 50, 2)

    p0 = ds._detector_probe(0)
    masked_p0_idx = (p0 >= 1200.0) & (p0 <= 1300.0)
    assert np.any(masked_p0_idx)

    # Detector 0 signal in masked region should be NaN
    assert np.all(np.isnan(ds.Zavg_R[:, masked_p0_idx, 0]))
    # Detector 0 signal outside masked region should be finite
    assert np.all(np.isfinite(ds.Zavg_R[:, ~masked_p0_idx, 0]))

    # Detector 1 signal should be completely unmasked (no NaNs)
    assert not np.any(np.isnan(ds.Zavg_R[:, :, 1]))

    # Mask range [2200, 2300] on detector 1
    ds.mask_probe_regions([(2200.0, 2300.0)], detector=1)
    p1 = ds._detector_probe(1)
    masked_p1_idx = (p1 >= 2200.0) & (p1 <= 2300.0)

    assert np.all(np.isnan(ds.Zavg_R[:, masked_p1_idx, 1]))
    assert np.all(np.isfinite(ds.Zavg_R[:, ~masked_p1_idx, 1]))
    # Detector 0 masked range remains intact
    assert np.all(np.isnan(ds.Zavg_R[:, masked_p0_idx, 0]))


def test_export_pdat_multi_detector(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from pymorgan.gui.main_window import MainWindow

    delays = np.linspace(-1, 10, 5)
    probe = np.column_stack([np.linspace(1000, 1500, 20), np.linspace(2000, 2500, 20)])
    Zavg_R = np.ones((5, 20, 2))
    units = {"unitsL_lbl": "Wavenumber", "unitsL_ltx": r"$\mathrm{cm^{-1}}$", "unitsT_ltx": "ps"}

    ds = Dataset1D(Zavg_R, delays, probe, units)

    mw = MainWindow.__new__(MainWindow)
    mw.dataset = ds
    mw._current_path = str(tmp_path / "sample.mat")
    mw._apply_background = lambda: None
    mw._rootdir_text = lambda: str(tmp_path)
    mw.statusBar = lambda: MagicMock()

    target_file = tmp_path / "sample_PROCESSED.pdat"
    monkeypatch.setattr("PyQt6.QtWidgets.QFileDialog.getSaveFileName", lambda *args, **kwargs: (str(target_file), ""))
    monkeypatch.setattr("PyQt6.QtWidgets.QMessageBox.information", lambda *args, **kwargs: None)

    mw._export_pdat()

    det1_file = tmp_path / "sample_PROCESSED_DET1.pdat"
    det2_file = tmp_path / "sample_PROCESSED_DET2.pdat"

    assert det1_file.exists()
    assert det2_file.exists()


def test_multi_detector_chirp_apply():
    from pymorgan.oneD.chirp import ChirpFit, default_chirp_filename, save_chirp_fit

    delays = np.linspace(-1, 10, 10)
    probe = np.column_stack([np.linspace(1000, 1500, 20), np.linspace(2000, 2500, 20)])
    Zavg_R = np.ones((10, 20, 2))
    # Distinct signal on detector 1 vs detector 0
    Zavg_R[:, :, 1] = 2.0
    units = {"unitsL_lbl": "Wavenumber", "unitsL_ltx": r"$\mathrm{cm^{-1}}$", "unitsT_ltx": "ps"}

    ds = Dataset1D(Zavg_R, delays, probe, units)

    # Fake fit for detector 0
    fit_det0 = ChirpFit(
        mode="automatic",
        coeffs=[0.5, 0.0],
        fit_wl=np.linspace(1000, 1500, 20),
        Pfit=np.ones((20, 8)),
        lambda_ref=1250.0,
        detector=0,
        Dfit=np.ones((10, 20)),
    )

    ds.apply_chirp_correction(fit_det0, detector=0)

    # Detector 1 data should remain untouched at 2.0
    assert np.allclose(ds.Zavg_R[:, :, 1], 2.0)


def test_multi_detector_chirp_save_filename(tmp_path):
    from pymorgan.oneD.chirp import ChirpFit, default_chirp_filename, save_chirp_fit

    fn_det1 = default_chirp_filename(detector=0, n_detectors=2)
    fn_det2 = default_chirp_filename(detector=1, n_detectors=2)
    assert "_DET1_" in fn_det1
    assert "_DET2_" in fn_det2

    fit = ChirpFit(
        mode="automatic",
        coeffs=[0.5, 0.0],
        fit_wl=np.linspace(1000, 1500, 20),
        Pfit=np.ones((20, 8)),
        lambda_ref=1250.0,
        detector=0,
        Dfit=np.ones((10, 20)),
    )

    save_path = tmp_path / "my_chirp.mat"
    save_chirp_fit(fit, save_path, n_detectors=2)

    expected_path = tmp_path / "my_chirp_DET1.mat"
    assert expected_path.exists()


# --------------------------------------------------------------------------- #
#                               Steady State                                  #
# --------------------------------------------------------------------------- #
def test_load_steady_state(tmp_path):
    csv_file = tmp_path / "abs.csv"
    data = np.column_stack([np.linspace(400, 700, 50), np.sin(np.linspace(0, np.pi, 50))])
    np.savetxt(csv_file, data, delimiter=",", header="Wavelength,Absorbance", comments="")
    ds = load_spectrum(csv_file)
    assert ds.x.ndim == 1
    assert ds.y.ndim == 1
    assert len(ds.x) == len(ds.y)


# --------------------------------------------------------------------------- #
#                             Chirp Correction                                #
# --------------------------------------------------------------------------- #
def test_cauchy_t0():
    coeffs = [1.0, -2.0, -3.0]
    probe = np.array([400.0, 500.0, 600.0])
    lambda_ref = 500.0

    expected = []
    for L in probe:
        val = 1.0 + (-2.0) * (500.0 / L) ** 2 + (-3.0) * (500.0 / L) ** 4
        expected.append(val)
    expected = np.array(expected)

    res = cauchy_t0(coeffs, probe, lambda_ref)
    assert np.allclose(res, expected)


def test_bisquare_weights():
    resid = np.array([0.1, -0.2, 0.3, 0.0, 10.0])
    w = _bisquare_weights(resid)
    assert w.shape == resid.shape
    assert w[4] < 0.1
    assert w[3] > 0.9

    resid_zeros = np.zeros(5)
    w_zeros = _bisquare_weights(resid_zeros)
    assert np.all(w_zeros == 1.0)


def test_interpolate_nans():
    x = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    x_int = _interpolate_nans(x)
    assert np.allclose(x_int, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_fit_cauchy_dispersion_basic():
    probe = np.linspace(400.0, 600.0, 20)
    true_t0 = 2.0 - 0.5 * (500.0 / probe) ** 2
    t0_obs = true_t0 + np.random.normal(0, 0.02, size=probe.shape)

    coeffs, lambda_ref = fit_cauchy_dispersion(probe, t0_obs, n_terms=2, lambda_ref=500.0)
    assert len(coeffs) == 2
    assert lambda_ref == 500.0


def test_save_load_chirp_fit(tmp_path):
    probe = np.linspace(400.0, 600.0, 10)
    fitted_t0 = np.linspace(1.0, 3.0, 10)
    coeffs = np.array([1.5, -0.2])
    fit = ChirpFit(mode="manual", coeffs=coeffs, fit_wl=probe, Pfit=np.column_stack([fitted_t0]), lambda_ref=500.0)

    save_file = tmp_path / "chirp_fit.pdat"
    save_chirp_fit(fit, save_file)
    assert save_file.exists()

    loaded = load_chirp_fit(save_file)
    assert loaded.lambda_ref == 500.0
    assert np.allclose(loaded.coeffs, coeffs)
    assert np.allclose(loaded.fit_wl, probe)


def test_apply_chirp_correction_dataset(dataset):
    ds = dataset
    fit = ChirpFit(
        mode="manual",
        coeffs=np.array([0.5, 0.0]),
        fit_wl=ds.probe,
        Pfit=np.full((ds.probe.size, 1), 0.5),
        lambda_ref=500.0,
    )

    ds.apply_chirp_correction(fit)
    assert ds.chirp_fit is fit


# --------------------------------------------------------------------------- #
#                           Shockwave Subtraction                            #
# --------------------------------------------------------------------------- #
def test_shockwave_subtraction_on_dataset(dataset):
    ds = dataset
    ds.subtract_shockwave(pixels=[0, 1], one_based=False)
    assert ds.is_corrected
    assert ds.Zavg_C is not None
    np.testing.assert_allclose(ds._shockwave_pixels, [0, 1])
    assert ds._shockwave_trace is not None


# --------------------------------------------------------------------------- #
#                             Time Derivative                                 #
# --------------------------------------------------------------------------- #
def test_time_derivative_process_basic():
    from pymorgan.oneD.process import time_derivative

    t = np.linspace(0.0, 50.0, 101)
    # Z(t) = 10 * exp(-t / 10.0) -> dZ/dt = -exp(-t / 10.0)
    Z = 10.0 * np.exp(-t / 10.0)[:, np.newaxis, np.newaxis]
    t_out, Z_dt, _ = time_derivative(t, Z)
    assert np.array_equal(t_out, t)
    # Check interior points for second-order accuracy
    expected = -np.exp(-t[1:-1] / 10.0)[:, np.newaxis, np.newaxis]
    np.testing.assert_allclose(Z_dt[1:-1], expected, atol=0.01, rtol=0.02)


def test_time_derivative_t_min_cutoff():
    from pymorgan.oneD.process import time_derivative

    t = np.linspace(-5.0, 50.0, 111)
    Z = np.sin(t)[:, np.newaxis]
    t_out, Z_dt, _ = time_derivative(t, Z, t_min=1.0)
    assert np.all(t_out >= 1.0)
    assert len(t_out) < len(t)


def test_time_derivative_interpolation_and_smoothing():
    from pymorgan.oneD.process import time_derivative

    t = np.geomspace(0.1, 100.0, 40)
    Z = np.exp(-t / 20.0)[:, np.newaxis]
    t_out, Z_dt, _ = time_derivative(
        t,
        Z,
        interpolate=True,
        n_interp=80,
        interp_kind="pchip",
        smooth=True,
        smooth_method="savgol",
        smooth_window=7,
    )
    assert len(t_out) == 80
    assert Z_dt.shape[0] == 80


def test_time_derivative_on_dataset(dataset):
    ds = dataset
    ds_dt = ds.time_derivative(t_min=0.0, smooth=True, smooth_window=5)
    assert isinstance(ds_dt, Dataset1D)
    assert np.all(ds_dt.delays >= 0.0)
    assert ds_dt.units.get("unitsZ_lbl") == r"$\mathrm{d}(\Delta A)/\mathrm{d}t$"
    assert ds_dt.Z.shape[0] == ds_dt.delays.shape[0]
    assert ds_dt.Z.shape[1] == ds.probe.shape[0]
