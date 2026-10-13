"""Unit tests for 1D dataset processing, background correction, chirp correction, shockwave subtraction, and steady-state loading."""

import copy

import numpy as np
import pytest

import pymorgan as pm
from pymorgan.oneD.chirp import (
    ChirpFit,
    _bisquare_weights,
    _interpolate_nans,
    cauchy_t0,
    fit_cauchy_dispersion,
    load_chirp_fit,
    save_chirp_fit,
)
from pymorgan.oneD.dataset import Dataset1D
from pymorgan.oneD.load import load_1D_hdf5, save_1D_hdf5
from pymorgan.oneD.stitch import (
    StitchResult,
    _optimize_linear_wl_shift,
    stitch_time,
    stitch_wavelength,
)
from pymorgan.oneD.svd import SVDResult, compute_svd
from pymorgan.steadyState.dataset import (
    Spectrum,
    SpectrumKind,
    SpectrumSeries,
    load_spectrum,
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
    from pymorgan.oneD.chirp import ChirpFit

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
    assert ds_dt.Z.shape[0] == ds_dt.delays.shape[0]
    assert ds_dt.Z.shape[1] == ds.probe.shape[0]


def test_recalc_average_harpia(tmp_path):
    import synthetic

    info = synthetic.make_synthetic_harpia_dataset(tmp_path / "sample_harpia.dat", nscans=4)
    ds = pm.load_1D(info["path"], data_type="HARPIA_TA")
    assert ds.has_single_scans
    assert ds.total_scans == 4
    assert ds.nscans == 4

    orig_zavg = ds.Zavg_R.copy()
    scan0 = ds.Zss_R[:, :, :, 0].copy()

    # Recalculate with scan 0 only
    ds.recalc_average([0])
    assert ds.nscans == 1
    assert ds.total_scans == 4
    np.testing.assert_allclose(ds.Zavg_R, scan0)
    assert np.all(ds.Zstdv == 0.0)

    # Recalculate with scans 0 and 1
    ds.recalc_average([0, 1])
    assert ds.nscans == 2
    assert ds.total_scans == 4
    expected_avg = np.nanmean(ds.Zss_R[:, :, :, :2], axis=-1)
    np.testing.assert_allclose(ds.Zavg_R, expected_avg)

    # Recalculate with all scans reproduces original average
    ds.recalc_average([0, 1, 2, 3])
    assert ds.nscans == 4
    np.testing.assert_allclose(ds.Zavg_R, orig_zavg)


# --------------------------------------------------------------------------- #
# Dataset1D Copy & Crop                                                       #
# --------------------------------------------------------------------------- #
def test_dataset1d_copy(dataset):
    cp = dataset.copy(deep=True)
    assert cp is not dataset
    assert np.array_equal(cp.delays, dataset.delays)
    assert np.array_equal(cp.probe, dataset.probe)
    assert np.array_equal(cp.Z, dataset.Z)
    assert cp.is_corrected == dataset.is_corrected

    # Mutate copy and verify original is unaffected
    cp.delays[0] = 999.0
    cp.Zavg_R[0, 0, 0] = -123.0
    cp.units["test_key"] = "test_val"

    assert dataset.delays[0] != 999.0
    assert dataset.Zavg_R[0, 0, 0] != -123.0
    assert "test_key" not in dataset.units


def test_dataset1d_crop(dataset):
    t_mid = float(dataset.delays[len(dataset.delays) // 2])
    p_mid = float(dataset.probe[len(dataset.probe) // 2])
    cropped = dataset.copy().crop(tmin=0.0, tmax=t_mid, probe_min=dataset.probe[0], probe_max=p_mid)
    assert cropped.delays[0] >= 0.0
    assert cropped.delays[-1] <= t_mid
    assert cropped.probe[0] >= dataset.probe[0]
    assert cropped.probe[-1] <= p_mid
    assert cropped.Zavg_R.shape[0] == len(cropped.delays)
    assert cropped.Zavg_R.shape[1] == len(cropped.probe)

    # Out of range crop raises ValueError
    with pytest.raises(ValueError, match="No delay points fall in range"):
        dataset.copy().crop(tmin=1e6, tmax=2e6)

    with pytest.raises(ValueError, match="No probe points fall in range"):
        dataset.copy().crop(probe_min=1e6, probe_max=2e6)


# --------------------------------------------------------------------------- #
# Dataset1D PDAT Export & Dataframe                                           #
# --------------------------------------------------------------------------- #
def test_write_pdat_and_to_pdat(tmp_path, dataset):
    out_file = tmp_path / "test_out.pdat"
    paths = dataset.to_pdat(out_file, use_corrected=True)
    reloaded = pm.load_1D(paths[0], data_type="PDAT")
    assert np.allclose(reloaded.delays, dataset.delays)
    assert np.allclose(reloaded.probe, dataset.probe)

    direct_pdat = tmp_path / "direct.pdat"
    pm.write_pdat(direct_pdat, dataset.delays, dataset.probe, dataset.Z[:, :, 0])
    assert direct_pdat.is_file()


def test_to_dataframe(dataset):
    df_tidy = dataset.to_dataframe(format="tidy")
    assert "delay" in df_tidy.columns
    assert "probe" in df_tidy.columns
    assert "signal" in df_tidy.columns
    assert len(df_tidy) == len(dataset.delays) * len(dataset.probe)

    df_wide = dataset.to_dataframe(format="wide")
    assert df_wide.shape == (len(dataset.delays), len(dataset.probe))
    assert np.allclose(df_wide.index.values, dataset.delays)

    with pytest.raises(ValueError, match="Unknown format"):
        dataset.to_dataframe(format="invalid")


# --------------------------------------------------------------------------- #
# HDF5 Serialization                                                          #
# --------------------------------------------------------------------------- #
def test_hdf5_roundtrip(tmp_path, dataset):
    h5_file = tmp_path / "dataset.h5"
    save_1D_hdf5(dataset, h5_file)
    assert h5_file.is_file()

    reloaded = load_1D_hdf5(h5_file)
    assert isinstance(reloaded, Dataset1D)
    assert np.allclose(reloaded.delays, dataset.delays)
    assert np.allclose(reloaded.probe, dataset.probe)
    assert np.allclose(reloaded.Zavg_R, dataset.Zavg_R)


# --------------------------------------------------------------------------- #
# Jupyter Notebook _repr_html_                                                #
# --------------------------------------------------------------------------- #
def test_repr_html(dataset):
    h1 = dataset._repr_html_()
    assert "PyMORGAN Dataset1D" in h1
    assert "delays" in h1

    # Spectrum
    x = np.linspace(400, 700, 50)
    y = np.exp(-((x - 550) ** 2) / (2 * 30**2))
    spec = Spectrum(x, y, kind=SpectrumKind.ABSORPTION, label="Q-band")
    h_spec = spec._repr_html_()
    assert "PyMORGAN Spectrum" in h_spec
    assert "Absorption" in h_spec

    # SpectrumSeries
    series = SpectrumSeries([spec, Spectrum(x, y * 0.8, kind=SpectrumKind.EMISSION, label="PL")])
    h_series = series._repr_html_()
    assert "PyMORGAN SpectrumSeries" in h_series
    assert "2 spectra" in h_series


# --------------------------------------------------------------------------- #
# Polarization Anisotropy                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def sample_1d_multidet():
    """Create a 2-detector 1D dataset for anisotropy testing."""
    delays = np.array([-2.0, 0.0, 1.0, 5.0])
    probe_d1 = np.linspace(1900, 2000, 15)
    probe_d2 = np.linspace(2000, 2100, 15)
    probe = np.column_stack([probe_d1, probe_d2])
    Z = np.ones((len(delays), 15, 2))
    Z[:, :, 0] *= 1.5
    Z[:, :, 1] *= 2.5
    noise = np.full_like(Z, 0.05)
    units = {
        "unitsT_lbl": "Delay",
        "unitsT_ltx": "ps",
        "unitsL_lbl": "Wavenumber",
        "unitsL_ltx": r"$\mathrm{cm^{-1}}$",
        "unitsZ_lbl": r"$\Delta$A",
        "unitsZ_ltx": "mOD",
    }
    return Dataset1D(
        Zavg_R=Z,
        delays=delays,
        probe=probe,
        units=units,
        nscans=5,
        Zstdv=noise,
        source="mock_multidet",
        data_type="RAL_IRpp",
    )


@pytest.fixture
def sample_1d():
    """Create a standard 1D dataset."""
    delays = np.array([-5.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
    probe = np.linspace(1900, 2100, 20)
    Z = np.sin(delays[:, None, None] / 2.0) * np.cos(probe[None, :, None] / 100.0)
    units = {
        "unitsT_lbl": "Delay",
        "unitsT_ltx": "ps",
        "unitsL_lbl": "Wavenumber",
        "unitsL_ltx": r"$\mathrm{cm^{-1}}$",
        "unitsZ_lbl": r"$\Delta$A",
        "unitsZ_ltx": "mOD",
    }
    return Dataset1D(
        Zavg_R=Z,
        delays=delays,
        probe=probe,
        units=units,
        nscans=10,
        source="mock_data",
        data_type="PDAT",
    )


def test_calculate_anisotropy_multidet(sample_1d_multidet):
    # sample_1d_multidet has det 0 = 1.5 and det 1 = 2.5
    # r = (1.5 - 2.5) / (1.5 + 2 * 2.5) = -1.0 / 6.5 = -0.153846...
    r_ds = sample_1d_multidet.calculate_anisotropy()
    assert isinstance(r_ds, Dataset1D)
    assert r_ds.data_type == "Anisotropy"
    assert r_ds.units["unitsZ_lbl"] == "r"
    assert np.allclose(r_ds.Z[:, :, 0], -1.0 / 6.5)

    # Test error propagation
    assert r_ds.noise_array() is not None


def test_calculate_anisotropy_two_datasets(sample_1d):
    par = sample_1d.copy()
    perp = sample_1d.copy()
    par.Zavg_R = np.full_like(par.Zavg_R, 10.0)
    par.Zavg_C = None
    perp.Zavg_R = np.full_like(perp.Zavg_R, 5.0)
    perp.Zavg_C = None

    # r = (10 - 1*5) / (10 + 2*5) = 5 / 20 = 0.25
    r_ds = pm.calculate_anisotropy(par, perp, g_factor=1.0)
    assert np.allclose(r_ds.Z[:, :, 0], 0.25)

    # With G = 1.5: r = (10 - 1.5*5) / (10 + 2*1.5*5) = 2.5 / 25 = 0.1
    r_g = pm.calculate_anisotropy(par, perp, g_factor=1.5)
    assert np.allclose(r_g.Z[:, :, 0], 0.1)

    # Test threshold masking: denominator is 20. With threshold=25, all should be NaN
    r_th = pm.calculate_anisotropy(par, perp, threshold=25.0)
    assert np.all(np.isnan(r_th.Z[:, :, 0]))

    # Test return_components
    comps = pm.calculate_anisotropy(par, perp, return_components=True)
    assert "anisotropy" in comps
    assert "magic_angle" in comps
    assert "difference" in comps
    assert "parallel" in comps
    assert "perpendicular" in comps

    # Magic angle = (10 + 2*5)/3 = 20/3
    assert np.allclose(comps["magic_angle"].Z[:, :, 0], 20.0 / 3.0)
    # Difference = 10 - 5 = 5
    assert np.allclose(comps["difference"].Z[:, :, 0], 5.0)


# --------------------------------------------------------------------------- #
# Singular Value Decomposition (SVD)                                          #
# --------------------------------------------------------------------------- #
@pytest.fixture
def synthetic_svd_data():
    """Create a synthetic 2-component transient absorption dataset."""
    rng = np.random.default_rng(12345)
    t = np.linspace(-2, 50, 45)
    p = np.linspace(420, 680, 50)

    kin1 = np.where(t > 0, np.exp(-t / 6.0), 0.0)
    kin2 = np.where(t > 0, (1 - np.exp(-t / 6.0)) * np.exp(-t / 25.0), 0.0)

    spec1 = np.exp(-((p - 510) / 25.0) ** 2)
    spec2 = -0.8 * np.exp(-((p - 600) / 35.0) ** 2)

    pure = np.outer(kin1, spec1) + np.outer(kin2, spec2)
    noise = 0.003 * rng.standard_normal(pure.shape)
    z = pure + noise

    units = {
        "unitsT_lbl": "Delay",
        "unitsT_ltx": "ps",
        "unitsL_lbl": "Wavelength",
        "unitsL_ltx": "nm",
        "unitsZ_lbl": r"$\Delta$A",
        "unitsZ_ltx": "mOD",
    }
    ds = pm.Dataset1D(Zavg_R=z, delays=t, probe=p, units=units)
    return ds, pure, z, t, p


def test_svd_basic(synthetic_svd_data):
    """Test basic SVD decomposition and variance properties."""
    ds, pure, z, t, p = synthetic_svd_data
    res = ds.svd(k=2)

    assert isinstance(res, SVDResult)
    assert res.n_components == min(len(t), len(p))
    assert res.k == 2
    assert len(res.S) == res.n_components
    # First 2 singular values should be dominant
    assert res.S[0] > res.S[1] > res.S[2] * 10
    # Cumulative variance for 2 components should exceed 99%
    assert res.variance_at(2) > 0.99
    assert np.all(np.diff(res.cumulative_variance) >= 0)
    assert np.isclose(res.cumulative_variance[-1], 1.0)


def test_svd_recompose_and_residuals(synthetic_svd_data):
    """Test rank-k recomposition and residual matrices."""
    ds, pure, z, t, p = synthetic_svd_data
    res = ds.svd(k=2)

    recon = res.recompose(k=2)
    resid = res.residuals(k=2)

    assert recon.shape == z.shape
    assert resid.shape == z.shape
    np.testing.assert_allclose(recon + resid, z, rtol=1e-10, atol=1e-12)

    # Recomposition with all components should reproduce input matrix exactly
    full_recon = res.recompose(k=res.n_components)
    np.testing.assert_allclose(full_recon, z, rtol=1e-10, atol=1e-12)


def test_svd_components_access(synthetic_svd_data):
    """Test temporal and spectral component extractors."""
    ds, pure, z, t, p = synthetic_svd_data
    res = ds.svd(k=2)

    u0_scaled = res.component_time(0, scaled=True)
    u0_raw = res.component_time(0, scaled=False)
    np.testing.assert_allclose(u0_scaled, u0_raw * res.S[0])

    v0 = res.component_probe(0)
    assert len(v0) == len(p)
    assert len(u0_raw) == len(t)

    with pytest.raises(IndexError):
        res.component_time(-1)
    with pytest.raises(IndexError):
        res.component_probe(999)


def test_svd_cropping(synthetic_svd_data):
    """Test cropping time and probe limits prior to SVD."""
    ds, pure, z, t, p = synthetic_svd_data
    tmin, tmax = 0.5, 30.0
    pmin, pmax = 480.0, 620.0

    res = ds.svd(k=2, tmin=tmin, tmax=tmax, probe_min=pmin, probe_max=pmax)

    assert np.all(res.delays >= tmin)
    assert np.all(res.delays <= tmax)
    assert np.all(res.probe >= pmin)
    assert np.all(res.probe <= pmax)
    assert res.raw_matrix.shape == (len(res.delays), len(res.probe))
    assert res.recompose().shape == (len(res.delays), len(res.probe))


def test_svd_centering_modes(synthetic_svd_data):
    """Test all centering modes."""
    _, _, z, t, p = synthetic_svd_data

    for mode in ["none", "spectrum", "kinetic", "both"]:
        res = compute_svd(z, t, p, k=2, centering=mode)
        recon = res.recompose(k=res.n_components)
        np.testing.assert_allclose(recon, z, rtol=1e-7, atol=1e-7)

    with pytest.raises(ValueError, match="Unknown centering"):
        compute_svd(z, t, p, centering="invalid_mode")


def test_svd_nan_handling(synthetic_svd_data):
    """Test handling of NaN values via interpolation and drop."""
    _, _, z, t, p = synthetic_svd_data
    z_nan = np.copy(z)
    z_nan[10, 15] = np.nan
    z_nan[11, 15:18] = np.nan

    # 1. interpolate mode (default)
    res_interp = compute_svd(z_nan, t, p, k=2, handle_nan="interpolate")
    assert not np.any(np.isnan(res_interp.raw_matrix))
    assert res_interp.raw_matrix.shape == z.shape

    # 2. raise mode
    with pytest.raises(ValueError, match="Data matrix contains NaN"):
        compute_svd(z_nan, t, p, k=2, handle_nan="raise")

    # 3. drop mode
    res_drop = compute_svd(z_nan, t, p, k=2, handle_nan="drop")
    assert not np.any(np.isnan(res_drop.raw_matrix))
    assert res_drop.raw_matrix.shape[0] < z.shape[0]
    assert res_drop.raw_matrix.shape[1] < z.shape[1]


def test_svd_to_dataset(synthetic_svd_data):
    """Test exporting SVD recomposition to Dataset1D."""
    ds, _, _, _, _ = synthetic_svd_data
    res = ds.svd(k=2)

    new_ds = res.to_dataset(k=2, name="TestRecomposed")
    assert isinstance(new_ds, pm.Dataset1D)
    assert new_ds.source == "TestRecomposed"
    assert new_ds.Zavg_R.shape == ds.Zavg_R.shape
    assert new_ds.Zavg_C.shape == ds.Zavg_R.shape
    np.testing.assert_allclose(new_ds.delays, ds.delays)
    np.testing.assert_allclose(new_ds.probe, ds.probe)


def test_svd_export_csv_and_dataframe(synthetic_svd_data, tmp_path):
    """Test CSV export and DataFrame conversion."""
    ds, _, _, _, _ = synthetic_svd_data
    res = ds.svd(k=2)

    csv_file = tmp_path / "svd_output.csv"
    res.export_csv(csv_file, k=2)
    assert csv_file.exists()
    content = csv_file.read_text(encoding="utf-8")
    assert "SVD Summary & Scree Plot" in content
    assert "Temporal Components" in content
    assert "Spectral Components" in content

    dfs = res.to_dataframe(k=2)
    assert "scree" in dfs
    assert "temporal" in dfs
    assert "spectral" in dfs
    assert "recomposed" in dfs
    assert "residuals" in dfs
    assert len(dfs["scree"]) == res.n_components


def test_svd_multi_detector(synthetic_svd_data):
    """Test SVD on multi-detector dataset."""
    _, _, z, t, p = synthetic_svd_data
    z_multi = np.stack([z, z * 0.5], axis=2)  # [delays, probe, 2]

    ds_multi = pm.Dataset1D(Zavg_R=z_multi, delays=t, probe=p, units={})
    assert ds_multi.n_detectors == 2

    res0 = ds_multi.svd(detector=0, k=2)
    res1 = ds_multi.svd(detector=1, k=2)

    np.testing.assert_allclose(res0.S, res1.S * 2.0, rtol=1e-5)


# --------------------------------------------------------------------------- #
# Wavelength & Timescale Stitching                                            #
# --------------------------------------------------------------------------- #
@pytest.fixture
def uv_vis_nir_datasets():
    """Create a pair of synthetic UV/Vis and NIR datasets with an overlapping spectral window."""
    rng = np.random.default_rng(100)
    t = np.linspace(-1, 50, 40)  # delays in ps

    # UV/Vis probe: 350 to 750 nm
    p_uv = np.linspace(350, 750, 50)
    # NIR probe: 700 to 1200 nm (overlap 700 - 750 nm)
    p_nir = np.linspace(700, 1200, 60)

    kin = np.where(t > 0, np.exp(-t / 12.0), 0.0)

    # Underlying physical spectrum: peak at 500 nm and peak at 900 nm
    spec_uv = np.exp(-((p_uv - 500) / 60.0) ** 2) + 0.3 * np.exp(-((p_uv - 720) / 40.0) ** 2)
    spec_nir_true = 0.3 * np.exp(-((p_nir - 720) / 40.0) ** 2) + 0.8 * np.exp(-((p_nir - 900) / 80.0) ** 2)

    # In experiment, NIR detector has lower sensitivity -> scaled by 0.5
    scale_factor_true = 2.0  # NIR * 2.0 = UV
    spec_nir_measured = spec_nir_true / scale_factor_true

    z_uv = np.outer(kin, spec_uv) + 0.001 * rng.standard_normal((len(t), len(p_uv)))
    z_nir = np.outer(kin, spec_nir_measured) + 0.001 * rng.standard_normal((len(t), len(p_nir)))

    units = {
        "unitsT_lbl": "Delay",
        "unitsT_ltx": "ps",
        "unitsL_lbl": "Wavelength",
        "unitsL_ltx": "nm",
        "unitsZ_lbl": r"$\Delta$A",
        "unitsZ_ltx": "mOD",
    }
    ds_uv = pm.Dataset1D(Zavg_R=z_uv, delays=t, probe=p_uv, units=units)
    ds_nir = pm.Dataset1D(Zavg_R=z_nir, delays=t, probe=p_nir, units=units)
    return ds_uv, ds_nir, scale_factor_true


@pytest.fixture
def fs_ns_datasets():
    """Create a pair of synthetic fs (ps) and ns (ns) datasets."""
    rng = np.random.default_rng(200)

    # fs delays: -2 to 1400 ps (i.e. -0.002 to 1.4 ns)
    t_fs = np.linspace(-2, 1400, 50)
    # ns delays: -10 to 500 ns
    t_ns = np.concatenate([np.linspace(-10, 0, 5), np.geomspace(0.1, 500, 45)])

    # Common probe
    p = np.linspace(400, 700, 40)

    # 2-exponential decay: tau1 = 0.5 ns (500 ps), tau2 = 50 ns
    kin_fs = np.where(t_fs > 0, 0.6 * np.exp(-t_fs / 500.0) + 0.4 * np.exp(-t_fs / 50000.0), 0.0)
    kin_ns_true = np.where(t_ns > 0, 0.6 * np.exp(-t_ns / 0.5) + 0.4 * np.exp(-t_ns / 50.0), 0.0)

    # Probe shape
    spec = np.exp(-((p - 550) / 50.0) ** 2)

    # fs signal scaled down or up relative to ns
    scale_true = 2.5
    z_fs = np.outer(kin_fs, spec) + 0.001 * rng.standard_normal((len(t_fs), len(p)))
    z_ns = np.outer(kin_ns_true * scale_true, spec) + 0.001 * rng.standard_normal((len(t_ns), len(p)))

    units_fs = {
        "unitsT_lbl": "Delay",
        "unitsT_ltx": "ps",
        "unitsL_lbl": "Wavelength",
        "unitsL_ltx": "nm",
        "unitsZ_lbl": r"$\Delta$A",
        "unitsZ_ltx": "mOD",
    }
    units_ns = {
        "unitsT_lbl": "Delay",
        "unitsT_ltx": "ns",
        "unitsL_lbl": "Wavelength",
        "unitsL_ltx": "nm",
        "unitsZ_lbl": r"$\Delta$A",
        "unitsZ_ltx": "mOD",
    }
    ds_fs = pm.Dataset1D(Zavg_R=z_fs, delays=t_fs, probe=p, units=units_fs)
    ds_ns = pm.Dataset1D(Zavg_R=z_ns, delays=t_ns, probe=p, units=units_ns)
    return ds_fs, ds_ns, scale_true


# =========================================================================
# Wavelength Stitching Tests
# =========================================================================


def test_stitch_wavelength_blend(uv_vis_nir_datasets):
    """Test wavelength stitching in blend mode with automatic scale fitting."""
    ds_uv, ds_nir, scale_true = uv_vis_nir_datasets
    res = stitch_wavelength(ds_uv, ds_nir, mode="blend", stitch_region=(710, 740))

    assert isinstance(res, StitchResult)
    assert res.kind == "wavelength"
    assert isinstance(res.scaling_factor, (float, np.floating))
    assert res.scaling_factor > 0

    # Output grid should span from min of UV to max of NIR
    assert np.isclose(res.probe[0], 350, atol=2)
    assert np.isclose(res.probe[-1], 1200, atol=2)
    assert len(res.delays) == len(ds_nir.delays)
    assert res.Z.shape == (len(res.delays), len(res.probe))

    # Verify to_dataset()
    ds_stitched = res.to_dataset()
    assert isinstance(ds_stitched, pm.Dataset1D)
    assert ds_stitched.Z.shape == (len(res.delays), len(res.probe))


def test_stitch_wavelength_cutover(uv_vis_nir_datasets):
    """Test wavelength stitching in cutover mode with manual scale factor."""
    ds_uv, ds_nir, _ = uv_vis_nir_datasets
    res = stitch_wavelength(ds_uv, ds_nir, mode="cutover", cutover_wl=725.0, scale_override=2.0)

    assert res.scaling_factor == 2.0
    assert res.cutover_val == 725.0
    # Probe grid should be monotonic
    assert np.all(np.diff(res.probe) > 0)
    assert np.min(np.diff(res.probe)) > 0


def test_stitch_wavelength_reverse_order(uv_vis_nir_datasets):
    """Test passing NIR as first argument and UV as second argument (should auto-sort)."""
    ds_uv, ds_nir, _ = uv_vis_nir_datasets
    res = stitch_wavelength(ds_nir, ds_uv, mode="blend")
    assert res.probe[0] < res.probe[-1]
    assert np.isclose(res.probe[0], 350, atol=2)


def test_stitch_wavelength_no_overlap():
    """Test error when datasets have no spectral overlap."""
    t = np.linspace(0, 10, 5)
    dummy_units = {"unitsT_lbl": "T", "unitsT_ltx": "ps", "unitsL_lbl": "L", "unitsL_ltx": "nm"}
    ds1 = pm.Dataset1D(Zavg_R=np.ones((5, 10)), delays=t, probe=np.linspace(300, 400, 10), units=dummy_units)
    ds2 = pm.Dataset1D(Zavg_R=np.ones((5, 10)), delays=t, probe=np.linspace(600, 700, 10), units=dummy_units)

    with pytest.raises(ValueError, match="[Nn]o spectral overlap"):
        stitch_wavelength(ds1, ds2)


def test_stitch_wavelength_dataset_method(uv_vis_nir_datasets):
    """Test calling ds.stitch_wavelength(other)."""
    ds_uv, ds_nir, _ = uv_vis_nir_datasets
    res = ds_uv.stitch_wavelength(ds_nir)
    assert isinstance(res, StitchResult)


# =========================================================================
# Timescale Stitching Tests
# =========================================================================


def test_stitch_time_scalar(fs_ns_datasets):
    """Test timescale stitching with scalar scale model and ps-to-ns unit harmonization."""
    ds_fs, ds_ns, _ = fs_ns_datasets
    res = stitch_time(
        ds_fs,
        ds_ns,
        time_overlap=(0.5, 1.2),
        blend_window=(0.6, 1.0),
        scale_model="scalar",
    )

    assert isinstance(res, StitchResult)
    assert res.kind == "timescale"
    assert res.dataset.units["unitsT_ltx"] == "ns"
    assert isinstance(res.scaling_factor, (float, np.floating))

    # Delays should be strictly monotonic in ns
    assert np.all(np.diff(res.delays) > 0)
    # Delays should cover from < 0 ns up to 500 ns
    assert res.delays[0] < 0
    assert res.delays[-1] >= 450

    # Dataset conversion
    ds_stitched = res.to_dataset()
    assert isinstance(ds_stitched, pm.Dataset1D)
    assert ds_stitched.units["unitsT_ltx"] == "ns"


def test_stitch_time_quadratic(fs_ns_datasets):
    """Test timescale stitching with quadratic scaling model."""
    ds_fs, ds_ns, _ = fs_ns_datasets
    res = stitch_time(
        ds_fs,
        ds_ns,
        time_overlap=(0.5, 1.2),
        blend_window=(0.6, 1.0),
        scale_model="quadratic",
    )
    assert res.kind == "timescale"
    assert "poly_coeffs" in res.scaling_params
    assert len(res.scaling_params["poly_coeffs"]) == 3


def test_stitch_time_wavelength_calibration():
    """Test _optimize_linear_wl_shift helper."""
    wl_ref = np.linspace(400, 700, 100)
    spec_ref = np.exp(-((wl_ref - 520) / 30.0) ** 2)

    # Shifted moving spectrum: wl_moving = (wl_ref - 3) / 1.01
    wl_moving = np.linspace(390, 710, 110)
    spec_moving = np.exp(-((1.01 * wl_moving + 3.0 - 520) / 30.0) ** 2)

    m, b = _optimize_linear_wl_shift(wl_ref, spec_ref, wl_moving, spec_moving)
    assert np.isclose(m, 1.01, atol=0.05)
    assert np.isclose(b, 3.0, atol=2.0)


def test_stitch_time_scatter_exclusion(fs_ns_datasets):
    """Test timescale stitching with laser scatter window excluded."""
    ds_fs, ds_ns, _ = fs_ns_datasets
    res = stitch_time(
        ds_fs,
        ds_ns,
        time_overlap=(0.4, 1.3),
        cut_scatter=True,
        scatter_region=(520.0, 540.0),
    )
    assert isinstance(res, StitchResult)


def test_stitch_time_dataset_method(fs_ns_datasets):
    """Test calling ds.stitch_time(other)."""
    ds_fs, ds_ns, _ = fs_ns_datasets
    res = ds_fs.stitch_time(ds_ns, time_overlap=(0.5, 1.2))
    assert isinstance(res, StitchResult)


def test_stitch_result_export(uv_vis_nir_datasets, tmp_path):
    """Test exporting StitchResult to .pdat and .csv files."""
    ds_uv, ds_nir, _ = uv_vis_nir_datasets
    res = stitch_wavelength(ds_uv, ds_nir)

    pdat_file = tmp_path / "stitched.pdat"
    res.export(pdat_file, file_format="pdat")
    assert pdat_file.exists()
    assert pdat_file.stat().st_size > 0

    csv_file = tmp_path / "stitched.csv"
    res.export(csv_file, file_format="csv")
    assert csv_file.exists()
    assert csv_file.stat().st_size > 0


@pytest.mark.parametrize("method", ["cubic", "quintic", "cosine", "linear", "step"])
def test_blend_methods_wavelength(uv_vis_nir_datasets, method):
    """Test all blend methods in wavelength stitching."""
    ds_uv, ds_nir, _ = uv_vis_nir_datasets
    res = stitch_wavelength(ds_uv, ds_nir, mode="blend", blend_method=method)
    assert res.blend_method == method
    assert np.all(np.isfinite(res.Z))
    assert res.probe[0] <= 350
    assert res.probe[-1] >= 1100


@pytest.mark.parametrize("method", ["cubic", "quintic", "cosine", "linear", "step"])
def test_blend_methods_timescale(fs_ns_datasets, method):
    """Test all blend methods in timescale stitching."""
    ds_fs, ds_ns, _ = fs_ns_datasets
    res = stitch_time(ds_fs, ds_ns, blend_method=method)
    assert res.blend_method == method
    assert np.all(np.isfinite(res.Z))
    assert res.delays[0] <= 0
    assert res.delays[-1] >= 300


# --------------------------------------------------------------------------- #
#                     Scan Selection and Recalc Average                       #
# --------------------------------------------------------------------------- #
def test_parse_scan_selection():
    from pymorgan.oneD.load import parse_scan_selection

    assert parse_scan_selection("1-3, 5", 6) == [0, 1, 2, 4]
    assert parse_scan_selection("4 2 1", 5) == [0, 1, 3]
    assert parse_scan_selection("3-1", 5) == [0, 1, 2]

    with pytest.raises(ValueError, match="Scan numbers must be between"):
        parse_scan_selection("0 2", 5)
    with pytest.raises(ValueError, match="Scan numbers must be between"):
        parse_scan_selection("1 6", 5)
    with pytest.raises(ValueError, match="Invalid token"):
        parse_scan_selection("abc", 5)


def test_mess_recalc_average_weighted_vs_direct(tmp_path):
    from pymorgan.oneD.load import mess_recalc_average

    run_dir = tmp_path / "test_mess"
    temp_dir = run_dir / "temp"
    temp_dir.mkdir(parents=True)

    delays = np.array([-1.0, 0.5, 10.0])
    np.savetxt(run_dir / "test_mess_delays.csv", delays, delimiter=",")

    # Scan 1: sig = 10, noise = 1
    # Scan 2: sig = 20, noise = 3
    shape = (len(delays), 4)
    s1 = np.full(shape, 10.0)
    n1 = np.full(shape, 1.0)
    s2 = np.full(shape, 20.0)
    n2 = np.full(shape, 3.0)

    np.savetxt(temp_dir / "test_mess_signal_sp0_sm0_du0_1.csv", s1, delimiter=",")
    np.savetxt(temp_dir / "test_mess_signal_noise_sp0_sm0_du0_1.csv", n1, delimiter=",")
    np.savetxt(temp_dir / "test_mess_signal_sp0_sm0_du0_2.csv", s2, delimiter=",")
    np.savetxt(temp_dir / "test_mess_signal_noise_sp0_sm0_du0_2.csv", n2, delimiter=",")

    # 1. Noise-weighted average
    # w1 = 1, w2 = 1/9; avg = (10*1 + 20/9)/(10/9) = 11.0; comb = sqrt(9/10) ~ 0.94868
    avg_w, std_w = mess_recalc_average(str(run_dir), [0, 1], noise_weighted=True)
    assert np.allclose(avg_w[:, :, 0], 11.0, atol=1e-5)
    assert np.allclose(std_w[:, :, 0], np.sqrt(0.9), atol=1e-5)

    # 2. Direct arithmetic average
    # avg = (10 + 20) / 2 = 15.0; comb = sqrt(1^2 + 3^2) / 2 = sqrt(10) / 2 ~ 1.58114
    avg_d, std_d = mess_recalc_average(str(run_dir), [0, 1], noise_weighted=False)
    assert np.allclose(avg_d[:, :, 0], 15.0, atol=1e-5)
    assert np.allclose(std_d[:, :, 0], np.sqrt(10.0) / 2.0, atol=1e-5)

    # 3. Single scan selection gives identical result for both modes
    avg_s1_w, std_s1_w = mess_recalc_average(str(run_dir), [0], noise_weighted=True)
    avg_s1_d, std_s1_d = mess_recalc_average(str(run_dir), [0], noise_weighted=False)
    assert np.allclose(avg_s1_w, avg_s1_d)
    assert np.allclose(avg_s1_w[:, :, 0], 10.0)
    assert np.allclose(std_s1_w[:, :, 0], 1.0)


def test_dataset_recalc_average_single_scans(dataset):
    # Give dataset mock single scans: [Ndelays, Npixels, 1, 3]
    delays = dataset.delays
    n_delays = len(delays)
    n_pix = dataset.probe.size
    zss = np.zeros((n_delays, n_pix, 1, 3))
    zss[:, :, 0, 0] = 5.0
    zss[:, :, 0, 1] = 15.0
    zss[:, :, 0, 2] = 25.0
    dataset.Zss_R = zss

    # Direct average over scans 0 and 2: (5 + 25) / 2 = 15.0
    dataset.recalc_average([0, 2], noise_weighted=False)
    assert dataset.nscans == 2
    assert np.allclose(dataset.Zavg_R[:, :, 0], 15.0)


