"""Tests for multi-band 1-D scatter & dynamic background removal and its GUI."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox

import pymorgan as pm
from pymorgan.gui.main_window import MainWindow
from pymorgan.gui.scatter_dialog import ScatterRemovalDialog
from pymorgan.oneD.scatter import (
    ScatterBand,
    ScatterRemovalResult,
    compute_scatter_removal,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_scatter_dataset(
    *,
    n_scans: int = 4,
    descending_cm1: bool = False,
    with_harmonic: bool = True,
    harmonic_amp: float = 2.0,
    with_nan_gap: bool = False,
    two_d: bool = False,
):
    """Build a synthetic TA dataset with dimming + walking scatter, fluorescence & GSB."""
    rng = np.random.default_rng(123)
    t = np.concatenate([np.linspace(-5.0, -1.0, 6), np.linspace(-0.8, 2.0, 15), np.geomspace(2.5, 1800.0, 24)])
    wl = np.linspace(330.0, 740.0, 200)
    probe = (1e7 / wl)[::-1] if descending_cm1 else wl.copy()

    # True underlying TA signal: smooth GSB around 340..420nm + ESA at 480nm
    kin_gsb = np.where(t > 0.0, -4.0 * np.exp(-t / 120.0) - 1.5, 0.0)
    # Coherent artifact near t0 across all wavelengths (chirped slightly)
    t0_wl = 0.0 + 0.002 * (wl - 355.0)
    ca = 6.0 * (-(t[:, None] - t0_wl[None, :]) / 0.15) * np.exp(-(((t[:, None] - t0_wl[None, :]) / 0.15) ** 2))
    spec_gsb = np.exp(-((wl - 365.0) / 38.0) ** 2)
    spec_esa = 2.5 * np.exp(-((wl - 480.0) / 30.0) ** 2)
    Z_true_base = np.outer(kin_gsb, spec_gsb) + np.outer(np.where(t > 0.0, np.exp(-t / 50.0), 0.0), spec_esa) + ca

    # Broadband fluorescence background that dims slowly along with stage travel
    fl_spec = -0.6 * np.exp(-((wl - 580.0) / 90.0) ** 2) - 0.8 * np.exp(-((wl - 325.0) / 18.0) ** 2)

    # Scatter trajectory: dims from 1.0 -> 0.55 and shifts by +0.4 nm across delay
    stage_amp = 1.0 - 0.45 * (np.maximum(t, 0.0) / 1800.0)
    stage_shift = 0.45 * (np.maximum(t, 0.0) / 1800.0)

    if two_d:
        amp_t = stage_amp * (1.0 + 0.03 * rng.standard_normal(t.size))
        cen = 355.0 + stage_shift
        sc1 = -130.0 * amp_t[:, None] * np.exp(-0.5 * ((wl[None, :] - cen[:, None]) / 2.2) ** 2)
        sc2 = -harmonic_amp * amp_t[:, None] * np.exp(-0.5 * ((wl[None, :] - 710.0) / 2.8) ** 2) if with_harmonic else 0.0
        Z = Z_true_base + sc1 + sc2 + stage_amp[:, None] * fl_spec[None, :] + 0.08 * rng.standard_normal((t.size, wl.size))
        if descending_cm1:
            Z = Z[:, ::-1]
        units = {
            "unitsT_lbl": "Delay",
            "unitsT_ltx": "ps",
            "unitsL_lbl": "Wavenumber" if descending_cm1 else "Wavelength",
            "unitsL_ltx": r"$\mathrm{cm^{-1}}$" if descending_cm1 else "nm",
            "unitsZ_lbl": r"$\Delta$A",
            "unitsZ_ltx": "mOD",
        }
        return pm.Dataset1D(Zavg_R=Z, delays=t, probe=probe, units=units), Z_true_base

    Zss = np.zeros((t.size, wl.size, 1, n_scans), dtype=float)
    for s in range(n_scans):
        scan_scale = 0.90 + 0.08 * s
        amp_t = scan_scale * stage_amp * (1.0 + 0.04 * rng.standard_normal(t.size))
        cen = 355.0 + 0.08 * s + stage_shift
        sc1 = -130.0 * amp_t[:, None] * np.exp(-0.5 * ((wl[None, :] - cen[:, None]) / 2.2) ** 2)
        sc2 = -harmonic_amp * amp_t[:, None] * np.exp(-0.5 * ((wl[None, :] - 710.0) / 2.8) ** 2) if with_harmonic else 0.0
        noise = 0.08 * rng.standard_normal((t.size, wl.size))
        Z_s = Z_true_base + sc1 + sc2 + scan_scale * stage_amp[:, None] * fl_spec[None, :] + noise
        if descending_cm1:
            Z_s = Z_s[:, ::-1]
        if with_nan_gap:
            Z_s[:, 120:124] = np.nan
        Zss[:, :, 0, s] = Z_s

    Zavg = np.mean(Zss, axis=-1)
    Zstdv = np.std(Zss, axis=-1, ddof=1)
    if descending_cm1:
        Z_true_out = Z_true_base[:, ::-1]
    else:
        Z_true_out = Z_true_base

    units = {
        "unitsT_lbl": "Delay",
        "unitsT_ltx": "ps",
        "unitsL_lbl": "Wavenumber" if descending_cm1 else "Wavelength",
        "unitsL_ltx": r"$\mathrm{cm^{-1}}$" if descending_cm1 else "nm",
        "unitsZ_lbl": r"$\Delta$A",
        "unitsZ_ltx": "mOD",
        "pymess_metadata": {"pump_wl_nm": 355.0} if not descending_cm1 else {},
    }
    ds = pm.Dataset1D(
        Zavg_R=Zavg,
        delays=t,
        probe=probe,
        units=units,
        nscans=n_scans,
        Zss_R=Zss,
        Zstdv=Zstdv,
        source="synthetic_scatter.h5",
    )
    return ds, Z_true_out


def test_detect_scatter_bands_primary_and_harmonic():
    # 1. Strong 2nd-order harmonic (> 0.5 mOD floor) detected by Pass 1 and auto-enabled
    ds, _ = _make_scatter_dataset(with_harmonic=True, harmonic_amp=2.0)
    bands = ds.detect_scatter_bands()
    assert len(bands) >= 2
    primary = bands[0]
    assert primary.enabled is True
    assert primary.p_min <= 355.0 <= primary.p_max

    harm = [b for b in bands if b.p_min <= 710.0 <= b.p_max]
    assert len(harm) == 1
    assert harm[0].enabled is True

    # 2. Moderate Pass-2 harmonic (>= 5 sigma, < 0.5 mOD floor) auto-enabled & coupled to primary
    ds_mod, _ = _make_scatter_dataset(with_harmonic=True, harmonic_amp=0.45)
    bands_mod = ds_mod.detect_scatter_bands()
    harm_mod = [b for b in bands_mod if b.p_min <= 710.0 <= b.p_max]
    assert len(harm_mod) == 1
    assert harm_mod[0].enabled is True
    assert harm_mod[0].couple_to == 0
    assert harm_mod[0].rank == 1

    # 3. Borderline Pass-2 harmonic (3..5 sigma) suggested disabled & coupled to primary
    ds_weak, _ = _make_scatter_dataset(with_harmonic=True, harmonic_amp=0.07)
    bands_weak = ds_weak.detect_scatter_bands()
    harm_weak = [b for b in bands_weak if b.p_min <= 710.0 <= b.p_max]
    assert len(harm_weak) == 1
    assert harm_weak[0].enabled is False
    assert harm_weak[0].couple_to == 0


def test_detect_scatter_bands_descending_cm1_and_positive():
    ds, _ = _make_scatter_dataset(descending_cm1=True, with_harmonic=False)
    bands = ds.detect_scatter_bands()
    assert len(bands) >= 1
    pump_cm1 = 1e7 / 355.0
    assert any(b.p_min <= pump_cm1 <= b.p_max for b in bands)

    # Positive scatter artifact detection only triggers when allow_positive=True
    ds_pos = ds.copy()
    ds_pos.Zavg_R = -ds_pos.Zavg_R
    assert ds_pos.detect_scatter_bands(allow_positive=False) == []
    assert len(ds_pos.detect_scatter_bands(allow_positive=True)) >= 1


@pytest.mark.parametrize("method", ["local_svd", "taylor", "warped"])
def test_scatter_removal_methods_and_preserves_signal(method):
    ds, Z_true = _make_scatter_dataset(with_harmonic=True, harmonic_amp=2.0)
    orig_std = ds.Zstdv.copy()
    bands = ds.detect_scatter_bands()

    res = compute_scatter_removal(
        ds,
        bands=bands,
        method=method,
        rank=3,
        per_scan=True,
        update_noise=True,
        scale_broadband=True,
    )
    assert isinstance(res, ScatterRemovalResult)
    assert res.Zavg_C.shape == ds.Zavg_R.shape
    assert res.Zss_C is not None and res.Zss_C.shape == ds.Zss_R.shape
    assert res.Zstdv is not None and res.Zstdv.shape == orig_std.shape

    m = res.metrics
    assert m["used_per_scan"] is True
    assert m["pre_zero_rms_after"] < m["pre_zero_rms_before"] * 0.2
    assert m["pos_roughness_after"] < m["pos_roughness_before"] * 0.5
    assert m["interscan_std_after"] < m["interscan_std_before"]

    # Check underlying GSB at 355 nm for t > 5 ps is faithfully recovered
    wl = ds.probe
    t = ds.delays
    i355 = int(np.argmin(np.abs(wl - 355.0)))
    assert np.median(res.Zstdv[:, i355, 0]) < np.median(orig_std[:, i355, 0]) * 0.25
    late = t >= 10.0
    err = np.sqrt(np.mean((res.Zavg_C[late, i355, 0] - Z_true[late, i355]) ** 2))
    assert err < 0.6, f"{method} residual error at 355 nm too large: {err:.3f}"

    # Check coherent artifact around t0 at 355 nm is preserved (peak-to-peak > 5 mOD)
    t0_win = (t >= -0.4) & (t <= 0.4)
    assert np.ptp(res.Zavg_C[t0_win, i355, 0]) > 5.0


def test_dataset1d_remove_copy_and_clear_restores_noise():
    ds, _ = _make_scatter_dataset(with_harmonic=False)
    orig_std = ds.Zstdv.copy()

    ret = ds.remove_scatter(method="local_svd", per_scan=True, update_noise=True)
    assert ret is ds
    assert ds._scatter_removal_result is not None
    assert ds._scatter_removal_params is not None
    assert not np.allclose(ds.Zstdv, orig_std)

    ds_copy = ds.copy()
    assert ds_copy._scatter_removal_result is not None
    assert np.allclose(ds_copy.Zavg_C, ds.Zavg_C)

    ds.clear_scatter_removal()
    assert ds._scatter_removal_result is None
    assert ds._scatter_removal_params is None
    assert np.allclose(ds.Zstdv, orig_std)


def test_scatter_edge_cases_overlapping_nan_2d_transmittance_saturated():
    # 1. Overlapping bands + NaN gap + transmittance_space + mask_saturated
    ds, _ = _make_scatter_dataset(with_nan_gap=True, with_harmonic=False)
    overlapping = [
        ScatterBand(348.0, 357.0, enabled=True, label="Part A"),
        ScatterBand(354.0, 362.0, enabled=True, label="Part B"),
    ]
    res = compute_scatter_removal(
        ds,
        bands=overlapping,
        transmittance_space=True,
        mask_saturated=True,
        saturation_threshold=100.0,
    )
    # NaN gap stays NaN and does not leak into clean neighbors
    assert np.all(np.isnan(res.Zavg_C[:, 120:124, 0]))
    assert np.all(np.isfinite(res.Zavg_C[:, 130:150, 0]))
    assert res.metrics["n_saturated_masked"] > 0
    assert res.metrics["pre_zero_rms_after"] < 1.5

    # 2. 2-D Zavg_R compatibility + descending cm-1
    ds2d, _ = _make_scatter_dataset(two_d=True, descending_cm1=True, with_harmonic=False)
    ds2d.remove_scatter()
    assert ds2d.Zavg_C.shape == ds2d.Zavg_R.shape
    assert ds2d.Zss_C is None

    # 3. Invalid method raises
    with pytest.raises(ValueError, match="Unknown scatter removal method"):
        compute_scatter_removal(ds2d, method="invalid_algo")


def test_scatter_dialog_headless(qapp, monkeypatch):
    ds, _ = _make_scatter_dataset(with_harmonic=True, harmonic_amp=2.0)
    dlg = ScatterRemovalDialog(None, ds, detector=0)
    assert dlg.band_table.rowCount() >= 2
    assert dlg.chk_per_scan.isChecked()
    assert dlg.chk_update_noise.isChecked()
    assert dlg.result is not None

    # Switch through all 4 tabs and render each
    for tab_idx in range(4):
        dlg.tabs.setCurrentIndex(tab_idx)
        qapp.processEvents()
        assert dlg._dirty_tabs[tab_idx] is False

    # Test interactive SpanSelector pick callback on Diagnostics tab
    n_before = dlg.band_table.rowCount()
    dlg._on_span_picked(420.0, 430.0)
    assert dlg.band_table.rowCount() == n_before + 1

    # Delete the added band
    dlg.band_table.selectRow(n_before)
    dlg._on_delete_band_clicked()
    assert dlg.band_table.rowCount() == n_before

    # Switch algorithm method + toggles
    for m_val in ("taylor", "warped", "local_svd"):
        idx = dlg.combo_method.findData(m_val)
        dlg.combo_method.setCurrentIndex(idx)
        dlg._recompute_and_render()
        assert dlg.result.params["method"] == m_val

    dlg.chk_asinh.setChecked(False)
    dlg.chk_broadband.setChecked(False)
    dlg._recompute_and_render()

    applied_results = []
    dlg.scatter_applied.connect(applied_results.append)
    dlg._on_apply_clicked()
    assert len(applied_results) == 1

    # Reset signal (mock confirmation Yes)
    reset_calls = []
    dlg.scatter_reset.connect(lambda: reset_calls.append(True))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **kw: QMessageBox.StandardButton.Yes)
    dlg._on_reset_clicked()
    assert reset_calls == [True]
    dlg.close()


def test_main_window_scatter_wiring_and_persistence(qapp):
    win = MainWindow(init_rootdir=False)
    try:
        assert hasattr(win, "actionAdvancedScatterRemoval")
        assert hasattr(win, "PP_advScatter_btn")
        assert win.actionAdvancedScatterRemoval in win.menuPreProcessing.actions()
        assert win.actionAdvancedScatterRemoval.isEnabled() is False

        ds, _ = _make_scatter_dataset(with_harmonic=False)
        win.dataset = ds
        win._on_main_tabs_changed(0)
        assert win.actionAdvancedScatterRemoval.isEnabled() is True

        res = compute_scatter_removal(ds)
        win._on_scatter_applied(res)
        assert win.dataset._scatter_removal_result is res
        assert np.allclose(win.dataset.Zavg_C, res.Zavg_C)

        # Calling _apply_background (as contour/cut plots do) must keep advanced clean result
        win._apply_background()
        assert np.allclose(win.dataset.Zavg_C, res.Zavg_C)

        # Resetting restores static background subtraction
        win._on_scatter_reset()
        assert win.dataset._scatter_removal_result is None
    finally:
        win.close()
