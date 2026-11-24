"""Tests covering PyMORGAN fixes and enhancements from GitHub issue #6:

1. SVD / Stitch / Dataset1D 3-D signal invariant & PDAT export fixes
2. Global GUI excepthook & MainWindow slot error guards + background control skipping
3. 1D & 2D crop backend + Undo/Redo provenance snapshots
4. Interactive CropDialog, scissors toggle buttons (PC_cropBox / twoD_PC_cropBox),
   and Pre-processing -> Crop Dataset menu action
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import synthetic
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

import pymorgan as pm
from pymorgan.gui import app as gui_app
from pymorgan.gui.anisotropy_dialog import AnisotropyDialog
from pymorgan.gui.crop_dialog import CropDialog
from pymorgan.gui.main_window import MainWindow
from pymorgan.gui.svd_dialog import SVDDialog
from pymorgan.oneD.dataset import Dataset1D
from pymorgan.oneD.stitch import stitch_time, stitch_wavelength
from pymorgan.oneD.svd import compute_svd
from pymorgan.twoD.dataset import Dataset2D


@pytest.fixture(scope="module")
def qapp():
    """Ensure a single QApplication instance for offscreen Qt tests."""
    return QApplication.instance() or QApplication([])


def _make_1d_dataset(*, corrected: bool = False, n_det: int = 1) -> Dataset1D:
    rng = np.random.default_rng(123)
    delays = np.array([-5.0, -1.0, 0.0, 0.5, 1.5, 4.0, 10.0, 25.0, 60.0])
    probe = np.linspace(450.0, 650.0, 20)
    kin = np.where(delays > 0, np.exp(-delays / 8.0), 0.0)
    spec = np.exp(-((probe - 540.0) / 35.0) ** 2)
    z2d = np.outer(kin, spec) + 0.002 * rng.standard_normal((len(delays), len(probe)))
    if n_det == 1:
        z = z2d[:, :, np.newaxis]
        zstd = np.full_like(z, 0.05)
    else:
        z = np.stack([z2d, 0.5 * z2d], axis=2)
        zstd = np.full_like(z, 0.05)

    units = pm.helpers.units2dic("nm", "ps", "mOD")
    units["counts"] = np.tile(np.linspace(100.0, 200.0, len(probe)), (len(delays), 1))
    units["rms"] = np.full((len(delays), len(probe)), 0.02)
    if n_det == 2:
        units["detectors"] = ["Parallel", "Perpendicular"]

    ds = Dataset1D(
        Zavg_R=z,
        delays=delays,
        probe=probe,
        units=units,
        nscans=5,
        Zstdv=zstd,
        source="synthetic_issue6.pdat",
        data_type="PDAT",
        counts=units["counts"].copy(),
    )
    ds._orig_Zstdv = zstd.copy()
    if corrected:
        ds.background_correct(tmin=-5.0, tmax=-0.5)
    return ds


def _make_2d_dataset_with_detectors() -> Dataset2D:
    pump = np.linspace(1980.0, 2060.0, 16)
    probe = np.linspace(1970.0, 2070.0, 18)
    delays = np.array([0.2, 0.5, 1.0, 3.0, 10.0])
    P, Q = np.meshgrid(pump, probe, indexing="ij")
    cube = np.stack(
        [np.exp(-d / 4.0) * np.exp(-((P - 2020.0) ** 2 + (Q - 2020.0) ** 2) / 100.0) for d in delays],
        axis=2,
    )
    det_b_probe = probe + 5.0
    det_data = {"DetA": cube.copy(), "DetB": cube * 0.8}
    det_probes = {"DetA": probe.copy(), "DetB": det_b_probe.copy()}
    units = {"unitsL": "cm^{-1}", "unitsT": "ps"}
    ds = Dataset2D(
        Z=cube,
        pump=pump,
        probe=probe,
        delays=delays,
        units=units,
        source="synthetic_2d.p2dat",
        data_type="P2DAT",
        detector_names=["DetA", "DetB"],
        detector_probes=det_probes,
        detector_data=det_data,
        active_detector="DetA",
    )
    return ds


# =========================================================================== #
# 1. SVD / Stitch / Dataset1D 3-D Invariant & PDAT Export Fixes               #
# =========================================================================== #


def test_svd_to_dataset_canonical_3d_and_pdat_roundtrip(tmp_path):
    ds = _make_1d_dataset(corrected=True)
    svd_res = compute_svd(ds.Z[:, :, 0], ds.delays, ds.probe, units=ds.units)
    recomp = svd_res.to_dataset(k=2)

    assert recomp.Zavg_R.shape == (len(ds.delays), len(ds.probe), 1)
    assert recomp.Zavg_C is not None
    assert recomp.Zavg_C.shape == (len(ds.delays), len(ds.probe), 1)
    assert recomp.Zstdv is not None
    assert recomp.Zstdv.shape == (len(ds.delays), len(ds.probe), 1)
    assert np.allclose(recomp.Zstdv, 0.0)
    assert np.isnan(recomp.nscans)
    assert recomp.Zss_R is None
    assert recomp.noise_array() is None
    assert recomp.data_type == "SVD-recomposed"

    out_pdat = tmp_path / "svd_recon.pdat"
    written = recomp.to_pdat(out_pdat)
    assert written and Path(written[0]).is_file()

    loaded = pm.load_1D(written[0], data_type="PDAT")
    assert loaded.Zavg_R.shape == recomp.Zavg_R.shape
    assert np.allclose(loaded.Zavg_R[:, :, 0], recomp.Z[:, :, 0], atol=1e-4)


def test_dataset1d_init_promotes_2d_arrays_and_stitch_is_3d(tmp_path):
    delays = np.linspace(0.0, 10.0, 6)
    probe = np.linspace(400.0, 750.0, 12)
    z2d = np.ones((6, 12)) * 2.0
    zstd_2d = np.full((6, 12), 0.1)
    ds2 = Dataset1D(
        Zavg_R=z2d,
        delays=delays,
        probe=probe,
        units=pm.helpers.units2dic("nm", "ps", "mOD"),
        Zstdv=zstd_2d,
    )
    assert ds2.Zavg_R.shape == (6, 12, 1)
    assert ds2.Zstdv.shape == (6, 12, 1)

    # Wavelength stitch dataset invariant + PDAT export
    ds_a = Dataset1D(
        Zavg_R=np.ones((6, 12)),
        delays=delays,
        probe=np.linspace(400.0, 720.0, 12),
        units=pm.helpers.units2dic("nm", "ps", "mOD"),
    )
    ds_b = Dataset1D(
        Zavg_R=np.ones((6, 12)) * 1.5,
        delays=delays,
        probe=np.linspace(680.0, 1000.0, 12),
        units=pm.helpers.units2dic("nm", "ps", "mOD"),
    )
    w_res = stitch_wavelength(ds_a, ds_b, mode="blend", stitch_region=(685.0, 715.0))
    w_ds = w_res.to_dataset()
    assert w_ds.Zavg_R.ndim == 3 and w_ds.Zavg_R.shape[2] == 1
    assert w_ds.Zavg_C.shape == w_ds.Zavg_R.shape
    assert w_ds.Zstdv.shape == w_ds.Zavg_R.shape
    assert w_ds.data_type == "Wavelength-Stitched"
    assert Path(w_ds.to_pdat(tmp_path / "w_stitched.pdat")[0]).is_file()

    # Timescale stitch dataset invariant + PDAT export
    t_fs = np.linspace(-1.0, 1000.0, 15)
    t_ns = np.array([-0.5, 0.2, 0.8, 2.0, 10.0, 50.0])
    pr = np.linspace(500.0, 600.0, 8)
    ds_fs = Dataset1D(
        Zavg_R=np.ones((len(t_fs), len(pr))),
        delays=t_fs,
        probe=pr,
        units=pm.helpers.units2dic("nm", "ps", "mOD"),
    )
    ds_ns = Dataset1D(
        Zavg_R=np.ones((len(t_ns), len(pr))) * 2.0,
        delays=t_ns,
        probe=pr,
        units=pm.helpers.units2dic("nm", "ns", "mOD"),
    )
    t_res = stitch_time(ds_fs, ds_ns, time_overlap=(0.2, 0.9), blend_window=(0.3, 0.8))
    t_ds = t_res.to_dataset()
    assert t_ds.Zavg_R.ndim == 3 and t_ds.Zavg_R.shape[2] == 1
    assert t_ds.Zavg_C.shape == t_ds.Zavg_R.shape
    assert t_ds.Zstdv.shape == t_ds.Zavg_R.shape
    assert t_ds.data_type == "Timescale-Stitched"
    assert Path(t_ds.to_pdat(tmp_path / "t_stitched.pdat")[0]).is_file()


def test_svd_and_anisotropy_dialogs_export_pdat(qapp, tmp_path, monkeypatch):
    info_msgs = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **kw: info_msgs.append(a[2]))

    ds1 = _make_1d_dataset(corrected=True)
    svd_dlg = SVDDialog(None, ds1)
    try:
        svd_pdat = tmp_path / "svd_from_dialog.pdat"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **kw: (str(svd_pdat), ""))
        svd_dlg._export_recomposed()
        assert svd_pdat.is_file()
    finally:
        svd_dlg.close()

    ds2 = _make_1d_dataset(corrected=False, n_det=2)
    aniso_dlg = AnisotropyDialog(None, ds2)
    try:
        aniso_pdat = tmp_path / "aniso_from_dialog.pdat"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **kw: (str(aniso_pdat), ""))
        aniso_dlg._export_anisotropy()
        # Multi-channel or single suffix write: at least one matching .pdat file exists
        written = list(tmp_path.glob("aniso_from_dialog*.pdat"))
        assert len(written) >= 1
        assert len(info_msgs) >= 2
    finally:
        aniso_dlg.close()


# =========================================================================== #
# 2. Global GUI Excepthook & MainWindow Slot Error Guards                     #
# =========================================================================== #


def test_install_gui_excepthook_logs_reentrancy_and_keyboard_interrupt(caplog):
    orig_sys = sys.excepthook
    orig_thr = threading.excepthook
    orig_flag = gui_app._IN_EXCEPTHOOK
    kb_calls = []
    orig_default = sys.__excepthook__

    try:
        gui_app.install_gui_excepthook("PyMORGAN")

        with caplog.at_level(logging.ERROR, logger="pymorgan"):
            try:
                raise RuntimeError("boom in slot")
            except RuntimeError as exc:
                sys.excepthook(type(exc), exc, exc.__traceback__)
        assert any("Unhandled exception in PyMORGAN" in r.message for r in caplog.records)

        # Re-entrancy guard prevents recursive hook work
        gui_app._IN_EXCEPTHOOK = True
        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="pymorgan"):
            try:
                raise RuntimeError("reentrant boom")
            except RuntimeError as exc:
                sys.excepthook(type(exc), exc, exc.__traceback__)
        assert not any("reentrant boom" in str(r.exc_info) for r in caplog.records)
        gui_app._IN_EXCEPTHOOK = False

        # KeyboardInterrupt delegates to sys.__excepthook__
        sys.__excepthook__ = lambda et, ev, tb: kb_calls.append(et)
        try:
            raise KeyboardInterrupt()
        except KeyboardInterrupt as exc:
            sys.excepthook(type(exc), exc, exc.__traceback__)
        assert kb_calls == [KeyboardInterrupt]
    finally:
        sys.__excepthook__ = orig_default
        sys.excepthook = orig_sys
        threading.excepthook = orig_thr
        gui_app._IN_EXCEPTHOOK = orig_flag


def test_mainwindow_svd_apply_error_guard_and_derived_background_checkbox(qapp, tmp_path, monkeypatch):
    critical_calls = []
    info_calls = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **kw: critical_calls.append(a[2]))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **kw: info_calls.append(a[2]))
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **kw: QMessageBox.StandardButton.Yes
    )

    ds = _make_1d_dataset(corrected=True)
    recomp = compute_svd(ds.Z[:, :, 0], ds.delays, ds.probe, units=ds.units).to_dataset(k=2)

    win = MainWindow(init_rootdir=False)
    try:
        # 1. Background checkbox stays unchecked for derived/recomposed datasets
        win.dataset = recomp
        win._setup_background_controls()
        assert win.PP_SubtactBkg_chk.isChecked() is False

        # 2. _on_svd_dataset_applied catches failure, sets _apply_failed=True, and prevents "Applied" popup
        win.dataset = ds
        win._preview_contour = lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("plot crash"))
        dlg = SVDDialog(win, ds)
        dlg.dataset_applied.connect(win._on_svd_dataset_applied)
        try:
            dlg._apply_to_workspace()
            assert win._apply_failed is True
            assert win.dataset is ds
            assert any("plot crash" in msg for msg in critical_calls)
            assert info_calls == []
        finally:
            dlg.close()
    finally:
        win.close()


# =========================================================================== #
# 3. 1D & 2D Crop Backend + Undo/Redo Provenance                              #
# =========================================================================== #


def test_dataset1d_crop_swapped_bounds_auxiliary_arrays_and_undo_redo():
    ds = _make_1d_dataset(corrected=True)
    orig_delays_n = len(ds.delays)
    orig_probe_n = len(ds.probe)

    # Swapped bounds must normalize cleanly and slice Zstdv, _orig_Zstdv, counts, rms
    ds.crop(tmin=10.0, tmax=0.0, probe_min=610.0, probe_max=490.0)
    nt, npix = len(ds.delays), len(ds.probe)
    assert 0 < nt < orig_delays_n
    assert 0 < npix < orig_probe_n
    assert float(ds.delays.min()) >= 0.0 and float(ds.delays.max()) <= 10.0
    assert float(ds.probe.min()) >= 490.0 and float(ds.probe.max()) <= 610.0
    assert ds.Zavg_R.shape == (nt, npix, 1)
    assert ds.Zavg_C.shape == (nt, npix, 1)
    assert ds.Zstdv.shape == (nt, npix, 1)
    assert ds._orig_Zstdv.shape == (nt, npix, 1)
    assert ds.counts.shape == (nt, npix)
    assert ds.units["counts"].shape == (nt, npix)
    assert ds.units["rms"].shape == (nt, npix)

    assert ds.undo() is True
    assert len(ds.delays) == orig_delays_n
    assert len(ds.probe) == orig_probe_n
    assert ds.counts.shape == (orig_delays_n, orig_probe_n)

    assert ds.redo() is True
    assert len(ds.delays) == nt
    assert len(ds.probe) == npix
    assert ds.counts.shape == (nt, npix)


def test_dataset2d_crop_swapped_bounds_detectors_and_undo_redo():
    ds2 = _make_2d_dataset_with_detectors()
    orig_pu, orig_pr, orig_t2 = len(ds2.pump), len(ds2.probe), len(ds2.delays)

    ds2.crop(
        pump_min=2045.0,
        pump_max=1995.0,
        probe_min=2045.0,
        probe_max=1990.0,
        t2_min=3.5,
        t2_max=0.4,
    )
    npu, npr, nt2 = len(ds2.pump), len(ds2.probe), len(ds2.delays)
    assert 0 < npu < orig_pu
    assert 0 < npr < orig_pr
    assert 0 < nt2 < orig_t2
    assert ds2.Z_R.shape == (npu, npr, nt2)
    assert ds2.detector_data["DetA"].shape == (npu, npr, nt2)
    assert ds2.detector_probes["DetA"].shape == (npr,)
    # DetB probe was shifted +5, so independently sliced within [1990, 2045]
    npr_b = len(ds2.detector_probes["DetB"])
    assert npr_b > 0
    assert ds2.detector_data["DetB"].shape == (npu, npr_b, nt2)

    assert ds2.undo() is True
    assert ds2.Z_R.shape == (orig_pu, orig_pr, orig_t2)
    assert ds2.detector_probes["DetB"].shape == (orig_pr,)
    assert ds2.detector_data["DetB"].shape == (orig_pu, orig_pr, orig_t2)

    assert ds2.redo() is True
    assert ds2.Z_R.shape == (npu, npr, nt2)
    assert ds2.detector_probes["DetB"].shape == (npr_b,)


# =========================================================================== #
# 4. CropDialog, Scissors Toggle Buttons & MainWindow Integration             #
# =========================================================================== #


def test_crop_dialog_1d_and_2d_apply_export_presets(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **kw: QMessageBox.StandardButton.Ok)

    # --- 1D CropDialog ---
    ds1 = _make_1d_dataset(corrected=True)
    orig_shape = ds1.Zavg_R.shape
    dlg1 = CropDialog(
        None,
        ds1,
        initial_bounds={"probe_min": 480.0, "probe_max": 600.0, "tmin": 0.0, "tmax": 25.0},
        view_bounds={"probe_min": 500.0, "probe_max": 580.0, "tmin": 1.0, "tmax": 10.0},
    )
    emitted1 = []
    dlg1.dataset_cropped.connect(emitted1.append)
    try:
        assert np.isclose(dlg1.spn_probe_min.value(), 480.0)
        assert np.isclose(dlg1.spn_tmax.value(), 25.0)

        dlg1.btn_match_view.click()
        assert np.isclose(dlg1.spn_probe_min.value(), 500.0)
        assert np.isclose(dlg1.spn_tmin.value(), 1.0)

        # Export non-destructive copy -> must NOT mutate workspace ds1
        out_pdat = tmp_path / "cropped_copy.pdat"
        monkeypatch.setattr(
            QFileDialog,
            "getSaveFileName",
            lambda *a, **kw: (str(out_pdat), "PDAT format (*.pdat)"),
        )
        dlg1._on_export_copy()
        assert out_pdat.is_file()
        assert ds1.Zavg_R.shape == orig_shape

        dlg1.btn_reset_full.click()
        assert np.isclose(dlg1.spn_probe_min.value(), float(ds1.probe.min()))
        assert np.isclose(dlg1.spn_tmin.value(), float(ds1.delays.min()))

        dlg1.spn_probe_min.setValue(480.0)
        dlg1.spn_probe_max.setValue(600.0)
        dlg1.spn_tmin.setValue(0.0)
        dlg1.spn_tmax.setValue(25.0)
        dlg1._on_apply()
        assert len(emitted1) == 1
        assert ds1.Zavg_R.shape != orig_shape
    finally:
        dlg1.close()

    # --- 2D CropDialog ---
    ds2 = _make_2d_dataset_with_detectors()
    orig_2d_shape = ds2.Z_R.shape
    dlg2 = CropDialog(
        None,
        ds2,
        initial_bounds={"pump_min": 1995.0, "pump_max": 2045.0, "probe_min": 1990.0, "probe_max": 2050.0},
        view_bounds={"pump_min": 2000.0, "pump_max": 2040.0, "probe_min": 2000.0, "probe_max": 2040.0},
    )
    emitted2 = []
    dlg2.dataset_cropped.connect(emitted2.append)
    try:
        assert dlg2.t2_slider is not None
        dlg2.t2_slider.setValue(min(2, dlg2.t2_slider.maximum()))
        dlg2.btn_match_view.click()
        assert np.isclose(dlg2.spn_pump_min.value(), 2000.0)

        out_h5 = tmp_path / "cropped_copy_2d.h5"
        monkeypatch.setattr(
            QFileDialog,
            "getSaveFileName",
            lambda *a, **kw: (str(out_h5), "PyMORGAN 2D HDF5 (*.h5)"),
        )
        dlg2._on_export_copy()
        assert out_h5.is_file()
        assert ds2.Z_R.shape == orig_2d_shape

        dlg2._on_apply()
        assert len(emitted2) == 1
        assert ds2.Z_R.shape != orig_2d_shape
    finally:
        dlg2.close()


def test_mainwindow_scissors_buttons_crop_action_and_bkg_preservation(qapp, tmp_path, monkeypatch):
    pdat_info = synthetic.make_synthetic_pdat(tmp_path / "crop_1d.pdat")
    p2_info = synthetic.make_synthetic_p2dat(tmp_path / "crop_2d.p2dat")

    win = MainWindow(init_rootdir=False)
    try:
        # Before load: crop action disabled
        assert hasattr(win, "actionCropDataset")
        assert hasattr(win, "PC_cropBox")
        assert hasattr(win, "twoD_PC_cropBox")
        assert win.actionCropDataset.isEnabled() is False

        # Load 1D and verify background stays preserved when cropping to tmin > 0
        win.MainTabs.setCurrentIndex(0)
        win.load_path(pdat_info["path"], "PDAT")
        qapp.processEvents()
        assert win.actionCropDataset.isEnabled() is True

        # Trigger 1D box crop toggle and ensure tab switch cancels it cleanly
        win.PC_cropBox.setChecked(True)
        assert win.PC_cropBox.isChecked() is True
        win.MainTabs.setCurrentIndex(1)
        assert win.PC_cropBox.isChecked() is False

        # Apply 1D crop removing negative delays (tmin > 0) while preserving Zavg_C
        win.MainTabs.setCurrentIndex(0)
        ds1 = win.dataset
        ds1.background_correct(tmin=float(ds1.delays.min()), tmax=-0.1)
        expected_zavg_c_at_pos = ds1.Zavg_C[ds1.delays >= 1.0].copy()

        # Fail test immediately if _apply_background is mistakenly re-called
        win._apply_background = lambda *_a, **_kw: (_ for _ in ()).throw(
            AssertionError("_apply_background must not run on post-crop tmin > 0")
        )
        ds1.crop(tmin=1.0, tmax=float(ds1.delays.max()))
        win._on_1d_cropped(ds1)
        assert np.allclose(win.dataset.Zavg_C, expected_zavg_c_at_pos)
        assert win.PP_bkg_tmin.minimum() >= 0.0

        # Load 2D dataset and verify actionCropDataset + 2D scissors toggle + _on_2d_cropped
        win.MainTabs.setCurrentIndex(1)
        win.twoD_load_path(p2_info["path"], data_type="P2DAT")
        qapp.processEvents()
        assert win.actionCropDataset.isEnabled() is True

        win.twoD_PC_cropBox.setChecked(True)
        assert win.twoD_PC_cropBox.isChecked() is True
        win.MainTabs.setCurrentIndex(0)
        assert win.twoD_PC_cropBox.isChecked() is False

        win.MainTabs.setCurrentIndex(1)
        ds2 = win.twoD_dataset
        old_shape = ds2.Z_R.shape
        ds2.crop(pump_min=2010.0, pump_max=2060.0)
        win._on_2d_cropped(ds2)
        assert win.twoD_dataset.Z_R.shape != old_shape
    finally:
        win.close()
