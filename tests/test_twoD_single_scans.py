"""Tests for 2D single-scan exploration, contour array plotting, and scan average recalculation."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pymorgan as pm
from pymorgan.twoD.plot import plot_scan_map_array

matplotlib.use("Agg")


def _make_synthetic_pymess_2d_h5(tmp_path: Path, n_scans: int = 4) -> Path:
    """Create a synthetic PyMESS_2D HDF5 file with individual scans (Schema A)."""
    h5_path = tmp_path / "synth_2d_scans.h5"
    n_t2 = 3
    n_t1 = 32
    n_probe = 16

    t2_delays = np.array([0.2, 1.0, 5.0], dtype=float)
    t1_delays = np.linspace(0.0, 1550.0, n_t1)
    probe_wn = np.linspace(1950.0, 2100.0, n_probe)

    rng = np.random.default_rng(42)
    # Shape: (N_scans, N_t2, N_t1, N_probe)
    scans = np.zeros((n_scans, n_t2, n_t1, n_probe), dtype=float)
    base_sig = np.outer(
        np.cos(2 * np.pi * t1_delays / 200.0) * np.exp(-t1_delays / 600.0),
        np.exp(-((probe_wn - 2025.0) / 25.0) ** 2),
    )
    for s in range(n_scans):
        scale = 10.0 if s == 2 else 1.0  # Make scan 3 (index 2) a 10x outlier
        for d in range(n_t2):
            scans[s, d] = scale * base_sig * np.exp(-d * 0.2) + rng.normal(
                0.0, 1e-4, size=(n_t1, n_probe)
            )

    avg_cube = np.mean(scans, axis=0)  # (N_t2, N_t1, N_probe)

    with h5py.File(h5_path, "w") as f:
        data_grp = f.create_group("data")
        data_grp.create_dataset("t2_delays_fs", data=t2_delays * 1000.0)
        data_grp.create_dataset("t1_delays_fs", data=t1_delays)
        data_grp.create_dataset("omega3_axis", data=probe_wn)
        data_grp.create_dataset("interferograms_raw", data=avg_cube)
        data_grp.create_dataset("individual_scans", data=scans)

        meta = f.create_group("metadata")
        meta.attrs["measurement_type"] = "2d_spectra"
        meta.attrs["domain_mode"] = "time_domain"
        meta.attrs["rotating_frame_cm1"] = 1900.0
        shaper = meta.create_group("shaper")
        shaper.attrs["shaper_mode"] = "time"
        shaper.attrs["form"] = 1
        shaper.attrs["effective_rotating_frame_cm1"] = 1900.0

    return h5_path


def test_synthetic_pymess_2d_single_scans_and_recalc(tmp_path: Path):
    """Verify loading single scans, computing per-scan maps at t2, and recalculating average."""
    h5_path = _make_synthetic_pymess_2d_h5(tmp_path, n_scans=4)
    ds = pm.load_2D(h5_path, data_type="PyMESS_2D")
    ds.process()

    assert ds.has_single_scans
    assert ds.nscans == 4
    assert ds.raw_signal_ss is not None
    assert ds.raw_signal_ss.shape == (32, 16, 3, 4)
    assert ds.active_scan_indices == [0, 1, 2, 3]

    # Compute scan maps at t2 = 1.0 ps (bin_size=1)
    maps_1 = ds.compute_scan_maps_at_t2(1.0, bin_size=1)
    assert len(maps_1) == 4
    assert maps_1[0][0] == "Scan 1"
    assert maps_1[0][1] == [0]
    assert maps_1[2][0] == "Scan 3"
    # Scan 3 has ~10x larger amplitude than Scan 1
    vmax_s1 = float(np.nanmax(np.abs(maps_1[0][2])))
    vmax_s3 = float(np.nanmax(np.abs(maps_1[2][2])))
    assert vmax_s3 > 5.0 * vmax_s1

    # Compute binned scan maps (bin_size=2)
    maps_2 = ds.compute_scan_maps_at_t2(1.0, bin_size=2)
    assert len(maps_2) == 2
    assert maps_2[0][0] == "Scans 1\u20132"
    assert maps_2[0][1] == [0, 1]
    assert maps_2[1][0] == "Scans 3\u20134"
    assert maps_2[1][1] == [2, 3]

    # Plot scan map array with Scan 3 excluded
    fig, axes, panels = plot_scan_map_array(
        ds,
        1.0,
        bin_size=1,
        include_average=True,
        excluded_scans={2},
        scan_maps=maps_1,
    )
    try:
        assert len(panels) == 5  # 4 scans + 1 Average panel
        assert panels[-1]["is_average"] is True
        assert "1\u20132 4" in panels[-1]["label"]
        assert "[Excluded]" in panels[2]["ax"].get_title()
    finally:
        plt.close(fig)

    # Recalculate average excluding outlier Scan 3 (0-based index 2)
    z_max_before = float(np.nanmax(np.abs(ds.Z)))
    ds.recalc_average([0, 1, 3])
    ds.process()
    z_max_after = float(np.nanmax(np.abs(ds.Z)))

    assert ds.active_scan_indices == [0, 1, 3]
    assert z_max_after < 0.5 * z_max_before
    assert "3 of 4" in ds.sample_info()


def test_gui_twoD_single_scans_workflow(qapp, tmp_path: Path, monkeypatch):
    """Test the GUI controls, interactive scan contour array click handler, and Recalc. Avg."""
    from PyQt6.QtWidgets import QInputDialog, QMessageBox

    from pymorgan.gui.main_window import MainWindow

    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **kw: errors.append(("critical", a)))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **kw: errors.append(("warning", a)))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **kw: errors.append(("info", a)))
    monkeypatch.setattr(plt, "show", lambda *a, **kw: None)

    h5_path = _make_synthetic_pymess_2d_h5(tmp_path, n_scans=4)
    win = MainWindow()
    try:
        pm.update_settings(load_single_scans=True)
        win.twoD_load_path(str(h5_path), data_type="PyMESS_2D", force=True)
        assert not errors, f"Unexpected QMessageBox errors: {errors}"
        assert win.twoD_dataset is not None
        assert win.twoD_dataset.has_single_scans
        assert not win.twoD_singleScan_box.isHidden()
        assert win.twoD_BinScans.maximum() == 2

        win._twoD_plot_scan_contours()
        assert not errors, f"Unexpected QMessageBox errors in plot_scan_contours: {errors}"

        # Simulate excluding Scan 3 (0-based index 2) and running _twoD_recalc_average
        win._twoD_excluded_scans = {2}
        captured_defaults = []

        def _fake_get_text(parent, title, label, text=""):
            captured_defaults.append(text)
            return text, True

        monkeypatch.setattr(QInputDialog, "getText", _fake_get_text)
        win._twoD_recalc_average()
        assert not errors, f"Unexpected QMessageBox errors in recalc_average: {errors}"

        assert captured_defaults == ["1-2 4"]
        assert win.twoD_dataset.active_scan_indices == [0, 1, 3]
    finally:
        plt.close("all")
        win.close()
