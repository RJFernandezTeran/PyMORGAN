"""Tests for Dataset2D multi-detector support and UoS_2DIR / RAL_RAW / RAL_Proc loaders."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import pymorgan as pm
from pymorgan.twoD.dataset import Dataset2D
from pymorgan.twoD.load import is_map_dataset_dir

TESTDATA_ROOT = Path(r"c:\Users\ricar\switchdrive\Ambizione UniGE\Scripts\testData")
UOS_BASE = TESTDATA_ROOT / "UoS" / "20220505 - IrH SEC Good"
UOS_2DIR_GOOD = UOS_BASE / "IrHCOP3_DCM_OTTLE_0V_2DIR_20k64"
UOS_2DIR_P2DAT = UOS_BASE / "IrHCOP3_DCM_OTTLE_0V_2DIR_20k64_PROCESSED_DET1.p2dat"
UOS_2DIR_MEH = UOS_BASE / "IrHCOP3_DCM_OTTLE_+700mV_2DIR_20k64-meh"

RAL_BASE = Path(r"E:\UoS Google Drive\RAL STUFF\20221017 - 2DIR")
RAL_RAW_DIR = RAL_BASE / "NAP-13CC-Pt-CC-Ph-CN__DCM__2DIR_10s3cy"
RAL_PROC_DIR = RAL_BASE / "PROC" / "NAP-13CC-Pt-CC-Ph-CN__DCM__2DIR_10s3cy"


# --------------------------------------------------------------------------- #
#                     Synthetic Helpers for UoS / RAL 2D                      #
# --------------------------------------------------------------------------- #
def _make_synthetic_uos_2dir(
    root: Path,
    *,
    delays_ps: list[float],
    present_indices: list[int] | None = None,
    n_t1: int = 32,
    n_cols: int = 192,
    w0_cm1: float = 1950.0,
    dt1_fs: float = 15.0,
    with_cal_probe: bool = True,
) -> Path:
    ds_dir = root / "Sample_2DIR_scan"
    ds_dir.mkdir(parents=True, exist_ok=True)
    stem = ds_dir.name

    # UoS 2DIR marker: 0-byte <stem>.2D alongside .DT and D{k} slices
    (ds_dir / f"{stem}.2D").write_bytes(b"")
    (ds_dir / f"{stem}.DT").write_text(
        "\n".join(f"{d:.6g}" for d in delays_ps) + "\n", encoding="utf-8"
    )
    (ds_dir / f"{stem}.LG").write_text(
        "Sample: TestComplex\nSolvent: DCM\nScans: 100\n", encoding="utf-8"
    )
    (ds_dir / "w0.csv").write_text(f"{w0_cm1},{dt1_fs}\n", encoding="utf-8")

    if with_cal_probe:
        probe1 = np.linspace(1920.0, 2060.0, 96)
        probe2 = np.linspace(1900.0, 2040.0, 96)
        np.savetxt(ds_dir / "CalibratedProbe.csv", np.concatenate([probe1, probe2]), delimiter=",")

    t1 = np.arange(n_t1, dtype=float) * dt1_fs
    indices = list(range(len(delays_ps))) if present_indices is None else list(present_indices)
    for k in indices:
        d_ps = delays_ps[k]
        mat = np.zeros((n_t1, n_cols), dtype=float)
        freq = 0.03 * (1980.0 - w0_cm1)
        envelope = np.exp(-t1 / 250.0) * np.exp(-max(d_ps, 0.0) / 10.0)
        sig_1d = envelope * np.cos(0.188 * freq * t1 + 0.1 * t1)
        mat[:, : min(96, n_cols)] = (
            sig_1d[:, None]
            * np.exp(-0.5 * ((np.arange(min(96, n_cols)) - 48) / 10.0) ** 2)[None, :]
        )
        if n_cols >= 192:
            mat[:, 96:192] = (
                0.5 * sig_1d[:, None] * np.exp(-0.5 * ((np.arange(96) - 40) / 12.0) ** 2)[None, :]
            )
        np.savetxt(ds_dir / f"{stem}_D{k}_{d_ps:.3f}_ps.2D", mat, delimiter="\t", fmt="%.6e")

    return ds_dir


def _make_synthetic_ral_raw(root: Path) -> Path:
    ds_dir = root / "RAL_RAW_Sample"
    ds_dir.mkdir(parents=True, exist_ok=True)
    n_t1 = 24
    delays_ps = [2.0, -0.5, 0.5, 5.0]
    (ds_dir / "w0.csv").write_text("1600.0,20.0\n", encoding="utf-8")
    (ds_dir / "Run 0 info.txt").write_text("Sample: RAL Test\nPump: 1650 cm-1\n", encoding="utf-8")

    det1 = np.linspace(1780.0, 2240.0, 128)
    det2 = np.linspace(1500.0, 1775.0, 128)
    np.savetxt(ds_dir / "CalibratedProbe.csv", np.concatenate([det1, det2]), delimiter=",")

    headers: list[float] = []
    cols: list[np.ndarray] = []
    # Header encodes t1_idx * 1e7 + t2_ps; note BOTH leading t1_idx==0 columns exist per t2 block
    for d_ps in delays_ps:
        t2_rem = d_ps
        t1_indices = [0] + list(range(n_t1))
        for k in t1_indices:
            headers.append(k * 1e7 + t2_rem)
            col = np.zeros(256, dtype=float)
            amp = np.cos(0.25 * k) * np.exp(-k / 12.0) * np.exp(-max(d_ps, 0.0) / 8.0)
            col[:128] = amp * np.exp(-0.5 * ((det1 - 2010.0) / 30.0) ** 2)
            col[128:] = -0.8 * amp * np.exp(-0.5 * ((det2 - 1650.0) / 25.0) ** 2)
            cols.append(col)

    mat = np.column_stack(cols)
    hdr_line = "Pixel," + ",".join(f"{h:.10g}" for h in headers)
    body = "\n".join(
        f"{pix_idx + 1}," + ",".join(f"{v:.6e}" for v in row) for pix_idx, row in enumerate(mat)
    )
    (ds_dir / "Run 0_AvgDiff.csv").write_text(hdr_line + "\n" + body + "\n", encoding="utf-8")

    # Add a 1D Avg Diff spec sibling to confirm RAL_RAW priority/disambiguation
    (ds_dir / "Run 0 Avg Diff spec 01.csv").write_text(
        "Pixel,0.0,1.0\n1,0.1,0.2\n", encoding="utf-8"
    )
    return ds_dir


def _make_synthetic_ral_proc(root: Path) -> Path:
    ds_dir = root / "RAL_PROC_Sample"
    ds_dir.mkdir(parents=True, exist_ok=True)
    # Build even-length Hermitian-symmetric pump axis (2 * n_half = 32 cols)
    n_half = 16
    pump_low = np.linspace(1500.0, 1900.0, n_half)
    pump_hi = np.linspace(1925.0, 2300.0, n_half)
    pump_full = np.concatenate([pump_low, pump_hi])
    det1 = np.linspace(1780.0, 2240.0, 128)
    det2 = np.linspace(1500.0, 1775.0, 128)
    probe_full = np.concatenate([det1, det2])

    specs = [("a_500fs_2DIR.csv", 0.5), ("b_-0.100ps_2DIR.csv", -0.1), ("c_0.001ns_2DIR.csv", 1.0)]
    for fname, d_ps in specs:
        out = np.zeros((1 + 256, 1 + len(pump_full)), dtype=float)
        out[0, 1:] = pump_full
        out[1:, 0] = probe_full
        low_grid = np.outer(
            np.exp(-0.5 * ((probe_full - 1950.0) / 40.0) ** 2),
            np.exp(-0.5 * ((pump_low - 1700.0) / 35.0) ** 2),
        ) * np.exp(-max(d_ps, 0.0) / 3.0)
        out[1:, 1 : 1 + n_half] = low_grid
        out[1:, 1 + n_half] = low_grid[:, -1] * 0.5
        out[1:, 2 + n_half :] = low_grid[:, 1:n_half][:, ::-1]
        lines = [",".join(f"{v:.8g}" for v in row) for row in out]
        (ds_dir / fname).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ds_dir


# --------------------------------------------------------------------------- #
#                         Dataset2D Multi-Detector API                        #
# --------------------------------------------------------------------------- #
def test_dataset2d_multi_detector_switch_copy_and_p2dat(tmp_path):
    pump = np.linspace(1900.0, 2050.0, 8)
    probe1 = np.linspace(1920.0, 2060.0, 6)
    probe2 = np.linspace(1780.0, 1910.0, 5)
    delays = np.array([0.5, 2.0, 10.0])

    z1 = np.ones((8, 6, 3)) * 1.5
    z2 = np.ones((8, 5, 3)) * -2.5

    ds = Dataset2D(
        z1,
        pump,
        probe1,
        delays,
        {},
        datatype="processed",
        is_frequency_domain=True,
        detector_names=["DET1", "DET2"],
        detector_probes={"DET1": probe1, "DET2": probe2},
        detector_data={"DET1": z1, "DET2": z2},
        active_detector="DET1",
        data_type="RAL_Proc",
    )

    assert ds.detector_names == ["DET1", "DET2"]
    assert ds.active_detector == "DET1"
    assert ds.n_detectors == 2
    assert "<b>Detector:</b> DET1" in ds.sample_info()

    ds.background_correct(0)
    assert ds._is_corrected is True
    assert ds.Z_C is not None
    ds._cls_cache = {"dummy": True}

    assert ds.set_active_detector("DOES_NOT_EXIST") is False
    assert ds.set_active_detector("DET2") is True
    assert ds.active_detector == "DET2"
    assert ds.probe.shape == (5,)
    assert np.allclose(ds.Z_R, z2)
    assert ds._is_corrected is False
    assert ds.Z_C is None
    assert not hasattr(ds, "_cls_cache")

    copied = ds.copy()
    assert copied.detector_names == ["DET1", "DET2"]
    assert copied.active_detector == "DET2"
    assert np.allclose(copied.detector_probes["DET1"], probe1)

    single_out = tmp_path / "single.p2dat"
    assert ds.to_p2dat(single_out) is None
    assert single_out.is_file()

    all_out = ds.to_p2dat(tmp_path / "multi.p2dat", detector="all")
    assert isinstance(all_out, list) and len(all_out) == 2
    assert all(p.is_file() for p in all_out)
    reloaded = pm.load_2D(all_out[0], data_type="P2DAT")
    assert reloaded.Z_R.shape == (8, 6, 3)


# --------------------------------------------------------------------------- #
#                           UoS_2DIR / RAL Loaders                            #
# --------------------------------------------------------------------------- #
def test_uos_2dir_synthetic_complete_and_incomplete(tmp_path):
    complete_dir = _make_synthetic_uos_2dir(tmp_path / "complete", delays_ps=[-1.0, 0.5, 2.0, 10.0])
    assert is_map_dataset_dir("UoS_2DIR", complete_dir) is True

    ds = pm.load_2D(complete_dir, data_type="UoS_2DIR")
    assert ds.raw_signal.shape == (32, 96, 4)
    assert np.allclose(ds.delays, [-1.0, 0.5, 2.0, 10.0])
    assert ds.detector_names == ["DET1", "DET2"]
    assert ds.active_detector == "DET1"
    assert ds.in_progress is False
    assert ds.w0_cm1 == pytest.approx(1950.0)
    assert ds.dt1_fs == pytest.approx(15.0)
    assert ds.datatype == "shaper"
    assert ds.cal_level == "datadir"

    ds.process(zeropad_enable=True, zeropad_factor=4)
    assert ds.Z_R.ndim == 3 and ds.Z_R.shape[1:] == (96, 4)
    assert ds.set_active_detector("DET2") is True
    ds.process(zeropad_enable=True, zeropad_factor=4)
    assert ds.Z_R.shape[1:] == (96, 4)

    partial_dir = _make_synthetic_uos_2dir(
        tmp_path / "partial", delays_ps=[0.0, 1.0, 5.0, 20.0], present_indices=[0, 2]
    )
    ds_part = pm.load_2D(partial_dir, data_type="UoS_2DIR")
    assert ds_part.in_progress is True
    assert ds_part.raw_signal.shape == (32, 96, 2)
    assert np.allclose(ds_part.delays, [0.0, 5.0])


def test_ral_raw_and_proc_synthetic(tmp_path):
    raw_dir = _make_synthetic_ral_raw(tmp_path)
    assert is_map_dataset_dir("RAL_RAW", raw_dir) is True

    ds_raw = pm.load_2D(raw_dir, data_type="RAL_RAW")
    assert ds_raw.raw_signal.shape == (25, 128, 4)
    assert np.allclose(ds_raw.delays, [-0.5, 0.5, 2.0, 5.0])
    assert ds_raw.detector_names == ["DET1", "DET2", "Combined (DET1+DET2)"]
    assert ds_raw.set_active_detector("Combined (DET1+DET2)") is True
    assert ds_raw.raw_signal.shape == (25, 256, 4)
    assert np.all(np.diff(ds_raw.probe) > 0)

    proc_dir = _make_synthetic_ral_proc(tmp_path)
    assert is_map_dataset_dir("RAL_Proc", proc_dir) is True

    ds_proc = pm.load_2D(proc_dir, data_type="RAL_Proc")
    assert ds_proc.is_frequency_domain is True
    assert ds_proc.datatype == "processed"
    assert ds_proc.Z_R.shape == (16, 128, 3)
    assert np.allclose(ds_proc.delays, [-0.1, 0.5, 1.0])
    assert ds_proc.set_active_detector("Combined (DET1+DET2)") is True
    assert ds_proc.Z_R.shape == (16, 256, 3)


def test_twoD_gui_detector_combo_switches_detectors(qapp, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from pymorgan.gui.main_window import MainWindow
    from pymorgan.gui.mw_common import set_combo_datatype

    uos_dir = _make_synthetic_uos_2dir(tmp_path, delays_ps=[0.5, 2.0])
    win = MainWindow()
    try:
        win.show()
        set_combo_datatype(win.twoD_datatype_cbx, "UoS_2DIR")
        win.MainTabs.setCurrentIndex(1)
        win.twoD_load_path(str(uos_dir), data_type="UoS_2DIR")
        qapp.processEvents()

        assert win.twoD_dataset is not None
        det_cbx = getattr(win, "_twoD_detector_cbx", None)
        assert det_cbx is not None
        assert det_cbx.isVisible() is True
        assert [det_cbx.itemText(i) for i in range(det_cbx.count())] == ["DET1", "DET2"]
        assert win.twoD_dataset.active_detector == "DET1"

        det_cbx.setCurrentText("DET2")
        qapp.processEvents()
        assert win.twoD_dataset.active_detector == "DET2"
        assert np.allclose(win.twoD_dataset.probe, win.twoD_dataset.detector_probes["DET2"])
    finally:
        win.close()


# --------------------------------------------------------------------------- #
#                        Auto-Skipping Real-Data Tests                        #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not UOS_2DIR_GOOD.is_dir(), reason="UoS 2DIR real testData absent")
def test_real_uos_2dir_good_and_meh():
    assert is_map_dataset_dir("UoS_2DIR", UOS_2DIR_GOOD) is True
    ds = pm.load_2D(UOS_2DIR_GOOD, data_type="UoS_2DIR")
    assert ds.in_progress is False
    assert ds.raw_signal.shape == (268, 96, 16)
    assert ds.detector_names == ["DET1", "DET2"]
    assert ds.w0_cm1 == pytest.approx(1450.0)
    assert ds.dt1_fs == pytest.approx(15.0)

    if UOS_2DIR_P2DAT.is_file():
        ds.process(zeropad_enable=True, zeropad_factor=4)
        p2 = pm.load_2D(UOS_2DIR_P2DAT, data_type="P2DAT")
        assert ds.Z_R.shape[1] == 96 and ds.Z_R.shape[2] == p2.Z_R.shape[2] == 16
        assert np.allclose(ds.delays, p2.delays)

    if UOS_2DIR_MEH.is_dir():
        ds_meh = pm.load_2D(UOS_2DIR_MEH, data_type="UoS_2DIR")
        assert ds_meh.in_progress is True
        assert ds_meh.raw_signal.shape[0] == 268
        assert ds_meh.raw_signal.shape[1] == 96
        assert ds_meh.raw_signal.shape[2] == 5


@pytest.mark.skipif(
    not (RAL_RAW_DIR.is_dir() and RAL_PROC_DIR.is_dir()),
    reason="RAL 2DIR Drive testData absent",
)
def test_real_ral_raw_and_proc_parity():
    assert is_map_dataset_dir("RAL_RAW", RAL_RAW_DIR) is True
    assert is_map_dataset_dir("RAL_Proc", RAL_PROC_DIR) is True

    ds_raw = pm.load_2D(RAL_RAW_DIR, data_type="RAL_RAW")
    ds_proc = pm.load_2D(RAL_PROC_DIR, data_type="RAL_Proc")

    assert ds_raw.raw_signal.shape == (234, 128, 66)
    assert ds_raw.detector_names == ["DET1", "DET2", "Combined (DET1+DET2)"]
    assert ds_proc.Z_R.shape == (234, 128, 66)
    assert ds_proc.detector_names == ["DET1", "DET2", "Combined (DET1+DET2)"]

    assert ds_proc.set_active_detector("Combined (DET1+DET2)") is True
    assert ds_proc.Z_R.shape == (234, 256, 66)
    assert np.all(np.diff(ds_proc.probe) > 0)
