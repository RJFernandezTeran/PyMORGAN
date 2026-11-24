"""Tests for Helios .ufs, Glotaran .ascii/.nc, FAIR 1D & 2D HDF5, and GUI export actions."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import h5py
import numpy as np
import pytest

import pymorgan as pm
from pymorgan.oneD.chirp import ChirpFit
from pymorgan.oneD.dataset import Dataset1D
from pymorgan.oneD.interop import (
    read_glotaran_ascii,
    read_glotaran_nc,
)
from pymorgan.oneD.load import (
    _has_valid_ufs_header,
    load_1D_hdf5,
    parse_ufs,
    save_1D_hdf5,
)
from pymorgan.twoD.dataset import Dataset2D
from pymorgan.twoD.load import load_2D_hdf5, read_2D_hdf5

HELIOS_REAL_DIR = Path(
    r"c:\Users\ricar\switchdrive\Ambizione UniGE\Scripts\testData\HeliosTA\3-RF47-MeCN-1000nJ"
)


def _pack_ufs_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack(">i", len(raw)) + raw


def _write_synthetic_ufs(
    path: Path,
    *,
    version: str = "Version2",
    probe: np.ndarray | None = None,
    delays: np.ndarray | None = None,
    signal: np.ndarray | None = None,
    z_unit: str = "DA",
    inject_nan: bool = False,
) -> Path:
    probe_arr = np.linspace(450.0, 650.0, 6) if probe is None else np.asarray(probe, dtype=float)
    delay_arr = (
        np.array([-0.5, 0.0, 1.0, 5.0]) if delays is None else np.asarray(delays, dtype=float)
    )
    if signal is None:
        # shape (n_delays, n_probe) in mOD -> store as DA (= OD) in UFS when z_unit="DA"
        sig_mod = np.outer(
            np.exp(-np.maximum(delay_arr, 0.0) / 3.0), np.sin((probe_arr - 450.0) / 50.0)
        )
        sig_mod = sig_mod * 12.0 + 1.0
    else:
        sig_mod = np.asarray(signal, dtype=float)

    stored = sig_mod / 1000.0 if z_unit.upper() == "DA" else sig_mod.copy()
    if inject_nan:
        stored = stored.copy()
        stored[1, 2] = -np.inf

    # UFS stores matrix shape (len(probe), len(delays)) row-major big-endian
    mat_wl_t = np.ascontiguousarray(stored.T)
    dtype_fmt = ">f8" if version == "Version2" else ">f4"
    count_fmt = ">i"

    buf = bytearray()
    buf += _pack_ufs_string(version)
    buf += _pack_ufs_string("Wavelength")
    buf += _pack_ufs_string("nm")
    buf += struct.pack(count_fmt, len(probe_arr))
    buf += probe_arr.astype(dtype_fmt).tobytes()
    buf += _pack_ufs_string("Time")
    buf += _pack_ufs_string("ps")
    buf += struct.pack(count_fmt, len(delay_arr))
    buf += delay_arr.astype(dtype_fmt).tobytes()
    buf += _pack_ufs_string("Absorption")
    buf += _pack_ufs_string(z_unit)
    buf += struct.pack(">ii", mat_wl_t.shape[0], mat_wl_t.shape[1])
    buf += mat_wl_t.astype(dtype_fmt).tobytes()
    buf += _pack_ufs_string("Sample: TestDye\rPump Wavelength: 400 nm\rComments: Synthetic")

    path.write_bytes(bytes(buf))
    return path


# --------------------------------------------------------------------------- #
#                             Helios UFS (.ufs)                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("version", ["Version2", "Version1"])
def test_parse_ufs_versions_and_scan_loading(tmp_path, version):
    ds_dir = tmp_path / f"helios_{version}"
    ds_dir.mkdir()
    main_ufs = _write_synthetic_ufs(ds_dir / "exp.ufs", version=version, inject_nan=True)
    _write_synthetic_ufs(ds_dir / "exp_scan1.ufs", version=version)
    _write_synthetic_ufs(ds_dir / "exp_scan2.ufs", version=version)

    assert _has_valid_ufs_header(main_ufs) is True
    bad = tmp_path / "dummy_zeros.ufs"
    bad.write_bytes(b"\x00" * 32)
    assert _has_valid_ufs_header(bad) is False

    delays, probe, mat, meta = parse_ufs(main_ufs)
    assert mat.shape == (4, 6)
    assert np.isfinite(mat).all()
    assert delays.shape == (4,)
    assert probe.shape == (6,)
    assert meta["unitsT"] == "ps"
    assert "Sample" in meta["fields"]

    ds = pm.load_1D(ds_dir, data_type="Helios_UFS")
    assert ds.Zavg_R.shape == (4, 6, 1)
    assert ds.nscans == 2
    assert ds.Zss_R.shape == (4, 6, 1, 2)
    assert np.isfinite(ds.Zavg_R).all()


@pytest.mark.skipif(not HELIOS_REAL_DIR.is_dir(), reason="HeliosTA real testData absent")
def test_real_helios_ufs_matches_csv():
    ds_ufs = pm.load_1D(HELIOS_REAL_DIR / "3-RF47-MeCN-1000nJ.ufs", data_type="Helios_UFS")
    ds_csv = pm.load_1D(HELIOS_REAL_DIR / "3-RF47-MeCN-1000nJ.csv", data_type="Helios_TA")

    assert ds_ufs.Zavg_R.shape == ds_csv.Zavg_R.shape == (200, 512, 1)
    assert ds_ufs.nscans == ds_csv.nscans == 10
    assert np.max(np.abs(ds_ufs.delays - ds_csv.delays)) < 1e-6
    assert np.max(np.abs(ds_ufs.probe - ds_csv.probe)) < 1e-6
    both_finite = np.isfinite(ds_ufs.Zavg_R) & np.isfinite(ds_csv.Zavg_R)
    assert np.max(np.abs(ds_ufs.Zavg_R[both_finite] - ds_csv.Zavg_R[both_finite])) < 1e-4


# --------------------------------------------------------------------------- #
#                         Glotaran .ascii & .nc                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("orientation", ["time", "wavelength"])
@pytest.mark.parametrize("signal_unit", ["mOD", "OD"])
def test_glotaran_ascii_roundtrip(tmp_path, orientation, signal_unit):
    delays = np.array([-0.5, 0.2, 1.5, 10.0])
    probe = np.linspace(1920.0, 2060.0, 7)
    z = np.outer(np.exp(-np.maximum(delays, 0.0) / 4.0), np.cos((probe - 1990.0) / 30.0))[
        :, :, None
    ]
    z = z * 25.0

    units = pm.helpers.units2dic("cm^{-1}", "ps", "x1E3")
    ds = Dataset1D(z, delays, probe, units, nscans=1, data_type="PDAT")

    out_path = ds.to_glotaran_ascii(
        tmp_path / f"test_{orientation}_{signal_unit}.ascii",
        orientation=orientation,
        signal_unit=signal_unit,
    )
    assert out_path.is_file()

    z_r, d_r, p_r, u_r, _, _, _ = read_glotaran_ascii(out_path)
    assert np.allclose(d_r, delays, atol=1e-6)
    assert np.allclose(p_r, probe, atol=1e-6)
    assert np.allclose(z_r[:, :, 0], z[:, :, 0], atol=1e-5)
    assert u_r["unitsL_lbl"] == "Wavenumber"

    ds_back = pm.load_1D(out_path, data_type="Glotaran_ASCII")
    assert ds_back.Zavg_R.shape == (4, 7, 1)


def test_glotaran_nc_cf_roundtrip(tmp_path):
    delays = np.array([0.0, 1.0, 5.0, 25.0])
    probe = np.linspace(450.0, 600.0, 5)
    z = np.arange(20, dtype=float).reshape(4, 5, 1) * 1.5
    stdv = np.ones((4, 5, 1)) * 0.25
    units = pm.helpers.units2dic("nm", "ps", "x1E3")
    ds = Dataset1D(z, delays, probe, units, Zstdv=stdv, data_type="PDAT")

    nc_path = ds.to_glotaran_nc(tmp_path / "glotaran_cf.nc", signal_unit="OD")
    assert nc_path.is_file()

    with h5py.File(nc_path, "r") as f:
        assert f.attrs["Conventions"] == "CF-1.8"
        assert "time" in f and "spectral" in f and "data" in f and "data_std" in f
        assert f["data"].attrs["units"] == "OD"

    z_r, d_r, p_r, u_r, _, _, zstd_r = read_glotaran_nc(nc_path)
    assert np.allclose(d_r, delays)
    assert np.allclose(p_r, probe)
    assert np.allclose(z_r, z, atol=1e-6)
    assert np.allclose(zstd_r, stdv, atol=1e-6)
    assert u_r["unitsL_lbl"] == "Wavelength"

    ds_back = pm.load_1D(nc_path, data_type="Glotaran_NC")
    assert np.allclose(ds_back.Zavg_R, z, atol=1e-6)


# --------------------------------------------------------------------------- #
#                 FAIR 1D & 2D HDF5 + History Contract #2                     #
# --------------------------------------------------------------------------- #
def test_fair_1d_hdf5_v2_roundtrip_history_and_v1_compat(tmp_path):
    src_file = tmp_path / "source_origin.pdat"
    src_file.write_text("dummy", encoding="utf-8")

    delays = np.array([-0.5, 0.0, 2.0, 10.0])
    probe = np.linspace(1920.0, 2020.0, 6)
    z = np.arange(24, dtype=float).reshape(4, 6, 1)
    z_ss = np.repeat(z[:, :, :, None], 2, axis=3)
    counts = np.full_like(z, 1200.0)

    units = pm.helpers.units2dic("cm^{-1}", "ps", "x1E3")
    units["sample_info"] = "IrHCOP3 in DCM"
    ds = Dataset1D(
        z,
        delays,
        probe,
        units,
        nscans=2,
        Zss_R=z_ss,
        Zstdv=np.ones_like(z) * 0.1,
        source=str(src_file),
        data_type="UoS_IRpp",
        counts=counts,
    )
    ds.background_correct(-1.0, -0.1)
    ds.mask_ranges = [(1940.0, 1950.0)]
    ds.chirp_fit = ChirpFit(
        mode="manual",
        coeffs=np.array([0.1, 1e-4]),
        fit_wl=probe[:3],
        Pfit=np.array([0.0, 0.2, 0.5]),
        fit_pixels=np.array([0, 1, 2]),
    )
    ds._scatter_removal_params = {"method": "svs", "k": 2}

    history_payload = [{"op": "background_correct", "params": {"n_neg": 1}}]
    restored_history: list[dict] = []
    ds.history_to_dicts = lambda: list(history_payload)

    h5_path = save_1D_hdf5(ds, tmp_path / "archive_1d.h5")
    with h5py.File(h5_path, "r") as f:
        assert f.attrs["NX_class"] == "NXroot"
        assert f.attrs["pymorgan_format"] == "1D"
        assert f.attrs["format_version"] == "2.0"
        assert "source_sha256" in f.attrs
        assert json.loads(f.attrs["history_json"]) == history_payload
        assert f["Zavg_R"].compression == "gzip"

    loaded = load_1D_hdf5(h5_path)
    assert loaded.counts is not None and np.allclose(loaded.counts, counts)
    assert loaded.mask_ranges == [(1940.0, 1950.0)]
    assert loaded.chirp_fit is not None and np.allclose(loaded.chirp_fit.coeffs, [0.1, 1e-4])
    assert loaded._scatter_removal_params == {"method": "svs", "k": 2}

    # Test restore_dataset_history hook on a class supporting load_history_from_dicts
    with h5py.File(h5_path, "r") as f:
        from pymorgan.hdf5_meta import restore_dataset_history

        class _DummyHist:
            def load_history_from_dicts(self, items):
                restored_history.extend(items)

        restore_dataset_history(f, _DummyHist())
    assert restored_history == history_payload

    # Legacy v1.0 HDF5 without format_version / history_json loads cleanly
    v1_path = tmp_path / "legacy_v1.h5"
    with h5py.File(v1_path, "w") as f:
        f.create_dataset("delays", data=delays)
        f.create_dataset("probe", data=probe)
        f.create_dataset("Zavg_R", data=z)
    v1_loaded = load_1D_hdf5(v1_path)
    assert np.allclose(v1_loaded.Zavg_R, z)


def test_fair_2d_hdf5_roundtrip_and_cross_type_guard(tmp_path):
    pump = np.linspace(1920.0, 2040.0, 6)
    probe1 = np.linspace(1930.0, 2030.0, 5)
    probe2 = np.linspace(1700.0, 1800.0, 4)
    delays = np.array([0.5, 2.0])
    z1 = np.ones((6, 5, 2)) * 2.0
    z2 = np.ones((6, 4, 2)) * -3.0

    ds2d = Dataset2D(
        z1,
        pump,
        probe1,
        delays,
        {},
        datatype="shaper",
        raw_signal=np.ones((6, 5, 2)) * 4.0,
        raw_t1delays=np.arange(6, dtype=float) * 15.0,
        w0_cm1=1950.0,
        dt1_fs=15.0,
        detector_names=["DET1", "DET2"],
        detector_probes={"DET1": probe1, "DET2": probe2},
        detector_data={"DET1": z1, "DET2": z2},
        active_detector="DET1",
        data_type="UoS_2DIR",
    )
    history_2d = [{"op": "process", "params": {"zero_pad": 4}}]
    ds2d.history_to_dicts = lambda: list(history_2d)

    out_h5 = ds2d.to_hdf5(tmp_path / "archive_2d.h5")
    with h5py.File(out_h5, "r") as f:
        assert f.attrs["NX_class"] == "NXroot"
        assert f.attrs["pymorgan_format"] == "2D"
        assert f.attrs["format_version"] == "2.0"
        assert json.loads(f.attrs["history_json"]) == history_2d

    res = read_2D_hdf5(out_h5)
    assert res[0].shape == (6, 5, 2)
    loaded2d = load_2D_hdf5(out_h5)
    assert loaded2d.detector_names == ["DET1", "DET2"]
    assert loaded2d.set_active_detector("DET2") is True
    assert np.allclose(loaded2d.probe, probe2)

    with pytest.raises(ValueError, match="2D PyMORGAN HDF5"):
        load_1D_hdf5(out_h5)

    ds1d = Dataset1D(np.ones((2, 3, 1)), np.array([0.0, 1.0]), np.array([1.0, 2.0, 3.0]), {})
    h5_1d = save_1D_hdf5(ds1d, tmp_path / "only_1d.h5")
    with pytest.raises(ValueError, match="1D PyMORGAN HDF5"):
        load_2D_hdf5(h5_1d)


def test_gui_export_hdf5_and_glotaran_actions(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    from pymorgan.gui.main_window import MainWindow

    delays = np.array([0.0, 1.0, 5.0])
    probe = np.linspace(1920.0, 2000.0, 4)
    z1d = np.ones((3, 4, 1)) * 5.0
    ds1d = Dataset1D(z1d, delays, probe, pm.helpers.units2dic("cm^{-1}", "ps", "x1E3"))
    ds2d = Dataset2D(
        np.ones((4, 4, 3)),
        probe,
        probe,
        delays,
        {},
        datatype="processed",
        is_frequency_domain=True,
    )

    monkeypatch.setattr(QMessageBox, "information", lambda *a, **kw: QMessageBox.StandardButton.Ok)

    win = MainWindow()
    try:
        assert hasattr(win, "actionExportHDF5")
        assert hasattr(win, "actionExportGlotaran")
        assert win.actionExportHDF5.isEnabled() is False
        assert win.actionExportGlotaran.isEnabled() is False

        win.MainTabs.setCurrentIndex(0)
        win.dataset = ds1d
        win._current_path = str(tmp_path / "sample1d.pdat")
        win._update_export_actions_state()
        assert win.actionExportHDF5.isEnabled() is True
        assert win.actionExportGlotaran.isEnabled() is True

        h5_1d_out = tmp_path / "gui_1d.h5"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **kw: (str(h5_1d_out), ""))
        win._export_hdf5()
        assert h5_1d_out.is_file()

        ascii_out = tmp_path / "gui_1d.ascii"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **kw: (str(ascii_out), ""))
        win._export_glotaran()
        assert ascii_out.is_file()

        win.twoD_dataset = ds2d
        win._twoD_current_path = str(tmp_path / "sample2d.p2dat")
        win.MainTabs.setCurrentIndex(1)
        assert win.actionExportHDF5.isEnabled() is True
        assert win.actionExportGlotaran.isEnabled() is False

        h5_2d_out = tmp_path / "gui_2d.h5"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **kw: (str(h5_2d_out), ""))
        win._export_hdf5()
        assert h5_2d_out.is_file()
    finally:
        win.close()


# --------------------------------------------------------------------------- #
#           Helios UFS Edge Cases & load_1D/load_2D Auto-Detection            #
# --------------------------------------------------------------------------- #
def test_ufs_descending_probe_sorted_ascending(tmp_path):
    probe_desc = np.array([700.0, 620.0, 540.0, 460.0])
    delays = np.array([-1.0, 0.5, 2.0])
    # Distinct column values per probe point to verify column permutation matches axis sort
    sig = np.outer(np.array([1.0, 2.0, 3.0]), np.array([70.0, 62.0, 54.0, 46.0]))
    ufs_path = _write_synthetic_ufs(
        tmp_path / "desc.ufs",
        probe=probe_desc,
        delays=delays,
        signal=sig,
        z_unit="DA",
    )

    _, probe_out, sig_out, _ = parse_ufs(ufs_path)
    assert np.all(np.diff(probe_out) > 0)
    assert np.allclose(probe_out, probe_desc[::-1])
    assert np.allclose(sig_out, sig[:, ::-1], atol=1e-9)

    ds = pm.load_1D(ufs_path, data_type="Helios_UFS")
    assert np.allclose(ds.probe, probe_desc[::-1])
    assert np.allclose(ds.Zavg_R[:, :, 0], sig[:, ::-1], atol=1e-9)


def test_ufs_pix_unit_normalized_to_px(tmp_path):
    probe = np.arange(10.0, 16.0)
    delays = np.array([-1.0, 0.0, 2.0, 5.0])
    sig = np.ones((4, 6)) * 3.0
    mat_wl_t = np.ascontiguousarray((sig / 1000.0).T)

    buf = bytearray()
    buf += _pack_ufs_string("Version2")
    buf += _pack_ufs_string("Pixels")
    buf += _pack_ufs_string("pix")
    buf += struct.pack(">i", len(probe))
    buf += probe.astype(">f8").tobytes()
    buf += _pack_ufs_string("Time")
    buf += _pack_ufs_string("ps")
    buf += struct.pack(">i", len(delays))
    buf += delays.astype(">f8").tobytes()
    buf += _pack_ufs_string("Absorption")
    buf += _pack_ufs_string("DA")
    buf += struct.pack(">ii", mat_wl_t.shape[0], mat_wl_t.shape[1])
    buf += mat_wl_t.astype(">f8").tobytes()
    ufs_path = tmp_path / "pixels.ufs"
    ufs_path.write_bytes(bytes(buf))

    _, _, _, meta = parse_ufs(ufs_path)
    assert meta["unitsL"] == "px"

    ds = pm.load_1D(ufs_path, data_type="Helios_UFS")
    assert ds.units["unitsL_ltx"] == "px"
    assert ds.units["unitsL_lbl"] == "Pixel"
    assert pm.helpers.units2dic("pixels", "ps", "mOD")["unitsL_ltx"] == "px"


def test_ufs_quality_masking_tiny_signal_preserved_and_warning(tmp_path, caplog):
    import logging

    probe = np.linspace(450.0, 650.0, 5)
    delays = np.array([-0.5, 0.0, 1.0, 5.0])
    # All legitimate small/normal signals first: must keep status="ok" and preserve 1e-10 mOD
    clean_sig = np.full((4, 5), 2.5)
    clean_sig[1, 1] = 1e-10
    clean_path = _write_synthetic_ufs(
        tmp_path / "clean_tiny.ufs",
        probe=probe,
        delays=delays,
        signal=clean_sig,
        z_unit="mOD",
    )
    _, _, clean_out, clean_meta = parse_ufs(clean_path)
    assert clean_meta["ufs_quality"]["n_filled"] == 0
    assert clean_meta["ufs_quality"]["status"] == "ok"
    assert np.isclose(clean_out[1, 1], 1e-10, rtol=1e-6)

    # Inject >20% corrupted cells: NaN, extreme spike >=1e5 mOD, and IEEE subnormal <1e-300
    bad_sig = np.full((4, 5), 4.0)
    bad_sig[0, :] = np.nan
    bad_sig[1, 0] = 2.0e6
    bad_sig[2, 3] = 1e-315  # 7 / 20 = 35% > 0.2
    bad_path = _write_synthetic_ufs(
        tmp_path / "corrupted.ufs",
        probe=probe,
        delays=delays,
        signal=bad_sig,
        z_unit="mOD",
    )

    _, _, fixed_out, bad_meta = parse_ufs(bad_path)
    qual = bad_meta["ufs_quality"]
    assert qual["status"] == "cleaned"
    assert qual["n_filled"] >= 7
    assert qual["filled_fraction"] > 0.2
    assert np.all(np.isfinite(fixed_out))
    assert np.max(np.abs(fixed_out)) < 100.0

    with caplog.at_level(logging.WARNING, logger="pymorgan"):
        ds = pm.load_1D(bad_path, data_type="Helios_UFS")
    assert ds.units["ufs_quality"]["filled_fraction"] > 0.2
    assert "padded/invalid cells interpolated" in ds.units.get("sample_info_full", "")
    assert any("invalid/padded cells interpolated" in rec.message for rec in caplog.records)


def test_ufs_loader_alias_and_load_1d(tmp_path):
    from pymorgan.oneD.load import get_loader, read_Helios_UFS

    assert get_loader("UFS") is read_Helios_UFS
    assert get_loader("ufs") is read_Helios_UFS

    ufs_path = _write_synthetic_ufs(tmp_path / "alias.ufs")
    ds_alias = pm.load_1D(ufs_path, data_type="UFS")
    ds_full = pm.load_1D(ufs_path, data_type="Helios_UFS")
    assert np.allclose(ds_alias.Zavg_R, ds_full.Zavg_R)


def test_ufs_folder_filter_and_bg_chirp_fallback(tmp_path):
    from pymorgan.oneD.load import _find_helios_main_ufs, _helios_ufs_file_filter

    # Folder with ONLY processed -bg -chirp.ufs falls back cleanly
    only_proc = tmp_path / "only_processed"
    only_proc.mkdir()
    proc_file = _write_synthetic_ufs(only_proc / "meas-bg -chirp.ufs")
    corr_file = _write_synthetic_ufs(only_proc / "meas-CORRECTED.ufs")
    assert _helios_ufs_file_filter(proc_file) is False
    assert _helios_ufs_file_filter(corr_file) is False
    assert _find_helios_main_ufs(only_proc) in {proc_file, corr_file}
    ds_fallback = pm.load_1D(only_proc, data_type="Helios_UFS")
    assert isinstance(ds_fallback, Dataset1D)

    # Folder with both raw and processed prefers the raw .ufs and shows only raw in filter
    both_dir = tmp_path / "raw_and_proc"
    both_dir.mkdir()
    both_proc = _write_synthetic_ufs(both_dir / "meas-bg -chirp.ufs")
    raw_file = _write_synthetic_ufs(both_dir / "meas.ufs")
    assert _helios_ufs_file_filter(raw_file) is True
    assert _helios_ufs_file_filter(both_proc) is False
    assert _find_helios_main_ufs(both_dir) == raw_file


def test_ufs_converted_matrix_fallback(tmp_path):
    probe = np.linspace(450.0, 650.0, 6)
    delays = np.array([-0.5, 0.0, 1.0, 5.0])
    sig_mod = np.outer(np.array([0.2, 1.0, 2.0, 0.5]), np.linspace(1.0, 6.0, 6))
    mat_wl_t = np.ascontiguousarray((sig_mod / 1000.0).T)

    # Third-party CSV->UFS writer: omit z_unit string and supply mismatched (n1, n2)
    buf = bytearray()
    buf += _pack_ufs_string("Version2")
    buf += _pack_ufs_string("Wavelength")
    buf += _pack_ufs_string("nm")
    buf += struct.pack(">i", len(probe))
    buf += probe.astype(">f8").tobytes()
    buf += _pack_ufs_string("Time")
    buf += _pack_ufs_string("ps")
    buf += struct.pack(">i", len(delays))
    buf += delays.astype(">f8").tobytes()
    buf += _pack_ufs_string("Absorption")
    buf += _pack_ufs_string("DA")
    buf += struct.pack(">ii", 999, 999)  # wrong dimensions -> triggers _fallback_converted_ufs_matrix
    buf += mat_wl_t.astype(">f8").tobytes()
    buf += _pack_ufs_string("Sample: ConvertedCSV\rComments: Third-party")
    conv_path = tmp_path / "converted.ufs"
    conv_path.write_bytes(bytes(buf))

    delays_out, probe_out, sig_out, meta = parse_ufs(conv_path)
    assert np.allclose(delays_out, delays)
    assert np.allclose(probe_out, probe)
    assert np.allclose(sig_out, sig_mod, atol=1e-6)
    assert "ConvertedCSV" in meta.get("sample_info_full", "")


def test_load_1d_and_2d_auto_detect_and_format_hint(tmp_path):
    import synthetic

    ufs_path = _write_synthetic_ufs(tmp_path / "auto.ufs")
    pdat_info = synthetic.make_synthetic_pdat(tmp_path / "auto.pdat")
    p2_info = synthetic.make_synthetic_p2dat(tmp_path / "auto.p2dat")

    # Omitting data_type or passing "auto" resolves the true loader
    ds_ufs_auto = pm.load_1D(ufs_path)
    assert ds_ufs_auto.data_type == "Helios_UFS"
    ds_pdat_auto = pm.load_1D(pdat_info["path"], data_type="auto")
    assert ds_pdat_auto.data_type == "PDAT"

    ds_2d_auto = pm.load_2D(p2_info["path"])
    assert isinstance(ds_2d_auto, Dataset2D)
    h5_2d = ds_2d_auto.to_hdf5(tmp_path / "auto_2d.h5")
    ds_2d_h5 = pm.load_2D(h5_2d, data_type="auto")
    assert isinstance(ds_2d_h5, Dataset2D)

    # Mismatched explicit data_type raises ValueError with detected format hint
    with pytest.raises(ValueError, match="Helios_UFS"):
        pm.load_1D(ufs_path, data_type="PDAT")

    with pytest.raises(ValueError, match="HDF5_2D|Detected format"):
        pm.load_2D(h5_2d, data_type="P2DAT")
