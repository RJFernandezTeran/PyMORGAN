"""Unit and integration tests for PyMESS pump-probe and anisotropy loader in PyMORGAN."""

from pathlib import Path

import h5py
import numpy as np
import pytest

import pymorgan as pm
from pymorgan import load_1D, load_pymess_anisotropy
from pymorgan.oneD.load import (
    _pymess_pp_is_file,
    read_PyMESS_PP,
)


@pytest.fixture
def standard_pp_file(tmp_path: Path) -> Path:
    """Create a mock standard PyMESS pump-probe .h5 file."""
    p = tmp_path / "standard_scan.h5"
    n_delays, n_pix = 15, 32
    delays_fs = np.linspace(-500.0, 5000.0, n_delays)
    wl = np.linspace(400.0, 700.0, n_pix)
    delta_A = np.outer(np.exp(-delays_fs / 2000.0), np.sin(np.linspace(0, 3, n_pix))) * 12.0

    with h5py.File(p, "w") as f:
        grp = f.create_group("data")
        grp.create_dataset("delta_A", data=delta_A)
        grp.create_dataset("delays_fs", data=delays_fs)
        grp.create_dataset("wavelengths", data=wl)
        grp.create_dataset("delta_A_noise", data=np.full_like(delta_A, 0.05))

        meta = f.create_group("metadata")
        meta.attrs["measurement_type"] = "pump_probe"
        meta.attrs["sample_id"] = "Rhodamine_6G"
        meta.attrs["pump_wl_nm"] = "532.0"
        meta.attrs["setup_name"] = "Ultrafast Spectroscopy Workstation"

    return p


@pytest.fixture
def slow_mod_anisotropy_file(tmp_path: Path) -> Path:
    """Create a mock slow-modulated polarization scan (Parallel and Perpendicular)."""
    p = tmp_path / "anisotropy_scan.h5"
    n_delays, n_pix = 10, 16
    delays_fs = np.linspace(-200.0, 2000.0, n_delays)
    wl = np.linspace(1900.0, 2100.0, n_pix)  # Mid-IR TRIR (cm-1)

    # 2 scans x 2 states (Par, Perp) = 4 passes
    indiv = np.zeros((4, n_delays, n_pix), dtype=np.float64)
    # State 0 (Parallel) = 10.0 mOD
    indiv[0] = 10.0
    indiv[2] = 10.0
    # State 1 (Perpendicular) = 4.0 mOD
    indiv[1] = 4.0
    indiv[3] = 4.0

    # Theoretical:
    # Isotropic: (10 + 2*4) / 3 = 18 / 3 = 6.0 mOD
    # Anisotropy: (10 - 4) / 18 = 6 / 18 = 0.333333

    with h5py.File(p, "w") as f:
        grp = f.create_group("data")
        grp.create_dataset("delta_A", data=np.mean(indiv, axis=0))
        grp.create_dataset("delays_fs", data=delays_fs)
        grp.create_dataset("wavelengths", data=wl)
        grp.create_dataset("individual_scans", data=indiv)

        # Pre-calculated group
        grp_slow = f.create_group("data/slow_modulation")
        grp_slow.create_dataset("delta_A_parallel", data=np.full((n_delays, n_pix), 10.0))
        grp_slow.create_dataset("delta_A_perpendicular", data=np.full((n_delays, n_pix), 4.0))
        grp_slow.create_dataset("isotropic", data=np.full((n_delays, n_pix), 6.0))
        grp_slow.create_dataset("anisotropy", data=np.full((n_delays, n_pix), 1.0 / 3.0))

        meta = f.create_group("metadata")
        meta.attrs["measurement_type"] = "pump_probe"
        meta.attrs["slow_modulation_enabled"] = "True"
        meta.attrs["slow_modulation_num_states"] = "2"
        meta.attrs["slow_modulation_state_names"] = "Parallel,Perpendicular"
        meta.attrs["sample_id"] = "ReCO_DCM"

    return p


@pytest.fixture
def slow_mod_no_precalc_file(tmp_path: Path) -> Path:
    """Create a mock slow-modulated polarization scan without pre-calculated surfaces."""
    p = tmp_path / "anisotropy_scan_raw.h5"
    n_delays, n_pix = 8, 12
    delays_fs = np.linspace(-100.0, 1000.0, n_delays)
    wl = np.linspace(500.0, 600.0, n_pix)

    # 3 scans x 2 states (Parallel, Perpendicular) = 6 passes
    indiv = np.zeros((6, n_delays, n_pix), dtype=np.float64)
    # Even passes (0, 2, 4) = Parallel: 8.0 mOD
    indiv[0::2] = 8.0
    # Odd passes (1, 3, 5) = Perpendicular: 2.0 mOD
    indiv[1::2] = 2.0

    # Put a region with very small signal to test threshold masking
    indiv[:, 0, :] = 0.01

    with h5py.File(p, "w") as f:
        grp = f.create_group("data")
        grp.create_dataset("delta_A", data=np.mean(indiv, axis=0))
        grp.create_dataset("delays_fs", data=delays_fs)
        grp.create_dataset("wavelengths", data=wl)
        grp.create_dataset("individual_scans", data=indiv)

        meta = f.create_group("metadata")
        meta.attrs["measurement_type"] = "pump_probe"
        meta.attrs["slow_modulation_enabled"] = "True"
        meta.attrs["slow_modulation_num_states"] = "2"
        meta.attrs["slow_modulation_state_names"] = "Parallel,Perpendicular"
        meta.attrs["sample_id"] = "Test_Raw"

    return p


def test_pymess_file_filter(standard_pp_file: Path, slow_mod_anisotropy_file: Path, tmp_path: Path):
    """Verify that file detector recognizes pump-probe files and rejects other types."""
    assert _pymess_pp_is_file(standard_pp_file) is True
    assert _pymess_pp_is_file(slow_mod_anisotropy_file) is True

    # Non-HDF5 file
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("hello")
    assert _pymess_pp_is_file(txt_file) is False

    # 2D spectroscopy file
    two_d = tmp_path / "2d.h5"
    with h5py.File(two_d, "w") as f:
        f.create_group("2D_Spectra")
        m = f.create_group("metadata")
        m.attrs["measurement_type"] = "2d_spectra"
    assert _pymess_pp_is_file(two_d) is False


def test_read_standard_pymess_pp(standard_pp_file: Path):
    """Verify standard pump-probe loader output."""
    Zavg_R, delays, probe, Units, Nscans, Zss_R, Zstdv = read_PyMESS_PP(str(standard_pp_file))

    assert Zavg_R.shape == (15, 32, 1)
    assert len(delays) == 15
    assert len(probe) == 32
    assert delays[0] == pytest.approx(-0.5)  # -500 fs converted to -0.5 ps
    assert delays[-1] == pytest.approx(5.0)  # 5000 fs converted to 5.0 ps
    assert Units["T"] == "ps"
    assert Units["L"] == "nm"
    assert Units["Z"] == "x1E3"
    assert Units["sample"] == "Rhodamine_6G"
    assert Units["pump_wl"] == "532.0"

    # Also test via pm.load_1D
    ds = load_1D(standard_pp_file, data_type="PyMESS_PP")
    assert ds.Zavg_R.shape == (15, 32, 1)
    assert ds.Units["sample"] == "Rhodamine_6G"


def test_read_slow_mod_multi_detector(slow_mod_anisotropy_file: Path):
    """Verify multi-state slow modulation loaded into multi-detector dimension."""
    Zavg_R, delays, probe, Units, Nscans, Zss_R, Zstdv = read_PyMESS_PP(str(slow_mod_anisotropy_file))

    assert Zavg_R.shape == (10, 16, 2)
    assert Zss_R.shape == (10, 16, 2, 2)  # 2 repeats
    assert Nscans == 2
    assert Units["L"] == "cm-1"
    assert Units["detectors"] == ["Parallel", "Perpendicular"]

    # Channel 0 (Parallel)
    np.testing.assert_allclose(Zavg_R[:, :, 0], 10.0)
    # Channel 1 (Perpendicular)
    np.testing.assert_allclose(Zavg_R[:, :, 1], 4.0)


def test_load_pymess_anisotropy_bundle(slow_mod_anisotropy_file: Path):
    """Verify dedicated anisotropy bundle unpacks into 4 valid Dataset1D instances."""
    bundle = load_pymess_anisotropy(str(slow_mod_anisotropy_file), G_factor=1.0, threshold_mOD=0.05)

    assert "parallel" in bundle
    assert "perpendicular" in bundle
    assert "isotropic" in bundle
    assert "anisotropy" in bundle

    ds_par = bundle["parallel"]
    ds_perp = bundle["perpendicular"]
    ds_iso = bundle["isotropic"]
    ds_aniso = bundle["anisotropy"]

    # Verify theoretical values
    np.testing.assert_allclose(ds_par.Zavg_R[:, :, 0], 10.0)
    np.testing.assert_allclose(ds_perp.Zavg_R[:, :, 0], 4.0)
    np.testing.assert_allclose(ds_iso.Zavg_R[:, :, 0], 6.0)
    np.testing.assert_allclose(ds_aniso.Zavg_R[:, :, 0], 1.0 / 3.0, rtol=1e-5)

    assert ds_aniso.Units["Z"] == "dimensionless"
    assert ds_iso.Units["Z"] == "x1E3"


def test_load_pymess_anisotropy_calculated_and_g_factor(slow_mod_no_precalc_file: Path):
    """Verify on-the-fly calculation when pre-calculated surfaces are absent and G-factor applied."""
    # With G_factor = 1.0:
    # Par = 8.0, Perp = 2.0
    # Iso = (8.0 + 2*2.0)/3 = 4.0
    # Aniso = (8.0 - 2.0) / (8.0 + 4.0) = 6/12 = 0.5
    bundle = pm.load_pymess_anisotropy(slow_mod_no_precalc_file, G_factor=1.0, threshold_mOD=0.1)

    ds_par = bundle["parallel"]
    ds_perp = bundle["perpendicular"]
    ds_iso = bundle["isotropic"]
    ds_aniso = bundle["anisotropy"]

    # Check non-masked rows (delay index >= 1)
    np.testing.assert_allclose(ds_par.Zavg_R[1:, :, 0], 8.0)
    np.testing.assert_allclose(ds_perp.Zavg_R[1:, :, 0], 2.0)
    np.testing.assert_allclose(ds_iso.Zavg_R[1:, :, 0], 4.0)
    np.testing.assert_allclose(ds_aniso.Zavg_R[1:, :, 0], 0.5)

    # First row had signal 0.01 mOD (< threshold 0.1 mOD) -> must be NaN in anisotropy
    assert np.isnan(ds_aniso.Zavg_R[0, :, 0]).all()

    # Now test with G_factor = 1.5:
    # Par = 8.0, Perp = 2.0, G = 1.5
    # Iso = (8.0 + 2 * 1.5 * 2.0) / 3 = (8 + 6) / 3 = 14 / 3 = 4.666667
    # Aniso = (8.0 - 1.5 * 2.0) / (8.0 + 2 * 1.5 * 2.0) = (8 - 3) / 14 = 5 / 14 = 0.357143
    bundle_g = pm.load_pymess_anisotropy(slow_mod_no_precalc_file, G_factor=1.5, threshold_mOD=0.1)
    np.testing.assert_allclose(bundle_g["isotropic"].Zavg_R[1:, :, 0], 14.0 / 3.0)
    np.testing.assert_allclose(bundle_g["anisotropy"].Zavg_R[1:, :, 0], 5.0 / 14.0)


def test_load_pymess_anisotropy_export():
    """Verify load_pymess_anisotropy is exposed at both pymorgan and pymorgan.oneD levels."""
    assert hasattr(pm, "load_pymess_anisotropy")
    assert hasattr(pm.oneD, "load_pymess_anisotropy")
    assert pm.load_pymess_anisotropy is pm.oneD.load_pymess_anisotropy


def test_read_real_20261007_datasets():
    """Verify loading real datasets recorded on 20261007 if present."""
    import os
    env_dir = os.environ.get("PYMORGAN_TESTDATA_DIR")
    if not env_dir:
        pytest.skip("PYMORGAN_TESTDATA_DIR environment variable not set")
    base_dir = Path(env_dir) / "PyMESS" / "20261007"
    if not base_dir.is_dir():
        pytest.skip(f"Directory {base_dir} not found")

    files = [
        "ReCO_DCM_again_IRpp_181712.h5",
        "ReCO_DCM_again_IRpp_181830.h5",
        "ReCO_DCM_again_IRpp_narrowPump_182920.h5",
    ]
    for fname in files:
        p = base_dir / fname
        if not p.is_file():
            continue

        ds = pm.load_1D(p, data_type="PyMESS_PP")
        assert ds.data_type == "PyMESS_PP"
        assert ds.Zavg_R.ndim == 3
        assert ds.Zavg_R.shape[2] == 1  # Standard single-channel scan
        assert ds.nscans >= 1
        assert ds.delays.shape[0] == ds.Zavg_R.shape[0]
        assert ds.probe.shape[0] == ds.Zavg_R.shape[1]
        assert ds.Units["L"] == "cm-1"
        assert ds.Units["T"] == "ps"
        assert ds.Units["Z"] == "x1E3"

        # Background correction test
        ds.background_correct(tmin=-50.0, tmax=-10.0)
        assert ds.Zavg_C.shape == ds.Zavg_R.shape
        assert np.isfinite(ds.Zavg_C).all()


def test_pymess_pp_describe(standard_pp_file: Path, slow_mod_anisotropy_file: Path):
    """Verify PyMESS describer reports correct state counts and polarization names."""
    from pymorgan.oneD.load import describe_dataset

    # Standard file: 1 state
    desc_std = describe_dataset("PyMESS_PP", standard_pp_file)
    assert desc_std is not None
    assert desc_std["n_slowmod"] == 1
    assert desc_std["is_pymess"] is True

    # Slow mod anisotropy file: 2 states (Parallel, Perpendicular)
    desc_sm = describe_dataset("PyMESS_PP", slow_mod_anisotropy_file)
    assert desc_sm is not None
    assert desc_sm["n_slowmod"] == 2
    assert desc_sm["state_names"] == ["Parallel", "Perpendicular"]
    assert desc_sm["is_pymess"] is True


def test_new_schema_slow_mod_state_00_and_intensity_mode(tmp_path: Path):
    """Verify new PyMESS .h5 schema (state_00_<name> without individual_scans, plus Intensity mode)."""
    from pymorgan.oneD.load import describe_dataset

    # 1. Slow-modulated pump-probe with state_00_Parallel / state_01_Perpendicular and no individual_scans
    p_sm = tmp_path / "new_schema_aniso.h5"
    n_delays, n_pix = 10, 16
    delays_fs = np.linspace(-200.0, 2000.0, n_delays)
    wl = np.linspace(1900.0, 2100.0, n_pix)
    par = np.full((n_delays, n_pix), 12.0)
    perp = np.full((n_delays, n_pix), 3.0)
    iso = (par + 2.0 * perp) / 3.0
    aniso = (par - perp) / (par + 2.0 * perp)

    with h5py.File(p_sm, "w") as f:
        grp = f.create_group("data")
        grp.create_dataset("delta_A", data=0.5 * (par + perp))
        grp.create_dataset("delays_fs", data=delays_fs)
        grp.create_dataset("wavelengths", data=wl)
        grp_slow = f.create_group("data/slow_modulation")
        grp_slow.create_dataset("state_00_Parallel", data=par)
        grp_slow.create_dataset("state_01_Perpendicular", data=perp)
        grp_slow.create_dataset("isotropic", data=iso)
        grp_slow.create_dataset("anisotropy", data=aniso)

        meta = f.create_group("metadata")
        meta.attrs["measurement_type"] = "pump_probe"
        meta.attrs["slow_modulation_enabled"] = True
        meta.attrs["slow_modulation_num_states"] = 2
        meta.attrs["slow_modulation_state_names"] = "Parallel,Perpendicular"
        meta.attrs["num_scans_completed"] = 3
        meta.attrs["sample_id"] = "NewSchema_Aniso"

    desc = describe_dataset("PyMESS_PP", p_sm)
    assert desc is not None
    assert desc["n_slowmod"] == 2
    assert desc["state_names"] == ["Parallel", "Perpendicular"]

    ds_sm = load_1D(p_sm, data_type="PyMESS_PP")
    assert ds_sm.Zavg_R.shape == (n_delays, n_pix, 2)
    assert ds_sm.Units["detectors"] == ["Parallel", "Perpendicular"]
    np.testing.assert_allclose(ds_sm.Zavg_R[:, :, 0], 12.0)
    np.testing.assert_allclose(ds_sm.Zavg_R[:, :, 1], 3.0)

    bundle = load_pymess_anisotropy(p_sm, G_factor=1.0, threshold_mOD=0.05)
    np.testing.assert_allclose(bundle["parallel"].Zavg_R[:, :, 0], 12.0)
    np.testing.assert_allclose(bundle["perpendicular"].Zavg_R[:, :, 0], 3.0)
    np.testing.assert_allclose(bundle["isotropic"].Zavg_R[:, :, 0], 6.0)
    np.testing.assert_allclose(bundle["anisotropy"].Zavg_R[:, :, 0], 0.5)

    # 2. Intensity-mode pump-probe scan (/data/intensity instead of /data/delta_A)
    p_int = tmp_path / "new_schema_intensity.h5"
    intensity_mat = np.full((n_delays, n_pix), 2.5)
    with h5py.File(p_int, "w") as f:
        grp = f.create_group("data")
        grp.create_dataset("intensity", data=intensity_mat)
        grp.create_dataset("delays_fs", data=delays_fs)
        grp.create_dataset("wavelengths", data=wl)
        meta = f.create_group("metadata")
        meta.attrs["measurement_type"] = "pump_probe"
        meta.attrs["save_as"] = "Intensity"
        meta.attrs["sample_id"] = "Intensity_Test"

    assert _pymess_pp_is_file(p_int) is True
    ds_int = load_1D(p_int, data_type="PyMESS_PP")
    assert ds_int.Zavg_R.shape == (n_delays, n_pix, 1)
    np.testing.assert_allclose(ds_int.Zavg_R[:, :, 0], 2.5)
    np.testing.assert_allclose(ds_int.Units["counts"], 2.5)

    # 3. Multi-signal fast modulation (/data/signals/signal_01_*, signal_02_*)
    p_sig = tmp_path / "new_schema_multisig.h5"
    s1 = np.full((n_delays, n_pix), 5.0)
    s2 = np.full((n_delays, n_pix), -2.0)
    with h5py.File(p_sig, "w") as f:
        grp = f.create_group("data")
        grp.create_dataset("delta_A", data=s1)
        grp.create_dataset("delays_fs", data=delays_fs)
        grp.create_dataset("wavelengths", data=wl)
        sig_grp = f.create_group("data/signals")
        ds1 = sig_grp.create_dataset("signal_01_Pump_Only", data=s1)
        ds1.attrs["label"] = "S1: Pump Only"
        ds2 = sig_grp.create_dataset("signal_02_Push_Only", data=s2)
        ds2.attrs["label"] = "S2: Push Only"
        meta = f.create_group("metadata")
        meta.attrs["measurement_type"] = "pump_probe"

    desc_sig = describe_dataset("PyMESS_PP", p_sig)
    assert desc_sig is not None
    assert desc_sig["n_slowmod"] == 2
    assert desc_sig["state_names"] == ["S1: Pump Only", "S2: Push Only"]
    ds_sig = load_1D(p_sig, data_type="PyMESS_PP")
    assert ds_sig.Zavg_R.shape == (n_delays, n_pix, 2)
    assert ds_sig.Units["detectors"] == ["S1: Pump Only", "S2: Push Only"]
    np.testing.assert_allclose(ds_sig.Zavg_R[:, :, 0], 5.0)
    np.testing.assert_allclose(ds_sig.Zavg_R[:, :, 1], -2.0)


def _write_basic_pp_h5(path: Path, wavelengths: np.ndarray, n_delays: int = 8) -> Path:
    """Create a minimal PyMESS pump-probe .h5 file at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_pix = len(wavelengths)
    delays_fs = np.linspace(-200.0, 1000.0, n_delays)
    delta_A = np.ones((n_delays, n_pix), dtype=np.float64) * 2.5
    with h5py.File(path, "w") as f:
        grp = f.create_group("data")
        grp.create_dataset("delta_A", data=delta_A)
        grp.create_dataset("delays_fs", data=delays_fs)
        grp.create_dataset("wavelengths", data=wavelengths)
        meta = f.create_group("metadata")
        meta.attrs["measurement_type"] = "pump_probe"
        meta.attrs["sample_id"] = "CSV_Test"
    return path


def _write_probe_csv(path: Path, values: np.ndarray, header: bool = False) -> Path:
    """Write 1D probe calibration values to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Wavelength"] if header else []
    lines.extend(f"{val:.6f}" for val in values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_pymess_pp_calibrated_probe_csv_precedence_and_mismatch(tmp_path: Path):
    """CalibratedProbe.csv overrides internal .h5 wavelengths with data-folder priority and ignores length mismatches."""
    n_pix = 32
    orig_wl = np.linspace(400.0, 700.0, n_pix)
    csv_root = np.linspace(405.0, 705.0, n_pix)
    csv_cal_chirp = np.linspace(410.0, 710.0, n_pix)
    csv_calib = np.linspace(415.0, 715.0, n_pix)
    csv_data = np.linspace(420.0, 720.0, n_pix)

    # 1. Separate search locations in subtree
    root = tmp_path / "proj_loc"
    data_dir = root / "day1"
    h5_path = _write_basic_pp_h5(data_dir / "scan.h5", orig_wl)

    # Root folder fallback
    _write_probe_csv(root / "CalibratedProbe.csv", csv_root)
    ds = load_1D(h5_path, data_type="PyMESS_PP")
    np.testing.assert_allclose(ds.probe, csv_root)
    assert ds.units["unitsL_ltx"] == "nm"
    assert "root folder" in ds.cal_source

    # Also auto-delegates via generic HDF5 data_type
    ds_h5 = load_1D(h5_path, data_type="HDF5")
    np.testing.assert_allclose(ds_h5.probe, csv_root)

    # Cal+Chirp subfolder beats root folder
    _write_probe_csv(data_dir / "Cal+Chirp" / "CalibratedProbe.csv", csv_cal_chirp, header=True)
    ds = load_1D(h5_path, data_type="PyMESS_PP")
    np.testing.assert_allclose(ds.probe, csv_cal_chirp)
    assert "Cal+Chirp folder" in ds.cal_source

    # calib/ subfolder beats Cal+Chirp/
    _write_probe_csv(data_dir / "calib" / "CalibratedProbe.csv", csv_calib)
    ds = load_1D(h5_path, data_type="PyMESS_PP")
    np.testing.assert_allclose(ds.probe, csv_calib)
    assert "calib folder" in ds.cal_source

    # data folder beats calib/ and everything else
    _write_probe_csv(data_dir / "CalibratedProbe.csv", csv_data)
    ds = load_1D(h5_path, data_type="PyMESS_PP")
    np.testing.assert_allclose(ds.probe, csv_data)
    assert ds.cal_source == "CalibratedProbe.csv (data folder)"

    # Mismatched length CSV in data folder is ignored, falling back to calib/
    _write_probe_csv(data_dir / "CalibratedProbe.csv", np.linspace(1.0, 10.0, 16))
    ds_mismatch = load_1D(h5_path, data_type="PyMESS_PP")
    np.testing.assert_allclose(ds_mismatch.probe, csv_calib)
    assert "calib folder" in ds_mismatch.cal_source


def test_pymess_pp_slowmod_and_anisotropy_prefer_calibrated_probe_csv(slow_mod_anisotropy_file: Path):
    """Slow-modulated PyMESS_PP and load_pymess_anisotropy use CalibratedProbe.csv for all channels."""
    csv_cm1 = np.linspace(1912.0, 2088.0, 16)
    _write_probe_csv(slow_mod_anisotropy_file.parent / "CalibratedProbe.csv", csv_cm1)

    ds_pp = load_1D(slow_mod_anisotropy_file, data_type="PyMESS_PP")
    np.testing.assert_allclose(ds_pp.probe, csv_cm1)
    assert ds_pp.units["unitsL_ltx"] == "cm$^{-1}$"
    assert "CalibratedProbe.csv" in ds_pp.cal_source
    assert ds_pp.n_detectors == 2
    assert ds_pp.Zavg_R.shape[2] == 2

    # Pre-calculated anisotropy (G_factor == 1.0)
    res_1 = load_pymess_anisotropy(slow_mod_anisotropy_file, G_factor=1.0)
    assert set(res_1.keys()) == {"parallel", "perpendicular", "isotropic", "anisotropy"}
    for sub_ds in res_1.values():
        np.testing.assert_allclose(sub_ds.probe, csv_cm1)
        assert sub_ds.units["unitsL_ltx"] == "cm$^{-1}$"
        assert "CalibratedProbe.csv" in sub_ds.cal_source

    # On-the-fly anisotropy (G_factor != 1.0)
    res_g = load_pymess_anisotropy(slow_mod_anisotropy_file, G_factor=1.2)
    assert set(res_g.keys()) == {"parallel", "perpendicular", "isotropic", "anisotropy"}
    for sub_ds in res_g.values():
        np.testing.assert_allclose(sub_ds.probe, csv_cm1)
        assert sub_ds.units["unitsL_ltx"] == "cm$^{-1}$"
        assert "CalibratedProbe.csv" in sub_ds.cal_source


def test_pymess_pp_uncalibrated_512_pixel_axis_with_and_without_csv(tmp_path: Path):
    """512-px 0..511 array reads as uncalibrated px without CSV and calibrated nm with CSV."""
    px_axis = np.arange(512, dtype=np.float64)
    p = _write_basic_pp_h5(tmp_path / "uncal_512.h5", px_axis)

    ds_uncal = load_1D(p, data_type="PyMESS_PP")
    assert ds_uncal.units["unitsL_ltx"] == "px"
    assert ds_uncal.cal_source == "Uncalibrated pixel indices"
    np.testing.assert_allclose(ds_uncal.probe, px_axis)

    nm_axis = np.linspace(323.0, 745.0, 512)
    _write_probe_csv(tmp_path / "CalibratedProbe.csv", nm_axis)
    ds_cal = load_1D(p, data_type="PyMESS_PP")
    assert ds_cal.units["unitsL_ltx"] == "nm"
    assert ds_cal.cal_source == "CalibratedProbe.csv (data folder)"
    np.testing.assert_allclose(ds_cal.probe, nm_axis)


def test_embed_calibrated_probe_h5_and_dataset_roundtrip(tmp_path: Path):
    """Embedding CalibratedProbe.csv sticks the axis inside .h5 and preserves original backup."""
    orig_wl = np.linspace(400.0, 700.0, 32)
    cal1 = np.linspace(410.0, 710.0, 32)
    cal2 = np.linspace(415.0, 715.0, 32)
    cal3 = np.linspace(420.0, 720.0, 32)

    folder_a = tmp_path / "folder_a"
    p = _write_basic_pp_h5(folder_a / "scan.h5", orig_wl)
    _write_probe_csv(folder_a / "CalibratedProbe.csv", cal1)

    # Length-mismatched CSV raises ValueError and leaves .h5 untouched
    bad_csv = tmp_path / "bad_len.csv"
    _write_probe_csv(bad_csv, np.linspace(400.0, 500.0, 12))
    with pytest.raises(ValueError):
        pm.embed_calibrated_probe_h5(p, bad_csv)
    with h5py.File(p, "r") as f:
        np.testing.assert_allclose(f["data/wavelengths"][:], orig_wl)

    # Embed via Dataset1D method while preserving shifted t0 in memory
    ds = load_1D(p, data_type="PyMESS_PP")
    ds.shift_t0(50.0)
    delays_shifted = ds.delays.copy()

    res = ds.embed_calibrated_probe()
    assert res["unitsL"] == "nm"
    assert any("wavelengths" in item for item in res["datasets_updated"])
    np.testing.assert_allclose(ds.probe, cal1)
    np.testing.assert_allclose(ds.delays, delays_shifted)
    assert ds.units["unitsL_ltx"] == "nm"
    assert ds.cal_level is None

    with h5py.File(p, "r") as f:
        np.testing.assert_allclose(f["data/wavelengths"][:], cal1)
        np.testing.assert_allclose(f["calibration/original_probe_axis"][:], orig_wl)

    # Re-embed with a different CSV; original backup must remain orig_wl
    cal2_csv = tmp_path / "cal2.csv"
    _write_probe_csv(cal2_csv, cal2)
    pm.embed_calibrated_probe_h5(p, cal2_csv)
    with h5py.File(p, "r") as f:
        np.testing.assert_allclose(f["data/wavelengths"][:], cal2)
        np.testing.assert_allclose(f["calibration/original_probe_axis"][:], orig_wl)

    # Copy .h5 to isolated folder (no CSV): embedded calibration persists
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    iso_h5 = isolated / "scan_copy.h5"
    iso_h5.write_bytes(p.read_bytes())

    ds_iso = load_1D(iso_h5, data_type="PyMESS_PP")
    np.testing.assert_allclose(ds_iso.probe, cal2)
    assert "Embedded CalibratedProbe.csv" in ds_iso.cal_source

    # Adding a newer CalibratedProbe.csv next to scan_copy.h5 still overrides the embedded one
    _write_probe_csv(isolated / "CalibratedProbe.csv", cal3)
    ds_ext = load_1D(iso_h5, data_type="PyMESS_PP")
    np.testing.assert_allclose(ds_ext.probe, cal3)
    assert ds_ext.cal_source == "CalibratedProbe.csv (data folder)"

    # Also test native PyMORGAN HDF5 embedding
    native_h5 = tmp_path / "native.h5"
    ds_iso.save_hdf5(native_h5)
    pm.embed_calibrated_probe_h5(native_h5, cal2_csv)
    ds_native_reloaded = load_1D(native_h5, data_type="HDF5")
    np.testing.assert_allclose(ds_native_reloaded.probe, cal2)





