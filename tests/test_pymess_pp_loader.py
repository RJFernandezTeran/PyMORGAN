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
    base_dir = Path(r"C:\Users\ricar\switchdrive\Ambizione UniGE\Scripts\testData\PyMESS\20261007")
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
