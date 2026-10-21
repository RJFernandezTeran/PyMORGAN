"""Unit and integration tests for PyMESS shaper t2-scan (1D) and 2D loaders."""

from pathlib import Path

import h5py
import numpy as np
import pytest

import pymorgan as pm
from pymorgan.oneD.load import (
    _pymess_pp_is_file,
    classify_pymess_h5,
    describe_dataset,
    read_PyMESS_PP,
)
from pymorgan.twoD.load import (
    is_map_dataset_file,
    read_PyMESS_2D,
)


@pytest.fixture
def shaper_t2_scan_file(tmp_path: Path) -> Path:
    """Create a mock PyMESS shaper_t2_scan .h5 file matching the real PyMESS schema."""
    p = tmp_path / "shaper_t2_scan.h5"
    n_delays, n_pix = 12, 32
    t2_fs = np.linspace(-500.0, 2000.0, n_delays)
    omega3 = np.linspace(1900.0, 2100.0, n_pix)
    surf = np.outer(np.exp(-np.maximum(t2_fs, 0.0) / 1000.0), np.sin(np.linspace(0, 3, n_pix))) * 5.0

    with h5py.File(p, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["measurement_type"] = "2d_spectra"
        meta.attrs["domain_mode"] = "time_domain"
        meta.attrs["fast_modulation"] = "2f_pp"
        meta.attrs["sample_id"] = "ReCO_t2"
        shaper = meta.create_group("shaper")
        shaper.attrs["mode"] = "Time"
        shaper.attrs["form"] = 0

        grp = f.create_group("data")
        grp.create_dataset("t1_delays_fs", data=t2_fs)
        grp.create_dataset("omega3_axis", data=omega3)
        grp.create_dataset("interferograms_raw", data=surf)
        grp.create_dataset("surface_absorptive", data=surf)

        scans = f.create_group("scans")
        s1 = scans.create_group("scan_001")
        s1.create_dataset("interferogram", data=surf * 0.95)
        s2 = scans.create_group("scan_002")
        s2.create_dataset("interferogram", data=surf * 1.05)

    return p


@pytest.fixture
def shaper_t2_slowmod_file(tmp_path: Path) -> Path:
    """Create a mock slow-modulated shaper_t2_scan .h5 file."""
    p = tmp_path / "shaper_t2_slowmod.h5"
    n_delays, n_pix = 8, 16
    t2_fs = np.linspace(-200.0, 1000.0, n_delays)
    omega3 = np.linspace(1900.0, 2050.0, n_pix)

    par = np.full((n_delays, n_pix), 9.0)
    perp = np.full((n_delays, n_pix), 3.0)

    with h5py.File(p, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["measurement_type"] = "2d_spectra"
        meta.attrs["domain_mode"] = "time_domain"
        meta.attrs["slow_modulation_enabled"] = "True"
        meta.attrs["slow_modulation_num_states"] = "2"
        meta.attrs["slow_modulation_state_names"] = "Parallel,Perpendicular"
        shaper = meta.create_group("shaper")
        shaper.attrs["mode"] = "Time"
        shaper.attrs["form"] = 0

        grp = f.create_group("data")
        grp.create_dataset("t1_delays_fs", data=t2_fs)
        grp.create_dataset("omega3_axis", data=omega3)
        grp.create_dataset("interferograms_raw", data=0.5 * (par + perp))

        scans = f.create_group("scans")
        st0 = scans.create_group("state_0")
        st0.attrs["state_name"] = "Parallel"
        st0.create_dataset("interferograms_raw", data=par)
        st0.create_group("scan_001").create_dataset("interferogram", data=par)
        st0.create_group("scan_002").create_dataset("interferogram", data=par)

        st1 = scans.create_group("state_1")
        st1.attrs["state_name"] = "Perpendicular"
        st1.create_dataset("interferograms_raw", data=perp)
        st1.create_group("scan_001").create_dataset("interferogram", data=perp)
        st1.create_group("scan_002").create_dataset("interferogram", data=perp)

    return p


@pytest.fixture
def td_2d_file(tmp_path: Path) -> Path:
    """Create a mock Time-Domain 2D .h5 file with 2 waiting times and slow modulation."""
    p = tmp_path / "td_2d_scan.h5"
    n_t1, n_pix = 21, 16
    dt1_fs = 35.0
    t1_fs = np.arange(n_t1, dtype=float) * dt1_fs
    omega3 = np.linspace(1950.0, 2050.0, n_pix)
    omega1 = np.linspace(1800.0, 2150.0, 32)
    w0_cm1 = 1800.0
    freq_cm1 = 2025.0
    c_cm_fs = 2.99792458e-5

    # Construct damped cosine along t1 at (freq_cm1 - w0_cm1)
    osc = np.cos(2.0 * np.pi * (freq_cm1 - w0_cm1) * c_cm_fs * t1_fs) * np.exp(-t1_fs / 600.0)
    probe_prof = np.exp(-((omega3 - 2020.0) / 15.0) ** 2)
    td_signal = np.outer(osc, probe_prof) * -10.0
    fd_dummy = np.zeros((len(omega1), n_pix), dtype=float)

    with h5py.File(p, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["measurement_type"] = "2d_spectra"
        meta.attrs["domain_mode"] = "time_domain"
        meta.attrs["fast_modulation"] = "4f_2d"
        meta.attrs["sample_id"] = "ReCO_TD2D"
        meta.attrs["slow_modulation_enabled"] = "True"
        meta.attrs["slow_modulation_num_states"] = "2"
        meta.attrs["slow_modulation_state_names"] = "Parallel,Perpendicular"
        shaper = meta.create_group("shaper")
        shaper.attrs["mode"] = "Time"
        shaper.attrs["form"] = 1
        shaper.attrs["effective_rotating_frame_cm1"] = str(w0_cm1)
        shaper.attrs["rotating_frame_scanned_cm1"] = "1550.0"
        shaper.attrs["rotating_frame_fixed_cm1"] = "-250.0"
        shaper.attrs["step_fs"] = str(dt1_fs)

        grp = f.create_group("data")
        grp.create_dataset("t1_delays_fs", data=t1_fs)
        grp.create_dataset("omega1_axis", data=omega1)
        grp.create_dataset("omega3_axis", data=omega3)
        grp.create_dataset("t2_delays_fs", data=np.array([200.0, 1000.0]))
        grp.create_dataset("interferograms_raw", data=td_signal)
        grp.create_dataset("surface_absorptive", data=fd_dummy)

        scans = f.create_group("scans")
        for idx, (sk, t2_val, scale) in enumerate([("scan_001", 200.0, 1.0), ("scan_002", 1000.0, 0.5)]):
            sgrp = scans.create_group(sk)
            sgrp.attrs["t2_delay_fs"] = t2_val
            sgrp.create_dataset("interferograms_raw", data=td_signal * scale)

        for st_idx, (st_name, st_scale) in enumerate([("Parallel", 1.2), ("Perpendicular", 0.4)]):
            st_grp = scans.create_group(f"state_{st_idx}")
            st_grp.attrs["state_name"] = st_name
            for sk, t2_val, scale in [("scan_001", 200.0, 1.0), ("scan_002", 1000.0, 0.5)]:
                sgrp = st_grp.create_group(sk)
                sgrp.attrs["t2_delay_fs"] = t2_val
                sgrp.create_dataset("interferograms_raw", data=td_signal * scale * st_scale)

    return p


@pytest.fixture
def fd_2d_file(tmp_path: Path) -> Path:
    """Create a mock Frequency-Domain 2D .h5 file."""
    p = tmp_path / "fd_2d_scan_500fs.h5"
    n_pump, n_pix = 15, 20
    pump_axis = np.linspace(1950.0, 2060.0, n_pump)
    probe_axis = np.linspace(1940.0, 2070.0, n_pix)
    surf = np.outer(
        np.exp(-((pump_axis - 2015.0) / 12.0) ** 2),
        np.exp(-((probe_axis - 2015.0) / 12.0) ** 2),
    ) * -8.0

    with h5py.File(p, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["measurement_type"] = "2d_spectra"
        meta.attrs["domain_mode"] = "frequency_domain"
        meta.attrs["sample_id"] = "ReCO_FD2D"
        shaper = meta.create_group("shaper")
        shaper.attrs["mode"] = "Freq (cm-1)"
        shaper.attrs["form"] = 0

        grp = f.create_group("data")
        grp.create_dataset("t1_delays_fs", data=pump_axis)
        grp.create_dataset("omega1_axis", data=pump_axis)
        grp.create_dataset("omega3_axis", data=probe_axis)
        grp.create_dataset("surface_absorptive", data=surf)
        grp.create_dataset("interferograms_raw", data=surf)

    return p


def test_classify_pymess_h5(
    shaper_t2_scan_file: Path,
    td_2d_file: Path,
    fd_2d_file: Path,
    tmp_path: Path,
):
    """Verify classify_pymess_h5 distinguishes all 4 PyMESS modes and rejects invalid files."""
    assert classify_pymess_h5(shaper_t2_scan_file) == "shaper_t2_scan"
    assert classify_pymess_h5(td_2d_file) == "2d_time_domain"
    assert classify_pymess_h5(fd_2d_file) == "2d_freq_domain"

    # Aborted / empty 2D file (omega1_axis empty, surface_absorptive all-NaN)
    aborted = tmp_path / "aborted_2d.h5"
    with h5py.File(aborted, "w") as f:
        m = f.create_group("metadata")
        m.attrs["measurement_type"] = "2d_spectra"
        g = f.create_group("data")
        g.create_dataset("t1_delays_fs", data=np.linspace(0, 500, 10))
        g.create_dataset("omega1_axis", data=np.array([], dtype=float))
        g.create_dataset("omega3_axis", data=float("nan"))
        g.create_dataset("surface_absorptive", data=np.full((10, 16), np.nan))
    assert classify_pymess_h5(aborted) == "other"

    # File filters for 1D vs 2D
    assert _pymess_pp_is_file(shaper_t2_scan_file) is True
    assert _pymess_pp_is_file(td_2d_file) is False
    assert _pymess_pp_is_file(fd_2d_file) is False
    assert _pymess_pp_is_file(aborted) is False

    assert is_map_dataset_file("PyMESS_2D", td_2d_file) is True
    assert is_map_dataset_file("PyMESS_2D", fd_2d_file) is True
    assert is_map_dataset_file("PyMESS_2D", shaper_t2_scan_file) is False
    assert is_map_dataset_file("PyMESS_2D", aborted) is False


def test_read_shaper_t2_scan_1d(shaper_t2_scan_file: Path, shaper_t2_slowmod_file: Path, tmp_path: Path):
    """Verify shaper_t2_scan loads into Dataset1D with ps delays and fallback omega3 axis."""
    ds = pm.load_1D(shaper_t2_scan_file, data_type="PyMESS_PP")
    assert ds.Zavg_R.shape == (12, 32, 1)
    assert ds.Zss_R.shape == (12, 32, 1, 2)
    assert ds.nscans == 2
    assert ds.delays[0] == pytest.approx(-0.5)
    assert ds.delays[-1] == pytest.approx(2.0)
    assert ds.Units["T"] == "ps"
    assert ds.Units["L"] == "cm-1"
    assert ds.Units["sample"] == "ReCO_t2"

    # Slow-modulated shaper_t2_scan
    ds_sm = pm.load_1D(shaper_t2_slowmod_file, data_type="PyMESS_PP")
    assert ds_sm.Zavg_R.shape == (8, 16, 2)
    assert ds_sm.Zss_R.shape == (8, 16, 2, 2)
    assert ds_sm.Units["detectors"] == ["Parallel", "Perpendicular"]
    np.testing.assert_allclose(ds_sm.Zavg_R[:, :, 0], 9.0)
    np.testing.assert_allclose(ds_sm.Zavg_R[:, :, 1], 3.0)

    desc = describe_dataset("PyMESS_PP", shaper_t2_slowmod_file)
    assert desc is not None
    assert desc["n_slowmod"] == 2
    assert desc["state_names"] == ["Parallel", "Perpendicular"]

    # Stopped scan with scalar-NaN omega3_axis in same folder recovers omega3 from sibling
    stopped = tmp_path / "ReCO_t2scan_stopped.h5"
    with h5py.File(stopped, "w") as f:
        m = f.create_group("metadata")
        m.attrs["measurement_type"] = "2d_spectra"
        g = f.create_group("data")
        g.create_dataset("t1_delays_fs", data=np.linspace(-100.0, 500.0, 6))
        g.create_dataset("omega1_axis", data=np.array([-0.05, 0.0, 0.05]))
        g.create_dataset("omega3_axis", data=float("nan"))
        g.create_dataset("interferograms_raw", data=np.ones((6, 32)))
    assert classify_pymess_h5(stopped) == "shaper_t2_scan"
    ds_stopped = pm.load_1D(stopped, data_type="PyMESS_PP")
    np.testing.assert_allclose(ds_stopped.probe, ds.probe)


def test_load_and_process_td_2d(td_2d_file: Path):
    """Verify loading and FFT processing of Time-Domain PyMESS 2D datasets."""
    ds = pm.load_2D(td_2d_file, data_type="PyMESS_2D")
    assert ds.is_frequency_domain is False
    assert ds.datatype == "shaper"
    assert ds.w0_cm1 == pytest.approx(1800.0)
    assert ds.dt1_fs == pytest.approx(35.0)
    np.testing.assert_allclose(ds.delays, [0.2, 1.0])
    assert len(ds.raw_t1delays) == 21
    assert len(ds.probe) == 16
    assert ds.state_names == ["Parallel", "Perpendicular", "Magic Angle"]

    # Process without needing w0.csv in folder (uses dataset.w0_cm1 and dataset.dt1_fs)
    ds.process(zeropad_factor=4)
    assert len(ds.pump) > 21
    assert ds.Z.shape[2] == 2
    # Check peak pump frequency is near the simulated 2025 cm-1 oscillation
    w1_mask = (ds.pump >= 1850.0) & (ds.pump <= 2150.0)
    sub_map = ds.Z[w1_mask, :, 0]
    w1_sub = ds.pump[w1_mask]
    peak_w1 = w1_sub[np.unravel_index(np.argmax(np.abs(sub_map)), sub_map.shape)[0]]
    assert abs(peak_w1 - 2025.0) < 10.0

    # Switch state to Perpendicular and re-process
    ds.set_active_state("Perpendicular")
    ds.process(zeropad_factor=4)
    assert ds.active_state == "Perpendicular"


def test_load_and_process_fd_2d(fd_2d_file: Path):
    """Verify loading and direct processing of Frequency-Domain PyMESS 2D datasets."""
    ds = pm.load_2D(fd_2d_file, data_type="PyMESS_2D")
    assert ds.is_frequency_domain is True
    assert ds.datatype == "FreqDomain"
    np.testing.assert_allclose(ds.delays, [0.5])
    assert len(ds.pump) == 15
    assert len(ds.probe) == 20

    # Process should bypass FFT and preserve shape (n_pump, n_probe, n_t2)
    ds.process(bkg_sub=False)
    assert ds.Z.shape == (15, 20, 1)
    assert np.isfinite(ds.Z).all()


def test_real_20261016_datasets():
    """Integration test against all experimental files in 20261016 if present."""
    base_dir = Path(r"C:\Users\ricar\switchdrive\Ambizione UniGE\Scripts\testData\PyMESS\20261016")
    if not base_dir.is_dir():
        pytest.skip(f"Directory {base_dir} not found")

    # 1. t2scan subfolder -> all 11 files should load as 1D PyMESS_PP
    t2_dir = base_dir / "t2scan"
    if t2_dir.is_dir():
        h5_files = sorted(t2_dir.glob("*.h5"))
        assert len(h5_files) >= 5
        for f in h5_files:
            assert classify_pymess_h5(f) == "shaper_t2_scan"
            assert _pymess_pp_is_file(f) is True
            ds1d = pm.load_1D(f, data_type="PyMESS_PP")
            assert ds1d.Zavg_R.shape[1] == 64
            assert len(ds1d.probe) == 64
            assert np.isfinite(ds1d.probe).all()
            assert ds1d.Units["T"] == "ps"
            assert ds1d.Units["L"] == "cm-1"

    # 2. 2D data subfolder -> 2 FD files and 4 TD files
    two_d_dir = base_dir / "2D data"
    if two_d_dir.is_dir():
        for f in sorted(two_d_dir.glob("*.h5")):
            kind = classify_pymess_h5(f)
            assert kind in ("2d_freq_domain", "2d_time_domain")
            assert is_map_dataset_file("PyMESS_2D", f) is True
            ds2d = pm.load_2D(f, data_type="PyMESS_2D")
            ds2d.process()
            assert ds2d.Z.shape == (len(ds2d.pump), len(ds2d.probe), len(ds2d.delays))

        # Check specific rotating-frame peak agreement on ReCO_timeDomain_2D_195443.h5
        f_td = two_d_dir / "ReCO_timeDomain_2D_195443.h5"
        if f_td.is_file():
            ds_td = pm.load_2D(f_td, data_type="PyMESS_2D")
            assert ds_td.w0_cm1 == pytest.approx(1800.0)
            assert ds_td.dt1_fs == pytest.approx(35.0)
            ds_td.process(apodise_method="0", zeropad_factor=3)
            w1_mask = (ds_td.pump >= 1880.0) & (ds_td.pump <= 2100.0)
            sub = ds_td.Z[w1_mask, :, 0]
            w1_sub = ds_td.pump[w1_mask]
            peak_w1 = w1_sub[np.unravel_index(np.argmin(sub), sub.shape)[0]]
            assert peak_w1 == pytest.approx(2025.23022053, abs=1e-4)


def test_new_schema_shaper_t2_scan_3d_active_and_slowmod(tmp_path: Path):
    """Verify new PyMESS .h5 schema for shaper t2 scans (3D interferograms_raw, active AWG mode, slow_modulation)."""
    p = tmp_path / "new_schema_t2_scan.h5"
    n_delays, n_pix = 10, 24
    t2_fs = np.linspace(-300.0, 1500.0, n_delays)
    omega3 = np.linspace(1920.0, 2080.0, n_pix)
    par_2d = np.full((n_delays, n_pix), 8.0)
    perp_2d = np.full((n_delays, n_pix), 2.0)
    avg_3d = (0.5 * (par_2d + perp_2d))[np.newaxis, :, :]  # (1, n_t1, n_pix)
    indiv_4d = np.stack([avg_3d * 0.9, avg_3d * 1.1], axis=0)  # (2, 1, n_t1, n_pix)

    with h5py.File(p, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["measurement_type"] = "2d_spectra"
        meta.attrs["domain_mode"] = "time_domain"
        meta.attrs["sample_id"] = "NewSchema_t2"
        meta.attrs["slow_modulation_enabled"] = True
        meta.attrs["slow_modulation_num_states"] = 2
        meta.attrs["slow_modulation_state_names"] = "Parallel,Perpendicular"
        shaper = meta.create_group("shaper")
        shaper.attrs["mode"] = "active"
        shaper.attrs["shaper_mode"] = "time"
        shaper.attrs["pulse_form"] = "Single Pulse (Compression / Chirp)"

        grp = f.create_group("data")
        grp.create_dataset("t1_delays_fs", data=t2_fs)
        grp.create_dataset("t2_delays_fs", data=np.array([0.0]))
        grp.create_dataset("omega3_axis", data=omega3)
        grp.create_dataset("interferograms_raw", data=avg_3d)
        grp.create_dataset("pump_interferogram", data=np.ones((1, n_delays)))
        grp.create_dataset("individual_scans", data=indiv_4d)

        sm = f.create_group("data/slow_modulation")
        sm.create_dataset("state_00_Parallel", data=par_2d[np.newaxis, :, :])
        sm.create_dataset("state_01_Perpendicular", data=perp_2d[np.newaxis, :, :])
        sm.create_dataset("isotropic", data=((par_2d + 2.0 * perp_2d) / 3.0)[np.newaxis, :, :])
        sm.create_dataset("anisotropy", data=((par_2d - perp_2d) / (par_2d + 2.0 * perp_2d))[np.newaxis, :, :])

    assert classify_pymess_h5(p) == "shaper_t2_scan"
    assert _pymess_pp_is_file(p) is True
    assert is_map_dataset_file("PyMESS_2D", p) is False

    desc = describe_dataset("PyMESS_PP", p)
    assert desc is not None
    assert desc["n_slowmod"] == 2
    assert desc["state_names"] == ["Parallel", "Perpendicular"]

    ds = pm.load_1D(p, data_type="PyMESS_PP")
    assert ds.Zavg_R.shape == (n_delays, n_pix, 2)
    assert ds.Units["detectors"] == ["Parallel", "Perpendicular"]
    np.testing.assert_allclose(ds.Zavg_R[:, :, 0], 8.0)
    np.testing.assert_allclose(ds.Zavg_R[:, :, 1], 2.0)
    assert ds.Units["counts"].shape == (n_delays,)



def test_new_schema_td_2d_3d_no_omega1_axis(tmp_path: Path):
    """Verify new PyMESS .h5 schema for Time-Domain 2D (3D interferograms_raw, no omega1_axis/surface_absorptive)."""
    p = tmp_path / "new_schema_td_2d.h5"
    n_t2, n_t1, n_pix = 3, 21, 16
    dt1_fs = 35.0
    t1_fs = np.arange(n_t1, dtype=float) * dt1_fs
    t2_fs = np.array([200.0, 1000.0, 5000.0])  # 3rd t2 slice will be unmeasured (NaN)
    omega3 = np.linspace(1950.0, 2050.0, n_pix)
    w0_cm1 = 1800.0
    freq_cm1 = 2025.0
    c_cm_fs = 2.99792458e-5

    osc = np.cos(2.0 * np.pi * (freq_cm1 - w0_cm1) * c_cm_fs * t1_fs) * np.exp(-t1_fs / 600.0)
    probe_prof = np.exp(-((omega3 - 2020.0) / 15.0) ** 2)
    td_base = np.outer(osc, probe_prof) * -10.0  # (n_t1, n_pix)

    cube_avg = np.full((n_t2, n_t1, n_pix), np.nan, dtype=float)
    cube_avg[0] = td_base * 1.0
    cube_avg[1] = td_base * 0.5
    # cube_avg[2] left as NaN to simulate a live/partially completed multi-t2 scan

    with h5py.File(p, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["measurement_type"] = "2d_spectra"
        meta.attrs["domain_mode"] = "time_domain"
        meta.attrs["sample_id"] = "NewSchema_TD2D"
        meta.attrs["slow_modulation_enabled"] = True
        meta.attrs["slow_modulation_num_states"] = 2
        meta.attrs["slow_modulation_state_names"] = "Parallel,Perpendicular"
        shaper = meta.create_group("shaper")
        shaper.attrs["mode"] = "active"
        shaper.attrs["shaper_mode"] = "time"
        shaper.attrs["pulse_form"] = "Double Pulse (2D Scan)"
        shaper.attrs["rotating_frame_cm1"] = w0_cm1
        shaper.attrs["step_fs"] = dt1_fs

        grp = f.create_group("data")
        grp.create_dataset("t1_delays_fs", data=t1_fs)
        grp.create_dataset("t2_delays_fs", data=t2_fs)
        grp.create_dataset("omega3_axis", data=omega3)
        grp.create_dataset("interferograms_raw", data=cube_avg)
        grp.create_dataset("pump_interferogram", data=np.ones((n_t2, n_t1)))

        sm = f.create_group("data/slow_modulation")
        sm.create_dataset("state_00_Parallel", data=cube_avg * 1.2)
        sm.create_dataset("state_01_Perpendicular", data=cube_avg * 0.4)
        sm.create_dataset("isotropic", data=cube_avg * ((1.2 + 0.8) / 3.0))

    assert classify_pymess_h5(p) == "2d_time_domain"
    assert is_map_dataset_file("PyMESS_2D", p) is True
    assert _pymess_pp_is_file(p) is False

    ds = pm.load_2D(p, data_type="PyMESS_2D")
    assert ds.is_frequency_domain is False
    assert ds.w0_cm1 == pytest.approx(1800.0)
    assert ds.dt1_fs == pytest.approx(35.0)
    # Only the 2 measured t2 slices should be kept (3rd all-NaN slice filtered out)
    np.testing.assert_allclose(ds.delays, [0.2, 1.0])
    assert ds.raw_signal.shape == (n_t1, n_pix, 2)
    assert ds.raw_interferogram.shape == (n_t1, 2)
    assert ds.state_names == ["Parallel", "Perpendicular", "Magic Angle"]


    ds.process(zeropad_factor=4)
    assert ds.Z.shape[2] == 2
    assert np.isfinite(ds.Z).all()


def test_new_schema_fd_2d_3d_no_surface_absorptive(tmp_path: Path):
    """Verify new PyMESS .h5 schema for Frequency-Domain 2D (3D interferograms_raw, no surface_absorptive)."""
    p = tmp_path / "new_schema_fd_2d.h5"
    n_t2, n_pump, n_pix = 2, 15, 20
    pump_axis = np.linspace(1950.0, 2060.0, n_pump)
    probe_axis = np.linspace(1940.0, 2070.0, n_pix)
    t2_fs = np.array([300.0, 1500.0])
    surf_base = (
        np.outer(
            np.exp(-((pump_axis - 2015.0) / 12.0) ** 2),
            np.exp(-((probe_axis - 2015.0) / 12.0) ** 2),
        )
        * -8.0
    )
    cube_avg = np.stack([surf_base, surf_base * 0.6], axis=0)  # (n_t2, n_pump, n_pix)

    with h5py.File(p, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["measurement_type"] = "2d_spectra"
        meta.attrs["domain_mode"] = "frequency_domain"
        meta.attrs["sample_id"] = "NewSchema_FD2D"
        meta.attrs["slow_modulation_enabled"] = True
        meta.attrs["slow_modulation_num_states"] = 2
        meta.attrs["slow_modulation_state_names"] = "Parallel,Perpendicular"
        shaper = meta.create_group("shaper")
        shaper.attrs["mode"] = "active"
        shaper.attrs["shaper_mode"] = "freq"

        grp = f.create_group("data")
        grp.create_dataset("t1_delays_fs", data=pump_axis)
        grp.create_dataset("omega1_axis", data=pump_axis)
        grp.create_dataset("omega3_axis", data=probe_axis)
        grp.create_dataset("t2_delays_fs", data=t2_fs)
        grp.create_dataset("interferograms_raw", data=cube_avg)

        sm = f.create_group("data/slow_modulation")
        sm.create_dataset("state_00_Parallel", data=cube_avg * 1.2)
        sm.create_dataset("state_01_Perpendicular", data=cube_avg * 0.4)
        sm.create_dataset("isotropic", data=cube_avg * ((1.2 + 0.8) / 3.0))

    assert classify_pymess_h5(p) == "2d_freq_domain"
    assert is_map_dataset_file("PyMESS_2D", p) is True
    assert _pymess_pp_is_file(p) is False

    ds = pm.load_2D(p, data_type="PyMESS_2D")
    assert ds.is_frequency_domain is True
    np.testing.assert_allclose(ds.delays, [0.3, 1.5])
    assert ds.Z_R.shape == (n_pump, n_pix, 2)
    assert ds.state_names == ["Parallel", "Perpendicular", "Magic Angle", "Anisotropy"]



    ds.process(bkg_sub=False)
    assert ds.Z.shape == (n_pump, n_pix, 2)
    assert np.isfinite(ds.Z).all()
    ds.set_active_state("Perpendicular")
    ds.process(bkg_sub=False)
    np.testing.assert_allclose(ds.Z[:, :, 0], surf_base * 0.4)

