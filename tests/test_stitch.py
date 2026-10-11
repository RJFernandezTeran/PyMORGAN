"""Unit tests for wavelength and timescale stitching backend functions."""

import numpy as np
import pytest

import pymorgan as pm
from pymorgan.oneD.stitch import (
    StitchResult,
    _optimize_linear_wl_shift,
    stitch_time,
    stitch_wavelength,
)


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
