"""Unit tests for Singular Value Decomposition (SVD) backend and dataset methods."""

import numpy as np
import pytest

import pymorgan as pm
from pymorgan.oneD.svd import SVDResult, compute_svd


@pytest.fixture
def synthetic_data():
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


def test_svd_basic(synthetic_data):
    """Test basic SVD decomposition and variance properties."""
    ds, pure, z, t, p = synthetic_data
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


def test_svd_recompose_and_residuals(synthetic_data):
    """Test rank-k recomposition and residual matrices."""
    ds, pure, z, t, p = synthetic_data
    res = ds.svd(k=2)

    recon = res.recompose(k=2)
    resid = res.residuals(k=2)

    assert recon.shape == z.shape
    assert resid.shape == z.shape
    np.testing.assert_allclose(recon + resid, z, rtol=1e-10, atol=1e-12)

    # Recomposition with all components should reproduce input matrix exactly
    full_recon = res.recompose(k=res.n_components)
    np.testing.assert_allclose(full_recon, z, rtol=1e-10, atol=1e-12)


def test_svd_components_access(synthetic_data):
    """Test temporal and spectral component extractors."""
    ds, pure, z, t, p = synthetic_data
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


def test_svd_cropping(synthetic_data):
    """Test cropping time and probe limits prior to SVD."""
    ds, pure, z, t, p = synthetic_data
    tmin, tmax = 0.5, 30.0
    pmin, pmax = 480.0, 620.0

    res = ds.svd(k=2, tmin=tmin, tmax=tmax, probe_min=pmin, probe_max=pmax)

    assert np.all(res.delays >= tmin)
    assert np.all(res.delays <= tmax)
    assert np.all(res.probe >= pmin)
    assert np.all(res.probe <= pmax)
    assert res.raw_matrix.shape == (len(res.delays), len(res.probe))
    assert res.recompose().shape == (len(res.delays), len(res.probe))


def test_svd_centering_modes(synthetic_data):
    """Test all centering modes."""
    _, _, z, t, p = synthetic_data

    for mode in ["none", "spectrum", "kinetic", "both"]:
        res = compute_svd(z, t, p, k=2, centering=mode)
        recon = res.recompose(k=res.n_components)
        np.testing.assert_allclose(recon, z, rtol=1e-7, atol=1e-7)

    with pytest.raises(ValueError, match="Unknown centering"):
        compute_svd(z, t, p, centering="invalid_mode")


def test_svd_nan_handling(synthetic_data):
    """Test handling of NaN values via interpolation and drop."""
    _, _, z, t, p = synthetic_data
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


def test_svd_to_dataset(synthetic_data):
    """Test exporting SVD recomposition to Dataset1D."""
    ds, _, _, _, _ = synthetic_data
    res = ds.svd(k=2)

    new_ds = res.to_dataset(k=2, name="TestRecomposed")
    assert isinstance(new_ds, pm.Dataset1D)
    assert new_ds.source == "TestRecomposed"
    assert new_ds.Zavg_R.shape == ds.Zavg_R.shape
    assert new_ds.Zavg_C.shape == ds.Zavg_R.shape
    np.testing.assert_allclose(new_ds.delays, ds.delays)
    np.testing.assert_allclose(new_ds.probe, ds.probe)


def test_svd_export_csv_and_dataframe(synthetic_data, tmp_path):
    """Test CSV export and DataFrame conversion."""
    ds, _, _, _, _ = synthetic_data
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


def test_svd_multi_detector(synthetic_data):
    """Test SVD on multi-detector dataset."""
    _, _, z, t, p = synthetic_data
    z_multi = np.stack([z, z * 0.5], axis=2)  # [delays, probe, 2]

    ds_multi = pm.Dataset1D(Zavg_R=z_multi, delays=t, probe=p, units={})
    assert ds_multi.n_detectors == 2

    res0 = ds_multi.svd(detector=0, k=2)
    res1 = ds_multi.svd(detector=1, k=2)

    np.testing.assert_allclose(res0.S, res1.S * 2.0, rtol=1e-5)
