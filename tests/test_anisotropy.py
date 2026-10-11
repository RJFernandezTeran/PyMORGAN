"""Unit tests for Polarization Anisotropy r(t) calculations."""

import numpy as np
import pytest

import pymorgan as pm
from pymorgan.oneD.dataset import Dataset1D


@pytest.fixture
def sample_1d_multidet():
    """Create a 2-detector 1D dataset for anisotropy testing."""
    delays = np.array([-2.0, 0.0, 1.0, 5.0])
    probe_d1 = np.linspace(1900, 2000, 15)
    probe_d2 = np.linspace(2000, 2100, 15)
    probe = np.column_stack([probe_d1, probe_d2])
    Z = np.ones((len(delays), 15, 2))
    Z[:, :, 0] *= 1.5
    Z[:, :, 1] *= 2.5
    noise = np.full_like(Z, 0.05)
    units = {
        "unitsT_lbl": "Delay",
        "unitsT_ltx": "ps",
        "unitsL_lbl": "Wavenumber",
        "unitsL_ltx": r"$\mathrm{cm^{-1}}$",
        "unitsZ_lbl": r"$\Delta$A",
        "unitsZ_ltx": "mOD",
    }
    return Dataset1D(
        Zavg_R=Z,
        delays=delays,
        probe=probe,
        units=units,
        nscans=5,
        Zstdv=noise,
        source="mock_multidet",
        data_type="RAL_IRpp",
    )


@pytest.fixture
def sample_1d():
    """Create a standard 1D dataset."""
    delays = np.array([-5.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
    probe = np.linspace(1900, 2100, 20)
    Z = np.sin(delays[:, None, None] / 2.0) * np.cos(probe[None, :, None] / 100.0)
    units = {
        "unitsT_lbl": "Delay",
        "unitsT_ltx": "ps",
        "unitsL_lbl": "Wavenumber",
        "unitsL_ltx": r"$\mathrm{cm^{-1}}$",
        "unitsZ_lbl": r"$\Delta$A",
        "unitsZ_ltx": "mOD",
    }
    return Dataset1D(
        Zavg_R=Z,
        delays=delays,
        probe=probe,
        units=units,
        nscans=10,
        source="mock_data",
        data_type="PDAT",
    )


def test_calculate_anisotropy_multidet(sample_1d_multidet):
    # sample_1d_multidet has det 0 = 1.5 and det 1 = 2.5
    # r = (1.5 - 2.5) / (1.5 + 2 * 2.5) = -1.0 / 6.5 = -0.153846...
    r_ds = sample_1d_multidet.calculate_anisotropy()
    assert isinstance(r_ds, Dataset1D)
    assert r_ds.data_type == "Anisotropy"
    assert r_ds.units["unitsZ_lbl"] == "r"
    assert np.allclose(r_ds.Z[:, :, 0], -1.0 / 6.5)

    # Test error propagation
    assert r_ds.noise_array() is not None


def test_calculate_anisotropy_two_datasets(sample_1d):
    par = sample_1d.copy()
    perp = sample_1d.copy()
    par.Zavg_R = np.full_like(par.Zavg_R, 10.0)
    par.Zavg_C = None
    perp.Zavg_R = np.full_like(perp.Zavg_R, 5.0)
    perp.Zavg_C = None

    # r = (10 - 1*5) / (10 + 2*5) = 5 / 20 = 0.25
    r_ds = pm.calculate_anisotropy(par, perp, g_factor=1.0)
    assert np.allclose(r_ds.Z[:, :, 0], 0.25)

    # With G = 1.5: r = (10 - 1.5*5) / (10 + 2*1.5*5) = 2.5 / 25 = 0.1
    r_g = pm.calculate_anisotropy(par, perp, g_factor=1.5)
    assert np.allclose(r_g.Z[:, :, 0], 0.1)

    # Test threshold masking: denominator is 20. With threshold=25, all should be NaN
    r_th = pm.calculate_anisotropy(par, perp, threshold=25.0)
    assert np.all(np.isnan(r_th.Z[:, :, 0]))

    # Test return_components
    comps = pm.calculate_anisotropy(par, perp, return_components=True)
    assert "anisotropy" in comps
    assert "magic_angle" in comps
    assert "difference" in comps
    assert "parallel" in comps
    assert "perpendicular" in comps

    # Magic angle = (10 + 2*5)/3 = 20/3
    assert np.allclose(comps["magic_angle"].Z[:, :, 0], 20.0 / 3.0)
    # Difference = 10 - 5 = 5
    assert np.allclose(comps["difference"].Z[:, :, 0], 5.0)
