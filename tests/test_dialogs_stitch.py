"""Unit tests for WavelengthStitchDialog and TimescaleStitchDialog in headless mode."""

import os

import numpy as np
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PyQt6.QtWidgets import QApplication

import pymorgan as pm
from pymorgan.gui.timescale_stitch_dialog import TimescaleStitchDialog
from pymorgan.gui.wavelength_stitch_dialog import WavelengthStitchDialog


@pytest.fixture(scope="session")
def qapp():
    """Ensure a single QApplication instance for Qt tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def synthetic_uv_nir():
    """Create UV and NIR synthetic datasets with spectral overlap."""
    rng = np.random.default_rng(42)
    t = np.linspace(-1, 30, 25)

    p_uv = np.linspace(350, 750, 45)
    p_nir = np.linspace(700, 1100, 50)

    kin = np.where(t > 0, np.exp(-t / 10.0), 0.0)
    spec_uv = np.exp(-((p_uv - 520) / 50.0) ** 2) + 0.2 * np.exp(-((p_uv - 720) / 30.0) ** 2)
    spec_nir = 0.2 * np.exp(-((p_nir - 720) / 30.0) ** 2) + 0.7 * np.exp(-((p_nir - 950) / 80.0) ** 2)

    z_uv = np.outer(kin, spec_uv) + 0.001 * rng.standard_normal((len(t), len(p_uv)))
    z_nir = np.outer(kin, spec_nir * 0.5) + 0.001 * rng.standard_normal((len(t), len(p_nir)))

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
    return ds_uv, ds_nir


@pytest.fixture
def synthetic_fs_ns():
    """Create fs (ps) and ns (ns) synthetic datasets with delay overlap."""
    rng = np.random.default_rng(84)
    t_fs = np.linspace(-2, 1400, 35)  # ps -> -0.002 to 1.4 ns
    t_ns = np.concatenate([np.linspace(-5, 0, 5), np.geomspace(0.1, 400, 35)])  # ns
    p = np.linspace(420, 680, 40)

    kin_fs = np.where(t_fs > 0, 0.7 * np.exp(-t_fs / 600.0) + 0.3 * np.exp(-t_fs / 50000.0), 0.0)
    kin_ns = np.where(t_ns > 0, 0.7 * np.exp(-t_ns / 0.6) + 0.3 * np.exp(-t_ns / 50.0), 0.0)
    spec = np.exp(-((p - 540) / 45.0) ** 2)

    z_fs = np.outer(kin_fs, spec) + 0.001 * rng.standard_normal((len(t_fs), len(p)))
    z_ns = np.outer(kin_ns * 1.5, spec) + 0.001 * rng.standard_normal((len(t_ns), len(p)))

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
    return ds_fs, ds_ns


# =========================================================================
# Wavelength Stitch Dialog Tests
# =========================================================================


def test_wavelength_stitch_dialog_init(qapp, synthetic_uv_nir):
    """Test WavelengthStitchDialog initialization and parameter modification."""
    ds_uv, ds_nir = synthetic_uv_nir
    dlg = WavelengthStitchDialog(None, ds_uv, ds_nir)

    assert dlg.result is not None
    assert dlg.result.kind == "wavelength"
    assert dlg.rad_blend.isChecked()

    # Switch to Cutover mode
    dlg.rad_cutover.setChecked(True)
    assert dlg.rad_cutover.isChecked()
    assert dlg.result.cutover_val is not None

    # Manual scale factor override
    dlg.chk_override.setChecked(True)
    dlg.spin_override.setValue(2.05)
    assert dlg.spin_override.isEnabled()
    assert dlg.result.scaling_factor == 2.05

    # Switch back to automatic fit
    dlg.chk_override.setChecked(False)
    assert not dlg.spin_override.isEnabled()

    # Adjust fit bounds
    dlg.spin_fit_min.setValue(715.0)
    dlg.spin_fit_max.setValue(740.0)
    assert dlg.result.stitch_region == (715.0, 740.0)


def test_wavelength_stitch_dialog_apply_signal(qapp, synthetic_uv_nir):
    """Test that WavelengthStitchDialog emits stitch_applied with a Dataset1D."""
    ds_uv, ds_nir = synthetic_uv_nir
    dlg = WavelengthStitchDialog(None, ds_uv, ds_nir)

    emitted_datasets = []
    dlg.stitch_applied.connect(emitted_datasets.append)

    # Click apply without modal prompt
    dlg._apply_to_workspace(prompt_confirmation=False)

    assert len(emitted_datasets) == 1
    applied_ds = emitted_datasets[0]
    assert isinstance(applied_ds, pm.Dataset1D)
    assert applied_ds.probe[0] <= 350
    assert applied_ds.probe[-1] >= 1100


# =========================================================================
# Timescale Stitch Dialog Tests
# =========================================================================


def test_timescale_stitch_dialog_init(qapp, synthetic_fs_ns):
    """Test TimescaleStitchDialog initialization and parameter controls."""
    ds_fs, ds_ns = synthetic_fs_ns
    dlg = TimescaleStitchDialog(None, ds_fs, ds_ns)

    assert dlg.result is not None
    assert dlg.result.kind == "timescale"
    assert dlg.rad_quad.isChecked()

    # Switch to scalar scaling model
    dlg.rad_scalar.setChecked(True)
    assert dlg.result.scaling_model == "scalar"

    # Scatter cut toggle
    dlg.chk_scatter.setChecked(True)
    dlg.spin_sc_min.setValue(520.0)
    dlg.spin_sc_max.setValue(540.0)
    assert dlg.result is not None

    # Overlap and blend spinboxes
    dlg.spin_t_ov_min.setValue(0.4)
    dlg.spin_t_ov_max.setValue(1.3)
    dlg.spin_blend_start.setValue(0.7)
    dlg.spin_blend_end.setValue(1.1)
    assert dlg.result.stitch_region == (0.4, 1.3)
    assert dlg.result.blend_window == (0.7, 1.1)


def test_timescale_stitch_dialog_apply_signal(qapp, synthetic_fs_ns):
    """Test that TimescaleStitchDialog emits stitch_applied with a Dataset1D in ns."""
    ds_fs, ds_ns = synthetic_fs_ns
    dlg = TimescaleStitchDialog(None, ds_fs, ds_ns)

    emitted_datasets = []
    dlg.stitch_applied.connect(emitted_datasets.append)

    # Click apply without modal prompt
    dlg._apply_to_workspace(prompt_confirmation=False)

    assert len(emitted_datasets) == 1
    applied_ds = emitted_datasets[0]
    assert isinstance(applied_ds, pm.Dataset1D)
    assert applied_ds.units["unitsT_ltx"] == "ns"
    assert applied_ds.delays[0] < 0
    assert applied_ds.delays[-1] >= 350
