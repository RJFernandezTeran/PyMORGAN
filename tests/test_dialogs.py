"""Unit tests for specialized GUI dialogs in headless mode:

- SVDDialog (Singular Value Decomposition)
- AnisotropyDialog (Polarization Anisotropy r(t))
- WavelengthStitchDialog (Stitch Spectral Windows)
- TimescaleStitchDialog (Stitch Timescales)
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PyQt6.QtWidgets import QApplication

import pymorgan as pm
from pymorgan.gui.anisotropy_dialog import AnisotropyDialog
from pymorgan.gui.svd_dialog import SVDDialog
from pymorgan.gui.timescale_stitch_dialog import TimescaleStitchDialog
from pymorgan.gui.wavelength_stitch_dialog import WavelengthStitchDialog


@pytest.fixture(scope="session")
def qapp():
    """Ensure a single QApplication instance for Qt tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# =========================================================================== #
# Synthetic Dataset Fixtures                                                  #
# =========================================================================== #


@pytest.fixture
def synthetic_ta_dataset():
    """Synthetic transient absorption dataset."""
    rng = np.random.default_rng(42)
    t = np.linspace(-2, 40, 30)
    p = np.linspace(450, 650, 35)

    kin = np.where(t > 0, np.exp(-t / 8.0), 0.0)
    spec = np.exp(-((p - 530) / 30.0) ** 2)
    z = np.outer(kin, spec) + 0.005 * rng.standard_normal((len(t), len(p)))

    units = {
        "unitsT_lbl": "Delay",
        "unitsT_ltx": "ps",
        "unitsL_lbl": "Wavelength",
        "unitsL_ltx": "nm",
        "unitsZ_lbl": r"$\Delta$A",
        "unitsZ_ltx": "mOD",
    }
    return pm.Dataset1D(Zavg_R=z, delays=t, probe=p, units=units)


@pytest.fixture
def synthetic_multi_dataset():
    """Synthetic multi-detector dataset for anisotropy."""
    rng = np.random.default_rng(42)
    t = np.linspace(-2, 40, 30)
    p = np.linspace(450, 650, 35)

    kin_par = np.where(t > 0, np.exp(-t / 8.0) * (1 + 2 * 0.4 * np.exp(-t / 15.0)), 0.0)
    kin_perp = np.where(t > 0, np.exp(-t / 8.0) * (1 - 0.4 * np.exp(-t / 15.0)), 0.0)
    spec = np.exp(-((p - 530) / 30.0) ** 2)

    z_par = np.outer(kin_par, spec) + 0.002 * rng.standard_normal((len(t), len(p)))
    z_perp = np.outer(kin_perp, spec) + 0.002 * rng.standard_normal((len(t), len(p)))
    z_multi = np.stack([z_par, z_perp], axis=2)

    units = {
        "unitsT_lbl": "Delay",
        "unitsT_ltx": "ps",
        "unitsL_lbl": "Wavelength",
        "unitsL_ltx": "nm",
        "unitsZ_lbl": r"$\Delta$A",
        "unitsZ_ltx": "mOD",
    }
    return pm.Dataset1D(Zavg_R=z_multi, delays=t, probe=p, units=units)


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


# =========================================================================== #
# SVD Dialog Tests                                                            #
# =========================================================================== #


def test_svd_dialog_initialization_and_controls(qapp, synthetic_ta_dataset):
    """Test SVDDialog initialization, controls, and updates."""
    dlg = SVDDialog(None, synthetic_ta_dataset)
    assert dlg.svd_res is not None
    assert dlg.k_spin.value() == 2

    # Switch rank k
    dlg.k_spin.setValue(3)
    assert dlg.k_slider.value() == 3

    # Switch display scales
    dlg.rad_time_symlog.setChecked(True)
    assert dlg.rad_time_symlog.isChecked()
    dlg.rad_scree_linear.setChecked(True)
    assert dlg.rad_scree_linear.isChecked()

    # Presets
    dlg._set_k_for_target_variance(99.0)
    assert dlg.k_spin.value() >= 1

    # Reset limits
    dlg._reset_time_limits(0.0, 20.0)
    assert dlg.tmin_spin.value() == 0.0
    assert dlg.tmax_spin.value() == 20.0


def test_svd_dialog_apply_signal(qapp, synthetic_ta_dataset):
    """Test SVDDialog dataset_applied signal."""
    dlg = SVDDialog(None, synthetic_ta_dataset)
    applied_datasets = []
    dlg.dataset_applied.connect(applied_datasets.append)

    # Directly emit for test verification
    recon = dlg.svd_res.to_dataset(k=2)
    dlg.dataset_applied.emit(recon)

    assert len(applied_datasets) == 1
    assert isinstance(applied_datasets[0], pm.Dataset1D)


# =========================================================================== #
# Anisotropy Dialog Tests                                                     #
# =========================================================================== #


def test_anisotropy_dialog_multi_detector(qapp, synthetic_multi_dataset):
    """Test AnisotropyDialog on multi-detector dataset."""
    dlg = AnisotropyDialog(None, synthetic_multi_dataset)
    assert dlg.rad_channels.isChecked()
    assert dlg.aniso_dataset is not None
    assert dlg.iso_dataset is not None

    # Change G-factor
    dlg.spin_g.setValue(1.05)
    assert dlg.aniso_dataset is not None
    assert dlg.spin_g.value() == 1.05

    # Target component selection
    assert dlg.combo_target_component.count() == 5
    dlg.combo_target_component.setCurrentIndex(1)  # Magic angle
    assert dlg.combo_target_component.currentData() == "magic_angle"


def test_anisotropy_dialog_pymess_channel_names(qapp):
    """Test AnisotropyDialog with PyMESS-style detector channel names."""
    delays = np.array([-1.0, 0.0, 1.0, 5.0])
    probe = np.linspace(500, 600, 10)
    z = np.ones((len(delays), 10, 2))
    z[:, :, 0] = 12.0  # Parallel
    z[:, :, 1] = 3.0   # Perpendicular
    units = {
        "detectors": ["Parallel_pol", "Perp_pol"],
        "pymess_metadata": {"g_factor": "1.12"},
        "unitsT_lbl": "Delay",
        "unitsT_ltx": "ps",
        "unitsL_lbl": "Wavelength",
        "unitsL_ltx": "nm",
        "unitsZ_lbl": r"$\Delta$A",
        "unitsZ_ltx": "mOD",
    }
    ds = pm.Dataset1D(Zavg_R=z, delays=delays, probe=probe, units=units, data_type="PyMESS_PP")
    dlg = AnisotropyDialog(None, ds)

    # Combos should contain descriptive names
    assert "Parallel_pol" in dlg.combo_par.itemText(0)
    assert "Perp_pol" in dlg.combo_perp.itemText(1)
    # G-factor auto-fill from PyMESS metadata
    assert np.isclose(dlg.spin_g.value(), 1.12)
    # Auto-detected indices
    assert dlg.combo_par.currentIndex() == 0
    assert dlg.combo_perp.currentIndex() == 1

    # Verify components computed
    assert "magic_angle" in dlg.components
    assert "difference" in dlg.components
    assert "anisotropy" in dlg.components


def test_mainwindow_anisotropy_menu_integration(qapp, monkeypatch):
    """Test unified Anisotropy menu in MainWindow."""
    from pymorgan.gui.main_window import MainWindow

    monkeypatch.setattr(MainWindow, "show", lambda self: None)
    mw = MainWindow()

    menu = getattr(mw, "_aniso_menu", None)
    assert menu is not None
    assert getattr(mw, "actionAnisotropy", None) is not None
    assert len(mw._quick_aniso_actions) == 5
    assert getattr(mw, "_legacy_aniso_menu", None) is not None

    # Load multi-detector dataset
    delays = np.array([-1.0, 0.0, 1.0, 5.0])
    probe = np.linspace(500, 600, 10)
    z = np.ones((len(delays), 10, 2))
    z[:, :, 0] = 10.0
    z[:, :, 1] = 4.0
    ds = pm.Dataset1D(
        Zavg_R=z,
        delays=delays,
        probe=probe,
        units={"detectors": ["Parallel", "Perpendicular"]},
    )
    mw.dataset = ds
    mw._configure_state_controls(None)

    assert menu.isEnabled()
    for act in mw._quick_aniso_actions:
        assert act.isEnabled()

    # Test quick magic angle calculation
    par_i, perp_i = mw._resolve_polarization_detectors()
    assert par_i == 0
    assert perp_i == 1

    mw._apply_quick_anisotropy("magic_angle")
    # Expected magic angle: (10 + 2*4) / 3 = 18 / 3 = 6.0
    assert mw.dataset is not None
    assert np.allclose(mw.dataset.Z[:, :, 0], 6.0)

    # Test quick anisotropy calculation
    mw.dataset = ds  # Reset to 2-detector dataset
    mw._apply_quick_anisotropy("anisotropy")
    # Expected anisotropy: (10 - 4) / 18 = 6 / 18 = 0.333333
    assert np.allclose(mw.dataset.Z[:, :, 0], 6.0 / 18.0)


# =========================================================================== #
# Wavelength Stitch Dialog Tests                                              #
# =========================================================================== #


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


def test_wavelength_stitch_dialog_mid_ir(qapp):
    """Test WavelengthStitchDialog handling of Mid-IR windows (cm-1)."""
    t = np.linspace(0, 10, 15)
    p1 = np.linspace(1500, 1750, 25)
    p2 = np.linspace(1700, 1950, 25)
    z1 = np.ones((len(t), len(p1))) * 2.0
    z2 = np.ones((len(t), len(p2))) * 4.0

    units = {
        "unitsT_lbl": "Delay",
        "unitsT_ltx": "ps",
        "unitsL_lbl": "Wavenumber",
        "unitsL_ltx": "cm-1",
        "unitsZ_lbl": r"$\Delta$A",
        "unitsZ_ltx": "mOD",
    }
    ds1 = pm.Dataset1D(Zavg_R=z1, delays=t, probe=p1, units=units)
    ds2 = pm.Dataset1D(Zavg_R=z2, delays=t, probe=p2, units=units)

    dlg = WavelengthStitchDialog(None, ds1, ds2)
    assert dlg._probe_unit_display() == "cm-1"
    assert dlg._probe_name() == "Wavenumber"
    assert dlg.windowTitle() == "Stitch Spectral Windows - PyMORGAN"
    assert dlg.result is not None
    assert dlg.result.probe[0] <= 1500
    assert dlg.result.probe[-1] >= 1950


# =========================================================================== #
# Timescale Stitch Dialog Tests                                               #
# =========================================================================== #


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


def test_stitch_dialogs_arcsinh_and_draggable_legends(qapp, synthetic_uv_nir, synthetic_fs_ns):
    """Test arcsinh switch, percentage controls, and draggable legends in both stitch dialogs."""
    ds_uv, ds_nir = synthetic_uv_nir
    dlg_wl = WavelengthStitchDialog(None, ds_uv, ds_nir)

    assert hasattr(dlg_wl, "chk_asinh")
    assert hasattr(dlg_wl, "spin_asinh_pct")
    assert not dlg_wl.chk_asinh.isChecked()
    assert not dlg_wl.spin_asinh_pct.isEnabled()

    # Enable arcsinh
    dlg_wl.chk_asinh.setChecked(True)
    assert dlg_wl.spin_asinh_pct.isEnabled()
    dlg_wl.spin_asinh_pct.setValue(12.5)

    # Check that top legend is draggable
    fig_wl = dlg_wl.canvas.figure
    axes_wl = fig_wl.axes
    assert len(axes_wl) >= 2
    leg_wl = axes_wl[0].get_legend()
    assert leg_wl is not None
    assert leg_wl.get_draggable()

    # Timescale dialog
    ds_fs, ds_ns = synthetic_fs_ns
    dlg_ts = TimescaleStitchDialog(None, ds_fs, ds_ns)

    assert hasattr(dlg_ts, "chk_asinh")
    assert hasattr(dlg_ts, "spin_asinh_pct")
    dlg_ts.chk_asinh.setChecked(True)
    assert dlg_ts.spin_asinh_pct.isEnabled()
    dlg_ts.spin_asinh_pct.setValue(8.0)

    fig_ts = dlg_ts.canvas.figure
    axes_ts = fig_ts.axes
    assert len(axes_ts) >= 3
    leg_kin = axes_ts[0].get_legend()
    assert leg_kin is not None
    assert leg_kin.get_draggable()


def test_dialog_blend_combobox_selection(qapp, synthetic_uv_nir, synthetic_fs_ns):
    """Test switching blend method via combo_blend in both dialogs."""
    ds_uv, ds_nir = synthetic_uv_nir
    dlg_wl = WavelengthStitchDialog(None, ds_uv, ds_nir)
    assert hasattr(dlg_wl, "combo_blend")

    # Select Linear
    idx_lin = dlg_wl.combo_blend.findData("linear")
    dlg_wl.combo_blend.setCurrentIndex(idx_lin)
    assert dlg_wl.result.blend_method == "linear"

    # Select Step
    idx_step = dlg_wl.combo_blend.findData("step")
    dlg_wl.combo_blend.setCurrentIndex(idx_step)
    assert dlg_wl.result.blend_method == "step"

    # Timescale dialog
    ds_fs, ds_ns = synthetic_fs_ns
    dlg_ts = TimescaleStitchDialog(None, ds_fs, ds_ns)
    assert hasattr(dlg_ts, "combo_blend")

    idx_quintic = dlg_ts.combo_blend.findData("quintic")
    dlg_ts.combo_blend.setCurrentIndex(idx_quintic)
    assert dlg_ts.result.blend_method == "quintic"


def test_recalc_average_dialog(qapp):
    """Test RecalcAverageDialog scan text input and noise-weighted toggle."""
    from pymorgan.gui.tabs.oneD import RecalcAverageDialog

    # Default is noise-weighted = True
    dlg = RecalcAverageDialog(None, n_scans=8, default_weighted=True)
    assert dlg.scan_text() == "1-8"
    assert dlg.is_noise_weighted() is True

    # User toggles to direct averaging
    dlg.noise_weighted_chk.setChecked(False)
    assert dlg.is_noise_weighted() is False

    # Changing scan text
    dlg.scans_edit.setText("1-4, 6")
    assert dlg.scan_text() == "1-4, 6"

    # Default initialized as False
    dlg_direct = RecalcAverageDialog(None, n_scans=5, default_weighted=False)
    assert dlg_direct.is_noise_weighted() is False

