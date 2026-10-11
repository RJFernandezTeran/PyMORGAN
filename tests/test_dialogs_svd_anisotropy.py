"""Unit tests for SVDDialog and AnisotropyDialog GUI components in headless mode."""

import os

import numpy as np
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PyQt6.QtWidgets import QApplication

import pymorgan as pm
from pymorgan.gui.anisotropy_dialog import AnisotropyDialog
from pymorgan.gui.svd_dialog import SVDDialog


@pytest.fixture(scope="session")
def qapp():
    """Ensure a single QApplication instance for Qt tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


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


def test_svd_dialog_initialization_and_controls(qapp, synthetic_ta_dataset):
    """Test SVDDialog initialization, controls, and updates."""
    dlg = SVDDialog(None, synthetic_ta_dataset)
    assert dlg.svd_res is not None
    assert dlg.k_spin.value() == 2

    # Switch rank k
    dlg.k_spin.setValue(3)
    assert dlg.k_slider.value() == 3

    # Switch display scales (User Requirement)
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

    # Monkeypatch show/exec
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
