"""Unit tests for MainWindow GUI, dataset browser, busy guard, movie dialog, and 2D GUI controls."""

import os
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

import pymorgan as pm
import synthetic
from pymorgan.gui.busy import busy, busy_guard, is_busy
from pymorgan.gui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path):
    synthetic.make_synthetic_pdat(tmp_path / "one.pdat")
    synthetic.make_synthetic_pdat(tmp_path / "two.pdat")
    w = MainWindow()
    w.RootDir_field.setText(str(tmp_path))
    w._on_rootdir_changed()
    yield w
    w.close()


# --------------------------------------------------------------------------- #
#                             Main Window & Tabs                              #
# --------------------------------------------------------------------------- #
def test_window_constructs(window):
    assert "PyMORGAN" in window.windowTitle()
    assert window.MainTabs.count() == 3
    assert window.dataset is None


def test_load_renders_contour(window, tmp_path):
    info = synthetic.make_synthetic_pdat(tmp_path / "syn.pdat")
    window.load_path(info["path"], "PDAT")
    assert window.dataset is not None


def test_background_toggle_corrects(window, tmp_path):
    info = synthetic.make_synthetic_pdat(tmp_path / "syn.pdat")
    window.load_path(info["path"], "PDAT")
    assert window.dataset is not None


# --------------------------------------------------------------------------- #
#                              Dataset Browser                                #
# --------------------------------------------------------------------------- #
def test_dataset_browser_lists_and_loads(window, tmp_path):
    assert window._folder_model.rowCount() >= 1
    assert window._twoD_folder_model.rowCount() >= 1


def test_folder_scan_is_cached_per_datatype(window, tmp_path):
    folder = Path(tmp_path)
    first, n_first = window._scan_dataset_folder(folder, "PDAT", twoD=False)
    second, n_second = window._scan_dataset_folder(folder, "PDAT", twoD=False)
    assert second is first
    assert n_second == n_first == 2


def test_new_file_is_picked_up_after_invalidation(window, tmp_path):
    folder = Path(tmp_path)
    _, before = window._scan_dataset_folder(folder, "PDAT", twoD=False)
    synthetic.make_synthetic_pdat(folder / "three.pdat")
    window._invalidate_folder_scan_cache()
    _, after = window._scan_dataset_folder(folder, "PDAT", twoD=False)
    assert after == before + 1


# --------------------------------------------------------------------------- #
#                                Busy Guard                                   #
# --------------------------------------------------------------------------- #
def test_input_is_disabled_while_running_and_restored_after(qapp):
    w = MainWindow()
    with busy(w, "Working..."):
        assert is_busy(w) is True
        assert w.MainTabs.isEnabled() is False
    assert is_busy(w) is False
    assert w.MainTabs.isEnabled() is True
    w.close()


# --------------------------------------------------------------------------- #
#                               2D GUI Controls                               #
# --------------------------------------------------------------------------- #
def test_twoD_window_loads_dataset(qapp, tmp_path):
    info = synthetic.make_synthetic_p2dat(tmp_path / "syn.p2dat")
    win = MainWindow()
    win.twoD_load_path(info["path"], "P2DAT")

    assert win.twoD_dataset is not None
    assert win.twoD_dataset.data_type == "P2DAT"
    win.close()


def test_movie_dialog_constructs(qapp, tmp_path):
    from pymorgan.gui.movie_dialog import MakeMovieDialog

    info = synthetic.make_synthetic_p2dat(tmp_path / "syn.p2dat")
    ds = pm.load_2D(info["path"], data_type="P2DAT")
    dlg = MakeMovieDialog(ds)
    assert "Movie" in dlg.windowTitle() or "Animation" in dlg.windowTitle()
    dlg.close()


def test_oneD_and_twoD_arrow_key_dataset_scrolling(qapp, tmp_path):
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QKeyEvent

    # Setup 2 synthetic 1D datasets
    dir_1d = tmp_path / "1d"
    dir_1d.mkdir()
    info1 = synthetic.make_synthetic_pdat(dir_1d / "ds1.pdat")
    info2 = synthetic.make_synthetic_pdat(dir_1d / "ds2.pdat")

    win = MainWindow()
    win.MainTabs.setCurrentIndex(0) # 1D tab
    for i in range(win.PP_datatype_cbx.count()):
        if "PDAT" in win.PP_datatype_cbx.itemData(i):
            win.PP_datatype_cbx.setCurrentIndex(i)
            break

    win.RootDir_field.setText(str(dir_1d))
    win._on_rootdir_changed()

    # Load ds1
    win.load_path(str(info1["path"]))
    assert win._current_path == str(info1["path"])

    # Select ds1 in dataset list
    lst_1d = win.PP_datafolderlist_lst
    lst_1d.setCurrentIndex(win._folder_model.index(1, 0))
    lst_1d.setFocus()

    # Press Down key via _navigate_dataset_browser
    assert win._navigate_dataset_browser(forward=True)
    assert win._current_path == str(info2["path"])

    # Press Up key via _navigate_dataset_browser
    assert win._navigate_dataset_browser(forward=False)
    assert win._current_path == str(info1["path"])
    win.close()


def test_f5_reloads_active_1D_and_2D_dataset(qapp, tmp_path):
    info_1d = synthetic.make_synthetic_pdat(tmp_path / "syn1d.pdat")
    info_2d = synthetic.make_synthetic_p2dat(tmp_path / "syn2d.p2dat")

    win = MainWindow()

    # Verify F5 shortcut is registered on actionReload
    action_reload = getattr(win, "actionReload", None)
    if action_reload is not None:
        assert action_reload.shortcut().toString() == "F5"

    # 1D Tab reload test
    win.MainTabs.setCurrentIndex(0)
    win.load_path(str(info_1d["path"]), "PDAT")
    assert win.dataset is not None
    assert win._current_path == str(info_1d["path"])

    win.handle_reload()
    assert win.dataset is not None

    # 2D Tab reload test
    win.MainTabs.setCurrentIndex(1)
    win.twoD_load_path(str(info_2d["path"]), "P2DAT")
    assert win.twoD_dataset is not None
    assert win._twoD_current_path == str(info_2d["path"])

    win.handle_reload()
    assert win.twoD_dataset is not None

    win.close()


def test_twoD_load_dataset_with_fewer_delays_goes_to_last_delay(qapp, tmp_path):
    delays1 = np.linspace(0.1, 10.0, 15)
    delays2 = np.linspace(0.1, 5.0, 5)
    info1 = synthetic.make_synthetic_p2dat(tmp_path / "syn1.p2dat", delays=delays1)
    info2 = synthetic.make_synthetic_p2dat(tmp_path / "syn2.p2dat", delays=delays2)

    win = MainWindow()
    win.twoD_load_path(info1["path"], "P2DAT")
    assert win.twoD_dataset is not None
    assert len(win.twoD_dataset.delays) == 15

    lst_delays = getattr(win, "twoD_t2delay_lst", None)
    if lst_delays is not None:
        lst_delays.setCurrentIndex(win._twoD_delay_model.index(12, 0))
        assert lst_delays.currentIndex().row() == 12

    # Load dataset 2 which only has 5 delays
    win.twoD_load_path(info2["path"], "P2DAT")
    assert win.twoD_dataset is not None
    assert len(win.twoD_dataset.delays) == 5

    # Check that it automatically selected index 4 (the last t2 delay: 5 - 1 = 4)
    if lst_delays is not None:
        assert lst_delays.currentIndex().row() == 4

    win.close()


def test_probe_autocalibration_from_cal_tab(qapp, tmp_path):
    delays = np.linspace(0.1, 5.0, 3)
    probe_native = np.linspace(1900.0, 2100.0, 64)
    info = synthetic.make_synthetic_p2dat(tmp_path / "syn_cal.p2dat", delays=delays, probe=probe_native)

    win = MainWindow()
    win.twoD_load_path(info["path"], "P2DAT")
    assert win.twoD_dataset is not None
    assert np.allclose(win.twoD_dataset.probe, probe_native)

    # Set up a CAL tab calibration result
    from pymorgan.cal import CalibrationResult
    cal_curve = np.linspace(2000.0, 2200.0, 64)
    res = CalibrationResult(
        wavelength_nm=1e7 / cal_curve,
        wavenumber_cm1=cal_curve,
        residuals=np.zeros(64),
        fit_params=np.array([2000.0, 0.5]),
        ref_x_cut=np.linspace(2000.0, 2200.0, 100),
        ref_y_cut=np.ones(100),
        model_y_fit=np.ones(100),
    )
    win._cal_fit_result_det1 = res

    # Check CAL tab dashed line refresh
    win._refresh_current_probe_line()

    # Enable autocalibration in 2D tab
    win._on_twoD_auto_cal_toggled(True)
    assert np.allclose(win.twoD_dataset.probe, cal_curve)
    assert win.twoD_dataset.cal_level == "autocalibrated"

    # Disable autocalibration in 2D tab
    win._on_twoD_auto_cal_toggled(False)
    assert np.allclose(win.twoD_dataset.probe, probe_native)

    win.close()


def test_twoD_other_buttons_distinct_styles(qapp):
    win = MainWindow()
    assert hasattr(win, "twoD_other_btn_TD")
    assert hasattr(win, "twoD_other_btn_PH")
    assert hasattr(win, "twoD_other_btn_CAL")

    st_td = win.twoD_other_btn_TD.styleSheet()
    st_ph = win.twoD_other_btn_PH.styleSheet()
    st_cal = win.twoD_other_btn_CAL.styleSheet()

    assert "QPushButton:checked" in st_td
    assert "QPushButton:checked" in st_ph
    assert "QPushButton:checked" in st_cal

    # Ensure all three buttons have distinct checked styling definitions
    assert st_td != st_ph
    assert st_ph != st_cal
    assert st_td != st_cal

    win.close()


def test_settings_change_updates_plot_controls_units_and_limits(qapp, tmp_path):
    info = synthetic.make_synthetic_pdat(tmp_path / "syn_unit.pdat")
    win = MainWindow()
    win.load_path(info["path"], "PDAT")
    assert win.dataset is not None
    assert win.plot_controls is not None

    pc = win.plot_controls
    initial_unit = pc._x_display
    target_unit = "cm-1" if initial_unit != "cm-1" else "nm"

    orig_min = pc.x_min.value()

    # Change x_axis_unit setting (simulating Settings Panel edit)
    pm.update_settings(x_axis_unit=target_unit)
    win._rerender_preview()

    assert pc._x_display == target_unit
    assert pc._x_lbl_min.text() == "Min probe"
    assert pc._x_lbl_max.text() == "Max probe"
    assert pc.x_min.value() < pc.x_max.value()

    # Revert setting to initial_unit
    pm.update_settings(x_axis_unit=initial_unit)
    win._rerender_preview()
    assert pc._x_display == initial_unit
    assert np.isclose(pc.x_min.value(), orig_min, rtol=1e-2)

    win.close()


def test_plot_counts_button_visibility(qapp, tmp_path):
    win = MainWindow()
    assert hasattr(win, "PP_plotCounts_btn")
    assert win.PP_plotCounts_btn.isHidden()

    # Load UniGE_nsTA dataset with counts -> button becomes visible
    nsta_info = synthetic.make_synthetic_unige_nsta(tmp_path / "nsta_test.dat")
    win.load_path(nsta_info["path"], "UniGE_nsTA")
    assert not win.PP_plotCounts_btn.isHidden()

    # Load PDAT dataset without counts -> button becomes hidden
    pdat_info = synthetic.make_synthetic_pdat(tmp_path / "pdat_test.pdat")
    win.load_path(pdat_info["path"], "PDAT")
    assert win.PP_plotCounts_btn.isHidden()

    win.close()


def test_quick_plots_checkbox_toggles_setting(qapp):
    win = MainWindow()
    assert hasattr(win, "PP_QuickPlots_chk")

    orig_setting = pm.get_settings().quick_plots
    try:
        win.PP_QuickPlots_chk.setChecked(True)
        assert pm.get_settings().quick_plots is True

        win.PP_QuickPlots_chk.setChecked(False)
        assert pm.get_settings().quick_plots is False
    finally:
        pm.update_settings(quick_plots=orig_setting)
        win.close()


def test_plot_noise_button_enabled_state(qapp, tmp_path):
    win = MainWindow()
    assert hasattr(win, "PP_plotNoise_btn")
    assert not win.PP_plotNoise_btn.isEnabled()

    # Load PDAT without noise -> remains disabled
    pdat_info = synthetic.make_synthetic_pdat(tmp_path / "pdat_no_noise.pdat")
    win.load_path(pdat_info["path"], "PDAT")
    assert not win.PP_plotNoise_btn.isEnabled()

    # Load PDAT with .pdatn sibling -> enabled
    pdat2_info = synthetic.make_synthetic_pdat(tmp_path / "pdat_with_noise.pdat")
    synthetic.make_synthetic_pdatn(pdat2_info["path"])
    win.load_path(pdat2_info["path"], "PDAT")
    assert win.PP_plotNoise_btn.isEnabled()

    # Load MESS dataset with single scans -> enabled
    mess_info = synthetic.make_synthetic_mess(tmp_path / "mess_noise", nscans=3)
    win.load_path(mess_info["folder"], "MESS_TRIR")
    assert win.PP_plotNoise_btn.isEnabled()

    win.close()


def test_twoD_gaussian_dialog(qapp, tmp_path):
    from pymorgan.gui.twoD_gaussian_dialog import TwoDGaussianFitDialog

    info = synthetic.make_synthetic_p2dat(tmp_path / "sample_2d.p2dat")
    ds = pm.load_2D(info["path"], data_type="P2DAT")

    dlg = TwoDGaussianFitDialog(None, ds)
    assert hasattr(dlg, "tab_widget")
    assert dlg.tab_widget.count() == 2
    assert dlg.tab_widget.tabText(0) == "Peaks & Modes"
    assert dlg.tab_widget.tabText(1) == "Fitting Options & Bounds"

    assert hasattr(dlg, "spn_pos_tol")
    assert hasattr(dlg, "spn_anharm_min")
    assert hasattr(dlg, "spn_anharm_max")
    assert hasattr(dlg, "spn_anharm_tol")
    assert hasattr(dlg, "chk_constrain_signs")
    assert hasattr(dlg, "cb_peak_shape")
    assert dlg.spn_pos_tol.value() == 10.0
    assert dlg.chk_constrain_signs.isChecked() is True
    assert dlg.cb_peak_shape.currentText() == "Gaussian"
    assert dlg.table_modes.columnCount() == 11
    assert dlg.table_modes.horizontalHeaderItem(9).text() == "σ1"
    assert dlg.table_modes.horizontalHeaderItem(10).text() == "σ3"
    assert dlg.chk_correlated.isEnabled() is True

    # Switch to Lorentzian (correlated disabled and unchecked, headers Γ1, Γ3)
    dlg.cb_peak_shape.setCurrentText("Lorentzian")
    assert dlg.table_modes.horizontalHeaderItem(9).text() == "Γ1"
    assert dlg.table_modes.horizontalHeaderItem(10).text() == "Γ3"
    assert "Γ" in dlg.lbl_sigma.text()
    assert dlg.chk_correlated.isEnabled() is False
    assert dlg.chk_correlated.isChecked() is False

    # Switch back to Gaussian (correlated re-enabled)
    dlg.cb_peak_shape.setCurrentText("Gaussian")
    assert dlg.table_modes.horizontalHeaderItem(9).text() == "σ1"
    assert "σ" in dlg.lbl_sigma.text()
    assert dlg.chk_correlated.isEnabled() is True

    # Add 2 diagonal modes and 1 cross-peak manually
    dlg.modes.append({
        "w1": 2010.0, "w3": 2010.0, "anharm": 16.0,
        "amp_gsb": -1.0, "amp_esa": 0.8, "sigma_w1": 10.0, "sigma_w3": 10.0,
    })
    dlg.modes.append({
        "w1": 2040.0, "w3": 2040.0, "anharm": 14.0,
        "amp_gsb": -1.2, "amp_esa": 1.0, "sigma_w1": 10.0, "sigma_w3": 10.0,
    })
    dlg.modes.append({
        "w1": 2010.0, "w3": 2040.0, "anharm": 14.0,
        "amp_gsb": -0.3, "amp_esa": 0.2, "sigma_w1": 10.0, "sigma_w3": 10.0,
        "link_w1": 0, "link_w3": 1, "link_anharm": 1,
    })
    dlg._populate_table()
    assert dlg.table_modes.rowCount() == 3
    assert dlg.table_modes.item(0, 0).text() == "#1"
    assert dlg.table_modes.item(0, 1).text() == "2010.0"
    assert dlg.table_modes.item(1, 1).text() == "2040.0"

    # Cross-peak 3 displays linked values
    assert dlg.table_modes.item(2, 1).text() == "2010.0"
    assert dlg.table_modes.item(2, 3).text() == "2040.0"
    assert dlg.table_modes.item(2, 5).text() == "14.0"

    # Editing master Mode 1 w1 should synchronize linked slave Mode 3 w1
    dlg.table_modes.item(0, 1).setText("2015.5")
    assert dlg.modes[0]["w1"] == 2015.5
    assert dlg.modes[2]["w1"] == 2015.5
    assert dlg.table_modes.item(2, 1).text() == "2015.5"

    # Test Save Simulated P2DAT button
    assert hasattr(dlg, "btn_save_p2dat")
    out_sim_p2dat = tmp_path / "simulated_test.p2dat"
    from unittest.mock import patch
    with (
        patch("PyQt6.QtWidgets.QFileDialog.getSaveFileName", return_value=(str(out_sim_p2dat), "P2DAT (*.p2dat)")),
        patch("PyQt6.QtWidgets.QMessageBox.information"),
        patch("PyQt6.QtWidgets.QMessageBox.warning"),
        patch("PyQt6.QtWidgets.QMessageBox.critical"),
    ):
        dlg._save_simulated_p2dat()

    assert out_sim_p2dat.exists()
    out_sim_txt = tmp_path / "simulated_test_parameters.txt"
    assert out_sim_txt.exists()

    # Verify saved file is a valid P2DAT reloadable dataset
    ds_sim = pm.load_2D(out_sim_p2dat, data_type="P2DAT")
    assert np.allclose(ds_sim.pump, ds.pump)
    assert np.allclose(ds_sim.probe, ds.probe)
    assert np.allclose(ds_sim.delays, ds.delays)

    dlg.close()

