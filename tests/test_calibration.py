"""Unit tests for spectrometer calibration math models, optimisation backend, load utilities, and GUI controls."""

import os

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from pymorgan.cal import (
    fit_wavelength_axis,
    linear_grating_pixel_mapping,
    load_experimental_spectrum,
    load_reference_spectrum,
    merge_calibration,
    n_sellmeier_sf10,
    polynomial_grating_pixel_mapping,
    prism_output_angle,
    save_calibration_file,
    split_calibration,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# --------------------------------------------------------------------------- #
#                           Backend Calibration Math                          #
# --------------------------------------------------------------------------- #
def test_sf10_sellmeier_refractive_index():
    n_1100 = n_sellmeier_sf10(1100.0)
    assert 1.68 < n_1100 < 1.75

    n_500 = n_sellmeier_sf10(500.0)
    n_1500 = n_sellmeier_sf10(1500.0)
    assert n_500 > n_1500


def test_prism_output_angle():
    th_out = prism_output_angle(1100.0, 60.0)
    assert 0.5 < th_out < 1.5


def test_grating_pixel_mappings():
    ref_wl = np.array([400.0, 500.0, 600.0])
    pixels = linear_grating_pixel_mapping(ref_wl, central_wl=500.0, ppnm=1.0, n_pix=128)
    np.testing.assert_allclose(pixels, [-35.5, 64.5, 164.5])

    pix_poly = polynomial_grating_pixel_mapping(
        ref_wl, central_wl=350.0, ppnm=1.0, c2=-4.0, c3=6.0
    )
    assert len(pix_poly) == 3


def test_fit_wavelength_axis_synthetic():
    n_pix = 128
    meas_y = np.sin(np.linspace(0, np.pi, n_pix)) ** 2
    ref_x = np.linspace(350, 750, 500)
    ref_y = np.sin(np.linspace(0, np.pi, 500)) ** 2

    res = fit_wavelength_axis(
        meas_y,
        ref_x,
        ref_y,
        cwl=530.0,
        ppnm_guess=1.1,
        cal_type_code=2,
    )

    assert len(res.wavelength_nm) == n_pix
    assert len(res.wavenumber_cm1) == n_pix
    assert np.all(res.wavelength_nm > 0)


# --------------------------------------------------------------------------- #
#                       Load Utilities & Split/Merge                           #
# --------------------------------------------------------------------------- #
def test_split_and_merge_calibration(tmp_path):
    cm_synth = np.linspace(1000, 2000, 256)

    lhs, rhs = split_calibration(cm_synth)
    assert len(lhs) == 128
    assert len(rhs) == 128
    np.testing.assert_allclose(lhs, cm_synth[:128])
    np.testing.assert_allclose(rhs, cm_synth[128:])

    merged = merge_calibration(lhs, rhs)
    assert len(merged) == 256
    np.testing.assert_allclose(merged, cm_synth)

    p_main, p_stamp = save_calibration_file(merged, tmp_path, save_timestamped=True)
    assert p_main.exists()
    assert p_stamp is not None and p_stamp.exists()


def test_load_reference_spectrum(tmp_path):
    csv_file = tmp_path / "Polystyrene.csv"
    data = np.column_stack([np.linspace(400, 700, 100), np.sin(np.linspace(0, np.pi, 100))])
    np.savetxt(csv_file, data, delimiter=",")

    ref = load_reference_spectrum(csv_file)
    assert ref.name == "Polystyrene"
    assert ref.axis_unit == "nm"
    assert len(ref.spectral_axis) == 100
    assert len(ref.absorbance) == 100


# --------------------------------------------------------------------------- #
#                            Calibration GUI Controls                         #
# --------------------------------------------------------------------------- #
def test_led_indicator_states(qapp):
    from pymorgan.gui.widgets import LEDIndicator

    led = LEDIndicator()
    assert led.state() == "off"

    led.set_state("loading")
    assert led.state() == "loading"

    led.set_state("loaded")
    assert led.state() == "loaded"

    led.set_state("error")
    assert led.state() == "error"

    led.set_state("gray")
    assert led.state() == "gray"


def test_unige_fsta_nsta_calibration_defaults():
    from pymorgan.cal import get_default_calibration_params

    p_fsta = get_default_calibration_params(2)
    assert p_fsta["cwl"] == 535.0
    assert p_fsta["ppnm_guess"] == 1.1

    p_nsta = get_default_calibration_params(3)
    assert p_nsta["cwl"] == 540.0
    assert p_nsta["ppnm_guess"] == 1.4


def test_unige_nsta_loader(tmp_path):
    # Synthetic tab-delimited 5-column dat file (pixel, signal, rms signal, reference, rms reference)
    pix = np.arange(100)
    sig = np.sin(np.linspace(0, np.pi, 100)) + 10.0
    rms_sig = np.ones(100) * 0.1
    ref = np.ones(100) * 12.0
    rms_ref = np.ones(100) * 0.2

    mat = np.column_stack([pix, sig, rms_sig, ref, rms_ref])
    dat_path = tmp_path / "nsta_sample.dat"
    np.savetxt(dat_path, mat, delimiter="\t", header="pixel\tsignal\trms signal\treference\trms reference", comments="% ")

    exp = load_experimental_spectrum(dat_path, cal_type_code=3)
    assert len(exp.detector_data) == 1
    assert len(exp.detector_data[0]) == 100
    np.testing.assert_allclose(exp.detector_data[0], sig)

    # ps-ta style export with double tab (\t\t) between signal and rms signal
    double_tab_path = tmp_path / "nsta_double_tab.dat"
    lines = ["% pixel\tsignal\trms signal\treference\trms reference"]
    for p, s, rs, r, rr in zip(pix, sig, rms_sig, ref, rms_ref, strict=True):
        lines.append(f"{p}\t{s:.6f}\t\t{rs:.4f}\t{r:.2f}\t{rr:.4f}")
    double_tab_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    exp_dt = load_experimental_spectrum(double_tab_path, cal_type_code=3)
    assert len(exp_dt.detector_data) == 1
    assert len(exp_dt.detector_data[0]) == 100
    np.testing.assert_allclose(exp_dt.detector_data[0], sig, atol=1e-6)


def test_shaper_panel_visibility_toggle(qapp):
    from pymorgan.gui.main_window import MainWindow

    win = MainWindow()
    win._ensure_calibration_initialized()
    assert hasattr(win, "cal_shaper_group")
    assert hasattr(win, "cal_type_combo")

    idx_fsta = win.cal_type_combo.findText("UniGE fsTA")
    idx_nsta = win.cal_type_combo.findText("UniGE nsTA")
    idx_trir = win.cal_type_combo.findText("UniGE TRIR (Intensity)")

    assert idx_fsta >= 0
    assert idx_nsta >= 0
    assert idx_trir >= 0

    win.cal_type_combo.setCurrentIndex(idx_fsta)
    assert win.cal_shaper_group.isHidden()

    win.cal_type_combo.setCurrentIndex(idx_nsta)
    assert win.cal_shaper_group.isHidden()

    win.cal_type_combo.setCurrentIndex(idx_trir)
    assert not win.cal_shaper_group.isHidden()


def test_load_harpia_ta_calibration_spectrum(tmp_path):
    from pymorgan.cal import load_HARPIA_calibration_spectrum

    # Test synthetic HARPIA calibration file
    wl = np.linspace(330.0, 530.0, 200)
    sig = np.sin(np.linspace(0, np.pi, 200)) * 5.0
    mat = np.column_stack([wl, sig])
    fpath = tmp_path / "harpia_cal.txt"
    np.savetxt(fpath, mat, delimiter="\t", header="Wavelength (nm)\tDetector signal (V)", comments="")

    exp = load_HARPIA_calibration_spectrum(fpath)
    assert len(exp.detector_data) == 1
    assert len(exp.detector_data[0]) == 200
    np.testing.assert_allclose(exp.detector_data[0], sig)
    assert exp.wavelengths is not None
    np.testing.assert_allclose(exp.wavelengths, wl)
    assert exp.min_wl == pytest.approx(330.0)
    assert exp.max_wl == pytest.approx(530.0)

    # Test load_experimental_spectrum with cal_type_code=11
    exp11 = load_experimental_spectrum(fpath, cal_type_code=11)
    assert len(exp11.detector_data) == 1
    np.testing.assert_allclose(exp11.detector_data[0], sig)


def test_available_calibration_types():
    import pymorgan as pm
    types = pm.available_calibration_types()
    assert isinstance(types, list)
    assert len(types) >= 11
    assert "Light Conversion HARPIA-TA (Int.)" in types
    assert "UoS TRIR" in types

    mapping = pm.available_calibration_types(as_dict=True)
    assert isinstance(mapping, dict)
    assert mapping[11] == "Light Conversion HARPIA-TA (Int.)"
    assert mapping[1] == "UoS TRIR"

    code = pm.available_calibration_types("HARPIA")
    assert code == 11

    pymess_code = pm.available_calibration_types("PyMESS")
    assert pymess_code == 12
    assert mapping[12] == "PyMESS"


def test_harpia_uv_and_vis_defaults():
    from pymorgan.cal import get_default_calibration_params

    p_uv = get_default_calibration_params(11, spectral_range="UV Range")
    p_vis = get_default_calibration_params(11, spectral_range="Vis Range")

    assert p_uv["cwl"] == pytest.approx(430.0)
    assert p_uv["ppnm_guess"] == pytest.approx(1.23)
    assert p_uv["rel_min_nm"] == pytest.approx(-40.0)
    assert p_uv["rel_max_nm"] == pytest.approx(115.0)

    assert p_vis["cwl"] == pytest.approx(685.0)
    assert p_vis["ppnm_guess"] == pytest.approx(0.62)
    assert p_vis["rel_min_nm"] == pytest.approx(-190.0)
    assert p_vis["rel_max_nm"] == pytest.approx(190.0)
    assert p_uv["ppnm_guess"] != p_vis["ppnm_guess"]


def test_pymess_calibration_defaults_and_pathways():
    from pymorgan.cal import get_default_calibration_params

    p_ir = get_default_calibration_params(12, spectral_range="Mid-IR")
    p_uv = get_default_calibration_params(12, spectral_range="UV/Vis")
    p_nir = get_default_calibration_params(12, spectral_range="NIR")

    assert p_ir == get_default_calibration_params(8)
    assert p_uv == get_default_calibration_params(10)
    assert p_nir == get_default_calibration_params(5)


def test_pymess_h5_calibration_loader_synthetic(tmp_path):
    import h5py

    from pymorgan.cal import load_PyMESS_calibration_spectrum

    # 1. Mid-IR PyMESS .h5 file -> follows UniGE TRIR pathway
    ir_h5 = tmp_path / "Probe_air_ir.h5"
    with h5py.File(ir_h5, "w") as f:
        d = f.create_group("data")
        d.create_dataset("probe_spectrum", data=np.linspace(1000.0, 5000.0, 64))
        d.create_dataset("wavelengths", data=np.linspace(1990.0, 2180.0, 64))
        m = f.create_group("metadata")
        m.attrs["setup_id"] = "TRIR-2DIR_UniGE"
        m.attrs["spectrometer_cwl_nm"] = "4807.707"
        m.attrs["spectrometer_cwn_cm1"] = "2079.994"
        m.attrs["spectrometer_grating"] = "150 l/mm"

    exp_ir = load_PyMESS_calibration_spectrum(ir_h5)
    assert exp_ir.spectral_range == "mid-ir"
    assert len(exp_ir.detector_data[0]) == 64
    assert exp_ir.cwl[0] == pytest.approx(2079.994)
    assert exp_ir.gratings[0] == pytest.approx(150.0)
    assert exp_ir.wavenumbers is not None

    # 2. UV/Vis PyMESS .h5 file -> follows TRUVIS-II pathway
    uv_h5 = tmp_path / "Probe_air_uv.h5"
    with h5py.File(uv_h5, "w") as f:
        d = f.create_group("data")
        d.create_dataset("probe_spectrum", data=np.linspace(1.0, 10.0, 512))
        d.create_dataset("wavelengths", data=np.linspace(350.0, 750.0, 512))
        m = f.create_group("metadata")
        m.attrs["setup_id"] = "TRUVIS_UniGE"

    exp_uv = load_PyMESS_calibration_spectrum(uv_h5)
    assert exp_uv.spectral_range == "uv-vis"
    assert len(exp_uv.detector_data[0]) == 512
    assert exp_uv.gratings[0] == 0.0
    assert exp_uv.wavelengths is not None

    # 3. NIR PyMESS .h5 file -> follows UniGE NIR-TA pathway
    nir_h5 = tmp_path / "Probe_air_nir.h5"
    with h5py.File(nir_h5, "w") as f:
        d = f.create_group("data")
        d.create_dataset("probe_spectrum", data=np.linspace(1.0, 10.0, 256))
        d.create_dataset("wavelengths", data=np.linspace(850.0, 1550.0, 256))
        m = f.create_group("metadata")
        m.attrs["spectrometer_cwl_nm"] = "1150.0"

    exp_nir = load_PyMESS_calibration_spectrum(nir_h5)
    assert exp_nir.spectral_range == "nir"
    assert len(exp_nir.detector_data[0]) == 256
    assert exp_nir.cwl[0] == pytest.approx(1150.0)
    assert exp_nir.gratings[0] == 0.0


def test_gui_pymess_and_harpia_range_selectors(qapp):
    from pymorgan.gui.main_window import MainWindow

    win = MainWindow()
    win._ensure_calibration_initialized()
    assert hasattr(win, "cal_range_combo")
    assert hasattr(win, "cal_range_widget")

    # Verify 3-column layout: Left file loading | Center detector plots | Right fit settings inspector
    assert win.cal_main_layout.count() == 3
    assert win.cal_detector_tabWidget.count() == 3  # Detector 1, Detector 2, Pump Spectrum (no Fit Settings tab)
    assert hasattr(win, "cal_settings_scroll_area")
    assert win.cal_settings_scroll_area.widget() is win.cal_settings_tab
    assert hasattr(win, "cal_set_btn_refit_spec")
    assert hasattr(win, "cal_set_btn_refit_shaper")

    # 1. Select PyMESS
    idx_pymess = win.cal_type_combo.findText("PyMESS")
    assert idx_pymess >= 0
    win.cal_type_combo.setCurrentIndex(idx_pymess)
    assert not win.cal_range_widget.isHidden()
    assert win.cal_range_combo.count() == 3

    # Default PyMESS range is Mid-IR -> shaper group visible, CWL = 2000.0
    assert win.cal_range_combo.currentIndex() == 0
    assert not win.cal_shaper_group.isHidden()
    assert win.cal_set_cwl.value() == pytest.approx(2000.0)
    assert win.cal_set_ppnm.value() == pytest.approx(0.30)

    # Switch PyMESS to UV/Vis (TRUVIS-II pathway) -> shaper group hidden, CWL = 540.0, cross-corr on
    win.cal_range_combo.setCurrentIndex(1)
    assert win.cal_shaper_group.isHidden()
    assert win.cal_set_cwl.value() == pytest.approx(540.0)
    assert win.cal_set_ppnm.value() == pytest.approx(1.20)
    assert win.cal_set_cross_corr.isChecked() is True

    # Switch PyMESS to NIR (UniGE NIR-TA pathway) -> shaper group hidden, Prism model active, CWL = 1000.0
    win.cal_range_combo.setCurrentIndex(2)
    assert win.cal_shaper_group.isHidden()
    assert win.cal_set_cwl.value() == pytest.approx(1000.0)
    assert not win.cal_set_grating_model.isEnabled()

    # 2. Select Light Conversion HARPIA-TA (Int.)
    idx_harpia = win.cal_type_combo.findText("Light Conversion HARPIA-TA (Int.)")
    assert idx_harpia >= 0
    win.cal_type_combo.setCurrentIndex(idx_harpia)
    assert not win.cal_range_widget.isHidden()
    assert win.cal_range_combo.count() == 2

    # UV Range default
    assert win.cal_range_combo.currentIndex() == 0
    assert win.cal_set_cwl.value() == pytest.approx(430.0)
    assert win.cal_set_ppnm.value() == pytest.approx(1.23)
    assert win.cal_set_rel_min_nm.value() == pytest.approx(-40.0)
    assert win.cal_set_rel_max_nm.value() == pytest.approx(115.0)

    # Switch to Vis Range
    win.cal_range_combo.setCurrentIndex(1)
    assert win.cal_set_cwl.value() == pytest.approx(685.0)
    assert win.cal_set_ppnm.value() == pytest.approx(0.62)
    assert win.cal_set_rel_min_nm.value() == pytest.approx(-190.0)
    assert win.cal_set_rel_max_nm.value() == pytest.approx(190.0)


def _write_synthetic_nsta_cal_pair(tmp_path):
    """Create a synthetic 524-px UniGE nsTA blank + filter calibration file pair."""
    from pathlib import Path

    n_pix = 524
    pix = np.arange(n_pix, dtype=float)
    wl = 540.0 + (pix - 261.5) / 1.4
    ref_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "pymorgan"
        / "cal"
        / "ref_spectra"
        / "UVVis-Holmium.csv"
    )
    ref = load_reference_spectrum(ref_path)
    abs_ref = np.interp(wl, ref.spectral_axis, ref.absorbance)
    i_blank = 1000.0 + 150.0 * np.sin(np.linspace(0.0, np.pi, n_pix))
    i_filter = i_blank * (10.0 ** (-abs_ref))
    rms = np.ones(n_pix) * 0.1
    ref_col = np.ones(n_pix) * 12.0

    blank_path = tmp_path / "blank.dat"
    filter_path = tmp_path / "filter.dat"
    np.savetxt(
        blank_path,
        np.column_stack([pix, i_blank, rms, ref_col, rms]),
        delimiter="\t",
        header="pixel\tsignal\trms signal\treference\trms reference",
        comments="% ",
    )
    np.savetxt(
        filter_path,
        np.column_stack([pix, i_filter, rms, ref_col, rms]),
        delimiter="\t",
        header="pixel\tsignal\trms signal\treference\trms reference",
        comments="% ",
    )
    return blank_path, filter_path


def test_gui_unige_nsta_load_order_populates_raw_and_abs(qapp, tmp_path):
    from pymorgan.gui.main_window import MainWindow

    blank_path, filter_path = _write_synthetic_nsta_cal_pair(tmp_path)

    win = MainWindow()
    try:
        win._ensure_calibration_initialized()
        idx_nsta = win.cal_type_combo.findText("UniGE nsTA")
        assert idx_nsta >= 0
        win.cal_type_combo.setCurrentIndex(idx_nsta)
        assert win.cal_set_cross_corr.isChecked() is True

        # Load filter into _cal_svt_data first, then blank into _cal_exp_data
        win._cal_svt_data = load_experimental_spectrum(filter_path, cal_type_code=3)
        win._update_cal_raw_plots()
        win._update_cal_abs_plot()
        assert "Calculated Abs" not in {
            line.get_label() for line in win.cal_canvas_det1.ax_abs.lines
        }

        win._cal_exp_data = load_experimental_spectrum(blank_path, cal_type_code=3)
        win._update_cal_raw_plots()
        win._update_cal_abs_plot()

        raw_labels = {line.get_label() for line in win.cal_canvas_det1.ax_raw.lines}
        abs_labels = {line.get_label() for line in win.cal_canvas_det1.ax_abs.lines}
        assert {"Probe", "Solvent"} <= raw_labels
        assert "Calculated Abs" in abs_labels
    finally:
        win.close()


def test_cal_tab_left_panel_not_squashed_on_load(qapp):
    from pymorgan.gui.main_window import MainWindow

    win = MainWindow()
    win._ensure_calibration_initialized()
    expected_w = win.cal_scroll_area.minimumWidth()
    assert expected_w == win.cal_scroll_area.maximumWidth()
    assert expected_w >= 204
    if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
        assert expected_w < 290
    assert win.cal_main_layout.stretch(0) == 0
    assert win.cal_main_layout.stretch(1) == 1
    assert win.cal_main_layout.stretch(2) == 0

    win.move(-10000, -10000)
    win.show()
    win.MainTabs.setCurrentIndex(2)
    qapp.processEvents()

    try:
        assert win.cal_scroll_area.width() == expected_w
        win.cal_canvas_det1.show()
        qapp.processEvents()
        assert win.cal_scroll_area.width() == expected_w
    finally:
        win.close()


def test_cal_ref_plot_shared_yaxis_and_tight_layout(qapp, tmp_path):
    from pathlib import Path

    from pymorgan.gui.main_window import MainWindow

    blank_path, filter_path = _write_synthetic_nsta_cal_pair(tmp_path)

    win = MainWindow()
    win.move(-10000, -10000)
    win.show()
    win.MainTabs.setCurrentIndex(2)
    win._ensure_calibration_initialized()
    idx_nsta = win.cal_type_combo.findText("UniGE nsTA")
    win.cal_type_combo.setCurrentIndex(idx_nsta)

    win._cal_exp_data = load_experimental_spectrum(blank_path, cal_type_code=3)
    win._cal_svt_data = load_experimental_spectrum(filter_path, cal_type_code=3)
    win._update_cal_raw_plots()
    win._update_cal_abs_plot()
    ref_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "pymorgan"
        / "cal"
        / "ref_spectra"
        / "UVVis-Holmium.csv"
    )
    win._cal_ref_spec = load_reference_spectrum(ref_path)
    win._on_cal_do_fit()
    qapp.processEvents()

    try:
        canvas = win.cal_canvas_det1
        # Shared Y-axis: only the 4 subplots exist (no _ref_twin axis)
        assert len(canvas.fig.axes) == 4
        ref_lines = {line.get_label(): line for line in canvas.ax_ref.lines}
        assert {"Raw Ref", "Exp (BC)"} <= set(ref_lines)
        exp_line = ref_lines["Exp (BC)"]
        raw_ref_line = ref_lines["Raw Ref"]
        assert exp_line.get_linewidth() >= 2.4
        assert exp_line.get_linewidth() > raw_ref_line.get_linewidth()
        assert exp_line.get_zorder() > raw_ref_line.get_zorder()
        assert str(exp_line.get_color()).lower() in ("#0066cc", "#58baff")
        p_raw = canvas.ax_raw.get_position()
        p_abs = canvas.ax_abs.get_position()
        p_ref = canvas.ax_ref.get_position()
        assert p_raw.height > 0.35
        assert p_abs.y1 < p_raw.y0
        assert p_raw.x1 < p_ref.x0
    finally:
        win.close()


def test_baseline_estimators_synthetic_and_scale_invariance():
    from pymorgan.cal import (
        baseline_arpls,
        baseline_asls,
        baseline_modpoly,
        baseline_snip,
        estimate_baseline,
    )

    for n in (64, 1000):
        x = np.linspace(0.0, 1.0, n)
        true_bl = 0.12 + 0.25 * x + 0.08 * np.sin(np.pi * x)
        peaks = (
            0.85 * np.exp(-0.5 * ((x - 0.22) / 0.03) ** 2)
            + 1.15 * np.exp(-0.5 * ((x - 0.55) / 0.035) ** 2)
            + 0.65 * np.exp(-0.5 * ((x - 0.80) / 0.025) ** 2)
        )
        y = true_bl + peaks

        for fn in (
            lambda arr: baseline_arpls(arr, smoothness=0.08),
            lambda arr: baseline_asls(arr, smoothness=0.08),
            lambda arr: baseline_snip(arr, smoothness=0.10),
            lambda arr: baseline_modpoly(arr, poly_order=3),
        ):
            est = fn(y)
            assert est.shape == y.shape
            rmse = float(np.sqrt(np.mean((est - true_bl) ** 2)))
            assert rmse < 0.032

        np.testing.assert_array_equal(estimate_baseline(y, method="legacy_poly"), np.zeros(n))
        np.testing.assert_array_equal(estimate_baseline(y, method="none"), np.zeros(n))

    with pytest.raises(ValueError, match="Unknown baseline method"):
        estimate_baseline(np.ones(20), method="not_a_real_method")


def test_varpro_global_fit_synthetic_ground_truth():
    # High-resolution reference with 5 sharp peaks and rising wing baseline
    ref_x = np.linspace(380.0, 720.0, 1400)
    ref_bl = 0.05 + 0.0012 * (ref_x - 400.0) + 0.35 * np.exp(-((ref_x - 375.0) / 14.0) ** 2)
    peaks = (
        0.9 * np.exp(-0.5 * ((ref_x - 435.0) / 3.0) ** 2)
        + 1.2 * np.exp(-0.5 * ((ref_x - 485.0) / 3.5) ** 2)
        + 0.7 * np.exp(-0.5 * ((ref_x - 535.0) / 3.0) ** 2)
        + 1.0 * np.exp(-0.5 * ((ref_x - 590.0) / 4.0) ** 2)
        + 0.8 * np.exp(-0.5 * ((ref_x - 640.0) / 3.5) ** 2)
    )
    ref_y = ref_bl + peaks

    # Quadratic grating ground truth over 96 pixels centered near 535 nm
    n_pix = 96
    pix = np.arange(1, n_pix + 1, dtype=float)
    c_pix = (1.0 + n_pix) / 2.0
    wl0_true, ppnm_true, c2_true = 537.5, 0.42, 0.35
    pix_true = (ref_x - wl0_true) * ppnm_true + 1e-4 * c2_true * ((ref_x - wl0_true) ** 2) + c_pix
    true_wl = np.interp(pix, pix_true, ref_x)

    # Sample spectrum with strong curved + ramped baseline and low edge SNR
    rng = np.random.default_rng(42)
    from scipy.ndimage import gaussian_filter1d

    ref_conv = gaussian_filter1d(peaks, sigma=2.5)
    meas_clean = np.interp(true_wl, ref_x, ref_conv) * 0.65
    u = (pix - c_pix) / (0.5 * n_pix)
    meas_bl = 0.18 + 0.32 * u + 0.12 * (u**2)
    probe_i0 = 100.0 * np.exp(-1.5 * (u**2)) + 5.0
    noise = rng.normal(0.0, 0.002 / np.sqrt(probe_i0 / 100.0))
    meas_y = meas_clean + meas_bl + noise

    res_global = fit_wavelength_axis(
        meas_y,
        ref_x,
        ref_y,
        cal_type_code=2,
        cwl=530.0,
        ppnm_guess=0.40,
        rel_min_nm=-130.0,
        rel_max_nm=150.0,
        grating_degree=2,
        use_derivative="global",
        baseline_method="arpls",
        baseline_target="both",
        convolve_ref=True,
        probe_weights=probe_i0,
    )
    rmse_global = float(np.sqrt(np.mean((res_global.wavelength_nm - true_wl) ** 2)))
    assert rmse_global < 0.12
    assert res_global.sample_baseline_px is not None
    assert len(res_global.sample_baseline_px) == n_pix
    assert res_global.exp_raw_y_fit is not None

    # Verify all target and derivative variants execute cleanly
    for target in ("none", "sample", "standard", "both"):
        r_t = fit_wavelength_axis(
            meas_y,
            ref_x,
            ref_y,
            cal_type_code=2,
            cwl=530.0,
            ppnm_guess=0.40,
            rel_min_nm=-130.0,
            rel_max_nm=150.0,
            grating_degree=2,
            use_derivative="global",
            baseline_method="arpls",
            baseline_target=target,
            convolve_ref=False,
        )
        assert len(r_t.wavelength_nm) == n_pix

    for mode in ("none", "1st", "2nd", "1st+2nd"):
        r_m = fit_wavelength_axis(
            meas_y,
            ref_x,
            ref_y,
            cal_type_code=2,
            cwl=530.0,
            ppnm_guess=0.40,
            rel_min_nm=-130.0,
            rel_max_nm=150.0,
            grating_degree=2,
            use_derivative=mode,
            baseline_method="asls",
            baseline_target="both",
            convolve_ref=False,
        )
        assert len(r_m.wavelength_nm) == n_pix


def test_gui_calibration_baseline_and_global_controls(qapp):
    from pymorgan.gui.main_window import MainWindow

    win = MainWindow()
    win._ensure_calibration_initialized()
    try:
        assert win.cal_set_bl_method.count() == 5
        assert win.cal_set_bl_method.currentIndex() == 0
        assert win.cal_set_bl_target.count() == 4
        assert win.cal_set_bl_target.currentIndex() == 0
        assert win.cal_set_bl_smooth.value() == pytest.approx(0.20)
        assert win.cal_set_snr_weights.isChecked() is True
        assert win.cal_set_deriv_mode.count() == 4
        assert win.cal_set_deriv_mode.currentIndex() == 3
        assert "Global" in win.cal_set_deriv_mode.currentText()
        assert win.cal_set_bl_smooth.isEnabled() is True

        # Selecting Legacy Polynomial disables target and smoothness sub-controls
        win.cal_set_bl_method.setCurrentIndex(4)
        assert win.cal_set_bl_target.isEnabled() is False
        assert win.cal_set_bl_smooth.isEnabled() is False

        # Switching back to arPLS restores controls
        win.cal_set_bl_method.setCurrentIndex(0)
        assert win.cal_set_bl_target.isEnabled() is True
        assert win.cal_set_bl_smooth.isEnabled() is True

        # Setting Apply To = None disables smoothness spinbox
        win.cal_set_bl_target.setCurrentIndex(3)
        assert win.cal_set_bl_smooth.isEnabled() is False

        # Restore defaults and verify shaper fit forwards new baseline/global controls
        win.cal_set_bl_target.setCurrentIndex(0)
        idx_trir = win.cal_type_combo.findText("UniGE TRIR (Intensity)")
        win.cal_type_combo.setCurrentIndex(idx_trir)
        from pathlib import Path

        ref_dir = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "pymorgan"
            / "cal"
            / "ref_spectra"
        )
        ref_diox = load_reference_spectrum(ref_dir / "FTIR-DioxaneRAL_2cm-1.csv")
        wn = np.linspace(2022.0, 2208.0, 64)
        abs_diox = np.interp(wn, ref_diox.spectral_axis, ref_diox.absorbance)
        pu_air_y = 1000.0 * np.exp(-0.5 * ((wn - 2115.0) / 80.0) ** 2)
        pu_diox_y = pu_air_y * (10.0 ** (-abs_diox))
        win._cal_shaper_pump_air_y = pu_air_y
        win._cal_shaper_pump_solvent_y = pu_diox_y
        win.cal_set_cwl.setValue(2115.0)
        win.cal_set_ppnm.setValue(63.0 / 186.0)
        win._on_cal_shaper_do_fit()
        shaper_res = win._cal_shaper_fit_res
        assert shaper_res is not None
        assert shaper_res.baseline_method == "arpls"
        assert shaper_res.baseline_target == "both"
        assert shaper_res.sample_baseline_px is not None
        assert 2015.0 < shaper_res.wavenumber_cm1[0] < 2030.0
        assert 2200.0 < shaper_res.wavenumber_cm1[-1] < 2215.0
    finally:
        win.close()


def test_truvis_ii_default_cwl_540_cross_corr_and_uncalibrated_meta(tmp_path, qapp):
    from unittest.mock import patch

    from pymorgan.cal import get_default_calibration_params
    from pymorgan.gui.main_window import MainWindow

    p_def = get_default_calibration_params(10)
    assert p_def["cwl"] == pytest.approx(540.0)
    assert p_def["use_cross_corr"] is True

    # Synthetic uncalibrated 512-px TRUVIS-II folder with 0..511 pixel-index axis in _meta.txt
    scan_dir = tmp_path / "probe_air_120000"
    scan_dir.mkdir()
    spec_file = scan_dir / "probe_air_120000_sp0_ds0_intensity_0.csv"
    np.savetxt(spec_file, np.linspace(1000.0, 2000.0, 512), fmt="%.4f")
    pix_axis = ", ".join(f"{i:.6f}" for i in range(512))
    (scan_dir / "probe_air_120000_meta.txt").write_text(
        f"-- SPECTROMETER --\nWavelength axis: [{pix_axis}]\nCenter wavelength: 0.000000\nGrating: 150 l/mm\n",
        encoding="utf-8",
    )

    exp = load_experimental_spectrum(spec_file, cal_type_code=10)
    assert exp.cwl[0] == 0.0
    assert exp.wavelengths is None

    # Synthetic uncalibrated 512-px PyMESS UV/Vis .h5 file where DAQ wrote CWL = 256 (N // 2 pixels)
    import h5py

    uv_bad_h5 = tmp_path / "Probe_air_truvis_uncal.h5"
    with h5py.File(uv_bad_h5, "w") as f:
        d = f.create_group("data")
        d.create_dataset("probe_spectrum", data=np.linspace(1.0, 10.0, 512))
        d.create_dataset("wavelengths", data=np.arange(512, dtype=float))
        m = f.create_group("metadata")
        m.attrs["setup_id"] = "TRUVIS_UniGE"
        m.attrs["spectrometer_cwl_nm"] = "256.0"

    exp_h5 = load_experimental_spectrum(uv_bad_h5, cal_type_code=12)
    assert exp_h5.spectral_range == "uv-vis"
    assert exp_h5.cwl[0] == 0.0
    assert exp_h5.wavelengths is None

    win = MainWindow()
    win._ensure_calibration_initialized()
    try:
        idx_truvis = win.cal_type_combo.findText("UniGE TRUVIS-II (Intensity)")
        assert idx_truvis >= 0
        win.cal_type_combo.setCurrentIndex(idx_truvis)
        assert win.cal_set_cwl.value() == pytest.approx(540.0)
        assert win.cal_set_cross_corr.isChecked() is True

        with patch(
            "pymorgan.gui.tabs.calibration.QFileDialog.getOpenFileName",
            return_value=(str(spec_file), ""),
        ):
            win._on_cal_load_probe()

        assert win.cal_set_cwl.value() == pytest.approx(540.0)

        with patch(
            "pymorgan.gui.tabs.calibration.QFileDialog.getOpenFileName",
            return_value=(str(uv_bad_h5), ""),
        ):
            win._on_cal_load_probe()

        assert win.cal_set_cwl.value() == pytest.approx(540.0)
    finally:
        win.close()



