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


def test_load_harpia_ta_real_testdata():
    from pathlib import Path
    env_dir = os.environ.get("PYMORGAN_TESTDATA_DIR")
    if not env_dir:
        pytest.skip("PYMORGAN_TESTDATA_DIR environment variable not set")
    real_file = Path(env_dir) / "HARPIA" / "probe spectra for wl calibration" / "260629_newWL_0ns_1_average (number of measured spectra 250).txt"
    if not real_file.is_file():
        pytest.skip("HARPIA test data file not present")

    exp = load_experimental_spectrum(real_file, cal_type_code=11)
    assert len(exp.detector_data) == 1
    assert len(exp.detector_data[0]) == 256
    assert exp.wavelengths is not None
    assert len(exp.wavelengths) == 256
    assert exp.min_wl == pytest.approx(326.373077392578)
    assert exp.max_wl == pytest.approx(533.762634277344)


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


def test_pymess_real_20261022_calibration_testdata():
    from pathlib import Path

    from pymorgan.cal import (
        fit_spectrum_gaussian,
        fit_wavelength_axis,
        load_PyMESS_calibration_spectrum,
        load_reference_spectrum,
    )

    folder = Path(r"C:\Users\ricar\switchdrive\Ambizione UniGE\Scripts\testData\PyMESS\20261022\Calibration_Shaper_passiv_PyMess")
    if not folder.is_dir():
        pytest.skip("PyMESS 20261022 calibration testData folder not present")

    probe_air = load_PyMESS_calibration_spectrum(folder / "Probe_air_110859.h5")
    probe_diox = load_PyMESS_calibration_spectrum(folder / "Probe_diox_110927.h5")
    pump_air = load_PyMESS_calibration_spectrum(folder / "Pump_air_110610.h5")
    pump_diox = load_PyMESS_calibration_spectrum(folder / "Pump_diox_110655.h5")
    multimask = load_PyMESS_calibration_spectrum(folder / "Multimask_111613.h5")
    singlemask = load_PyMESS_calibration_spectrum(folder / "Singlemask_111631.h5")

    for ds in (probe_air, probe_diox, pump_air, pump_diox, multimask, singlemask):
        assert ds.spectral_range == "mid-ir"
        assert len(ds.detector_data[0]) == 64
        assert ds.cwl[0] == pytest.approx(2079.994)
        assert ds.gratings[0] == pytest.approx(150.0)

    ref_dir = Path(__file__).resolve().parent.parent / "src" / "pymorgan" / "cal" / "ref_spectra"
    ref = load_reference_spectrum(ref_dir / "FTIR-DioxaneRAL_2cm-1.csv")

    meas_probe = -np.log10(probe_diox.detector_data[0] / probe_air.detector_data[0])
    res_probe = fit_wavelength_axis(
        meas_probe,
        ref.spectral_axis,
        ref.absorbance,
        cal_type_code=12,
        cwl=probe_air.cwl[0],
        ppnm_guess=probe_air.gratings[0] / 500.0,
        spectral_range=probe_air.spectral_range,
    )
    assert len(res_probe.wavenumber_cm1) == 64
    assert 1980.0 < res_probe.wavenumber_cm1[0] < 2015.0
    assert 2170.0 < res_probe.wavenumber_cm1[-1] < 2205.0

    meas_pump = np.abs(-np.log10(pump_diox.detector_data[0] / pump_air.detector_data[0]))
    res_pump = fit_wavelength_axis(
        meas_pump,
        ref.spectral_axis,
        ref.absorbance,
        cal_type_code=12,
        cwl=pump_air.cwl[0],
        ppnm_guess=pump_air.gratings[0] / 500.0,
        spectral_range=pump_air.spectral_range,
    )
    gauss = fit_spectrum_gaussian(pump_air.detector_data[0], res_pump.wavenumber_cm1)
    assert 2000.0 < gauss.w0 < 2160.0
    assert gauss.fwhm > 20.0


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

    # Switch PyMESS to UV/Vis (TRUVIS-II pathway) -> shaper group hidden, CWL = 530.0
    win.cal_range_combo.setCurrentIndex(1)
    assert win.cal_shaper_group.isHidden()
    assert win.cal_set_cwl.value() == pytest.approx(530.0)
    assert win.cal_set_ppnm.value() == pytest.approx(1.20)

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


def test_unige_nsta_real_testdata_load_and_fit():
    from pathlib import Path

    from pymorgan.cal import get_default_calibration_params

    base = Path(r"C:\Users\ricar\switchdrive\Ambizione UniGE\Scripts\testData\UniGE_nsTA")
    folders = [
        (base / "nsTA" / "Raw" / "18", 113, 235),
        (base / "cal", 120, 242),
    ]
    if not all(f.is_dir() for f, _, _ in folders):
        pytest.skip("UniGE nsTA testData directories not present")

    ref_path = Path(__file__).resolve().parent.parent / "src" / "pymorgan" / "cal" / "ref_spectra" / "UVVis-Holmium.csv"
    ref = load_reference_spectrum(ref_path)
    p_def = get_default_calibration_params(3)
    assert p_def["use_cross_corr"] is True

    for folder, pk_445, pk_536 in folders:
        wl = load_experimental_spectrum(folder / "WL.dat", cal_type_code=3)
        ho = load_experimental_spectrum(folder / "HOLMIUM.dat", cal_type_code=3)

        assert len(wl.detector_data) == 1
        assert len(ho.detector_data) == 1
        i_wl = wl.detector_data[0]
        i_ho = ho.detector_data[0]
        assert len(i_wl) == 524
        assert len(i_ho) == 524
        assert np.all(np.isfinite(i_wl)) and np.all(i_wl > 0)
        assert np.all(np.isfinite(i_ho)) and np.all(i_ho > 0)

        meas_y = -np.log10(i_ho / i_wl)
        res = fit_wavelength_axis(
            meas_y,
            ref.spectral_axis,
            ref.absorbance,
            cal_type_code=3,
            cwl=p_def["cwl"],
            ppnm_guess=p_def["ppnm_guess"],
            rel_min_nm=p_def["rel_min_nm"],
            rel_max_nm=p_def["rel_max_nm"],
            grating_degree=p_def["grating_degree"],
            use_derivative=p_def["use_derivative"],
            use_cross_corr=p_def["use_cross_corr"],
        )
        assert len(res.wavelength_nm) == 524
        assert res.wavelength_nm[pk_445] == pytest.approx(445.2, abs=1.5)
        assert res.wavelength_nm[pk_536] == pytest.approx(536.3, abs=2.0)


def test_gui_unige_nsta_load_order_populates_raw_and_abs(qapp):
    from pathlib import Path

    from pymorgan.gui.main_window import MainWindow

    folder = Path(r"C:\Users\ricar\switchdrive\Ambizione UniGE\Scripts\testData\UniGE_nsTA\nsTA\Raw\18")
    if not folder.is_dir():
        pytest.skip("UniGE nsTA Raw/18 testData directory not present")

    win = MainWindow()
    win._ensure_calibration_initialized()
    idx_nsta = win.cal_type_combo.findText("UniGE nsTA")
    assert idx_nsta >= 0
    win.cal_type_combo.setCurrentIndex(idx_nsta)
    assert win.cal_set_cross_corr.isChecked() is True

    # Load filter (HOLMIUM.dat) into _cal_svt_data first, then blank (WL.dat) into _cal_exp_data
    win._cal_svt_data = load_experimental_spectrum(folder / "HOLMIUM.dat", cal_type_code=3)
    win._update_cal_raw_plots()
    win._update_cal_abs_plot()
    assert "Calculated Abs" not in {line.get_label() for line in win.cal_canvas_det1.ax_abs.lines}

    win._cal_exp_data = load_experimental_spectrum(folder / "WL.dat", cal_type_code=3)
    win._update_cal_raw_plots()
    win._update_cal_abs_plot()

    raw_labels = {line.get_label() for line in win.cal_canvas_det1.ax_raw.lines}
    abs_labels = {line.get_label() for line in win.cal_canvas_det1.ax_abs.lines}
    assert {"Probe", "Solvent"} <= raw_labels
    assert "Calculated Abs" in abs_labels


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


def test_cal_ref_plot_shared_yaxis_and_tight_layout(qapp):
    from pathlib import Path

    from pymorgan.gui.main_window import MainWindow

    folder = Path(r"C:\Users\ricar\switchdrive\Ambizione UniGE\Scripts\testData\UniGE_nsTA\nsTA\Raw\18")
    if not folder.is_dir():
        pytest.skip("UniGE nsTA Raw/18 testData directory not present")

    win = MainWindow()
    win.move(-10000, -10000)
    win.show()
    win.MainTabs.setCurrentIndex(2)
    win._ensure_calibration_initialized()
    idx_nsta = win.cal_type_combo.findText("UniGE nsTA")
    win.cal_type_combo.setCurrentIndex(idx_nsta)

    win._cal_exp_data = load_experimental_spectrum(folder / "WL.dat", cal_type_code=3)
    win._cal_svt_data = load_experimental_spectrum(folder / "HOLMIUM.dat", cal_type_code=3)
    win._update_cal_raw_plots()
    win._update_cal_abs_plot()
    ref_path = Path(__file__).resolve().parent.parent / "src" / "pymorgan" / "cal" / "ref_spectra" / "UVVis-Holmium.csv"
    win._cal_ref_spec = load_reference_spectrum(ref_path)
    win._on_cal_do_fit()
    qapp.processEvents()

    try:
        canvas = win.cal_canvas_det1
        # Shared Y-axis: only the 4 subplots exist (no _ref_twin axis)
        assert len(canvas.fig.axes) == 4
        ref_labels = {line.get_label() for line in canvas.ax_ref.lines}
        assert {"Raw Ref", "Exp (BC)"} <= ref_labels
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


def test_real_nsta_and_trir_improved_calibration_accuracy():
    from pathlib import Path

    from scipy.ndimage import gaussian_filter1d

    from pymorgan.cal import estimate_baseline, get_default_calibration_params

    nsta_dir = Path(r"C:\Users\ricar\switchdrive\Ambizione UniGE\Scripts\testData\UniGE_nsTA\nsTA\Raw\18")
    trir_dir = Path(r"C:\Users\ricar\switchdrive\Ambizione UniGE\Scripts\testData\Calibration_Data\TRIR+shaper")
    if not nsta_dir.is_dir() or not trir_dir.is_dir():
        pytest.skip("nsTA Raw/18 or TRIR+shaper testData directories not present")

    ref_dir = Path(__file__).resolve().parent.parent / "src" / "pymorgan" / "cal" / "ref_spectra"
    ref_ho = load_reference_spectrum(ref_dir / "UVVis-Holmium.csv")
    ref_diox = load_reference_spectrum(ref_dir / "FTIR-DioxaneRAL_2cm-1.csv")

    def _subpixel_peak(x: np.ndarray, y: np.ndarray, target_x: float, win_dx: float) -> float:
        m = (x >= target_x - win_dx) & (x <= target_x + win_dx)
        xs, ys = x[m], y[m]
        idx = int(np.argmax(ys))
        if 0 < idx < len(ys) - 1:
            c = np.polyfit(xs[idx - 1 : idx + 2], ys[idx - 1 : idx + 2], 2)
            if c[0] < 0:
                xv = -c[1] / (2.0 * c[0])
                if xs[0] <= xv <= xs[-1]:
                    return float(xv)
        return float(xs[idx])

    # 1. UniGE nsTA Raw/18 (8 Holmium Oxide bands 385.6-637.2 nm)
    wl_i = load_experimental_spectrum(nsta_dir / "WL.dat", 3).detector_data[0]
    ho_i = load_experimental_spectrum(nsta_dir / "HOLMIUM.dat", 3).detector_data[0]
    abs_nsta = -np.log10(ho_i / wl_i)

    ux_ho = np.arange(300.0, 780.0, 0.02)
    uy_ho = np.interp(ux_ho, ref_ho.spectral_axis, ref_ho.absorbance)
    uy_ho_conv = gaussian_filter1d(uy_ho - estimate_baseline(uy_ho, "arpls", 0.15), sigma=1.0 / 0.02)
    ho_ref_pks = np.array([
        _subpixel_peak(ux_ho, uy_ho_conv, t, 2.0)
        for t in (385.6, 418.4, 445.65, 453.15, 459.85, 485.3, 536.4, 637.2)
    ])
    pix_nsta = np.arange(1, len(abs_nsta) + 1, dtype=float)
    ho_exp_pix = np.array([
        _subpixel_peak(pix_nsta, abs_nsta - estimate_baseline(abs_nsta, "arpls", 0.15), p, 4.0)
        for p in (34.4, 77.7, 114.1, 124.1, 133.0, 165.8, 235.9, 373.1)
    ])

    p_nsta = get_default_calibration_params(3)
    res_nsta_new = fit_wavelength_axis(
        abs_nsta,
        ref_ho.spectral_axis,
        ref_ho.absorbance,
        cal_type_code=3,
        cwl=p_nsta["cwl"],
        ppnm_guess=p_nsta["ppnm_guess"],
        rel_min_nm=p_nsta["rel_min_nm"],
        rel_max_nm=p_nsta["rel_max_nm"],
        grating_degree=p_nsta["grating_degree"],
        use_cross_corr=p_nsta["use_cross_corr"],
        use_derivative=p_nsta["use_derivative"],
        convolve_ref=p_nsta["convolve_ref"],
        baseline_method=p_nsta["baseline_method"],
        baseline_target=p_nsta["baseline_target"],
        baseline_smoothness=p_nsta["baseline_smoothness"],
        probe_weights=wl_i,
    )
    res_nsta_leg = fit_wavelength_axis(
        abs_nsta,
        ref_ho.spectral_axis,
        ref_ho.absorbance,
        cal_type_code=3,
        cwl=p_nsta["cwl"],
        ppnm_guess=p_nsta["ppnm_guess"],
        rel_min_nm=p_nsta["rel_min_nm"],
        rel_max_nm=p_nsta["rel_max_nm"],
        grating_degree=2,
        use_cross_corr=True,
        use_derivative="1st",
        convolve_ref=False,
        baseline_method="legacy_poly",
        baseline_target="none",
    )
    rmse_nsta_new = float(np.sqrt(np.mean((np.interp(ho_exp_pix, pix_nsta, res_nsta_new.wavelength_nm) - ho_ref_pks) ** 2)))
    rmse_nsta_leg = float(np.sqrt(np.mean((np.interp(ho_exp_pix, pix_nsta, res_nsta_leg.wavelength_nm) - ho_ref_pks) ** 2)))
    assert rmse_nsta_new < 0.20
    assert rmse_nsta_new < rmse_nsta_leg

    # 2. TRIR+shaper Probe and Shaper Pump (6 Dioxane bands 2071.9-2191.8 cm-1)
    pr_air = load_experimental_spectrum(trir_dir / "probe_air_172426" / "probe_air_172426_sp0_ds0_intensity_0.csv", 8)
    pr_diox = load_experimental_spectrum(trir_dir / "probe_diox_172445" / "probe_diox_172445_sp0_ds0_intensity_0.csv", 8)
    pu_air = load_experimental_spectrum(trir_dir / "pump_air_171745" / "pump_air_171745_sp0_ds0_intensity_0.csv", 8)
    pu_diox = load_experimental_spectrum(trir_dir / "pump_diox_171859" / "pump_diox_171859_sp0_ds0_intensity_0.csv", 8)

    abs_pr = -np.log10(pr_diox.detector_data[0] / pr_air.detector_data[0])
    abs_pu = -np.log10(pu_diox.detector_data[0] / pu_air.detector_data[0])

    ux_ir = np.arange(1990.0, 2220.0, 0.02)
    uy_ir = np.interp(ux_ir, ref_diox.spectral_axis, ref_diox.absorbance)
    uy_ir_conv = gaussian_filter1d(uy_ir - estimate_baseline(uy_ir, "arpls", 0.20), sigma=1.5 / 0.02)
    ir_ref_pks = np.array([
        _subpixel_peak(ux_ir, uy_ir_conv, t, 2.5)
        for t in (2071.9, 2087.8, 2104.7, 2130.2, 2175.7, 2191.8)
    ])
    pix_ir = np.arange(1, 65, dtype=float)
    pr_exp_pix = np.array([
        _subpixel_peak(pix_ir, abs_pr - estimate_baseline(abs_pr, "arpls", 0.22), p, 2.0)
        for p in (20.3, 25.7, 30.8, 39.6, 53.5, 58.2)
    ])
    pu_exp_pix = np.array([
        _subpixel_peak(pix_ir, abs_pu - estimate_baseline(abs_pu, "arpls", 0.22), p, 2.0)
        for p in (19.5, 24.9, 30.5, 39.2, 52.9, 57.6)
    ])

    res_pr_new = fit_wavelength_axis(
        abs_pr,
        ref_diox.spectral_axis,
        ref_diox.absorbance,
        cal_type_code=8,
        cwl=pr_air.cwl[0],
        ppnm_guess=pr_air.gratings[0] / 500.0,
        rel_min_nm=-200.0,
        rel_max_nm=200.0,
        grating_degree=2,
        use_derivative="global",
        baseline_method="arpls",
        baseline_target="both",
        convolve_ref=True,
        probe_weights=pr_air.detector_data[0],
    )
    res_pr_leg = fit_wavelength_axis(
        abs_pr,
        ref_diox.spectral_axis,
        ref_diox.absorbance,
        cal_type_code=8,
        cwl=pr_air.cwl[0],
        ppnm_guess=pr_air.gratings[0] / 500.0,
        rel_min_nm=-200.0,
        rel_max_nm=200.0,
        grating_degree=2,
        use_derivative="1st",
        baseline_method="legacy_poly",
        baseline_target="none",
        convolve_ref=False,
    )
    rmse_pr_new = float(np.sqrt(np.mean((np.interp(pr_exp_pix, pix_ir, res_pr_new.wavenumber_cm1) - ir_ref_pks) ** 2)))
    rmse_pr_leg = float(np.sqrt(np.mean((np.interp(pr_exp_pix, pix_ir, res_pr_leg.wavenumber_cm1) - ir_ref_pks) ** 2)))
    assert rmse_pr_new < 0.80
    assert rmse_pr_leg > 3.5
    assert rmse_pr_new < 0.25 * rmse_pr_leg

    res_pu_new = fit_wavelength_axis(
        abs_pu,
        ref_diox.spectral_axis,
        ref_diox.absorbance,
        cal_type_code=8,
        cwl=pu_air.cwl[0],
        ppnm_guess=pu_air.gratings[0] / 500.0,
        rel_min_nm=-200.0,
        rel_max_nm=200.0,
        grating_degree=2,
        use_derivative="global",
        baseline_method="arpls",
        baseline_target="both",
        convolve_ref=True,
        probe_weights=pu_air.detector_data[0],
    )
    rmse_pu_new = float(np.sqrt(np.mean((np.interp(pu_exp_pix, pix_ir, res_pu_new.wavenumber_cm1) - ir_ref_pks) ** 2)))
    assert rmse_pu_new < 0.55


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
        trir_dir = Path(r"C:\Users\ricar\switchdrive\Ambizione UniGE\Scripts\testData\Calibration_Data\TRIR+shaper")
        if trir_dir.is_dir():
            pu_air = load_experimental_spectrum(trir_dir / "pump_air_171745" / "pump_air_171745_sp0_ds0_intensity_0.csv", 8)
            pu_diox = load_experimental_spectrum(trir_dir / "pump_diox_171859" / "pump_diox_171859_sp0_ds0_intensity_0.csv", 8)
            win._cal_shaper_pump_air_y = pu_air.detector_data[0]
            win._cal_shaper_pump_solvent_y = pu_diox.detector_data[0]
            win.cal_set_cwl.setValue(pu_air.cwl[0])
            win.cal_set_ppnm.setValue(pu_air.gratings[0] / 500.0)
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



