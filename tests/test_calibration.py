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


