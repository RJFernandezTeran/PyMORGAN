"""Unit tests for spectrometer calibration math models, optimisation backend, load utilities, and GUI controls."""

import os
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from pymorgan.cal import (
    fit_probe_spectrum,
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

