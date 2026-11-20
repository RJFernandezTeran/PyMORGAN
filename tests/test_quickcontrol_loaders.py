"""Unit and GUI integration tests for PhaseTech QuickControl 1D and 2D loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import synthetic
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

import pymorgan as pm
from pymorgan.gui.main_window import MainWindow
from pymorgan.gui.mw_common import DATA_TYPE_DISPLAY_NAMES, set_combo_datatype
from pymorgan.oneD.load import (
    classify_quickcontrol_info,
    describe_dataset,
    is_dataset_file,
    is_dataset_file_status,
)
from pymorgan.twoD.load import (
    is_map_dataset_file,
    is_map_dataset_file_status,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# --------------------------------------------------------------------------- #
#                       QuickControl 1D (QuickControl_PP)                     #
# --------------------------------------------------------------------------- #
def test_quickcontrol_1d_polarised_magic_angle_default(tmp_path: Path):
    """Verify polarised 1D dataset loads with Magic Angle first, followed by Parallel and Perpendicular."""
    meta = synthetic.make_synthetic_quickcontrol_1d(
        tmp_path,
        prefix="ir_sample#01",
        n_delays=14,
        n_pixels=20,
        n_scans=3,
        polarised=True,
    )

    kind, has_scans = classify_quickcontrol_info(meta["info_path"])
    assert kind == "1d"
    assert has_scans is True
    assert is_dataset_file("QuickControl_PP", meta["info_path"]) is True
    assert is_map_dataset_file("QuickControl_2D", meta["info_path"]) is False

    desc = describe_dataset("QuickControl_PP", meta["info_path"])
    assert desc is not None
    assert desc["state_names"] == ["Magic Angle", "Parallel", "Perpendicular"]

    ds = pm.load_1D(meta["info_path"], data_type="QuickControl_PP")
    assert ds.n_detectors == 3
    assert ds.Units["detectors"] == ["Magic Angle", "Parallel", "Perpendicular"]
    assert ds.nscans == 3
    assert ds.Zavg_R.shape == (14, 20, 3)
    assert ds.Zss_R.shape == (14, 20, 3, 3)
    assert ds.Zstdv.shape == (14, 20, 3)
    assert ds.has_noise is True

    np.testing.assert_allclose(ds.delays, meta["delays_ps"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(ds.probe, meta["probe"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ds.Zavg_R[:, :, 0], meta["avg_ma_mOD"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ds.Zavg_R[:, :, 1], meta["avg_par_mOD"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ds.Zavg_R[:, :, 2], meta["avg_per_mOD"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ds.Units["probe_spectrum"], meta["spec"], rtol=1e-5)
    assert ds.counts is None

    # Verify anisotropy calculation auto-detects Parallel (1) and Perpendicular (2)
    comps = ds.calculate_anisotropy(return_components=True, detector_par=1, detector_perp=2)
    np.testing.assert_allclose(
        comps["magic_angle"].Zavg_R[:, :, 0], meta["avg_ma_mOD"], rtol=1e-5, atol=1e-6
    )


def test_quickcontrol_1d_unpolarised_uvvis_and_path_resolution(tmp_path: Path):
    """Verify unpolarised UV-Vis 1D dataset, linear-grating fallback, and flexible path inputs."""
    sub = tmp_path / "single_run"
    meta = synthetic.make_synthetic_quickcontrol_1d(
        sub,
        prefix="vis_run#07",
        n_delays=10,
        n_pixels=16,
        n_scans=2,
        polarised=False,
        cwl_nm=520.0,
        advanced_cal=False,
    )

    # Load from bare prefix, sibling .scan file, and parent directory
    bare_prefix = sub / "vis_run#07"
    sibling_scan = sub / "vis_run#07#001.scan"

    ds_info = pm.load_1D(meta["info_path"], data_type="QuickControl_PP")
    ds_bare = pm.load_1D(bare_prefix, data_type="QuickControl_PP")
    ds_scan = pm.load_1D(sibling_scan, data_type="QuickControl_PP")
    ds_dir = pm.load_1D(sub, data_type="QuickControl_PP", load_single_scans=False)

    assert ds_info.n_detectors == 1
    assert ds_info.Units["detectors"] == ["Signal"]
    assert ds_info.Units["unitsL_ltx"] == "nm"
    assert "linear grating" in ds_info.calibration_status()
    np.testing.assert_allclose(ds_info.Zavg_R, ds_bare.Zavg_R)
    np.testing.assert_allclose(ds_info.Zavg_R, ds_scan.Zavg_R)
    np.testing.assert_allclose(ds_info.Zavg_R, ds_dir.Zavg_R)
    assert np.all(np.isnan(ds_dir.Zss_R))


def test_quickcontrol_1d_calibrated_probe_override_and_empty_candidate(tmp_path: Path):
    """Verify CalibratedProbe.csv override and zero-scan candidate status."""
    meta = synthetic.make_synthetic_quickcontrol_1d(
        tmp_path, prefix="cal_run#01", n_delays=8, n_pixels=12, n_scans=1
    )
    custom_wn = np.linspace(1950.0, 2060.0, 12)
    np.savetxt(tmp_path / "CalibratedProbe.csv", custom_wn, delimiter=",")

    ds = pm.load_1D(meta["info_path"], data_type="QuickControl_PP")
    np.testing.assert_allclose(ds.probe, custom_wn)
    assert "CalibratedProbe.csv" in ds.calibration_status()

    empty = synthetic.make_synthetic_quickcontrol_1d(
        tmp_path, prefix="pending_run#02", write_scans=False
    )
    assert is_dataset_file_status("QuickControl_PP", empty["info_path"]) == (True, False)
    assert is_dataset_file("QuickControl_PP", empty["info_path"]) is False
    with pytest.raises(ValueError, match="No 1D .scan files"):
        pm.load_1D(empty["info_path"], data_type="QuickControl_PP")


# --------------------------------------------------------------------------- #
#                       QuickControl 2D (QuickControl_2D)                     #
# --------------------------------------------------------------------------- #
def test_quickcontrol_2d_time_domain_fft_and_state_switching(tmp_path: Path):
    """Verify Time-Domain 2D loading, Magic Angle default, shaper FFT peak recovery, and state switching."""
    meta = synthetic.make_synthetic_quickcontrol_2d(
        tmp_path,
        prefix="td_map#10",
        domain="Time",
        n_t1=61,
        dt1_fs=25.0,
        n_pixels=20,
        t2_fs=(250.0, 1500.0),
        n_scans=2,
        polarised=True,
        w0_scanned_cm1=1980.0,
        peak_cm1=2085.0,
    )

    assert is_map_dataset_file("QuickControl_2D", meta["info_path"]) is True
    assert is_dataset_file("QuickControl_PP", meta["info_path"]) is False

    with pytest.raises(ValueError, match="QuickControl_2D"):
        pm.load_1D(meta["info_path"], data_type="QuickControl_PP")

    ds = pm.load_2D(meta["info_path"], data_type="QuickControl_2D")
    assert ds.datatype == "shaper"
    assert ds.is_frequency_domain is False
    assert ds.in_progress is False
    assert ds.active_state == "Magic Angle"
    assert ds.state_names == ["Magic Angle", "Parallel", "Perpendicular"]
    assert ds.has_single_scans is True
    assert ds.nscans == 2
    assert ds.w0_cm1 == pytest.approx(1980.0)
    assert ds.dt1_fs == pytest.approx(25.0)
    np.testing.assert_allclose(ds.delays, [0.25, 1.5])
    np.testing.assert_allclose(ds.probe, meta["probe_cm1"], rtol=1e-6)
    np.testing.assert_allclose(ds.raw_signal, meta["ma_cube_mOD"], rtol=1e-5, atol=1e-6)

    # Process shaper FID into 2D absorptive frequency map and check pump peak frequency
    ds.process(apodise_method="Box", zeropad_factor=4)
    assert np.all(np.diff(ds.probe) > 0)
    w1_mask = (ds.pump >= 2000.0) & (ds.pump <= 2170.0)
    sub = ds.Z[w1_mask, :, 0]
    w1_sub = ds.pump[w1_mask]
    i_min, _ = np.unravel_index(np.argmin(sub), sub.shape)
    assert w1_sub[i_min] == pytest.approx(2085.0, abs=2.5)

    # Verify diagonal extraction and P2DAT round-trip export preserve ascending probe order
    diag_f, diag_s = ds.diagonal(t2=0.25)
    assert len(diag_f) == len(diag_s) and len(diag_f) > 2
    p2_out = tmp_path / "exported.p2dat"
    ds.to_p2dat(p2_out)
    ds_rt = pm.load_2D(p2_out, data_type="P2DAT")
    np.testing.assert_allclose(ds_rt.probe, ds.probe, atol=1e-3)
    np.testing.assert_allclose(ds_rt.Z, ds.Z, rtol=1e-4, atol=1e-4)

    # Switch active state to Parallel and verify raw_signal and interferogram update
    assert ds.set_active_state("Parallel") is True
    assert ds.active_state == "Parallel"
    np.testing.assert_allclose(ds.raw_signal, meta["par_cube_mOD"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ds.raw_interferogram, np.nanmean(meta["par_cube_mOD"], axis=1))


def test_quickcontrol_2d_freq_domain_partial_and_unsorted_wts(tmp_path: Path):
    """Verify Frequency-Domain 2D states (including Anisotropy), partial scan flag, and t2 sorting."""
    fd = synthetic.make_synthetic_quickcontrol_2d(
        tmp_path,
        prefix="fd_map#11",
        domain="Frequency",
        n_t1=25,
        n_pixels=16,
        t2_fs=(1500.0, 300.0, 5000.0),
        n_written_t2=2,
        n_scans=2,
        polarised=True,
    )

    ds = pm.load_2D(fd["info_path"], data_type="QuickControl_2D")
    assert ds.is_frequency_domain is True
    assert ds.datatype == "FreqDomain"
    assert ds.in_progress is True
    assert ds.active_state == "Magic Angle"
    assert ds.state_names == ["Magic Angle", "Parallel", "Perpendicular", "Anisotropy"]
    # Unsorted [1.5, 0.3] ps should be sorted into ascending [0.3, 1.5] ps
    np.testing.assert_allclose(ds.delays, [0.3, 1.5])

    empty_2d = tmp_path / "empty_2d#12.info"
    empty_2d.write_text(
        "Experiment Type\t2D\nDomain\tTime\nStep Size (fs)\t25.0\nQC Version\t4.0\n",
        encoding="utf-8",
    )
    assert is_map_dataset_file_status("QuickControl_2D", empty_2d) == (True, False)


# --------------------------------------------------------------------------- #
#                     GUI Browser & Selector Integration                      #
# --------------------------------------------------------------------------- #
def test_gui_quickcontrol_browser_filtering_and_stripped_labels(qapp, tmp_path: Path):
    """Verify 1D/2D dropdown labels, .info stem display, companion hiding, and Magic Angle default."""
    assert DATA_TYPE_DISPLAY_NAMES["QuickControl_PP"] == "PhaseTech QC 1D (.info)"
    assert DATA_TYPE_DISPLAY_NAMES["QuickControl_2D"] == "PhaseTech QC 2D (.info)"

    m1 = synthetic.make_synthetic_quickcontrol_1d(
        tmp_path, prefix="exp_alpha#01", n_delays=10, n_pixels=16, n_scans=2, polarised=True
    )
    synthetic.make_synthetic_quickcontrol_1d(
        tmp_path, prefix="exp_pending#02", write_scans=False
    )
    m2 = synthetic.make_synthetic_quickcontrol_2d(
        tmp_path, prefix="exp_alpha#03", n_t1=81, n_pixels=32, t2_fs=(200.0, 800.0), n_scans=2
    )

    win = MainWindow()
    try:
        win.RootDir_field.setText(str(tmp_path))
        win._on_rootdir_changed()

        # Select PhaseTech QC 1D (.info) in 1D tab
        set_combo_datatype(win.PP_datatype_cbx, "QuickControl_PP")
        win.refresh_dataset_list()

        items_1d = [
            win._folder_model.item(r) for r in range(win._folder_model.rowCount())
        ]
        labels_1d = [it.text() for it in items_1d if it.text() != "[..]"]
        # Only 1D .info stems appear (no .info suffix, no 2D datasets, no .scan/.spec/.wts files)
        assert labels_1d == ["exp_alpha#01", "exp_pending#02"]
        by_lbl_1d = {it.text(): it for it in items_1d}
        assert bool(by_lbl_1d["exp_alpha#01"].flags() & Qt.ItemFlag.ItemIsEnabled) is True
        assert bool(by_lbl_1d["exp_pending#02"].flags() & Qt.ItemFlag.ItemIsEnabled) is False

        # Load 1D dataset and confirm Magic Angle is active at channel 1 (0-based 0)
        win.load_path(str(m1["info_path"]), data_type="QuickControl_PP")
        assert win.dataset is not None
        assert win.plot_controls.detector == 0
        assert win.plot_controls.detector_lbl.text() == "Magic Angle:"

        # Select PhaseTech QC 2D (.info) in 2D tab
        set_combo_datatype(win.twoD_datatype_cbx, "QuickControl_2D")
        win.twoD_refresh_dataset_list()

        items_2d = [
            win._twoD_folder_model.item(r) for r in range(win._twoD_folder_model.rowCount())
        ]
        labels_2d = [it.text() for it in items_2d if it.text() != "[..]"]
        assert labels_2d == ["exp_alpha#03"]

        # Load 2D dataset and confirm Magic Angle is active in the Polarisation State selector
        win.twoD_load_path(str(m2["info_path"]))
        assert win.twoD_dataset is not None
        assert win.twoD_dataset.active_state == "Magic Angle"
        assert win._twoD_state_cbx.currentText() == "Magic Angle"
    finally:
        win.close()


def test_quickcontrol_2d_v094_negative_footer_encoder_t2_and_probe_spectrum(tmp_path: Path):
    """Verify v0.9.4 trailing single-beam footer stripping, negative T2 recovery, and no false in_progress."""
    sub_list = tmp_path / "with_list"
    meta = synthetic.make_synthetic_quickcontrol_2d(
        sub_list,
        prefix="ir_v094_01",
        n_t1=41,
        dt1_fs=25.0,
        n_pixels=20,
        t2_fs=(-20000.0, 250.0, 1500.0),
        n_scans=2,
        polarised=False,
        write_wts=False,
        qc_v094_footer=True,
        include_wt_list=True,
    )

    ds = pm.load_2D(meta["info_path"], data_type="QuickControl_2D")
    assert ds.in_progress is False
    assert len(ds.raw_t1delays) == 41
    np.testing.assert_allclose(ds.raw_t1delays, meta["axis0"])
    np.testing.assert_allclose(ds.delays, [-20.0, 0.25, 1.5])
    # Single-beam ~50 OD (~50,000 mOD) footer row must be completely stripped
    assert float(np.nanmax(np.abs(ds.raw_signal))) < 50.0
    np.testing.assert_allclose(ds.raw_signal, meta["ma_cube_mOD"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ds.units["probe_spectrum"], meta["single_beam"], rtol=1e-5)

    # Also verify fallback to footer column-0 encoder T2 when neither .wts nor Waiting Time list exists
    sub_enc = tmp_path / "encoder_only"
    meta_enc = synthetic.make_synthetic_quickcontrol_2d(
        sub_enc,
        prefix="ir_v094_enc01",
        n_t1=41,
        dt1_fs=25.0,
        n_pixels=20,
        t2_fs=(-20000.0, 250.0, 1500.0),
        n_scans=2,
        polarised=False,
        write_wts=False,
        qc_v094_footer=True,
        include_wt_list=False,
    )
    ds_enc = pm.load_2D(meta_enc["info_path"], data_type="QuickControl_2D")
    np.testing.assert_allclose(ds_enc.delays, [-20.0, 0.25, 1.5])
    assert len(ds_enc.raw_t1delays) == 41


def test_quickcontrol_2d_partial_interrupted_scan_nan_padding(tmp_path: Path):
    """Verify an interrupted partial .scan file preserves its final row and NaN-pads missing rows."""
    meta = synthetic.make_synthetic_quickcontrol_2d(
        tmp_path,
        prefix="ir_partial_01",
        n_t1=30,
        dt1_fs=25.0,
        n_pixels=16,
        t2_fs=(500.0,),
        n_scans=2,
        polarised=False,
        qc_v094_footer=True,
    )
    # Truncate scan #002 to 15 genuine rows without a trailing single-beam footer
    scan2_path = tmp_path / "ir_partial_01_T01#002.scan"
    full_lines = scan2_path.read_text(encoding="utf-8").splitlines()
    scan2_path.write_text("\n".join(full_lines[:15]) + "\n", encoding="utf-8")

    ds = pm.load_2D(meta["info_path"], data_type="QuickControl_2D")
    assert ds.in_progress is True
    assert ds.raw_signal.shape == (30, 16, 1)
    assert ds.raw_signal_ss is not None
    # Scan 2 must retain all 15 measured rows and have NaN for rows 15..29
    assert np.all(np.isfinite(ds.raw_signal_ss[:15, :, 0, 1]))
    assert np.all(np.isnan(ds.raw_signal_ss[15:, :, 0, 1]))


def test_quickcontrol_2d_electronic_2des_unfolding_dual_axes_and_gui(qapp, tmp_path: Path):
    """Verify visible 2D-ES Nyquist unfolding, decreasing 10^3 cm-1 + dual nm axes, and GUI limits."""
    import matplotlib.pyplot as plt

    meta = synthetic.make_synthetic_quickcontrol_2des(
        tmp_path,
        prefix="vis_2des_01",
        n_t1=107,
        dt1_fs=12.0,
        n_pixels=128,
        t2_fs=(250.0, 1500.0),
        n_scans=2,
        w0_scanned_cm1=880.0,
        peak_cm1=20800.0,
    )

    ds = pm.load_2D(meta["info_path"], data_type="QuickControl_2D")
    ds.process(apodise_method="Cos", zeropad_factor=2)

    assert ds.is_electronic is True
    assert ds.freq_units == "cm-1"
    assert np.all(np.diff(ds.probe) > 0)
    assert float(np.min(ds.probe)) > 12000.0
    assert ds.w0_cm1 > 18000.0
    assert ds.units["unitsZ_lbl"] == r"S$_{\text{2DES}}$"
    assert "nm)" in ds.sample_info()

    # Peak in processed 2D-ES map must land near 20,800 cm-1 along both pump and probe
    i_pu, i_pr = np.unravel_index(np.argmin(ds.Z[:, :, 0]), ds.Z[:, :, 0].shape)
    assert ds.pump[i_pu] == pytest.approx(20800.0, abs=20.0)
    assert ds.probe[i_pr] == pytest.approx(20800.0, abs=60.0)

    # Round-trip display <-> native coordinate conversion
    to_disp = pm.twoD.plot.display_converter(ds)
    to_nat = pm.twoD.plot.native_converter(ds)
    np.testing.assert_allclose(to_disp([20500.0, 21200.0]), [20.5, 21.2])
    np.testing.assert_allclose(to_nat([20.5, 21.2]), [20500.0, 21200.0])

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    try:
        ds.plot_map(0.25, ax=ax)
        # Option A: wavenumbers decrease left-to-right / bottom-to-top so nm increases
        assert ax.get_xlim()[0] > ax.get_xlim()[1]
        assert ax.get_ylim()[0] > ax.get_ylim()[1]
        assert 10.0 < ax.get_xlim()[0] < 35.0
        sec_axes = getattr(ax, "_pymorgan_sec_axes", None)
        assert sec_axes is not None and len(sec_axes) == 2
        assert "nm" in sec_axes[0].get_xlabel()
        assert "nm" in sec_axes[1].get_ylabel()
    finally:
        plt.close(fig)

    win = MainWindow()
    try:
        win.twoD_load_path(str(meta["info_path"]), data_type="QuickControl_2D")
        pc = win.twoD_plot_controls
        assert pc.freq_scale() == 1000.0
        assert pc.pump_min.suffix() == " 10³ cm⁻¹"
        assert 15.0 < pc.pump_min.value() < pc.pump_max.value() < 30.0
        assert 10.0 < pc.probe_min.value() < pc.probe_max.value() < 30.0
        assert win.twoD_PC_dualAxis.isChecked() is True
        assert win.twoD_PC_square.isChecked() is False
        assert win.twoDaxes.ax.get_xlim()[0] > win.twoDaxes.ax.get_xlim()[1]
        assert win.twoDaxes.ax.get_ylim()[0] > win.twoDaxes.ax.get_ylim()[1]

        # Syncing limits from an inverted axis must keep spinboxes ordered (Min <= Max) in 10^3 cm-1
        win._on_twoD_sync_limits()
        assert pc.pump_min.value() < pc.pump_max.value()
        assert pc.probe_min.value() < pc.probe_max.value()
    finally:
        win.close()


def test_quickcontrol_partial_first_scan_preserves_full_scans(tmp_path: Path):
    """Verify partial first .scan files NaN-pad to expected rows without truncating later scans."""
    # 1D multi-scan: scan #001 is truncated to 12 of 30 rows, scan #002 is full 30 rows
    m1_multi = synthetic.make_synthetic_quickcontrol_1d(
        tmp_path / "pp_multi",
        prefix="pp_part_01",
        n_delays=30,
        n_pixels=12,
        n_scans=2,
        polarised=False,
    )
    pp_scan1 = tmp_path / "pp_multi" / "pp_part_01#001.scan"
    pp_scan1.write_text(
        "\n".join(pp_scan1.read_text(encoding="utf-8").splitlines()[:12]) + "\n",
        encoding="utf-8",
    )
    ds1_multi = pm.load_1D(m1_multi["info_path"], data_type="QuickControl_PP")
    assert len(ds1_multi.delays) == 30
    assert ds1_multi.Zavg_R.shape == (30, 12, 1)
    assert np.all(np.isfinite(ds1_multi.Zss_R[:12, :, 0, 0]))
    assert np.all(np.isnan(ds1_multi.Zss_R[12:, :, 0, 0]))
    assert np.all(np.isfinite(ds1_multi.Zss_R[:, :, 0, 1]))

    # 1D single-scan truncated to 12 of 30 rows
    m1_single = synthetic.make_synthetic_quickcontrol_1d(
        tmp_path / "pp_single",
        prefix="pp_part_02",
        n_delays=30,
        n_pixels=12,
        n_scans=1,
        polarised=False,
    )
    pp_s1 = tmp_path / "pp_single" / "pp_part_02#001.scan"
    pp_s1.write_text(
        "\n".join(pp_s1.read_text(encoding="utf-8").splitlines()[:12]) + "\n",
        encoding="utf-8",
    )
    ds1_single = pm.load_1D(m1_single["info_path"], data_type="QuickControl_PP")
    assert len(ds1_single.delays) == 30
    assert np.all(np.isfinite(ds1_single.Zss_R[:12, :, 0, 0]))
    assert np.all(np.isnan(ds1_single.Zss_R[12:, :, 0, 0]))

    # 2D multi-scan: scan #001 is truncated to 12 of 30 rows, scan #002 is complete
    m2_multi = synthetic.make_synthetic_quickcontrol_2d(
        tmp_path / "td_multi",
        prefix="ir_first_part_01",
        n_t1=30,
        dt1_fs=25.0,
        n_pixels=16,
        t2_fs=(500.0,),
        n_scans=2,
        polarised=False,
        qc_v094_footer=True,
    )
    scan1_2d = tmp_path / "td_multi" / "ir_first_part_01_T01#001.scan"
    scan1_2d.write_text(
        "\n".join(scan1_2d.read_text(encoding="utf-8").splitlines()[:12]) + "\n",
        encoding="utf-8",
    )
    ds2_multi = pm.load_2D(m2_multi["info_path"], data_type="QuickControl_2D")
    assert ds2_multi.in_progress is True
    assert len(ds2_multi.raw_t1delays) == 30
    assert ds2_multi.raw_signal.shape == (30, 16, 1)
    assert np.all(np.isfinite(ds2_multi.raw_signal_ss[:12, :, 0, 0]))
    assert np.all(np.isnan(ds2_multi.raw_signal_ss[12:, :, 0, 0]))
    assert np.all(np.isfinite(ds2_multi.raw_signal_ss[:, :, 0, 1]))

    # 2D single-scan truncated to 12 of 30 rows
    m2_single = synthetic.make_synthetic_quickcontrol_2d(
        tmp_path / "td_single",
        prefix="ir_first_part_02",
        n_t1=30,
        dt1_fs=25.0,
        n_pixels=16,
        t2_fs=(500.0,),
        n_scans=1,
        polarised=False,
        qc_v094_footer=True,
    )
    scan1_single = tmp_path / "td_single" / "ir_first_part_02_T01#001.scan"
    scan1_single.write_text(
        "\n".join(scan1_single.read_text(encoding="utf-8").splitlines()[:12]) + "\n",
        encoding="utf-8",
    )
    ds2_single = pm.load_2D(m2_single["info_path"], data_type="QuickControl_2D")
    assert ds2_single.in_progress is True
    assert len(ds2_single.raw_t1delays) == 30
    assert ds2_single.raw_signal.shape == (30, 16, 1)


def test_quickcontrol_2des_negative_nyquist_unfolding_and_ph_plot(qapp, tmp_path: Path):
    """Verify negative-Nyquist alias selection and GUI PH diagnostic upper-half slicing."""
    meta = synthetic.make_synthetic_quickcontrol_2des(
        tmp_path,
        prefix="vis_2des_neg_nyq",
        n_t1=107,
        dt1_fs=12.0,
        n_pixels=128,
        t2_fs=(250.0, 1500.0),
        n_scans=2,
        w0_scanned_cm1=0.0,
        peak_cm1=21500.0,
        cwl_nm=480.0,
    )

    ds = pm.load_2D(meta["info_path"], data_type="QuickControl_2D")
    ds.process(apodise_method="Cos", zeropad_factor=2)

    assert ds.units.get("nyquist_invert") is True
    i_pu, i_pr = np.unravel_index(np.argmin(ds.Z[:, :, 0]), ds.Z[:, :, 0].shape)
    assert ds.pump[i_pu] == pytest.approx(21500.0, abs=30.0)
    assert ds.probe[i_pr] == pytest.approx(21500.0, abs=80.0)

    win = MainWindow()
    try:
        win.twoD_load_path(str(meta["info_path"]), data_type="QuickControl_2D")
        win.twoD_other_btn_PH.setChecked(True)
        win._twoD_update_other_plot()
        pump_lines = [ln for ln in win.twoD_other_plot.ax.get_lines() if ln.get_label() == "Pump"]
        assert len(pump_lines) == 1
        xdata = np.asarray(pump_lines[0].get_xdata(), dtype=float)
        assert float(np.min(xdata)) >= 20500.0
        assert float(np.min(xdata)) <= 21500.0 <= float(np.max(xdata))
    finally:
        win.close()


def test_quickcontrol_2d_calibrated_probe_csv_wavelength_preserved(tmp_path: Path):
    """Verify Dataset2D.from_file does not overwrite QuickControl's nm->cm-1 CalibratedProbe.csv axis."""
    sub = tmp_path / "qc_nm_cal"
    meta = synthetic.make_synthetic_quickcontrol_2des(
        sub,
        prefix="vis_cal_01",
        n_t1=64,
        dt1_fs=12.0,
        n_pixels=64,
        t2_fs=(250.0,),
        n_scans=1,
    )
    cal_nm = np.linspace(480.0, 600.0, 64)
    np.savetxt(sub / "CalibratedProbe.csv", cal_nm, delimiter=",")

    ds = pm.load_2D(meta["info_path"], data_type="QuickControl_2D")
    assert ds.freq_units == "cm-1"
    assert ds.cal_level == "datadir"
    assert "CalibratedProbe.csv" in ds.calibration_status()
    assert np.all(np.diff(ds.probe) > 0)
    assert ds.probe[0] == pytest.approx(1.0e7 / 600.0, rel=1e-5)
    assert ds.probe[-1] == pytest.approx(1.0e7 / 480.0, rel=1e-5)
