"""Unit tests for 1D and 2D axis unit conversions and display settings."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import synthetic

import pymorgan as pm
from pymorgan import helpers as hlp
from pymorgan.helpers import DatasetUnits


@pytest.fixture
def dataset2d(tmp_path):
    info = synthetic.make_synthetic_p2dat(tmp_path / "syn.p2dat")
    return pm.load_2D(info["path"], data_type="P2DAT")


def test_1d_reports_probe_delay_and_signal(dataset):
    units = dataset.axis_units()
    assert isinstance(units, DatasetUnits)
    assert units.x.unit == "cm-1" and units.x.quantity == "Wavenumber"
    assert units.y.unit == "ps" and units.y.quantity == "Delay"
    assert units.z.unit
    assert units.t2 is None


def test_1d_display_follows_the_x_axis_unit(dataset):
    pm.update_settings(x_axis_unit="nm")
    assert dataset.axis_units(display=True).x.unit == "nm"
    pm.update_settings(x_axis_unit="native")
    assert dataset.axis_units(display=True).x.unit == "cm-1"


def test_2d_reports_probe_pump_and_signal(dataset2d):
    units = dataset2d.axis_units()
    assert isinstance(units, DatasetUnits)
    assert units.x.unit == "cm-1"
    assert units.y.unit == "cm-1"
    assert units.z.unit
    assert units.t2 is not None and units.t2.unit == "ps"


def test_unit_formatting_and_time_value_list_parsing():
    assert hlp.plain_time_unit(r"$\mu$s") == "µs"
    assert hlp.plain_time_unit({"unitsT_ltx": "ns"}) == "ns"
    assert hlp.plain_spectral_unit("cm-1", scaled=True) == "10³ cm⁻¹"
    assert hlp.plain_spectral_unit("cm-1", scaled=False) == "cm⁻¹"
    assert hlp.format_spectral_value(0.24797, unit="eV", rounded=True) == "0.248"
    assert hlp.format_spectral_value(1955.2, unit="cm-1", rounded=True) == "1955"
    assert hlp.format_spectral_value(20.0, unit="cm-1", scaled=True, rounded=True) == "20"

    vals_ps = hlp.parse_value_list("500 fs, 2.5, 1 ns, 2µs, 10:5:20", target_time_unit="ps")
    assert vals_ps == pytest.approx([0.5, 2.5, 1000.0, 2e6, 10.0, 15.0, 20.0])

    vals_ns = hlp.parse_value_list("500 ns, 1 us, 1.5µs", target_time_unit="ns")
    assert vals_ns == pytest.approx([500.0, 1000.0, 1500.0])

    # A comma prevents a trailing unit word from binding to the previous clause.
    vals_comma = hlp.parse_value_list("500, ns", target_time_unit="ps")
    assert vals_comma == pytest.approx([500.0])


def test_plot_kinetics_and_scan_kinetics_unit_aware_labels(tmp_path):
    from pymorgan.oneD.dataset import Dataset1D

    probe_nm = np.linspace(400.0, 700.0, 31)
    delays = np.array([-1.0, 0.5, 2.0, 10.0])
    # Distinct signal per pixel so selected column is unambiguous
    Z = np.outer(np.arange(1, len(delays) + 1, dtype=float), np.arange(len(probe_nm), dtype=float))
    ds_uv = Dataset1D(Z, delays, probe_nm, units=hlp.units2dic("nm", "ps", "mOD"))

    # Native nm
    ax, _, Y = ds_uv.plot_kinetics([500.0])
    assert ax.get_legend().get_texts()[0].get_text() == "500 nm"
    idx_500 = int(np.argmin(np.abs(probe_nm - 500.0)))
    np.testing.assert_allclose(Y[:, 0], Z[:, idx_500])
    plt.close(ax.figure)

    # UV-Vis displayed as 10^3 cm-1
    pm.update_settings(x_axis_unit="cm-1")
    ax, _, Y = ds_uv.plot_kinetics([500.0])
    lbl_wn = ax.get_legend().get_texts()[0].get_text()
    assert "10^{3}" in lbl_wn and "cm" in lbl_wn
    assert lbl_wn.startswith("20 ")
    np.testing.assert_allclose(Y[:, 0], Z[:, idx_500])
    plt.close(ax.figure)

    # Display-unit input_unit="eV" selects 500 nm via 2.4797 eV and formats in eV
    pm.update_settings(x_axis_unit="eV")
    ax, _, Y = ds_uv.plot_kinetics([1239.841984 / 500.0], input_unit="eV")
    assert ax.get_legend().get_texts()[0].get_text() == "2.48 eV"
    np.testing.assert_allclose(Y[:, 0], Z[:, idx_500])
    plt.close(ax.figure)

    # Mid-IR dataset in eV must not collapse to "0 eV"
    probe_ir = np.linspace(1950.0, 2050.0, 21)
    Z_ir = np.outer(np.ones(len(delays)), np.arange(len(probe_ir), dtype=float))
    ds_ir = Dataset1D(Z_ir, delays, probe_ir, units=hlp.units2dic("cm-1", "ps", "mOD"))
    ax, _, Y_ir = ds_ir.plot_kinetics([2000.0], x_axis_unit="eV")
    lbl_ir = ax.get_legend().get_texts()[0].get_text()
    assert lbl_ir != "0 eV"
    assert "0.248" in lbl_ir and lbl_ir.endswith("eV")
    plt.close(ax.figure)

    # Uncalibrated pixel dataset
    units_px = hlp.units2dic("cm-1", "ps", "mOD")
    units_px["unitsL_lbl"] = "Pixel"
    units_px["unitsL_ltx"] = "px"
    units_px["L"] = "px"
    units_px["cal_source"] = "uncalibrated"
    ds_px = Dataset1D(Z_ir, delays, np.arange(len(probe_ir), dtype=float), units=units_px)
    ax, _, _ = ds_px.plot_kinetics([10.0], x_axis_unit="eV")
    assert ax.get_legend().get_texts()[0].get_text() == "10 px"
    plt.close(ax.figure)

    # Per-scan kinetics follows Settings.x_axis_unit
    harp = synthetic.make_synthetic_harpia_dataset(tmp_path / "scan_unit.dat", nscans=2)
    ds_scan = pm.load_1D(harp["path"], data_type="HARPIA_TA")
    pm.update_settings(x_axis_unit="eV")
    mid_nm = float(ds_scan.probe[len(ds_scan.probe) // 2])
    ax_s1, _, _ = ds_scan.plot_scan_kinetics([mid_nm], binsize=1)
    mgr = getattr(ax_s1.figure.canvas, "manager", None)
    if mgr is not None and hasattr(mgr, "get_window_title"):
        title = mgr.get_window_title()
        if title:
            assert "eV" in title
    plt.close(ax_s1.figure)

    # Multi-position per-scan kinetics legend prefix
    pos1_ev = float(hlp.convert_spectral(ds_scan.probe[0], "nm", "eV"))
    pos2_ev = float(hlp.convert_spectral(ds_scan.probe[-1], "nm", "eV"))
    ax_s2_list, _, _ = ds_scan.plot_scan_kinetics([pos1_ev, pos2_ev], ax=plt.subplots()[1], input_unit="eV")
    texts = [t.get_text() for t in ax_s2_list.get_legend().get_texts()]
    assert any("eV" in txt for txt in texts)
    plt.close("all")
