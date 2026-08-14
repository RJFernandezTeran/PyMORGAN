"""Unit tests for 1D and 2D axis unit conversions and display settings."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
import synthetic

import pymorgan as pm
from pymorgan.helpers import AxisUnit, DatasetUnits
from pymorgan.twoD.dataset import Dataset2D


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
