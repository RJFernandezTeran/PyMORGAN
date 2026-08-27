"""Regression tests ensuring UK/US spelling changes do not break core commands,
third-party library calls, or keyword arguments.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import pymorgan as pm
from pymorgan import helpers as hlp
from pymorgan.display import add_subplot_labels
from pymorgan.oneD.chirp import _movmedian, fit_chirp_wavelet
from pymorgan.steadyState.dataset import Spectrum, SpectrumKind, SpectrumSeries
import synthetic


def test_movmedian_pandas_rolling():
    """Verify that _movmedian does not fail with pandas center=True parameter."""
    x = np.array([1.0, 2.0, 10.0, 3.0, 2.0, 1.0, 0.5])
    res = _movmedian(x, window=3)
    assert len(res) == len(x)
    assert not np.any(np.isnan(res))


def test_add_subplot_labels_alignments():
    """Verify that add_subplot_labels accepts both US and UK centre/center alignments."""
    fig, axes = plt.subplots(2, 2)
    # Test top-centre and top-center
    text_objs_uk = add_subplot_labels(axes, position="top-centre")
    assert len(text_objs_uk) == 4
    for txt in text_objs_uk:
        assert txt.get_ha() == "center"

    text_objs_us = add_subplot_labels(axes, position="top-center")
    assert len(text_objs_us) == 4
    for txt in text_objs_us:
        assert txt.get_ha() == "center"

    # Test bottom-centre and bottom-center
    text_objs_bc_uk = add_subplot_labels(axes, position="bottom-centre")
    for txt in text_objs_bc_uk:
        assert txt.get_ha() == "center"

    text_objs_bc_us = add_subplot_labels(axes, position="bottom-center")
    for txt in text_objs_bc_us:
        assert txt.get_ha() == "center"

    plt.close(fig)


def test_add_glow_color_and_colour():
    """Verify that add_glow accepts both 'colour' and 'color' and renders without NameError."""
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 50)
    y = np.sin(x)
    ax.set_ylim(-2, 2)

    # Calling with colour
    hlp.add_glow(ax, x, y, colour="red")
    # Calling with color
    hlp.add_glow(ax, x, y, color="blue")
    plt.close(fig)


def test_spectrum_and_series_normalize():
    """Verify Spectrum and SpectrumSeries support both normalise/normalize and plot overlays."""
    x = np.linspace(1000, 2000, 50)
    y = np.exp(-((x - 1500) ** 2) / (2 * 50**2))
    sp1 = Spectrum(x, y, kind=SpectrumKind.ABSORPTION, label="sp1")
    sp2 = Spectrum(x, y * 2, kind=SpectrumKind.ABSORPTION, label="sp2")

    # Method aliases
    assert np.isclose(np.nanmax(sp1.normalised().y), 1.0)
    assert np.isclose(np.nanmax(sp1.normalized().y), 1.0)

    # Plot single spectrum
    fig, ax = plt.subplots()
    sp1.plot(ax=ax, normalize=True)
    sp1.plot(ax=ax, normalise=True)
    plt.close(fig)

    # Spectrum series
    series = SpectrumSeries([sp1, sp2])
    assert np.isclose(np.nanmax(series.normalised()[1].y), 1.0)
    assert np.isclose(np.nanmax(series.normalized()[1].y), 1.0)

    # Series plot overlay
    fig, ax = plt.subplots()
    series.plot(ax=ax, normalize=True)
    series.plot(ax=ax, normalise=True)
    plt.close(fig)


def test_draw_overlay_helper_cls(tmp_path):
    """Verify that _draw_overlay_helper does not throw NameError on color=colour."""
    from pymorgan.twoD.plot import _draw_overlay_helper
    fig, ax = plt.subplots()
    x_raw = np.array([1900.0, 1950.0, 2000.0])
    y_raw = np.array([1900.0, 1950.0, 2000.0])
    _draw_overlay_helper(
        ax,
        x_raw=x_raw,
        y_raw=y_raw,
        fit_params=(1.0, 0.0),
        colour="cyan",
        style="Both",
        lw=2.0,
    )
    plt.close(fig)


def test_2d_antidiagonal_center_and_centre(tmp_path):
    """Verify Dataset2D antidiagonal & compare_diag_antidiag accept centre & center."""
    p2 = synthetic.make_synthetic_p2dat(tmp_path / "syn.p2dat")
    ds = pm.load_2D(p2["path"], data_type="P2DAT")

    # antidiagonal with centre
    rel1, sig1 = ds.antidiagonal(centre=(2040.0, 2040.0))
    # antidiagonal with center
    rel2, sig2 = ds.antidiagonal(center=(2040.0, 2040.0))
    assert np.allclose(rel1, rel2)
    assert np.allclose(sig1, sig2)

    # compare_diag_antidiag with centre and center
    res1 = ds.compare_diag_antidiag(centre=(2040.0, 2040.0))
    res2 = ds.compare_diag_antidiag(center=(2040.0, 2040.0))
    assert np.allclose(res1[0], res2[0])
    assert np.allclose(res1[1], res2[1])
    assert np.allclose(res1[2], res2[2])

    # centre_line_slope alias
    assert hasattr(ds, "centre_line_slope")
    assert hasattr(ds, "center_line_slope")
