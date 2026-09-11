"""Unit tests for plotting, quick-plot rendering, display helpers, legend pruning, and themes."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pytest
import synthetic
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.collections import QuadMesh
from matplotlib.contour import QuadContourSet

import pymorgan as pm
from pymorgan import helpers as hlp
from pymorgan.oneD import plot as P
from pymorgan.gui.app import _resolve_theme
from pymorgan.gui.theme import DARK_BG, DARK_FG, LIGHT_BG, style_figure


@pytest.fixture
def p2dat_dataset(tmp_path):
    info = synthetic.make_synthetic_p2dat(tmp_path / "syn.p2dat")
    return pm.load_2D(info["path"], data_type="P2DAT")


# --------------------------------------------------------------------------- #
#                               Method Plotters                               #
# --------------------------------------------------------------------------- #
def test_all_method_plotters(corrected):
    assert isinstance(corrected.plot_contour(Zscale=80), Axes)
    assert isinstance(corrected.plot_surface(), Axes)
    assert isinstance(corrected.plot_spectra([0.5, 1, 5, 20], doSmooth=1), Axes)
    ax, t, Y = corrected.plot_kinetics([1955, 1980], plotStyle="-")
    assert isinstance(ax, Axes)
    assert Y.shape[0] == t.shape[0]


def test_module_function_form_and_override(corrected):
    ax = P.plot_contour(corrected, label_style="/", cmap_ID="Rd/Wh/Bu v2", Yscale="lin")
    assert isinstance(ax, Axes)


def test_species_spectra(corrected):
    n = corrected.probe.size
    Sfit = np.random.RandomState(0).randn(n, 2)
    ax = corrected.plot_species_spectra(
        Sfit, [12.0, 120.0], [0.5, 5.0], [False, False], "Sequential"
    )
    assert isinstance(ax, Axes)


def test_species_spectra_target_labels_only_species_names(corrected):
    n = corrected.probe.size
    Sfit = np.random.RandomState(0).randn(n, 3)
    ax = corrected.plot_species_spectra(
        Sfit,
        [1.0, 5.0, 20.0, 50.0],  # 4 rates, 3 species
        None,
        None,
        "Target",
        species_labels=["Q", "I", "D"],
    )
    labels = [line.get_label() for line in ax.get_lines() if not line.get_label().startswith("_")]
    assert labels == ["Q", "I", "D"]



def test_settings_drive_defaults(corrected):
    pm.update_settings(cmap="DkRd/Wh/DkBu", time_axis_scale="lin", label_style="[]")
    assert isinstance(corrected.plot_contour(), Axes)


# --------------------------------------------------------------------------- #
#                               Quick plots                                   #
# --------------------------------------------------------------------------- #
def test_quick_plot_draws_an_image_instead_of_contours(dataset):
    _, ax = plt.subplots()
    dataset.plot_contour(ax=ax, quick=False)
    assert isinstance(ax.collections[0], QuadContourSet | matplotlib.collections.Collection)
    assert not isinstance(ax.collections[0], QuadMesh)

    _, ax_quick = plt.subplots()
    dataset.plot_contour(ax=ax_quick, quick=True)
    assert isinstance(ax_quick.collections[0], QuadMesh)


def test_quick_plots_setting_drives_the_default(dataset):
    pm.update_settings(quick_plots=True)
    _, ax = plt.subplots()
    dataset.plot_contour(ax=ax)
    assert isinstance(ax.collections[0], QuadMesh)

    _, ax_off = plt.subplots()
    dataset.plot_contour(ax=ax_off, quick=False)
    assert not isinstance(ax_off.collections[0], QuadMesh)


# --------------------------------------------------------------------------- #
#                              Display Helpers                                #
# --------------------------------------------------------------------------- #
def test_display_helpers_exist():
    assert callable(pm.show_plots)
    assert callable(pm.show)
    assert callable(pm.close_plots)
    assert callable(pm.add_subplot_labels)


def test_show_nonblocking_then_close():
    plt.figure()
    pm.show_plots(block=False)
    pm.close_plots()
    assert not plt.get_fignums()


def test_add_subplot_labels():
    fig, axes = plt.subplots(2, 2)
    texts = pm.add_subplot_labels(axes)
    assert len(texts) == 4
    assert texts[0].get_text() == "(a)"
    assert texts[1].get_text() == "(b)"
    assert texts[2].get_text() == "(c)"
    assert texts[3].get_text() == "(d)"
    plt.close(fig)


# --------------------------------------------------------------------------- #
#                              Theme & Styling                                #
# --------------------------------------------------------------------------- #
class _FakeFactory:
    @staticmethod
    def keys():
        return ["Fusion", "Windows", "windows11", "macos"]


def test_resolve_theme_plain_is_light():
    assert _resolve_theme(_FakeFactory, "macos") == ("macos", False)
    assert _resolve_theme(_FakeFactory, "Fusion") == ("Fusion", False)


def test_resolve_theme_unknown_falls_back_to_fusion():
    assert _resolve_theme(_FakeFactory, "nonsense") == ("Fusion", False)


def test_resolve_theme_dark_forces_fusion():
    assert _resolve_theme(_FakeFactory, "Fusion Dark") == ("Fusion", True)
    assert _resolve_theme(_FakeFactory, "dark") == ("Fusion", True)


def test_style_figure_light_is_white():
    fig, ax = plt.subplots()
    style_figure(fig, dark=False)
    assert to_rgba(fig.get_facecolor()) == to_rgba(LIGHT_BG)
    assert to_rgba(ax.get_facecolor()) == to_rgba(LIGHT_BG)
    plt.close(fig)


def test_style_figure_dark_recolours_face_and_ink():
    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_title("left title", loc="left")
    style_figure(fig, dark=True)
    assert to_rgba(fig.get_facecolor()) == to_rgba(DARK_BG)
    assert to_rgba(ax.get_facecolor()) == to_rgba(DARK_BG)
    assert to_rgba(ax.xaxis.label.get_color()) == to_rgba(DARK_FG)
    if hasattr(ax, "_left_title"):
        assert to_rgba(ax._left_title.get_color()) == to_rgba(DARK_FG)
    plt.close(fig)


def test_style_figure_dark_colorbar_top_zlabel(tmp_path):
    info = synthetic.make_synthetic_pdat(tmp_path / "test.pdat")
    ds = pm.load_1D(info["path"], data_type="PDAT")
    fig, ax = plt.subplots()
    ds.plot_contour(ax=ax, cbarLbl="top")
    style_figure(fig, dark=True)
    cax = fig.axes[1]
    if hasattr(cax, "_left_title"):
        assert to_rgba(cax._left_title.get_color()) == to_rgba(DARK_FG)
    plt.close(fig)


def test_plot_contour_aspect_auto(tmp_path):
    info = synthetic.make_synthetic_pdat(tmp_path / "test.pdat")
    ds = pm.load_1D(info["path"], data_type="PDAT")
    fig, ax = plt.subplots()
    ax_out = ds.plot_contour(ax=ax, aspect="auto")
    assert ax_out.get_aspect() == "auto"
    plt.close(fig)


def test_get_trace_cmap():
    colors_rb = hlp.get_trace_cmap("rainbow", 5)
    assert len(colors_rb) == 5

    colors_rb_rev = hlp.get_trace_cmap("rainbow reversed", 5)
    assert len(colors_rb_rev) == 5
    # First colour of reversed rainbow matches last colour of rainbow
    assert np.allclose(colors_rb[0], colors_rb_rev[-1])

    colors_rpb = hlp.get_trace_cmap("red-purple-blue", 5)
    assert len(colors_rpb) == 5
    # Red at start, Blue at end
    assert colors_rpb[0][0] > 0.8  # Red channel strong
    assert colors_rpb[-1][2] > 0.6  # Blue channel strong


def test_plot_spectra_and_kinetics_traces_cmap(tmp_path):
    info = synthetic.make_synthetic_pdat(tmp_path / "test.pdat")
    ds = pm.load_1D(info["path"], data_type="PDAT")

    fig, ax = plt.subplots()
    ds.plot_spectra(ds.delays[:3], ax=ax, traces_cmap="red-purple-blue")
    lines = ax.get_lines()
    assert len(lines) >= 3
    # Check red-purple-blue colours on spectra traces
    c0 = to_rgba(lines[0].get_color())
    c2 = to_rgba(lines[2].get_color())
    assert c0[0] > 0.8  # Red channel
    assert c2[2] > 0.6  # Blue channel
    plt.close(fig)

    fig, ax = plt.subplots()
    ds.plot_kinetics(ds.probe[:3], ax=ax, traces_cmap="rainbow reversed")
    lines_k = ax.get_lines()
    assert len(lines_k) >= 3
    plt.close(fig)


def test_format_coord_1D(dataset):
    fig, ax = plt.subplots()
    dataset.plot_contour(ax=ax)
    assert hasattr(ax, "format_coord")
    # Query within data bounds
    coord_str = ax.format_coord(dataset.probe[5], dataset.delays[5])
    assert "z=" in coord_str
    assert "x=" in coord_str and "y=" in coord_str
    plt.close(fig)


def test_format_coord_2D(p2dat_dataset):
    fig, ax = plt.subplots()
    map_axes = p2dat_dataset.plot_map(p2dat_dataset.delays[0], ax=ax)
    assert hasattr(map_axes.ax, "format_coord")
    coord_str = map_axes.ax.format_coord(p2dat_dataset.probe[2], p2dat_dataset.pump[2])
    assert "z=" in coord_str
    assert "x=" in coord_str and "y=" in coord_str
    plt.close(fig)


def test_asinh_contour_and_ticks(dataset):
    fig, ax = plt.subplots()
    ax_out = dataset.plot_contour(ax=ax, Asinh=True, asinh_pct=5.0)
    assert isinstance(ax_out, Axes)
    # Check that norm is AsinhNorm
    mappable = ax.collections[0]
    assert isinstance(mappable.norm, mcolors.AsinhNorm)
    plt.close(fig)


def test_masked_probe_contour_plot_descending_and_ascending():
    from pymorgan.oneD.dataset import Dataset1D

    delays = np.linspace(-1, 10, 30)
    # Descending probe (e.g. wavenumbers)
    probe_desc = np.linspace(2000, 1000, 60)
    Zavg_R = (np.sin(delays[:, None] / 2) * np.cos(probe_desc[None, :] / 100))[:, :, None]
    units = {
        "unitsL_lbl": "Wavenumber",
        "unitsL_ltx": r"$\mathrm{cm^{-1}}$",
        "unitsT_lbl": "Delay",
        "unitsT_ltx": "ps",
        "unitsZ_lbl": "Delta A",
        "unitsZ_ltx": "mOD",
    }
    ds_desc = Dataset1D(Zavg_R.copy(), delays, probe_desc, units)
    ds_desc.mask_probe_regions([(1400.0, 1600.0)])

    fig, ax = plt.subplots()
    ds_desc.plot_contour(ax=ax, ShowLines=True, smooth=2)
    # Should render filled contours and contour lines covering both segments (below 1400 and above 1600)
    n_paths = sum(len(c.get_paths()) for c in ax.collections)
    assert n_paths > 20
    plt.close(fig)

    # Ascending probe (e.g. nm)
    probe_asc = np.linspace(400, 700, 60)
    Zavg_R_asc = (np.sin(delays[:, None] / 2) * np.cos(probe_asc[None, :] / 50))[:, :, None]
    ds_asc = Dataset1D(Zavg_R_asc, delays, probe_asc, units)
    ds_asc.mask_probe_regions([(500.0, 550.0)])

    fig, ax = plt.subplots()
    ds_asc.plot_contour(ax=ax, ShowLines=True, smooth=3)
    n_paths_asc = sum(len(c.get_paths()) for c in ax.collections)
    assert n_paths_asc > 20
    plt.close(fig)





