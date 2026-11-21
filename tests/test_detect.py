"""Tests for universal format auto-detection (detect.py) and GUI external-open dispatch."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import synthetic

import pymorgan as pm
from pymorgan.detect import FormatMatch, detect_format, load_auto
from pymorgan.oneD.dataset import Dataset1D
from pymorgan.oneD.load import save_1D_hdf5
from pymorgan.twoD.dataset import Dataset2D
from tests.test_twoD_uos_ral import (
    RAL_PROC_DIR,
    RAL_RAW_DIR,
    UOS_2DIR_GOOD,
    _make_synthetic_ral_proc,
    _make_synthetic_ral_raw,
    _make_synthetic_uos_2dir,
)
from tests.test_ufs_interop import HELIOS_REAL_DIR, _write_synthetic_ufs

TESTDATA_ROOT = Path(r"c:\Users\ricar\switchdrive\Ambizione UniGE\Scripts\testData")


def test_detect_missing_path_returns_none_or_empty(tmp_path):
    missing = tmp_path / "does_not_exist"
    assert detect_format(missing) is None
    assert detect_format(missing, return_all=True) == []


def test_detect_and_load_auto_1d_and_2d_formats(tmp_path):
    pdat_dir = tmp_path / "pdat_dir"
    pdat_dir.mkdir()
    pdat_info = synthetic.make_synthetic_pdat(pdat_dir / "sample.pdat")
    pdat_match = detect_format(pdat_info["path"])
    assert isinstance(pdat_match, FormatMatch)
    assert pdat_match.dimension == "1D" and pdat_match.data_type == "PDAT"
    synthetic.make_synthetic_pdatn(pdat_info["path"])
    assert (
        detect_format(pdat_dir / "sample.pdatn").resolved_path == Path(pdat_info["path"]).resolve()
    )
    assert isinstance(load_auto(pdat_info["path"]), Dataset1D)

    p2_dir = tmp_path / "p2dat_dir"
    p2_dir.mkdir()
    p2_info = synthetic.make_synthetic_p2dat(p2_dir / "map.p2dat")
    p2_match = detect_format(p2_info["path"])
    assert p2_match.dimension == "2D" and p2_match.data_type == "P2DAT"
    assert isinstance(load_auto(p2_info["path"]), Dataset2D)

    ufs_dir = tmp_path / "ufs_only"
    ufs_dir.mkdir()
    ufs_file = _write_synthetic_ufs(ufs_dir / "meas.ufs")
    assert detect_format(ufs_file).data_type == "Helios_UFS"
    assert detect_format(ufs_dir).data_type == "Helios_UFS"
    assert isinstance(load_auto(ufs_dir), Dataset1D)

    ds1d = load_auto(pdat_info["path"])
    ascii_file = ds1d.to_glotaran_ascii(tmp_path / "glot.ascii")
    assert detect_format(ascii_file).data_type == "Glotaran_ASCII"
    assert isinstance(load_auto(ascii_file), Dataset1D)

    nc_file = ds1d.to_glotaran_nc(tmp_path / "glot.nc")
    assert detect_format(nc_file).data_type == "Glotaran_NC"
    assert isinstance(load_auto(nc_file), Dataset1D)

    h5_1d = save_1D_hdf5(ds1d, tmp_path / "fair_1d.h5")
    assert detect_format(h5_1d).data_type == "HDF5"
    assert isinstance(load_auto(h5_1d), Dataset1D)

    ds2d = load_auto(p2_info["path"])
    h5_2d = ds2d.to_hdf5(tmp_path / "fair_2d.h5")
    assert detect_format(h5_2d).data_type == "HDF5_2D"
    assert isinstance(load_auto(h5_2d), Dataset2D)

    pymess_pp = synthetic.make_synthetic_pymess_pp(tmp_path / "pm_pp.h5")
    assert detect_format(pymess_pp["path"]).data_type == "PyMESS_PP"

    qc_dir = tmp_path / "qc_2d_scan"
    synthetic.make_synthetic_quickcontrol_2d(qc_dir, prefix="qc_2d_scan")
    assert detect_format(qc_dir / "qc_2d_scan.info").data_type == "QuickControl_2D"
    assert detect_format(qc_dir).data_type == "QuickControl_2D"


def test_detect_disambiguation_uos_ral_steady_project_recipe(tmp_path):
    uos_dir = _make_synthetic_uos_2dir(tmp_path / "u_2d", delays_ps=[0.0, 2.0])
    m_u = detect_format(uos_dir)
    assert m_u is not None and m_u.dimension == "2D" and m_u.data_type == "UoS_2DIR"
    slice_file = next(uos_dir.glob("*_D0_*.2D"))
    assert detect_format(slice_file).data_type == "UoS_2DIR"

    ral_raw = _make_synthetic_ral_raw(tmp_path / "r_raw")
    m_raw = detect_format(ral_raw)
    assert m_raw.dimension == "2D" and m_raw.data_type == "RAL_RAW"
    assert detect_format(ral_raw / "Run 0 Avg Diff spec 01.csv").data_type == "RAL_IRpp"

    ral_proc = _make_synthetic_ral_proc(tmp_path / "r_proc")
    assert detect_format(ral_proc).data_type == "RAL_Proc"
    assert detect_format(ral_proc / "a_500fs_2DIR.csv").data_type == "RAL_Proc"

    ss_csv = tmp_path / "abs_spec.csv"
    ss_csv.write_text("450.0,0.12\n451.0,0.15\n452.0,0.18\n", encoding="utf-8")
    m_ss = detect_format(ss_csv)
    assert m_ss.dimension == "steady_state" and m_ss.data_type == "csv"
    spec = load_auto(ss_csv)
    assert len(spec.x) == 3

    pmproj = tmp_path / "study.pmproj"
    pmproj.write_text("{}", encoding="utf-8")
    m_proj = detect_format(pmproj)
    assert m_proj == FormatMatch(
        "project", "PMPROJ", 1.0, "PyMORGAN project file (.pmproj)", pmproj.resolve()
    )

    pmrecipe = tmp_path / "default.pmrecipe.toml"
    pmrecipe.write_text("version = 1\n", encoding="utf-8")
    m_rec = detect_format(pmrecipe)
    assert m_rec == FormatMatch(
        "recipe", "PMRECIPE", 1.0, "PyMORGAN recipe file (.pmrecipe.toml)", pmrecipe.resolve()
    )


def test_gui_open_external_path_tie_dialog_and_formats(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QInputDialog

    from pymorgan.gui.batch_dialog import BatchProcessDialog
    from pymorgan.gui.main_window import MainWindow
    from pymorgan.gui.mw_common import get_combo_datatype
    from pymorgan.project import ProjectSession
    from pymorgan.recipe import Recipe

    plain_dir = tmp_path / "plain_browse"
    plain_dir.mkdir()
    ufs_dir = tmp_path / "ufs_drop"
    ufs_dir.mkdir()
    ufs_file = _write_synthetic_ufs(ufs_dir / "dropped.ufs")

    twoD_dir = tmp_path / "twoD_dir"
    twoD_dir.mkdir()
    p2_info = synthetic.make_synthetic_p2dat(twoD_dir / "dropped_2d.p2dat")
    ds2d = pm.load_2D(p2_info["path"], data_type="P2DAT")
    h5_2d = ds2d.to_hdf5(tmp_path / "twoD_dir" / "dropped_2d.h5")

    ss_csv = tmp_path / "uvvis.csv"
    ss_csv.write_text("Wavelength,Abs\n400,0.1\n450,0.3\n500,0.2\n", encoding="utf-8")
    proj = tmp_path / "workspace.pmproj"
    ProjectSession().save(proj)
    recipe = tmp_path / "pipe.pmrecipe.toml"
    Recipe(name="x", dimension="1D", steps=[]).save(recipe)

    opened_recipes: list[str] = []
    monkeypatch.setattr(
        BatchProcessDialog,
        "exec",
        lambda self: (opened_recipes.append(self._recipe.name if self._recipe else ""), 1)[1],
    )

    win = MainWindow()
    try:
        assert win.open_external_path(plain_dir) is True
        assert Path(win.RootDir_field.text()).resolve() == plain_dir.resolve()

        assert win.open_external_path(ufs_file) is True
        assert win.MainTabs.currentIndex() == 0
        assert get_combo_datatype(win.PP_datatype_cbx, "") == "Helios_UFS"
        assert win.dataset is not None

        assert win.open_external_path(h5_2d) is True
        assert win.MainTabs.currentIndex() == 1
        assert get_combo_datatype(win.twoD_datatype_cbx, "") == "HDF5_2D"
        assert win.twoD_dataset is not None

        assert win.open_external_path(ss_csv) is True
        assert win._ss_abs_spectrum is not None
        assert win.MainTabs.currentIndex() == 0

        assert win.open_external_path(proj) is True
        assert win.open_external_path(recipe) is True
        assert opened_recipes == ["x"]

        # Top-confidence tie prompts QInputDialog.getItem and routes to chosen format
        tied_calls: list[str] = []
        monkeypatch.setattr(
            pm,
            "detect_format",
            lambda p, return_all=False: (
                [
                    FormatMatch("1D", "Helios_UFS", 0.90, "Option A", ufs_file),
                    FormatMatch("2D", "HDF5_2D", 0.90, "Option B", h5_2d),
                ]
                if return_all
                else FormatMatch("1D", "Helios_UFS", 0.90, "Option A", ufs_file)
            ),
        )

        def _fake_item(parent, title, label, items, current, editable):
            tied_calls.extend(items)
            return items[1], True

        monkeypatch.setattr(QInputDialog, "getItem", _fake_item)
        assert win.open_external_path(ufs_file) is True
        assert len(tied_calls) == 2
        assert win.MainTabs.currentIndex() == 1
        assert get_combo_datatype(win.twoD_datatype_cbx, "") == "HDF5_2D"
    finally:
        win.close()


# --------------------------------------------------------------------------- #
#                         Auto-Skipping Real Data Tests                       #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not TESTDATA_ROOT.is_dir(), reason="Switchdrive testData absent")
def test_real_testdata_auto_detection():
    if (HELIOS_REAL_DIR / "3-RF47-MeCN-1000nJ.ufs").is_file():
        assert detect_format(HELIOS_REAL_DIR / "3-RF47-MeCN-1000nJ.ufs").data_type == "Helios_UFS"
    if HELIOS_REAL_DIR.is_dir():
        assert detect_format(HELIOS_REAL_DIR).data_type == "Helios_TA"
    if UOS_2DIR_GOOD.is_dir():
        m = detect_format(UOS_2DIR_GOOD)
        assert m.dimension == "2D" and m.data_type == "UoS_2DIR"
    uos_1d = TESTDATA_ROOT / "UoS" / "20220505 - IrH SEC Good" / "IrHCOP3_DCM_OTTLE_+700mV_pp_20k"
    if uos_1d.is_dir():
        m1 = detect_format(uos_1d)
        assert m1.dimension == "1D" and m1.data_type == "UoS_IRpp"
    ss_real = TESTDATA_ROOT / "Spectra" / "CSV" / "C153-EtOH_dil_spectra.csv"
    if ss_real.is_file():
        assert detect_format(ss_real).dimension == "steady_state"
    qc_real = TESTDATA_ROOT / "QuickControl" / "TillStensitzki" / "MeSCN_H2O"
    if qc_real.is_dir():
        m_qc = detect_format(qc_real)
        assert m_qc is not None and "QuickControl" in m_qc.data_type

    if RAL_RAW_DIR.is_dir():
        assert detect_format(RAL_RAW_DIR).data_type == "RAL_RAW"
    if RAL_PROC_DIR.is_dir():
        assert detect_format(RAL_PROC_DIR).data_type == "RAL_Proc"


def test_detect_unrecognized_nc_without_data_ignored(tmp_path):
    empty_nc = tmp_path / "empty.nc"
    with h5py.File(empty_nc, "w") as f:
        f.create_dataset("metadata_only", data=np.array([1.0, 2.0]))
    assert detect_format(empty_nc) is None
