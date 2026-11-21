"""Tests for Milestone 4: processing provenance, undo/redo, Python export, and GUI."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from synthetic import (
    make_synthetic_p2dat,
    make_synthetic_pdat,
    make_synthetic_pymess_pp,
    make_synthetic_quickcontrol_2d,
)

import pymorgan as pm
from pymorgan.oneD.chirp import ChirpFit, save_chirp_fit
from pymorgan.provenance import ProcessingStep, generate_python_script


def test_1d_history_undo_redo_reset_and_revert(tmp_path):
    pdat_path = tmp_path / "sample.pdat"
    make_synthetic_pdat(pdat_path, baseline=0.05)

    ds = pm.load_1D(pdat_path, data_type="PDAT")
    raw_z = np.array(ds.Z, copy=True)
    raw_delays = np.array(ds.delays, copy=True)
    assert ds.history == []
    assert not ds.can_undo
    assert not ds.can_redo

    ds.background_correct(-20.0, -5.0, do_correct=True)
    z_after_bkg = np.array(ds.Z, copy=True)
    ds.mask_probe_regions([(1950.0, 1960.0)])
    z_after_mask = np.array(ds.Z, copy=True)
    ds.shift_t0(0.5)
    z_after_shift = np.array(ds.Z, copy=True)
    delays_after_shift = np.array(ds.delays, copy=True)

    assert [s.op for s in ds.history] == [
        "background_correct",
        "mask_probe_regions",
        "shift_t0",
    ]
    assert ds.can_undo
    assert not ds.can_redo

    # Undo shift_t0 -> back to masked
    assert ds.undo()
    assert len(ds.history) == 2
    assert ds.can_redo
    assert np.allclose(ds.delays, raw_delays, equal_nan=True)
    assert np.allclose(ds.Z, z_after_mask, equal_nan=True)

    # Redo restores shift_t0
    assert ds.redo()
    assert len(ds.history) == 3
    assert np.allclose(ds.delays, delays_after_shift, equal_nan=True)
    assert np.allclose(ds.Z, z_after_shift, equal_nan=True)

    # Undo twice, then new op clears redo stack
    assert ds.undo()
    assert ds.undo()
    assert np.allclose(ds.Z, z_after_bkg, equal_nan=True)
    assert ds.can_redo
    ds.shift_t0(1.0)
    assert not ds.can_redo
    assert [s.op for s in ds.history] == ["background_correct", "shift_t0"]

    # revert_to_step(0) returns to post-background state
    ds.revert_to_step(0)
    assert len(ds.history) == 1
    assert ds.history[0].op == "background_correct"
    assert np.allclose(ds.Z, z_after_bkg, equal_nan=True)
    assert np.allclose(ds.delays, raw_delays, equal_nan=True)

    # reset_to_raw restores unmodified baseline and clears stacks
    ds.reset_to_raw()
    assert ds.history == []
    assert not ds.can_undo
    assert np.allclose(ds.Z, raw_z, equal_nan=True)
    assert not ds.can_redo


def test_1d_coalescing_reentrancy_and_bounded_snapshots(tmp_path):
    h5_path = tmp_path / "pymess.h5"
    make_synthetic_pymess_pp(
        h5_path, n_scans=3, n_delays=20, n_pixels=32, with_individual_scans=True
    )
    ds = pm.load_1D(h5_path, data_type="PyMESS_PP")
    assert ds.Zss_R is not None

    # 1. Coalescing repeated background_correct calls into a single step
    ds.background_correct(-2.0, -0.5, do_correct=True)
    ds.background_correct(-3.0, -1.0, do_correct=True)
    ds.background_correct(-2.5, -0.8, do_correct=True)
    assert len(ds.history) == 1
    assert ds.history[0].params["tmin"] == pytest.approx(-2.5)
    assert ds.history[0].params["tmax"] == pytest.approx(-0.8)

    # Memory-smart check: snapshot holds reference to 4D/3D Zss_R without copying it
    _, undo_snap = ds._history._undo_stack[-1]
    assert undo_snap["Zss_R"] is ds.Zss_R

    # Undo back to raw in one step
    assert ds.undo()
    assert ds.history == []

    # 2. Compound remove_scatter -> apply_scatter_removal records ONLY outer semantic op
    ds.remove_scatter(tmin=-2.0, tmax=-0.2, method="local_svd", rank=2)
    assert [s.op for s in ds.history] == ["remove_scatter"]

    # 3. Bounded max_snapshots=12
    ds.reset_to_raw()
    for _ in range(16):
        ds.shift_t0(0.05)
    assert len(ds.history) == 16
    assert len(ds._history._undo_stack) == 12
    # Undoing 12 times succeeds; 13th returns False, but reset_to_raw still works cleanly
    undone = 0
    while ds.undo():
        undone += 1
    assert undone == 12
    ds.reset_to_raw()
    assert ds.history == []


def test_copy_and_history_dict_serialization(tmp_path):
    pdat_path = tmp_path / "s.pdat"
    make_synthetic_pdat(pdat_path)
    ds = pm.load_1D(pdat_path, data_type="PDAT")
    ds.background_correct(-20.0, -5.0)
    ds.shift_t0(0.25)

    copied = ds.copy()
    assert len(copied.history) == 2
    assert copied.undo()
    assert len(copied.history) == 1
    assert len(ds.history) == 2  # original unaffected

    dicts = ds.history_to_dicts()
    assert len(dicts) == 2
    ds2 = pm.load_1D(pdat_path, data_type="PDAT")
    ds2.load_history_from_dicts(dicts)
    assert [s.op for s in ds2.history] == ["background_correct", "shift_t0"]
    assert isinstance(ProcessingStep.from_dict(dicts[0]), ProcessingStep)


def test_2d_history_undo_redo_and_memory_smart_snapshots(tmp_path):
    p2_path = tmp_path / "map.p2dat"
    make_synthetic_p2dat(p2_path, delays=(-1.0, 0.5, 2.0, 5.0))
    ds2 = pm.load_2D(p2_path, data_type="P2DAT")
    raw_z = np.array(ds2.Z, copy=True)

    # Coalesced background_correct
    ds2.background_correct(reference=0, do_correct=True)
    ds2.background_correct(reference=0, do_correct=True)
    assert len(ds2.history) == 1

    # subtract_spectrum + reset_subtraction
    ref_ds = pm.load_2D(p2_path, data_type="P2DAT")
    ds2.subtract_spectrum(ref_ds, scale=0.25, ref_t2_idx=0)
    z_sub = np.array(ds2.Z, copy=True)
    ds2.reset_subtraction()
    assert [s.op for s in ds2.history] == [
        "background_correct",
        "subtract_spectrum",
        "reset_subtraction",
    ]
    assert ds2.undo()
    assert np.allclose(ds2.Z, z_sub, equal_nan=True)
    assert ds2.redo()
    ds2.reset_to_raw()
    assert np.allclose(ds2.Z, raw_z, equal_nan=True)

    # QuickControl 2D: process coalescing & zero-copy sharing of raw_signal_ss
    qc_dir = tmp_path / "qc2d"
    qc_dir.mkdir()
    qc_info = make_synthetic_quickcontrol_2d(qc_dir, n_scans=2, n_t1=25, n_pixels=16)
    qc_ds = pm.load_2D(qc_info["info_path"], data_type="QuickControl_2D")
    qc_ds.process(zeropad_factor=2)
    qc_ds.process(zeropad_factor=4)
    assert len(qc_ds.history) == 1
    assert qc_ds.history[0].op == "process"
    assert qc_ds.history[0].params["zeropad_factor"] == 4
    _, snap = qc_ds._history._undo_stack[-1]
    if "raw_signal_ss" in snap and qc_ds.raw_signal_ss is not None:
        assert snap["raw_signal_ss"] is qc_ds.raw_signal_ss


def test_generate_python_script_reproduces_1d_and_passes_ruff(tmp_path):
    pdat_path = tmp_path / "main.pdat"
    sol_path = tmp_path / "solvent.pdat"
    make_synthetic_pdat(pdat_path, baseline=0.04)
    make_synthetic_pdat(sol_path, amps=(0.1, 0.05), baseline=0.01, seed=7)

    ds = pm.load_1D(pdat_path, data_type="PDAT")
    sol = pm.load_1D(sol_path, data_type="PDAT")
    ds.background_correct(-20.0, -5.0, do_correct=True)
    ds.mask_probe_regions([(1950.0, 1958.0)])
    ds.shift_t0(0.3)

    # Saved chirp file + solvent subtraction + shockwave subtraction
    fit = ChirpFit(
        mode="manual",
        coeffs=np.array([0.1, 200.0]),
        fit_wl=np.linspace(1920.0, 1990.0, 10),
        Pfit=np.empty((0, 1)),
        n_cauchy_terms=2,
        lambda_ref=1955.0,
        detector=0,
    )
    chirp_json = tmp_path / "chirp.json"
    save_chirp_fit(fit, chirp_json)
    ds.apply_chirp_correction(fit)
    ds.subtract_solvent(sol, dt=0.1, scale=0.5)
    ds.subtract_shockwave(pixels=[0, 1], one_based=False)

    script_text = generate_python_script(ds, include_plot=False, include_settings=True)
    script_file = tmp_path / "reproduce_1d.py"
    script_file.write_text(script_text, encoding="utf-8")

    # Also test inline ChirpFit fallback when chirp was never saved to disk
    ds_inline = pm.load_1D(pdat_path, data_type="PDAT")
    fit_unsaved = ChirpFit(
        mode="manual",
        coeffs=np.array([0.05, 100.0]),
        fit_wl=np.linspace(1920.0, 1990.0, 10),
        Pfit=np.empty((0, 1)),
        n_cauchy_terms=2,
        lambda_ref=1955.0,
        detector=0,
    )
    ds_inline.apply_chirp_correction(fit_unsaved)
    inline_script = ds_inline.to_python_script(include_plot=False, include_settings=False)
    inline_file = tmp_path / "reproduce_inline.py"
    inline_file.write_text(inline_script, encoding="utf-8")

    # Validate syntax & Ruff cleanliness (E, F, W, B, UP)
    ruff_exe = Path(sys.executable).with_name("ruff.exe")
    if not ruff_exe.exists():
        ruff_exe = Path(sys.executable).with_name("ruff")
    if ruff_exe.exists():
        chk = subprocess.run(
            [
                str(ruff_exe),
                "check",
                "--select",
                "E,F,I,W,B,UP",
                "--ignore",
                "E501,UP031",
                str(script_file),
                str(inline_file),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert chk.returncode == 0, f"Ruff failed on generated script:\n{chk.stdout}\n{chk.stderr}"

    # Execute generated script in isolated subprocess & compare Z against original ds.Z
    verify_code = (
        f"import runpy, numpy as np\n"
        f"ns = runpy.run_path({str(script_file)!r})\n"
        f"ref = np.load({str(tmp_path / 'ref_z.npy')!r})\n"
        f"assert np.allclose(ns['ds'].Z, ref, equal_nan=True)\n"
        f"ns_in = runpy.run_path({str(inline_file)!r})\n"
        f"ref_in = np.load({str(tmp_path / 'ref_in_z.npy')!r})\n"
        f"assert np.allclose(ns_in['ds'].Z, ref_in, equal_nan=True)\n"
    )
    np.save(tmp_path / "ref_z.npy", np.asarray(ds.Z))
    np.save(tmp_path / "ref_in_z.npy", np.asarray(ds_inline.Z))

    env = dict(os.environ)
    repo_src = str(Path(pm.__file__).resolve().parent.parent)
    env["PYTHONPATH"] = repo_src
    env["MPLBACKEND"] = "Agg"
    res = subprocess.run(
        [sys.executable, "-c", verify_code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"Generated script execution failed:\n{res.stdout}\n{res.stderr}"


def test_gui_edit_menu_undo_redo_preview_safety_and_history_dialog(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    from pymorgan.gui.history_dialog import ProcessingHistoryDialog
    from pymorgan.gui.main_window import MainWindow

    pdat_path = tmp_path / "gui_sample.pdat"
    make_synthetic_pdat(pdat_path, baseline=0.05)

    win = MainWindow(init_rootdir=False)
    try:
        # Edit menu is positioned before menuPreProcessing
        bar_actions = win.menuBar().actions()
        edit_idx = bar_actions.index(win.menuEdit.menuAction())
        pre_idx = bar_actions.index(win.menuPreProcessing.menuAction())
        assert edit_idx < pre_idx

        win.load_path(str(pdat_path), data_type="PDAT")
        assert win.dataset.history == []
        assert not win.actionUndo.isEnabled()

        # Toggle background subtraction via UI control
        win.PP_SubtactBkg_chk.setChecked(True)
        assert len(win.dataset.history) == 1
        assert win.actionUndo.isEnabled()
        assert "Background" in win.actionUndo.text()

        # Perform mask_probe_regions and shift_t0, then verify _preview_contour doesn't clobber history!
        win.dataset.mask_probe_regions([(1950.0, 1955.0)])
        win._sync_oneD_ui_after_history_change()
        z_masked = np.array(win.dataset.Z, copy=True)

        # Trigger preview redraw (simulating Zscale/cmap change) -> history preserved!
        win._preview_contour()
        assert [s.op for s in win.dataset.history] == ["background_correct", "mask_probe_regions"]
        assert np.allclose(win.dataset.Z, z_masked, equal_nan=True)

        # Undo via QAction -> restores post-background state & clears redo check after redo
        win.actionUndo.trigger()
        assert len(win.dataset.history) == 1
        assert win.actionRedo.isEnabled()

        # Preview redraw MUST NOT clear _redo_stack!
        win._preview_contour()
        assert win.actionRedo.isEnabled()

        win.actionRedo.trigger()
        assert len(win.dataset.history) == 2
        assert np.allclose(win.dataset.Z, z_masked, equal_nan=True)

        # History dialog actions
        dlg = ProcessingHistoryDialog(
            win.dataset,
            win,
            on_reverted=win._sync_oneD_ui_after_history_change,
            default_dir=str(tmp_path),
        )
        assert dlg.step_list.count() == 3  # Raw + 2 steps
        dlg.step_list.setCurrentRow(1)
        assert dlg.btn_revert.isEnabled()
        dlg.btn_revert.click()
        assert len(win.dataset.history) == 1

        saved_py = tmp_path / "exported.py"
        monkeypatch.setattr(
            QFileDialog,
            "getSaveFileName",
            lambda *_args, **_kw: (str(saved_py), "Python Scripts (*.py)"),
        )
        dlg._save_script()
        assert saved_py.is_file()
        assert "background_correct" in saved_py.read_text(encoding="utf-8")

        saved_rec = tmp_path / "from_dlg.pmrecipe.toml"
        monkeypatch.setattr(
            QFileDialog,
            "getSaveFileName",
            lambda *_args, **_kw: (str(saved_rec), "Recipes (*.pmrecipe.toml)"),
        )
        dlg._save_recipe()
        assert saved_rec.is_file()
        assert "background_correct" in saved_rec.read_text(encoding="utf-8")
        dlg.close()
    finally:
        win.close()
