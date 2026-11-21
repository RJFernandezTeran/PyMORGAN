"""Cross-track integration tests uniting Item 1 (reproducibility/workflows) and Item 4 (loaders/interop)."""

from __future__ import annotations

import numpy as np
import synthetic

import pymorgan as pm
from pymorgan.batch import discover_batch_inputs, run_batch
from pymorgan.gui.batch_dialog import BatchProcessDialog
from pymorgan.gui.main_window import MainWindow
from pymorgan.recipe import Recipe
from tests.test_twoD_uos_ral import _make_synthetic_ral_proc
from tests.test_ufs_interop import _write_synthetic_ufs


def test_fair_hdf5_history_roundtrip_and_script_export_1d_and_2d(tmp_path):
    # --- 1D path: PDAT -> processing ops -> FAIR HDF5 v2.0 -> load_auto & HDF5 -> script ---
    pdat_info = synthetic.make_synthetic_pdat(tmp_path / "sample1d.pdat", baseline=0.06)
    ds1 = pm.load_1D(pdat_info["path"], data_type="PDAT")
    ds1.background_correct(tmin=-20.0, tmax=-5.0, do_correct=True)
    ds1.shift_t0(0.3)
    ds1.mask_probe_regions([(1950.0, 1955.0)])

    h5_1d = ds1.to_hdf5(tmp_path / "fair_1d.h5")
    for loader_fn in (
        lambda p: pm.load_auto(p),
        lambda p: pm.load_1D(p, data_type="HDF5"),
    ):
        reloaded1 = loader_fn(h5_1d)
        assert [s.op for s in reloaded1.history] == [
            "background_correct",
            "shift_t0",
            "mask_probe_regions",
        ]
        assert np.allclose(reloaded1.Z, ds1.Z, equal_nan=True)
        assert np.allclose(reloaded1.delays, ds1.delays, equal_nan=True)

        script1 = reloaded1.to_python_script()
        compile(script1, "<fair_1d_history_script>", "exec")
        assert "background_correct(" in script1
        assert "shift_t0(" in script1
        assert "mask_probe_regions(" in script1

    # --- 2D path: multi-detector RAL_Proc -> detector switch + bkg -> HDF5_2D -> load_auto & HDF5_2D -> script ---
    ral_dir = _make_synthetic_ral_proc(tmp_path / "ral_2d")
    ds2 = pm.load_2D(ral_dir, data_type="RAL_Proc")
    assert ds2.active_detector == "DET1"
    assert ds2.set_active_detector("DET2") is True
    ds2.background_correct(reference=0, do_correct=True)

    h5_2d = ds2.to_hdf5(tmp_path / "fair_2d.h5")
    for loader_fn in (
        lambda p: pm.load_auto(p),
        lambda p: pm.load_2D(p, data_type="HDF5_2D"),
    ):
        reloaded2 = loader_fn(h5_2d)
        assert reloaded2.active_detector == "DET2"
        assert [s.op for s in reloaded2.history] == [
            "set_active_detector",
            "background_correct",
        ]
        assert np.allclose(reloaded2.Z, ds2.Z, equal_nan=True)

        script2 = reloaded2.to_python_script()
        compile(script2, "<fair_2d_history_script>", "exec")
        assert "set_active_detector('DET2')" in script2
        assert "background_correct(" in script2


def test_twoD_multi_detector_undo_redo_and_recipe_replay(tmp_path):
    ral_dir = _make_synthetic_ral_proc(tmp_path / "ral_undo")
    ds = pm.load_2D(ral_dir, data_type="RAL_Proc")
    assert ds.active_detector == "DET1"
    z_det1 = np.array(ds.Z, copy=True)
    probe_det1 = np.array(ds.probe, copy=True)

    assert ds.set_active_detector("DET2") is True
    assert ds.active_detector == "DET2"
    assert [s.op for s in ds.history] == ["set_active_detector"]
    z_det2 = np.array(ds.Z, copy=True)
    probe_det2 = np.array(ds.probe, copy=True)
    assert not np.allclose(probe_det1, probe_det2)

    ds.background_correct(reference=0, do_correct=True)
    z_corr = np.array(ds.Z, copy=True)
    assert [s.op for s in ds.history] == ["set_active_detector", "background_correct"]

    # Undo background_correct -> still DET2 with un-corrected DET2 cube
    assert ds.undo() is True
    assert ds.active_detector == "DET2"
    assert np.allclose(ds.Z, z_det2, equal_nan=True)

    # Undo set_active_detector -> back to DET1 probe & cube
    assert ds.undo() is True
    assert ds.active_detector == "DET1"
    assert np.allclose(ds.probe, probe_det1, equal_nan=True)
    assert np.allclose(ds.Z, z_det1, equal_nan=True)
    assert ds.undo() is False

    # Redo both steps -> identical DET2 corrected state
    assert ds.redo() is True
    assert ds.active_detector == "DET2"
    assert np.allclose(ds.probe, probe_det2, equal_nan=True)
    assert ds.redo() is True
    assert np.allclose(ds.Z, z_corr, equal_nan=True)

    # Save & reload .pmrecipe.toml, then replay onto a fresh RAL_Proc dataset
    recipe_path = tmp_path / "ral_det2.pmrecipe.toml"
    recipe = Recipe.from_dataset(ds, name="ral_det2_recipe", export_formats=["hdf5", "p2dat"])
    recipe.save(recipe_path)

    reloaded_recipe = Recipe.load(recipe_path)
    fresh = pm.load_2D(ral_dir, data_type="RAL_Proc")
    reloaded_recipe.apply(fresh)
    assert fresh.active_detector == "DET2"
    assert np.allclose(fresh.Z, z_corr, equal_nan=True)


def test_batch_auto_discovery_and_execution_across_item4_formats(tmp_path):
    mixed = tmp_path / "mixed_inputs"
    mixed.mkdir()

    base_info = synthetic.make_synthetic_pdat(tmp_path / "seed.pdat", baseline=0.05)
    seed_ds = pm.load_1D(base_info["path"], data_type="PDAT")
    seed_ds.to_glotaran_ascii(mixed / "spec_a.ascii")
    seed_ds.to_glotaran_nc(mixed / "spec_b.nc")
    seed_ds.to_hdf5(mixed / "spec_c.hdf5")
    _write_synthetic_ufs(mixed / "meas_d.ufs")
    _write_synthetic_ufs(mixed / "meas_d_scan1.ufs")
    ral_dir = _make_synthetic_ral_proc(mixed)

    found_1d = discover_batch_inputs(mixed, data_type="auto", dimension="1D")
    names_types_1d = {(p.name, dt) for p, dt in found_1d}
    assert names_types_1d == {
        ("spec_a.ascii", "Glotaran_ASCII"),
        ("spec_b.nc", "Glotaran_NC"),
        ("spec_c.hdf5", "HDF5"),
        ("meas_d.ufs", "Helios_UFS"),
    }

    found_2d = discover_batch_inputs(mixed, data_type="auto", dimension="2D")
    assert [(p.resolve(), dt) for p, dt in found_2d] == [(ral_dir.resolve(), "RAL_Proc")]

    out_1d = tmp_path / "batch_out_1d"
    rec_1d = Recipe(
        name="interop_1d",
        dimension="1D",
        data_type="auto",
        steps=[
            {
                "op": "background_correct",
                "enabled": True,
                "params": {"tmin": -1.0, "tmax": 0.0, "do_correct": True},
            }
        ],
        export_formats=["hdf5", "glotaran_ascii", "glotaran_nc"],
        export_suffix="_BATCH",
    )
    rep_1d = run_batch(found_1d, rec_1d, out_1d, workers=1, on_error="stop")
    assert rep_1d.total == 4
    assert rep_1d.succeeded == 4
    assert rep_1d.failed == 0
    for item in rep_1d.items:
        assert item.status == "ok"
        assert len(item.output_files) == 3
        for out_file in item.output_files:
            loaded = pm.load_auto(out_file)
            assert loaded.Z is not None

    out_2d = tmp_path / "batch_out_2d"
    rec_2d = Recipe(
        name="interop_2d",
        dimension="2D",
        data_type="auto",
        steps=[
            {
                "op": "set_active_detector",
                "enabled": True,
                "params": {"name": "DET2"},
            },
            {
                "op": "background_correct",
                "enabled": True,
                "params": {"reference": 0, "do_correct": True},
            },
        ],
        export_formats=["hdf5", "p2dat"],
        export_suffix="_DET2",
    )
    rep_2d = run_batch(found_2d, rec_2d, out_2d, workers=1, on_error="stop")
    assert rep_2d.total == 1
    assert rep_2d.succeeded == 1
    assert rep_2d.failed == 0
    h5_out = out_2d / f"{ral_dir.name}_DET2.hdf5"
    p2_out = out_2d / f"{ral_dir.name}_DET2.p2dat"
    assert h5_out.is_file()
    assert p2_out.is_file()
    reloaded_batch_2d = pm.load_auto(h5_out)
    assert reloaded_batch_2d.active_detector == "DET2"


def test_gui_multidetector_undo_redo_pmproj_and_recipe_drop(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    ral_dir = _make_synthetic_ral_proc(tmp_path / "gui_exp")
    proj_path = tmp_path / "det2_session.pmproj"
    recipe_path = tmp_path / "det2_gui.pmrecipe.toml"

    win = MainWindow(init_rootdir=False)
    try:
        win.MainTabs.setCurrentIndex(1)
        win.twoD_load_path(str(ral_dir), data_type="RAL_Proc")
        assert win.twoD_dataset is not None
        assert win.twoD_dataset.active_detector == "DET1"
        assert getattr(win, "_twoD_detector_cbx", None) is not None
        assert win._twoD_detector_cbx.currentText() == "DET1"

        win._twoD_detector_cbx.setCurrentText("DET2")
        assert win.twoD_dataset.active_detector == "DET2"
        assert win.actionUndo.isEnabled()

        win._edit_undo()
        assert win.twoD_dataset.active_detector == "DET1"
        assert win._twoD_detector_cbx.currentText() == "DET1"
        assert win.actionRedo.isEnabled()

        win._edit_redo()
        assert win.twoD_dataset.active_detector == "DET2"
        assert win._twoD_detector_cbx.currentText() == "DET2"

        monkeypatch.setattr(
            QFileDialog,
            "getSaveFileName",
            lambda *_a, **_k: (str(proj_path), "PyMORGAN Projects (*.pmproj)"),
        )
        win._save_project()
        assert proj_path.is_file()

        Recipe.from_dataset(win.twoD_dataset, name="det2_gui").save(recipe_path)
    finally:
        win.close()

    pm.set_settings(pm.Settings())
    captured_recipes: list[str] = []
    monkeypatch.setattr(
        BatchProcessDialog,
        "exec",
        lambda self: (captured_recipes.append(self._recipe.name if self._recipe else ""), 1)[1],
    )

    win2 = MainWindow(init_rootdir=False)
    try:
        assert win2.open_external_path(proj_path) is True
        assert win2.twoD_dataset is not None
        assert win2.twoD_dataset.active_detector == "DET2"
        assert win2._twoD_detector_cbx.currentText() == "DET2"

        assert win2.open_external_path(recipe_path) is True
        assert captured_recipes == ["det2_gui"]
    finally:
        win2.close()
