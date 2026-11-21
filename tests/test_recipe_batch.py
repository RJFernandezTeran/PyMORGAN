"""Tests for Milestone 5: .pmrecipe.toml recipes, batch processor (API/CLI), and GUI dialog."""

from __future__ import annotations

import csv
import json
import threading
from pathlib import Path

import numpy as np
from synthetic import (
    make_synthetic_p2dat,
    make_synthetic_pdat,
    make_synthetic_pymess_pp,
)

import pymorgan as pm
from pymorgan.batch import cli_main, discover_batch_inputs, run_batch
from pymorgan.launcher import main as launcher_main
from pymorgan.oneD.chirp import ChirpFit, save_chirp_fit
from pymorgan.recipe import Recipe


def test_recipe_save_load_relative_aux_paths_and_apply(tmp_path):
    orig_dir = tmp_path / "orig"
    aux_dir = orig_dir / "aux"
    aux_dir.mkdir(parents=True)

    raw_pdat = orig_dir / "scan1.pdat"
    sol_pdat = aux_dir / "solvent.pdat"
    chirp_json = aux_dir / "chirp.json"
    cal_csv = aux_dir / "cal.csv"
    make_synthetic_pdat(raw_pdat, baseline=0.05)
    make_synthetic_pdat(sol_pdat, amps=(0.05, 0.02), baseline=0.01, seed=9)

    fit = ChirpFit(
        mode="manual",
        coeffs=np.array([0.08, 150.0]),
        fit_wl=np.linspace(1920.0, 1990.0, 10),
        Pfit=np.empty((0, 1)),
        n_cauchy_terms=2,
        lambda_ref=1955.0,
        detector=0,
    )
    save_chirp_fit(fit, chirp_json)

    ds = pm.load_1D(raw_pdat, data_type="PDAT")
    cal_probe = ds.probe + 1.5
    np.savetxt(cal_csv, cal_probe, delimiter=",")
    ds.probe = cal_probe.copy()
    sol = pm.load_1D(sol_pdat, data_type="PDAT")

    ds.background_correct(-20.0, -5.0, do_correct=True)
    ds.apply_chirp_correction(fit)
    ds.subtract_solvent(sol, dt=0.0, scale=0.3)
    ds.mask_probe_regions([(1952.0, 1956.0)])

    recipe_path = orig_dir / "workflow.pmrecipe.toml"
    recipe = Recipe.from_dataset(
        ds,
        name="TA_standard",
        calibration_file=str(cal_csv),
        export_formats=["pdat", "hdf5", "png"],
    )
    recipe.save(recipe_path)
    toml_txt = recipe_path.read_text(encoding="utf-8")
    assert 'schema_version = "1.0"' in toml_txt
    assert "aux/chirp.json" in toml_txt
    assert "aux/solvent.pdat" in toml_txt

    # Move entire folder tree to simulate sharing with another machine
    moved_dir = tmp_path / "moved"
    orig_dir.rename(moved_dir)
    moved_recipe = Recipe.load(moved_dir / "workflow.pmrecipe.toml")
    target_ds = pm.load_1D(moved_dir / "scan1.pdat", data_type="PDAT")
    moved_recipe.apply(target_ds, base_dir=moved_dir)

    assert np.allclose(target_ds.probe, ds.probe)
    assert [s.op for s in target_ds.history] == [
        "background_correct",
        "apply_chirp_correction",
        "subtract_solvent",
        "mask_probe_regions",
    ]
    assert np.allclose(target_ds.Z, ds.Z, equal_nan=True)

    # Disabled step is skipped cleanly
    moved_recipe.steps[-1]["enabled"] = False
    ds_nomask = pm.load_1D(moved_dir / "scan1.pdat", data_type="PDAT")
    moved_recipe.apply(ds_nomask, base_dir=moved_dir)
    assert "mask_probe_regions" not in [s.op for s in ds_nomask.history]


def test_discover_and_run_batch_1d_and_2d_with_error_resilience(tmp_path):
    in_1d = tmp_path / "in_1d"
    sub = in_1d / "nested"
    sub.mkdir(parents=True)
    make_synthetic_pdat(in_1d / "a.pdat", baseline=0.05)
    make_synthetic_pdat(sub / "b.pdat", baseline=0.08)
    make_synthetic_pymess_pp(in_1d / "c.h5", n_scans=2, n_delays=12, n_pixels=20)
    # Companion noise (valid size) + orphan .pdatn + corrupted file that should fail gracefully
    (in_1d / "orphan.pdatn").write_text("0\n", encoding="utf-8")
    (in_1d / "broken.pdat").write_text("not a valid pdat file\ngarbage", encoding="utf-8")
    (in_1d / "ignored.pmrecipe.toml").write_text('dimension = "1D"\n', encoding="utf-8")

    discovered = discover_batch_inputs(in_1d, data_type="auto", dimension="1D", recursive=True)
    discovered_names = {p.name for p, _ in discovered}
    assert discovered_names == {"a.pdat", "b.pdat", "c.h5", "broken.pdat"}

    recipe = Recipe(
        name="batch_1d",
        dimension="1D",
        data_type="auto",
        steps=[
            {
                "op": "background_correct",
                "enabled": True,
                "params": {"tmin": -2.0, "tmax": -0.5, "do_correct": True},
            }
        ],
        export_formats=["pdat", "hdf5", "glotaran_ascii", "png"],
        export_suffix="_PROC",
        write_noise=True,
    )

    out_dry = tmp_path / "out_dry"
    dry_report = run_batch(discovered, recipe, out_dry, dry_run=True)
    assert dry_report.dry_run
    assert dry_report.total == 4
    assert all(item.status == "dry_run" for item in dry_report.items)
    assert not out_dry.exists()

    out_dir = tmp_path / "out_1d"
    report = run_batch(discovered, recipe, out_dir, workers=1, on_error="skip")
    assert report.total == 4
    assert report.succeeded == 3
    assert report.failed == 1
    assert report.report_csv and Path(report.report_csv).is_file()
    assert report.report_json and Path(report.report_json).is_file()

    assert (out_dir / "a_PROC.pdat").is_file()
    assert (out_dir / "a_PROC.hdf5").is_file()
    assert (out_dir / "a_PROC.ascii").is_file()
    assert (out_dir / "a_PROC.png").is_file()
    assert (out_dir / "b_PROC.pdat").is_file()
    assert (out_dir / "c_PROC.pdat").is_file()

    with open(report.report_csv, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    json_data = json.loads(Path(report.report_json).read_text(encoding="utf-8"))
    assert json_data["succeeded"] == 3
    assert json_data["failed"] == 1

    # on_error="stop" stops remaining items after failure
    stop_dir = tmp_path / "out_stop"
    ordered_inputs = [
        (in_1d / "broken.pdat", "PDAT"),
        (in_1d / "a.pdat", "PDAT"),
    ]
    stop_rep = run_batch(ordered_inputs, recipe, stop_dir, workers=1, on_error="stop")
    assert stop_rep.failed == 1
    assert stop_rep.skipped == 1
    assert stop_rep.items[1].status == "skipped"

    # 2D batch test
    in_2d = tmp_path / "in_2d"
    in_2d.mkdir()
    make_synthetic_p2dat(in_2d / "m1.p2dat", delays=(-1.0, 0.5, 2.0))
    make_synthetic_p2dat(in_2d / "m2.p2dat", delays=(-1.0, 0.5, 2.0))
    rec_2d = Recipe(
        name="batch_2d",
        dimension="2D",
        data_type="P2DAT",
        steps=[
            {
                "op": "background_correct",
                "enabled": True,
                "params": {"reference": 0, "do_correct": True},
            }
        ],
        export_formats=["p2dat", "png"],
        export_suffix="_PROC",
    )
    out_2d = tmp_path / "out_2d"
    rep_2d = run_batch(
        discover_batch_inputs(in_2d, data_type="auto", dimension="2D"), rec_2d, out_2d
    )
    assert rep_2d.succeeded == 2
    assert (out_2d / "m1_PROC.p2dat").is_file()
    assert (out_2d / "m2_PROC.png").is_file()


def test_run_batch_multiprocessing_and_cancellation(tmp_path):
    in_dir = tmp_path / "mp_in"
    in_dir.mkdir()
    for i in range(4):
        make_synthetic_pdat(in_dir / f"s{i}.pdat")

    recipe = Recipe(
        name="mp_recipe",
        dimension="1D",
        data_type="PDAT",
        steps=[{"op": "shift_t0", "enabled": True, "params": {"t0_shift": 0.2}}],
        export_formats=["pdat"],
    )
    out_dir = tmp_path / "mp_out"
    inputs = discover_batch_inputs(in_dir, data_type="PDAT")
    rep = run_batch(inputs, recipe, out_dir, workers=2)
    assert rep.succeeded == 4
    assert rep.failed == 0
    for i in range(4):
        assert (out_dir / f"s{i}_PROCESSED.pdat").is_file()

    # Pre-set cancellation stops immediately
    cancel_ev = threading.Event()
    cancel_ev.set()
    rep_c = run_batch(inputs, recipe, tmp_path / "cancel_out", workers=1, cancel_event=cancel_ev)
    assert rep_c.cancelled
    assert rep_c.succeeded == 0
    assert all(r.status == "cancelled" for r in rep_c.items)


def test_batch_cli_and_launcher_entrypoint(tmp_path):
    in_dir = tmp_path / "cli_in"
    in_dir.mkdir()
    make_synthetic_pdat(in_dir / "x1.pdat")
    make_synthetic_pdat(in_dir / "x2.pdat")

    rec_path = tmp_path / "cli.pmrecipe.toml"
    Recipe(
        name="cli_rec",
        dimension="1D",
        data_type="auto",
        steps=[
            {
                "op": "background_correct",
                "enabled": True,
                "params": {"tmin": -20.0, "tmax": -5.0, "do_correct": True},
            }
        ],
        export_formats=["pdat"],
    ).save(rec_path)

    out_dir = tmp_path / "cli_out"
    rc_dry = cli_main(
        ["--recipe", str(rec_path), "--input", str(in_dir), "--output", str(out_dir), "--dry-run"]
    )
    assert rc_dry == 0
    assert not out_dir.exists()

    rc = launcher_main(
        [
            "batch",
            "--recipe",
            str(rec_path),
            "--input",
            str(in_dir),
            "--output",
            str(out_dir),
            "--workers",
            "1",
        ]
    )
    assert rc == 0
    assert (out_dir / "x1_PROCESSED.pdat").is_file()
    assert (out_dir / "x2_PROCESSED.pdat").is_file()


def test_gui_batch_dialog_runs_on_folder(qapp, tmp_path):
    from PyQt6.QtTest import QTest

    from pymorgan.gui.batch_dialog import BatchProcessDialog

    in_dir = tmp_path / "gui_batch_in"
    out_dir = tmp_path / "gui_batch_out"
    in_dir.mkdir()
    p1 = in_dir / "d1.pdat"
    p2 = in_dir / "d2.pdat"
    make_synthetic_pdat(p1)
    make_synthetic_pdat(p2)

    ds = pm.load_1D(p1, data_type="PDAT")
    ds.background_correct(-20.0, -5.0)
    ds.shift_t0(0.4)

    dlg = BatchProcessDialog(active_dataset=ds, default_dir=str(in_dir))
    try:
        assert dlg.table.rowCount() == 2
        assert dlg.step_list.count() == 2
        dlg.output_dir_edit.setText(str(out_dir))
        dlg.btn_run.click()

        # Spin Qt event loop briefly until worker thread completes
        for _ in range(100):
            qapp.processEvents()
            if dlg._last_report is not None:
                break
            QTest.qWait(50)

        assert dlg._last_report is not None
        assert dlg._last_report.succeeded == 2
        assert (out_dir / "d1_PROCESSED.pdat").is_file()
        assert (out_dir / "d2_PROCESSED.pdat").is_file()
    finally:
        dlg.close()
