"""Tests for Milestone 6: .pmproj sessions, recent folders, and zero-I/O browser filter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QFileDialog
from synthetic import make_spectrum_csv, make_synthetic_p2dat, make_synthetic_pdat

import pymorgan as pm
from pymorgan.gui.main_window import MainWindow
from pymorgan.project import ProjectSession


def test_project_session_roundtrip_portable_move_and_missing_recovery(qapp, tmp_path, monkeypatch):
    proj_dir = tmp_path / "orig_proj"
    data_dir = proj_dir / "data"
    data_dir.mkdir(parents=True)

    pdat1 = data_dir / "sample1d.pdat"
    p2dat = data_dir / "sample2d.p2dat"
    ss_abs = data_dir / "abs.csv"
    make_synthetic_pdat(pdat1, baseline=0.06)
    make_synthetic_p2dat(p2dat, delays=(-1.0, 0.5, 2.0, 5.0))
    make_spectrum_csv(ss_abs)

    win = MainWindow(init_rootdir=False)
    try:
        win.RootDir_field.setText(str(data_dir))
        win._on_rootdir_changed()

        win.load_path(str(pdat1), data_type="PDAT")
        win.dataset.background_correct(-20.0, -5.0, do_correct=True)
        win.dataset.mask_probe_regions([(1950.0, 1955.0)])
        win.dataset.shift_t0(0.25)
        win._sync_oneD_ui_after_history_change()
        assert "*" in win.windowTitle()

        win._load_steady_state("absorption", path=str(ss_abs))

        win.MainTabs.setCurrentIndex(1)
        win.twoD_load_path(str(p2dat), data_type="P2DAT")
        win.twoD_dataset.background_correct(reference=0, do_correct=True)
        win._sync_twoD_ui_after_history_change()

        pm.update_settings(cmap="Seismic")

        z1_expected = np.array(win.dataset.Z, copy=True)
        z2_expected = np.array(win.twoD_dataset.Z, copy=True)

        proj_file = proj_dir / "analysis.pmproj"
        monkeypatch.setattr(
            QFileDialog,
            "getSaveFileName",
            lambda *_a, **_k: (str(proj_file), "PyMORGAN Projects (*.pmproj)"),
        )
        win._save_project()
        assert proj_file.is_file()
        assert not win._project_dirty
        assert "*" not in win.windowTitle()

        # Verify portable relative path saved in TOML
        toml_txt = proj_file.read_text(encoding="utf-8")
        assert 'path = "data/sample1d.pdat"' in toml_txt
        assert 'path = "data/sample2d.p2dat"' in toml_txt
        assert "Seismic" in toml_txt
    finally:
        win.close()

    # Reset settings & move entire project folder -> relative path resolves with old abs gone
    pm.set_settings(pm.Settings())
    moved_dir = tmp_path / "moved_proj"
    proj_dir.rename(moved_dir)
    moved_pmproj = moved_dir / "analysis.pmproj"

    win2 = MainWindow(init_rootdir=False)
    try:
        warnings = win2.open_project(moved_pmproj, interactive=False)
        assert warnings == []
        assert win2.MainTabs.currentIndex() == 1
        assert Path(win2.RootDir_field.text()) == (moved_dir / "data")
        assert win2.dataset is not None
        assert win2.twoD_dataset is not None
        assert np.allclose(win2.dataset.Z, z1_expected, equal_nan=True)
        assert np.allclose(win2.twoD_dataset.Z, z2_expected, equal_nan=True)
        assert win2._ss_abs_spectrum is not None
        assert pm.get_settings().cmap == "Seismic"
        assert not win2._project_dirty
    finally:
        win2.close()

    # Missing file recovery: replace deleted 1D via callback, skip deleted 2D
    replacement = tmp_path / "relocated_sample1d.pdat"
    make_synthetic_pdat(replacement, baseline=0.06)
    (moved_dir / "data" / "sample1d.pdat").unlink()
    (moved_dir / "data" / "sample2d.p2dat").unlink()

    called_roles: list[str] = []

    def _relocate_cb(role: str, missing: Path):
        called_roles.append(role)
        if role == "oneD.path":
            return replacement
        return None  # Skip 2D

    win3 = MainWindow(init_rootdir=False)
    try:
        warnings3 = win3.open_project(
            moved_pmproj, relocate_callback=_relocate_cb, interactive=False
        )
        assert "oneD.path" in called_roles
        assert "twoD.path" in called_roles
        assert any("twoD.path" in w for w in warnings3)
        assert win3.dataset is not None
        assert np.allclose(win3.dataset.Z, z1_expected, equal_nan=True)
        assert win3.twoD_dataset is None
    finally:
        win3.close()


def test_project_defensive_active_detector_roundtrip(tmp_path):
    p2dat = tmp_path / "m.p2dat"
    make_synthetic_p2dat(p2dat, delays=(-1.0, 0.5, 2.0))

    class _FakeWin:
        RootDir_field = None
        _mess_slowmod = 0
        _mess_spectrum = 0
        _current_project_path = None
        dataset = None
        twoD_plot_controls = None
        plot_controls = None

        def _rootdir_text(self):
            return str(tmp_path)

    ds2 = pm.load_2D(p2dat, data_type="P2DAT")
    ds2.detector_names = ["Det1", "Det2"]
    ds2.active_detector = "Det2"
    set_calls: list[str] = []

    def _set_det(name: str) -> bool:
        set_calls.append(name)
        ds2.active_detector = name
        return True

    ds2.set_active_detector = _set_det
    w = _FakeWin()
    w.twoD_dataset = ds2
    sess = ProjectSession.from_main_window(w, base_dir=tmp_path)
    assert sess.twoD["active_detector"] == "Det2"
    proj_p = tmp_path / "det.pmproj"
    sess.save(proj_p)
    loaded = ProjectSession.load(proj_p)
    assert loaded.twoD["active_detector"] == "Det2"


def test_recent_folders_mru_persistence_and_menu(qapp, tmp_path, monkeypatch):
    cfg_path = tmp_path / "settings.toml"
    monkeypatch.setattr(pm.settings, "settings_path", lambda: cfg_path)
    pm.set_settings(pm.Settings(max_recent_dirs=2))

    d1 = tmp_path / "exp_a"
    d2 = tmp_path / "exp_b"
    d3 = tmp_path / "exp_c"
    for d in (d1, d2, d3):
        d.mkdir()

    win = MainWindow(init_rootdir=False)
    pm.update_settings(max_recent_dirs=2)
    try:
        for d in (d1, d2, d1, d3):
            win.RootDir_field.setText(str(d))
            win._on_rootdir_changed()

        # Max recent_dirs = 2 -> MRU is [d3, d1]
        recent = pm.get_settings().recent_dirs
        assert [Path(p).resolve() for p in recent] == [d3.resolve(), d1.resolve()]

        # Persisted to isolated tmp settings.toml
        reloaded = pm.Settings.load(cfg_path)
        assert [Path(p).resolve() for p in reloaded.recent_dirs] == [d3.resolve(), d1.resolve()]

        # Submenu + QCompleter contain MRU paths + Clear Recent
        act_texts = [a.text() for a in win.menuRecentFolders.actions() if not a.isSeparator()]
        assert str(d3.resolve()) in act_texts[0]
        assert str(d1.resolve()) in act_texts[1]
        assert "Clear Recent" in act_texts[-1]
        assert win._rootdir_completer_model.stringList() == recent

        # Clear Recent removes everything cleanly
        win._clear_recent_dirs()
        assert pm.get_settings().recent_dirs == []
        assert win._rootdir_completer_model.stringList() == []
    finally:
        win.close()


def test_zero_io_browser_filter_1d_and_2d(qapp, tmp_path):
    root = tmp_path / "browser_root"
    (root / "sub_alpha").mkdir(parents=True)
    (root / "sub_gamma").mkdir(parents=True)
    make_synthetic_pdat(root / "alpha_scan1.pdat")
    make_synthetic_pdat(root / "beta_scan2.pdat")
    make_synthetic_pdat(root / "alpha_scan3.pdat")
    make_synthetic_p2dat(root / "map_alpha.p2dat")
    make_synthetic_p2dat(root / "map_beta.p2dat")

    from pymorgan.gui.mw_common import set_combo_datatype

    win = MainWindow(init_rootdir=False)
    try:
        set_combo_datatype(win.PP_datatype_cbx, "PDAT")
        set_combo_datatype(win.twoD_datatype_cbx, "P2DAT")
        win.RootDir_field.setText(str(root))
        win._on_rootdir_changed()

        def _row_labels(model):
            return [model.item(r).text() for r in range(model.rowCount())]

        assert "[..]" in _row_labels(win._folder_model)[0]
        initial_count = win._folder_model.rowCount()
        assert initial_count >= 5

        # Spy on _scan_dataset_folder -> typing in filter MUST trigger 0 disk scans
        scan_calls = 0
        orig_scan = win._scan_dataset_folder

        def _spy_scan(*args, **kwargs):
            nonlocal scan_calls
            scan_calls += 1
            return orig_scan(*args, **kwargs)

        win._scan_dataset_folder = _spy_scan

        win._focus_dataset_filter()
        assert win.PP_filter_edit.placeholderText().startswith("Filter datasets")
        win.PP_filter_edit.setText("ALPHA")
        qapp.processEvents()

        assert scan_calls == 0
        labels_filtered = _row_labels(win._folder_model)
        assert labels_filtered[0] == "[..]"
        assert any("alpha_scan1" in lbl for lbl in labels_filtered)
        assert any("alpha_scan3" in lbl for lbl in labels_filtered)
        assert any("sub_alpha" in lbl for lbl in labels_filtered)
        assert not any("beta" in lbl.lower() for lbl in labels_filtered)

        win.PP_filter_edit.clear()
        qapp.processEvents()
        assert scan_calls == 0
        assert win._folder_model.rowCount() == initial_count

        # Verify 2D tab filter is also zero-I/O
        win.MainTabs.setCurrentIndex(1)
        scan_calls = 0
        win._focus_dataset_filter()
        win.twoD_filter_edit.setText("map_alpha")
        qapp.processEvents()
        assert scan_calls == 0
        labels_2d = _row_labels(win._twoD_folder_model)
        assert labels_2d[0] == "[..]"
        assert any("map_alpha" in lbl for lbl in labels_2d)
        assert not any("map_beta" in lbl for lbl in labels_2d)
    finally:
        win.close()
