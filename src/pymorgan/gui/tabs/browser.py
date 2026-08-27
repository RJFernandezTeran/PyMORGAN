"""Root-folder navigation and the 1D / 2D dataset list widgets.

Part of the :class:`~pymorgan.gui.main_window.MainWindow` implementation, split out
as a mixin so each tab lives in its own module. The methods are unchanged moves:
they run as ``MainWindow`` methods and bind to the widgets declared in
``main_window.ui``, so ``self`` is always the main window.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyqt6")

from pathlib import Path

from PyQt6.QtGui import (
    QBrush,
    QColor,
    QStandardItem,
)
from PyQt6.QtWidgets import (
    QFileDialog,
)

import pymorgan as pm
from pymorgan.oneD.load import (
    dataset_glob,
    is_dataset_dir,
    is_dataset_dir_status,
    is_dataset_file,
)

from ..mw_common import (
    _ISDIR_ROLE,
    _PATH_ROLE,
    get_combo_datatype,
)


class DatasetBrowserMixin:
    """Root-folder navigation and the 1D / 2D dataset list widgets."""

    def _on_rootdir_changed(self):
        self.refresh_dataset_list()
        self.twoD_refresh_dataset_list()

    def _force_refresh_dataset_lists(self):
        """Re-read both listings from disk, bypassing the folder-scan cache."""
        self._invalidate_folder_scan_cache()
        self._on_rootdir_changed()

    def _refresh_list_keeping_selection(self, view_name: str, model_name: str, refresh):
        """Re-populate a dataset list without re-triggering a dataset load.

        The selection model is blocked while the model is rebuilt, and the
        previously selected path is restored afterwards if it still exists.
        This is used when the lists are synchronised on tab changes, where the
        currently loaded dataset must not be reloaded.
        """
        view = getattr(self, view_name, None)
        model = getattr(self, model_name, None)
        if view is None or model is None:
            refresh()
            return

        sel = view.selectionModel()
        previous = view.currentIndex().data(_PATH_ROLE) if view.currentIndex().isValid() else None
        if sel is not None:
            sel.blockSignals(True)
        try:
            refresh()
        finally:
            if sel is not None:
                sel.blockSignals(False)

        if previous is None:
            return
        for row in range(model.rowCount()):
            index = model.index(row, 0)
            if index.data(_PATH_ROLE) == previous:
                if sel is not None:
                    sel.blockSignals(True)
                try:
                    view.setCurrentIndex(index)
                finally:
                    if sel is not None:
                        sel.blockSignals(False)
                break

    def _sync_dataset_lists(self, index: int | None = None):
        """Refresh the dataset list(s) so the active tab always shows current contents.

        Both tabs share the same root folder (``RootDir_field``); only the
        listing of the tab being shown is rebuilt, since the other one is
        rebuilt when it becomes visible in turn. ``index`` is the main-tab
        index (0 = 1D, 1 = 2D); ``None`` refreshes both.
        """
        if not self._rootdir_text():
            return
        if index is None or index == 0:
            self._refresh_list_keeping_selection(
                "PP_datafolderlist_lst", "_folder_model", self.refresh_dataset_list
            )
        if index is None or index == 1:
            self._refresh_list_keeping_selection(
                "twoD_datafolderlist_lst", "_twoD_folder_model", self.twoD_refresh_dataset_list
            )

    def _rootdir_text(self) -> str:
        field = getattr(self, "RootDir_field", None)
        return field.text().strip() if field is not None else ""

    def _current_datatype(self) -> str:
        combo = getattr(self, "PP_datatype_cbx", None)
        return get_combo_datatype(combo, "PDAT")

    def _current_twoD_datatype(self) -> str:
        combo = getattr(self, "twoD_datatype_cbx", None)
        return get_combo_datatype(combo, "P2DAT")

    def _current_glob(self) -> str | None:
        """File glob for a file-based data type, or ``None`` for directory formats."""
        return dataset_glob(self._current_datatype())

    def browse_rootdir(self):
        """Pick the root folder with a dialog (the field can also be typed)."""
        start = self._rootdir_text() or ""
        folder = QFileDialog.getExistingDirectory(self, "Select root folder", start)
        if folder and hasattr(self, "RootDir_field"):
            self.RootDir_field.setText(folder)
            self._on_rootdir_changed()

    def _add_dir_item(self, label: str, path: Path, model=None):
        """Append a navigable folder row, shown as ``[label]``."""
        item = QStandardItem(f"[{label}]")
        item.setForeground(QBrush(QColor(150, 150, 150)))
        item.setData(str(path), _PATH_ROLE)
        item.setData(True, _ISDIR_ROLE)
        item.setEditable(False)
        (model if model is not None else self._folder_model).appendRow(item)

    def _add_dataset_item(self, label: str, path: Path, model=None, is_valid: bool = True):
        """Append a loadable dataset row (loads on selection). If not valid, shown grayed out."""
        item = QStandardItem(label)
        item.setData(str(path), _PATH_ROLE)
        item.setEditable(False)
        if not is_valid:
            item.setForeground(QBrush(QColor(150, 150, 150)))
            item.setEnabled(False)
        (model if model is not None else self._folder_model).appendRow(item)

    def _scan_dataset_folder(self, folder: Path, dt: str, *, twoD: bool):
        """Classify the contents of ``folder`` for data type ``dt``.

        Returns ``(entries, n_datasets)`` with ``entries`` a list of
        ``(label, path, is_dir, is_valid)`` tuples in display order (excluding the ``..``
        row). The dataset detectors probe the filesystem once per sub-folder,
        which dominates the cost on network shares, so results are cached per
        ``(folder, dt, twoD)`` and invalidated by the folder's mtime.
        """
        cache = getattr(self, "_folder_scan_cache", None)
        if cache is None:
            cache = self._folder_scan_cache = {}
        try:
            stamp = folder.stat().st_mtime_ns
        except OSError:
            stamp = None
        key = (str(folder), dt, twoD)
        if stamp is not None:
            hit = cache.get(key)
            if hit is not None and hit[0] == stamp:
                return hit[1], hit[2]

        is_dataset = pm.twoD.is_map_dataset_dir if twoD else is_dataset_dir
        pattern = pm.twoD.map_dataset_glob(dt) if twoD else dataset_glob(dt)

        try:
            subdirs = sorted(
                (d for d in folder.iterdir() if d.is_dir()),
                key=lambda d: d.name.lower(),
            )
        except OSError:
            subdirs = []
        entries = []
        n_datasets = 0
        for d in subdirs:
            if twoD:
                if pm.twoD.is_map_dataset_dir(dt, d):
                    entries.append((d.name, d, False, True))
                    n_datasets += 1
                else:
                    entries.append((d.name, d, True, True))
            else:
                is_cand, is_val = is_dataset_dir_status(dt, d)
                if is_cand:
                    entries.append((d.name, d, False, is_val))
                    if is_val:
                        n_datasets += 1
                else:
                    entries.append((d.name, d, True, True))
        if pattern:
            try:
                files = sorted(
                    (f for f in folder.glob(pattern) if f.is_file()),
                    key=lambda f: f.name.lower(),
                )
            except OSError:
                files = []
            for f in files:
                valid = True if twoD else is_dataset_file(dt, f)
                if not valid:
                    continue
                entries.append((f.name, f, False, True))
                n_datasets += 1

        if stamp is not None:
            if len(cache) > 32:
                cache.clear()
            cache[key] = (stamp, entries, n_datasets)
        return entries, n_datasets

    def _invalidate_folder_scan_cache(self):
        """Drop the cached folder listings (used by the explicit Refresh button)."""
        cache = getattr(self, "_folder_scan_cache", None)
        if cache is not None:
            cache.clear()

    def refresh_dataset_list(self):
        """Populate the list with sub-folders and datasets in the root folder.

        Every sub-folder is inspected with the active data type's dataset
        detector: a folder that contains data named after itself is shown as
        ``<name>`` and loads on selection; any other folder is shown as
        ``[name]`` and is entered by double-click. File-based formats (e.g.
        PDAT) additionally list matching data files in the current folder.
        """
        self._folder_model.clear()
        root = self._rootdir_text()
        if not root:
            return
        folder = Path(root)
        if not folder.is_dir():
            self.statusBar().showMessage(f"Not a folder: {root}")
            return
        dt = self._current_datatype()
        self._add_dir_item("..", folder.parent)
        entries, n_datasets = self._scan_dataset_folder(folder, dt, twoD=False)
        for item in entries:
            label, path, is_dir = item[0], item[1], item[2]
            is_valid = item[3] if len(item) > 3 else True
            if is_dir:
                self._add_dir_item(label, path)
            else:
                self._add_dataset_item(label, path, is_valid=is_valid)
        self.statusBar().showMessage(f"{n_datasets} dataset(s) of type {dt} in {folder.name}")

    def _on_dataset_activated(self, index, _previous=None):
        if index is None or not index.isValid():
            return
        if index.data(_ISDIR_ROLE):
            return  # folders are entered by double-click, not loaded
        path = index.data(_PATH_ROLE)
        if path:
            self.load_path(path)

    def _on_item_double_clicked(self, index):
        """Enter a folder on double-click (datasets load on selection instead)."""
        if index is None or not index.isValid() or not index.data(_ISDIR_ROLE):
            return
        target = index.data(_PATH_ROLE)
        if target and hasattr(self, "RootDir_field"):
            self.RootDir_field.setText(str(target))
            self._on_rootdir_changed()

    def _go_up_dir(self):
        root = self._rootdir_text()
        if root and hasattr(self, "RootDir_field"):
            self.RootDir_field.setText(str(Path(root).parent))
            self._on_rootdir_changed()

    def _go_to_today(self):
        """Navigate to ``<default_datadir>/<YYYYMMDD>`` from the settings file."""
        from datetime import date

        base = getattr(pm.get_settings(), "default_datadir", "")
        if not base:
            self.statusBar().showMessage(
                "Set 'default_datadir' in settings.toml to use the Today button"
            )
            return
        target = Path(base) / date.today().strftime("%Y%m%d")
        if hasattr(self, "RootDir_field"):
            self.RootDir_field.setText(str(target))
        self._on_rootdir_changed()

    def _init_rootdir(self):
        """Pre-fill the root folder with the settings ``default_datadir`` if empty."""
        d_dir = getattr(pm.get_settings(), "default_datadir", "")
        field = getattr(self, "RootDir_field", None)
        if d_dir and field is not None and not field.text().strip():
            field.setText(str(d_dir))
            self._on_rootdir_changed()

    def twoD_refresh_dataset_list(self):
        """Populate the 2D file list with sub-folders and 2D data files."""
        self._twoD_folder_model.clear()
        root = self._rootdir_text()
        if not root:
            return
        folder = Path(root)
        if not folder.is_dir():
            self.statusBar().showMessage(f"Not a folder: {root}")
            return
        dt = self._current_twoD_datatype()
        model = self._twoD_folder_model
        self._add_dir_item("..", folder.parent, model)
        entries, n_datasets = self._scan_dataset_folder(folder, dt, twoD=True)
        for item in entries:
            label, path, is_dir = item[0], item[1], item[2]
            is_valid = item[3] if len(item) > 3 else True
            if is_dir:
                self._add_dir_item(label, path, model)
            else:
                self._add_dataset_item(label, path, model, is_valid=is_valid)
        self.statusBar().showMessage(f"{n_datasets} 2D dataset(s) of type {dt} in {folder.name}")

    def _on_twoD_dataset_activated(self, index, _previous=None):
        if index is None or not index.isValid():
            return
        if index.data(_ISDIR_ROLE):
            return
        path = index.data(_PATH_ROLE)
        if path:
            self.twoD_load_path(path)

    def _on_twoD_item_double_clicked(self, index):
        if index is None or not index.isValid() or not index.data(_ISDIR_ROLE):
            return
        target = index.data(_PATH_ROLE)
        if target and hasattr(self, "RootDir_field"):
            self.RootDir_field.setText(str(target))
            self._on_rootdir_changed()
