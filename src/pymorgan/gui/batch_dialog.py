"""Batch Processing GUI dialog for applying `.pmrecipe.toml` recipes across datasets."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..batch import (
    SUPPORTED_EXPORT_FORMATS,
    BatchItemResult,
    BatchReport,
    discover_batch_inputs,
    run_batch,
)
from ..recipe import Recipe


class _BatchWorkerThread(QThread):
    """Background worker driving :func:`pymorgan.batch.run_batch`."""

    item_progress = pyqtSignal(int, int, object)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        inputs: list[tuple[Path, str]],
        recipe: Recipe,
        output_dir: Path,
        *,
        workers: int,
        on_error: str,
        dry_run: bool,
        base_dir: Path | None,
        cancel_event: threading.Event,
    ):
        super().__init__()
        self._inputs = inputs
        self._recipe = recipe
        self._output_dir = output_dir
        self._workers = workers
        self._on_error = on_error
        self._dry_run = dry_run
        self._base_dir = base_dir
        self._cancel_event = cancel_event

    def run(self) -> None:
        try:
            report = run_batch(
                self._inputs,
                self._recipe,
                self._output_dir,
                workers=self._workers,
                on_error=self._on_error,  # type: ignore[arg-type]
                dry_run=self._dry_run,
                base_dir=self._base_dir,
                progress_callback=lambda done, total, res: self.item_progress.emit(
                    done, total, res
                ),
                cancel_event=self._cancel_event,
            )
            self.completed.emit(report)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class BatchProcessDialog(QDialog):
    """Interactive batch-processing dialog with recipe editor and live table progress."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        active_dataset: Any = None,
        default_dir: str | None = None,
        initial_recipe: str | Path | None = None,
    ):
        super().__init__(parent)
        self._active_dataset = active_dataset
        self._default_dir = default_dir or ""
        self._recipe: Recipe | None = None
        self._recipe_base_dir: Path | None = None
        self._discovered: list[tuple[Path, str]] = []
        self._worker_thread: _BatchWorkerThread | None = None
        self._cancel_event = threading.Event()
        self._last_report: BatchReport | None = None

        self.setWindowTitle("Batch Process Datasets")
        self.resize(940, 700)

        root = QVBoxLayout(self)
        root.setSpacing(8)

        # --- 1. Inputs Group ---
        inp_grp = QGroupBox("1. Input Datasets")
        inp_lay = QVBoxLayout(inp_grp)

        row_top = QHBoxLayout()
        row_top.addWidget(QLabel("Folder:"))
        self.input_dir_edit = QLineEdit(self._default_dir)
        self.input_dir_edit.setPlaceholderText(
            "Select an input folder or add individual dataset files..."
        )
        row_top.addWidget(self.input_dir_edit, 1)

        self.btn_browse_input = QPushButton("Browse Folder...")
        self.btn_browse_input.clicked.connect(self._browse_input_dir)
        row_top.addWidget(self.btn_browse_input)

        self.btn_add_files = QPushButton("Add Files...")
        self.btn_add_files.clicked.connect(self._add_files)
        row_top.addWidget(self.btn_add_files)

        row_top.addWidget(QLabel("Data type:"))
        self.data_type_cbx = QComboBox()
        self.data_type_cbx.addItems(self._available_datatype_choices())
        row_top.addWidget(self.data_type_cbx)

        self.chk_recursive = QCheckBox("Recursive")
        row_top.addWidget(self.chk_recursive)

        self.btn_rescan = QPushButton("Scan")
        self.btn_rescan.clicked.connect(self.rescan_inputs)
        row_top.addWidget(self.btn_rescan)
        inp_lay.addLayout(row_top)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Include", "Dataset", "Type", "Status", "Outputs"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hdr = self.table.horizontalHeader()
        if hdr is not None:
            hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
            hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        inp_lay.addWidget(self.table, 1)

        sel_row = QHBoxLayout()
        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.clicked.connect(lambda: self._set_all_checked(True))
        sel_row.addWidget(self.btn_select_all)
        self.btn_select_none = QPushButton("Select None")
        self.btn_select_none.clicked.connect(lambda: self._set_all_checked(False))
        sel_row.addWidget(self.btn_select_none)
        self.lbl_input_count = QLabel("0 datasets discovered")
        sel_row.addWidget(self.lbl_input_count)
        sel_row.addStretch(1)
        inp_lay.addLayout(sel_row)
        root.addWidget(inp_grp, 2)

        # --- 2. Recipe Group ---
        rec_grp = QGroupBox("2. Processing Recipe (.pmrecipe.toml)")
        rec_lay = QVBoxLayout(rec_grp)

        rec_top = QHBoxLayout()
        self.lbl_recipe = QLabel("No recipe loaded")
        rec_top.addWidget(self.lbl_recipe, 1)

        self.btn_load_recipe = QPushButton("Load .pmrecipe.toml...")
        self.btn_load_recipe.clicked.connect(self._load_recipe_dialog)
        rec_top.addWidget(self.btn_load_recipe)

        self.btn_import_active = QPushButton("Import from Active Dataset")
        self.btn_import_active.setEnabled(active_dataset is not None)
        self.btn_import_active.clicked.connect(self._import_from_active_dataset)
        rec_top.addWidget(self.btn_import_active)

        self.btn_save_recipe = QPushButton("Save Recipe...")
        self.btn_save_recipe.setEnabled(False)
        self.btn_save_recipe.clicked.connect(self._save_recipe_dialog)
        rec_top.addWidget(self.btn_save_recipe)
        rec_lay.addLayout(rec_top)

        self.step_list = QListWidget()
        self.step_list.setMaximumHeight(130)
        self.step_list.itemChanged.connect(self._on_step_toggled)
        rec_lay.addWidget(self.step_list)
        root.addWidget(rec_grp, 1)

        # --- 3. Output & Options Group ---
        out_grp = QGroupBox("3. Output & Execution Options")
        out_grid = QGridLayout(out_grp)

        out_grid.addWidget(QLabel("Output folder:"), 0, 0)
        default_out = str(Path(self._default_dir) / "batch_output") if self._default_dir else ""
        self.output_dir_edit = QLineEdit(default_out)
        out_grid.addWidget(self.output_dir_edit, 0, 1, 1, 4)
        self.btn_browse_output = QPushButton("Browse...")
        self.btn_browse_output.clicked.connect(self._browse_output_dir)
        out_grid.addWidget(self.btn_browse_output, 0, 5)

        out_grid.addWidget(QLabel("Export formats:"), 1, 0)
        fmt_row = QHBoxLayout()
        self.format_checks: dict[str, QCheckBox] = {}
        for fmt in SUPPORTED_EXPORT_FORMATS:
            chk = QCheckBox(fmt)
            if fmt == "pdat":
                chk.setChecked(True)
            self.format_checks[fmt] = chk
            fmt_row.addWidget(chk)
        fmt_row.addStretch(1)
        out_grid.addLayout(fmt_row, 1, 1, 1, 5)

        out_grid.addWidget(QLabel("Filename suffix:"), 2, 0)
        self.suffix_edit = QLineEdit("_PROCESSED")
        out_grid.addWidget(self.suffix_edit, 2, 1)

        self.chk_write_noise = QCheckBox("Write .pdatn noise")
        self.chk_write_noise.setChecked(True)
        out_grid.addWidget(self.chk_write_noise, 2, 2)

        out_grid.addWidget(QLabel("Workers:"), 2, 3)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, max(1, min(16, os.cpu_count() or 4)))
        self.workers_spin.setValue(1)
        out_grid.addWidget(self.workers_spin, 2, 4)

        self.on_error_cbx = QComboBox()
        self.on_error_cbx.addItems(["skip", "stop"])
        out_grid.addWidget(self.on_error_cbx, 2, 5)

        root.addWidget(out_grp)

        # --- 4. Progress & Action Footer ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        root.addWidget(self.progress_bar)

        footer = QHBoxLayout()
        self.lbl_status = QLabel("Ready")
        footer.addWidget(self.lbl_status, 1)

        self.btn_dry_run = QPushButton("Dry Run Preview")
        self.btn_dry_run.clicked.connect(lambda: self._start_run(dry_run=True))
        footer.addWidget(self.btn_dry_run)

        self.btn_run = QPushButton("Run Batch")
        self.btn_run.clicked.connect(lambda: self._start_run(dry_run=False))
        footer.addWidget(self.btn_run)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_run)
        footer.addWidget(self.btn_cancel)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        footer.addWidget(self.btn_close)
        root.addLayout(footer)

        if initial_recipe is not None:
            self.load_recipe_path(initial_recipe)
        elif self._active_dataset is not None:
            self._import_from_active_dataset()
        if self._default_dir and Path(self._default_dir).is_dir():
            self.rescan_inputs()

    @staticmethod
    def _available_datatype_choices() -> list[str]:
        choices = ["auto"]
        try:
            from ..oneD import load as l1
            from ..twoD import load as l2

            for k in list(getattr(l1, "_LOADERS", {}).keys()) + list(
                getattr(l2, "_LOADERS", {}).keys()
            ):
                if k not in choices:
                    choices.append(k)
        except Exception:
            choices.extend(["PDAT", "HDF5", "PyMESS_PP", "P2DAT", "PyMESS_2D"])
        return choices

    def set_recipe(self, recipe: Recipe, *, base_dir: Path | None = None) -> None:
        """Bind a :class:`Recipe` to the dialog and populate steps/export checkboxes."""
        self._recipe = recipe
        self._recipe_base_dir = base_dir
        self.lbl_recipe.setText(
            f"<b>{recipe.name}</b> &nbsp;|&nbsp; Dimension: {recipe.dimension} &nbsp;|&nbsp; "
            f"Steps: {len(recipe.steps)}"
        )
        self.btn_save_recipe.setEnabled(True)

        self.step_list.blockSignals(True)
        self.step_list.clear()
        for idx, st in enumerate(recipe.steps):
            lbl = st.get("label") or st.get("op", f"step_{idx}")
            item = QListWidgetItem(f"[{idx + 1}] {lbl}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if st.get("enabled", True) else Qt.CheckState.Unchecked
            )
            item.setData(Qt.ItemDataRole.UserRole, idx)
            self.step_list.addItem(item)
        self.step_list.blockSignals(False)

        fmts = {str(f).strip().lower() for f in (recipe.export_formats or [])}
        for fmt, chk in self.format_checks.items():
            chk.setChecked(fmt in fmts)
        self.suffix_edit.setText(recipe.export_suffix)
        self.chk_write_noise.setChecked(bool(recipe.write_noise))

    def _on_step_toggled(self, item: QListWidgetItem) -> None:
        if self._recipe is None:
            return
        idx = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(idx, int) and 0 <= idx < len(self._recipe.steps):
            self._recipe.steps[idx]["enabled"] = item.checkState() == Qt.CheckState.Checked

    def _import_from_active_dataset(self) -> None:
        if self._active_dataset is None:
            return
        dim = "2D" if hasattr(self._active_dataset, "Z_R") else "1D"
        src = getattr(self._active_dataset, "source", None)
        name = f"{Path(src).stem}_recipe" if src else "active_dataset_recipe"
        recipe = Recipe.from_dataset(
            self._active_dataset,
            name=name,
            export_formats=["p2dat" if dim == "2D" else "pdat"],
        )
        base = Path(src).parent if src else None
        self.set_recipe(recipe, base_dir=base)

    def load_recipe_path(self, path: str | Path) -> bool:
        """Load a ``.pmrecipe.toml`` file into the dialog and update base directory."""
        try:
            p = Path(path)
            recipe = Recipe.load(p)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid Recipe", str(exc))
            return False
        self.set_recipe(recipe, base_dir=p.parent)
        if not self.input_dir_edit.text().strip():
            self.input_dir_edit.setText(str(p.parent))
            self._default_dir = str(p.parent)
        if not self.output_dir_edit.text().strip():
            self.output_dir_edit.setText(str(p.parent / "batch_output"))
        self.rescan_inputs()
        return True

    def _load_recipe_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Processing Recipe",
            self._default_dir,
            "PyMORGAN Recipes (*.pmrecipe.toml *.toml);;All Files (*)",
        )
        if path:
            self.load_recipe_path(path)

    def _save_recipe_dialog(self) -> None:
        recipe = self._sync_recipe_from_ui()
        if recipe is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processing Recipe",
            str(Path(self._default_dir or ".") / f"{recipe.name}.pmrecipe.toml"),
            "PyMORGAN Recipes (*.pmrecipe.toml);;TOML Files (*.toml);;All Files (*)",
        )
        if not path:
            return
        target = Path(path)
        if not target.name.lower().endswith(".toml"):
            target = target.with_name(target.name + ".pmrecipe.toml")
        recipe.save(target)
        self._recipe_base_dir = target.parent

    def _browse_input_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder", self._default_dir)
        if not folder:
            return
        self.input_dir_edit.setText(folder)
        if not self.output_dir_edit.text().strip():
            self.output_dir_edit.setText(str(Path(folder) / "batch_output"))
        self.rescan_inputs()

    def _add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add Dataset Files", self._default_dir, "All Files (*)"
        )
        if not files:
            return
        dt = self.data_type_cbx.currentText()
        dim = self._recipe.dimension if self._recipe and dt == "auto" else None
        added = discover_batch_inputs(files, data_type=dt, dimension=dim)
        existing = {str(p) for p, _ in self._discovered}
        for p, resolved_dt in added:
            if str(p) not in existing:
                self._discovered.append((p, resolved_dt))
                existing.add(str(p))
        self._populate_table()

    def _browse_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", self.output_dir_edit.text() or self._default_dir
        )
        if folder:
            self.output_dir_edit.setText(folder)

    def rescan_inputs(self) -> None:
        """Scan ``input_dir_edit`` and refresh the discovered-datasets table."""
        raw = self.input_dir_edit.text().strip()
        if not raw or not Path(raw).exists():
            self._discovered = []
            self._populate_table()
            return
        dt = self.data_type_cbx.currentText()
        dim = self._recipe.dimension if self._recipe and dt == "auto" else None
        self._discovered = discover_batch_inputs(
            raw,
            data_type=dt,
            dimension=dim,
            recursive=self.chk_recursive.isChecked(),
        )
        self._populate_table()

    def _populate_table(self) -> None:
        self.table.setRowCount(len(self._discovered))
        for row, (p, dt) in enumerate(self._discovered):
            chk_item = QTableWidgetItem()
            chk_item.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
            chk_item.setCheckState(Qt.CheckState.Checked)
            chk_item.setData(Qt.ItemDataRole.UserRole, str(p))
            self.table.setItem(row, 0, chk_item)
            self.table.setItem(row, 1, QTableWidgetItem(p.name if p.is_file() else str(p)))
            self.table.setItem(row, 2, QTableWidgetItem(dt))
            self.table.setItem(row, 3, QTableWidgetItem("Pending"))
            self.table.setItem(row, 4, QTableWidgetItem(""))
        self.lbl_input_count.setText(f"{len(self._discovered)} dataset(s) discovered")

    def _set_all_checked(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None:
                item.setCheckState(state)

    def _selected_inputs(self) -> list[tuple[int, Path, str]]:
        selected: list[tuple[int, Path, str]] = []
        for row, (p, dt) in enumerate(self._discovered):
            item = self.table.item(row, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                selected.append((row, p, dt))
        return selected

    def _sync_recipe_from_ui(self) -> Recipe | None:
        if self._recipe is None:
            return None
        formats = [fmt for fmt, chk in self.format_checks.items() if chk.isChecked()]
        if not formats:
            formats = ["p2dat" if self._recipe.dimension == "2D" else "pdat"]
        self._recipe.export_formats = formats
        self._recipe.export_suffix = self.suffix_edit.text() or "_PROCESSED"
        self._recipe.write_noise = self.chk_write_noise.isChecked()
        return self._recipe

    def _start_run(self, *, dry_run: bool) -> None:
        recipe = self._sync_recipe_from_ui()
        if recipe is None:
            QMessageBox.information(self, "Batch Process", "Load or import a recipe first.")
            return
        selected = self._selected_inputs()
        if not selected:
            QMessageBox.information(
                self, "Batch Process", "Select at least one dataset to process."
            )
            return
        out_text = self.output_dir_edit.text().strip()
        if not out_text:
            QMessageBox.information(self, "Batch Process", "Choose an output folder first.")
            return

        self._row_by_path = {str(p): row for row, p, _ in selected}
        for row, _, _ in selected:
            self.table.setItem(row, 3, QTableWidgetItem("Queued"))
            self.table.setItem(row, 4, QTableWidgetItem(""))

        self._cancel_event.clear()
        self.progress_bar.setRange(0, len(selected))
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Previewing..." if dry_run else "Running batch...")
        self._set_running_ui(True)

        self._worker_thread = _BatchWorkerThread(
            [(p, dt) for _, p, dt in selected],
            recipe,
            Path(out_text),
            workers=int(self.workers_spin.value()),
            on_error=self.on_error_cbx.currentText(),
            dry_run=dry_run,
            base_dir=self._recipe_base_dir,
            cancel_event=self._cancel_event,
        )
        self._worker_thread.item_progress.connect(self._on_item_progress)
        self._worker_thread.completed.connect(self._on_run_completed)
        self._worker_thread.failed.connect(self._on_run_failed)
        self._worker_thread.start()

    def _set_running_ui(self, running: bool) -> None:
        self.btn_run.setEnabled(not running)
        self.btn_dry_run.setEnabled(not running)
        self.btn_cancel.setEnabled(running)

    def _cancel_run(self) -> None:
        self._cancel_event.set()
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText("Cancelling...")

    def _on_item_progress(self, done: int, total: int, res: BatchItemResult) -> None:
        self.progress_bar.setMaximum(max(1, total))
        self.progress_bar.setValue(done)
        row = self._row_by_path.get(res.input_path)
        if row is not None:
            status_txt = (
                res.status.upper() if not res.error else f"{res.status.upper()}: {res.error}"
            )
            out_txt = ", ".join(Path(f).name for f in res.output_files)
            self.table.setItem(row, 3, QTableWidgetItem(status_txt))
            self.table.setItem(row, 4, QTableWidgetItem(out_txt))
        self.lbl_status.setText(f"Processed {done}/{total}...")

    def _on_run_completed(self, report: BatchReport) -> None:
        self._last_report = report
        self._set_running_ui(False)
        if report.dry_run:
            self.lbl_status.setText(f"Dry run complete: {report.total} planned dataset(s).")
        elif report.cancelled:
            self.lbl_status.setText(
                f"Cancelled: {report.succeeded} succeeded, {report.failed} failed, {report.skipped} skipped."
            )
        else:
            self.lbl_status.setText(
                f"Done ({report.elapsed_s:.1f}s): {report.succeeded} succeeded, "
                f"{report.failed} failed, {report.skipped} skipped."
            )

    def _on_run_failed(self, err: str) -> None:
        self._set_running_ui(False)
        self.lbl_status.setText(f"Batch error: {err}")
        QMessageBox.critical(self, "Batch Processing Error", err)

    def done(self, result: int) -> None:
        if self._worker_thread is not None and self._worker_thread.isRunning():
            self._cancel_event.set()
            self._worker_thread.wait(5000)
        super().done(result)

    def closeEvent(self, event) -> None:
        if self._worker_thread is not None and self._worker_thread.isRunning():
            self._cancel_event.set()
            self._worker_thread.wait(5000)
        super().closeEvent(event)
