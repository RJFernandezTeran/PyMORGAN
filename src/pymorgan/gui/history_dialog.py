"""Dialog for inspecting, reverting, and exporting dataset processing history."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


class ProcessingHistoryDialog(QDialog):
    """Inspect processing steps, revert to any step, or export Python/Recipe."""

    def __init__(
        self,
        dataset: Any,
        parent: QWidget | None = None,
        *,
        on_reverted: Callable[[], None] | None = None,
        default_dir: str | None = None,
    ):
        super().__init__(parent)
        self._dataset = dataset
        self._on_reverted = on_reverted
        self._default_dir = default_dir or ""

        self.setWindowTitle("Processing History & Reproducibility")
        self.resize(860, 560)

        root = QVBoxLayout(self)
        root.setSpacing(8)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        root.addWidget(self.summary_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        # Left pane: step list + step parameters inspector
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.addWidget(QLabel("Recorded Steps:"))

        self.step_list = QListWidget()
        self.step_list.currentRowChanged.connect(self._on_step_selected)
        left_lay.addWidget(self.step_list, 2)

        left_lay.addWidget(QLabel("Step Parameters:"))
        self.params_view = QPlainTextEdit()
        self.params_view.setReadOnly(True)
        mono = QFont("Consolas" if hasattr(QFont, "Monospace") else "")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.params_view.setFont(mono)
        left_lay.addWidget(self.params_view, 1)
        splitter.addWidget(left)

        # Right pane: reproducible Python script preview
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.addWidget(QLabel("Reproducible Python Script:"))

        self.script_preview = QPlainTextEdit()
        self.script_preview.setReadOnly(True)
        self.script_preview.setFont(mono)
        right_lay.addWidget(self.script_preview, 1)
        splitter.addWidget(right)
        splitter.setSizes([340, 520])

        # Bottom action buttons
        btn_row = QHBoxLayout()
        self.btn_revert = QPushButton("Revert to Selected Step")
        self.btn_revert.clicked.connect(self._revert_selected)
        btn_row.addWidget(self.btn_revert)

        self.btn_reset = QPushButton("Reset to Raw")
        self.btn_reset.clicked.connect(self._reset_raw)
        btn_row.addWidget(self.btn_reset)

        btn_row.addStretch(1)

        self.btn_copy_py = QPushButton("Copy Python Script")
        self.btn_copy_py.clicked.connect(self._copy_script)
        btn_row.addWidget(self.btn_copy_py)

        self.btn_save_py = QPushButton("Save Python Script (.py)...")
        self.btn_save_py.clicked.connect(self._save_script)
        btn_row.addWidget(self.btn_save_py)

        self.btn_save_recipe = QPushButton("Save as Recipe (.pmrecipe.toml)...")
        self.btn_save_recipe.clicked.connect(self._save_recipe)
        btn_row.addWidget(self.btn_save_recipe)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_close)

        root.addLayout(btn_row)
        self.refresh()

    def refresh(self) -> None:
        """Reload steps, parameter view, and generated Python script."""
        steps = list(getattr(self._dataset, "history", []) or [])
        dim = "2D" if hasattr(self._dataset, "Z_R") else "1D"
        src = getattr(self._dataset, "source", None)
        src_name = Path(src).name if src else "<in-memory>"
        self.summary_label.setText(
            f"<b>Dataset ({dim}):</b> {src_name} &nbsp;|&nbsp; "
            f"<b>Steps:</b> {len(steps)} &nbsp;|&nbsp; "
            f"<b>Can Undo:</b> {'Yes' if getattr(self._dataset, 'can_undo', False) else 'No'} &nbsp;|&nbsp; "
            f"<b>Can Redo:</b> {'Yes' if getattr(self._dataset, 'can_redo', False) else 'No'}"
        )

        self.step_list.blockSignals(True)
        self.step_list.clear()
        raw_item = QListWidgetItem("[0] Raw Loaded Baseline")
        raw_item.setData(Qt.ItemDataRole.UserRole, -1)
        self.step_list.addItem(raw_item)
        for idx, st in enumerate(steps):
            item = QListWidgetItem(f"[{idx + 1}] {st.label} ({st.op})")
            item.setData(Qt.ItemDataRole.UserRole, idx)
            self.step_list.addItem(item)
        self.step_list.setCurrentRow(self.step_list.count() - 1)
        self.step_list.blockSignals(False)

        self._on_step_selected(self.step_list.currentRow())

        if hasattr(self._dataset, "to_python_script"):
            self.script_preview.setPlainText(self._dataset.to_python_script())
        else:
            self.script_preview.setPlainText("# Python script generation unavailable.")

        self.btn_reset.setEnabled(bool(steps))

    def _on_step_selected(self, row: int) -> None:
        steps = list(getattr(self._dataset, "history", []) or [])
        if row <= 0 or row - 1 >= len(steps):
            self.params_view.setPlainText("Initial unmodified dataset state after load.")
            self.btn_revert.setEnabled(bool(steps) and row == 0)
            return
        st = steps[row - 1]
        payload = {
            "op": st.op,
            "label": st.label,
            "timestamp_utc": st.timestamp_utc,
            "params": st.params,
        }
        self.params_view.setPlainText(json.dumps(payload, indent=2))
        # Enable Revert whenever selected step is earlier than the latest step
        self.btn_revert.setEnabled(row - 1 < len(steps) - 1)

    def _revert_selected(self) -> None:
        row = self.step_list.currentRow()
        if row < 0:
            return
        try:
            if row == 0:
                self._dataset.reset_to_raw()
            else:
                self._dataset.revert_to_step(row - 1)
        except Exception as exc:
            QMessageBox.warning(self, "Revert Failed", str(exc))
            return
        self.refresh()
        if self._on_reverted is not None:
            self._on_reverted()

    def _reset_raw(self) -> None:
        try:
            self._dataset.reset_to_raw()
        except Exception as exc:
            QMessageBox.warning(self, "Reset Failed", str(exc))
            return
        self.refresh()
        if self._on_reverted is not None:
            self._on_reverted()

    def _copy_script(self) -> None:
        cb = QApplication.clipboard()
        if cb is not None:
            cb.setText(self.script_preview.toPlainText())

    def _save_script(self) -> None:
        src = getattr(self._dataset, "source", None)
        base = (
            str(Path(src).with_suffix(".py"))
            if src
            else str(Path(self._default_dir) / "reproduce.py")
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Python Script",
            base,
            "Python Scripts (*.py);;All Files (*)",
        )
        if not path:
            return
        Path(path).write_text(self.script_preview.toPlainText(), encoding="utf-8")

    def _save_recipe(self) -> None:
        from ..recipe import Recipe

        src = getattr(self._dataset, "source", None)
        stem = Path(src).stem if src else "processing_recipe"
        default_path = (
            str(Path(src).parent / f"{stem}.pmrecipe.toml")
            if src
            else str(Path(self._default_dir) / f"{stem}.pmrecipe.toml")
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processing Recipe",
            default_path,
            "PyMORGAN Recipes (*.pmrecipe.toml);;TOML Files (*.toml);;All Files (*)",
        )
        if not path:
            return
        target = Path(path)
        if not target.name.lower().endswith(".toml"):
            target = target.with_name(target.name + ".pmrecipe.toml")
        recipe = Recipe.from_dataset(self._dataset, name=target.stem.replace(".pmrecipe", ""))
        recipe.save(target)
