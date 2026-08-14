"""Application entry point for the standalone PyMORGAN settings editor."""

import os
import sys
from pathlib import Path

# Pin matplotlib and widgets to the PyQt6 Qt binding
os.environ.setdefault("QT_API", "pyqt6")

def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else list(sys.argv)

    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QMessageBox, QVBoxLayout

    app = QApplication.instance() or QApplication(args)

    import pymorgan as pm

    from .settings_panel import SettingsPanel

    # Load settings from settings.toml
    cfg = pm.settings_path()
    pm.load_settings(cfg)

    dlg = QDialog()
    dlg.setWindowTitle("PyMORGAN Settings Editor")
    dlg.resize(600, 500)

    _icon = Path(__file__).with_name("icons") / "pirate-hat.png"
    if _icon.exists():
        dlg.setWindowIcon(QIcon(str(_icon)))

    layout = QVBoxLayout(dlg)
    panel = SettingsPanel(dlg)
    layout.addWidget(panel)

    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close
    )
    save_btn = buttons.button(QDialogButtonBox.StandardButton.Save)
    save_btn.setText("Save permanently")

    def _save_settings():
        try:
            pm.save_settings(cfg)
            QMessageBox.information(dlg, "Settings Saved", f"Settings successfully saved to {cfg}")
        except Exception as exc:
            QMessageBox.critical(dlg, "Save Failed", str(exc))

    buttons.accepted.connect(_save_settings)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)

    dlg.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
