"""Shared pytest fixtures. Forces a headless matplotlib backend and offscreen Qt platform."""

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import matplotlib

matplotlib.use("Agg")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))  # make `import synthetic` work
import synthetic  # noqa: E402

import pymorgan as pm  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_settings():
    """Isolate tests from each other's settings mutations."""
    pm.set_settings(pm.Settings())
    yield
    pm.set_settings(pm.Settings())


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


@pytest.fixture(autouse=True)
def _cleanup_qt():
    yield
    try:
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            for widget in app.topLevelWidgets():
                widget.close()
                widget.deleteLater()
            app.processEvents()
    except Exception:
        pass


@pytest.fixture
def pdat(tmp_path):
    """Ground-truth metadata for a freshly written synthetic .pdat."""
    return synthetic.make_synthetic_pdat(tmp_path / "syn.pdat")


@pytest.fixture
def dataset(pdat):
    return pm.load_1D(pdat["path"], data_type="PDAT")


@pytest.fixture
def corrected(dataset):
    return dataset.background_correct(tmin=-20, tmax=-1, do_correct=True)


@pytest.fixture
def abs_csv(tmp_path):
    return synthetic.make_spectrum_csv(tmp_path / "abs.csv")
