.. _installation:

Installation
============

PyMORGAN requires **Python 3.11 or later** (Python 3.12+ recommended). It can be installed
directly from `PyPI <https://pypi.org/project/pymorgan/>`_, installed from source using
`uv <https://docs.astral.sh/uv/>`_, or deployed on Windows lab workstations using the
standalone **PyMORGAN Suite** installer.

Quick Install via PyPI (Recommended)
------------------------------------

To install the latest official release into your active Python environment:

.. code-block:: bash

   pip install pymorgan
   # or with uv:
   uv pip install pymorgan

   # Register bundled fonts into matplotlib:
   pymorgan-install-fonts

You can also launch the graphical interface directly without prior installation using ``uvx``:

.. code-block:: bash

   uvx --from pymorgan pymorgan-gui

Installation from Source
------------------------

1. Clone the Repository
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/RJFernandezTeran/PyMORGAN.git
   cd PyMORGAN

2. Create the Virtual Environment and Install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using `uv <https://docs.astral.sh/uv/>`_ from the repository root:

.. code-block:: bash

   uv venv                       # creates .venv/
   uv pip install -e ".[dev]"    # editable install + ruff + pytest
   uv run pymorgan-install-fonts # copy bundled fonts into matplotlib

3. Verify the Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   uv run python -c "import pymorgan; print(pymorgan.__version__)"
   uv run pymorgan-gui
   uv run pytest
   uv run python scripts/run_checks.py

Core Dependencies
-----------------

.. list-table:: Core Runtime Dependencies
   :header-rows: 1
   :widths: 28 18 54

   * - Package
     - Minimum
     - Purpose
   * - ``numpy``
     - ``>=1.26.0``
     - Multi-dimensional array operations
   * - ``scipy``
     - ``>=1.12.0``
     - Signal processing, Savitzky–Golay filters, and non-linear least squares
   * - ``matplotlib``
     - ``>=3.8.0``
     - Publication-grade plotting back-end
   * - ``pandas``
     - ``>=2.1.0``
     - C-accelerated tabular I/O
   * - ``h5py``
     - ``>=3.10.0``
     - HDF5 file I/O (PyMESS ``.h5`` datasets and calibration files)
   * - ``mpl-axes-aligner``
     - ``>=1.3``
     - Zero-level alignment for dual-axis reference overlays
   * - ``tomlkit``
     - ``>=0.12.0``
     - Comment-preserving ``settings.toml`` read/write
   * - ``PyQt6`` / ``PyQt6-sip``
     - ``>=6.6.0``
     - Graphical user interface framework
   * - ``cmcrameri``
     - ``>=1.8``
     - Perceptually uniform scientific colourmaps
   * - ``defusedxml``
     - ``>=0.7.1``
     - Hardened XML parsing for Qt Designer ``.ui`` files

Console Scripts
---------------

.. list-table:: Installed Console Entry Points
   :header-rows: 1
   :widths: 32 68

   * - Script
     - Purpose
   * - ``pymorgan-gui``
     - Launch the PyQt6 graphical interface
   * - ``pymorgan-suite``
     - Launch the unified PyMORGAN + PyRATE-TA Suite Launcher
   * - ``pymorgan-install-fonts``
     - Install bundled ``TeX Gyre Heros`` / ``TgTermes`` fonts into Matplotlib's font cache
   * - ``pymorgan-edit-gui``
     - Open ``src/pymorgan/gui/main_window.ui`` in Qt Designer
   * - ``pymorgan-settings``
     - Standalone settings editor (no dataset required)
   * - ``pymorgan-setup-env``
     - 1-click local Python environment setup wizard (CLI or GUI)

Windows Standalone Installer (PyMORGAN Suite)
---------------------------------------------

For lab workstations and end-users who prefer a self-contained installation without managing a
Python environment, PyMORGAN provides **PyMORGAN Suite Setup** (``PyMORGAN_Suite_Setup_*.exe``)
and an accompanying portable archive (``PyMORGAN_Suite_*_Portable.zip``):

- **Zero external dependencies:** Bundles a standalone Python runtime, C-extensions, PyQt6, and
  all scientific libraries.
- **Integrated Suite Launcher:** Includes ``PyMORGAN_Suite.exe`` for one-click startup of both
  **PyMORGAN** (processing and visualisation) and **PyRATE-TA** (kinetic and target fitting),
  plus a **1-Click Local Python Environment Setup** dialog under *Tools*.
- **System integration:** Registers Windows Start Menu shortcuts, desktop icons, and native file
  associations for 1D datasets (``.pdat``) and 2D-IR datasets (``.p2dat``).
- **In-app software updates:** Automatically checks GitHub Releases in the background (throttled
  to once per 24 hours) and provides one-click download and installation.
