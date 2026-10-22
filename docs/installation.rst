.. _installation:

Installation
============

PyMORGAN requires **Python 3.11 or later** (Python 3.12+ recommended). It can be installed
directly from `PyPI <https://pypi.org/project/pymorgan/>`_, installed from source using
`uv <https://docs.astral.sh/uv/>`_, or deployed on **Windows**, **macOS**, and **Linux** lab
workstations using the standalone **PyMORGAN Suite** installers.

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

Cross-Platform Standalone Installers (PyMORGAN Suite)
-----------------------------------------------------

For lab workstations and end-users who prefer a self-contained installation without manually
managing a Python environment, **PyMORGAN Suite** ships native standalone distributions for
**Windows**, **macOS**, and **Linux** on `GitHub Releases <https://github.com/RJFernandezTeran/PyMORGAN/releases>`_:

Windows (``x64``)
~~~~~~~~~~~~~~~~~

- **Windows Setup Wizard (**\ ``PyMORGAN_Suite_Setup_v*.exe``\ **):** Compiled with Inno Setup;
  registers Windows Start Menu shortcuts, Desktop icons featuring official software emblems, and
  native file associations for 1D datasets (``.pdat``) and 2D datasets (``.p2dat``).
  Double-clicking a ``.pdat`` or ``.p2dat`` file opens it directly in PyMORGAN.
- **Windows Portable Archive (**\ ``PyMORGAN_Suite_v*_Windows_x64_Portable.zip``\ **):**
  Zero-install portable archive containing ``PyMORGAN_Suite.exe``, ``PyMORGAN.exe``, and
  ``PyRATE-TA.exe``.

macOS (``Apple Silicon / Intel``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **macOS Disk Image (**\ ``PyMORGAN_Suite_v*_macOS.dmg``\ **) & App Bundle (**\ ``PyMORGAN Suite.app``\ **):**
  Deep code-signed (``codesign --verify --deep --strict``) macOS application bundle containing
  ``PyMORGAN_Suite``, ``PyMORGAN``, and ``PyRATE-TA`` inside ``Contents/MacOS`` with native
  high-resolution ``pymorgan.icns`` icons.

Linux (``x86_64``)
~~~~~~~~~~~~~~~~~~

- **Linux Tarball & Desktop Installer (**\ ``PyMORGAN_Suite_v*_Linux_x86_64.tar.gz``\ **):**
  Contains standalone binaries (``bin/PyMORGAN_Suite``, ``bin/PyMORGAN``, ``bin/PyRATE-TA``),
  freedesktop ``.desktop`` application launchers (``desktop/*.desktop``), and automated
  ``install.sh`` / ``uninstall.sh`` scripts for desktop menu integration.

Key Features of the Standalone Suite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Zero external dependencies:** Bundles a standalone Python runtime, C-extensions, PyQt6, and
  all scientific libraries. Executables launch directly into the graphical interface without
  exposing background terminal windows.
- **Integrated Suite Launcher:** Includes the unified ``PyMORGAN_Suite`` launcher for one-click
  startup of both **PyMORGAN** (processing and visualisation) and **PyRATE-TA** (kinetic and
  target fitting).
- **1-Click Local Python Environment Setup (**\ ``pymorgan-setup-env``\ **):** Even when installed
  from the standalone Suite installer on Windows, macOS, or Linux, users can open
  **Tools → Setup in Local Python Environment...** in the Suite Launcher (or click
  **Setup in Python...** in the **About** dialog) to automatically configure any local Conda,
  Homebrew, ``uv``, or system Python environment so ``import pymorgan`` and ``import pyrate_ta``
  work directly in Jupyter notebooks and scripts.

Installation from Source
------------------------

Step 1 — Install Python 3.12+
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Windows:** Download the official installer from `python.org <https://www.python.org/downloads/>`_
  (tick *Add Python to PATH*), or use the Windows Store / Python Standalone Builds:

  .. code-block:: bash

     python --version        # should print Python 3.12.x or later

- **macOS:**

  .. code-block:: bash

     brew install python@3.12

- **Linux (Debian/Ubuntu):**

  .. code-block:: bash

     sudo apt update
     sudo apt install python3.12 python3.12-venv python3.12-dev

Step 2 — Install ``uv``
~~~~~~~~~~~~~~~~~~~~~~~

``uv`` is a single self-contained binary that manages virtual environments and dependencies:

- **Windows (PowerShell):**

  .. code-block:: powershell

     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

- **macOS / Linux:**

  .. code-block:: bash

     curl -LsSf https://astral.sh/uv/install.sh | sh

- **Via pip (any OS):**

  .. code-block:: bash

     pip install uv

Step 3 — Clone the Repository & Create the Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/RJFernandezTeran/PyMORGAN.git
   cd PyMORGAN

   uv venv                       # creates .venv/
   uv pip install -e ".[dev]"    # editable install + ruff + pytest
   uv run pymorgan-install-fonts # copy bundled fonts into matplotlib

- ``uv venv`` creates a ``.venv/`` virtual environment in the repository root.
- ``uv pip install -e ".[dev]"`` installs all core runtime dependencies plus development extras
  (``ruff`` and ``pytest``) in editable mode so source edits take effect immediately.
- ``uv run pymorgan-install-fonts`` copies the bundled ``TeX Gyre Heros`` and ``TgTermes`` fonts
  used by the plot style profiles into Matplotlib's font cache.

Step 4 — Verify the Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   uv run python -c "import pymorgan; print(pymorgan.__version__)"
   uv run pymorgan-gui
   uv run pytest
   uv run python scripts/run_checks.py

Dependencies
------------

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
     - C-accelerated tabular data and I/O
   * - ``h5py``
     - ``>=3.10.0``
     - HDF5 file I/O (PyMESS ``.h5`` datasets and calibration files)
   * - ``mpl-axes-aligner``
     - ``>=1.3``
     - Zero-level alignment for dual-axis reference overlays
   * - ``tomlkit``
     - ``>=0.12.0``
     - Comment-preserving ``settings.toml`` read/write
   * - ``Pillow``
     - ``>=10.0.0``
     - Image and animated GIF export
   * - ``PyQt6`` / ``PyQt6-sip``
     - ``>=6.6.0``
     - Graphical user interface framework
   * - ``cmcrameri``
     - ``>=1.8``
     - Perceptually uniform scientific colourmaps
   * - ``defusedxml``
     - ``>=0.7.1``
     - Hardened XML parsing for Qt Designer ``.ui`` files

.. list-table:: Optional Dependency Groups
   :header-rows: 1
   :widths: 24 38 38

   * - Extra Group
     - Packages
     - Install Command
   * - ``dev``
     - ``ruff>=0.16.2``, ``pytest>=9.1.1``
     - ``uv pip install -e ".[dev]"``
   * - ``gui-dev``
     - ``pytest-qt>=4``
     - ``uv pip install -e ".[gui-dev]"``
   * - ``docs``
     - ``sphinx>=7.0``, ``pydata-sphinx-theme>=0.15``, ``sphinx-copybutton>=0.5``, ``sphinx-design>=0.5``
     - ``uv pip install -e ".[docs]"``

Keeping Packages Up to Date
---------------------------

.. code-block:: bash

   uv lock --upgrade     # resolve latest versions, update uv.lock
   uv sync               # install the resolved versions into .venv

Console Scripts
---------------

.. list-table:: Installed Console Entry Points
   :header-rows: 1
   :widths: 32 68

   * - Script
     - Purpose
   * - ``pymorgan`` / ``pymorgan-gui``
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
   * - ``pymorgan-test``
     - Run the built-in self-test verification runner

GUI Layout and Compiled UI
--------------------------

The GUI layout is defined in ``src/pymorgan/gui/main_window.ui`` (Qt Designer XML). For startup
performance (~1.5× faster), PyMORGAN ships a pre-compiled Python translation,
``main_window_ui.py``, generated by ``pyuic6``. Whenever ``main_window.ui`` is modified in Qt
Designer, PyMORGAN detects the timestamp difference on launch and automatically regenerates
``main_window_ui.py`` before opening the window. To force a manual recompile:

.. code-block:: bash

   uv run pyuic6 src/pymorgan/gui/main_window.ui -o src/pymorgan/gui/main_window_ui.py

Software Updates (Releases & Bleeding Edge)
-------------------------------------------

PyMORGAN includes an integrated software update delivery system supporting both official releases
and bleeding-edge development builds:

- **Automatic background checks:** At launch, PyMORGAN performs a non-blocking GitHub API query
  (throttled to once every 24 hours). When a new version is available, a status bar badge
  appears (e.g. ``[PyMORGAN vX.Y.Z Available]``).
- **Release vs. Bleeding-Edge channel:** Users can check for updates at any time via
  **Help → Check for Updates (Release / Bleeding Edge)...** or from the **About** dialog,
  choosing either the latest tagged GitHub Release or syncing directly with the bleeding-edge
  ``main`` branch.
- **In-app download and installation:** Streams the platform-matched installer or package update
  with a real-time progress bar and launches the upgrade automatically.
