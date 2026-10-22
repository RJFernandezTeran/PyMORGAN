.. _user_guide_extending:

Extending PyMORGAN
==================

This guide explains how to (i) modify the GUI layout and behaviour and (ii) add a reader for a
new instrument file format.

Editing the GUI Layout (Qt Designer)
------------------------------------

The window layout lives in a single Qt Designer file, ``src/pymorgan/gui/main_window.ui``; the
controller ``src/pymorgan/gui/main_window.py`` binds to widgets **by their object name**.

Opening the Layout
~~~~~~~~~~~~~~~~~~

Launch Qt Designer directly via the console script:

.. code-block:: bash

   pymorgan-edit-gui
   # or via uvx:
   uvx --from pyside6-essentials pyside6-designer src/pymorgan/gui/main_window.ui

Moving, Resizing, or Adding Widgets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Moving or resizing:** Rearranging layouts in Qt Designer breaks nothing as long as widget
  object names are unchanged. On next launch, PyMORGAN automatically recompiles
  ``main_window_ui.py`` if ``main_window.ui`` is newer.
- **Adding a widget:**

  1. Add the widget in Designer and assign a clear ``objectName`` (e.g. ``PP_exportFig_btn``).
  2. In ``main_window.py``, inside ``_wire()``, connect it using the defensive helpers:

     .. code-block:: python

        self._connect("PP_exportFig_btn", self.export_figure)   # buttons (clicked)
        self._connect_text("SomeField", self.on_text_changed)   # QLineEdit (editingFinished)

- **Promoting a plot area:** Promote a ``QWidget`` to ``WidgetPlot`` with header
  ``pymorgan.gui.widgetplot``. In Python it exposes ``.figure``, ``.ax``, and ``.canvas``.
- **Icons:** Place PNG/ICO files in ``src/pymorgan/gui/icons/`` and register them in
  ``_BUTTON_ICONS`` in ``main_window.py`` (do not use compiled ``.qrc`` resources).

Verify GUI integrity headlessly after any edit:

.. code-block:: bash

   QT_QPA_PLATFORM=offscreen uv run pytest tests/test_gui.py

Adding a New Data-Format Parser
-------------------------------

A **loader** maps a file path to the fields a dataset needs. Registering it with the matching
decorator makes its data-type name appear in ``pm.available_*_loaders()`` and in the GUI's
data-type combo box:

.. list-table:: Loader Registration Contracts
   :header-rows: 1
   :widths: 18 32 50

   * - Branch
     - Decorator
     - Return Tuple
   * - **1D**
     - :func:`@pm.register_loader <pymorgan.register_loader>`
     - ``(Zavg_R, delays, probe, Units, Nscans, Zss_R, Zstdv)``
   * - **2D**
     - :func:`@pm.register_map_loader <pymorgan.register_map_loader>`
     - ``(Z, pump, probe, delays, units, freq_units)``
   * - **Steady-State**
     - :func:`@pm.register_spectrum_loader <pymorgan.register_spectrum_loader>`
     - ``(x, y, x_units)``

Writing a 1D Reader
~~~~~~~~~~~~~~~~~~~

The signal is stored as ``[Ndelays x Npixels x Ndetectors]``; the unit dictionary is built with
:func:`pymorgan.helpers.units2dic`:

.. code-block:: python

   import numpy as np
   import pymorgan as pm
   from pymorgan import helpers as hlp

   @pm.register_loader("MyTA")
   def read_my_ta(path):
       raw = np.loadtxt(path, delimiter=",")
       delays = raw[1:, 0]                       # first column = delays
       probe  = raw[0, 1:]                       # first row    = probe axis
       Zavg   = raw[1:, 1:][..., np.newaxis]     # add detector axis
       Units  = hlp.units2dic("nm", "ps", "mOD") # (probe, time, signal) units
       Nscans = np.nan
       Zss    = np.nan * np.zeros_like(Zavg[..., np.newaxis])
       Zstdv  = np.zeros_like(Zavg)
       return Zavg, delays, probe, Units, Nscans, Zss, Zstdv

The data type is then available everywhere:

.. code-block:: python

   data = pm.load_1D("scan.myta", data_type="MyTA")

To expose a custom file extension in the GUI file browser, add an entry to ``_DATATYPE_GLOB`` in
``src/pymorgan/gui/mw_common.py`` (e.g. ``"MyTA": "*.myta"``).
