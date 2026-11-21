.. _user_guide_workflows:

Reproducible Workflows, Interoperability & Automation
=====================================================

PyMORGAN combines interactive GUI convenience with strict script-level reproducibility across both
1D (:class:`~pymorgan.oneD.dataset.Dataset1D`) and 2D (:class:`~pymorgan.twoD.dataset.Dataset2D`)
pipelines.

Universal Format Auto-Detection & Drag-and-Drop
-----------------------------------------------

Instead of memorising instrument type tokens, :func:`pymorgan.detect_format` probes files and
directories by signature (magic headers, NetCDF4/HDF5 schemas, folder markers) and returns ranked
:class:`~pymorgan.detect.FormatMatch` results. :func:`pymorgan.load_auto` dispatches directly to the
highest-confidence 1D, 2D, or steady-state loader:

.. code-block:: python

   import pymorgan as pm

   match = pm.detect_format("measurement.ufs")
   print(match.dimension, match.data_type, match.confidence)
   ds = pm.load_auto("measurement.ufs")

In ``pymorgan-gui``:

- **Open File... (**\ ``Ctrl+O``\ **)** and **drag-and-drop** onto the main window invoke
  ``open_external_path()``, automatically routing files and folders to 1D datasets, 2D maps,
  steady-state spectra, ``.pmproj`` workspace sessions, or ``.pmrecipe.toml`` batch recipes.
- When multiple formats tie at the top confidence score, the GUI prompts for disambiguation.

Processing History, Undo/Redo & Python Script Export
----------------------------------------------------

Every mutating operation on :class:`~pymorgan.oneD.dataset.Dataset1D` and
:class:`~pymorgan.twoD.dataset.Dataset2D` (including ``background_correct``, ``shift_t0``,
``crop``, ``mask_probe_regions``, ``remove_scatter``, ``apply_chirp_correction``,
``subtract_solvent``, ``subtract_shockwave``, ``process``, ``subtract_spectrum``,
``set_active_state``, and ``set_active_detector``) is recorded automatically by
:class:`~pymorgan.provenance.HistoryManager` with copy-on-write snapshot capture.

.. code-block:: python

   ds = pm.load_1D("scan.pdat", data_type="PDAT")
   ds.background_correct(tmin=-20.0, tmax=-5.0, do_correct=True)
   ds.shift_t0(0.25)

   ds.undo()        # reverts shift_t0
   ds.redo()        # re-applies shift_t0

   script = ds.to_python_script()
   print(script)    # self-contained Python script importing pymorgan + settings overrides

- **Undo / Redo (**\ ``Ctrl+Z`` / ``Ctrl+Shift+Z`` / ``Ctrl+Y``\ **):** Accessible from the
  *Edit* menu, displaying the active step label (e.g. ``Undo Shift t0``).
- **Processing History Inspector (**\ *Edit → Processing History...*\ **):** Inspect timestamps,
  parameters, preview or revert to any intermediate state, copy/save a runnable Python script, or
  export the step sequence as a reusable ``.pmrecipe.toml`` recipe.
- **Reset to Raw (**\ *Edit → Reset Dataset to Raw*\ **):** Restores the initial state recorded at
  load time (accessible programmatically via ``ds.reset_to_raw()``).

Saved Processing Recipes (``.pmrecipe.toml``) & Batch Execution
---------------------------------------------------------------

A :class:`~pymorgan.recipe.Recipe` serialises an ordered pipeline of processing steps and export
targets in human-readable TOML format (``.pmrecipe.toml``). Auxiliary files (e.g. solvent
references, chirp fits, calibration files, or 2D reference maps) are stored as portable relative
paths alongside absolute fallbacks.

.. code-block:: python

   from pymorgan.recipe import Recipe

   recipe = Recipe.from_dataset(ds, name="routine_trir", export_formats=["pdat", "hdf5"])
   recipe.save("routine_trir.pmrecipe.toml")

Apply saved recipes across files or folders via the GUI (**File → Batch Process Folder...**, or by
dropping a ``.pmrecipe.toml`` file onto the window), programmatically via :mod:`pymorgan.batch`, or
from the command line using ``pymorgan batch`` / ``pymorgan-batch``:

.. code-block:: bash

   pymorgan batch --recipe routine_trir.pmrecipe.toml --input ./scans --output ./out --recursive --workers 4 --format hdf5
   pymorgan-batch --recipe routine_trir.pmrecipe.toml --input ./scans --output ./out --dry-run

Batch runs emit structured ``batch_report.csv`` and ``batch_report.json`` summaries tracking per-item
status, runtime, and exported paths, with ``--on-error skip|stop`` resilience.

Project Workspace Sessions (``.pmproj``)
----------------------------------------

Use **File → Save Project** (``Ctrl+S``), **File → Save Project As...** (``Ctrl+Shift+S``), and
**File → Open Project (.pmproj)...** to save and reopen the complete workspace state:

- Active root folder (stored portably relative to the ``.pmproj`` file when moved together),
- Loaded 1D and 2D datasets, slow-modulation state, active 2D detector channel, and complete
  processing history,
- Steady-state / FTIR overlays, plot axis limits, Z-scale / symmetric bounds, and per-project
  settings overrides.

Missing paths are flagged gracefully with non-fatal warnings and optional interactive relocation.

Recent Folders & Zero-I/O Dataset Filtering
-------------------------------------------

- **Recent Folders (**\ *File → Recent Folders*\ **):** Tracks most-recently-used data directories
  in ``settings.toml`` (``recent_dirs`` / ``max_recent_dirs``) and populates path autocompletion on
  the ``RootDir`` box.
- **Dataset Filter (**\ ``Ctrl+F`` / ``Esc``\ **):** Both 1D and 2D browser panels include an inline
  filter box (*Filter datasets... (Ctrl+F)*) backed by an in-memory ``QSortFilterProxyModel`` for
  instant substring filtering across large measurement folders without filesystem re-scans.
