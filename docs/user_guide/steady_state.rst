.. _user_guide_steady_state:

Steady-State Spectra
====================

The :mod:`pymorgan.steadyState` module provides lightweight wrappers for absorption, emission,
and excitation spectra. Its design mirrors the one-dimensional branch (:ref:`user_guide_oneD`):
data are held in a plain object (:class:`~pymorgan.steadyState.dataset.Spectrum`), file formats
are resolved through a loader registry, and presentation is driven by the shared
:class:`~pymorgan.settings.Settings`.

Data Model
----------

``SpectrumKind``
~~~~~~~~~~~~~~~~

:class:`~pymorgan.steadyState.dataset.SpectrumKind` is a string enum that labels the physical
quantity measured:

.. code-block:: python

   class SpectrumKind(StrEnum):
       ABSORPTION = "absorption"
       EMISSION   = "emission"
       EXCITATION = "excitation"

``Spectrum``
~~~~~~~~~~~~

A single steady-state measurement. The spectral axis ``x`` and signal ``y`` are stored as NumPy
arrays; the remaining fields are metadata:

.. list-table:: Attributes of a ``Spectrum`` Object
   :header-rows: 1
   :widths: 22 28 50

   * - Attribute
     - Type
     - Meaning
   * - ``x``
     - ``ndarray``
     - Spectral axis (values in ``x_units``)
   * - ``y``
     - ``ndarray``
     - Signal (absorbance or intensity)
   * - ``kind``
     - ``SpectrumKind``
     - Measurement type (``absorption``, ``emission``, ``excitation``)
   * - ``x_units``
     - ``str``
     - Spectral-axis unit token (``"nm"``, ``"cm-1"``, ``"eV"``)
   * - ``label``
     - ``str | None``
     - Legend label
   * - ``source``
     - ``str | None``
     - File path (set automatically on load)

``SpectrumSeries``
~~~~~~~~~~~~~~~~~~

An ordered list of :class:`~pymorgan.steadyState.dataset.Spectrum` objects for overlay plots. It
supports iteration, indexing, and the standard sequence protocol. Construction from multiple
files returns a :class:`~pymorgan.steadyState.dataset.SpectrumSeries` directly.

Loading and the Loader Registry
-------------------------------

The module exposes a loader registry analogous to the 1D and 2D registries. The default ``"csv"``
reader handles comma- and tab-delimited files with automatic delimiter detection. Built-in
loaders include:

- ``csv``: Generic delimiter-detected text files
- ``opus``: Bruker OPUS binary files
- ``uv_vis``: Perkin-Elmer / Shimadzu UV–Vis text exports
- ``fluorimeter``: Horiba FluorEssence ``.fes`` files

.. code-block:: python

   import pymorgan as pm

   # List available loaders
   print(pm.available_spectrum_loaders())

   # Load a single spectrum
   sp = pm.load_spectrum("sample.csv", kind="absorption")

   # Specify units explicitly (overrides the reader's guess)
   sp = pm.load_spectrum("sample.csv", kind="emission", x_units="nm", label="Sample A")

   # Load via class method
   sp2 = pm.Spectrum.from_file("sample.opus", kind="absorption", data_type="opus")

Custom readers are registered with :func:`@pm.register_spectrum_loader <pymorgan.register_spectrum_loader>`:

.. code-block:: python

   @pm.register_spectrum_loader("MySpectrometer")
   def read_my_spectrometer(path):
       ...
       return x, y, x_units    # x_units may be None (falls back to "nm")

Transformations
---------------

Two per-``Spectrum`` transformations return a new :class:`~pymorgan.steadyState.dataset.Spectrum`
(leaving the original unchanged):

- :meth:`~pymorgan.steadyState.dataset.Spectrum.normalised`: Scales the signal so that
  :math:`\max(|y|) = 1`.
- :meth:`~pymorgan.steadyState.dataset.Spectrum.to_stimulated_emission`: Computes the
  stimulated-emission cross-section :math:`F_{\text{stim}} \propto \tilde{\nu}^4 \, F_{\text{spont}}`
  (valid for emission spectra on a wavelength axis). Useful when overlaying on transient
  absorption spectra.

``SpectrumSeries.normalised()`` applies the same operation to every member and returns a new
series.

Integration with the 1D and 2D Pipelines
----------------------------------------

A :class:`~pymorgan.steadyState.dataset.Spectrum` can be passed directly to
``Dataset1D.plot_spectra`` as a steady-state overlay or to ``Dataset2D.plot_map`` as a top
panel spectrum:

.. code-block:: python

   ftir = pm.load_spectrum("background_FTIR.csv", kind="absorption")
   data1d.plot_spectra([0.5, 1, 5], ss_overlay=ftir)
   data2d.plot_map(0.5, top_spectrum=ftir, top_label="FTIR")

Plotting Steady-State Series
----------------------------

.. code-block:: python

   import pymorgan as pm

   pm.load_settings("settings.toml")
   pm.apply_style()

   # Single spectrum
   sp = pm.load_spectrum("sample.csv", kind="absorption", label="Sample")
   sp.plot()

   # Normalised emission
   em = pm.load_spectrum("emission.csv", kind="emission", label="Emission")
   em.normalised().plot()

   # Series: multiple spectra coloured along a colourmap
   series = pm.SpectrumSeries.from_files(
       ["a.csv", "b.csv", "c.csv"],
       kind="absorption",
       labels=["10 uM", "50 uM", "100 uM"],
   )
   series.plot(cmap="viridis", normalise=True)

   pm.show_plots()
