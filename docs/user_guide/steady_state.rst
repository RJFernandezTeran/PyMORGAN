.. _user_guide_steady_state:

Steady-State Spectra
====================

The :mod:`pymorgan.steadyState` module provides lightweight wrappers for absorption, emission,
and excitation spectra. Data are held in :class:`~pymorgan.steadyState.dataset.Spectrum` and
:class:`~pymorgan.steadyState.dataset.SpectrumSeries` objects, file formats are resolved through
a loader registry, and presentation is driven by the shared :class:`~pymorgan.settings.Settings`.

Loading Steady-State Spectra
----------------------------

.. code-block:: python

   import pymorgan as pm

   # List registered steady-state loaders ('csv', 'opus', 'uv_vis', 'fluorimeter')
   print(pm.available_spectrum_loaders())

   # Load a single absorption or emission spectrum
   sp = pm.load_spectrum("sample.csv", kind="absorption", x_units="nm", label="Sample A")
   sp.plot()

   # Convert spontaneous emission to stimulated emission cross-section (nu^4 scaling)
   em = pm.load_spectrum("emission.csv", kind="emission", x_units="nm")
   em_stim = em.to_stimulated_emission().normalised()

   # Overlay on 1D transient absorption spectra
   data.plot_spectra([0.5, 1, 5], ss_overlay=sp)
