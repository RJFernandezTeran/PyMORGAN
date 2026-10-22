.. _user_guide_settings:

Settings & Aesthetics
=====================

Every presentation choice that is not intrinsic to the data lives in a single
:class:`~pymorgan.settings.Settings` object: the Matplotlib style profile (``regular``,
``poster``, ``inset``), label and unit conventions, colourmaps, axis units, and spectral-diffusion
parameters.

.. code-block:: python

   import pymorgan as pm

   pm.load_settings("settings.toml")     # read a TOML configuration file
   pm.update_settings(cmap="Seismic", quick_plots=True)
   s = pm.get_settings()                 # retrieve active Settings instance
   pm.save_settings("settings.toml")     # write back while preserving TOML comments

   with pm.use_settings(profile="poster", delta_a_units="mOD"):
       data.plot_contour()
