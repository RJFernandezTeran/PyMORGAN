.. _user_guide_extending:

Extending PyMORGAN
==================

Adding a Custom Data-Format Loader
----------------------------------

A loader maps a file path to the fields required by a dataset. Registering it with the matching
decorator makes its format token appear in ``pm.available_*_loaders()`` and in the GUI dropdowns:

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

.. code-block:: python

   import numpy as np
   import pymorgan as pm
   from pymorgan import helpers as hlp

   @pm.register_loader("MyTA")
   def read_my_ta(path):
       raw = np.loadtxt(path, delimiter=",")
       delays = raw[1:, 0]
       probe  = raw[0, 1:]
       Zavg   = raw[1:, 1:][..., np.newaxis]
       Units  = hlp.units2dic("nm", "ps", "mOD")
       Nscans = np.nan
       Zss    = np.nan * np.zeros_like(Zavg[..., np.newaxis])
       Zstdv  = np.zeros_like(Zavg)
       return Zavg, delays, probe, Units, Nscans, Zss, Zstdv
