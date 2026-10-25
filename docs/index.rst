PyMORGAN Documentation
======================

**Multidimensional Optical Spectroscopy Graphical Analysis Interface (1D & 2D)**

PyMORGAN provides a unified Python interface and PyQt6 graphical application to load, calibrate,
process, and plot ultrafast time-resolved spectroscopy data. The package is organised by
dimensionality:

- **One-Dimensional Spectroscopy (**\ :mod:`pymorgan.oneD`\ **):** Transient absorption (fs-TA,
  ns-TA / flash photolysis), time-resolved infrared (TRIR), fluorescence up-conversion, and
  polarisation anisotropy / magic-angle spectroscopy — including PyMESS ``.h5`` slow-modulation
  states with per-state single scans and multi-signal fast modulation.
- **Two-Dimensional Spectroscopy (**\ :mod:`pymorgan.twoD`\ **):** General two-dimensional optical
  correlation spectroscopy across homonuclear and cross-regime modalities — **2D-IR**
  (vibrational–vibrational), **2D-ES** (electronic–electronic), **2D-EV**
  (electronic–vibrational), and **2D-VE** (vibrational–electronic) — from raw interferograms
  through closed-form bin-zero detection, apodisation, IRLS phasing, Fourier transformation, and
  interactive 2D single-scan contour array exploration / scan recombination to spectral-diffusion
  (CLS / IvCLS / NLS), 2D Gaussian peak fitting, ROI volume integration, and generalized
  multi-FFCF Kubo lineshape response-function fitting with Variable Projection (VarPro).
- **Wavelength & Pulse-Shaper Calibration (**\ :mod:`pymorgan.cal`\ **):** Non-linear least-squares
  dispersion fitting (linear grating, 2nd/3rd-order polynomial grating, and SF10 glass Sellmeier
  prism models) across 12 pre-configured spectrograph and pulse-shaper setups.
- **Steady-State Reference Spectra (**\ :mod:`pymorgan.steadyState`\ **):** Absorption, emission,
  excitation, and stimulated-emission cross-section (:math:`\tilde{\nu}^4`) overlays.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🚀 Installation & Cross-Platform Suite
      :link: installation
      :link-type: doc

      Install via ``pip`` / ``uv``, deploy standalone **PyMORGAN Suite** installers on
      **Windows**, **macOS**, and **Linux**, or link local Python environments in 1 click.

   .. grid-item-card:: 📈 1D Pump–Probe Pipeline
      :link: user_guide/oneD
      :link-type: doc

      Load 12+ instrument formats (including PyMESS ``.h5`` slow-modulation & multi-signal schemas,
      HARPIA, Helios, TRIR), correct background, solvent, shockwaves & Cauchy chirp dispersion,
      stitch wavelength/time windows, compute time derivatives, perform SVD, and calculate
      polarisation anisotropy.

   .. grid-item-card:: 🗺️ 2D Spectroscopy (2D-IR, 2D-ES, 2D-EV, 2D-VE)
      :link: user_guide/twoD
      :link-type: doc

      Closed-form Fourier Shift Theorem bin-zero detection, apodisation, IRLS phasing,
      interactive 2D single-scan contour array explorer & scan recombination, spectral diffusion
      (CLS / IvCLS / NLS), generalized multi-FFCF Kubo lineshape fitting (VarPro), 2D Gaussian
      peak fitting, ROI volume integration, and 2D difference maps.

   .. grid-item-card:: 🎯 Wavelength & Shaper Calibration
      :link: user_guide/calibration
      :link-type: doc

      Calibrate spectrometer probe arrays and pulse-shaper pump masks across 12 hardware
      setups (including HARPIA UV/Vis presets and auto-routed PyMESS ``.h5`` regimes) in a
      unified 3-column inspector GUI.

   .. grid-item-card:: ⚙️ Steady-State, Aesthetics & Extending
      :link: user_guide/steady_state
      :link-type: doc

      Overlay steady-state FTIR/UV-Vis/emission spectra, customize publication-ready
      Matplotlib profiles via ``settings.toml``, edit the Qt Designer GUI, or register custom
      instrument loaders.

   .. grid-item-card:: 📚 Full Python API Reference
      :link: api/index
      :link-type: doc

      Auto-generated API documentation for ``pymorgan.oneD``, ``pymorgan.twoD``,
      ``pymorgan.cal``, ``pymorgan.steadyState``, and ``pymorgan.settings``.

.. admonition:: Download Complete LaTeX Manual (PDF)
   :class: tip

   In addition to this interactive web documentation, the complete typeset manual is compiled
   from ``docs/*.tex`` and available as a standalone PDF:
   `Download PyMORGAN Manual (main.pdf) <main.pdf>`_.

Quick Example
-------------

.. code-block:: python

   import pymorgan as pm

   # 1. Load & apply publication style profile
   pm.load_settings("settings.toml")
   pm.apply_style()

   # 2. Load 1D transient absorption dataset & subtract pre-zero background
   data = pm.load_1D("testData/testTRIR.pdat", data_type="PDAT")
   data.background_correct(tmin=-20, tmax=-5)

   # 3. Fit and apply Cauchy chirp dispersion correction
   chirp_fit = data.fit_chirp_automatic(deriv_mask=(True, True), use_exp=True)
   data.apply_chirp_correction(chirp_fit, boundary_handling="crop")

   # 4. Plot contour map, transient spectra, and kinetic traces
   data.plot_contour(Zscale=20, Asinh=True)
   data.plot_spectra([0.5, 1, 5, 20, 100], secondary_axis=True)
   ax, t, Y = data.plot_kinetics([1980, 2030], plotStyle="-")
   pm.show_plots()


Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index
   user_guide/oneD
   user_guide/twoD
   user_guide/calibration
   user_guide/steady_state
   user_guide/settings
   user_guide/extending

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/oneD
   api/twoD
   api/calibration
   api/steady_state
   api/settings_and_helpers
