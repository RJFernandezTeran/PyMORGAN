PyMORGAN Documentation
======================

**Plotting, Calibration, and Analysis of Ultrafast Time-Resolved Spectroscopy Data (1D & 2D)**

PyMORGAN is a scientific Python package and PyQt6 graphical user interface for loading,
calibrating, processing, and visualising one-dimensional (pump–probe TA, TRIR, nsTA,
fluorescence upconversion) and two-dimensional (2D-IR, 2D-ES) time-resolved spectroscopy
datasets, as well as steady-state reference spectra.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🚀 Installation & Setup
      :link: installation
      :link-type: doc

      Install via ``pip`` / ``uv``, set up local Python environments in 1 click, or run the
      standalone Windows **PyMORGAN Suite** installer.

   .. grid-item-card:: 📈 1D Pump–Probe Pipeline
      :link: user_guide/oneD
      :link-type: doc

      Load 12+ instrument formats (including PyMESS ``.h5``, HARPIA, Helios, TRIR), correct
      background & Cauchy chirp dispersion, stitch windows, perform SVD, and calculate
      polarisation anisotropy.

   .. grid-item-card:: 🗺️ 2D-IR & 2D Spectroscopy
      :link: user_guide/twoD
      :link-type: doc

      Closed-form Fourier Shift Theorem bin-zero detection, apodisation, IRLS phasing,
      spectral diffusion (CLS / IvCLS / NLS), and Kubo lineshape analysis.

   .. grid-item-card:: 🎯 Wavelength & Shaper Calibration
      :link: user_guide/calibration
      :link-type: doc

      Calibrate spectrometer probe arrays and pulse-shaper pump masks across 12 hardware
      setups (including HARPIA UV/Vis presets and auto-routed PyMESS ``.h5`` regimes).

   .. grid-item-card:: ⚙️ Steady-State, Aesthetics & Extending
      :link: user_guide/steady_state
      :link-type: doc

      Overlay steady-state FTIR/UV-Vis/emission spectra, customize publication-ready
      Matplotlib profiles via ``settings.toml``, or register custom instrument loaders.

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
