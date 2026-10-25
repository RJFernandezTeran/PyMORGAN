.. _user_guide_oneD:

One-Dimensional Data (Pump–Probe, TRIR, nsTA & Anisotropy)
==========================================================

Pipeline Overview
-----------------

The one-dimensional branch is organised as a **load → process → plot → analyse** pipeline.
Each stage is a module under :mod:`pymorgan.oneD`, and the :class:`~pymorgan.oneD.dataset.Dataset1D`
object ties them together: it holds the raw arrays and exposes one set of methods per stage.
The stage functions take the dataset as their first argument, so they remain reusable on their
own and new routines integrate with the whole pipeline.

.. list-table:: The 1D Pipeline Stages and Where They Live
   :header-rows: 1
   :widths: 18 34 48

   * - Stage
     - Module
     - Entry Point(s)
   * - **load**
     - :mod:`pymorgan.oneD.load`
     - :func:`pymorgan.load_1D`, :meth:`Dataset1D.from_file <pymorgan.oneD.dataset.Dataset1D.from_file>`
   * - **process**
     - :mod:`pymorgan.oneD.process`, :mod:`pymorgan.oneD.chirp`, :mod:`pymorgan.oneD.stitch`, :mod:`pymorgan.oneD.svd`
     - :meth:`Dataset1D.background_correct <pymorgan.oneD.dataset.Dataset1D.background_correct>`,
       :func:`~pymorgan.oneD.chirp.apply_chirp_correction`,
       :func:`~pymorgan.oneD.process.subtract_shockwave`,
       :func:`~pymorgan.oneD.process.time_derivative`
   * - **plot**
     - :mod:`pymorgan.oneD.plot`
     - ``Dataset1D.plot_contour``, ``plot_surface``, ``plot_spectra``, ``plot_kinetics``, ``plot_species_spectra``
   * - **analyse**
     - *PyRATE-TA*
     - Companion package (`PyRATE-TA <https://rjfernandezteran.github.io/PyRATE-TA/>`_) via ``Action → Launch PyRATE-TA``

Data Model
----------

A one-dimensional measurement is represented by a :class:`~pymorgan.oneD.dataset.Dataset1D`
object, which bundles the arrays produced by a loader. The signal is stored as a
three-dimensional array to generalise over multiple detectors:

.. math::

   Z \in \mathbb{R}^{N_\tau \times N_\lambda \times N_d}

indexed by delay :math:`\tau`, probe coordinate :math:`\lambda` (wavelength, wavenumber, energy,
or frequency), and detector :math:`d`. The accompanying axes are the delay vector
:math:`\tau \in \mathbb{R}^{N_\tau}` and the probe vector :math:`\lambda \in \mathbb{R}^{N_\lambda}`.

.. list-table:: Fields Returned by a 1D Loader and Stored on ``Dataset1D``
   :header-rows: 1
   :widths: 22 32 46

   * - Field
     - Symbol / Shape
     - Meaning
   * - ``Zavg_R``
     - :math:`N_\tau \times N_\lambda \times N_d`
     - Averaged signal
   * - ``delays``
     - :math:`N_\tau`
     - Delay axis
   * - ``probe``
     - :math:`N_\lambda`
     - Probe (spectral) axis
   * - ``Units``
     - ``dict``
     - Unit labels dictionary (see :ref:`user_guide_settings`)
   * - ``Nscans``
     - scalar
     - Number of scans (``nan`` if unknown)
   * - ``Zss_R``
     - :math:`N_\tau \times N_\lambda \times N_d \times N_s`
     - Single-scan signal tensor
   * - ``Zstdv``
     - :math:`N_\tau \times N_\lambda`
     - Per-point standard deviation

The most-processed signal is exposed through the ``Z`` property, which returns the
background-corrected array once a correction has been applied and falls back to the raw array
otherwise.

Loading and the Loader Registry
-------------------------------

Data formats are decoupled from the dataset through a **loader registry**
(:mod:`pymorgan.oneD.load`). A loader is any callable that maps a file path to the seven fields
listed above. Loaders are registered under one or more data-type names and resolved at load time:

.. code-block:: python

   import pymorgan as pm

   data = pm.load_1D("scan.pdat", data_type="PDAT")
   print(pm.available_loaders())

.. list-table:: Supported One-Dimensional Data Formats in PyMORGAN
   :header-rows: 1
   :widths: 22 26 52

   * - Format Token
     - Instrument / Origin
     - Description
   * - ``PDAT``
     - PyMORGAN standard
     - Comma-separated averaged data (``.pdat``) + optional noise (``.pdatn``)
   * - ``UniGE_fsTA``
     - UniGE fs-TA
     - Step-scan femtosecond TA (``.dat``) with C-engine vectorised parsing
   * - ``UniGE_nsTA``
     - UniGE ns-TA
     - Nanosecond TA (``.dat``) with delay, pixel, and TA signal columns
   * - ``UoS_IRpp``
     - Univ. of Sheffield
     - Transient IR (``.2D`` and ``.DT``), 96 or 192 pixels, single-scan detection
   * - ``MESS_TRIR``
     - MESS TRIR
     - State-modulated TRIR directories with inverse-variance scan averaging
   * - ``MESS_TRUVIS``
     - MESS TRUVIS
     - State-modulated transient UV–Vis directories
   * - ``HARPIA``
     - Light Conversion
     - HARPIA ultrafast transient absorption files (``.dat``)
   * - ``Helios_TA``
     - Ultrafast Systems
     - Helios TA files (``.csv``) with multi-scan discovery and interpolation
   * - ``PyMESS_PP``
     - PyMESS HDF5
     - Broadband pump–probe HDF5 datasets (``.h5``), including slow-modulation states and multi-signal fast modulation
   * - ``PyMESS_Aniso``
     - PyMESS HDF5
     - Slow-modulation polarisation anisotropy HDF5 datasets (``.h5``)
   * - ``PyMESS_t2scan``
     - PyMESS HDF5
     - 2D-IR population waiting-time (:math:`t_2`) scan HDF5 datasets (``.h5``)
   * - ``Exported_TXT``
     - Generic exports
     - Wavelength-vs-delay tabular text files (``.txt``)

A new format is added by decorating a reader with :func:`@pm.register_loader("MyInstrument") <pymorgan.register_loader>`:

.. code-block:: python

   @pm.register_loader("MyInstrument")
   def read_my_instrument(path):
       ...
       return Zavg_R, delays, probe, Units, Nscans, Zss_R, Zstdv

High-Performance File Parsing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Large experimental files (such as UniGE fsTA and Helios TA scans spanning tens of thousands of
lines) are parsed using high-throughput C-accelerated engines (``pandas.read_csv`` with
``engine="c"``) with automated fallbacks for non-standard whitespace or formatting anomalies.
Probe wavelength calibration is automatically resolved from local directory structures (e.g.
``CalibratedProbe.csv`` or ``pix2lam.mat``).

The PDAT Format
~~~~~~~~~~~~~~~

A ``.pdat`` file is comma-separated. The first line carries the time and probe units separated by
an asterisk (e.g. ``ps*cm^{-1}``). The first data row holds the probe axis (with a leading ``0``),
and each subsequent row begins with a delay value followed by the signal at every probe
coordinate. PDAT files store averaged data already in :math:`\text{mOD}`, so no rescaling is
applied on read. Per-point standard deviations are optionally stored in a sibling ``.pdatn`` file
with an identical layout.

PyMESS HDF5 Slow-Modulation & Multi-Signal Schemas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PyMESS HDF5 loaders (``PyMESS_PP``, ``PyMESS_Aniso``, and ``PyMESS_t2scan``) automatically
classify and load both standard single-state pump–probe datasets and multi-state acquisition
layouts:

- **Slow modulation (**\ ``/data/slow_modulation/state_XX_<name>``\ **):** State datasets are
  enumerated while ignoring their ``*_individual_scans`` companion datasets. Per-state single
  scans (``state_XX_<name>_individual_scans``) are loaded into ``Zss_R`` (shape
  :math:`N_\tau \times N_\lambda \times N_{\text{states}} \times N_{\text{scans}}`) and used to
  compute a true empirical per-point standard deviation ``Zstdv``. For polarisation anisotropy
  datasets (``PyMESS_Aniso``), the isotropic Magic Angle
  (:math:`\Delta A_{\mathrm{iso}} = (\Delta A_\parallel + 2G\,\Delta A_\perp)/3`) and Anisotropy
  (:math:`r = (\Delta A_\parallel - G\,\Delta A_\perp)/(\Delta A_\parallel + 2G\,\Delta A_\perp)`)
  channels are appended automatically alongside the parallel and perpendicular states, including
  their derived per-scan tensors.
- **Multi-signal fast modulation (**\ ``/data/signals/signal_XX_*``\ **):** Multi-beam or
  multi-phase chop/modulation schemes stored under ``/data/signals`` (such as pump-only,
  push-only, and pump+push difference channels) are exposed as named detector/modulation states in
  ``Units["detectors"]``, with companion ``*_noise`` datasets mapped directly to ``Zstdv``.
- **Shaper population-time scans (**\ ``PyMESS_t2scan``\ **):** 2D-IR waiting-time (:math:`t_2`)
  scans acquired at a fixed or zero coherence time (:math:`t_1 = 0`) are supported whether raw
  interferograms are stored under ``/data/interferograms_raw`` or pre-demodulated state matrices
  are stored under ``/data/slow_modulation``.

Background Correction
---------------------

The pre-time-zero background is estimated by averaging the signal over a delay window
:math:`[\tau_{\mathrm{min}}, \tau_{\mathrm{max}}]` and subtracting it from every delay:

.. math::

   Z_C(\tau, \lambda, d) = Z(\tau, \lambda, d)
   - \big\langle Z(\tau', \lambda, d) \big\rangle_{\tau' \in [\tau_{\mathrm{min}}, \tau_{\mathrm{max}}]}

where the average is taken with ``nan``-aware statistics (:mod:`pymorgan.oneD.process`). The
operation is applied through:

.. code-block:: python

   data.background_correct(tmin=-20, tmax=-5)                    # apply
   data.background_correct(tmin=-20, tmax=-5, do_correct=False)  # passthrough

With ``do_correct=False`` the raw signal is copied unchanged into the corrected fields, so
downstream code can always read ``Z`` uniformly. The method returns ``self``, allowing calls to
be chained.

Solvent Subtraction
-------------------

For measurements in solution, the pure-solvent response often has to be subtracted. Two modes are
available:

- **Manual subtraction:** A scaled, time-shifted solvent trace is removed from the sample signal:

  .. code-block:: python

     data.subtract_solvent(solvent, dt=0.0, scale=1.0)

  ``dt`` shifts the solvent delay axis before interpolation (accounting for a timing offset
  between the two measurements) and ``scale`` multiplies the solvent signal before subtraction.

- **Automatic solvent fitting:** :meth:`~pymorgan.oneD.dataset.Dataset1D.fit_solvent_auto` models
  the solvent contribution pixel by pixel using the solvent trace convolved with the Instrument
  Response Function (IRF) and removes it:

  .. code-block:: python

     fit = data.fit_solvent_auto(solvent)
     data.subtract_solvent_fit(fit)

  Optional per-pixel fits for the time-zero offset and amplitude scale are controlled via the
  ``solvent_per_pixel_dt`` (default ``False``) and ``solvent_per_pixel_scale`` (default ``True``)
  settings.

Shockwave Subtraction
---------------------

In transient IR / UV–Vis measurements using flowing samples, acoustic shockwaves generated by the
pump beam can appear as a time-dependent artefact that is correlated across *all* probe
wavelengths. The artefact is removed by averaging a selected subset of detector pixels (or a
probe-axis range) into a 1D shockwave reference trace and subtracting it from every pixel:

.. code-block:: python

   from pymorgan.oneD.process import subtract_shockwave

   Z_corrected = subtract_shockwave(Z, pixels=[0, 1, 2])   # pixel indices (0-based)

The GUI exposes this through the **Subtract Shockwave...** button in the 1D tab, which opens
``ShockwaveSubtractionDialog`` with pixel-index or probe-axis-value selection and a live preview.

Chirp (Dispersion) Correction
-----------------------------

Ultrafast time-resolved spectroscopy measurements often suffer from Group Delay Dispersion (GDD),
commonly referred to as "chirp". As the probe pulse passes through dispersive optical elements
(white-light generation crystals, lenses, sample cell windows, and the solvent itself), different
spectral components are temporally stretched so that the effective time-zero :math:`t_0(\lambda)`
varies across the probe wavelength range :math:`\lambda`.

PyMORGAN models the time-zero profile :math:`t_0(\lambda)` using a generalised Cauchy dispersion
equation:

.. math::

   t_0(\lambda) = c_0 + \sum_{n=1}^{N_C-1} c_n \left(\frac{\lambda_{\mathrm{ref}}}{\lambda}\right)^{2n}

where :math:`c_0, \dots, c_{N_C-1}` are the dispersion coefficients,
:math:`\lambda_{\mathrm{ref}}` is a reference wavelength (defaulting to the mean of the fitted
range to make the coefficients dimensionless and well-scaled), and :math:`N_C` is the number of
terms (typically ``6``).

Three chirp determination modes are implemented in :mod:`pymorgan.oneD.chirp`:

- **Automatic:** Fits the coherent artefact and early dynamics at every pixel independently. The
  model consists of a Gaussian Instrument Response Function (IRF), its optional first and second
  derivatives, a constant offset, and an optional erf-broadened exponential decay. PyMORGAN uses
  Variable Projection (VARPRO) to eliminate the linear amplitudes in closed form, optimising only
  the non-linear parameters :math:`(t_0, \mathrm{FWHM}, \tau_{\mathrm{exp}})` per pixel. A second
  pass uses smoothed parameters from the first pass as initial guesses to improve stability.
- **Step function:** Jointly fits a Gaussian IRF (+ derivatives) and a Heaviside step convolved
  with the same Gaussian over a narrow early-time window — robust when early-time exponential
  decay is not needed or is ill-conditioned.
- **Manual:** Fits the Cauchy dispersion equation directly to a set of user-picked
  :math:`(\lambda, t_0)` points clicked interactively on the contour map in the GUI.

Once :math:`t_0(\lambda)` is obtained, the Cauchy dispersion equation is fitted using
bound-constrained, iteratively reweighted linear least-squares (IRLS with Tukey's bisquare
weights) to reject outlier pixel fits.

To apply the correction, the dataset's delay axes are interpolated onto a common chirp-free delay
grid using one of two boundary-handling strategies:

- ``crop``: Slices the delay axis to keep only the delay region where all wavelength channels are
  defined, discarding out-of-bounds regions that become ``NaN`` after shifting.
- ``extrapolate``: Fills out-of-bounds values with the nearest valid boundary endpoint value,
  keeping the original delay range.

.. code-block:: python

   fit = data.fit_chirp_automatic(deriv_mask=(True, True), use_exp=True)
   data.plot_chirp_diagnostics(fit)
   data.apply_chirp_correction(fit, boundary_handling="crop")

Plotting and :math:`\operatorname{arcsinh}` Colour Scaling
----------------------------------------------------------

Five plotters live in :mod:`pymorgan.oneD.plot`; each is exposed both as a
:class:`~pymorgan.oneD.dataset.Dataset1D` method and as a stage-level function that takes the
dataset as its first argument:

.. list-table:: The 1D Plotters (``pymorgan.oneD.plot``)
   :header-rows: 1
   :widths: 34 66

   * - Method / Function
     - View
   * - ``plot_contour``
     - Delay-vs-probe contour map
   * - ``plot_surface``
     - Filled delay-vs-probe surface
   * - ``plot_spectra``
     - Transient spectra at selected delays
   * - ``plot_kinetics``
     - Kinetic traces at selected probe positions (also returns extracted ``(ax, t, Y)``)
   * - ``plot_species_spectra``
     - DAS / EAS / SAS from a global or target analysis fit

.. code-block:: python

   data.plot_contour(Zscale=20)                          # method form
   data.plot_spectra([0.5, 1, 5, 20, 100], doSmooth=1)
   ax, t, Y = data.plot_kinetics([2132, 2218], plotStyle="-")

   from pymorgan.oneD import plot                        # stage-function form
   plot.plot_contour(data, label_style="/", cmap_ID="Rd/Wh/Bu v2")

For datasets with high dynamic range where weak features would otherwise be washed out by
dominant bleach or ESA peaks, ``plot_contour`` provides non-linear :math:`\operatorname{arcsinh}`
colour scaling via ``Asinh=True`` (also toggled in the GUI plot controls). PyMORGAN applies an
inverse hyperbolic sine colour normalisation (``matplotlib.colors.AsinhNorm``) paired with
non-linear :math:`\sinh`-spaced contour levels:

.. math::

   z_k = s \cdot \sinh(u_k), \quad u_k \in \left[-\operatorname{arcsinh}\!\left(\frac{|Z_{\min}|}{s}\right),\; \operatorname{arcsinh}\!\left(\frac{Z_{\max}}{s}\right)\right]

where :math:`s = \frac{p}{100} \cdot \max(|Z_{\min}|, |Z_{\max}|)` is the linear crossover
threshold set by ``asinh_pct`` (:math:`p`, default ``5.0%``). Signals within :math:`\pm s` behave
linearly, while larger amplitudes are compressed logarithmically. Because the underlying data
array is not mutated, coordinates and hover values retain their physical units (:math:`\mathrm{mOD}`),
and colorbar ticks remain linear and centred at zero.

Composing Multi-Panel Figures and Displaying
--------------------------------------------

By default every plotter creates its own figure and axis. Passing an existing axis through
``ax=`` draws into it instead, allowing multi-panel layouts or dataset overlays on a single axis:

.. list-table:: Per-Call Toggles for Composing Multi-Panel Figures
   :header-rows: 1
   :widths: 30 18 52

   * - Keyword
     - Default
     - Plotters
   * - ``show_xlabel``
     - ``True``
     - All
   * - ``show_ylabel``
     - ``True``
     - All
   * - ``show_colorbar``
     - ``True``
     - ``plot_contour``, ``plot_surface``
   * - ``show_colorbar_label``
     - ``True``
     - ``plot_contour``, ``plot_surface``

.. code-block:: python

   import matplotlib.pyplot as plt
   import pymorgan as pm

   fig, (axL, axR) = plt.subplots(1, 2)

   # Two maps side by side; suppress duplicated colorbar / y-axis label
   dataA.plot_contour(ax=axL, show_colorbar=False)
   dataB.plot_contour(ax=axR, show_ylabel=False)

   pm.show_plots()   # wrapper around plt.show(); pm.show is an alias
   pm.close_plots()  # close all open figures

Kinetic Analysis Seam & Noise Propagation
-----------------------------------------

Kinetic analysis (multi-exponential decays, global and target analysis with rate-matrix models,
and Lifetime Density Analysis) lives in the companion project
`PyRATE-TA <https://rjfernandezteran.github.io/PyRATE-TA/>`_. PyMORGAN provides the two ends of
that seam plus the per-point noise:

1. **Trace extraction:** ``ax, t, Y = data.plot_kinetics([1950, 2000], plotStyle="-")`` returns
   the extracted traces ``Y`` of shape ``[Ndelays x Ncuts]``.
2. **Species spectra rendering:** ``data.plot_species_spectra(Sfit, Taus, TauErr, isFixTau, modelType)``
   accepts plain arrays directly via ``*fit.as_species_args()``.
3. **Noise propagation:** :meth:`Dataset1D.noise_array() <pymorgan.oneD.dataset.Dataset1D.noise_array>`
   returns ``[Ndelays x Npixels x Ndetectors]`` from ``Zstdv`` (sibling ``.pdatn`` file) or the
   spread across single scans (``Zss_R``), enabling true reduced-:math:`\chi^2` weighting in
   PyRATE-TA.
4. **One-click GUI handoff:** In ``pymorgan-gui``, **Action → Launch PyRATE-TA** sends the
   currently active (background-, solvent-, and chirp-corrected) dataset directly into PyRATE-TA.

Graphical Interface — 1D Analysis & Processing Dialogs
------------------------------------------------------

Dataset Stitching: Wavelength and Timescale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accessed via **Pre-Processing → Stitch Datasets**:

- **Wavelength Stitching (**\ ``Stitch Wavelength Windows...``\ **):** Automatically detects the
  shared probe interval :math:`[\lambda_{\mathrm{min}}^{\mathrm{ovl}}, \lambda_{\mathrm{max}}^{\mathrm{ovl}}]`
  between Dataset 1 and Dataset 2, evaluates the optimal least-squares amplitude scaling factor

  .. math::

     \alpha = \frac{\sum_{\tau, \lambda} Z_1(\tau, \lambda)\, Z_2(\tau, \lambda)}{\sum_{\tau, \lambda} Z_2(\tau, \lambda)^2}

  and blends the transition window using a raised-cosine taper weight:

  .. math::

     w(\lambda) = \frac{1}{2} \left[ 1 + \cos\!\left( \pi \frac{\lambda - \lambda_{\mathrm{min}}^{\mathrm{ovl}}}{\lambda_{\mathrm{max}}^{\mathrm{ovl}} - \lambda_{\mathrm{min}}^{\mathrm{ovl}}} \right) \right], \qquad
     Z_{\mathrm{stitched}}(\tau, \lambda) = w(\lambda)\, Z_1(\tau, \lambda) + \big(1 - w(\lambda)\big)\, \alpha Z_2(\tau, \lambda)

- **Timescale Stitching (**\ ``Stitch Timescales...``\ **):** Calibrates delay offsets
  :math:`\Delta \tau_0` and cross-scales overlapping femtosecond/picosecond optical delay scans
  with nanosecond/microsecond electronic delay measurements onto a unified temporal grid.

Singular Value Decomposition (SVD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accessed via **Analysis → Singular Value Decomposition (SVD)...** (:mod:`pymorgan.oneD.svd`),
this tool performs model-free factorisation of the transient matrix
:math:`Z \in \mathbb{R}^{N_\tau \times N_\lambda}`:

.. math::

   Z = U \Sigma V^{\mathsf{T}} = \sum_{k=1}^{\min(N_\tau, N_\lambda)} \sigma_k\, \mathbf{u}_k\, \mathbf{v}_k^{\mathsf{T}}

The dialog displays a diagnostic scree plot (singular values :math:`\sigma_k` and cumulative
variance explained) and allows interactive rank-:math:`K` noise filtering:

.. math::

   Z_{\mathrm{filtered}} = \sum_{k=1}^{K} \sigma_k\, \mathbf{u}_k\, \mathbf{v}_k^{\mathsf{T}}

with real-time contour previews of the raw data, rank-:math:`K` reconstruction, and residual map
:math:`Z - Z_{\mathrm{filtered}}`.

Polarisation Anisotropy and Magic Angle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accessed via the top-level **Anisotropy** menu, supporting PyMESS slow-modulation ``.h5``
datasets, multi-detector channels, paired parallel/perpendicular files, and legacy MESS folders.
With G-factor correction :math:`G = \Delta A_\parallel(\tau_{\mathrm{iso}}, \lambda) / \Delta A_\perp(\tau_{\mathrm{iso}}, \lambda)`,
PyMORGAN computes the time-resolved anisotropy :math:`r(\tau, \lambda)`, isotropic (magic-angle
:math:`54.7^\circ`) absorbance change :math:`\Delta A_{\mathrm{iso}}(\tau, \lambda)`, and
polarisation difference :math:`\Delta I(\tau, \lambda)`:

.. math::

   r(\tau, \lambda) = \frac{\Delta A_\parallel(\tau, \lambda) - G\, \Delta A_\perp(\tau, \lambda)}{\Delta A_\parallel(\tau, \lambda) + 2G\, \Delta A_\perp(\tau, \lambda)}

.. math::

   \Delta A_{\mathrm{iso}}(\tau, \lambda) = \frac{\Delta A_\parallel(\tau, \lambda) + 2G\, \Delta A_\perp(\tau, \lambda)}{3}, \qquad
   \Delta I(\tau, \lambda) = \Delta A_\parallel(\tau, \lambda) - G\, \Delta A_\perp(\tau, \lambda)

Time Derivative of Spectra (:math:`\mathrm{d}(\Delta A)/\mathrm{d}t`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accessed via **Pre-Processing → Time Derivative of Spectra...** (``time_derivative_dialog.py`` /
:func:`~pymorgan.oneD.process.time_derivative`), this dialog computes the numerical derivative
:math:`\partial \Delta A(\tau, \lambda) / \partial \tau` along the delay axis with:

- **Early-delay cutoff:** Trims delays :math:`\tau < \tau_{\min}` prior to differentiation so the
  coherent artefact around :math:`t = 0` does not dominate the derivative scale.
- **Time-domain interpolation:** Resamples onto a denser temporal grid (:math:`N_{\text{interp}}`
  points) prior to differentiation.
- **Pre-smoothing:** Savitzky–Golay polynomial filtering or moving-average smoothing with live
  kinetic trace preview.
