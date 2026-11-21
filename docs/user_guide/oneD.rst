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
     - :mod:`pymorgan.oneD.process`, :mod:`pymorgan.oneD.scatter`, :mod:`pymorgan.oneD.chirp`, :mod:`pymorgan.oneD.stitch`, :mod:`pymorgan.oneD.svd`
     - :meth:`Dataset1D.background_correct <pymorgan.oneD.dataset.Dataset1D.background_correct>`,
       :meth:`Dataset1D.remove_scatter <pymorgan.oneD.dataset.Dataset1D.remove_scatter>`,
       :func:`~pymorgan.oneD.scatter.compute_scatter_removal`,
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
   * - ``HDF5``
     - PyMORGAN FAIR HDF5
     - Gzip-compressed FAIR/NeXus HDF5 archive (``.h5`` / ``.hdf5``) storing arrays, chirp/mask/scatter state, and JSON history
   * - ``UniGE_fsTA``
     - UniGE fs-TA
     - Step-scan femtosecond TA (``.dat``) with C-engine vectorised parsing
   * - ``UniGE_nsTA``
     - UniGE ns-TA
     - Nanosecond TA (``.dat``) with delay, pixel, and TA signal columns
   * - ``UoS_IRpp``
     - Univ. of Sheffield
     - Transient IR (``.2D`` and ``.DT``), 96 or 192 pixels, single-scan detection
   * - ``RAL_IRpp``
     - STFC / Central Laser Facility
     - RAL LIFEtime 1D IR pump–probe CSV datasets (alias ``RAL_TRIR`` / ``RAL_LIFEtime``)
   * - ``MESS_TRIR``
     - MESS TRIR
     - State-modulated TRIR directories with inverse-variance scan averaging
   * - ``MESS_TRUVIS``
     - MESS TRUVIS
     - State-modulated transient UV–Vis directories
   * - ``HARPIA_TA``
     - Light Conversion
     - HARPIA ultrafast transient absorption files (``.dat``)
   * - ``Helios_TA``
     - Ultrafast Systems
     - Helios TA files (``.csv``) with multi-scan discovery and interpolation
   * - ``Helios_UFS``
     - Ultrafast Systems
     - Surface Xplorer binary datasets (``.ufs``) with metadata and ``_scan*.ufs`` single-scan loading
   * - ``QuickControl_PP``
     - PhaseTech Spectroscopy
     - QuickControl 1D pump–probe ASCII files (``.info`` + ``.scan`` + ``.spec``) with automatic Magic Angle channel ordering
   * - ``PyMESS_PP``
     - PyMESS HDF5
     - Broadband pump–probe HDF5 datasets (``.h5``), including slow-modulation states and multi-signal fast modulation
   * - ``PyMESS_Aniso``
     - PyMESS HDF5
     - Slow-modulation polarisation anisotropy HDF5 datasets (``.h5``)
   * - ``PyMESS_t2scan``
     - PyMESS HDF5
     - 2D-IR population waiting-time (:math:`t_2`) scan HDF5 datasets (``.h5``)
   * - ``Glotaran_ASCII``
     - Glotaran / TIMP
     - Time-explicit or wavelength-explicit ASCII exchange files (``.ascii``)
   * - ``Glotaran_NC``
     - Glotaran / pyglotaran
     - CF-1.8 compliant NetCDF4/HDF5 datasets (``.nc``) via ``h5py`` dimension scales
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

Surface Xplorer (``.ufs``), Glotaran & FAIR HDF5 Interoperability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Surface Xplorer (**\ ``Helios_UFS``\ **):** Directly parses binary ``.ufs`` files (IEEE-754
  big-endian v1/v2 streams), preserves metadata blocks (sample, solvent, excitation wavelength),
  converts ``DA`` values to :math:`\text{mOD}`, sanitises non-finite values, and discovers sibling
  ``_scan*.ufs`` single-scan files automatically.
- **Glotaran / pyglotaran (**\ ``Glotaran_ASCII``, ``Glotaran_NC``\ **):** Reads and writes both
  legacy Glotaran/TIMP time-/wavelength-explicit ``.ascii`` files and CF-1.8 NetCDF4 ``.nc`` files
  via ``Dataset1D.to_glotaran_ascii()`` and ``Dataset1D.to_glotaran_nc()`` (built directly on
  ``h5py`` dimension scales without external ``netCDF4`` or ``xarray`` dependencies).
- **FAIR HDF5 v2.0 (**\ ``HDF5``\ **):** ``Dataset1D.to_hdf5()`` and :func:`pymorgan.oneD.load.load_1D_hdf5`
  store raw/corrected arrays, chirp fits, masks, scatter parameters, units, and full JSON
  processing provenance (``history_json``), which is restored automatically on reload via
  :func:`pymorgan.load_auto` or ``pm.load_1D(..., data_type="HDF5")``.

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
be chained. When time-varying pump scatter or drifting broadband backgrounds defeat static
pre-zero subtraction, use :ref:`user_guide_oneD_scatter`.

.. _user_guide_oneD_scatter:

Advanced Scatter & Dynamic Background Removal
---------------------------------------------

Standard static pre-zero subtraction removes one fixed spectrum averaged over
:math:`[\tau_{\mathrm{min}}, \tau_{\mathrm{max}}]`. In real ultrafast transient absorption
measurements, pump scatter and broadband pre-zero backgrounds often evolve with delay and across
scans for four physical reasons:

1. **Delay-stage scatter dimming:** Translation of the mechanical delay stage subtly steers or
   clips scatter on the sample cell, causing a fixed static subtraction to oversubtract at positive
   delays and produce phantom positive structure across the pump core.
2. **Delay-dependent spectral walk:** Small centroid shifts or width changes of the scatter peak
   turn static residuals into bipolar derivative-like dispersive artefacts.
3. **Delay-to-delay speckle jitter & scan-to-scan drift:** Shot-to-shot fluctuations and scan-level
   scatter drift leave noisy vertical stripes in the averaged map and sharply inflate ``Zstdv``
   around the scatter wavelength, distorting weighted global analysis in **PyRATE-TA**.
4. **Broadband fluorescence & far-wing dimming:** Slow delay evolution of broadband luminescence or
   scatter tails introduces baseline tilt across all probe channels when subtracted statically.

PyMORGAN addresses all four failure modes in :mod:`pymorgan.oneD.scatter` while keeping genuine
transient absorption spectra, excited-state absorption, and coherent :math:`t_0` artefacts intact.

Two-Tier Automatic Band Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~pymorgan.oneD.scatter.detect_scatter_bands` (also wrapped by
:meth:`Dataset1D.detect_scatter_bands <pymorgan.oneD.dataset.Dataset1D.detect_scatter_bands>`)
locates scatter bands from the pre-zero average :math:`B_{\mathrm{pre}}(\lambda)` in two passes:

- **Pass 1 (primary bands):** Applies median-filter detrending (odd window size
  :math:`k = \max(21, \lfloor 0.08\,N_\lambda \rceil_{\mathrm{odd}})`) and robust MAD noise estimation
  (:math:`\sigma = 1.4826\,\mathrm{MAD}`) to threshold sharp pre-zero dips above
  :math:`\max(6\sigma, \text{floor})` (floor :math:`0.5\,\mathrm{mOD}` or
  :math:`5\times 10^{-4}\,\mathrm{OD}`; negative dips by default, or signed when
  ``allow_positive=True``). Contiguous runs of :math:`\ge 3` pixels (bridging 1–2 pixel gaps) are
  expanded outward until below :math:`\max(2\sigma, 1\%\,\text{peak})`, merged if overlapping, and
  sorted by descending amplitude so Band 0 is the strongest primary scatter band.
- **Pass 2 (2nd-order grating harmonics):** When the probe axis is calibrated in wavelength
  (:math:`\mathrm{nm}`), a :math:`\pm 12\,\mathrm{nm}` window around
  :math:`2\lambda_{\mathrm{pump}}` (resolved from ``pymess_metadata["pump_wl_nm"]`` or Band 0's
  peak) is inspected for second-order diffraction dips. Candidates with
  :math:`\mathrm{SNR} \ge 5\sigma` are added enabled with ``rank=1, couple_to=0``, while
  :math:`3\sigma\text{--}5\sigma` candidates are added disabled as optional suggestions.

Mathematical Formulation & Basis Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each :class:`~pymorgan.oneD.scatter.ScatterBand` specifies a core
:math:`[\lambda_{\min,m}, \lambda_{\max,m}]`, side-wing margin :math:`w_{\mathrm{wing},m}`
(default :math:`\max(3\Delta\lambda, 0.5\,\Delta\lambda_{\mathrm{band}})`), and raised-cosine
taper width :math:`w_{\mathrm{taper},m}` (default
:math:`\min(w_{\mathrm{wing},m}, \max(2\Delta\lambda, 0.25\,\Delta\lambda_{\mathrm{band}}))`):

.. math::

   T_m(\lambda) =
   \begin{cases}
     1, & \lambda \in [\lambda_{\min,m}, \lambda_{\max,m}], \\
     \tfrac{1}{2}\!\left[1 - \cos\!\left(\pi\frac{\lambda - (\lambda_{\min,m} - w_{\mathrm{taper},m})}{w_{\mathrm{taper},m}}\right)\right], & \lambda \in [\lambda_{\min,m} - w_{\mathrm{taper},m}, \lambda_{\min,m}), \\
     \tfrac{1}{2}\!\left[1 - \cos\!\left(\pi\frac{(\lambda_{\max,m} + w_{\mathrm{taper},m}) - \lambda}{w_{\mathrm{taper},m}}\right)\right], & \lambda \in (\lambda_{\max,m}, \lambda_{\max,m} + w_{\mathrm{taper},m}], \\
     0, & \text{otherwise.}
   \end{cases}

Touching or overlapping uncoupled fit windows
:math:`[\lambda_{\min,m} - w_{\mathrm{wing},m}, \lambda_{\max,m} + w_{\mathrm{wing},m}]` are merged
into joint groups. Subtracting a linear wing-estimated baseline
:math:`b_{\mathrm{wing}}(\lambda)` yields the localised pre-zero template
:math:`P_0(\lambda) = (B_{\mathrm{pre}}(\lambda) - b_{\mathrm{wing}}(\lambda))\,T(\lambda)` and
complementary broadband component
:math:`B_{\mathrm{broad}}(\lambda) = B_{\mathrm{pre}}(\lambda) - \sum_m P_{0,m}(\lambda)`.
When ``scale_broadband=True`` (default), a Savitzky–Golay-smoothed slow envelope
:math:`a_{\mathrm{smooth}}(\tau)` of the dominant primary band (re-normalised to :math:`1.0` at
pre-zero) scales :math:`B_{\mathrm{broad}}` across delays,
:math:`Y_{\mathrm{nb}}(\tau, \lambda) = Y(\tau, \lambda) - a_{\mathrm{smooth}}(\tau)\,B_{\mathrm{broad}}(\lambda)`,
eliminating fluorescence and far-wing dimming without injecting high-frequency speckle jitter.

Within each group's extended fit window, :func:`~pymorgan.oneD.scatter.compute_scatter_removal`
solves a joint least-squares system at every delay :math:`\tau` separating a tapered scatter basis
:math:`\{\phi_r(\lambda)\}_{r=1}^R` from a low-degree local TA polynomial
(:math:`p = \text{poly\_deg}`, default ``2``):

.. math::

   Y_{\mathrm{nb}}(\tau, \lambda) \approx \sum_{r=1}^{R} c_r(\tau)\,\phi_r(\lambda) + \sum_{j=0}^{p} \beta_j(\tau)\,\tilde{\lambda}^j

Only the scatter component :math:`\sum_r c_r(\tau)\,\phi_r(\lambda)` is subtracted; the smooth
physical TA polynomial :math:`\sum_j \beta_j(\tau)\,\tilde{\lambda}^j` is preserved intact.

.. list-table:: Comparison of Scatter Basis Methods (``method`` parameter)
   :header-rows: 1
   :widths: 20 35 45

   * - Method
     - Mathematical Model
     - Strengths & Typical Use
   * - ``"local_svd"`` (default)
     - Wing-detrended, tapered SVD learned on delays outside ``t0_exclude`` (default ``(-0.5, 1.5)``), re-tapered and solved jointly per delay.
     - Tracks amplitude dimming, centroid walk, width morphing, and higher-order slit structure while shielding :math:`t_0` coherent artefacts.
   * - ``"taylor"``
     - Strictly pre-zero cubic-spline template and tapered spectral derivatives :math:`[P_0,\, P_0'T,\, P_0''T,\, \dots]`.
     - Deterministic purely pre-zero basis absorbing small spectral walks and broadening without inspecting positive delays.
   * - ``"warped"``
     - Per-delay nonlinear VARPRO solving shift :math:`\delta\lambda(\tau)` and optional stretch :math:`\gamma(\tau)` (when ``rank >= 2``) via ``scipy.optimize.least_squares``.
     - Handles larger centroid walks relative to peak width using a strictly pre-zero template while solving amplitudes linearly.

Harmonic Coupling, Per-Scan Processing & Optical Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Harmonic coupling (``couple_to``):** Coupled harmonic bands scale the target band's smooth
  envelope with a single global scalar :math:`\alpha \in [0, 2.5]` fitted outside ``t0_exclude``,
  :math:`\alpha\,a_{\mathrm{smooth}}^{\mathrm{target}}(\tau)\,P_{0,c}(\lambda)`, avoiding overfitting
  weak harmonic peaks.
- **Per-scan cleaning & noise propagation (``per_scan=True``, ``update_noise=True``):** When
  single-scan tensors (``Zss_R``) are present, each scan is cleaned before recomputing ``Zavg_C``
  and ``Zstdv`` (``ddof=1``), removing inter-scan scatter drift from uncertainty arrays used by
  PyRATE-TA. Calling :meth:`~pymorgan.oneD.dataset.Dataset1D.clear_scatter_removal` restores the
  original ``Zstdv``.
- **Exact transmittance space & saturation masking:** ``transmittance_space=True`` transforms
  :math:`\Delta A` into linear transmittance modulation :math:`Y = 10^{-\Delta A / s} - 1`
  (:math:`s = 1000` for mOD, :math:`1` for OD) prior to separation; ``mask_saturated=True`` masks
  raw core pixels with :math:`|Z_{\mathrm{raw}}| > \text{saturation\_threshold}` (default ``500.0``)
  to ``NaN``.

.. important::

   Apply advanced scatter removal **before** chirp correction (:ref:`Chirp (Dispersion) Correction <user_guide_oneD_chirp>`),
   because pump scatter is physically tied to the delay-stage position and is unchirped, whereas
   chirp interpolation mixes delays across neighbouring pixels.

.. code-block:: python

   from pymorgan.oneD.scatter import ScatterBand, compute_scatter_removal

   # Inspect auto-detected bands (in native probe units, e.g. nm)
   bands = data.detect_scatter_bands(tmin=-20, tmax=-2)

   # Apply dynamic multi-band removal in place (per-scan + Zstdv update)
   data.remove_scatter(method="local_svd", rank=3, per_scan=True, update_noise=True)

   # Or evaluate explicitly to inspect quality metrics and diagnostics
   res = compute_scatter_removal(
       data,
       bands=[
           ScatterBand(348.2, 362.4, label="Primary"),
           ScatterBand(704.5, 715.3, rank=1, couple_to=0, label="2nd order"),
       ],
       method="local_svd",
   )
   print(res.metrics)
   data.apply_scatter_removal(res)
   # data.clear_scatter_removal()  # restores original Zstdv

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

.. _user_guide_oneD_chirp:

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

Advanced Scatter Removal Dialog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accessed via **Pre-processing → Advanced Scatter Removal...** or the compact **Adv...** button
beside the pre-zero background window controls, ``ScatterRemovalDialog``
(``src/pymorgan/gui/scatter_dialog.py``) provides a live split-pane interface around
:func:`~pymorgan.oneD.scatter.compute_scatter_removal`:

- **Left controls sidebar:** Pre-zero delay bounds, per-scan processing and ``Zstdv`` noise-update
  toggles, multi-band configuration table (enable toggle, editable label and bounds in the active
  spectral display unit, per-row Mode selector ``Auto`` / ``Rank 1``–``Rank 4`` / ``Linked → #1``,
  **Auto-Detect**, interactive **Pick on Plot**, **+** / **−**, shared **Wing** / **Taper** width
  overrides, positive-artefact toggle), algorithm selector (Adaptive Local SVD, Pre-Zero Template +
  Deriv., Warped Pre-Zero Template), ``Rank``, ``TA poly``, ``Exclude t0``, slow-envelope broadband
  scaling, exact transmittance space, and saturation masking.
- **Right 4-tab preview dashboard:** Shares **Zoom** preset (Full Range or focused band), **Asinh**,
  and **Zscale %** controls across four interactive tabs:

  1. **2D Maps:** Side-by-side maps comparing *Static Pre-Zero*, *Advanced Clean*, and
     *Removed Scatter (Diff)* with dashed band bounds and scroll-wheel zoom.
  2. **Spectral Slices:** Stacked before-vs-after transient spectra at user-chosen delays.
  3. **Kinetics:** Before-vs-after kinetic traces across side-wing and core probe channels
     (auto-populated via **From Focused Band**).
  4. **Scatter Diagnostics:** Pre-zero spectrum equipped with an interactive Matplotlib
     ``SpanSelector`` (activated by **Pick on Plot**), normalised basis shapes compared with
     template :math:`P_0`, and delay-dependent amplitude envelopes :math:`a(t)` /
     :math:`a_{\mathrm{smooth}}(t)` and centroid shift :math:`\delta\lambda(t)`.

- **Live metrics & state persistence:** A footer readout summarises before/after pre-zero RMS,
  positive-delay spectral second-difference roughness relative to the wing reference, and
  inter-scan STD. Clicking **Apply to Workspace** commits the cleaned dataset and highlights the
  **Adv...** button; re-clicking **Subtract bkg.** preserves the active scatter result while checked,
  **Reset to Static Bkg.** restores static subtraction and original ``Zstdv``, and geometric or
  scan edits (``crop``, ``mask_probe_regions``, ``shift_t0``, ``recalc_average``) clear it
  automatically.

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
