.. _user_guide_oneD:

One-Dimensional Data (Pump–Probe & TA)
======================================

Pipeline Overview
-----------------

The one-dimensional branch is organised as a **load → process → plot → analyse** pipeline.
Each stage is a module under :mod:`pymorgan.oneD`, and the :class:`~pymorgan.oneD.dataset.Dataset1D`
object ties them together: it holds the raw and corrected arrays and exposes one set of methods
per stage.

.. list-table:: 1D Pipeline Stages
   :header-rows: 1
   :widths: 18 34 48

   * - Stage
     - Module
     - Entry Point(s)
   * - **load**
     - :mod:`pymorgan.oneD.load`
     - :func:`pymorgan.load_1D`, :meth:`Dataset1D.from_file <pymorgan.oneD.dataset.Dataset1D.from_file>`
   * - **process**
     - :mod:`pymorgan.oneD.process`, :mod:`pymorgan.oneD.chirp`
     - :meth:`Dataset1D.background_correct <pymorgan.oneD.dataset.Dataset1D.background_correct>`,
       :func:`~pymorgan.oneD.chirp.apply_chirp_correction`
   * - **plot**
     - :mod:`pymorgan.oneD.plot`
     - ``Dataset1D.plot_contour``, ``plot_surface``, ``plot_spectra``, ``plot_kinetics``, ``plot_species_spectra``
   * - **analyse**
     - *PyRATE-TA*
     - Companion package (`PyRATE-TA <https://github.com/RJFernandezTeran/PyRATE-TA>`_)

Data Model
----------

A one-dimensional measurement is represented by a :class:`~pymorgan.oneD.dataset.Dataset1D`
object. The signal is stored as a three-dimensional array to generalise over multiple detectors:

.. math::

   Z \in \mathbb{R}^{N_\tau \times N_\lambda \times N_d}

indexed by delay :math:`\tau`, probe coordinate :math:`\lambda` (wavelength, wavenumber, energy,
or frequency), and detector :math:`d`.

.. list-table:: Fields Stored on ``Dataset1D``
   :header-rows: 1
   :widths: 22 32 46

   * - Field
     - Shape
     - Meaning
   * - ``Zavg_R``
     - :math:`N_\tau \times N_\lambda \times N_d`
     - Averaged raw signal
   * - ``delays``
     - :math:`N_\tau`
     - Pump–probe delay axis
   * - ``probe``
     - :math:`N_\lambda`
     - Probe (spectral) axis
   * - ``Units``
     - ``dict``
     - Unit labels dictionary
   * - ``Nscans``
     - scalar
     - Number of scans (``nan`` if unknown)
   * - ``Zss_R``
     - :math:`N_\tau \times N_\lambda \times N_d \times N_s`
     - Single-scan tensor
   * - ``Zstdv``
     - :math:`N_\tau \times N_\lambda`
     - Per-point standard deviation

Loading and the Loader Registry
-------------------------------

Data formats are decoupled from :class:`~pymorgan.oneD.dataset.Dataset1D` through a loader
registry (:mod:`pymorgan.oneD.load`).

.. code-block:: python

   import pymorgan as pm

   data = pm.load_1D("scan.pdat", data_type="PDAT")
   print(pm.available_loaders())

.. list-table:: Supported 1D Data Formats
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
     - Broadband pump–probe HDF5 datasets (``.h5``)
   * - ``PyMESS_Aniso``
     - PyMESS HDF5
     - Slow-modulation polarisation anisotropy HDF5 datasets (``.h5``)
   * - ``PyMESS_t2scan``
     - PyMESS HDF5
     - 2D-IR population waiting-time (:math:`t_2`) scan HDF5 datasets (``.h5``)
   * - ``Exported_TXT``
     - Generic exports
     - Wavelength-vs-delay tabular text files (``.txt``)

Background, Solvent, and Shockwave Subtraction
----------------------------------------------

Background Correction
~~~~~~~~~~~~~~~~~~~~~

The pre-time-zero background is estimated by averaging the signal over a delay window
:math:`[\tau_{\min}, \tau_{\max}]` and subtracting it from every delay:

.. math::

   Z_C(\tau, \lambda, d) = Z(\tau, \lambda, d)
   - \big\langle Z(\tau', \lambda, d) \big\rangle_{\tau' \in [\tau_{\min}, \tau_{\max}]}

.. code-block:: python

   data.background_correct(tmin=-20, tmax=-5)

Solvent Subtraction
~~~~~~~~~~~~~~~~~~~

Both manual and IRF-convolved automatic per-pixel solvent subtraction are supported:

.. code-block:: python

   # Manual scaled/shifted subtraction
   data.subtract_solvent(solvent, dt=0.0, scale=1.0)

   # Automatic per-pixel IRF-convolved fit
   fit = data.fit_solvent_auto(solvent)
   data.subtract_solvent_fit(fit)

Shockwave Subtraction
~~~~~~~~~~~~~~~~~~~~~

In flowing-sample measurements, acoustic shockwaves correlated across all probe wavelengths can
be removed by averaging selected baseline pixels and subtracting the trace across the detector:

.. code-block:: python

   from pymorgan.oneD.process import subtract_shockwave

   Z_corrected = subtract_shockwave(Z, pixels=[0, 1, 2])

Chirp (Group Delay Dispersion) Correction
-----------------------------------------

PyMORGAN models the wavelength-dependent time-zero profile :math:`t_0(\lambda)` using a
generalised Cauchy dispersion equation:

.. math::

   t_0(\lambda) = c_0 + \sum_{n=1}^{N_C-1} c_n \left(\frac{\lambda_{\mathrm{ref}}}{\lambda}\right)^{2n}

Three chirp determination modes are implemented in :mod:`pymorgan.oneD.chirp`:

- **Automatic:** Fits the coherent artefact and early dynamics at every pixel independently using
  Variable Projection (VARPRO) with a Gaussian IRF, optional 1st/2nd derivatives, and an
  erf-broadened exponential decay, followed by IRLS Cauchy regression with Tukey's bisquare
  weights.
- **Step function:** Jointly fits a Gaussian IRF (+ derivatives) and a Heaviside step convolved
  with the Gaussian over a narrow early-time window.
- **Manual:** Fits the Cauchy dispersion curve directly to user-selected :math:`(\lambda, t_0)`
  anchor points clicked on the contour map.

.. code-block:: python

   fit = data.fit_chirp_automatic(deriv_mask=(True, True), use_exp=True)
   data.plot_chirp_diagnostics(fit)
   data.apply_chirp_correction(fit, boundary_handling="crop")

Plotting and :math:`\operatorname{arcsinh}` Scaling
---------------------------------------------------

Five plotters live in :mod:`pymorgan.oneD.plot`: ``plot_contour``, ``plot_surface``,
``plot_spectra``, ``plot_kinetics``, and ``plot_species_spectra``.

For datasets with high dynamic range, ``plot_contour`` provides non-linear
:math:`\operatorname{arcsinh}` colour scaling via ``Asinh=True`` paired with
:math:`\sinh`-spaced contour levels:

.. math::

   z_k = s \cdot \sinh(u_k), \quad
   u_k \in \left[-\operatorname{arcsinh}\!\left(\frac{|Z_{\min}|}{s}\right),\;
   \operatorname{arcsinh}\!\left(\frac{Z_{\max}}{s}\right)\right]

where :math:`s = \frac{p}{100} \cdot \max(|Z_{\min}|, |Z_{\max}|)` is the linear crossover
threshold set by ``asinh_pct`` (default ``5.0%``).

Dataset Stitching, SVD, and Polarisation Anisotropy
---------------------------------------------------

Wavelength and Timescale Stitching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accessible programmatically via :mod:`pymorgan.oneD.stitch` or in the GUI under
**Spectra → Stitch Datasets**:

- **Wavelength Stitching:** Discovers the overlap region :math:`[\lambda_{\min}^{\mathrm{ovl}}, \lambda_{\max}^{\mathrm{ovl}}]`,
  evaluates the least-squares cross-scaling coefficient :math:`\alpha`, and blends windows using a
  smooth raised-cosine taper weight:

  .. math::

     w(\lambda) = \frac{1}{2} \left[ 1 + \cos\!\left( \pi \frac{\lambda - \lambda_{\min}^{\mathrm{ovl}}}{\lambda_{\max}^{\mathrm{ovl}} - \lambda_{\min}^{\mathrm{ovl}}} \right) \right]

- **Timescale Stitching:** Aligns time-zero offsets :math:`\Delta \tau_0` and cross-scales
  overlapping femtosecond/picosecond optical delay data with nanosecond/microsecond electronic
  delay scans.

Singular Value Decomposition (SVD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implemented in :mod:`pymorgan.oneD.svd` and **Analysis → Singular Value Decomposition (SVD)...**:

.. math::

   Z = U \Sigma V^{\mathsf{T}} = \sum_{k=1}^{\min(N_\tau, N_\lambda)} \sigma_k\, \mathbf{u}_k\, \mathbf{v}_k^{\mathsf{T}}

Provides diagnostic scree plots, cumulative variance curves, and low-rank noise filtering
(:math:`Z_{\mathrm{filtered}} = \sum_{k=1}^{K} \sigma_k \mathbf{u}_k \mathbf{v}_k^{\mathsf{T}}`).

Polarisation Anisotropy & Magic Angle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Computes the G-factor corrected anisotropy :math:`r(\tau, \lambda)`, isotropic (magic-angle)
absorbance change :math:`\Delta A_{\mathrm{iso}}(\tau, \lambda)`, and polarisation difference
:math:`\Delta I(\tau, \lambda)`:

.. math::

   r(\tau, \lambda) = \frac{\Delta A_\parallel(\tau, \lambda) - G\, \Delta A_\perp(\tau, \lambda)}{\Delta A_\parallel(\tau, \lambda) + 2G\, \Delta A_\perp(\tau, \lambda)}, \qquad
   \Delta A_{\mathrm{iso}}(\tau, \lambda) = \frac{\Delta A_\parallel(\tau, \lambda) + 2G\, \Delta A_\perp(\tau, \lambda)}{3}
