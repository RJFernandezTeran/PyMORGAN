.. _user_guide_calibration:

Spectrometer & Pulse Shaper Wavelength Calibration
==================================================

Wavelength and wavenumber axis calibration maps detector pixel indices
:math:`p \in \{1, \dots, N_{\text{pix}}\}` to physical spectral values (wavelength :math:`\lambda`
in nm or wavenumber :math:`\tilde{\nu}` in :math:`\text{cm}^{-1}`). PyMORGAN provides a dedicated
calibration engine in :mod:`pymorgan.cal`, supporting both **spectrometer calibration** (for
probe beam spectrographs) and **pulse shaper calibration** (for optical pulse shaper pump beam
masks).

Pipeline Overview
-----------------

The calibration workflow matches an experimentally measured absorption spectrum
:math:`I_{\text{meas}}(p)` against a high-resolution reference FTIR spectrum
:math:`A_{\text{ref}}(\lambda)` (such as liquid 1,4-dioxane, liquid polystyrene, or atmospheric
water vapour). Non-linear least-squares (NLS) optimisation fits optical dispersion parameters so
that the synthetic spectrum evaluated through the dispersion function reproduces the reference
absorption features.

The workflow proceeds in six stages:

1. **Load Reference Spectrum:** Load :math:`A_{\text{ref}}(\lambda)` or
   :math:`A_{\text{ref}}(\tilde{\nu})` with automatic delimiter and unit detection
   (:func:`~pymorgan.cal.load.load_reference_spectrum`).
2. **Load Experimental Data:** Parse raw detector intensity or absorbance arrays for one of twelve
   hardware setups (:func:`~pymorgan.cal.load.load_experimental_spectrum`).
3. **Pre-Alignment (Template Scanning):** Perform 1D cross-correlation scanning to estimate the
   initial wavelength shift :math:`\lambda_1` at pixel 1.
4. **NLS Parameter Fitting:** Optimise dispersion coefficients and additive polynomial baseline
   using ``scipy.optimize.least_squares`` with 1st or 2nd Savitzky–Golay derivative filtering.
5. **Axis Inversion:** Invert pixel-to-wavelength mapping :math:`p(\lambda) \to \lambda(p)`
   analytically or numerically.
6. **Export & Integration:** Save calibrated axis arrays (``CalibratedProbe.csv``,
   ``CalibratedPump.csv``, ``SingleMask.txt``, ``MultipleMask.txt``).

Data Model & Data Structures
----------------------------

The :mod:`pymorgan.cal` package defines five immutable data structures:

- :class:`~pymorgan.cal.models.ReferenceSpectrum`: Container for reference spectra library items:

  .. code-block:: python

     class ReferenceSpectrum(NamedTuple):
         name: str              # Spectrum name (e.g. 'FTIR-Dioxane')
         spectral_axis: ndarray # Spectral grid (nm or cm-1)
         absorbance: ndarray    # Absorbance array A(x)
         axis_unit: str         # Unit hint ('nm' or 'cm-1')

- :class:`~pymorgan.cal.models.ExperimentalData`: Container for loaded experimental spectrograph
  measurements:

  .. code-block:: python

     class ExperimentalData(NamedTuple):
         detector_data: list[ndarray]      # Raw detector intensity/absorbance arrays
         gratings: list[float]             # Grating groove density (grooves/mm)
         cwl: list[float]                  # Central wavelength setting (nm or cm-1)
         file_path: Path                   # File path on disk
         min_wl: float | None = None       # Lower spectral bound from metadata
         max_wl: float | None = None       # Upper spectral bound from metadata
         spectral_range: str | None = None # Detected regime ('Mid-IR', 'UV/Vis', 'NIR')

- :class:`~pymorgan.cal.models.CalibrationResult`: Output container returned by
  :func:`~pymorgan.cal.fit.fit_wavelength_axis`:

  .. code-block:: python

     class CalibrationResult(NamedTuple):
         wavelength_nm: ndarray   # Calibrated wavelength axis (nm)
         wavenumber_cm1: ndarray  # Calibrated wavenumber axis (cm-1)
         fit_params: ndarray      # Optimised dispersion parameters vector p
         residuals: ndarray       # Fit residual array
         ref_x_cut: ndarray       # Reference spectral axis in active cut region
         ref_y_cut: ndarray       # Reference absorbance in active cut region
         model_y_fit: ndarray     # Modelled experimental absorbance fit
         baseline_fit: ndarray    # Additive polynomial baseline curve
         exp_corr_y_fit: ndarray  # Baseline-corrected experimental absorbance
         ref_y_corr: ndarray      # Baseline-corrected reference spectrum

- :class:`~pymorgan.cal.models.SpectrumFitResult`: Container for 1D Gaussian pump/probe beam
  profile diagnostics:

  .. code-block:: python

     class SpectrumFitResult(NamedTuple):
         amplitude: float   # Peak amplitude (Counts)
         w0: float          # Central frequency w0 (cm-1 or nm)
         fwhm: float        # Full-Width at Half-Maximum (FWHM)
         fit_curve: ndarray # High-density fitted Gaussian profile
         axis_fit: ndarray  # High-density evaluation axis

- :class:`~pymorgan.cal.models.ProbeFitResult`: Container returned by
  :func:`~pymorgan.cal.spectrum_fit.fit_probe_spectrum`, wrapping multi-detector Gaussian
  diagnostics:

  .. code-block:: python

     class ProbeFitResult(NamedTuple):
         fits: list[SpectrumFitResult] # Per-detector Gaussian fit results
         detector_data: list[ndarray]  # Raw detector arrays that were fitted

Mathematical Optical Dispersion Models
--------------------------------------

PyMORGAN supports three physical models for mapping reference wavelength :math:`\lambda` (in nm)
to detector pixel index :math:`p(\lambda) \in [1, N_{\text{pix}}]`:

1. Linear Grating Dispersion Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For diffraction grating spectrographs in first-order paraxial approximation, the pixel position
:math:`p(\lambda)` varies linearly with wavelength around the central wavelength :math:`\lambda_0`:

.. math::

   p(\lambda) = (\lambda - \lambda_0) \cdot d + c_{\text{pix}}, \qquad
   c_{\text{pix}} = \frac{1 + N_{\text{pix}}}{2}

where :math:`d` is the linear dispersion in pixels/nm.

2. Polynomial Grating Dispersion Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For wide-bandwidth spectrographs or non-paraxial grating setups (such as UniGE TRUVIS-II),
higher-order geometric distortion is modelled via a 2nd- or 3rd-order polynomial expansion around
central wavelength :math:`\lambda_0`:

.. math::

   p(\lambda) = (\lambda - \lambda_0) \cdot d
   + 10^{-4} \, c_2 \, (\lambda - \lambda_0)^2
   + 10^{-7} \, c_3 \, (\lambda - \lambda_0)^3 + c_{\text{pix}}

where :math:`c_2` and :math:`c_3` are quadratic and cubic dispersion curvature coefficients.

3. Prism Dispersion Model (SF10 Glass)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For prism-based spectrographs (e.g. UniGE NIR-TA using an SF10 Flint glass equilateral prism),
dispersion is governed by Snell's law and the Sellmeier equation:

.. math::

   n(\lambda) = \sqrt{1 + \sum_{i=1}^{3} \frac{B_i \, \lambda_{\mu\text{m}}^2}{\lambda_{\mu\text{m}}^2 - C_i}}

with Sellmeier coefficients (:math:`\lambda_{\mu\text{m}} = \lambda / 1000`):

.. math::

   \begin{aligned}
     B_1 &= 1.62153902, & B_2 &= 0.256287842, & B_3 &= 1.64447552 \\
     C_1 &= 0.0122241457, & C_2 &= 0.0595736775, & C_3 &= 147.468793
   \end{aligned}

The deflection output angle :math:`\theta_{\text{out}}(\lambda)` from a prism with apex angle
:math:`\alpha = 60.84^\circ` at incidence angle :math:`\theta_{\text{in}}` and resulting pixel
index :math:`p(\lambda)` are:

.. math::

   \theta_{\text{out}}(\lambda) = \arcsin\!\left[ n(\lambda) \sin\!\left( \alpha - \arcsin\!\left( \frac{\sin\theta_{\text{in}}}{n(\lambda)} \right) \right) \right]

.. math::

   p(\lambda) = f_{\text{mm}} \cdot \left( \frac{1000}{25} \right) \cdot \sin\!\big[ \theta_{\text{out}}(\lambda) - \theta_{\text{out}}(\lambda_{\text{ref}}) \big] + x_{\text{shift}}

Forward Model, Baseline Correction & Derivative Cost Function
-------------------------------------------------------------

Given measured experimental spectrum array :math:`I_{\text{meas}}(p)`, the model synthesises an
experimental absorbance curve :math:`y_{\text{model}}(\lambda)` matched to reference grid points
:math:`\lambda`:

.. math::

   y_{\text{model}}(\lambda) = I_{\text{meas}}\!\big[p(\lambda)\big] \cdot s
   + b_0 + 10^{-6} \, b_1 \, (\lambda - \lambda_0)
   + 10^{-6} \, b_2 \, (\lambda - \lambda_0)^2

where :math:`s` is a global scaling factor and :math:`(b_0, b_1, b_2)` parameterise a smooth
background baseline polynomial. To eliminate broad solvent/background baseline drift during
fitting, optimisation is performed in derivative space using Savitzky–Golay numerical filters
(window size :math:`W`, polynomial order 2):

.. math::

   \mathbf{r}(\mathbf{p}) = \frac{\mathrm{d}^k}{\mathrm{d}\lambda^k} y_{\text{model}}(\lambda; \mathbf{p})
   - \frac{\mathrm{d}^k}{\mathrm{d}\lambda^k} A_{\text{ref}}(\lambda) \qquad (k \in \{1, 2\})

.. math::

   \min_{\mathbf{p}} \frac{1}{2} \|\mathbf{r}(\mathbf{p})\|_2^2 \quad \text{subject to } \mathbf{lb} \le \mathbf{p} \le \mathbf{ub}

Axis Inversion & Gaussian Pump Diagnostics
------------------------------------------

Once optimal parameters :math:`\mathbf{p}^*` are found, the calibrated wavelength axis
:math:`\lambda(p)` for pixel indices :math:`p \in \{1, \dots, N_{\text{pix}}\}` is computed:

- **Linear Grating:** :math:`\lambda(p) = \frac{p - 1}{d} + \lambda_1`.
- **Polynomial Grating:** Inverted via dense 1D grid interpolation over
  :math:`\lambda \in [250, 1000]\text{ nm}`.
- **Prism Model:** Inverted via KDTree spatial nearest-neighbour search against dense evaluation
  grid :math:`p(\lambda)`.
- **Wavenumbers:** :math:`\tilde{\nu}(p) = 10^7 / \lambda(p)`.

Pump beam profiles in optical pulse shaper calibration are fitted with a 3-parameter Gaussian
distribution:

.. math::

   G(x) = A \cdot \exp\!\left[ -4 \ln 2 \left( \frac{x - \omega_0}{\text{FWHM}} \right)^{\!2} \right]

Supported Hardware Setup Configurations
---------------------------------------

The calibration engine pre-configures twelve spectrograph hardware setups:

.. list-table:: Pre-Configured Calibration Setup Codes
   :header-rows: 1
   :widths: 8 30 16 18 10 18

   * - Code
     - Setup Name
     - Default :math:`\lambda_0`
     - Dispersion Guess
     - Degree
     - Default Bounds (:math:`\Delta\lambda_{\text{rel}}`)
   * - ``1``
     - UoS TRIR
     - :math:`2000\text{ cm}^{-1}`
     - :math:`0.30\text{ px/cm}^{-1}`
     - 2
     - ``[-1000, +1666] nm``
   * - ``2``
     - UniGE TA
     - ``530 nm``
     - ``1.10 px/nm``
     - 2
     - ``[-180, +210] nm``
   * - ``3``
     - UniGE nsTA
     - ``530 nm``
     - ``1.10 px/nm``
     - 2
     - ``[-180, +210] nm``
   * - ``4``
     - UZH Lab 2
     - :math:`2000\text{ cm}^{-1}`
     - :math:`0.30\text{ px/cm}^{-1}`
     - 2
     - ``[-455, +1250] nm``
   * - ``5``
     - UniGE NIR-TA
     - ``1000 nm``
     - ``1.00 px/nm``
     - Prism
     - ``[-200, +600] nm``
   * - ``6``
     - RAL Absorbance
     - :math:`2000\text{ cm}^{-1}`
     - :math:`0.30\text{ px/cm}^{-1}`
     - 2
     - ``[-653, +882] nm``
   * - ``7``
     - RAL Intensity
     - :math:`2000\text{ cm}^{-1}`
     - :math:`0.30\text{ px/cm}^{-1}`
     - 2
     - ``[-653, +882] nm``
   * - ``8``
     - UniGE TRIR (Int.)
     - :math:`2000\text{ cm}^{-1}`
     - :math:`0.30\text{ px/cm}^{-1}`
     - 2
     - ``[-200, +200] nm``
   * - ``9``
     - UniGE TRIR (Abs.)
     - :math:`2000\text{ cm}^{-1}`
     - :math:`0.30\text{ px/cm}^{-1}`
     - 2
     - ``[-200, +200] nm``
   * - ``10``
     - UniGE TRUVIS-II
     - ``530 nm``
     - ``1.20 px/nm``
     - 3
     - ``[-150, +220] nm``
   * - ``11``
     - Light Conversion HARPIA-TA (Int.)
     - ``515 / 440 nm``
     - ``1.00 / 1.15 px/nm``
     - 2
     - ``[-165,+235] / [-120,+160] nm``
   * - ``12``
     - PyMESS (``.h5``)
     - Auto-routed
     - Auto-routed
     - Auto
     - Routed by spectral range

HARPIA Range Presets (Setup 11)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Setup ``11`` (``Light Conversion HARPIA-TA (Int.)``) supports two optical presets via
``harpia_preset``:

- **Vis Range** (``"Vis"``): :math:`\lambda_0 = 515\text{ nm}`, dispersion guess
  :math:`d_0 = 1.0\text{ px/nm}`, relative fit bounds :math:`[-165, +235]\text{ nm}`
  (spanning :math:`350\text{–}750\text{ nm}`).
- **UV Range** (``"UV"``): :math:`\lambda_0 = 440\text{ nm}`, dispersion guess
  :math:`d_0 = 1.15\text{ px/nm}`, relative fit bounds :math:`[-120, +160]\text{ nm}`
  (spanning :math:`320\text{–}600\text{ nm}`).

PyMESS HDF5 Automatic Pathway Routing (Setup 12)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Setup ``12`` (``PyMESS``) reads ``.h5`` calibration datasets produced by PyMESS (both
spectrometer ``probe_*`` and pulse-shaper ``pump_*`` files). PyMORGAN inspects the HDF5 metadata
(``/meta/spectrometer/range``, ``type``, ``cwl``, and ``grating``) to detect the spectral regime
and routes the dispersion pathway via :func:`~pymorgan.cal.fit.resolve_effective_cal_type`:

- **Mid-IR** (:math:`\lambda_0 \ge 2500\text{ nm}` or :math:`1000 < \lambda_0 < 2500\text{ nm}`
  with grating :math:`< 200\text{ l/mm}`): Routes to **UniGE TRIR** (Setup ``8``).
- **UV/Vis** (:math:`\lambda_0 < 850\text{ nm}`): Routes to **UniGE TRUVIS-II** (Setup ``10``).
- **NIR** (:math:`850 \le \lambda_0 < 2500\text{ nm}`): Routes to **UniGE NIR-TA** (Setup ``5``).

Graphical User Interface (3-Column Inspector Layout)
----------------------------------------------------

The **Calibration Tab** combines 4 synchronised diagnostic subplots (*Raw Measurements*,
*Calculated Absorbance & Physical Baseline*, *Reference Overlay*, and *Calibration Fit*) above a
**3-Column Inspector Panel** where file selection, dispersion parameters, and fit controls remain
simultaneously visible without tab switching:

1. **Column 1 — Calibration Files & Type:** Hardware setup selector, contextual ``HARPIA Range``
   or ``PyMESS Regime`` sub-selector, Reference FTIR dropdown, and file browsers for Pump (Air),
   Pump (Solvent), Single Mask, and Multiple Mask.
2. **Column 2 — Calibration Parameters:** Side-by-side cards for **Spectrometer Setup & Bounds**
   (Detector array, Grating, CWL, Fit Axis Unit, Relative Cut Range) and **Dispersion Model & Fit
   Options** (Dispersion model radio buttons, Derivative mode, Baseline correction checkbox,
   Smoothing window, and Initial/Fitted Dispersion).
3. **Column 3 — Fit Control:** One-click **↻ Fit Spec.** and **↻ Fit Shaper** action buttons,
   live status banner, and export actions (*Save Spec. Cal.*, *Save Shaper Cal.*, *Pump Fit*).

Programmatic Usage Example
--------------------------

.. code-block:: python

   import pymorgan.cal as cal

   # 1. Load reference FTIR spectrum
   ref = cal.load_reference_spectrum("FTIR-Dioxane.csv")

   # 2. Load experimental probe spectrum (e.g. PyMESS .h5 setup 12)
   exp = cal.load_experimental_spectrum("probe_polystyrene.h5", cal_type_code=12)

   # 3. Perform wavelength calibration fit
   res = cal.fit_wavelength_axis(
       meas_y=exp.detector_data[0],
       ref_x=ref.spectral_axis,
       ref_y=ref.absorbance,
       cal_type_code=12,
       cwl=exp.cwl[0],
       use_derivative="1st",
       spectral_range=exp.spectral_range,
   )

   print(f"Calibrated span: {res.wavelength_nm[0]:.2f} nm to {res.wavelength_nm[-1]:.2f} nm")
   cal.save_calibration_file(res.wavenumber_cm1, "output_dir", filename="CalibratedProbe.csv")

   # 4. Fit probe beam Gaussian profiles (multi-detector)
   probe_fit = cal.fit_probe_spectrum(exp.detector_data, exp.cwl)
   for det_idx, fit in enumerate(probe_fit.fits):
       print(f"Detector {det_idx}: w0={fit.w0:.1f} cm-1, FWHM={fit.fwhm:.1f} cm-1")

   # 5. Query default parameters for a given setup
   params = cal.get_default_calibration_params(cal_type_code=11, harpia_preset="UV")
   print(params["degree"], params["cwl_guess"])
