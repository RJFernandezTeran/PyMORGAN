.. _user_guide_calibration:

Spectrometer & Pulse Shaper Wavelength Calibration
==================================================

Wavelength and wavenumber axis calibration maps detector pixel indices
:math:`p \in \{1, \dots, N_{\text{pix}}\}` to physical spectral values (wavelength :math:`\lambda`
in nm or wavenumber :math:`\tilde{\nu}` in :math:`\text{cm}^{-1}`). PyMORGAN provides a dedicated
calibration engine in :mod:`pymorgan.cal`, supporting both **spectrometer calibration** (for
probe beam spectrographs) and **pulse shaper calibration** (for optical pulse shaper pump beam
masks).

Mathematical Optical Dispersion Models
--------------------------------------

PyMORGAN supports three physical models for mapping reference wavelength :math:`\lambda` (in nm)
to detector pixel index :math:`p(\lambda) \in [1, N_{\text{pix}}]`:

1. **Linear Grating Dispersion Model:**

   .. math::

      p(\lambda) = (\lambda - \lambda_0) \cdot d + c_{\text{pix}}, \qquad
      c_{\text{pix}} = \frac{1 + N_{\text{pix}}}{2}

2. **Polynomial Grating Dispersion Model (2nd or 3rd Order):**

   .. math::

      p(\lambda) = (\lambda - \lambda_0) \cdot d
      + 10^{-4} \, c_2 \, (\lambda - \lambda_0)^2
      + 10^{-7} \, c_3 \, (\lambda - \lambda_0)^3 + c_{\text{pix}}

3. **Prism Dispersion Model (SF10 Glass Sellmeier Equation):**

   .. math::

      n(\lambda) = \sqrt{1 + \sum_{i=1}^{3} \frac{B_i \, \lambda_{\mu\text{m}}^2}{\lambda_{\mu\text{m}}^2 - C_i}}, \qquad
      \theta_{\text{out}}(\lambda) = \arcsin\!\left[ n(\lambda) \sin\!\left( \alpha - \arcsin\!\left( \frac{\sin\theta_{\text{in}}}{n(\lambda)} \right) \right) \right]

   .. math::

      p(\lambda) = f_{\text{mm}} \cdot \left( \frac{1000}{25} \right) \cdot \sin\!\big[ \theta_{\text{out}}(\lambda) - \theta_{\text{out}}(\lambda_{\text{ref}}) \big] + x_{\text{shift}}

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
