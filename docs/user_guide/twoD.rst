.. _user_guide_twoD:

Two-Dimensional Spectroscopy (2D-IR, 2D-ES, 2D-EV & 2D-VE)
==========================================================

Pipeline Overview & Supported 2D Modalities
-------------------------------------------

The two-dimensional branch follows the same **load → process → plot → analyse** pipeline as the
one-dimensional branch, built around a single :class:`~pymorgan.twoD.dataset.Dataset2D` object.
PyMORGAN is designed for general two-dimensional optical correlation spectroscopy across both
homonuclear and cross-regime heteronuclear experiments:

- **2D-IR (Two-Dimensional Infrared Spectroscopy):** Vibrational excitation (:math:`\omega_1` in
  the mid-IR) and vibrational detection (:math:`\omega_3` in the mid-IR).
- **2D-ES (Two-Dimensional Electronic Spectroscopy):** Electronic excitation (:math:`\omega_1` in
  the UV/Vis/NIR) and electronic detection (:math:`\omega_3` in the UV/Vis/NIR).
- **2D-EV (Two-Dimensional Electronic–Vibrational Spectroscopy):** Electronic excitation
  (:math:`\omega_1` in the UV/Vis) coupled to vibrational detection (:math:`\omega_3` in the
  mid-IR).
- **2D-VE (Two-Dimensional Vibrational–Electronic Spectroscopy):** Vibrational excitation
  (:math:`\omega_1` in the mid-IR) coupled to electronic detection (:math:`\omega_3` in the
  UV/Vis/NIR).

Because :class:`~pymorgan.twoD.dataset.Dataset2D` stores the pump (:math:`\omega_1`) and probe
(:math:`\omega_3`) axes independently, cross-regime experiments (**2D-EV** and **2D-VE**) where
the excitation and detection axes span disjoint spectral regions work seamlessly without special
casing.

.. list-table:: The 2D Pipeline Stages and Where They Live
   :header-rows: 1
   :widths: 18 34 48

   * - Stage
     - Module
     - Entry Point(s)
   * - **load**
     - :mod:`pymorgan.twoD.load`
     - :func:`pymorgan.load_2D`, :meth:`Dataset2D.from_file <pymorgan.twoD.dataset.Dataset2D.from_file>`
   * - **process**
     - :mod:`pymorgan.twoD.process`
     - :meth:`Dataset2D.process <pymorgan.twoD.dataset.Dataset2D.process>`,
       :meth:`Dataset2D.background_correct <pymorgan.twoD.dataset.Dataset2D.background_correct>`,
       :meth:`Dataset2D.recalc_average <pymorgan.twoD.dataset.Dataset2D.recalc_average>`,
       :meth:`Dataset2D.set_active_state <pymorgan.twoD.dataset.Dataset2D.set_active_state>`
   * - **plot**
     - :mod:`pymorgan.twoD.plot`
     - :meth:`Dataset2D.plot_map <pymorgan.twoD.dataset.Dataset2D.plot_map>`,
       :meth:`Dataset2D.plot_surface <pymorgan.twoD.dataset.Dataset2D.plot_surface>`,
       :meth:`Dataset2D.plot_scan_maps <pymorgan.twoD.dataset.Dataset2D.plot_scan_maps>`
   * - **analyse**
     - :mod:`pymorgan.twoD.analyse`, :mod:`pymorgan.twoD.kubo_fit`
     - ``Dataset2D.slice_at``, ``diagonal``, ``center_line_slope``, ``nodal_line_slope``,
       ``fit_2d_gaussian_map``, ``fit_2d_gaussian_global``, ``run_kubo_fit``

Data Model
----------

A two-dimensional measurement is a stack of 2D frequency–frequency correlation maps, one per
population waiting time :math:`t_2`. The signal is stored as a three-dimensional cube:

.. math::

   Z \in \mathbb{R}^{N_{\text{pump}} \times N_{\text{probe}} \times N_{t_2}}

indexed by pump frequency :math:`\omega_1`, probe frequency :math:`\omega_3`, and population time
:math:`t_2`. When single-scan loading (``load_single_scans=True``) or slow-modulation state
datasets are present, ``Dataset2D`` also tracks the per-scan and per-state tensors.

.. list-table:: Fields Stored on ``Dataset2D``
   :header-rows: 1
   :widths: 24 34 42

   * - Field
     - Shape / Type
     - Meaning
   * - ``Z``
     - :math:`N_{\text{pump}} \times N_{\text{probe}} \times N_{t_2}`
     - 2D signal cube (in :math:`\text{mOD}`)
   * - ``pump``
     - :math:`N_{\text{pump}}`
     - Pump (excitation) axis :math:`\omega_1`
   * - ``probe``
     - :math:`N_{\text{probe}}`
     - Probe (detection) axis :math:`\omega_3`
   * - ``delays``
     - :math:`N_{t_2}`
     - Population waiting times :math:`t_2`
   * - ``units``
     - ``dict``
     - Unit labels and metadata dictionary
   * - ``raw_signal_ss``
     - :math:`N_{t_1/\text{pump}} \times N_{\text{probe}} \times N_{t_2} \times N_s`
     - Per-scan raw signal tensor (when ``load_single_scans=True``)
   * - ``states_data``
     - ``dict[str, ndarray]``
     - Slow-modulation / polarisation state cubes
   * - ``states_signal_ss``
     - ``dict[str, ndarray]``
     - Per-state single-scan tensors
   * - ``nscans``
     - ``int``
     - Total number of individual scans :math:`N_s`
   * - ``active_scan_indices``
     - ``list[int]``
     - 0-based indices of scans included in the active average

Loading and the Loader Registry
-------------------------------

As in 1D, formats are resolved through a registry (:mod:`pymorgan.twoD.load`). Calling
:func:`pymorgan.available_map_loaders` lists all registered 2D readers:

.. code-block:: python

   import pymorgan as pm

   data = pm.load_2D("scan.p2dat", data_type="P2DAT")
   print(pm.available_map_loaders())

.. list-table:: Supported Two-Dimensional Data Formats in PyMORGAN
   :header-rows: 1
   :widths: 22 28 50

   * - Format Token
     - Instrument / Origin
     - Description
   * - ``P2DAT``
     - PyMORGAN standard
     - Comma-separated processed signal cube (``.p2dat``)
   * - ``MESS_2DIR``
     - MESS 2D
     - Multi-file directory with raw/signal states, count, probe, and ref files
   * - ``UoS_2DIR``
     - Univ. of Sheffield
     - Sheffield 2D acquisition files
   * - ``RAL_RAW``
     - STFC / Central Laser Facility
     - Raw Ultra 2D interferograms (phased & Fourier-transformed on load)
   * - ``RAL_Proc``
     - STFC / Central Laser Facility
     - Processed Ultra 2D signal cubes
   * - ``PyMESS_2D``
     - PyMESS HDF5
     - Unified PyMESS 2D HDF5 datasets (``.h5``, alias ``PyMESS_2DIR``)

- **The P2DAT format:** The first row is ``0, 0, t2^(1), t2^(2), ...`` (the population waiting
  times); each subsequent row is ``pump, probe, S(t2^(1)), ...``. The flat pump/probe ordering is
  reshaped with zero-copy Fortran order directly into the ``Z[pump, probe, t2]`` cube in
  :math:`\text{mOD}`.
- **The MESS 2DIR & PyMESS_2D formats:** Automatically detect wobbler and chopper modulation
  modes, perform interferogram normalisation (:math:`I / \text{counts}`), and compute optical
  density :math:`\Delta\mathrm{OD} = -1000 \log_{10}(I_{\text{probe}} / I_{\text{ref}})`. For
  PyMESS 2D HDF5 files, slow-modulated state datasets under
  ``/data/slow_modulation/state_XX_<name>`` (with or without ``/data/interferograms_raw``) and
  their ``*_individual_scans`` companions are loaded into ``states_data`` and
  ``states_signal_ss``. Derived ``Magic Angle`` and ``Anisotropy`` states automatically inherit
  per-scan tensors computed from the parallel and perpendicular states, and switching states via
  :meth:`~pymorgan.twoD.dataset.Dataset2D.set_active_state` preserves ``active_scan_indices``.

Interferogram Processing, Phasing, and Fourier Transformation
-------------------------------------------------------------

For raw time-domain 2D measurements (interferometer- or pulse-shaper-based acquisitions), the
data must be baseline-corrected, windowed, Fourier-transformed along the coherence time axis
:math:`t_1 \to \omega_1`, and phased to produce pure absorptive 2D correlation maps via
:meth:`Dataset2D.process <pymorgan.twoD.dataset.Dataset2D.process>` or the *Phasing/FT* GUI
sub-tab.

Baseline Correction and Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Pulse-shaper data:** Mean subtraction across the time-domain points removes DC offsets:

  .. math::

     S_{\text{corr}}(t_1, \omega_3, t_2) = S(t_1, \omega_3, t_2) - \langle S(t_1, \omega_3, t_2) \rangle_{t_1}

- **Interferometer data:** A running median filter with an odd kernel size of ``101`` points is
  applied along :math:`t_1` to both the reference interferogram and the transient signal,
  isolating the oscillatory fringe signal from slow baseline curvature.

Rotating-Frame Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

In rotating-frame acquisitions (such as AOTF or pulse-shaper spectrometers), the experimental
time step :math:`\Delta t_1` and rotating-frame carrier frequency :math:`\omega_0` (in
:math:`\text{cm}^{-1}`) define the spectral frequency axis:

.. math::

   \omega_1(k) = k \cdot \Delta\nu + \omega_0, \qquad
   \Delta\nu = \frac{1}{N_{\text{FT}} \Delta t_1 c_0}

where :math:`c_0 = 2.99792458\times 10^{-5}\text{ cm/fs}` is the speed of light and
:math:`N_{\text{FT}}` is the total number of Fourier transform points. For interferometer
systems, :math:`\omega_0 = 0` and :math:`\Delta t_1 = 2.11079\text{ fs}` (the HeNe laser
interference fringe period).

Interferogram Time-Zero (:math:`t_1 = 0`, Bin-Zero) via the Fourier Shift Theorem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Determining the exact zero-path-difference position (:math:`t_1 = 0`, or "binzero" :math:`b_0`)
is essential for obtaining pure absorptive spectra without phase distortion. Physically, placing
the time origin at :math:`t_1 = 0` requires that the phase spectrum
:math:`\phi(\omega) = \arg(\mathcal{F}\{I\})(\omega)` have zero first derivative across the
optical carrier frequency :math:`\omega_0`:

.. math::

   \left.\frac{\mathrm{d}\phi}{\mathrm{d}\omega}\right|_{\omega_0} = 0

Rather than iteratively rolling the time-domain signal and computing hundreds of redundant FFTs,
PyMORGAN determines :math:`b_0` analytically using the **Discrete Fourier Shift Theorem**.
Circularly shifting a discrete sequence of length :math:`N` by :math:`b` bins,
:math:`I'(t_1) = \operatorname{roll}(I, -b)`, corresponds to an exact linear phase ramp in the
frequency domain:

.. math::

   \mathcal{F}\{\operatorname{roll}(I, -b)\}[k] = X[k] \cdot \exp\!\left(+i \frac{2\pi k}{N} b\right)

Evaluating the unwrapped phase difference across a symmetric window
:math:`[k_{\max} - \Delta k/2, k_{\max} + \Delta k/2]` around the carrier peak gives:

.. math::

   \Delta\phi(b) = \phi_b[k_2] - \phi_b[k_1]
   = \underbrace{(\phi_0[k_2] - \phi_0[k_1])}_{\Delta\phi_0}
   + \underbrace{\left(\frac{2\pi \Delta k}{N}\right)}_{m} \cdot b

Because :math:`\Delta\phi(b)` is strictly linear in :math:`b`, the exact binzero position
:math:`b_0` where :math:`\Delta\phi(b_0) \equiv 0 \pmod{2\pi}` is solved in closed form in under
:math:`1\,\mu\text{s}` per delay:

.. math::

   b_0 = \operatorname{round}\!\left( \frac{2\pi m_{\text{wrap}} - \Delta\phi_0}{m} \right), \qquad
   m_{\text{wrap}} = \operatorname{round}\!\left(\frac{\Delta\phi_0 + m \cdot b_{\max}}{2\pi}\right)

Apodisation Window Functions & Zero-Padding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To eliminate truncation ripples (sinc sidelobes) arising from the finite :math:`t_1` acquisition
window, PyMORGAN applies an apodisation window :math:`w(v)` centred at binzero :math:`b_0` for
:math:`v = |q - (b_0 + 1)| \le M = N_{\text{bins}} - b_0 - 1`:

- **Box:** :math:`w(v) = 1.0`
- **Cosine (Cos, default):** :math:`w(v) = \cos\!\left(\frac{\pi v}{2 M}\right)`
- **Cos²:** :math:`w(v) = \cos^2\!\left(\frac{\pi v}{2 M}\right)`
- **Cos³:** :math:`w(v) = \cos^3\!\left(\frac{\pi v}{2 M}\right)`
- **Hanning:** :math:`w(v) = 0.5 + 0.5 \cos\!\left(\frac{\pi v}{M}\right)`
- **Hamming:** :math:`w(v) = 0.54 + 0.46 \cos\!\left(\frac{\pi v}{M}\right)`

Zero-padding supports integer multiplication factors (:math:`1\times`, :math:`2\times`,
:math:`4\times`) and automatic padding to the next power of two (:math:`2^k`).

IRLS Phasing Models, Pump Correction & Scatter Subtraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The frequency-dependent phase error :math:`\phi(\omega_1)` is modelled by fitting a Constant,
Linear, Quadratic, or Cubic polynomial by iteratively reweighted least squares (IRLS) over a
window of :math:`\pm N` points around the carrier peak:

.. math::

   S_{\text{phased}}(\omega_1, \omega_3, t_2) = \operatorname{Re}\!\left( S_{\text{raw}}(\omega_1, \omega_3, t_2) \cdot e^{-i \phi(\omega_1)} \right)

When **Pump correction** is enabled, the signal is normalised by the excitation spectrum profile:

.. math::

   S_{\text{norm}}(\omega_1, \omega_3, t_2) = \frac{S_{\text{phased}}(\omega_1, \omega_3, t_2)}{|I_{\text{pump}}(\omega_1)| / \max |I_{\text{pump}}|}

Background :math:`t_2` map subtraction is performed via:

.. code-block:: python

   data.background_correct(reference=-1)      # by t2 index
   data.background_correct(t2=50)             # by nearest t2 value
   data.background_correct(do_correct=False)  # passthrough

2D Single-Scan Contour Array Explorer & Scan Recombination
----------------------------------------------------------

When ``load_single_scans=True`` is enabled in :class:`~pymorgan.settings.Settings`, multi-scan 2D
datasets (``MESS_2DIR``, ``PyMESS_2D``, ``UoS_2DIR``, and ``RAL_RAW``) retain their individual scan
tensors in ``raw_signal_ss`` (shape
:math:`N_{t_1/\text{pump}} \times N_{\text{probe}} \times N_{t_2} \times N_s`) and optional
per-scan interferograms ``raw_interferogram_ss``.

- **Slice-level per-scan map evaluation (**\ :meth:`~pymorgan.twoD.dataset.Dataset2D.compute_scan_maps_at_t2`\ **):**
  Computes 2D frequency-domain maps ``(label, scan_indices_0based, Z_map)`` at a chosen :math:`t_2`
  delay for each individual scan or group of :math:`N` consecutive binned scans (``bin_size``).
  For time-domain shaper/interferometer datasets, only the requested :math:`t_2` slice (plus the
  scatter-subtraction reference slice if ``bkg_sub=True``) is Fourier-transformed and phased per
  scan/bin.
- **Multi-panel contour array (**\ :meth:`~pymorgan.twoD.dataset.Dataset2D.plot_scan_maps` /
  :func:`~pymorgan.twoD.plot.plot_scan_map_array`\ **):** Renders a grid of 2D contour maps across
  all individual or binned scans alongside an **Average** comparison panel on a shared colour scale.
- **Scan recombination (**\ :meth:`~pymorgan.twoD.dataset.Dataset2D.recalc_average`\ **):**
  Recomputes the averaged 2D dataset across any selected 0-based ``scan_indices``, updating
  ``raw_signal``, ``raw_interferogram``, ``Z_R``, and ``active_scan_indices``.

.. code-block:: python

   pm.update_settings(load_single_scans=True)
   d2 = pm.load_2D("scan_folder", data_type="MESS_2DIR")

   # Plot a multi-panel grid of 2D contour maps at t2 = 0.5 ps (binning 2 scans per panel)
   fig, axes, panels = d2.plot_scan_maps(0.5, bin_size=2, include_average=True)

   # Recompute the dataset average excluding outlier scan index 3 (0-based)
   keep = [i for i in range(d2.nscans) if i != 3]
   d2.recalc_average(keep).process(apodise_method="1", phase_method="Linear")

In ``pymorgan-gui``, the **Single Scans** box appears in the **2D** tab whenever a dataset with
multiple scans is loaded:

- **Bin Scans** sets the number of consecutive scans averaged into each subplot panel.
- **Plot Scan Contours** opens an interactive multi-panel figure where clicking any scan panel
  toggles its ``[Excluded]`` status (highlighted in red), and clicking the embedded
  **Recalc. Avg. (Exclude Selected)** button recomputes the dataset average and refreshes the main
  2D GUI in place.
- **Recalc. Avg.** opens a dialog to enter custom 1-based scan ranges (e.g. ``1-4, 6-10``).

Plotting, Cross-Regime Diagonal Handling & Unit Conversions
-----------------------------------------------------------

:meth:`~pymorgan.twoD.dataset.Dataset2D.plot_map` renders the map nearest a requested :math:`t_2`
as a pump-vs-probe contour and returns a ``Map2DAxes(ax, top, cbar)`` named tuple:

.. code-block:: python

   out = data.plot_map(0.5)                                  # single t2 map
   ftir = pm.load_spectrum("sample_FTIR.csv", kind="absorption")
   data.plot_map(0.5, top_spectrum=ftir, top_label="FTIR")   # with top reference panel

Two behaviours are specific to 2D plotting:

1. **Diagonal handling across 2D-IR, 2D-ES, 2D-EV, and 2D-VE:** The :math:`\omega_1 = \omega_3`
   diagonal line is drawn *only* where the pump and probe axes overlap; when their ranges are
   disjoint (as in **2D-EV** and **2D-VE**), no diagonal is drawn.
2. **Axis units and the** :math:`10^3` **rule:** Pump and probe axes can be displayed in any of
   the units below via ``twoD_freq_unit`` or ``freq_unit=``, pivoting through wavelength
   :math:`\lambda\,[\mathrm{nm}] = P_{\text{in}} / x_{\text{in}}`:

.. list-table:: Display Units of the 2D Spectral Axes
   :header-rows: 1
   :widths: 18 24 32 26

   * - Token
     - Quantity
     - :math:`P` (:math:`\text{unit}\cdot\text{nm}`)
     - Typical Mid-IR Value
   * - ``cm-1``
     - Wavenumber
     - :math:`10^{7}`
     - ``2000``
   * - ``in-1``
     - Wavenumber
     - :math:`2.54 \times 10^{7}`
     - ``5080``
   * - ``nm``
     - Wavelength
     - —
     - ``5000``
   * - ``THz``
     - Frequency
     - :math:`2.99792458 \times 10^{5}`
     - ``60``
   * - ``eV``
     - Energy
     - :math:`1239.841984`
     - ``0.25``

Whenever an axis exceeds :math:`5 \times 10^{3}` in the display unit (e.g. UV/Vis wavenumbers in
2D-ES, 2D-EV, or 2D-VE), its tick labels are divided by :math:`1000` and the axis label gains a
:math:`10^3` prefix (e.g. :math:`\omega_1\;(10^3\;\mathrm{cm}^{-1})`).

Spectral Diffusion Analysis (CLS, IvCLS, NLS)
---------------------------------------------

:mod:`pymorgan.twoD.analyse` extracts Centre-Line Slope (CLS), Inverse CLS (IvCLS), and
Nodal-Line Slope (NLS) across :math:`t_2` waiting times using Huber-loss linear regression and
two-stage sub-pixel extremum refinement:

1. Cubic spline upsampling onto a grid ``sd_interpolation_factor`` times finer.
2. Local least-squares peak fitting (``quadratic``, ``cubic``, ``quartic``, ``gaussian``, or
   ``lorentzian``) to the measured samples around the coarse extremum.

.. code-block:: python

   pump, probe, m = data.slice_at(0.5)          # 2D map at t2 = 0.5 ps
   freq, sig      = data.diagonal(0.5)          # signal along omega_1 = omega_3

   cls  = data.center_line_slope()              # [t2, slope]
   both = data.center_line_slope(both=True)     # [t2, CLS, IvCLS]
   nls  = data.nodal_line_slope()               # [t2, slope]

Advanced 2D Analysis Dialogs & Tools
------------------------------------

Direct Kubo Lineshape & Generalized FFCF Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **Kubo Fit...** dialog (``kubo_dialog.py``, backed by :mod:`pymorgan.twoD.kubo_fit`) performs
full third-order response-function simulation and multi-core non-linear least-squares fitting of 2D
spectra using generalized Frequency Fluctuation Correlation Functions (FFCFs):

.. math::

   C(t) = \Delta_0^2 + \sum_{m=1}^{M} \Delta_m^2 e^{-t/\tau_{c,m}}

The dephasing lineshape function :math:`g(t)` combines pure homogeneous dephasing
(:math:`\Delta\omega_{T_2}`, with :math:`1/T_2^* = \pi c_0 \Delta\omega_{T_2}`), static
inhomogeneous broadening (:math:`\Delta_0`), and an arbitrary number of Kubo exponential decay
terms (:math:`\Delta_m`, correlation time :math:`\tau_{c,m}`):

.. math::

   g(t) = \pi c_0 \Delta\omega_{T_2}\, t + \frac{1}{2}\sigma_0^2 t^2 + \sum_{m=1}^{M} \sigma_m^2 \tau_{c,m}^2 \left( e^{-t/\tau_{c,m}} + \frac{t}{\tau_{c,m}} - 1 \right),
   \qquad \sigma_i = \frac{2\pi c_0 \, \Delta_i}{2\sqrt{2\ln 2}}

- **Shared or per-band independent FFCF groups:** Multiple FFCF groups can be defined and assigned
  independently to the excitation (:math:`\omega_1`, ``ffcf_id``) and detection (:math:`\omega_3`,
  ``ffcf_id_w3``) coordinates of each oscillator band.
- **Diagonal and cross-peak bands with bath correlation:** Supports both diagonal oscillators
  (:math:`\omega_1 = \omega_3`) and off-diagonal cross-peaks (:math:`\omega_1 \ne \omega_3`) with
  parameter linking (``link_w1``, ``link_w3``, ``link_anharm``) and a tunable cross-peak bath
  correlation factor :math:`\eta_{AB} \in [-1, 1]`.
- **Variable Projection (VarPro) & multi-core parallelisation:** Uses Trust-Region Reflective
  (``scipy.optimize.least_squares``) with analytical Variable Projection to solve the linear GSB
  and ESA pathway amplitudes (:math:`A_{\text{GSB}}, A_{\text{ESA}}`) at each step, alongside
  full-parameter TRF and ``L-BFGS-B`` solvers. Time-domain evaluation uses separable 1D outer
  products, batched 2D FFTs, separable 2D rectilinear interpolation, and multi-core parallel
  execution across :math:`t_2` delays (``ProcessPool`` / ``ThreadPool``) with parallel Jacobian
  column evaluation.
- **Joint 2D + 1st-order FTIR fitting & diagnostics:** Simultaneously fits 2D correlation maps
  across selected :math:`t_2` delays and the analytical 1st-order steady-state FTIR spectrum
  (weighted by ``kubo_ftir_weight``), with live FFCF :math:`C(t)` and :math:`g(t)` diagnostic
  plots and JSON/TOML model preset import/export.

Interactive 2D Gaussian Peak Fitting & Cross-Peak Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **2D Gaussian Fit...** dialog (``twoD_gaussian_dialog.py``, backed by
:func:`~pymorgan.twoD.analyse.fit_2d_gaussian_map` and
:func:`~pymorgan.twoD.analyse.fit_2d_gaussian_global`) fits a sum of 2D correlated elliptical
Gaussian peaks to diagonal and cross-peak features:

.. math::

   G(\omega_1, \omega_3) = A \exp\!\left( -\frac{1}{2(1-\rho^2)} \left[ \frac{(\omega_1 - \omega_{1,0})^2}{\sigma_1^2} + \frac{(\omega_3 - \omega_{3,0})^2}{\sigma_3^2} - \frac{2\rho(\omega_1 - \omega_{1,0})(\omega_3 - \omega_{3,0})}{\sigma_1 \sigma_3} \right] \right)

Users can click directly on the 2D contour map to seed peaks, automatically generate linked
cross-peak pairs from diagonal modes via the ``CrossPeakGeneratorDialog``, link centres and
anharmonicities across peaks, and fit either single :math:`t_2` slices or globally across the
entire :math:`t_2` stack.

2D ROI Volume Integration & Dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **2D Integral Dynamics...** dialog (``twoD_integral_dialog.py``) provides interactive
rectangular region-of-interest (ROI) selection on the :math:`(\omega_1, \omega_3)` map using a
draggable ``RectangleSelector``. It integrates the 2D signal volume

.. math::

   I(t_2) = \iint_{\mathrm{ROI}} Z(\omega_1, \omega_3, t_2)\,\mathrm{d}\omega_1\,\mathrm{d}\omega_3

across all population waiting times :math:`t_2`, plots the resulting kinetic trace, and fits it
with single- or multi-exponential decay models.

2D Spectrum Subtraction
~~~~~~~~~~~~~~~~~~~~~~~

The **Subtract 2D Spectra...** dialog (``twoD_subtraction_dialog.py``) loads a reference 2D
dataset (such as pure solvent or buffer), interpolates it onto the sample dataset's
:math:`(\omega_1, \omega_3, t_2)` grid, and performs scaled subtraction
:math:`Z_{\mathrm{corr}} = Z_{\mathrm{sample}} - k \, Z_{\mathrm{ref}}` with a live three-panel
preview (Sample, Scaled Reference, and Difference Map).

Graphical Interface Sub-Tabs & Movie Export
-------------------------------------------

- **Phasing/FT Sub-Tab:** Controls Apodisation (``Box``, ``Cos``, ``Cos^2``, ``Cos^3``,
  ``Hanning``, ``Hamming``, ``None``), Phase fit order (``Constant``, ``Linear``, ``Quadratic``,
  ``Cubic``), Phase range (:math:`\pm N`), Zero-padding, and Pump correction, displaying the
  fitted polynomial phase coefficients live.
- **Other plt. Sub-Tab (**\ ``TD`` / ``PH`` / ``CAL`` + **Pop** button\ **):** Displays the
  time-domain interferogram (``TD``), Fourier amplitude + Gaussian fit + unwrapped phase & fit on
  a twin axis (``PH``), or the calibrated probe axis (``CAL``), with a **Pop** button to detach a
  synchronised resizable window.
- **Movie / GIF / WebP / APNG Export (**\ ``Make Movie...``\ **):** Exports animated GIF, WebP, or
  APNG videos (via Pillow) stepping through :math:`t_2` population times with configurable delay
  range, frame rate, figure dimensions, and fixed or per-frame normalised :math:`Z`-scale.
