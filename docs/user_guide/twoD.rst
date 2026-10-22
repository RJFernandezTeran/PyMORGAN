.. _user_guide_twoD:

Two-Dimensional Data (2D-IR & 2D-ES)
====================================

Pipeline Overview & Data Model
------------------------------

The two-dimensional branch follows the same **load → process → plot → analyse** architecture,
built around :class:`~pymorgan.twoD.dataset.Dataset2D`. A 2D measurement is stored as a signal
cube:

.. math::

   Z \in \mathbb{R}^{N_{\text{pump}} \times N_{\text{probe}} \times N_{t_2}}

indexed by pump frequency :math:`\omega_1`, probe frequency :math:`\omega_3`, and population
waiting time :math:`t_2`.

.. list-table:: Supported 2D Data Formats
   :header-rows: 1
   :widths: 22 28 50

   * - Format Token
     - Instrument / Origin
     - Description
   * - ``P2DAT``
     - PyMORGAN standard
     - Comma-separated processed signal cube (``.p2dat``)
   * - ``MESS_2DIR``
     - MESS 2D-IR
     - Multi-file directory with raw/signal states, count, probe, and ref files
   * - ``UoS_2DIR``
     - Univ. of Sheffield
     - Sheffield 2D-IR acquisition files
   * - ``RAL_RAW``
     - STFC / Central Laser Facility
     - Raw Ultra 2D-IR interferograms (phased & transformed on load)
   * - ``RAL_Proc``
     - STFC / Central Laser Facility
     - Processed Ultra 2D-IR cubes
   * - ``PyMESS_2DIR``
     - PyMESS HDF5
     - Unified PyMESS 2D-IR HDF5 datasets (``.h5``)

Interferogram Processing, Phasing, and Fourier Transformation
-------------------------------------------------------------

Closed-Form Bin-Zero (:math:`t_1 = 0`) via the Fourier Shift Theorem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rather than iteratively rolling the time-domain interferogram and computing redundant FFTs,
PyMORGAN determines the zero-path-difference bin :math:`b_0` analytically using the
**Discrete Fourier Shift Theorem**. Circularly shifting a sequence of length :math:`N` by
:math:`b` bins corresponds to a linear phase ramp in frequency space:

.. math::

   \mathcal{F}\{\operatorname{roll}(I, -b)\}[k] = X[k] \cdot \exp\!\left(+i \frac{2\pi k}{N} b\right)

Evaluating the unwrapped phase difference across a symmetric window
:math:`[k_{\max} - \Delta k / 2, k_{\max} + \Delta k / 2]` around the carrier peak yields a
strictly linear relation in :math:`b`, which is solved in closed form:

.. math::

   b_0 = \operatorname{round}\!\left( \frac{2\pi m_{\text{wrap}} - \Delta\phi_0}{m} \right), \qquad
   m = \frac{2\pi \Delta k}{N}

Apodisation, Zero-Padding, and IRLS Phasing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Apodisation windows:** ``Box``, ``Cos`` (default), ``Cos^2``, ``Cos^3``, ``Hanning``,
  ``Hamming``, or ``None``.
- **Zero-padding:** Supports :math:`1\times`, :math:`2\times`, :math:`4\times`, and next power of
  two (:math:`2^k`).
- **Phasing models:** Fits unwrapped phase :math:`\phi(\omega_1)` around the spectral peak via
  iteratively reweighted least squares (IRLS) using Constant, Linear, Quadratic, or Cubic
  polynomials:

  .. math::

     S_{\text{phased}}(\omega_1, \omega_3, t_2) = \operatorname{Re}\!\left( S_{\text{raw}}(\omega_1, \omega_3, t_2) \cdot e^{-i \phi(\omega_1)} \right)

Spectral Diffusion & Kubo Lineshape Analysis
--------------------------------------------

:mod:`pymorgan.twoD.analyse` provides Centre-Line Slope (CLS), Inverse CLS (IvCLS), and
Nodal-Line Slope (NLS) extraction with sub-pixel refinement (cubic spline upsampling by
``sd_interpolation_factor`` followed by local quadratic/cubic/quartic/Gaussian/Lorentzian peak
fitting and Huber-loss linear regression):

.. code-block:: python

   import pymorgan as pm

   data = pm.load_2D("testData/test2DIR.p2dat", data_type="P2DAT")
   data.background_correct(reference=-1)
   data.plot_map(0.5, ShowLines=True)

   cls  = data.center_line_slope()          # [t2, slope]
   both = data.center_line_slope(both=True) # [t2, CLS, IvCLS]
   nls  = data.nodal_line_slope()           # [t2, slope]
