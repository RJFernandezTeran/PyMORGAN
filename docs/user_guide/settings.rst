.. _user_guide_settings:

Settings & Aesthetics
=====================

Every presentation choice that is not intrinsic to the data lives in a single
:class:`~pymorgan.settings.Settings` object: the Matplotlib style profile, label and unit
conventions, colour scale, axis units, and the parameters of the spectral-diffusion and chirp
analyses. One instance is active at a time; the plotting routines read it, and any call may
override individual values.

.. code-block:: python

   import pymorgan as pm

   pm.load_settings("settings.toml")     # read a file
   pm.update_settings(cmap="Seismic", quick_plots=True)
   s = pm.get_settings()                 # the active object
   pm.save_settings("settings.toml")     # write it back, comments preserved

   with pm.use_settings(profile="poster"):   # temporary override
       data.plot_contour()

Values are coerced to the declared type, so a string from a text field or a TOML file becomes the
right enum, number, path, or pair: ``pm.update_settings(contour_figsize="9, 5")`` and
``pm.update_settings(x_axis_unit="nm")`` both work. An unknown name raises ``TypeError`` rather
than being silently stored.

The Configuration File & GUI Editor
-----------------------------------

- ``settings.toml`` is read at start-up (the GUI looks in the working directory first, then next
  to the package) and written back by **Save permanently** in the settings dialog. Comments and
  ordering survive a round trip.
- In ``pymorgan-gui``, **View → Aesthetics / Settings...** (or standalone ``pymorgan-settings``)
  opens a panel whose widgets are generated from the settings themselves and grouped into four
  tabs: **Common**, **1D**, **2D**, and **GUI & Defaults**.

Common Settings
---------------

Shared by every plot: style profile, labels, colour scale, and spectral axis units.

.. list-table:: Common Settings
   :header-rows: 1
   :widths: 26 28 32 14

   * - Key
     - Label in Panel
     - Type / Choices
     - Default
   * - ``profile``
     - Style profile
     - ``regular``, ``poster``, ``inset``
     - ``regular``
   * - ``font_scale``
     - Font scale
     - ``float``
     - ``1.0``
   * - ``label_style``
     - Axis-label style
     - ``()``, ``[]``, ``/``, ``""``
     - ``()``
   * - ``delta_a_units``
     - :math:`\Delta A` units
     - ``mOD``, ``x1E3``
     - ``x1E3``
   * - ``cmap``
     - Colourmap
     - ``DkRd/Wh/DkBu``, ``Rd/Wh/Bu v2``, ``Seismic``, ``Jet``, ``vik``, ``berlin``
     - ``DkRd/Wh/DkBu``
   * - ``white_levels``
     - White/Zero levels
     - ``float``
     - ``0``
   * - ``asinh_pct``
     - arcsinh linear %
     - ``float``
     - ``5.0``
   * - ``x_axis_unit``
     - Spectral X-axis unit
     - ``native``, ``nm``, ``cm-1``, ``eV``, ``THz``
     - ``native``
   * - ``secondary_axis``
     - Complementary X axis
     - ``bool``
     - ``False``
   * - ``round_labels``
     - Round trace labels
     - ``bool``
     - ``True``
   * - ``quick_plots``
     - Use quick plots (fast redraws)
     - ``bool``
     - ``False``
   * - ``ss_line_dashes``
     - Steady-state dash pattern
     - ``text``
     - ``4, 3``

1D Settings
-----------

Delay-axis scaling and the appearance of kinetic / transient-spectra traces.

.. list-table:: 1D Settings
   :header-rows: 1
   :widths: 28 28 30 14

   * - Key
     - Label in Panel
     - Type / Choices
     - Default
   * - ``time_axis_scale``
     - Time-axis scale
     - ``symlog``, ``log``, ``lin``
     - ``symlog``
   * - ``time_axis_label``
     - Time-axis labels
     - ``power``, ``decimal``, ``mixed``
     - ``power``
   * - ``kinetics_marker_size``
     - Kinetics marker size
     - ``float``
     - ``6``
   * - ``kinetics_line_width``
     - Kinetics line width
     - ``float``
     - ``1.5``
   * - ``legend_label_spacing``
     - Legend entry spacing
     - ``float``
     - ``0.5``
   * - ``prune_legend``
     - Prune legend
     - ``bool``
     - ``True``
   * - ``max_legend_entries``
     - Max legend entries
     - ``float``
     - ``15``
   * - ``ss_abs_color``
     - Steady-state abs. colour
     - ``text``
     - ``b``
   * - ``ss_em_color``
     - Steady-state em. colour
     - ``text``
     - ``r``
   * - ``ss_line_width``
     - Steady-state line width
     - ``float``
     - ``1.5``
   * - ``ss_line_alpha``
     - Steady-state line alpha
     - ``float``
     - ``0.15``
   * - ``ss_fill_alpha``
     - Steady-state fill alpha
     - ``float``
     - ``0.05``
   * - ``inherit_cut_limits``
     - Cuts inherit plot limits
     - ``bool``
     - ``True``
   * - ``load_single_scans``
     - Load single scans (1D & 2D)
     - ``bool``
     - ``False``
   * - ``chirp_boundary_handling``
     - Chirp boundary handling
     - ``crop``, ``extrapolate``
     - ``crop``
   * - ``parallel_fitting``
     - Chirp parallel fitting
     - ``threadpool``, ``processpool``, ``disabled``
     - ``threadpool``
   * - ``solvent_per_pixel_dt``
     - Per-pixel solvent :math:`t_0` offset
     - ``bool``
     - ``False``
   * - ``solvent_per_pixel_scale``
     - Per-pixel solvent amplitude scale
     - ``bool``
     - ``True``
   * - ``error_shading``
     - Error shading (:math:`\pm 1\;\text{std}`)
     - ``bool``
     - ``False``
   * - ``error_shading_alpha``
     - Error shading alpha
     - ``float``
     - ``0.2``

2D Settings
-----------

Map appearance, axis units, and spectral-diffusion / Kubo analysis parameters.

.. list-table:: 2D Settings
   :header-rows: 1
   :widths: 28 28 30 14

   * - Key
     - Label in Panel
     - Type / Choices
     - Default
   * - ``freq_label``
     - 2D frequency labels
     - ``Omega_n``, ``Omega_PP``, ``Pump-Probe``, ``Omega_n/2pic``
     - ``Omega_n``
   * - ``pump_axis``
     - 2D pump axis orientation
     - ``Horizontal``, ``Vertical``
     - ``Horizontal``
   * - ``twoD_freq_unit``
     - 2D axis unit
     - ``cm-1``, ``in-1``, ``nm``, ``THz``, ``eV``
     - ``cm-1``
   * - ``t2_label_style``
     - :math:`t_2` delay label format
     - ``t2=``, ``plain``
     - ``t2=``
   * - ``cls_color``
     - CLS colour
     - ``text``
     - ``b``
   * - ``ivcls_color``
     - IvCLS colour
     - ``text``
     - ``r``
   * - ``nls_color``
     - NLS colour
     - ``text``
     - ``g``
   * - ``overlay_draw_style``
     - Overlay draw style
     - ``Points``, ``Lines``, ``Both``
     - ``Points``
   * - ``overlay_linewidth``
     - Overlay linewidth
     - ``float``
     - ``1.5``
   * - ``sd_intensity_threshold``
     - Intensity threshold (top x%)
     - ``float``
     - ``0.1``
   * - ``sd_peak_range``
     - Peak search range (:math:`\text{cm}^{-1}`)
     - ``float``
     - ``10``
   * - ``sd_peak_type``
     - Target peak type
     - ``min``, ``max``
     - ``min``
   * - ``sd_interpolation``
     - Peak interpolation
     - ``quadratic``, ``cubic``, ``quartic``, ``gaussian``, ``lorentzian``, ``none``
     - ``quadratic``
   * - ``sd_interpolation_factor``
     - Interpolation factor (upsampling)
     - ``int``
     - ``4``
   * - ``kubo_ftir_weight``
     - Kubo joint fit FTIR vs 2D weight
     - ``float``
     - ``1.0``

GUI and Defaults Settings
-------------------------

Application preferences and default selections when the GUI starts.

.. list-table:: GUI and Defaults Settings
   :header-rows: 1
   :widths: 28 28 30 14

   * - Key
     - Label in Panel
     - Type / Choices
     - Default
   * - ``gui_theme``
     - GUI theme
     - ``Fusion``, ``Fusion Dark``
     - ``Fusion``
   * - ``splash_duration``
     - Splash-screen duration (s)
     - ``float``
     - ``2.0``
   * - ``default_datadir``
     - Default data directory
     - ``text``
     - ``None``
   * - ``default_oneD_datatype``
     - Default 1D data type
     - 1D loader token
     - ``PDAT``
   * - ``default_twoD_datatype``
     - Default 2D data type
     - 2D loader token
     - ``P2DAT``
   * - ``default_calibration_datatype``
     - Default calibration setup
     - ``text``
     - ``UniGE TRIR (Intensity)``
   * - ``styles_dir``
     - Style directory (blank = bundled)
     - ``text``
     - ``None``
   * - ``contour_figsize``
     - Contour figure size (w, h in)
     - ``text``
     - ``8, 6``
   * - ``kinetics_figsize``
     - Kinetics figure size (w, h in)
     - ``text``
     - ``7.5, 4.5``
   * - ``spectra_figsize``
     - Spectra figure size (w, h in)
     - ``text``
     - ``7.5, 4.375``

Declared Axis Units (``axis_units()``)
--------------------------------------

Both :class:`~pymorgan.oneD.dataset.Dataset1D` and :class:`~pymorgan.twoD.dataset.Dataset2D`
report the units of their axes through ``axis_units()``, which returns a
:class:`~pymorgan.helpers.DatasetUnits` record holding one entry per axis (``x``, ``y``, ``z``,
and ``t2``):

.. code-block:: python

   u = data.axis_units()            # 1D: x = probe, y = delay, z = signal
   u.x.unit                         # 'cm-1'
   u.x.label()                      # 'Wavenumber (cm$^{-1}$)'
   u.y.quantity                     # 'Delay'
   print(u)                         # readable summary of all axes
   u.as_dict()                      # plain nested dict for export

   v = d2.axis_units(display=True)  # 2D: converted to active display units & 10^3 factor
   v.t2.unit                        # 'ps'
