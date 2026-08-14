# PyMORGAN

<img src="src/pymorgan/gui/icons/pirate-hat.png" alt="" width="110" align="right">

**M**ultidimensional **O**ptical **R**esponse **G**raphical **A**nalysis i**N**terface in Python

[![PyPI - Version](https://img.shields.io/pypi/v/pymorgan.svg?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/pymorgan/)
[![GitHub Release](https://img.shields.io/github/v/release/RJFernandezTeran/PyMORGAN?logo=github&label=Release)](https://github.com/RJFernandezTeran/PyMORGAN/releases)
[![Python >= 3.12](https://img.shields.io/badge/python-%3E%3D3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![GUI: PyQt6](https://img.shields.io/badge/GUI-PyQt6-41CD52.svg?logo=qt&logoColor=white)](https://riverbankcomputing.com/software/pyqt/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-0A9EDC.svg?logo=pytest&logoColor=white)](tests/)
[![Kinetics: PyRATE-TA](https://img.shields.io/badge/kinetics-PyRATE--TA-8A2BE2.svg)](https://github.com/RJFernandezTeran/PyRATE-TA)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL_3.0-blue.svg)](LICENSE)
[![CodeFactor](https://www.codefactor.io/repository/github/RJFernandezTeran/PyMORGAN/badge)](https://www.codefactor.io/repository/github/RJFernandezTeran/PyMORGAN)
[![GitHub Stars](https://img.shields.io/github/stars/RJFernandezTeran/PyMORGAN?style=flat&logo=github)](https://github.com/RJFernandezTeran/PyMORGAN/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/RJFernandezTeran/PyMORGAN?style=flat&logo=github)](https://github.com/RJFernandezTeran/PyMORGAN/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/RJFernandezTeran/PyMORGAN?logo=github)](https://github.com/RJFernandezTeran/PyMORGAN/issues)
[![Downloads](https://img.shields.io/pepy/dt/pymorgan?label=downloads)](https://pepy.tech/project/pymorgan)

Plotting and analysis of ultrafast time-resolved spectroscopy data in **1D**
(pump–probe, transient IR, FLUPS, …) and **2D** (2D-IR, 2D-ES, 2D-VE, 2D-EV), plus
**steady-state** absorption / emission / excitation spectra.

---

## Relationship to PyRATE-TA

PyMORGAN is the **data** half of a two-project pair. Its sibling,
[PyRATE-TA](https://github.com/RJFernandezTeran/PyRATE-TA) (**R**ate **A**nalysis &
**T**arget-model **E**ngine for **T**ransient **A**bsorption), owns the **kinetic analysis**:
multi-exponential fitting, global analysis and target analysis with rate-matrix
(K-matrix) models, and the DAS / EAS / SAS spectra that come out of them.

**The dependency is one-way: PyRATE-TA imports PyMORGAN, never the reverse.**
PyMORGAN loads, processes and plots; PyRATE-TA fits. The seam is deliberately
narrow — `plot_species_spectra` takes plain arrays (`Sfit`, `Taus`, `TauErr`,
`isFixTau`, `modelType`) rather than a PyRATE-TA object, precisely so the arrow
never has to reverse. PyMORGAN's own `extract_kinetic` / `fit_kinetics` were
removed when PyRATE-TA took over analysis; use PyRATE-TA for anything kinetic.

```python
import pymorgan as pm
import pyrate_ta as pr

data = pm.load_1D("dataset1.pdat", data_type="PDAT")  # PyMORGAN loads
data.background_correct(tmin=-20, tmax=-5)            # PyMORGAN processes
fit  = pr.fit_global(data, n_components=3)            # PyRATE-TA fits
data.plot_species_spectra(*fit.as_species_args())     # PyMORGAN plots
```

PyMORGAN is usable entirely on its own; PyRATE-TA is an optional add-on and is not
a dependency of this package.

---

## Features

- **Unified 1-D pipeline** — `load → process → plot → analyse`, all hanging off
  a single `Dataset1D` object.
- **Unified 2-D pipeline** — the same stages for 2D-IR / 2D-EV via `Dataset2D`
  (pump–probe maps, overlap-aware diagonal, optional steady-state top panel).
- **Spectrometer & Pulse Shaper Calibration (`pymorgan.cal`)** — non-linear
  least-squares fitting of spectrograph dispersion curves to reference FTIR
  absorption spectra (e.g. Dioxane) across 10 hardware setup configurations (UniGE
  TRIR, UoS TRIR, UniGE TA, nsTA, UZH Lab 2, RAL LIFEtime, TRUVIS-II, etc.).
  Automatically parses dataset metadata (CWL, grating pitch) and exports calibrated
  frequency axes and pulse shaper mask files (`CalibratedPump.csv`, `SingleMask.txt`, `MultipleMask.txt`).
- **2-D interferometric processing** — apodisation (Box, Cos, Cos², Cos³, Hanning,
  Hamming, None), optional zero-padding, polynomial phase fitting (Constant /
  Linear / Quadratic / Cubic), and pump-correction; all exposed via the
  *Phasing/FT* sub-tab with live PH and TD diagnostic views.
- **Pluggable loader registry** — register new instrument formats with a
  decorator.
  - **1-D Time-Resolved**: `UniGE_fsTA` (with automatic non-transient `.dat` file filtering and `CalibratedProbe.csv` / `pix2lam.mat` detection), `UniGE_nsTA`, `HARPIA_TA`, `Helios_TA`, `MESS_TRIR`, `MESS_TRUVIS`, `PDAT`, `UniGE_FLUPSold`, `UniGE_FLUPSnew`, `UoS_IRpp`.
  - **2-D Spectroscopy**: `P2DAT`, `MESS_2DIR`, `UoS_2DIR`, `RAL_RAW`, `RAL_Proc`.
  - **Calibration Datasets**: UniGE fsTA / nsTA, UniGE TRIR Absorbance / ΔA, UniGE TRUVIS-II, UoS TRIR, UZH Lab 2, RAL LIFEtime.
  - **Steady-State**: FTIR / OPUS (`ftir`, `opus`), UV-Vis (`uvvis`), fluorimeter (`fluorimeter`), and generic delimited files (`csv`, `txt`, `xy`).
- **Steady-state spectra** — `Spectrum` / `SpectrumSeries` for absorption,
  emission and excitation, with automatic CSV/TXT delimiter detection.
- **Centralised, GUI-friendly settings** — style profile, font scale, axis-label
  convention, ΔA unit convention, spectral X-axis unit (nm / cm⁻¹ / eV / THz)
  with an optional complementary secondary axis, colourmap and time-axis scale,
  round-tripped to a commented `settings.toml`. The dataclass field is the single
  source of truth: serialisation, type coercion and the GUI widgets are all
  derived from it, and every setting is editable in the panel
  (*Common* / *1D* / *2D* / *GUI & Defaults* tabs).
- **2-D axis units** — pump/probe axes displayed in cm⁻¹ (default), in⁻¹, nm, THz
  or eV; axes reaching beyond 5000 (visible/UV maps) are shown divided by 1000
  with a `10³` prefix in the label. Limits and CLS/IvCLS/NLS overlays follow the
  conversion.
- **Sub-pixel spectral diffusion** — CLS/IvCLS centre lines are located on a
  spline-interpolated grid and refined by least-squares fitting a local model
  (quadratic/cubic/quartic, Gaussian or Lorentzian) to the measured points,
  reaching ~0.01 pixel on synthetic data instead of snapping to the pixel grid.
- **Shockwave subtraction** — removes correlated acoustic shockwave artefacts
  (common in ns-TA/ns-TRIR experiments) by averaging a user-selected pixel
  range into a 1-D reference trace and subtracting it from all channels; exposed
  via `ShockwaveSubtractionDialog` and the `subtract_shockwave` API.
- **Quick plots** — optional `pcolormesh` rendering of contour maps for fast
  interactive redraws (`quick_plots`), with the contour look restored for final
  figures.
- **Composable, `Dataset1D`-aware plotters** — per-call overrides, X/Y-label and
  colorbar toggles, and axis-handle reuse for multi-panel layouts and overlays.
- **matplotlib mathtext only** — the LaTeX text backend is never enabled.
- **Bundled PyQt6 GUI** (`PyMORGAN-GUI`) — load 1D/2D datasets and drive embedded
  contour maps, 2D interferometric phasing, and spectrograph/shaper calibration.
  Features a dedicated **Calibration Tab** with 4 interactive subplots:
  1. *Raw Measurements* (Pump Air, Pump Solvent, Single Mask shaded area, Multiple Mask).
  2. *Calculated Absorbance & Physical Baseline*.
  3. *Reference Fit Overlay* (Dual Y-axis zero-aligned via `mpl_axes_aligner`).
  4. *Calibration Fit* (Wavelength nm vs. Pixel dispersion curve).
  Includes global *Fit Controls* ("Do Fit", "Do baseline correction"), default `Wavelength (nm)` axis unit selection, and a standalone 1D Gaussian pump fit popup window with Matplotlib interactive zoom/pan toolbar and draggable legend.
- **Fast start-up** — the public API is lazy (PEP 562), so `import pymorgan`
  costs ~25 ms and the GUI shows its splash screen before the pipelines load.
  Heavy scientific imports (`scipy.ndimage`, `matplotlib.pyplot`) inside
  `helpers.py` and all dialog modules are deferred until first use. The GUI
  layout is loaded from a pre-compiled Python module (`main_window_ui.py`,
  generated by `pyuic6`) instead of parsing the 150 KB XML `.ui` file at
  every launch. **If `main_window.ui` is edited in Qt Designer, the compiled
  module is regenerated automatically on the next GUI start** — no manual
  step is needed.
- **Logging, not prints** — library messages go through the `pymorgan` logger
  (`pymorgan.configure_logging()`); set it to `DEBUG` to see the details behind a
  recovered failure.
- **uv-managed, ruff-linted, pytest-tested.**

## Supported Dataset Types & Loaders

PyMORGAN includes built-in loaders for a wide range of ultrafast time-resolved spectroscopy instruments, 2D spectroscopy formats, spectrograph calibration files, and steady-state spectra:

| Category | Type Identifier (`data_type`) | Format & Instrument Description |
| :--- | :--- | :--- |
| **1D Time-Resolved** | `UniGE_fsTA` | UniGE femtosecond Transient Absorption (`.dat` raw/processed TA datasets; auto-filters non-transient calibration files; checks `CalibratedProbe.csv` and `pix2lam.mat`) |
| | `UniGE_nsTA` | UniGE nanosecond Transient Absorption |
| | `HARPIA_TA` | Light Conversion HARPIA transient absorption spectrometer datasets |
| | `Helios_TA` | Ultrafast Systems Helios TA spectrometer datasets |
| | `MESS_TRIR` | MESS Transient IR spectrometer datasets (TRIR) |
| | `MESS_TRUVIS` | MESS Transient UV-Vis / TRUVIS-II spectrometer datasets |
| | `PDAT` | Standard PyMORGAN 1D binary/text transient dataset format |
| | `UniGE_FLUPSold` / `UniGE_FLUPSnew` | UniGE Fluorescence Upconversion Spectroscopy datasets |
| | `UoS_IRpp` | University of Sheffield IR pump–probe datasets |
| **2D Spectroscopy** | `P2DAT` | Standard PyMORGAN 2D binary/text dataset format |
| | `MESS_2DIR` | MESS 2D-IR spectrometer population-time maps |
| | `UoS_2DIR` | University of Sheffield 2D-IR population-time maps |
| | `RAL_RAW` | Rutherford Appleton Laboratory LIFEtime raw 2D-IR maps |
| | `RAL_Proc` | Rutherford Appleton Laboratory LIFEtime processed 2D-IR maps |
| **Calibration** | Spectrometer / Shaper | UniGE fsTA / nsTA, UniGE TRIR Absorbance / ΔA, UniGE TRUVIS-II, UoS TRIR, UZH Lab 2, RAL LIFEtime |
| **Steady-State** | `ftir`, `opus`, `uvvis`, `fluorimeter`, `csv`, `txt`, `xy` | FTIR absorption, Bruker OPUS files, UV-Vis, fluorimeter emission/excitation, and generic CSV/TXT/XY delimited matrices |

## Installation

### From PyPI (Recommended)

Install the latest release directly from [PyPI](https://pypi.org/project/pymorgan/):

```bash
pip install pymorgan
# or with uv:
uv pip install pymorgan

# After installation, register bundled fonts in matplotlib:
pymorgan-install-fonts
```

You can also run the GUI directly without installing using [`uvx`](https://docs.astral.sh/uv/concepts/tools/):
```bash
uvx --from pymorgan pymorgan-gui
```

### From Source (Development)

Clone the repository and install in editable mode with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/RJFernandezTeran/PyMORGAN.git
cd PyMORGAN

uv venv                       # create .venv (Python >= 3.12)

# Install the package (choose one):
uv pip install -e .           # core workflow (includes PyQt6 GUI)
# or:
uv pip install -e ".[dev]"    # + ruff and pytest

# After any install, run the font installer script:
uv run pymorgan-install-fonts # install bundled fonts into matplotlib
```

Console scripts installed with the package:

| Script | Purpose |
|--------|--------|
| `pymorgan-gui` | Launch the graphical interface |
| `pymorgan-install-fonts` | Install bundled fonts into matplotlib |
| `pymorgan-edit-gui` | Open the GUI layout in Qt Designer |
| `pymorgan-settings` | Standalone settings editor (no data loaded) |

## Quick start

```python
import pymorgan as pm

pm.load_settings("settings.toml")   # aesthetics: profile, labels, cmap, ...
pm.apply_style()

# --- Time-resolved (1-D): load -> process -> plot ---
data = pm.load_1D("scan.pdat", data_type="PDAT")
data.background_correct(tmin=-20, tmax=-5)
data.plot_contour(Zscale=20)
data.plot_spectra([0.5, 1, 5, 20, 100], doSmooth=1, x_axis_unit="nm", secondary_axis=True)
ax, t, Y = data.plot_kinetics([2132, 2218], plotStyle="-")   # Y is the extracted data

# --- 2-D (2D-IR / 2D-ES / 2D-VE / 2D-EV): one population-time (t2) map ---
d2 = pm.load_2D("scan.p2dat", data_type="P2DAT")
d2.plot_map(0.5)

# --- Spectrometer & Pulse Shaper Wavelength Calibration ---
from pymorgan.cal import load_experimental_spectrum, load_reference_spectrum, fit_wavelength_axis
exp = load_experimental_spectrum("pump_air.csv", cal_type_code=9)
ref = load_reference_spectrum("FTIR-Dioxane.csv")
res = fit_wavelength_axis(exp.detector_data[0], ref.spectral_axis, ref.absorbance, cal_type_code=9, cwl=2000.0)

# --- Steady-state (absorption / emission / excitation) ---
abs_sp = pm.load_spectrum("sample.csv", kind="absorption")  # delimiter auto-detected
abs_sp.plot()

pm.show_plots()
```

New instrument formats are added by registering a reader with
`@pm.register_loader(...)` (1-D), `@pm.register_map_loader(...)` (2-D) or
`@pm.register_spectrum_loader(...)` (steady-state); the matching
`pm.available_*_loaders()` helpers list what is recognised.

## Graphical interface

Launch the PyQt6 application with:

```bash
pymorgan-gui
```

It loads 1-D / 2-D datasets, shows sample diagnostics (noise, SNR, scan count,
probe resolution) and an embedded contour map driven by a plot-controls
panel. Kinetic and spectral cuts can be entered numerically or picked
interactively on the map; each opens in a new figure. The GUI includes dedicated
tabs for 2-D interferometric phasing/FT and spectrograph/shaper wavelength
calibration.

Additional interactive dialogs:
- **Chirp correction** (`Chirp…`) — guided automatic / step-function / manual
  chirp correction with per-pixel progress reporting.
- **Solvent subtraction** (`Subtract Solvent…`) — manual or automatic
  (per-pixel IRF-convoluted) solvent response removal.
- **Shockwave subtraction** (`Subtract Shockwave…`) — removes correlated
  acoustic artefacts by selecting a reference pixel range.
- **Make Movie** (`Make Movie…`) — exports a GIF or MP4 animation of 2-D maps
  stepping through all population times $t_2$.

The 1-D and 2-D dataset lists share one root folder and stay in sync; long
operations disable the interface while they run; and *View → Restore Default
Window Size* (`Ctrl+Shift+R`) brings the window back to its designed size. All
settings are editable under *View → Aesthetics / Settings…* and can be saved
back to `settings.toml`.

## Project layout

```
src/pymorgan/
  oneD/          load · process · plot · chirp · registry · dataset (Dataset1D)
  twoD/          load · process · plot · analyse · kubo_fit · progress · dataset (Dataset2D)
  steadyState/   Spectrum · SpectrumSeries · SpectrumKind · loader registry
  cal/           spectrometer & pulse shaper wavelength calibration (load, fit, models, export)
                 ref_spectra/ (reference FTIR / standard calibration spectra)
  gui/           PyQt6 application (pymorgan-gui)
    main_window.py / .ui   window assembly, wiring, menus (layout lives in the .ui)
    main_window_ui.py      pre-compiled Python UI (auto-regenerated from .ui on startup)
    tabs/                  per-tab mixins: browser · oneD · twoD · calibration
    mw_common.py           shared constants and helpers
    dialogs.py             spectral-diffusion, solvent-subtraction, detached plot window
    busy.py                re-entrancy guard for long operations
    plot_controls.py       plot-controls panel · settings_panel.py settings editor
    picker.py · canvas.py · cal_canvas.py · widgetplot.py · theme.py · widgets.py
    chirp_dialog.py · chirp_progress_dialog.py · shockwave_dialog.py
    kubo_dialog.py · movie_dialog.py · twoD_gaussian_dialog.py
    twoD_integral_dialog.py · twoD_subtraction_dialog.py
    icons/                 application icons and badges
  settings.py            Settings (TOML auto-merge, profiles, label/unit conventions, GUI field specs)
  settings.default.toml  canonical default settings template shipped with the package
  log.py                 logger factory and console configuration
  helpers.py             shared utilities (unit conversion, colourmaps, formatting)
  display.py             show_plots / close_plots / add_subplot_labels
  fonts.py               bundled-font installer (pymorgan-install-fonts)
  plot_styles/           matplotlib .mplstyle profiles (CMR, HLV, HLV_in, HLV_poster, JW)
examples/        runnable example scripts (1D pump-probe, composite 1D, 2D-IR, steady-state)
tests/           pytest suite (headless matplotlib, offscreen Qt)
scripts/         run_checks.py · run_changed_tests.py · bump_version.py
docs/            LaTeX manual (main + installation + oneD + twoD + steadyState + settings + calibration + extending)
```

## Testing

```bash
uv pip install -e ".[dev]"
uv run pymorgan-install-fonts        # install bundled fonts
uv run pytest                             # full test suite
uv run python scripts/run_changed_tests.py  # only the tests affected by your diff
uv run python scripts/run_checks.py       # quick end-to-end smoke check
```

The Qt-dependent suites (`test_gui`, `test_twoD_gui`, `test_cal_gui`,
`test_movie_dialog`, `test_picker`, `test_kubo`) need a Qt runtime; they run
offscreen and are the ones that catch GUI wiring regressions.

## Linting

[Ruff](https://docs.astral.sh/ruff/) handles linting and formatting
(configured in `pyproject.toml`):

```bash
uvx ruff@latest check src tests scripts examples
uvx ruff@latest format src tests
```

The version string in `src/pymorgan/__about__.py` follows `0.x.yymmdd.devN`
(PEP 440) and is maintained by `python scripts/bump_version.py` (`--minor` for a
design bump, `--check` to verify it was bumped today).

## Documentation

A LaTeX manual covering the data model, the settings, the full API and how to
extend the loader registry lives in `docs/` (`installation`, `oneD`, `twoD`,
`steadyState`, `settings`, `calibration`, `extending`). Build it with:

```bash
cd docs && pdflatex main.tex && pdflatex main.tex   # twice, for the table of contents
```

The kinetic-analysis side is documented separately, in
[PyRATE-TA](https://github.com/RJFernandezTeran/PyRATE-TA)'s own manual
(`docs/main.tex` there): the models and rate-matrix formalism, the fitting
engines, and the boundary between the two packages.


## Acknowledgements

- Development assisted by **Google Antigravity**, with all code, algorithms, and implementations manually verified and tested.
- PyMORGAN builds upon and modernizes the original MATLAB implementation from the now-deprecated [DataAnalysis](https://github.com/RJFernandezTeran/DataAnalysis) repository, written by Dr. Ricardo J. Fernández-Terán during his PhD and validated iteratively throughout the years.


## License
 
Released under the [GNU Affero General Public License v3.0 (AGPLv3)](LICENSE). © 2026 Dr. Ricardo J. Fernández-Terán
