# Calibration Reference Spectra Directory

This directory (`src/pymorgan/cal/ref_spectra/`) contains standard standard reference absorption/transmission spectra used for spectrograph wavelength and wavenumber calibration in PyMORGAN.

## Default Reference Spectra Included

- **Polystyrene Film (`FTIR-PS_*.csv`)**: Standard FTIR polystyrene film absorbance spectra across $4000-400\text{ cm}^{-1}$.
- **Dioxane / DCM (`FTIR-Dioxane_*.csv`, `FTIR-DCM_*.csv`)**: Solvent absorption spectra for mid-IR wavelength calibration.
- **Holmium Glass Filter (`UVVis-Holmium.csv`, `NIR-Holmium.csv`)**: Narrow absorption band reference standards for UV-Vis and NIR calibration.
- **Hoya Glass Filters (`Hoya-HY1.csv`, `Hoya-V11.csv`)**: Optical glass transmission/absorption standards.

## Adding Your Own Custom Reference Spectra

You can place your own custom reference spectra files directly in this directory:

1. **Format**: Save 2-column numerical CSV, TSV, or plain text files.
   - Column 1: Spectral Axis (Wavelength in `nm` or Wavenumber in `cm-1`)
   - Column 2: Absorbance or Transmittance intensity values
2. **Delimiters**: PyMORGAN automatically detects common separators:
   - Comma (`,`), Tab (`\t`), Semicolon (`;`), or Space (` `)
   - European decimal commas (e.g. `400,5; 0,123`) are converted automatically.
3. **Comments & Headers**: Lines starting with `#`, `%`, `//` or text header rows are skipped automatically.
4. **Git Version Control**: Custom spectra files added to this directory will **not** be tracked by git or pushed to GitHub, preserving your local reference library across repository updates.
