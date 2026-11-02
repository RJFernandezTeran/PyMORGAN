"""Synthetic-data generators for the test suite and the smoke script.

Deliberately free of PyMORGAN imports so the inputs are built independently of
the code under test. The generated pump-probe dataset is a clean bi-exponential
decay with a Gaussian spectral profile on a constant baseline, so background
correction and kinetic fitting have known ground truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_DEFAULT_DELAYS = np.array(
    [
        -20,
        -10,
        -5,
        -2,
        -1,
        -0.5,
        0.0,
        0.3,
        0.5,
        0.8,
        1,
        1.5,
        2,
        3,
        5,
        8,
        12,
        20,
        30,
        50,
        80,
        120,
        200,
        300,
        500,
    ],
    dtype=float,
)


def make_synthetic_pdat(
    path,
    *,
    taus=(12.0, 120.0),
    amps=(0.8, 0.2),
    centre=1955.0,
    width=18.0,
    baseline=0.05,
    noise=0.0,
    seed=0,
):
    """Write a synthetic ``.pdat`` file and return its ground-truth metadata."""
    delays = _DEFAULT_DELAYS
    probe = np.linspace(1900.0, 2010.0, 60)
    T, W = np.meshgrid(delays, probe, indexing="ij")
    spec = np.exp(-((W - centre) ** 2) / (2 * width**2))
    decay = np.zeros_like(T)
    pos = T > 0
    for A, tau in zip(amps, taus, strict=True):
        decay[pos] += A * np.exp(-T[pos] / tau)
    Z = decay * spec + baseline
    if noise:
        Z = Z + noise * np.random.RandomState(seed).randn(*Z.shape)

    lines = ["ps*cm^{-1}", "0," + ",".join(f"{w:.4f}" for w in probe)]
    for i, d in enumerate(delays):
        lines.append(f"{d}," + ",".join(f"{v:.6f}" for v in Z[i]))
    with open(str(path), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    return {
        "path": str(path),
        "delays": delays,
        "probe": probe,
        "taus": np.array(taus, dtype=float),
        "amps": np.array(amps, dtype=float),
        "centre": centre,
        "baseline": baseline,
    }


def make_synthetic_pdatn(pdat_path, *, sigma=0.5):
    """Write a sibling ``.pdatn`` (per-point std devs) for an existing ``.pdat``.

    The layout is identical to the ``.pdat``: a units header, the probe-axis
    header row and one delay-led row per delay, here filled with a constant
    standard deviation. Returns the ``.pdatn`` path.
    """
    import os

    with open(str(pdat_path)) as fh:
        header = fh.readline().rstrip()
        probe_row = fh.readline().rstrip()
        delay_first = [line.split(",", 1)[0] for line in fh if line.strip()]

    npix = probe_row.count(",")  # number of probe points
    lines = [header, probe_row]
    for d in delay_first:
        lines.append(f"{d}," + ",".join(f"{sigma:.6f}" for _ in range(npix)))

    npath = os.path.splitext(str(pdat_path))[0] + ".pdatn"
    with open(npath, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return npath


_SEP = {"tab": "\t", "comma": ",", "space": " "}


def make_spectrum_csv(
    path,
    *,
    delimiter="tab",
    header="Wavenumbers\tAbsorbance",
    centre=1960.0,
    width=25.0,
    n=200,
    x0=1900.0,
    x1=2010.0,
):
    """Write a synthetic two-column spectrum and return its metadata.

    ``delimiter`` is one of ``"tab"``, ``"comma"`` or ``"space"`` so the same
    spectrum can be emitted with different separators (to test auto-detection).
    """
    x = np.linspace(x0, x1, n)
    y = np.exp(-((x - centre) ** 2) / (2 * width**2))
    sep = _SEP.get(delimiter, delimiter)
    lines = [header] if header else []
    lines += [f"{a:.4f}{sep}{b:.6f}" for a, b in zip(x, y, strict=True)]
    with open(str(path), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return {"path": str(path), "x": x, "y": y, "centre": centre}


def make_synthetic_p2dat(
    path,
    *,
    pump=None,
    probe=None,
    delays=(0.25, 0.5, 1.0, 2.0, 5.0),
    t2_lifetime=3.0,
    centre=2040.0,
    width=8.0,
):
    """Write a synthetic P2DAT file (2D-IR-like) and return its ground truth.

    The cube is a diagonal bleach plus an off-diagonal cross peak that grows
    with t2; data is laid out exactly as ``read_P2DAT`` expects (header row of
    t2 values, then ``pump, probe, S(t2...)`` rows ordered pump-inner /
    probe-outer to match the Fortran-order reshape).
    """
    pump = np.linspace(2000.0, 2080.0, 24) if pump is None else np.asarray(pump, float)
    probe = np.linspace(2000.0, 2080.0, 24) if probe is None else np.asarray(probe, float)
    delays = np.asarray(delays, float)

    P, Q = np.meshgrid(pump, probe, indexing="ij")  # [Npump, Nprobe]
    Z = np.zeros((len(pump), len(probe), len(delays)))
    for k, t in enumerate(delays):
        diag = np.exp(-((P - centre) ** 2 + (Q - centre) ** 2) / (2 * width**2))
        cross = np.exp(-((P - centre) ** 2 + (Q - (centre - 20)) ** 2) / (2 * width**2))
        Z[:, :, k] = np.exp(-t / t2_lifetime) * diag - 0.5 * (1 - np.exp(-t / t2_lifetime)) * cross

    lines = [",".join(["0", "0"] + [f"{d}" for d in delays])]
    for j in range(len(probe)):  # probe outer
        for i in range(len(pump)):  # pump inner (Fortran order)
            row = [f"{pump[i]:.4f}", f"{probe[j]:.4f}"]
            row += [f"{Z[i, j, k]:.6f}" for k in range(len(delays))]
            lines.append(",".join(row))
    with open(str(path), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    return {"path": str(path), "pump": pump, "probe": probe, "delays": delays, "Z": Z}


_MESS_SLOWMOD_LABELS = (
    "Shutter",
    "Pol(pump)",
    "Pol(probe)",
    "Spectrometer",
    "Waveplate motor(s)",
    "TOPAS scan",
)


def make_synthetic_mess(
    folder,
    *,
    kind="TRIR",
    nscans=4,
    n_slowmod=1,
    n_spectra=1,
    npix=24,
    seed=0,
):
    """Write a synthetic MESS dataset folder (TRIR or TRUVIS) and return metadata.

    Mirrors the real acquisition layout: a folder whose files are prefixed with
    its own name, holding ``_delays.csv``, the probe axis
    (``_wavenumbers.csv`` for ``kind="TRIR"`` in cm^-1, ``_wavelengths.csv`` for
    ``kind="TRUVIS"`` in nm), ``_Nspectra.csv``, ``_slowModulation.csv`` and, for
    every ``(sp, sm)`` state, the averaged ``_signal`` / ``_signal_noise`` files
    plus a ``temp/`` folder of per-scan ``_signal`` / ``_signal_noise`` files.

    The averaged signal/noise are the **inverse-variance weighted** combination
    of the per-scan temp files (weight ``1/noise**2``), so combining every scan
    reproduces the averaged data -- the invariant exercised by the recalc tests.

    Delays are written in femtoseconds (the reader converts to ps). Returns a
    metadata dict with the folder, dataset name, ps delays, probe axis, the
    state counts, ``nscans`` and the per-(sp,sm) averaged signal/noise.
    """
    import os

    folder = str(folder)
    os.makedirs(folder, exist_ok=True)
    tempdir = os.path.join(folder, "temp")
    os.makedirs(tempdir, exist_ok=True)
    name = os.path.basename(os.path.normpath(folder))
    stem = os.path.join(folder, name)
    rng = np.random.RandomState(seed)

    # Delays in fs (reader divides by 1000 -> ps); a few negatives then positives.
    delays_ps = _DEFAULT_DELAYS
    delays_fs = delays_ps * 1000.0
    ndel = delays_fs.size

    if kind == "TRUVIS":
        probe = np.linspace(400.0, 700.0, npix)
        probe_suffix = "wavelengths"
    else:
        probe = np.linspace(1900.0, 2010.0, npix)
        probe_suffix = "wavenumbers"

    def _write_col(path, arr):
        with open(path, "w") as fh:
            fh.write("\n".join(f"{v:.6f}" for v in arr) + "\n")

    def _write_2d(path, M):
        lines = [",".join(f"{v:.6f}" for v in row) for row in M]
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")

    _write_col(f"{stem}_delays.csv", delays_fs)
    _write_col(f"{stem}_{probe_suffix}.csv", probe)
    with open(f"{stem}_Nspectra.csv", "w") as fh:
        fh.write(f"{n_spectra}\n")
    with open(f"{stem}_Ndatastates.csv", "w") as fh:
        fh.write("4\n")
    with open(f"{stem}_slowModulation.csv", "w") as fh:
        fh.write(
            f"{n_slowmod}\n" + "\n".join(f"{lbl}: [NaN]" for lbl in _MESS_SLOWMOD_LABELS) + "\n"
        )

    T, W = np.meshgrid(delays_ps, probe, indexing="ij")

    averaged = {}  # (sp, sm) -> (signal, noise)
    for sp in range(n_spectra):
        for sm in range(n_slowmod):
            # A distinct but smooth base signal per state.
            centre = probe.mean() + 5.0 * sm - 3.0 * sp
            width = (probe[-1] - probe[0]) / 6.0
            spec = np.exp(-((W - centre) ** 2) / (2 * width**2))
            base = np.where(T > 0, np.exp(-T / 60.0), 0.0) * spec * (1.0 + 0.1 * sm)

            scans_sig = []
            scans_noise = []
            for k in range(nscans):
                sigma = 0.4 + 0.2 * k  # noise level varies per scan -> weighting matters
                s_k = base + sigma * rng.randn(ndel, npix) * 0.05
                n_k = np.full((ndel, npix), sigma)
                scans_sig.append(s_k)
                scans_noise.append(n_k)
                end = f"_sp{sp}_sm{sm}_du0_{k}.csv"
                _write_2d(os.path.join(tempdir, f"{name}_signal{end}"), s_k)
                _write_2d(os.path.join(tempdir, f"{name}_signal_noise{end}"), n_k)

            S = np.stack(scans_sig, axis=-1)
            Nz = np.stack(scans_noise, axis=-1)
            w = 1.0 / Nz**2
            avg = np.sum(S * w, axis=-1) / np.sum(w, axis=-1)
            comb = np.sqrt(1.0 / np.sum(w, axis=-1))
            end = f"_sp{sp}_sm{sm}_du0.csv"
            _write_2d(f"{stem}_signal{end}", avg)
            _write_2d(f"{stem}_signal_noise{end}", comb)
            averaged[(sp, sm)] = (avg, comb)

    return {
        "folder": folder,
        "name": name,
        "kind": kind,
        "delays_ps": delays_ps,
        "probe": probe,
        "n_spectra": n_spectra,
        "n_slowmod": n_slowmod,
        "nscans": nscans,
        "averaged": averaged,
    }


def make_synthetic_helios_dataset(
    folder,
    main_name: str = "sample_data",
    *,
    nscans: int = 3,
    ndel: int = 15,
    npix: int = 20,
    seed: int = 42,
    create_aux_files: bool = True,
):
    """Write a synthetic Helios TA dataset directory structure."""
    import os
    from pathlib import Path

    folder_path = Path(folder)
    os.makedirs(str(folder_path), exist_ok=True)
    rng = np.random.RandomState(seed)

    delays = np.linspace(-2.0, 50.0, ndel)
    probe = np.linspace(350.0, 750.0, npix)

    T, W = np.meshgrid(delays, probe, indexing="xy")
    base_sig = np.where(T > 0, np.exp(-T / 10.0), 0.0) * np.exp(
        -((W - 500.0) ** 2) / (2 * 50.0**2)
    )

    def _write_helios_csv(target_path, sig_matrix):
        lines = ["0.0," + ",".join(f"{d:.6f}" for d in delays)]
        for i, p_val in enumerate(probe):
            row = [f"{p_val:.6f}"] + [f"{sig_matrix[i, j]:.6f}" for j in range(ndel)]
            lines.append(",".join(row))
        lines.append("Comments: Synthetic dataset")
        lines.append(f"Number of scans: {nscans}")
        lines.append("Z axis title: dA")
        target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    main_csv = folder_path / f"{main_name}.csv"
    _write_helios_csv(main_csv, base_sig)

    scans = []
    for s in range(1, nscans + 1):
        scan_sig = base_sig + rng.randn(npix, ndel) * 0.001
        scan_file = folder_path / f"{main_name}_scan{s}.csv"
        _write_helios_csv(scan_file, scan_sig)
        scans.append(scan_sig)

    if create_aux_files:
        _write_helios_csv(folder_path / f"{main_name}-bg.csv", base_sig)
        _write_helios_csv(folder_path / f"{main_name}-chirp.csv", base_sig)
        (folder_path / f"{main_name}-CORRECTED.pdat").write_text(
            "dummy pdat", encoding="utf-8"
        )
        (folder_path / f"{main_name}.ufs").write_bytes(b"\x00" * 32)

    return {
        "folder": folder_path,
        "main_file": main_csv,
        "main_name": main_name,
        "delays": delays,
        "probe": probe,
        "nscans": nscans,
        "base_sig": base_sig,
    }


def make_synthetic_harpia_dataset(file_path, *, nscans: int = 2, seed: int = 42):
    """Write a synthetic HARPIA-TA .dat file for unit testing."""
    from pathlib import Path

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    probe = np.linspace(300.0, 500.0, 10)
    delays = np.array([-2.0, 0.0, 1.0, 5.0, 10.0])

    lines = [
        "Pump-probe, Not referenced, 1000 Hz",
        "Background header",
        "Measurement header",
        "Wavelength:\t" + "\t".join(f"{p:.4f}" for p in probe),
    ]

    for s in range(1, nscans + 1):
        b_sp = "\t".join(f"{rng.uniform(0.1, 0.2):.6f}" for _ in probe)
        b_rp = "\t".join(f"{rng.uniform(0.1, 0.2):.6f}" for _ in probe)
        b_su = "\t".join(f"{rng.uniform(0.1, 0.2):.6f}" for _ in probe)
        b_ru = "\t".join(f"{rng.uniform(0.1, 0.2):.6f}" for _ in probe)
        lines.append(f"Background, scan {s}, Delay {delays[0]}")
        lines.append("Pump=0.0")
        lines.extend([b_sp, b_rp, b_su, b_ru])

        for d in delays:
            m_sp = "\t".join(f"{rng.uniform(1.0, 2.0):.6f}" for _ in probe)
            m_rp = "\t".join(f"{rng.uniform(1.0, 2.0):.6f}" for _ in probe)
            m_su = "\t".join(f"{rng.uniform(1.0, 2.0):.6f}" for _ in probe)
            m_ru = "\t".join(f"{rng.uniform(1.0, 2.0):.6f}" for _ in probe)
            lines.append(f"Measurement, scan {s}, Delay {d:.4f}")
            lines.append("Pump=1.0")
            lines.extend([m_sp, m_rp, m_su, m_ru])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"path": path, "delays": delays, "probe": probe, "nscans": nscans}


def make_synthetic_unige_fsta(
    file_path,
    *,
    nscans: int = 2,
    npixels: int = 60,
    ndelays: int = 5,
    seed: int = 42,
    create_mat: bool = True,
):
    """Write a synthetic UniGE fsTA .dat dataset file and optional pix2lam.mat."""
    from pathlib import Path

    import scipy.io as sio

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    delays_s = np.linspace(-2e-12, 10e-12, ndelays)  # seconds

    header = [
        "% clear-reads = 4",
        f"% number-of-pixels = {npixels}",
        f"% time-delays = {ndelays}",
        f"% scans = {nscans}",
        "% program-name = step-scan-ta",
    ]

    data_lines = []
    for _ in range(nscans):
        for d in delays_s:
            row_tokens = [f"{d:.6e}", "200"]
            for _ in range(npixels):
                sig_uod = float(rng.uniform(-50000.0, 50000.0))
                err_uod = float(rng.uniform(1000.0, 5000.0))
                row_tokens.extend([f"{sig_uod:.2f}", f"{err_uod:.2f}"])
            data_lines.append(" ".join(row_tokens))

    content = "\n".join(header + data_lines) + "\n"
    path.write_text(content, encoding="utf-8")

    probe_mat = None
    if create_mat:
        lam = np.linspace(350.0, 750.0, npixels)
        mat_path = path.parent / "pix2lam.mat"
        sio.savemat(mat_path, {"lam": lam})
        probe_mat = lam

    return {
        "path": path,
        "delays": delays_s * 1e12,
        "nscans": nscans,
        "npixels": npixels,
        "probe": probe_mat,
    }


def make_synthetic_unige_cal(folder_path, *, npixels: int = 60, seed: int = 42):
    """Write synthetic UniGE calibration sample and blank .dat files."""
    from pathlib import Path

    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    sample_path = folder / "synthetic_sample_WLCORR.dat"
    blank_path = folder / "synthetic_BLANK_WLCORR.dat"

    s_lines = []
    b_lines = []
    for p in range(npixels):
        s_intensity = 3000.0 + 500.0 * np.sin(p / 5.0) + float(rng.uniform(-10.0, 10.0))
        b_intensity = 3500.0 + float(rng.uniform(-10.0, 10.0))
        s_err = float(rng.uniform(1.0, 5.0))
        b_err = float(rng.uniform(1.0, 5.0))

        s_lines.append(f"{p}\t200\t{s_intensity:.2f}\t{s_err:.2f}\t3000.00\t5.00")
        b_lines.append(f"{p}\t200\t{b_intensity:.2f}\t{b_err:.2f}\t3000.00\t5.00")

    sample_path.write_text("\n".join(s_lines) + "\n", encoding="utf-8")
    blank_path.write_text("\n".join(b_lines) + "\n", encoding="utf-8")

    return {
        "sample_path": sample_path,
        "blank_path": blank_path,
        "npixels": npixels,
    }


def make_synthetic_uos_irpp(folder_path, *, n_detectors: int = 2, n_scans: int = 2, n_delays: int = 10, seed: int = 42):
    """Write synthetic University of Sheffield (UoS) TRIR dataset directory."""
    from pathlib import Path

    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    base_name = "UVIR_20260807_120000"
    twod_path = folder / f"{base_name}.2D"
    dt_path = folder / f"{base_name}.DT"
    lg_path = folder / f"{base_name}.LG"
    cal_path = folder / "CalibratedProbe.csv"

    single_delays = np.linspace(-10.0, 100.0, n_delays)
    all_delays = np.tile(single_delays, n_scans)
    np.savetxt(dt_path, all_delays, fmt="%.5f")

    pixels_per_det = 96
    total_pixels = pixels_per_det * n_detectors
    data_rows = []
    for _ in range(n_scans):
        for i in range(n_delays):
            row = rng.normal(loc=0.0, scale=0.05, size=total_pixels)
            if single_delays[i] > 0:
                row[:pixels_per_det] += np.exp(-single_delays[i] / 20.0) * 0.5
                if n_detectors > 1:
                    row[pixels_per_det:] += np.exp(-single_delays[i] / 50.0) * 0.3
            data_rows.append(row)

    data_mat = np.array(data_rows)
    np.savetxt(twod_path, data_mat, fmt="%.6f")

    lg_text = (
        "*************Detector Settings&Acquisition:     \n"
        "Grating grooves/mm (0>120,1>100,2>50):     1\n"
        "Central Wavelength nm:     5120.000000\n"
        "Grating grooves/mm:     2\n"
        "Central Wavelength nm:     4999.000000\n"
    )
    lg_path.write_text(lg_text, encoding="utf-8")

    probe_p1 = np.linspace(1800.0, 2100.0, pixels_per_det)
    if n_detectors > 1:
        probe_p2 = np.linspace(1900.0, 2200.0, pixels_per_det)
        probe_full = np.concatenate([probe_p1, probe_p2])
    else:
        probe_full = probe_p1

    np.savetxt(cal_path, probe_full, delimiter=",", fmt="%.4f")

    return {
        "folder": folder,
        "twod_path": twod_path,
        "dt_path": dt_path,
        "n_detectors": n_detectors,
        "n_scans": n_scans,
        "n_delays": n_delays,
    }


def make_synthetic_unige_nsta(path, *, n_delays=10, npixels=20, zero_counts=False):
    """Write a synthetic UniGE nsTA .dat file."""
    delays_sec = np.linspace(-5e-7, 1e-4, n_delays)
    lines = [
        "% program = ps-ta 3.0.1, 19.01.2015",
        "% file = synthetic.dat",
        "% format = dl-mach-xo-delay-1 Delay, pixel, TA signal, rms, error, n samples",
    ]
    for i, t in enumerate(delays_sec):
        counts = 1000 if not (zero_counts and i == 0) else 0
        for p in range(npixels):
            sig = -50.0 * np.exp(-max(0.0, t) / 1e-5) if t > 0 else 0.0
            rms = 10.0
            err = 0.5
            lines.append(f"{t:.6e}\t{p}\t{sig:.6e}\t{rms:.6e}\t{err:.6e}\t{counts}")

    path_obj = Path(path)
    path_obj.write_text("\n".join(lines), encoding="utf-8")
    return {"path": path_obj, "delays_sec": delays_sec, "npixels": npixels}


def make_synthetic_pymess_pp(
    file_path,
    *,
    n_scans: int = 3,
    n_delays: int = 20,
    n_pixels: int = 32,
    probe_type: str = "wavenumber",  # "wavenumber", "wavelength", or "uncalibrated"
    measurement_type: str | None = "pump_probe",
    with_noise: bool = True,
    with_individual_scans: bool = True,
    with_intensity: bool = False,
    trailing_nan_delays: int = 0,
    seed: int = 42,
):
    """Write a synthetic PyMESS pump-probe (.h5) dataset file."""
    import h5py

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    delays_fs = np.linspace(-5000.0, 50000.0, n_delays)  # in fs (-5 ps to 50 ps)

    if probe_type == "wavenumber":
        wavelengths = np.linspace(1900.0, 2100.0, n_pixels)  # cm-1
    elif probe_type == "wavelength":
        wavelengths = np.linspace(400.0, 700.0, n_pixels)  # nm
    else:
        wavelengths = None

    # Bi-exponential decay signal in mOD
    T, W = np.meshgrid(delays_fs / 1000.0, np.arange(n_pixels), indexing="ij")
    decay = np.zeros_like(T)
    pos = T > 0
    decay[pos] = -15.0 * np.exp(-T[pos] / 5.0) + 5.0 * np.exp(-T[pos] / 25.0)
    spec = np.exp(-((W - n_pixels / 2) ** 2) / (2 * (n_pixels / 6) ** 2))
    base_signal = decay * spec

    if with_individual_scans and n_scans > 0:
        indiv = np.zeros((n_scans, n_delays, n_pixels), dtype=np.float64)
        for s in range(n_scans):
            noise_pass = rng.normal(loc=0.0, scale=0.1, size=(n_delays, n_pixels))
            indiv[s] = base_signal + noise_pass
        delta_A = np.mean(indiv, axis=0)
    else:
        indiv = None
        delta_A = base_signal.copy()

    if with_noise:
        delta_A_noise = rng.uniform(0.05, 0.15, size=(n_delays, n_pixels))
    else:
        delta_A_noise = None

    if trailing_nan_delays > 0:
        cut = n_delays - trailing_nan_delays
        delta_A[cut:, :] = np.nan
        if indiv is not None:
            indiv[:, cut:, :] = np.nan
        if delta_A_noise is not None:
            delta_A_noise[cut:, :] = np.nan

    with h5py.File(path, "w") as f:
        data_grp = f.create_group("data")
        data_grp.create_dataset("delta_A", data=delta_A)
        data_grp.create_dataset("delays_fs", data=delays_fs)
        if wavelengths is not None:
            data_grp.create_dataset("wavelengths", data=wavelengths)
        if indiv is not None:
            data_grp.create_dataset("individual_scans", data=indiv)
        if delta_A_noise is not None:
            data_grp.create_dataset("delta_A_noise", data=delta_A_noise)
        if with_intensity:
            intensity = rng.uniform(1000.0, 5000.0, size=(n_delays, n_pixels))
            data_grp.create_dataset("intensity", data=intensity)

        meta_grp = f.create_group("metadata")
        if measurement_type is not None:
            meta_grp.attrs["measurement_type"] = measurement_type
        meta_grp.attrs["sample_id"] = "Synthetic Sample"
        meta_grp.attrs["solvent"] = "DCM"
        meta_grp.attrs["pump_wl_nm"] = "532.0"
        meta_grp.attrs["pump_power_mw"] = "1.5"
        meta_grp.attrs["pump_polarization"] = "Magic Angle"
        meta_grp.attrs["probe_polarization"] = "Horizontal"
        meta_grp.attrs["comments"] = "Synthetic PyMESS test dataset"

        cal_grp = f.create_group("calibration")
        cal_grp.create_dataset("b_matrix", data=np.eye(n_pixels))

    return {
        "path": path,
        "n_scans": n_scans,
        "n_delays": n_delays,
        "n_pixels": n_pixels,
        "delays_ps": delays_fs / 1000.0,
        "trailing_nan_delays": trailing_nan_delays,
    }


def _czerny_turner_wavelengths_nm(
    n_pix: int,
    cwl_nm: float,
    grooves_per_mm: float = 75.0,
    focal_mm: float = 150.0,
    dev_angle_deg: float = 49.32,
    pitch_um: float = 40.0,
    binning: int = 1,
) -> np.ndarray:
    """Independent Czerny-Turner wavelength calculation for synthetic fixtures."""
    d_nm = 1.0e6 / grooves_per_mm
    gamma = np.deg2rad(0.5 * dev_angle_deg)
    theta = np.arcsin(cwl_nm / (2.0 * d_nm * np.cos(gamma)))
    alpha = theta - gamma
    beta0 = theta + gamma
    pix = np.arange(n_pix, dtype=float)
    p_c = 0.5 * float(n_pix - 1)
    dx_mm = (pitch_um / 1000.0) * float(binning)
    return d_nm * (np.sin(alpha) + np.sin(beta0 + np.arctan((pix - p_c) * dx_mm / focal_mm)))


def _write_qc_scan_file(
    scan_path: Path,
    axis0: np.ndarray,
    signal_od: np.ndarray,
    trailing_row_od: np.ndarray | None = None,
) -> None:
    """Write a synthetic QuickControl ``.scan`` ASCII matrix with trailing zero-axis row."""
    n_pix = signal_od.shape[1]
    if trailing_row_od is None:
        trailing_row_od = np.zeros(n_pix, dtype=float)
    lines: list[str] = []
    for i in range(signal_od.shape[0]):
        vals = "\t".join(f"{v:.8e}" for v in signal_od[i])
        lines.append(f"{axis0[i]:.6f}\t{vals}")
    tail_vals = "\t".join(f"{v:.8e}" for v in trailing_row_od)
    lines.append(f"{0.0:.6f}\t{tail_vals}")
    scan_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_synthetic_quickcontrol_1d(
    folder: str | Path,
    prefix: str = "synthetic_run#01",
    *,
    n_delays: int = 15,
    n_pixels: int = 24,
    n_scans: int = 3,
    polarised: bool = True,
    cwl_nm: float = 4800.0,
    advanced_cal: bool = True,
    write_scans: bool = True,
    seed: int = 7,
) -> dict:
    """Write a synthetic PhaseTech QuickControl 1D dataset and return ground truth."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    delays_fs = np.linspace(-500.0, 5000.0, n_delays)
    wl_nm = _czerny_turner_wavelengths_nm(n_pixels, cwl_nm=cwl_nm)
    probe_expected = (1.0e7 / wl_nm) if cwl_nm > 1200.0 else wl_nm

    t_ps = delays_fs / 1000.0
    decay = np.where(t_ps >= 0.0, np.exp(-np.maximum(t_ps, 0.0) / 2.5), 0.0)
    p_mid = float(np.mean(probe_expected))
    p_span = max(1.0, 0.15 * float(np.ptp(probe_expected)))
    prof = -np.exp(-(((probe_expected - p_mid) / p_span) ** 2))
    base_mOD = 6.0 * np.outer(decay, prof)  # mOD

    info_lines = [
        "Experiment Type\t1D",
        "Comment\tSynthetic 1D pump-probe test run",
        "Date and Time\t2026-01-15 10:00:00",
        "Pump Type\tMid-IR Pulse Shaper",
        (
            "Detection Mode\tParallel, Perpendicular, & Reference"
            if polarised
            else "Detection Mode\tProbe & Reference"
        ),
        "Delays\t" + ",".join(f"{d:.3f}" for d in delays_fs),
        "Delay Units\tfs",
        f"1D Average\t{n_scans}",
        f"MONO1 Wavelength\t{cwl_nm:.6f}",
        'MONO1 Grating\t1, "1: 75 g/mm"',
        f"Detector Pixels\t{n_pixels}x{n_pixels}",
        "Detector Binning\t1",
        "Flip Detector Columns?\tFALSE",
        "QC Version\t4.0.0",
    ]
    if advanced_cal:
        info_lines.extend(
            [
                "MONO Calibration Type\tADVANCED",
                "MONO Focal Length\t150.000000",
                "MONO Deviation Angle\t49.320000",
                "MONO Pixel Pitch\t40.000000",
                "MONO Grating Grooves\t75.000000",
            ]
        )
    else:
        info_lines.append("MONO Calibration Type\tSIMPLE")

    info_path = folder / f"{prefix}.info"
    info_path.write_text("\n".join(info_lines) + "\n", encoding="utf-8")

    spec_vals = 1000.0 + 200.0 * np.exp(-(((probe_expected - p_mid) / (2.0 * p_span)) ** 2))
    (folder / f"{prefix}.spec").write_text(
        "\n".join(f"{v:.4f}" for v in spec_vals) + "\n", encoding="utf-8"
    )

    par_scans_mOD = []
    per_scans_mOD = []
    if write_scans and n_scans > 0:
        for s in range(1, n_scans + 1):
            jitter = rng.normal(0.0, 0.08, size=(n_delays, n_pixels))
            if polarised:
                par_mOD = 1.8 * base_mOD + jitter
                per_mOD = 0.6 * base_mOD - 0.5 * jitter
                par_scans_mOD.append(par_mOD)
                per_scans_mOD.append(per_mOD)
                _write_qc_scan_file(
                    folder / f"{prefix}_PAR#{s:03d}.scan", delays_fs, par_mOD / 1000.0
                )
                _write_qc_scan_file(
                    folder / f"{prefix}_PER#{s:03d}.scan", delays_fs, per_mOD / 1000.0
                )
            else:
                sig_mOD = base_mOD + jitter
                par_scans_mOD.append(sig_mOD)
                _write_qc_scan_file(
                    folder / f"{prefix}#{s:03d}.scan", delays_fs, sig_mOD / 1000.0
                )

    avg_par = np.mean(par_scans_mOD, axis=0) if par_scans_mOD else None
    avg_per = np.mean(per_scans_mOD, axis=0) if per_scans_mOD else None
    avg_ma = ((avg_par + 2.0 * avg_per) / 3.0) if (avg_par is not None and avg_per is not None) else None

    p_order = np.argsort(probe_expected)
    probe_out = probe_expected[p_order]
    spec_out = spec_vals[p_order]
    if avg_par is not None:
        avg_par = avg_par[:, p_order]
    if avg_per is not None:
        avg_per = avg_per[:, p_order]
    if avg_ma is not None:
        avg_ma = avg_ma[:, p_order]

    return {
        "info_path": info_path,
        "prefix": prefix,
        "delays_ps": delays_fs / 1000.0,
        "probe": probe_out,
        "avg_par_mOD": avg_par,
        "avg_per_mOD": avg_per,
        "avg_ma_mOD": avg_ma,
        "spec": spec_out,
    }


def make_synthetic_quickcontrol_2d(
    folder: str | Path,
    prefix: str = "synthetic_map#02",
    *,
    domain: str = "Time",
    n_t1: int = 41,
    dt1_fs: float = 25.0,
    n_pixels: int = 20,
    t2_fs: tuple[float, ...] = (200.0, 1000.0, 3000.0),
    n_written_t2: int | None = None,
    n_scans: int = 2,
    polarised: bool = True,
    w0_scanned_cm1: float = 1980.0,
    w0_fixed_cm1: float = 0.0,
    peak_cm1: float = 2080.0,
    cwl_nm: float = 4800.0,
    write_wts: bool = True,
    seed: int = 11,
) -> dict:
    """Write a synthetic PhaseTech QuickControl 2D dataset and return ground truth."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    wl_nm = _czerny_turner_wavelengths_nm(n_pixels, cwl_nm=cwl_nm)
    probe_cm1 = 1.0e7 / wl_nm
    t2_arr_fs = np.asarray(t2_fs, dtype=float)
    n_t2_total = len(t2_arr_fs)
    n_t2_disk = n_t2_total if n_written_t2 is None else min(n_t2_total, int(n_written_t2))

    is_freq = domain.strip().lower().startswith("freq")
    w0_eff = w0_scanned_cm1 - w0_fixed_cm1

    if is_freq:
        axis0 = np.linspace(2010.0, 2150.0, n_t1)
        pump_prof = np.exp(-(((axis0 - peak_cm1) / 12.0) ** 2))
    else:
        axis0 = np.arange(n_t1, dtype=float) * dt1_fs
        c_cm_fs = 2.99792458e-5
        pump_prof = np.cos(2.0 * np.pi * (peak_cm1 - w0_eff) * c_cm_fs * axis0) * np.exp(
            -axis0 / 450.0
        )

    probe_prof = -np.exp(-(((probe_cm1 - peak_cm1) / 14.0) ** 2)) + 0.65 * np.exp(
        -(((probe_cm1 - (peak_cm1 - 22.0)) / 14.0) ** 2)
    )
    base_2d_mOD = 8.0 * np.outer(pump_prof, probe_prof)

    info_lines = [
        "Experiment Type\t2D",
        f"Domain\t{'Frequency' if is_freq else 'Time'}",
        "Comment\tSynthetic 2D test run",
        "Date and Time\t2026-01-15 11:30:00",
        "Pump Type\tMid-IR Pulse Shaper",
        (
            "Detection Mode\tParallel, Perpendicular, & Reference"
            if polarised
            else "Detection Mode\tProbe & Reference"
        ),
        f"Step Size (fs)\t{dt1_fs:.6f}",
        f"Final Delay (fs)\t{(n_t1 - 1) * dt1_fs:.6f}",
        f"Rotating Frame (Scanned)\t{w0_scanned_cm1:.6f}",
        f"Rotating Frame (Fixed)\t{w0_fixed_cm1:.6f}",
        "Frames\t4",
        "Waiting Time Delays\t" + ",".join(f"{v:.3f}" for v in t2_arr_fs),
        "Waiting Time Delay Units\tfs",
        f"2D Average\t{n_scans}",
        f"MONO1 Wavelength\t{cwl_nm:.6f}",
        'MONO1 Grating\t1, "1: 75 g/mm"',
        "MONO Calibration Type\tADVANCED",
        "MONO Focal Length\t150.000000",
        "MONO Deviation Angle\t49.320000",
        "MONO Pixel Pitch\t40.000000",
        "MONO Grating Grooves\t75.000000",
        f"Detector Pixels\t{n_pixels}x{n_pixels}",
        "Detector Binning\t1",
        "Flip Detector Columns?\tFALSE",
        "QC Version\t4.0.0",
    ]

    info_path = folder / f"{prefix}.info"
    info_path.write_text("\n".join(info_lines) + "\n", encoding="utf-8")

    if write_wts:
        (folder / f"{prefix}.wts").write_text(
            "\n".join(f"{v:.3f}" for v in t2_arr_fs) + "\n", encoding="utf-8"
        )

    par_cube = np.zeros((n_t1, n_pixels, n_t2_disk), dtype=float)
    per_cube = np.zeros((n_t1, n_pixels, n_t2_disk), dtype=float)

    for m in range(n_t2_disk):
        decay_m = float(np.exp(-t2_arr_fs[m] / 4000.0))
        par_list = []
        per_list = []
        for s in range(1, n_scans + 1):
            jit = rng.normal(0.0, 0.04, size=(n_t1, n_pixels))
            stat_tail = 0.002 * probe_prof * decay_m
            if polarised:
                p_mOD = 1.8 * decay_m * base_2d_mOD + jit
                s_mOD = 0.6 * decay_m * base_2d_mOD - 0.5 * jit
                par_list.append(p_mOD)
                per_list.append(s_mOD)
                _write_qc_scan_file(
                    folder / f"{prefix}_T{m + 1:02d}_PAR#{s:03d}.scan",
                    axis0,
                    p_mOD / 1000.0,
                    trailing_row_od=stat_tail,
                )
                _write_qc_scan_file(
                    folder / f"{prefix}_T{m + 1:02d}_PER#{s:03d}.scan",
                    axis0,
                    s_mOD / 1000.0,
                    trailing_row_od=stat_tail * 0.5,
                )
            else:
                u_mOD = decay_m * base_2d_mOD + jit
                par_list.append(u_mOD)
                _write_qc_scan_file(
                    folder / f"{prefix}_T{m + 1:02d}#{s:03d}.scan",
                    axis0,
                    u_mOD / 1000.0,
                    trailing_row_od=stat_tail,
                )
        par_cube[:, :, m] = np.mean(par_list, axis=0)
        if polarised:
            per_cube[:, :, m] = np.mean(per_list, axis=0)

    ma_cube = (par_cube + 2.0 * per_cube) / 3.0 if polarised else par_cube
    p_order = np.argsort(probe_cm1)
    probe_out = probe_cm1[p_order]
    par_cube = par_cube[:, p_order, :]
    if polarised:
        per_cube = per_cube[:, p_order, :]
    ma_cube = ma_cube[:, p_order, :]

    return {
        "info_path": info_path,
        "prefix": prefix,
        "t2_ps": t2_arr_fs[:n_t2_disk] / 1000.0,
        "probe_cm1": probe_out,
        "axis0": axis0,
        "peak_cm1": peak_cm1,
        "w0_cm1": w0_eff,
        "dt1_fs": dt1_fs,
        "par_cube_mOD": par_cube,
        "per_cube_mOD": per_cube if polarised else None,
        "ma_cube_mOD": ma_cube,
    }





