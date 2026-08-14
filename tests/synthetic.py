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
    for s in range(nscans):
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
    for s in range(n_scans):
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




