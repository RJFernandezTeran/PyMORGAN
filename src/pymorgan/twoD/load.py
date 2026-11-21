"""Loading stage of the 2-D pipeline: readers and their registry.

A 2-D *loader* maps a file path to ``(Z, pump, probe, delays, units, freq_units)``:

==========  ===========================================================
Field       Meaning / shape
==========  ===========================================================
Z           Signal cube, ``[Npump x Nprobe x Nt2]`` (stack of 2-D maps)
pump        Pump axis omega_1, ``[Npump]``
probe       Probe axis omega_3, ``[Nprobe]``
delays      Population times t2, ``[Nt2]``
units       Unit-label dictionary
freq_units  Spectral-axis unit string for the labels (e.g. ``"cm-1"``)
==========  ===========================================================
"""

from __future__ import annotations

import re
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from .. import helpers as hlp
from ..log import get_logger

logger = get_logger(__name__)

LoaderResult = tuple[Any, Any, Any, Any, dict, str]
Loader = Callable[[str], LoaderResult]

_LOADERS: dict[str, Loader] = {}


def register_map_loader(*names: str) -> Callable[[Loader], Loader]:
    """Register a 2-D loader under one or more data-type names."""

    def decorator(fn: Loader) -> Loader:
        for name in names:
            _LOADERS[name] = fn
        return fn

    return decorator


def get_map_loader(name: str) -> Loader:
    """Return the loader registered for ``name`` or raise ``KeyError``."""
    try:
        return _LOADERS[name]
    except KeyError:
        raise KeyError(
            f"No 2-D loader registered for {name!r}. Available: {available_map_loaders()}"
        ) from None


def available_map_loaders() -> list[str]:
    """Return the sorted list of registered data-type names."""
    return sorted(_LOADERS)


def _units_2d(unitsT: str, unitsL: str) -> dict:
    return {
        "unitsL_lbl": "Wavenumber",
        "unitsL_ltx": r"cm$^{-1}$",
        "unitsZ_lbl": r"S$_{\text{2DIR}}$",
        "unitsZ_ltx": r"mOD cm$^{1/2}$",
        "unitsZ": "int2D",
        "unitsT_ltx": unitsT,
        "unitsL": unitsL,
    }


@register_map_loader("P2DAT")
def read_P2DAT(datafilename: str) -> LoaderResult:
    """Read a processed 2-D (P2DAT) file into a signal cube.

    Layout: the first row is ``0, 0, t2_1, t2_2, ...`` (population times); every
    subsequent row is ``pump, probe, S(t2_1), S(t2_2), ...``. The flat
    pump/probe column ordering is reshaped (Fortran order) into
    ``Z[pump, probe, t2]``. Data is in mOD.
    """
    import pandas as pd

    data = np.array(pd.read_csv(datafilename, skiprows=0, header=None))

    delays = data[0, 2:]
    pump = np.unique(data[1:, 0])
    probe = np.unique(data[1:, 1])
    Zflat = data[1:, 2:]

    Z = Zflat.reshape((len(pump), len(probe), len(delays)), order="F")

    return Z, pump, probe, delays, _units_2d("ps", "cm-1"), "cm-1"


def write_P2DAT(
    path: str | PathLike,
    pump: np.ndarray,
    probe: np.ndarray,
    delays: np.ndarray,
    Z_cube: np.ndarray,
) -> None:
    """Write a 3D dataset cube into standard P2DAT format.

    Parameters
    ----------
    path : str or Path
        Destination filepath.
    pump : 1D array-like
        Pump frequency axis (Npump).
    probe : 1D array-like
        Probe frequency axis (Nprobe).
    delays : 1D array-like
        Population delay times t2 (Nt2).
    Z_cube : 2D or 3D ndarray
        Signal array of shape (Npump, Nprobe, Nt2) or (Npump, Nprobe).
    """
    pump = np.asarray(pump, dtype=float)
    probe = np.asarray(probe, dtype=float)
    delays = np.asarray(delays, dtype=float)
    Z_cube = np.asarray(Z_cube, dtype=float)

    if Z_cube.ndim == 2:
        Z_cube = Z_cube[:, :, np.newaxis]
    if len(delays) == 0:
        delays = np.array([0.0])

    lines = [",".join(["0", "0"] + [f"{d}" for d in delays])]
    for j in range(len(probe)):
        for i in range(len(pump)):
            row = [f"{pump[i]:.4f}", f"{probe[j]:.4f}"]
            row += [f"{Z_cube[i, j, k]:.8g}" for k in range(len(delays))]
            lines.append(",".join(row))

    with open(str(path), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


save_P2DAT = write_P2DAT



@register_map_loader("MESS_2DIR")
def read_MESS_2DIR(path: str, progress_tracker=None) -> tuple[Any, Any, Any, Any, dict, str, str]:
    """Read a MESS 2DIR dataset (folder of CSV files) into a signal cube."""
    import pandas as pd

    folder = Path(path)
    from .progress import ProgressTracker

    close_tracker = False
    if progress_tracker is None:
        progress_tracker = ProgressTracker(1, title="Loading 2D Dataset", label="Initializing...")
        close_tracker = True

    def load_csv(p: Path) -> np.ndarray:
        import pandas as pd

        return pd.read_csv(p, header=None, dtype=float).to_numpy()

    def load_csv_1d(p: Path) -> np.ndarray:
        return load_csv(p).ravel()

    # 1. Load basic files
    cmprobe = load_csv_1d(folder / f"{folder.name}_wavenumbers.csv")
    bins = load_csv(folder / f"{folder.name}_bins.csv")
    t2delays = load_csv_1d(folder / f"{folder.name}_delays.csv")

    transient_delays_file = folder / f"{folder.name}_transientDelays.csv"
    if transient_delays_file.exists():
        transient_delays = load_csv_1d(transient_delays_file)
        t2delays = np.concatenate([t2delays, transient_delays])

    Nspectra = int(load_csv_1d(folder / f"{folder.name}_Nspectra.csv")[0])
    Ndatastates = int(load_csv_1d(folder / f"{folder.name}_Ndatastates.csv")[0])
    Ndummies = int(load_csv_1d(folder / f"{folder.name}_dummies.csv")[0])

    interleaves_file = folder / f"{folder.name}_interleaves.csv"
    if interleaves_file.exists():
        try:
            n_interleave = len(load_csv_1d(interleaves_file))
        except Exception:
            n_interleave = 1
    else:
        n_interleave = 1

    slowmod_file = folder / f"{folder.name}_slowModulation.csv"
    with open(slowmod_file) as f:
        slowmod_lines = [line.strip() for line in f if line.strip()]

    nslowmod = int(slowmod_lines[0])
    mod_a = slowmod_lines[1].split(":") if len(slowmod_lines) > 1 else ["", "NaN"]
    mod_b = slowmod_lines[2].split(":") if len(slowmod_lines) > 2 else ["", "NaN"]
    mod_c = slowmod_lines[3].split(":") if len(slowmod_lines) > 3 else ["", "NaN"]
    mod_spec = slowmod_lines[4].split(":") if len(slowmod_lines) > 4 else ["", "NaN"]
    mod_piezo = slowmod_lines[5] if len(slowmod_lines) > 5 else "-1"

    def parse_mod_vals(s: str) -> list[float]:
        s = s.strip().strip("[]")
        if not s or s.lower() == "nan":
            return []
        parts = s.replace(",", " ").split()
        vals = []
        for p in parts:
            try:
                val = float(p)
                if not np.isnan(val):
                    vals.append(val)
            except ValueError:
                logger.debug("Ignoring non-numeric metadata entry %r.", p, exc_info=True)
        return vals

    n_modA = len(parse_mod_vals(mod_a[1])) if len(mod_a) > 1 else 0
    n_modB = len(parse_mod_vals(mod_b[1])) if len(mod_b) > 1 else 0
    n_modC = len(parse_mod_vals(mod_c[1])) if len(mod_c) > 1 else 0
    n_modSpec = len(parse_mod_vals(mod_spec[1])) if len(mod_spec) > 1 else 0

    n_modA = 1 if n_modA == 0 else n_modA
    n_modB = 1 if n_modB == 0 else n_modB
    n_modC = 1 if n_modC == 0 else n_modC
    n_modSpec = 1 if n_modSpec == 0 else n_modSpec

    if mod_piezo != "-1":
        mod_piezo_parts = mod_piezo.split(":")
        if len(mod_piezo_parts) > 1:
            piezo_vals = parse_mod_vals(mod_piezo_parts[1])
            n_4pm = len(piezo_vals) // (n_modA * n_modB * n_modC * n_modSpec)
        else:
            n_4pm = 1
    else:
        n_4pm = 1
    if n_4pm == 0:
        n_4pm = 1

    # Correct interferograms for shaper data (integrated into loading step)
    if Ndatastates == 4:
        # Check if the raw files exist
        has_raw_ds = True
        for j in range(4):
            test_file = folder / f"{folder.name}_interferogram_ds{j}_sp0_sm0_de0_in0.csv"
            if not test_file.exists():
                has_raw_ds = False
                break

        if has_raw_ds:
            sign = [+1, -1, +1, -1]
            for p in range(n_interleave):
                for spec in range(Nspectra):
                    for sm in range(nslowmod):
                        for m in range(len(t2delays)):
                            avg_int_fn = folder / f"{folder.name}_interferogram_sp{spec}_sm{sm}_de{m}_in{p}.csv"

                            bak_file = avg_int_fn.with_suffix('.csv.bak')
                            if not bak_file.exists():
                                if avg_int_fn.exists():
                                    import shutil
                                    shutil.copy2(avg_int_fn, bak_file)

                            nbins = bins.shape[0]
                            avg_int = np.zeros((nbins, 2))

                            try:
                                for j in range(4):
                                    ds_int_fn = folder / f"{folder.name}_interferogram_ds{j}_sp{spec}_sm{sm}_de{m}_in{p}.csv"
                                    ds_cts_fn = folder / f"{folder.name}_count_ds{j}_sp{spec}_sm{sm}_de{m}_in{p}.csv"

                                    ds_int_data = load_csv(ds_int_fn)
                                    ds_cts_data = load_csv(ds_cts_fn)

                                    cts = np.squeeze(ds_cts_data)
                                    div = np.zeros_like(ds_int_data)
                                    for col in range(ds_int_data.shape[1]):
                                        mask = (cts != 0)
                                        div[mask, col] = ds_int_data[mask, col] / cts[mask]

                                    avg_int += sign[j] * div

                                avg_int /= 4.0

                                pd.DataFrame(avg_int).to_csv(avg_int_fn, index=False, header=False)
                            except Exception:
                                logger.warning(
                                    "Could not correct the interferogram for de%s; "
                                    "the uncorrected data is kept.", m, exc_info=True
                                )

    tempdir = folder / "temp"
    tempdir_out = folder.parent / f"{folder.name}temp"
    if not tempdir.exists() and tempdir_out.exists():
        tempdir = tempdir_out

    preview_mode = False
    n_scans = 1
    load_dir = folder

    if tempdir.exists() and tempdir.is_dir():
        temp_files = [f.name for f in tempdir.iterdir() if f.is_file()]
        temp_interferogram_files = [f for f in temp_files if "interferogram" in f.lower()]

        finished_delays = np.zeros(len(t2delays), dtype=int)
        for i in range(len(t2delays)):
            nfiles = sum(1 for f in temp_interferogram_files if f"de{i}_" in f.lower())
            if nfiles == nslowmod * Ndatastates * Nspectra:
                finished_delays[i] = 1

        if len(temp_interferogram_files) >= (len(t2delays) * Ndatastates * nslowmod):
            n_scans = round(len(temp_interferogram_files) / (len(t2delays) * Ndatastates * nslowmod))
        else:
            n_scans = 0

        if n_scans == 0:
            preview_mode = True
            finished_mask = (finished_delays == 1)
            t2delays = t2delays[finished_mask]
            load_dir = tempdir

    total_states = Ndummies * n_interleave * Nspectra * nslowmod * Ndatastates
    endings = [[None] * total_states for _ in range(len(t2delays))]
    endings_du = [[None] * total_states for _ in range(len(t2delays))]
    short_endings = [[None] * (total_states // Ndatastates) for _ in range(len(t2delays))]
    short_endings_du = [[None] * (total_states // Ndatastates) for _ in range(len(t2delays))]

    n = 0
    for q in range(Ndummies):
        for p in range(n_interleave):
            for i in range(Nspectra):
                for j in range(nslowmod):
                    for k in range(Ndatastates):
                        col_idx = n
                        short_col_idx = n // Ndatastates
                        for m in range(len(t2delays)):
                            if not preview_mode:
                                endings[m][col_idx] = f"_ds{k}_sp{i}_sm{j}_de{m}_in{p}.csv"
                                short_endings[m][short_col_idx] = f"_sp{i}_sm{j}_de{m}_in{p}.csv"
                                endings_du[m][col_idx] = f"_ds{k}_sp{i}_sm{j}_de{m}_in{p}_du{q}.csv"
                                short_endings_du[m][short_col_idx] = f"_sp{i}_sm{j}_de{m}_in{p}_du{q}.csv"
                            else:
                                endings[m][col_idx] = f"_ds{k}_sp{i}_sm{j}_de{m}_in{p}_0.csv"
                                endings_du[m][col_idx] = f"_ds{k}_sp{i}_sm{j}_de{m}_in{p}_du{q}.csv"
                                short_endings[m][short_col_idx] = f"_sp{i}_sm{j}_de{m}_in{p}_0.csv"
                                short_endings_du[m][short_col_idx] = f"_sp{i}_sm{j}_de{m}_in{p}_du{q}.csv"
                        n += 1

    dir_files = [f.name for f in load_dir.iterdir() if f.is_file()]
    dir_files_no_dummies = [f for f in dir_files if "dummies" not in f.lower()]
    has_du = any("_du" in f.lower() for f in dir_files_no_dummies)
    if has_du:
        endings = endings_du
        short_endings = short_endings_du

    autodetect_datatype = not preview_mode
    datatype = "Raw"
    if autodetect_datatype:
        has_signal_file = any("signal" in f.lower() for f in dir_files_no_dummies)
        if has_signal_file:
            datatype = "Signal"

    if Ndatastates == 1:
        chopper = "Chopper OFF"
    elif Ndatastates == 2:
        chopper = "Chopper ON"
    elif Ndatastates == 4:
        chopper = "Wobbler"
    else:
        chopper = "Chopper OFF"

    if n_4pm == 1:
        slowmod_type = "SlowMod OFF"
    elif n_4pm == 4:
        slowmod_type = "4PM"
    else:
        slowmod_type = "SlowMod OFF"

    Nstates_to_load = Ndatastates * nslowmod * n_interleave * Nspectra
    prefix = load_dir / folder.name

    orig_Ndatastates = Ndatastates
    if datatype == "Raw":
        total_steps = Nstates_to_load * len(t2delays)
        progress_tracker.set_total(total_steps)

        probe_data = [[None] * Nstates_to_load for _ in range(len(t2delays))]
        reference_data = [[None] * Nstates_to_load for _ in range(len(t2delays))]
        interferogram_data = [[None] * Nstates_to_load for _ in range(len(t2delays))]
        count_data = [[None] * Nstates_to_load for _ in range(len(t2delays))]
        signal_data = [[None] * Nstates_to_load for _ in range(len(t2delays))]
        t1delays_data = [[None] * Nstates_to_load for _ in range(len(t2delays))]

        start_NZ = np.zeros((len(t2delays), Nstates_to_load), dtype=int)
        end_NZ = np.zeros((len(t2delays), Nstates_to_load), dtype=int)
        int_size = np.zeros((len(t2delays), Nstates_to_load), dtype=int)

        for k in range(Nstates_to_load):
            for m in range(len(t2delays)):
                step = k * len(t2delays) + m + 1
                if hasattr(progress_tracker, "update_split"):
                    progress_tracker.update_split(step, total_steps, "loading")
                else:
                    progress_tracker.update(step, label=f"Loading ({step} of {total_steps})")
                ending = endings[m][k]
                cnt = load_csv_1d(Path(f"{prefix}_count{ending}"))
                if np.sum(cnt) == 0:
                    raise ValueError("2D-IR dataset incomplete - some delays have zero counts!")

                pr = load_csv(Path(f"{prefix}_probe{ending}"))
                ref = load_csv(Path(f"{prefix}_reference{ending}"))
                interf = load_csv(Path(f"{prefix}_interferogram{ending}"))

                if interf.ndim > 1:
                    interf = interf[:, 0]
                else:
                    interf = interf.ravel()

                nz_indices = np.nonzero(cnt)[0]
                start_idx = nz_indices[0]
                end_idx = nz_indices[-1]

                start_NZ[m, k] = start_idx
                end_NZ[m, k] = end_idx

                cnt = cnt[start_idx:end_idx + 1]
                pr = pr[start_idx:end_idx + 1, :] if pr.ndim > 1 else pr[start_idx:end_idx + 1]
                ref = ref[start_idx:end_idx + 1, :] if ref.ndim > 1 else ref[start_idx:end_idx + 1]
                interf = interf[start_idx:end_idx + 1]
                t1 = bins[start_idx:end_idx + 1]

                if pr.ndim > 1:
                    pr = pr / cnt[:, np.newaxis]
                    ref = ref / cnt[:, np.newaxis]
                else:
                    pr = pr / cnt
                    ref = ref / cnt
                interf = interf / cnt

                pr = pr / np.mean(pr, axis=0)
                ref = ref / np.mean(ref, axis=0)

                sig = -1000 * np.log10(pr / ref)

                count_data[m][k] = cnt
                probe_data[m][k] = pr
                reference_data[m][k] = ref
                interferogram_data[m][k] = interf
                signal_data[m][k] = sig
                t1delays_data[m][k] = t1
                int_size[m, k] = len(interf)

        min_size = int(np.min(int_size))
        for k in range(Nstates_to_load):
            for m in range(len(t2delays)):
                startcut = int_size[m, k] - min_size
                count_data[m][k] = count_data[m][k][startcut:]
                probe_data[m][k] = probe_data[m][k][startcut:, :] if probe_data[m][k].ndim > 1 else probe_data[m][k][startcut:]
                reference_data[m][k] = reference_data[m][k][startcut:, :] if reference_data[m][k].ndim > 1 else reference_data[m][k][startcut:]
                interferogram_data[m][k] = interferogram_data[m][k][startcut:]
                signal_data[m][k] = signal_data[m][k][startcut:, :] if signal_data[m][k].ndim > 1 else signal_data[m][k][startcut:]
                t1delays_data[m][k] = t1delays_data[m][k][startcut:]

        if n_interleave > 1:
            N_states_reduced = Ndatastates * nslowmod * Nspectra
            for k in range(N_states_reduced):
                for m in range(len(t2delays)):
                    for p in range(1, n_interleave):
                        idx_to_add = k + p * N_states_reduced
                        signal_data[m][k] += signal_data[m][idx_to_add]
                        interferogram_data[m][k] += interferogram_data[m][idx_to_add]
                    signal_data[m][k] /= n_interleave
                    interferogram_data[m][k] /= n_interleave
            signal_data = [row[:N_states_reduced] for row in signal_data]
            interferogram_data = [row[:N_states_reduced] for row in interferogram_data]

        if slowmod_type == "SlowMod OFF":
            if chopper == "Chopper ON":
                for m in range(len(t2delays)):
                    signal_data[m][0] = signal_data[m][0] - signal_data[m][1]
                    interferogram_data[m][0] = interferogram_data[m][0] - interferogram_data[m][1]
                signal_data = [row[:1] for row in signal_data]
                interferogram_data = [row[:1] for row in interferogram_data]
            elif chopper == "Wobbler":
                for m in range(len(t2delays)):
                    signal_data[m][0] = signal_data[m][0] - signal_data[m][1] + signal_data[m][2] - signal_data[m][3]
                    interferogram_data[m][0] = interferogram_data[m][0] - interferogram_data[m][1] + interferogram_data[m][2] - interferogram_data[m][3]
                signal_data = [row[:1] for row in signal_data]
                interferogram_data = [row[:1] for row in interferogram_data]
        elif slowmod_type == "4PM":
            temp_sig = [[None] * nslowmod for _ in range(len(t2delays))]
            temp_int = [[None] * nslowmod for _ in range(len(t2delays))]
            if chopper == "Chopper ON":
                for m in range(len(t2delays)):
                    for k in range(nslowmod):
                        temp_sig[m][k] = signal_data[m][2 * k + 1] - signal_data[m][2 * k]
                        temp_int[m][k] = interferogram_data[m][2 * k + 1] - interferogram_data[m][2 * k]
                    sig_avg = sum(temp_sig[m][k] for k in range(nslowmod)) / 4.0
                    int_avg = sum(temp_int[m][k] for k in range(nslowmod)) / 4.0
                    signal_data[m][0] = sig_avg
                    interferogram_data[m][0] = int_avg
                signal_data = [row[:1] for row in signal_data]
                interferogram_data = [row[:1] for row in interferogram_data]
            elif chopper == "Chopper OFF":
                for m in range(len(t2delays)):
                    sig_avg = sum(signal_data[m][k] for k in range(nslowmod)) / 4.0
                    int_avg = sum(interferogram_data[m][k] for k in range(nslowmod)) / 4.0
                    signal_data[m][0] = sig_avg
                    interferogram_data[m][0] = int_avg
                signal_data = [row[:1] for row in signal_data]
                interferogram_data = [row[:1] for row in interferogram_data]

    elif datatype == "Signal":
        Nstates_to_load = nslowmod * n_interleave * Nspectra * Ndummies
        total_steps = Nstates_to_load * len(t2delays)
        progress_tracker.set_total(total_steps)

        interferogram_data = [[None] * Nstates_to_load for _ in range(len(t2delays))]
        count_data = [[None] * Nstates_to_load for _ in range(len(t2delays))]
        signal_data = [[None] * Nstates_to_load for _ in range(len(t2delays))]
        t1delays_data = [[None] * Nstates_to_load for _ in range(len(t2delays))]

        start_nNaN = np.zeros((len(t2delays), Nstates_to_load), dtype=int)
        end_nNaN = np.zeros((len(t2delays), Nstates_to_load), dtype=int)
        int_size = np.zeros((len(t2delays), Nstates_to_load), dtype=int)

        for k in range(Nstates_to_load):
            for m in range(len(t2delays)):
                step = k * len(t2delays) + m + 1
                if hasattr(progress_tracker, "update_split"):
                    progress_tracker.update_split(step, total_steps, "loading")
                else:
                    progress_tracker.update(step, label=f"Loading ({step} of {total_steps})")
                ending = short_endings[m][k]
                cnt = load_csv_1d(Path(f"{prefix}_count{ending}"))
                interf = Ndatastates * load_csv(Path(f"{prefix}_interferogram{ending}"))
                sig = Ndatastates * load_csv(Path(f"{prefix}_signal{ending}"))

                if interf.ndim > 1:
                    interf = interf[:, 0]
                else:
                    interf = interf.ravel()

                not_nan_indices = np.nonzero(~np.isnan(interf))[0]
                start_idx = not_nan_indices[0]
                end_idx = not_nan_indices[-1]

                start_nNaN[m, k] = start_idx
                end_nNaN[m, k] = end_idx

                sig = sig[start_idx:end_idx + 1, :] if sig.ndim > 1 else sig[start_idx:end_idx + 1]
                interf = interf[start_idx:end_idx + 1]
                t1 = bins[start_idx:end_idx + 1]

                count_data[m][k] = cnt
                interferogram_data[m][k] = interf
                signal_data[m][k] = sig
                t1delays_data[m][k] = t1
                int_size[m, k] = len(interf)

        k_last = Nstates_to_load - 1
        min_size = int(np.min(int_size[:, k_last]))
        for m in range(len(t2delays)):
            startcut = int(int_size[m, k_last] - min_size)
            count_data[m][k_last] = count_data[m][k_last][startcut:]
            if chopper == "Wobbler":
                interferogram_data[m][k_last] = interferogram_data[m][k_last][startcut:] / 4.0
                signal_data[m][k_last] = (signal_data[m][k_last][startcut:, :] / 4.0) if signal_data[m][k_last].ndim > 1 else (signal_data[m][k_last][startcut:] / 4.0)
            else:
                interferogram_data[m][k_last] = interferogram_data[m][k_last][startcut:]
                signal_data[m][k_last] = signal_data[m][k_last][startcut:, :] if signal_data[m][k_last].ndim > 1 else signal_data[m][k_last][startcut:]
            t1delays_data[m][k_last] = t1delays_data[m][k_last][startcut:]

        Ndatastates = 1

        if slowmod_type == "4PM":
            temp_sig = [[None] * Ndummies for _ in range(len(t2delays))]
            temp_int = [[None] * Ndummies for _ in range(len(t2delays))]
            for q in range(Ndummies):
                for m in range(len(t2delays)):
                    sig_sum = sum(signal_data[m][i + q * n_4pm] for i in range(n_4pm)) / float(n_4pm)
                    int_sum = sum(interferogram_data[m][i + q * n_4pm] for i in range(n_4pm)) / float(n_4pm)
                    temp_sig[m][q] = sig_sum
                    temp_int[m][q] = int_sum
            signal_data = temp_sig
            interferogram_data = temp_int
            Ndatastates = 1

    if t2delays.ndim > 1 and t2delays.shape[1] == 2:
        delay_index = np.argsort(t2delays[:, 1])
    else:
        delay_index = np.argsort(t2delays)

    t2delays = t2delays[delay_index]
    signal_data = [signal_data[idx] for idx in delay_index]
    interferogram_data = [interferogram_data[idx] for idx in delay_index]
    t1delays_data = [t1delays_data[idx] for idx in delay_index]

    t1_col = t1delays_data[0][0]
    if t1_col.ndim > 1 and t1_col.shape[1] > 1:
        pump = t1_col[:, 1]
    else:
        pump = t1_col.ravel()

    probe = cmprobe

    Z = np.zeros((len(pump), len(probe), len(t2delays)))
    raw_interferogram = np.zeros((len(pump), len(t2delays)))
    for m in range(len(t2delays)):
        Z[:, :, m] = signal_data[m][0]
        raw_interferogram[:, m] = interferogram_data[m][0]

    raw_signal = Z.copy()
    raw_t1delays = pump.copy()

    if t2delays.ndim > 1 and t2delays.shape[1] == 2:
        t2delays_ps = t2delays[:, 1] / 1000.0
    else:
        t2delays_ps = t2delays / 1000.0

    meta_file = folder / f"{folder.name}_meta.txt"
    is_shaper = False
    if meta_file.exists():
        try:
            with open(meta_file) as f:
                for line in f:
                    if line.strip().startswith("Preset:"):
                        if "shaper" in line.lower():
                            is_shaper = True
                        break
        except Exception:
            logger.debug(
                "Could not read the acquisition preset from %s; "
                "assuming a non-shaper dataset.", meta_file, exc_info=True
            )
    else:
        is_shaper = len(bins) < 200

    datatype_str = "shaper" if is_shaper else "interferometer"

    if close_tracker:
        progress_tracker.close()

    raw_signal_ss = None
    raw_interferogram_ss = None
    if not preview_mode and n_scans > 1 and tempdir.exists() and tempdir.is_dir():
        try:
            temp_dir_files = [
                f.name for f in tempdir.iterdir() if f.is_file() and "dummies" not in f.name.lower()
            ]
            temp_has_du = any("_du" in f.lower() for f in temp_dir_files)
            temp_has_signal = any("signal" in f.lower() for f in temp_dir_files)
            temp_prefix = tempdir / folder.name
            n_t1 = len(pump)
            n_pr = len(probe)
            n_t2 = len(t2delays)

            sig_ss = np.full((n_t1, n_pr, n_t2, n_scans), np.nan, dtype=float)
            int_ss = np.full((n_t1, n_t2, n_scans), np.nan, dtype=float)

            for s_idx in range(n_scans):
                if temp_has_signal:
                    n_st_temp = nslowmod * n_interleave * Nspectra * Ndummies
                    s_sigs = [[None] * n_st_temp for _ in range(n_t2)]
                    s_ints = [[None] * n_st_temp for _ in range(n_t2)]
                    col_counter = 0
                    for q in range(Ndummies):
                        for p_i in range(n_interleave):
                            for sp_i in range(Nspectra):
                                for sm_i in range(nslowmod):
                                    for m in range(n_t2):
                                        if temp_has_du:
                                            end_s = f"_sp{sp_i}_sm{sm_i}_de{m}_in{p_i}_du{q}_{s_idx}.csv"
                                        else:
                                            end_s = f"_sp{sp_i}_sm{sm_i}_de{m}_in{p_i}_{s_idx}.csv"
                                        interf = orig_Ndatastates * load_csv(
                                            Path(f"{temp_prefix}_interferogram{end_s}")
                                        )
                                        sig = orig_Ndatastates * load_csv(
                                            Path(f"{temp_prefix}_signal{end_s}")
                                        )
                                        interf = interf[:, 0] if interf.ndim > 1 else interf.ravel()
                                        not_nan = np.nonzero(~np.isnan(interf))[0]
                                        st_i, en_i = not_nan[0], not_nan[-1]
                                        sig = sig[st_i : en_i + 1, :] if sig.ndim > 1 else sig[st_i : en_i + 1]
                                        interf = interf[st_i : en_i + 1]
                                        startcut = max(0, len(interf) - n_t1)
                                        div_factor = 4.0 if chopper == "Wobbler" else 1.0
                                        s_sigs[m][col_counter] = (sig[startcut : startcut + n_t1] / div_factor)
                                        s_ints[m][col_counter] = (interf[startcut : startcut + n_t1] / div_factor)
                                    col_counter += 1
                    if slowmod_type == "4PM":
                        for m in range(n_t2):
                            s_sigs[m][0] = sum(s_sigs[m][i] for i in range(n_4pm)) / float(n_4pm)
                            s_ints[m][0] = sum(s_ints[m][i] for i in range(n_4pm)) / float(n_4pm)
                else:
                    n_st_temp = orig_Ndatastates * nslowmod * n_interleave * Nspectra
                    s_sigs = [[None] * n_st_temp for _ in range(n_t2)]
                    s_ints = [[None] * n_st_temp for _ in range(n_t2)]
                    col_counter = 0
                    for p_i in range(n_interleave):
                        for sp_i in range(Nspectra):
                            for sm_i in range(nslowmod):
                                for ds_k in range(orig_Ndatastates):
                                    for m in range(n_t2):
                                        if temp_has_du:
                                            end_s = f"_ds{ds_k}_sp{sp_i}_sm{sm_i}_de{m}_in{p_i}_du0_{s_idx}.csv"
                                        else:
                                            end_s = f"_ds{ds_k}_sp{sp_i}_sm{sm_i}_de{m}_in{p_i}_{s_idx}.csv"
                                        cnt = load_csv_1d(Path(f"{temp_prefix}_count{end_s}"))
                                        pr = load_csv(Path(f"{temp_prefix}_probe{end_s}"))
                                        ref = load_csv(Path(f"{temp_prefix}_reference{end_s}"))
                                        interf = load_csv(Path(f"{temp_prefix}_interferogram{end_s}"))
                                        interf = interf[:, 0] if interf.ndim > 1 else interf.ravel()
                                        nz = np.nonzero(cnt)[0]
                                        st_i, en_i = nz[0], nz[-1]
                                        cnt = cnt[st_i : en_i + 1]
                                        pr = pr[st_i : en_i + 1, :] if pr.ndim > 1 else pr[st_i : en_i + 1]
                                        ref = ref[st_i : en_i + 1, :] if ref.ndim > 1 else ref[st_i : en_i + 1]
                                        interf = interf[st_i : en_i + 1] / cnt
                                        pr = pr / (cnt[:, np.newaxis] if pr.ndim > 1 else cnt)
                                        ref = ref / (cnt[:, np.newaxis] if ref.ndim > 1 else cnt)
                                        pr = pr / np.mean(pr, axis=0)
                                        ref = ref / np.mean(ref, axis=0)
                                        sig = -1000.0 * np.log10(pr / ref)
                                        startcut = max(0, len(interf) - n_t1)
                                        s_sigs[m][col_counter] = sig[startcut : startcut + n_t1]
                                        s_ints[m][col_counter] = interf[startcut : startcut + n_t1]
                                    col_counter += 1
                    if n_interleave > 1:
                        n_red = orig_Ndatastates * nslowmod * Nspectra
                        for k_i in range(n_red):
                            for m in range(n_t2):
                                for p_i in range(1, n_interleave):
                                    s_sigs[m][k_i] = s_sigs[m][k_i] + s_sigs[m][k_i + p_i * n_red]
                                    s_ints[m][k_i] = s_ints[m][k_i] + s_ints[m][k_i + p_i * n_red]
                                s_sigs[m][k_i] /= n_interleave
                                s_ints[m][k_i] /= n_interleave
                        s_sigs = [row[:n_red] for row in s_sigs]
                        s_ints = [row[:n_red] for row in s_ints]
                    if slowmod_type == "SlowMod OFF":
                        if chopper == "Chopper ON":
                            for m in range(n_t2):
                                s_sigs[m][0] = s_sigs[m][0] - s_sigs[m][1]
                                s_ints[m][0] = (s_ints[m][0] - s_ints[m][1]) / 2.0
                        elif chopper == "Wobbler":
                            sig_div = 2.0 if datatype == "Signal" else 1.0
                            int_div = 4.0 if datatype == "Signal" else 1.0
                            for m in range(n_t2):
                                s_sigs[m][0] = (
                                    s_sigs[m][0] - s_sigs[m][1] + s_sigs[m][2] - s_sigs[m][3]
                                ) / sig_div
                                s_ints[m][0] = (
                                    s_ints[m][0] - s_ints[m][1] + s_ints[m][2] - s_ints[m][3]
                                ) / int_div
                    elif slowmod_type == "4PM":
                        if chopper == "Chopper ON":
                            for m in range(n_t2):
                                s_sigs[m][0] = (
                                    sum(s_sigs[m][2 * k_i + 1] - s_sigs[m][2 * k_i] for k_i in range(nslowmod))
                                    / 4.0
                                )
                                s_ints[m][0] = (
                                    sum(s_ints[m][2 * k_i + 1] - s_ints[m][2 * k_i] for k_i in range(nslowmod))
                                    / 4.0
                                )
                        elif chopper == "Chopper OFF":
                            for m in range(n_t2):
                                s_sigs[m][0] = sum(s_sigs[m][k_i] for k_i in range(nslowmod)) / 4.0
                                s_ints[m][0] = sum(s_ints[m][k_i] for k_i in range(nslowmod)) / 4.0

                for m_sorted, m_orig in enumerate(delay_index):
                    sig_ss[:, :, m_sorted, s_idx] = s_sigs[m_orig][0]
                    int_ss[:, m_sorted, s_idx] = s_ints[m_orig][0]

            if not np.all(np.isnan(sig_ss)):
                raw_signal_ss = sig_ss
                raw_interferogram_ss = int_ss
        except Exception:
            logger.debug("Could not load per-scan MESS_2DIR data from %s", tempdir, exc_info=True)

    units = _units_2d("ps", "cm-1")
    if raw_signal_ss is not None:
        units["raw_signal_ss"] = raw_signal_ss
        units["raw_interferogram_ss"] = raw_interferogram_ss
        units["nscans"] = int(raw_signal_ss.shape[-1])
    try:
        from ..oneD.load import _find_mess_sample_info_file, parse_mess_sample_info

        info_file = _find_mess_sample_info_file(folder)
        if info_file is not None:
            info_str = parse_mess_sample_info(info_file)
            if info_str:
                units["sample_info"] = info_str
    except Exception:
        pass

    return Z, pump, probe, t2delays_ps, units, "cm-1", datatype_str, preview_mode, raw_signal, raw_interferogram, raw_t1delays


_MAP_DATATYPE_GLOBS: dict[str, str] = {"P2DAT": "*.p2dat"}
_MAP_DATASET_DETECTORS: dict[str, Callable[[Path], bool]] = {}
_MAP_DATASET_FILE_FILTERS: dict[str, Callable[[Path], bool]] = {}


def register_map_dataset_glob(name: str, pattern: str) -> None:
    """Declare ``name`` as a file-based map format whose datasets match ``pattern``."""
    _MAP_DATATYPE_GLOBS[name] = pattern


def register_map_dataset_file_filter(
    *names: str,
) -> Callable[[Callable[[Path], bool]], Callable[[Path], bool]]:
    """Register a ``file -> bool`` predicate filtering files for a file-based map format."""

    def decorator(fn: Callable[[Path], bool]) -> Callable[[Path], bool]:
        for name in names:
            _MAP_DATASET_FILE_FILTERS[name] = fn
        return fn

    return decorator


def register_map_dataset_detector(
    *names: str,
) -> Callable[[Callable[[Path], bool]], Callable[[Path], bool]]:
    """Register a ``folder -> bool`` predicate marking a directory-based map dataset."""

    def decorator(fn: Callable[[Path], bool]) -> Callable[[Path], bool]:
        for name in names:
            _MAP_DATASET_DETECTORS[name] = fn
        return fn

    return decorator


def map_dataset_glob(name: str) -> str | None:
    """Return the file glob for a file-based map format, or ``None``."""
    return _MAP_DATATYPE_GLOBS.get(name)


def is_map_dataset_file_status(name: str, filepath: str | PathLike) -> tuple[bool, bool]:
    """Whether ``filepath`` is a candidate/valid file dataset for map format ``name``.

    Returns ``(is_candidate, is_valid)``.
    """
    fn = _MAP_DATASET_FILE_FILTERS.get(name)
    if fn is None:
        return True, True
    try:
        res = fn(Path(filepath))
        if isinstance(res, tuple):
            return bool(res[0]), bool(res[1])
        if res:
            return True, True
        return False, False
    except OSError:
        return False, False


def is_map_dataset_file(name: str, filepath: str | PathLike) -> bool:
    """Whether ``filepath`` is a valid file dataset for map format ``name``."""
    _, valid = is_map_dataset_file_status(name, filepath)
    return valid


def is_map_directory_format(name: str) -> bool:
    """Whether ``name`` stores each map dataset as a directory."""
    return name in _MAP_DATASET_DETECTORS


def is_map_dataset_dir(name: str, folder: str | PathLike) -> bool:
    """Whether ``folder`` is a loadable map dataset of type ``name``."""
    fn = _MAP_DATASET_DETECTORS.get(name)
    try:
        return bool(fn and fn(Path(folder)))
    except OSError:
        return False


@register_map_dataset_detector("MESS_2DIR")
def detect_MESS_2DIR(folder: Path) -> bool:
    """Detect if a folder is a MESS_2DIR dataset."""
    if not folder.is_dir():
        return False
    return (
        (folder / f"{folder.name}_wavenumbers.csv").exists()
        and (folder / f"{folder.name}_bins.csv").exists()
        and (folder / f"{folder.name}_delays.csv").exists()
    )


# --------------------------------------------------------------------------- #
#                      PyMESS 2D Spectrum HDF5 reader                         #
# --------------------------------------------------------------------------- #
register_map_dataset_glob("PyMESS_2D", "*.h5")


@register_map_dataset_file_filter("PyMESS_2D")
def _pymess_2d_is_file(path: Path | str) -> bool:
    """Verify if ``path`` is a valid PyMESS 2D spectrum (Time-Domain or Frequency-Domain) HDF5 file."""
    from ..oneD.load import classify_pymess_h5

    return classify_pymess_h5(path) in ("2d_time_domain", "2d_freq_domain")


def _extract_pymess_2d_t2_ps(f, path: Path, meta_dict: dict[str, str], n_scans: int) -> np.ndarray:
    """Extract the waiting-time (t2) axis in picoseconds (ps) for a PyMESS 2D HDF5 file."""
    import re

    import h5py

    # 1. Explicit data/t2_delays_fs array
    if "data/t2_delays_fs" in f:
        arr = np.asarray(f["data/t2_delays_fs"][:], dtype=np.float64).ravel()
        if arr.size > 0 and not np.all(np.isnan(arr)):
            return arr / 1000.0

    # 2. Per-scan t2 attributes inside 2D_Spectra or scans
    for grp_name in ("2D_Spectra", "scans"):
        if grp_name in f and isinstance(f[grp_name], h5py.Group):
            scan_keys = sorted([k for k in f[grp_name].keys() if k.startswith("scan_")])
            if len(scan_keys) > 1:
                t2_vals = []
                for sk in scan_keys:
                    s_attrs = f[grp_name][sk].attrs
                    val_ps = None
                    for k_fs in ("t2_delay_fs", "waiting_time_fs", "delay_fs"):
                        if k_fs in s_attrs:
                            try:
                                val_ps = float(s_attrs[k_fs]) / 1000.0
                                break
                            except (ValueError, TypeError):
                                pass
                    if val_ps is None:
                        for k_ps in ("t2_delay_ps", "waiting_time_ps", "delay_ps", "stage_ir_pos_ps"):
                            if k_ps in s_attrs:
                                try:
                                    val_ps = float(s_attrs[k_ps])
                                    break
                                except (ValueError, TypeError):
                                    pass
                    if val_ps is not None:
                        t2_vals.append(val_ps)
                if len(t2_vals) == len(scan_keys):
                    return np.asarray(t2_vals, dtype=np.float64)

    # 3. Single-t2 metadata attributes
    for k_fs in ("t2_delay_fs", "waiting_time_fs", "t2_fs"):
        if k_fs in meta_dict:
            try:
                return np.full(n_scans, float(meta_dict[k_fs]) / 1000.0, dtype=np.float64)
            except (ValueError, TypeError):
                pass
    for k_ps in ("t2_delay_ps", "waiting_time_ps", "t2_ps", "stage_ir_pos_ps"):
        if k_ps in meta_dict:
            try:
                return np.full(n_scans, float(meta_dict[k_ps]), dtype=np.float64)
            except (ValueError, TypeError):
                pass

    # 4. Filename regex (e.g. sample_500fs.h5, sample_-1ps.h5)
    m = re.search(r"(?:^|[_\-\s])([+-]?\d+(?:\.\d+)?)\s*(fs|ps)(?:$|[_\-\s])", path.stem, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        unit = m.group(2).lower()
        val_ps = val / 1000.0 if unit == "fs" else val
        return np.full(n_scans, val_ps, dtype=np.float64)

    # 5. Parse last Stage move from log_notes if available
    log_notes = meta_dict.get("log_notes", "")
    if log_notes:
        matches = re.findall(
            r"Stage\s+\S+\s+moving\s+to\s+([+-]?\d+(?:\.\d+)?)\s*(ps|fs)",
            log_notes,
            re.IGNORECASE,
        )
        if matches:
            val_str, unit = matches[-1]
            try:
                val = float(val_str)
                val_ps = val / 1000.0 if unit.lower() == "fs" else val
                return np.full(n_scans, val_ps, dtype=np.float64)
            except ValueError:
                pass

    if n_scans > 1:
        return np.arange(n_scans, dtype=np.float64)
    return np.array([0.0], dtype=np.float64)


def _extract_pymess_2d_stack_from_group(
    grp: Any,
    is_freq_domain: bool,
    n_t1_or_w1_expected: int | None = None,
    n_pix_expected: int | None = None,
    n_t2_expected: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract a 3D stack ``(N_t1_or_w1, N_w3, N_t2)`` and optional 2D interferogram ``(N_t1, N_t2)`` from a group."""
    import h5py

    def _orient_3d_cube(arr_3d: np.ndarray) -> np.ndarray:
        # PyMESS new schema stores 3D cubes as (N_t2, N_t1_or_w1, N_pix);
        # PyMORGAN stores 3D cubes as (N_t1_or_w1, N_pix, N_t2).
        if (
            n_t1_or_w1_expected is not None
            and n_pix_expected is not None
            and arr_3d.shape[1] == n_t1_or_w1_expected
            and arr_3d.shape[2] == n_pix_expected
            and (
                arr_3d.shape[0] != n_t1_or_w1_expected
                or n_t2_expected is None
                or arr_3d.shape[0] == n_t2_expected
            )
        ):
            return np.transpose(arr_3d, (1, 2, 0))
        if (
            n_t1_or_w1_expected is not None
            and n_pix_expected is not None
            and arr_3d.shape[0] == n_t1_or_w1_expected
            and arr_3d.shape[1] == n_pix_expected
        ):
            return arr_3d
        if n_t1_or_w1_expected is not None:
            if arr_3d.shape[1] == n_t1_or_w1_expected and arr_3d.shape[0] != n_t1_or_w1_expected:
                return np.transpose(arr_3d, (1, 2, 0))
            if (
                n_t2_expected is not None
                and arr_3d.shape[0] == n_t2_expected
                and arr_3d.shape[1] == n_t1_or_w1_expected
            ):
                return np.transpose(arr_3d, (1, 2, 0))
        if (
            n_t2_expected is not None
            and n_pix_expected is not None
            and arr_3d.shape[0] == n_t2_expected
            and arr_3d.shape[2] == n_pix_expected
        ):
            return np.transpose(arr_3d, (1, 2, 0))
        return arr_3d

    def _orient_interf_2d(p_int: np.ndarray, n_t1: int, n_t2: int) -> np.ndarray:
        if p_int.ndim == 1:
            return np.repeat(p_int[:, np.newaxis], n_t2, axis=1)
        if p_int.ndim == 2:
            if p_int.shape == (n_t1, n_t2):
                return p_int
            if p_int.shape == (n_t2, n_t1):
                return p_int.T
            if p_int.shape[0] == n_t1:
                return p_int
            if p_int.shape[1] == n_t1:
                return p_int.T
        return p_int

    # 1. Check explicit 3D cubes first (e.g. data/surface_absorptive_cube, data/interferograms_cube)
    cube_mat = None
    if is_freq_domain and "surface_absorptive_cube" in grp:
        cube_mat = np.asarray(grp["surface_absorptive_cube"][:], dtype=np.float64)
    if (cube_mat is None or np.all(np.isnan(cube_mat))) and "interferograms_cube" in grp:
        cube_mat = np.asarray(grp["interferograms_cube"][:], dtype=np.float64)
    if (cube_mat is None or np.all(np.isnan(cube_mat))) and "surface_absorptive_cube" in grp:
        cube_mat = np.asarray(grp["surface_absorptive_cube"][:], dtype=np.float64)

    if cube_mat is not None and cube_mat.ndim == 3 and not np.all(np.isnan(cube_mat)):
        cube = _orient_3d_cube(cube_mat)

        interf_2d = None
        for ikey in ("pump_interferogram_cube", "pump_interferogram", "integrated_channel"):
            if ikey in grp:
                p_int = np.asarray(grp[ikey][:], dtype=np.float64)
                interf_2d = _orient_interf_2d(p_int, cube.shape[0], cube.shape[2])
                break
        if interf_2d is not None and np.all(np.isnan(interf_2d)):
            interf_2d = np.nanmean(cube, axis=1)
        return cube, interf_2d

    # 2. Check multi-scan subgroups (scan_001, scan_002, ...)
    scan_keys = sorted(
        [k for k in grp.keys() if k.startswith("scan_") and isinstance(grp[k], h5py.Group)]
    )
    if len(scan_keys) > 1:
        mats = []
        interfs = []
        for sk in scan_keys:
            sgrp = grp[sk]
            if is_freq_domain and "surface_absorptive" in sgrp:
                mat = np.asarray(sgrp["surface_absorptive"][:], dtype=np.float64)
            elif "interferograms_raw" in sgrp:
                mat = np.asarray(sgrp["interferograms_raw"][:], dtype=np.float64)
            elif "surface_absorptive" in sgrp:
                mat = np.asarray(sgrp["surface_absorptive"][:], dtype=np.float64)
            else:
                continue
            mats.append(mat)
            p_int_vec = None
            if "pump_interferogram" in sgrp:
                p_int_vec = np.asarray(sgrp["pump_interferogram"][:], dtype=np.float64).ravel()
            elif "integrated_channel" in sgrp:
                p_int_vec = np.asarray(sgrp["integrated_channel"][:], dtype=np.float64).ravel()
            if p_int_vec is None or np.all(np.isnan(p_int_vec)):
                p_int_vec = np.nanmean(mat, axis=1)
            interfs.append(p_int_vec)
        if mats:
            cube = np.stack(mats, axis=-1)
            interf_2d = np.stack(interfs, axis=-1) if interfs else None
            return cube, interf_2d

    # 3. Single scan in group or direct datasets
    target = grp[scan_keys[0]] if len(scan_keys) == 1 else grp
    mat = None
    if is_freq_domain and "surface_absorptive" in target:
        mat = np.asarray(target["surface_absorptive"][:], dtype=np.float64)
    if (mat is None or np.all(np.isnan(mat))) and "interferograms_raw" in target:
        mat = np.asarray(target["interferograms_raw"][:], dtype=np.float64)
    if (
        (mat is None or np.all(np.isnan(mat)))
        and "slow_modulation" in target
        and isinstance(target["slow_modulation"], h5py.Group)
    ):
        sm_grp = target["slow_modulation"]
        sm_st_keys = sorted(
            [
                k
                for k in sm_grp.keys()
                if k.startswith("state_")
                and not k.endswith("_individual_scans")
                and isinstance(sm_grp[k], h5py.Dataset)
            ]
        )
        if sm_st_keys:
            mat = np.asarray(sm_grp[sm_st_keys[0]][:], dtype=np.float64)
        else:
            for fallback_k in (
                "surface_parallel",
                "delta_A_parallel",
                "isotropic",
                "surface_isotropic",
            ):
                if fallback_k in sm_grp and isinstance(sm_grp[fallback_k], h5py.Dataset):
                    mat = np.asarray(sm_grp[fallback_k][:], dtype=np.float64)
                    break
    if (
        (mat is None or np.all(np.isnan(mat)))
        and "signals" in target
        and isinstance(target["signals"], h5py.Group)
    ):
        sig_grp = target["signals"]
        sig_keys = sorted(
            [
                k
                for k in sig_grp.keys()
                if not k.endswith("_noise") and isinstance(sig_grp[k], h5py.Dataset)
            ]
        )
        if sig_keys:
            mat = np.asarray(sig_grp[sig_keys[0]][:], dtype=np.float64)
    if (mat is None or np.all(np.isnan(mat))) and "surface_absorptive" in target:
        mat = np.asarray(target["surface_absorptive"][:], dtype=np.float64)
    if mat is None:
        raise ValueError("No 2D interferograms_raw or surface_absorptive dataset found in HDF5 group.")

    if mat.ndim == 2:
        cube = mat[:, :, np.newaxis]
    elif mat.ndim == 3:
        cube = _orient_3d_cube(mat)
    else:
        raise ValueError(f"Unexpected 2D dataset dimensionality: {mat.shape}")

    interf_2d = None
    for ikey in ("pump_interferogram_cube", "pump_interferogram", "integrated_channel"):
        if ikey in target:
            p_int = np.asarray(target[ikey][:], dtype=np.float64)
            interf_2d = _orient_interf_2d(p_int, cube.shape[0], cube.shape[2])
            break
    if interf_2d is not None and np.all(np.isnan(interf_2d)):
        interf_2d = np.nanmean(cube, axis=1)

    return cube, interf_2d


@register_map_loader("PyMESS_2D")
def read_PyMESS_2D(path: str, progress_tracker=None):
    """Read a PyMESS 2D spectrum (.h5) file (Time-Domain or Frequency-Domain)."""
    import re

    import h5py

    from ..oneD.load import (
        _detect_pymess_probe_units,
        _extract_pymess_metadata_bundle,
        _format_pymess_cal_source,
        _format_pymess_sample_info,
        _pymess_attr_str,
        _resolve_pymess_probe_axis,
        classify_pymess_h5,
    )

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"PyMESS 2D HDF5 file not found: {path}")

    subtype = classify_pymess_h5(p)
    if subtype not in ("2d_time_domain", "2d_freq_domain"):
        raise ValueError(
            f"File {p.name} is not a valid PyMESS 2D spectrum (detected type: {subtype})."
        )

    is_freq_domain = subtype == "2d_freq_domain"

    try:
        f = h5py.File(p, "r", swmr=True)
    except Exception:
        f = h5py.File(p, "r")

    with f:
        meta_dict, shaper_dict, fast_mod_dict, t2_n_batches = _extract_pymess_metadata_bundle(f)

        shaper_attrs: dict[str, Any] = {}
        if "metadata/shaper" in f:
            shaper_attrs = dict(f["metadata/shaper"].attrs)
        elif "2D_Spectra" in f:
            for sk in sorted(f["2D_Spectra"].keys()):
                sgrp = f["2D_Spectra"][sk]
                if isinstance(sgrp, h5py.Group) and "shaper" in sgrp:
                    shaper_attrs = dict(sgrp["shaper"].attrs)
                    break

        n_t1_or_w1_exp = None
        if is_freq_domain and "data/omega1_axis" in f and f["data/omega1_axis"].size > 1:
            n_t1_or_w1_exp = int(f["data/omega1_axis"].size)
        elif "data/t1_delays_fs" in f and f["data/t1_delays_fs"].size > 1:
            n_t1_or_w1_exp = int(f["data/t1_delays_fs"].size)
        n_pix_exp = None
        if "data/omega3_axis" in f and f["data/omega3_axis"].size > 1:
            n_pix_exp = int(f["data/omega3_axis"].size)
        n_t2_exp = None
        if "data/t2_delays_fs" in f and f["data/t2_delays_fs"].size > 0:
            n_t2_exp = int(f["data/t2_delays_fs"].size)

        # Extract primary 3D cube and optional pump interferogram:
        # Prioritize explicit 3D cubes in /data if present, then multi-scan groups
        if "data" in f and isinstance(f["data"], h5py.Group) and (
            "interferograms_cube" in f["data"] or "surface_absorptive_cube" in f["data"]
        ):
            raw_cube, interf_2d = _extract_pymess_2d_stack_from_group(
                f["data"], is_freq_domain, n_t1_or_w1_exp, n_pix_exp, n_t2_exp
            )
        elif "scans" in f and isinstance(f["scans"], h5py.Group) and any(
            k.startswith("scan_") for k in f["scans"].keys()
        ):
            raw_cube, interf_2d = _extract_pymess_2d_stack_from_group(
                f["scans"], is_freq_domain, n_t1_or_w1_exp, n_pix_exp, n_t2_exp
            )
        elif "2D_Spectra" in f and isinstance(f["2D_Spectra"], h5py.Group) and any(
            k.startswith("scan_") for k in f["2D_Spectra"].keys()
        ):
            raw_cube, interf_2d = _extract_pymess_2d_stack_from_group(
                f["2D_Spectra"], is_freq_domain, n_t1_or_w1_exp, n_pix_exp, n_t2_exp
            )
        elif "data" in f and isinstance(f["data"], h5py.Group):
            raw_cube, interf_2d = _extract_pymess_2d_stack_from_group(
                f["data"], is_freq_domain, n_t1_or_w1_exp, n_pix_exp, n_t2_exp
            )
        else:
            raise ValueError(f"File {p.name} contains no recognizable 2D data group.")

        # Check for slow modulation / polarization states:
        # 1. Datasets in /data/slow_modulation
        states_data: dict[str, np.ndarray] = {}
        states_raw_ss: dict[str, np.ndarray] = {}
        if "data/slow_modulation" in f and isinstance(f["data/slow_modulation"], h5py.Group):
            sm_grp = f["data/slow_modulation"]

            def _to_cube(arr: np.ndarray) -> np.ndarray:
                arr = np.asarray(arr, dtype=np.float64)
                if arr.ndim == 2:
                    return arr[:, :, np.newaxis]
                if arr.ndim == 3:
                    if arr.shape[:2] == raw_cube.shape[:2]:
                        return arr
                    if arr.shape[1:] == raw_cube.shape[:2]:
                        return np.transpose(arr, (1, 2, 0))
                return arr

            raw_names = meta_dict.get("slow_modulation_state_names", "")
            parsed_names = (
                [s.strip() for s in raw_names.split(",") if s.strip()]
                if raw_names
                else []
            )

            # 1a. New PyMESS schema: state_00_<name>, state_01_<name>, ...
            st_ds_keys = sorted(
                [
                    k
                    for k in sm_grp.keys()
                    if k.startswith("state_")
                    and not k.endswith("_individual_scans")
                    and isinstance(sm_grp[k], h5py.Dataset)
                ]
            )
            for idx, k in enumerate(st_ds_keys):
                c = _to_cube(sm_grp[k][:])
                if c.shape[:2] == raw_cube.shape[:2]:
                    if idx < len(parsed_names):
                        st_label = parsed_names[idx]
                    else:
                        st_label = re.sub(r"^state_\d+_", "", k) or k
                    states_data[st_label] = c
                    ind_k = f"{k}_individual_scans"
                    if ind_k in sm_grp and isinstance(sm_grp[ind_k], h5py.Dataset):
                        states_raw_ss[st_label] = np.asarray(sm_grp[ind_k][:], dtype=np.float64)

            # 1b. Legacy PyMESS schema or pre-computed isotropic/anisotropy datasets
            if is_freq_domain:
                pair_map = [
                    ("surface_parallel", "Parallel"),
                    ("surface_perpendicular", "Perpendicular"),
                    ("surface_isotropic", "Magic Angle"),
                    ("surface_anisotropy", "Anisotropy"),
                    ("isotropic", "Magic Angle"),
                    ("anisotropy", "Anisotropy"),
                ]
                prefix = "surface_"
            else:
                pair_map = [
                    ("delta_A_parallel", "Parallel"),
                    ("delta_A_perpendicular", "Perpendicular"),
                    ("isotropic", "Magic Angle"),
                ]
                prefix = "state_"

            for ds_key, label in pair_map:
                if label not in states_data and ds_key in sm_grp and isinstance(sm_grp[ds_key], h5py.Dataset):
                    c = _to_cube(sm_grp[ds_key][:])
                    if c.shape[:2] == raw_cube.shape[:2]:
                        states_data[label] = c

            for k in sorted(sm_grp.keys()):
                if (
                    k.startswith(prefix)
                    and not k.endswith("_individual_scans")
                    and k not in dict(pair_map)
                    and k not in st_ds_keys
                    and isinstance(sm_grp[k], h5py.Dataset)
                ):
                    c = _to_cube(sm_grp[k][:])
                    if c.shape[:2] == raw_cube.shape[:2]:
                        states_data[k] = c

        # 2. Subgroups state_0, state_1, ... in /scans, /data, or /2D_Spectra/scan_001
        if not states_data:
            for parent_key in ("scans", "data", "2D_Spectra/scan_001"):
                if parent_key in f and isinstance(f[parent_key], h5py.Group):
                    pgrp = f[parent_key]
                    st_keys = sorted(
                        [
                            k
                            for k in pgrp.keys()
                            if k.startswith("state_") and isinstance(pgrp[k], h5py.Group)
                        ]
                    )
                    if len(st_keys) >= 2:
                        raw_names = meta_dict.get("slow_modulation_state_names", "")
                        parsed_names = (
                            [s.strip() for s in raw_names.split(",") if s.strip()]
                            if raw_names
                            else []
                        )
                        for idx, sk in enumerate(st_keys):
                            sgrp = pgrp[sk]
                            st_cube, _ = _extract_pymess_2d_stack_from_group(
                                sgrp, is_freq_domain, n_t1_or_w1_exp, n_pix_exp, n_t2_exp
                            )
                            if idx < len(parsed_names):
                                st_label = parsed_names[idx]
                            else:
                                st_label = _pymess_attr_str(sgrp.attrs.get("state_name", sk))
                            states_data[st_label] = st_cube
                        break

        # 3. Multi-signal demodulated channels in /data/signals (when slow modulation is not active)
        if not states_data and "data/signals" in f and isinstance(f["data/signals"], h5py.Group):
            sig_grp = f["data/signals"]
            sig_keys = [
                k
                for k in sig_grp.keys()
                if not k.endswith("_noise")
                and isinstance(sig_grp[k], h5py.Dataset)
                and "label" in sig_grp[k].attrs
            ]
            if len(sig_keys) >= 2:
                for sk in sig_keys:
                    arr_s = np.asarray(sig_grp[sk][:], dtype=np.float64)
                    if arr_s.ndim == 2:
                        c = arr_s[:, :, np.newaxis]
                    elif arr_s.ndim == 3 and arr_s.shape[:2] == raw_cube.shape[:2]:
                        c = arr_s
                    elif arr_s.ndim == 3 and arr_s.shape[1:] == raw_cube.shape[:2]:
                        c = np.transpose(arr_s, (1, 2, 0))
                    else:
                        continue
                    if c.shape[:2] == raw_cube.shape[:2]:
                        s_lbl = _pymess_attr_str(sig_grp[sk].attrs.get("label", sk))
                        states_data[s_lbl] = c

        n_t1_or_w1, n_pix, n_t2 = raw_cube.shape
        n_t2_orig = n_t2

        def _orient_indiv_4d(indiv_arr: np.ndarray) -> np.ndarray | None:
            if indiv_arr.ndim == 4:
                # Commonly (N_scans, N_t2, N_t1_or_w1, N_pix) or (N_t1_or_w1, N_pix, N_t2, N_scans)
                if indiv_arr.shape[1:] == (n_t2_orig, n_t1_or_w1, n_pix):
                    return np.transpose(indiv_arr, (2, 3, 1, 0))
                if indiv_arr.shape[:3] == (n_t1_or_w1, n_pix, n_t2_orig):
                    return indiv_arr
                if indiv_arr.shape[2:] == (n_t1_or_w1, n_pix) and indiv_arr.shape[0] == n_t2_orig:
                    return np.transpose(indiv_arr, (2, 3, 0, 1))
            elif indiv_arr.ndim == 3 and n_t2_orig == 1:
                if indiv_arr.shape[1:] == (n_t1_or_w1, n_pix):
                    return np.transpose(indiv_arr, (1, 2, 0))[:, :, np.newaxis, :]
                if indiv_arr.shape[:2] == (n_t1_or_w1, n_pix):
                    return indiv_arr[:, :, np.newaxis, :]
            return None

        # Extract optional per-scan 4D cube (N_t1_or_w1, N_pix, N_t2_orig, N_scans)
        raw_signal_ss: np.ndarray | None = None
        raw_interferogram_ss: np.ndarray | None = None
        states_signal_ss: dict[str, np.ndarray] = {}
        if "data/individual_scans" in f and isinstance(f["data/individual_scans"], h5py.Dataset):
            raw_signal_ss = _orient_indiv_4d(np.asarray(f["data/individual_scans"][:], dtype=np.float64))

        for st_lbl, raw_4d in states_raw_ss.items():
            oriented_4d = _orient_indiv_4d(raw_4d)
            if oriented_4d is not None:
                states_signal_ss[st_lbl] = oriented_4d

        if raw_signal_ss is None:
            for grp_name in ("2D_Spectra", "scans"):
                if grp_name in f and isinstance(f[grp_name], h5py.Group):
                    sgrp_parent = f[grp_name]
                    s_keys = sorted(
                        [k for k in sgrp_parent.keys() if k.startswith("scan_") and isinstance(sgrp_parent[k], h5py.Group)]
                    )
                    if len(s_keys) > 1:
                        by_scan_t2: dict[tuple[int, int], tuple[np.ndarray, np.ndarray | None]] = {}
                        scan_id_set: set[int] = set()
                        t2_id_set: set[int] = set()
                        for sk in s_keys:
                            sg = sgrp_parent[sk]
                            if "scan_idx" not in sg.attrs:
                                continue
                            try:
                                s_id = int(sg.attrs["scan_idx"])
                                t_id = int(sg.attrs.get("t2_idx", 0))
                            except (ValueError, TypeError):
                                continue
                            mat_s = None
                            if is_freq_domain and "surface_absorptive" in sg:
                                mat_s = np.asarray(sg["surface_absorptive"][:], dtype=np.float64)
                            elif "interferograms_raw" in sg:
                                mat_s = np.asarray(sg["interferograms_raw"][:], dtype=np.float64)
                            elif "surface_absorptive" in sg:
                                mat_s = np.asarray(sg["surface_absorptive"][:], dtype=np.float64)
                            if mat_s is None or mat_s.shape != (n_t1_or_w1, n_pix):
                                continue
                            p_int_s = None
                            if "pump_interferogram" in sg:
                                p_int_s = np.asarray(sg["pump_interferogram"][:], dtype=np.float64).ravel()
                            elif "integrated_channel" in sg:
                                p_int_s = np.asarray(sg["integrated_channel"][:], dtype=np.float64).ravel()
                            by_scan_t2[(s_id, t_id)] = (mat_s, p_int_s)
                            scan_id_set.add(s_id)
                            t2_id_set.add(t_id)
                        sorted_s_ids = sorted(scan_id_set)
                        sorted_t_ids = sorted(t2_id_set)
                        if len(sorted_s_ids) > 1 and len(sorted_t_ids) == n_t2_orig:
                            sig_4d = np.full(
                                (n_t1_or_w1, n_pix, n_t2_orig, len(sorted_s_ids)), np.nan, dtype=np.float64
                            )
                            int_3d = np.full(
                                (n_t1_or_w1, n_t2_orig, len(sorted_s_ids)), np.nan, dtype=np.float64
                            )
                            has_int_s = False
                            for s_pos, s_id in enumerate(sorted_s_ids):
                                for t_pos, t_id in enumerate(sorted_t_ids):
                                    entry = by_scan_t2.get((s_id, t_id))
                                    if entry is not None:
                                        sig_4d[:, :, t_pos, s_pos] = entry[0]
                                        if entry[1] is not None and len(entry[1]) == n_t1_or_w1 and not np.all(np.isnan(entry[1])):
                                            int_3d[:, t_pos, s_pos] = entry[1]
                                            has_int_s = True
                                        else:
                                            int_3d[:, t_pos, s_pos] = np.nanmean(entry[0], axis=1)
                            raw_signal_ss = sig_4d
                            if has_int_s:
                                raw_interferogram_ss = int_3d
                            break

        # Resolve probe axis (omega3)
        probe, cal_kind, cal_loc, cal_level = _resolve_pymess_probe_axis(
            f, p, n_pix, prefer_omega3=True, default_zero_based=False
        )
        _unitsL, display_L = _detect_pymess_probe_units(probe, meta_dict)
        freq_units = display_L if display_L in ("cm-1", "nm") else "cm-1"
        cal_source = _format_pymess_cal_source(cal_kind, cal_loc, display_L, _unitsL)

        # Resolve t2 waiting times in ps
        t2delays_ps = _extract_pymess_2d_t2_ps(f, p, meta_dict, n_t2)
        if len(t2delays_ps) != n_t2:
            if len(t2delays_ps) > n_t2:
                t2delays_ps = t2delays_ps[:n_t2]
            else:
                t2delays_ps = np.arange(n_t2, dtype=np.float64)

        # Filter out completely unmeasured (all-NaN) t2 slices if scan is in progress
        in_progress_t2 = False
        if n_t2 > 1:
            valid_t2_mask = ~np.all(np.isnan(raw_cube), axis=(0, 1))
            if np.any(valid_t2_mask) and not np.all(valid_t2_mask):
                in_progress_t2 = True
                raw_cube = raw_cube[:, :, valid_t2_mask]
                t2delays_ps = t2delays_ps[valid_t2_mask]
                if interf_2d is not None and interf_2d.shape[1] == n_t2:
                    interf_2d = interf_2d[:, valid_t2_mask]
                if raw_signal_ss is not None and raw_signal_ss.shape[2] == n_t2:
                    raw_signal_ss = raw_signal_ss[:, :, valid_t2_mask, :]
                if raw_interferogram_ss is not None and raw_interferogram_ss.shape[1] == n_t2:
                    raw_interferogram_ss = raw_interferogram_ss[:, valid_t2_mask, :]
                for st_k in list(states_data.keys()):
                    if states_data[st_k].shape[2] == n_t2:
                        states_data[st_k] = states_data[st_k][:, :, valid_t2_mask]
                for st_k in list(states_signal_ss.keys()):
                    if states_signal_ss[st_k].shape[2] == n_t2:
                        states_signal_ss[st_k] = states_signal_ss[st_k][:, :, valid_t2_mask, :]
                n_t2 = raw_cube.shape[2]

        # Sort by ascending t2
        if n_t2 > 1:
            sort_idx = np.argsort(t2delays_ps)
            t2delays_ps = t2delays_ps[sort_idx]
            raw_cube = raw_cube[:, :, sort_idx]
            if interf_2d is not None and interf_2d.shape[1] == n_t2:
                interf_2d = interf_2d[:, sort_idx]
            if raw_signal_ss is not None and raw_signal_ss.shape[2] == n_t2:
                raw_signal_ss = raw_signal_ss[:, :, sort_idx, :]
            if raw_interferogram_ss is not None and raw_interferogram_ss.shape[1] == n_t2:
                raw_interferogram_ss = raw_interferogram_ss[:, sort_idx, :]
            for st_k in list(states_data.keys()):
                states_data[st_k] = states_data[st_k][:, :, sort_idx]
            for st_k in list(states_signal_ss.keys()):
                if states_signal_ss[st_k].shape[2] == n_t2:
                    states_signal_ss[st_k] = states_signal_ss[st_k][:, :, sort_idx, :]

        # Trim trailing all-NaN rows along axis 0 if scan was stopped early
        valid_rows = ~np.all(np.isnan(raw_cube[:, :, 0]), axis=1)
        in_progress = in_progress_t2 or (not bool(np.all(valid_rows)))
        cut_rows = n_t1_or_w1
        if not bool(np.all(valid_rows)):
            last_valid = np.where(valid_rows)[0]
            if len(last_valid) > 0:
                cut_rows = int(last_valid[-1]) + 1
                raw_cube = raw_cube[:cut_rows, :, :]
                if interf_2d is not None:
                    interf_2d = interf_2d[:cut_rows, :]
                if raw_signal_ss is not None:
                    raw_signal_ss = raw_signal_ss[:cut_rows, :, :, :]
                if raw_interferogram_ss is not None:
                    raw_interferogram_ss = raw_interferogram_ss[:cut_rows, :, :]
                for st_k in list(states_data.keys()):
                    states_data[st_k] = states_data[st_k][:cut_rows, :, :]
                for st_k in list(states_signal_ss.keys()):
                    states_signal_ss[st_k] = states_signal_ss[st_k][:cut_rows, :, :, :]

        for st_k in list(states_signal_ss.keys()):
            st_ss = states_signal_ss[st_k]
            v_mask = ~np.all(np.isnan(st_ss), axis=(0, 1, 2))
            if np.any(v_mask):
                st_ss = st_ss[:, :, :, v_mask]
            if st_ss.shape[-1] <= 1:
                del states_signal_ss[st_k]
            else:
                states_signal_ss[st_k] = st_ss

        if raw_signal_ss is not None:
            valid_scan_mask = ~np.all(np.isnan(raw_signal_ss), axis=(0, 1, 2))
            if np.any(valid_scan_mask):
                raw_signal_ss = raw_signal_ss[:, :, :, valid_scan_mask]
                if raw_interferogram_ss is not None and raw_interferogram_ss.shape[-1] == len(valid_scan_mask):
                    raw_interferogram_ss = raw_interferogram_ss[:, :, valid_scan_mask]
            if raw_signal_ss.shape[-1] <= 1:
                raw_signal_ss = None
                raw_interferogram_ss = None

        # If 2 polarization states exist, compute Magic Angle and Anisotropy cubes
        if len(states_data) >= 2:
            st_keys_list = list(states_data.keys())
            par_key, perp_key = None, None
            for k in st_keys_list:
                kl = k.lower()
                if "par" in kl and par_key is None:
                    par_key = k
                elif "perp" in kl and perp_key is None:
                    perp_key = k
            if par_key is None or perp_key is None:
                par_key, perp_key = st_keys_list[0], st_keys_list[1]

            g_factor = 1.0
            for g_key in ("g_factor", "G_factor", "gfactor"):
                if g_key in meta_dict:
                    try:
                        g_factor = float(meta_dict[g_key])
                        break
                    except (ValueError, TypeError):
                        pass

            if ("par" in par_key.lower() and "perp" in perp_key.lower()) or len(st_keys_list) == 2:
                c_par = states_data[par_key]
                c_perp = states_data[perp_key]
                c_ma = (c_par + 2.0 * g_factor * c_perp) / 3.0
                if "Magic Angle" not in states_data:
                    states_data["Magic Angle"] = c_ma
                if is_freq_domain and "Anisotropy" not in states_data:
                    denom = c_par + 2.0 * g_factor * c_perp
                    with np.errstate(divide="ignore", invalid="ignore"):
                        c_aniso = np.where(np.abs(denom) > 1e-6, (c_par - g_factor * c_perp) / denom, np.nan)
                    states_data["Anisotropy"] = c_aniso

                if par_key in states_signal_ss and perp_key in states_signal_ss:
                    ss_par = states_signal_ss[par_key]
                    ss_perp = states_signal_ss[perp_key]
                    n_sc_min = min(ss_par.shape[-1], ss_perp.shape[-1])
                    if n_sc_min > 1 and ss_par.shape[:3] == ss_perp.shape[:3]:
                        ss_ma = (ss_par[..., :n_sc_min] + 2.0 * g_factor * ss_perp[..., :n_sc_min]) / 3.0
                        states_signal_ss["Magic Angle"] = ss_ma
                        if is_freq_domain:
                            denom_ss = ss_par[..., :n_sc_min] + 2.0 * g_factor * ss_perp[..., :n_sc_min]
                            with np.errstate(divide="ignore", invalid="ignore"):
                                ss_aniso = np.where(
                                    np.abs(denom_ss) > 1e-6,
                                    (ss_par[..., :n_sc_min] - g_factor * ss_perp[..., :n_sc_min]) / denom_ss,
                                    np.nan,
                                )
                            states_signal_ss["Anisotropy"] = ss_aniso

        if raw_signal_ss is None and states_signal_ss:
            first_st = list(states_data.keys())[0] if states_data else list(states_signal_ss.keys())[0]
            raw_signal_ss = states_signal_ss.get(first_st)

        w0_cm1 = 0.0
        dt1_fs = 0.0

        if is_freq_domain:
            # Frequency-Domain 2D: axis 0 is already pump wavenumber (omega1)
            if "data/omega1_axis" in f:
                w1 = np.asarray(f["data/omega1_axis"][:], dtype=np.float64).ravel()
            elif "data/t1_delays_fs" in f:
                w1 = np.asarray(f["data/t1_delays_fs"][:], dtype=np.float64).ravel()
            else:
                w1 = np.arange(cut_rows, dtype=np.float64)
            w1 = w1[:cut_rows]

            Z = raw_cube.copy()
            pump = w1.copy()
            raw_signal = raw_cube.copy()
            raw_interferogram = (
                interf_2d[:cut_rows, :]
                if interf_2d is not None and interf_2d.shape[0] >= cut_rows
                else np.zeros((cut_rows, n_t2), dtype=np.float64)
            )
            raw_t1delays = w1.copy()
            datatype_str = "FreqDomain"
        else:
            # Time-Domain 2D: axis 0 of raw_cube is t1 delay in fs
            if "data/t1_delays_fs" in f:
                t1 = np.asarray(f["data/t1_delays_fs"][:], dtype=np.float64).ravel()[:cut_rows]
            else:
                t1 = np.arange(cut_rows, dtype=np.float64)

            # Determine rotating frame frequency (w0_cm1):
            # Prioritize effective_rotating_frame_cm1 (= scanned - fixed)
            if "effective_rotating_frame_cm1" in shaper_attrs:
                try:
                    w0_cm1 = float(shaper_attrs["effective_rotating_frame_cm1"])
                except (ValueError, TypeError):
                    w0_cm1 = 0.0
            elif "rotating_frame_cm1" in shaper_attrs:
                try:
                    w0_cm1 = float(shaper_attrs["rotating_frame_cm1"])
                except (ValueError, TypeError):
                    w0_cm1 = 0.0
            elif (
                "rotating_frame_scanned_cm1" in shaper_attrs
                and "rotating_frame_fixed_cm1" in shaper_attrs
            ):
                try:
                    w0_cm1 = float(shaper_attrs["rotating_frame_scanned_cm1"]) - float(
                        shaper_attrs["rotating_frame_fixed_cm1"]
                    )
                except (ValueError, TypeError):
                    w0_cm1 = 0.0
            elif "rotating_frame_scanned_cm1" in shaper_attrs:
                try:
                    w0_cm1 = float(shaper_attrs["rotating_frame_scanned_cm1"])
                except (ValueError, TypeError):
                    w0_cm1 = 0.0
            elif "rotating_frame_cm1" in meta_dict:
                try:
                    w0_cm1 = float(meta_dict["rotating_frame_cm1"])
                except (ValueError, TypeError):
                    w0_cm1 = 0.0

            # Fallback if metadata/shaper was absent (e.g. scan stopped early):
            # read the starting wavenumber of data/omega1_axis
            if w0_cm1 == 0.0 and "data/omega1_axis" in f:
                w1_saved = np.asarray(f["data/omega1_axis"]).squeeze()
                if w1_saved.ndim == 1 and w1_saved.size > 1 and not np.all(np.isnan(w1_saved)):
                    w0_cm1 = float(w1_saved[0])

            if "step_fs" in shaper_attrs:
                try:
                    dt1_fs = float(shaper_attrs["step_fs"])
                except (ValueError, TypeError):
                    dt1_fs = 0.0
            if dt1_fs <= 0.0 and len(t1) > 1:
                dt1_fs = float(np.nanmean(np.abs(np.diff(t1))))

            raw_signal = raw_cube.copy()
            if interf_2d is not None and interf_2d.shape[0] >= cut_rows:
                raw_interferogram = interf_2d[:cut_rows, :].copy()
            elif "data/integrated_channel" in f:
                ic = np.asarray(f["data/integrated_channel"][:], dtype=np.float64).ravel()[:cut_rows]
                raw_interferogram = np.repeat(ic[:, np.newaxis], n_t2, axis=1)
            else:
                raw_interferogram = np.nanmean(raw_signal, axis=1)
            raw_t1delays = t1.copy()
            datatype_str = "shaper"

            # Provide initial frequency-domain Z and pump (used before/unless process() is called)
            if (
                "data/surface_absorptive" in f
                and "data/omega1_axis" in f
                and n_t2 == 1
            ):
                surf = np.asarray(f["data/surface_absorptive"][:], dtype=np.float64)
                w1_saved = np.asarray(f["data/omega1_axis"][:], dtype=np.float64).ravel()
                if (
                    surf.ndim == 2
                    and w1_saved.ndim == 1
                    and surf.shape == (len(w1_saved), n_pix)
                    and not np.all(np.isnan(surf))
                ):
                    Z = surf[:, :, np.newaxis]
                    pump = w1_saved
                else:
                    Z = raw_signal.copy()
                    pump = t1.copy()
            else:
                Z = raw_signal.copy()
                pump = t1.copy()

        units = _units_2d("ps", freq_units)
        if freq_units == "nm":
            units["unitsL_lbl"] = "Wavelength"
            units["unitsL_ltx"] = "nm"
        units["cal_source"] = cal_source
        if cal_level is not None:
            units["cal_level"] = cal_level
        units["is_frequency_domain"] = is_freq_domain
        units["w0_cm1"] = w0_cm1
        units["dt1_fs"] = dt1_fs
        units["pymess_metadata"] = meta_dict
        units["pymess_subtype"] = subtype
        if raw_signal_ss is not None:
            units["raw_signal_ss"] = raw_signal_ss
            units["raw_interferogram_ss"] = raw_interferogram_ss
            units["nscans"] = int(raw_signal_ss.shape[-1])
        if states_signal_ss:
            units["states_signal_ss"] = states_signal_ss
        if states_data:
            units["state_names"] = list(states_data.keys())
            units["states_data"] = states_data
            units["active_state"] = list(states_data.keys())[0]

        compact_info, full_info = _format_pymess_sample_info(
            meta_dict,
            mode="2d_freq_domain" if is_freq_domain else "2d_time_domain",
            shaper_attrs=shaper_dict,
            fast_mod_attrs=fast_mod_dict,
            t2_n_batches=t2_n_batches,
            state_names=list(states_data.keys()) if states_data else None,
            w0_cm1=None if is_freq_domain else w0_cm1,
            n_t2=len(t2delays_ps),
        )
        units["sample_info"] = compact_info
        if full_info:
            units["sample_info_full"] = full_info

    return (
        Z,
        pump,
        probe,
        t2delays_ps,
        units,
        freq_units,
        datatype_str,
        in_progress,
        raw_signal,
        raw_interferogram,
        raw_t1delays,
    )


# --------------------------------------------------------------------------- #
#                     PhaseTech QuickControl 2D reader                        #
# --------------------------------------------------------------------------- #
register_map_dataset_glob("QuickControl_2D", "*.info")


@register_map_dataset_file_filter("QuickControl_2D")
def _quickcontrol_2d_is_file(path: Path | str) -> tuple[bool, bool]:
    """Return ``(is_candidate, is_valid)`` for a QuickControl 2D ``.info`` dataset."""
    from ..oneD.load import classify_quickcontrol_info

    kind, has_scans = classify_quickcontrol_info(path)
    if kind != "2d":
        return False, False
    return True, has_scans


def _extract_quickcontrol_2d_t2_ps(
    info_path: Path,
    info: dict[str, str],
    t_indices: list[int],
    *,
    footer_t2_raw: dict[int, float] | None = None,
) -> tuple[np.ndarray, bool]:
    """Resolve 2D population times in picoseconds and detect partial acquisitions.

    Priority:
    1. ``<prefix>.wts`` encoder waiting-time column (in ``Waiting Time Delay Units``).
    2. Per-waiting-time ``.scan`` footer encoder timestamps when ``.wts`` is absent.
    3. ``Waiting Time Delays`` or ``Waiting Time Delay List`` schedule in ``.info``.
    4. ``Waiting Time (fs)`` single float in ``.info``.
    5. Zero-based index fallback.
    """
    wt_unit = info.get("Waiting Time Delay Units", "fs").strip().lower()
    if wt_unit == "fs":
        scale_to_ps = 1.0 / 1000.0
    elif wt_unit == "ns":
        scale_to_ps = 1000.0
    else:
        scale_to_ps = 1.0

    sched_ps: np.ndarray | None = None
    raw_sched = info.get("Waiting Time Delays") or info.get("Waiting Time Delay List") or ""
    if raw_sched.strip():
        try:
            parts = [float(x.strip()) for x in raw_sched.split(",") if x.strip()]
            if parts:
                sched_ps = np.asarray(parts, dtype=np.float64) * scale_to_ps
        except ValueError:
            sched_ps = None

    full_t2_ps: np.ndarray | None = None
    wts_path = info_path.parent / f"{info_path.stem}.wts"
    if wts_path.is_file():
        try:
            arr = np.loadtxt(wts_path, dtype=np.float64).ravel()
            if arr.size > 0 and np.all(np.isfinite(arr)):
                full_t2_ps = arr * scale_to_ps
        except Exception:
            full_t2_ps = None

    if full_t2_ps is not None and full_t2_ps.size > 0:
        expected_len = max(full_t2_ps.size, sched_ps.size if sched_ps is not None else 0)
        in_progress = len(t_indices) < expected_len
        mapped = [
            float(full_t2_ps[t_idx - 1])
            if 0 <= (t_idx - 1) < full_t2_ps.size
            else (
                float(sched_ps[t_idx - 1])
                if sched_ps is not None and 0 <= (t_idx - 1) < sched_ps.size
                else float(t_idx - 1)
            )
            for t_idx in t_indices
        ]
        return np.asarray(mapped, dtype=np.float64), in_progress

    if footer_t2_raw:
        f_vals = [footer_t2_raw.get(t_idx, float("nan")) for t_idx in t_indices]
        f_arr = np.asarray(f_vals, dtype=np.float64)
        if np.all(np.isfinite(f_arr)) and not np.allclose(f_arr, 0.0, atol=1e-6):
            in_progress = bool(sched_ps is not None and len(t_indices) < sched_ps.size)
            return f_arr * scale_to_ps, in_progress

    if sched_ps is not None and sched_ps.size > 0:
        in_progress = len(t_indices) < sched_ps.size
        mapped = [
            float(sched_ps[t_idx - 1])
            if 0 <= (t_idx - 1) < sched_ps.size
            else float(t_idx - 1)
            for t_idx in t_indices
        ]
        return np.asarray(mapped, dtype=np.float64), in_progress

    if info.get("Waiting Time (fs)"):
        try:
            wt_fs = float(info["Waiting Time (fs)"])
            full_t2_ps = np.asarray([wt_fs / 1000.0], dtype=np.float64)
            return (
                np.asarray(
                    [
                        float(full_t2_ps[t_idx - 1])
                        if 0 <= (t_idx - 1) < full_t2_ps.size
                        else float(t_idx - 1)
                        for t_idx in t_indices
                    ],
                    dtype=np.float64,
                ),
                False,
            )
        except ValueError:
            pass

    return np.asarray([float(t_idx - 1) for t_idx in t_indices], dtype=np.float64), False


def _unfold_undersampled_shaper_w0(
    w0_rf_cm1: float,
    dt1_fs: float,
    info: dict[str, str],
    probe_cm1: np.ndarray,
    raw_signal: np.ndarray,
    probe_spectrum: np.ndarray | None = None,
) -> tuple[float, bool]:
    """Unfold an undersampled pulse-shaper rotating-frame offset into its optical Nyquist zone.

    Returns ``(w0_effective_cm1, nyquist_invert)``. When ``w0_rf_cm1 + f_Nyquist``
    already reaches ``min(probe_cm1)`` (e.g. standard mid-IR or fully shifted
    rotating-frame acquisitions), ``(w0_rf_cm1, False)`` is returned unchanged.
    """
    c_0 = 2.99792458e-5  # cm/fs
    thz_to_cm1 = 33.356409519815204
    if dt1_fs <= 0.0 or probe_cm1.size == 0:
        return w0_rf_cm1, False

    f_s = 1.0 / (c_0 * dt1_fs)
    f_nyq = 0.5 * f_s
    probe_min = float(np.nanmin(probe_cm1))
    probe_max = float(np.nanmax(probe_cm1))
    if not np.isfinite(probe_min) or not np.isfinite(probe_max) or w0_rf_cm1 + f_nyq >= probe_min:
        return w0_rf_cm1, False

    band_lo = probe_min
    band_hi = probe_max
    band_centre = 0.5 * (probe_min + probe_max)

    if probe_spectrum is not None and probe_spectrum.size == probe_cm1.size:
        sp_max = float(np.nanmax(probe_spectrum))
        if np.isfinite(sp_max) and sp_max > 0.0:
            active_mask = probe_spectrum >= 0.02 * sp_max
            if np.any(active_mask):
                band_lo = float(np.nanmin(probe_cm1[active_mask]))
                band_hi = float(np.nanmax(probe_cm1[active_mask]))
                band_centre = 0.5 * (band_lo + band_hi)

    try:
        p1 = float(info.get("Shaper Calibration P1", "nan"))
        p2 = float(info.get("Shaper Calibration P2", "nan"))
        p3 = float(info.get("Shaper Calibration P3", "nan"))
        mask_len = float(info.get("AWG Mask Length", "10000") or 10000.0)
    except ValueError:
        p1 = p2 = p3 = float("nan")
        mask_len = 10000.0

    if np.all(np.isfinite([p1, p2, p3])) and p3 > 0.0 and mask_len > 0.0:
        x_grid = np.linspace(0.0, mask_len, 512)
        w_shaper = (p1 * x_grid**2 + p2 * x_grid + p3) * thz_to_cm1
        sh_lo = float(np.nanmin(w_shaper))
        sh_hi = float(np.nanmax(w_shaper))
        if sh_hi > sh_lo > 0.0 and sh_hi >= probe_min and sh_lo <= probe_max:
            band_lo = max(band_lo, sh_lo)
            band_hi = min(band_hi, sh_hi)
            if band_lo >= band_hi:
                band_lo, band_hi = max(probe_min, sh_lo), min(probe_max, sh_hi)
            band_centre = 0.5 * (sh_lo + sh_hi)

    n_bins = int(raw_signal.shape[0])
    n_pos = max(1, n_bins // 2)
    res_m = 1.0 / (max(1, n_bins) * dt1_fs * c_0)
    with np.errstate(invalid="ignore"):
        sig_zm = raw_signal - np.nanmean(raw_signal, axis=0, keepdims=True)
        sig_zm = np.nan_to_num(sig_zm, nan=0.0)
        ft_mag = np.mean(np.abs(np.fft.fft(sig_zm, axis=0)), axis=2)
    diag_tol = max(200.0, 2.5 * res_m)

    def _score_zone(pump_cand: np.ndarray, amp_half: np.ndarray) -> float:
        z_lo = float(pump_cand[0])
        z_hi = float(pump_cand[-1])
        ov = max(0.0, min(z_hi, band_hi) - max(z_lo, band_lo))
        if ov <= 0.0:
            return -1.0
        in_band = (probe_cm1 >= band_lo) & (probe_cm1 <= band_hi)
        if not np.any(in_band):
            in_band = np.ones_like(probe_cm1, dtype=bool)
        pr_sub = probe_cm1[in_band]
        amp_sub = amp_half[:, in_band]
        diff = np.abs(pump_cand[:, np.newaxis] - pr_sub[np.newaxis, :])
        mask = diff <= diag_tol
        if np.any(mask):
            diag_weights = 1.0 - 0.5 * (diff[mask] / diag_tol) ** 2
            diag_energy = float(np.sum(amp_sub[mask] * diag_weights))
        else:
            diag_energy = 0.0
        centre_prox = 1.0 / (1.0 + abs(0.5 * (z_lo + z_hi) - band_centre) / max(1.0, f_nyq))
        return diag_energy * (0.5 + 0.5 * centre_prox) + 1e-6 * ov * centre_prox

    m_min = max(1, int(np.floor((band_lo - w0_rf_cm1 - f_nyq) / f_s)))
    m_max = max(m_min, int(np.ceil((band_hi - w0_rf_cm1 + f_nyq) / f_s)) + 1)

    best_w0 = w0_rf_cm1
    best_inv = False
    best_score = -1.0

    f_pos = np.arange(n_pos, dtype=np.float64) * res_m
    amp_pos = ft_mag[:n_pos, :]
    for m in range(m_min, m_max + 2):
        w0_cand = w0_rf_cm1 + m * f_s
        sc = _score_zone(w0_cand + f_pos, amp_pos)
        if sc > best_score:
            best_score = sc
            best_w0 = w0_cand
            best_inv = False

    if n_bins >= 4:
        f_neg = np.arange(n_bins - n_pos, n_bins, dtype=np.float64) * res_m
        amp_neg = ft_mag[n_bins - n_pos :, :]
        for m in range(m_min, m_max + 2):
            w0_cand = w0_rf_cm1 + (m - 1) * f_s
            sc = _score_zone(w0_cand + f_neg, amp_neg)
            if sc > best_score:
                best_score = sc
                best_w0 = w0_cand
                best_inv = True

    return best_w0, best_inv


@register_map_loader("QuickControl_2D")
def read_QuickControl_2D(path: str, progress_tracker=None):
    """Read a PhaseTech QuickControl 2D spectrum dataset (``.info`` + ``.scan`` + ``.wts``).

    Supports both pulse-shaper time-domain acquisitions (``Domain = Time``) and
    frequency-swept acquisitions (``Domain = Frequency``). When both parallel
    (``_PAR#``) and perpendicular (``_PER#``) scan series are present, the
    isotropic Magic Angle state ``(PAR + 2 * PER) / 3`` is listed first and
    selected as the active state by default.
    """
    import warnings

    from ..oneD.load import (
        _expected_quickcontrol_axis0,
        _extract_quickcontrol_footer_probe_spectrum,
        _find_quickcontrol_2d_scans,
        _format_quickcontrol_sample_info,
        _pad_scan_stack_rows,
        _quickcontrol_exp_kind,
        _resolve_quickcontrol_info_path,
        _resolve_quickcontrol_probe_axis,
        _stack_quickcontrol_scans,
        parse_quickcontrol_info,
    )

    info_path = _resolve_quickcontrol_info_path(path, expected_kind="2d")
    info = parse_quickcontrol_info(info_path)
    kind = _quickcontrol_exp_kind(info)
    if kind == "1d":
        raise ValueError(
            f"File {info_path.name} is a 1D QuickControl dataset; use data_type='QuickControl_PP'."
        )
    if kind != "2d":
        raise ValueError(f"File {info_path.name} is not a valid QuickControl 2D .info header.")

    t_indices, scans_by_t = _find_quickcontrol_2d_scans(info_path)
    if not t_indices:
        raise ValueError(f"No 2D .scan files found for QuickControl dataset {info_path.stem!r}.")

    n_t2 = len(t_indices)
    expected_axis0 = _expected_quickcontrol_axis0(info, "2d")
    expected_rows = int(expected_axis0.size) if expected_axis0 is not None else None

    has_par_any = any(len(scans_by_t[t]["par"]) > 0 for t in t_indices)
    has_per_any = any(len(scans_by_t[t]["per"]) > 0 for t in t_indices)
    dual_pol = has_par_any and has_per_any

    axis0_ref: np.ndarray | None = None
    axis0_candidates: list[tuple[np.ndarray, bool]] = []
    n_rows_ref: int | None = expected_rows
    n_rows = 0
    n_pix = 0
    footer_t2_by_idx: dict[int, float] = {}
    footer_spec_parts: list[np.ndarray] = []
    has_partial_any = False
    in_progress = False

    def _select_axis0_ref(target_rows: int) -> np.ndarray:
        best_ax0: np.ndarray | None = None
        for ax_cand, is_part in axis0_candidates:
            if not is_part and len(ax_cand) >= target_rows:
                return ax_cand[:target_rows].copy()
        for ax_cand, _is_part in axis0_candidates:
            if best_ax0 is None or len(ax_cand) > len(best_ax0):
                best_ax0 = ax_cand
            if len(ax_cand) >= target_rows:
                return ax_cand[:target_rows].copy()
        assert best_ax0 is not None
        if expected_axis0 is not None and expected_axis0.size >= target_rows:
            out = expected_axis0[:target_rows].astype(np.float64, copy=True)
            n_cp = min(len(best_ax0), target_rows)
            out[:n_cp] = best_ax0[:n_cp]
            return out
        out = np.empty(target_rows, dtype=np.float64)
        n_cp = min(len(best_ax0), target_rows)
        out[:n_cp] = best_ax0[:n_cp]
        if n_cp < target_rows:
            step = float(np.median(np.diff(best_ax0))) if len(best_ax0) >= 2 else 1.0
            last = float(best_ax0[-1]) if len(best_ax0) > 0 else 0.0
            out[n_cp:] = last + step * np.arange(1, target_rows - n_cp + 1, dtype=np.float64)
        return out

    if dual_pol:
        par_stacks: list[np.ndarray | None] = []
        per_stacks: list[np.ndarray | None] = []
        max_par = 0
        max_per = 0

        for idx_m, t_idx in enumerate(t_indices):
            if progress_tracker is not None:
                total_steps = max(1, 2 * n_t2)
                if hasattr(progress_tracker, "update_split"):
                    progress_tracker.update_split(idx_m + 1, total_steps, "loading")
                else:
                    progress_tracker.update(idx_m + 1, label=f"Loading T{t_idx:02d}...")

            p_files = scans_by_t[t_idx]["par"]
            s_files = scans_by_t[t_idx]["per"]
            t_fcols: list[np.ndarray] = []
            if p_files:
                ax0, p_st, p_fc, p_fr, p_part = _stack_quickcontrol_scans(
                    p_files,
                    n_rows_ref=n_rows_ref,
                    n_pix_ref=n_pix or None,
                    expected_n_rows=expected_rows,
                    expected_axis0=expected_axis0,
                )
                if p_part:
                    has_partial_any = True
                if p_fc.size > 0 and np.any(np.isfinite(p_fc)):
                    t_fcols.append(p_fc[np.isfinite(p_fc)])
                if not footer_spec_parts and p_fr is not None:
                    footer_spec_parts.append(p_fr)
                axis0_candidates.append((ax0, p_part))
                if n_pix == 0:
                    n_pix = int(p_st.shape[1])
                par_stacks.append(p_st)
                max_par = max(max_par, int(p_st.shape[-1]))
            else:
                par_stacks.append(None)

            if s_files:
                ax0, s_st, s_fc, s_fr, s_part = _stack_quickcontrol_scans(
                    s_files,
                    n_rows_ref=n_rows_ref,
                    n_pix_ref=n_pix or None,
                    expected_n_rows=expected_rows,
                    expected_axis0=expected_axis0,
                )
                if s_part:
                    has_partial_any = True
                if s_fc.size > 0 and np.any(np.isfinite(s_fc)):
                    t_fcols.append(s_fc[np.isfinite(s_fc)])
                if not footer_spec_parts and s_fr is not None:
                    footer_spec_parts.append(s_fr)
                axis0_candidates.append((ax0, s_part))
                if n_pix == 0:
                    n_pix = int(s_st.shape[1])
                per_stacks.append(s_st)
                max_per = max(max_per, int(s_st.shape[-1]))
            else:
                per_stacks.append(None)

            if t_fcols:
                footer_t2_by_idx[t_idx] = float(np.mean(np.concatenate(t_fcols)))

        obs_rows = [int(st.shape[0]) for st in (*par_stacks, *per_stacks) if st is not None]
        n_rows = max(obs_rows) if obs_rows else 0
        if expected_rows is not None:
            n_rows = max(n_rows, expected_rows)
        assert n_rows > 0 and n_pix > 0 and axis0_candidates
        axis0_ref = _select_axis0_ref(n_rows)

        cube_par = np.full((n_rows, n_pix, n_t2), np.nan, dtype=np.float64)
        cube_per = np.full((n_rows, n_pix, n_t2), np.nan, dtype=np.float64)
        ss_par_full = np.full((n_rows, n_pix, n_t2, max(1, max_par)), np.nan, dtype=np.float64)
        ss_per_full = np.full((n_rows, n_pix, n_t2, max(1, max_per)), np.nan, dtype=np.float64)

        for m in range(n_t2):
            pst = par_stacks[m]
            sst = per_stacks[m]
            if pst is not None:
                if pst.shape[0] < n_rows:
                    pst = _pad_scan_stack_rows(pst, n_rows)
                    has_partial_any = True
                with warnings.catch_warnings(), np.errstate(invalid="ignore"):
                    warnings.simplefilter("ignore", RuntimeWarning)
                    cube_par[:, :, m] = np.nanmean(pst, axis=-1)
                ss_par_full[:, :, m, : pst.shape[-1]] = pst
                if pst.shape[-1] < max_par:
                    in_progress = True
            else:
                in_progress = True
            if sst is not None:
                if sst.shape[0] < n_rows:
                    sst = _pad_scan_stack_rows(sst, n_rows)
                    has_partial_any = True
                with warnings.catch_warnings(), np.errstate(invalid="ignore"):
                    warnings.simplefilter("ignore", RuntimeWarning)
                    cube_per[:, :, m] = np.nanmean(sst, axis=-1)
                ss_per_full[:, :, m, : sst.shape[-1]] = sst
                if sst.shape[-1] < max_per:
                    in_progress = True
            else:
                in_progress = True

        cube_par = np.nan_to_num(cube_par, nan=0.0)
        cube_per = np.nan_to_num(cube_per, nan=0.0)
        cube_ma = (cube_par + 2.0 * cube_per) / 3.0

        states_data: dict[str, np.ndarray] = {
            "Magic Angle": cube_ma,
            "Parallel": cube_par,
            "Perpendicular": cube_per,
        }

        max_common = min(max_par, max_per)
        states_signal_ss: dict[str, np.ndarray] = {}
        if max_common > 1:
            ss_ma = (ss_par_full[:, :, :, :max_common] + 2.0 * ss_per_full[:, :, :, :max_common]) / 3.0
            states_signal_ss["Magic Angle"] = ss_ma
        if max_par > 1:
            states_signal_ss["Parallel"] = ss_par_full
        if max_per > 1:
            states_signal_ss["Perpendicular"] = ss_per_full

        raw_cube = cube_ma.copy()
        raw_signal_ss = states_signal_ss.get("Magic Angle")
        n_scans_total = max(1, max_common)
    else:
        ch_stacks: list[np.ndarray] = []
        max_ch = 0
        if has_par_any:
            chan_key = "par"
        elif has_per_any:
            chan_key = "per"
        else:
            chan_key = "unpol"

        for idx_m, t_idx in enumerate(t_indices):
            if progress_tracker is not None:
                total_steps = max(1, 2 * n_t2)
                if hasattr(progress_tracker, "update_split"):
                    progress_tracker.update_split(idx_m + 1, total_steps, "loading")
                else:
                    progress_tracker.update(idx_m + 1, label=f"Loading T{t_idx:02d}...")

            c_files = scans_by_t[t_idx][chan_key]
            ax0, c_st, c_fc, c_fr, c_part = _stack_quickcontrol_scans(
                c_files,
                n_rows_ref=n_rows_ref,
                n_pix_ref=n_pix or None,
                expected_n_rows=expected_rows,
                expected_axis0=expected_axis0,
            )
            if c_part:
                has_partial_any = True
            if c_fc.size > 0 and np.any(np.isfinite(c_fc)):
                footer_t2_by_idx[t_idx] = float(np.mean(c_fc[np.isfinite(c_fc)]))
            if not footer_spec_parts and c_fr is not None:
                footer_spec_parts.append(c_fr)
            axis0_candidates.append((ax0, c_part))
            if n_pix == 0:
                n_pix = int(c_st.shape[1])
            ch_stacks.append(c_st)
            max_ch = max(max_ch, int(c_st.shape[-1]))

        obs_rows = [int(cst.shape[0]) for cst in ch_stacks]
        n_rows = max(obs_rows) if obs_rows else 0
        if expected_rows is not None:
            n_rows = max(n_rows, expected_rows)
        assert n_rows > 0 and n_pix > 0 and axis0_candidates
        axis0_ref = _select_axis0_ref(n_rows)

        raw_cube = np.zeros((n_rows, n_pix, n_t2), dtype=np.float64)
        ss_ch_full = np.full((n_rows, n_pix, n_t2, max(1, max_ch)), np.nan, dtype=np.float64)
        for m, cst in enumerate(ch_stacks):
            if cst.shape[0] < n_rows:
                cst = _pad_scan_stack_rows(cst, n_rows)
                has_partial_any = True
            with warnings.catch_warnings(), np.errstate(invalid="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                raw_cube[:, :, m] = np.nan_to_num(np.nanmean(cst, axis=-1), nan=0.0)
            ss_ch_full[:, :, m, : cst.shape[-1]] = cst
            if cst.shape[-1] < max_ch:
                in_progress = True

        states_data = {}
        states_signal_ss = {}
        raw_signal_ss = ss_ch_full if max_ch > 1 else None
        n_scans_total = max(1, max_ch)

    t2delays_ps, in_progress_t2 = _extract_quickcontrol_2d_t2_ps(
        info_path, info, t_indices, footer_t2_raw=footer_t2_by_idx
    )
    in_progress = in_progress or in_progress_t2 or has_partial_any

    if n_t2 > 1 and not np.all(np.diff(t2delays_ps) >= 0):
        sort_idx = np.argsort(t2delays_ps)
        t2delays_ps = t2delays_ps[sort_idx]
        raw_cube = raw_cube[:, :, sort_idx]
        if raw_signal_ss is not None:
            raw_signal_ss = raw_signal_ss[:, :, sort_idx, :]
        for k_st in list(states_data.keys()):
            states_data[k_st] = states_data[k_st][:, :, sort_idx]
        for k_st in list(states_signal_ss.keys()):
            states_signal_ss[k_st] = states_signal_ss[k_st][:, :, sort_idx, :]

    probe, _unitsL, display_L, cal_source, cal_level = _resolve_quickcontrol_probe_axis(
        info, n_pix, info_path, check_csv_override=True, force_wavenumbers=True
    )
    freq_units = "cm-1" if display_L in ("cm-1", "px") else display_L

    spec_arr: np.ndarray | None = None
    spec_first = info_path.parent / f"{info_path.stem}_{t_indices[0]:02d}.spec"
    if not spec_first.is_file():
        spec_first = info_path.parent / f"{info_path.stem}.spec"
    if spec_first.is_file():
        try:
            cand_spec = np.loadtxt(spec_first, dtype=np.float64).ravel()
            if cand_spec.size == n_pix:
                spec_arr = cand_spec
        except Exception:
            spec_arr = None
    if spec_arr is None and footer_spec_parts:
        spec_arr = _extract_quickcontrol_footer_probe_spectrum(
            np.concatenate(footer_spec_parts, axis=-1)
        )

    if probe.size > 1 and not np.all(np.diff(probe) >= 0):
        p_order = np.argsort(probe)
        probe = probe[p_order]
        raw_cube = raw_cube[:, p_order, :]
        if raw_signal_ss is not None:
            raw_signal_ss = raw_signal_ss[:, p_order, :, :]
        for k_st in list(states_data.keys()):
            states_data[k_st] = states_data[k_st][:, p_order, :]
        for k_st in list(states_signal_ss.keys()):
            states_signal_ss[k_st] = states_signal_ss[k_st][:, p_order, :, :]
        if spec_arr is not None and spec_arr.size == p_order.size:
            spec_arr = spec_arr[p_order]

    domain_str = info.get("Domain", "Time").strip().lower()
    is_freq_domain = domain_str.startswith("freq")
    nyquist_invert = False

    if is_freq_domain:
        pump = axis0_ref.copy()
        if (
            pump.size > 0
            and np.all(np.isfinite(pump))
            and np.all(pump > 0.0)
            and 150.0 <= float(np.nanmean(pump)) <= 2000.0
            and float(np.nanmax(probe)) > hlp.DISPLAY_SCALE_THRESHOLD
        ):
            pump = 1.0e7 / pump
        if pump.size > 1 and not np.all(np.diff(pump) >= 0):
            pu_order = np.argsort(pump)
            pump = pump[pu_order]
            raw_cube = raw_cube[pu_order, :, :]
            if raw_signal_ss is not None:
                raw_signal_ss = raw_signal_ss[pu_order, :, :, :]
            for k_st in list(states_data.keys()):
                states_data[k_st] = states_data[k_st][pu_order, :, :]
            for k_st in list(states_signal_ss.keys()):
                states_signal_ss[k_st] = states_signal_ss[k_st][pu_order, :, :, :]

        datatype_str = "FreqDomain"
        w0_cm1 = 0.0
        dt1_fs = 0.0
        raw_signal = raw_cube.copy()
        raw_interferogram = None
        raw_t1delays = None
        Z = raw_cube.copy()

        if "Parallel" in states_data and "Perpendicular" in states_data:
            p_mat = states_data["Parallel"]
            s_mat = states_data["Perpendicular"]
            denom = p_mat + 2.0 * s_mat
            peak_iso = float(np.nanmax(np.abs(denom))) if denom.size > 0 else 0.0
            thresh = max(1e-6, 0.02 * peak_iso)
            with np.errstate(divide="ignore", invalid="ignore"):
                states_data["Anisotropy"] = np.where(
                    np.abs(denom) > thresh, (p_mat - s_mat) / denom, 0.0
                )
    else:
        t1 = axis0_ref.copy()
        w0_rf_cm1 = 0.0
        if "Rotating Frame (Scanned)" in info or "Rotating Frame (Fixed)" in info:
            try:
                rf_scan = float(info.get("Rotating Frame (Scanned)", "0") or 0.0)
                rf_fix = float(info.get("Rotating Frame (Fixed)", "0") or 0.0)
                w0_rf_cm1 = rf_scan - rf_fix
            except ValueError:
                w0_rf_cm1 = 0.0
        elif "Rotating Frame" in info:
            try:
                w0_rf_cm1 = float(info.get("Rotating Frame", "0") or 0.0)
            except ValueError:
                w0_rf_cm1 = 0.0

        dt1_fs = 0.0
        if "Step Size (fs)" in info:
            try:
                dt1_fs = float(info["Step Size (fs)"])
            except ValueError:
                dt1_fs = 0.0
        if dt1_fs <= 0.0 and len(t1) > 1:
            dt1_fs = float(np.nanmean(np.abs(np.diff(t1))))

        w0_cm1, nyquist_invert = _unfold_undersampled_shaper_w0(
            w0_rf_cm1, dt1_fs, info, probe, raw_cube, probe_spectrum=spec_arr
        )

        raw_signal = raw_cube.copy()
        raw_interferogram = np.nanmean(raw_signal, axis=1)
        raw_t1delays = t1.copy()
        datatype_str = "shaper"
        Z = raw_signal.copy()
        pump = t1.copy()

    units = _units_2d("ps", freq_units)
    if probe.size > 0 and float(np.nanmax(probe)) > hlp.DISPLAY_SCALE_THRESHOLD:
        units["unitsZ_lbl"] = r"S$_{\text{2DES}}$"
    if freq_units == "nm":
        units["unitsL_lbl"] = "Wavelength"
        units["unitsL_ltx"] = "nm"
    units["cal_source"] = cal_source
    if cal_level is not None:
        units["cal_level"] = cal_level
    units["is_frequency_domain"] = is_freq_domain
    units["w0_cm1"] = w0_cm1
    units["dt1_fs"] = dt1_fs
    if nyquist_invert:
        units["nyquist_invert"] = True
    units["quickcontrol_info"] = info
    units["interferogram_from_signal"] = not is_freq_domain
    units["nscans"] = int(n_scans_total)

    if raw_signal_ss is not None:
        units["raw_signal_ss"] = raw_signal_ss
        if not is_freq_domain:
            import warnings

            with warnings.catch_warnings(), np.errstate(invalid="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                units["raw_interferogram_ss"] = np.nanmean(raw_signal_ss, axis=1)
    if states_signal_ss:
        units["states_signal_ss"] = states_signal_ss
    if states_data:
        units["state_names"] = list(states_data.keys())
        units["states_data"] = states_data
        units["active_state"] = list(states_data.keys())[0]

    if spec_arr is not None:
        units["probe_spectrum"] = spec_arr

    mode_line = (
        "Mode: Frequency-Domain 2D"
        if is_freq_domain
        else f"Mode: Time-Domain 2D (w0 = {w0_cm1:.1f} cm-1, dt1 = {dt1_fs:g} fs)"
    )
    units["sample_info"] = _format_quickcontrol_sample_info(
        info,
        mode_line=mode_line,
        state_names=list(states_data.keys()) if states_data else None,
    )

    return (
        Z,
        pump,
        probe,
        t2delays_ps,
        units,
        freq_units,
        datatype_str,
        in_progress,
        raw_signal,
        raw_interferogram,
        raw_t1delays,
    )


# --------------------------------------------------------------------------- #
#              Shared helpers for UoS 2DIR & RAL 2DIR readers                 #
# --------------------------------------------------------------------------- #
def _read_w0_csv(folder: Path) -> tuple[float, float]:
    """Read ``(w0_cm1, dt1_fs)`` from ``w0.csv`` in ``folder`` or its parent."""
    for cand in (folder / "w0.csv", folder.parent / "w0.csv"):
        if cand.is_file():
            try:
                vals = np.loadtxt(cand, delimiter=",").ravel().astype(float)
                if vals.size >= 2:
                    return float(vals[0]), float(vals[1])
                if vals.size == 1:
                    return float(vals[0]), 15.0
            except Exception:
                try:
                    vals = np.loadtxt(cand).ravel().astype(float)
                    if vals.size >= 2:
                        return float(vals[0]), float(vals[1])
                    if vals.size == 1:
                        return float(vals[0]), 15.0
                except Exception:
                    logger.debug("Could not parse %s", cand, exc_info=True)
    return 0.0, 15.0


def _is_strictly_monotonic(arr: np.ndarray) -> bool:
    arr = np.asarray(arr, dtype=float)
    if arr.size <= 1 or not np.all(np.isfinite(arr)):
        return False
    diffs = np.diff(arr)
    return bool(np.all(diffs > 0) or np.all(diffs < 0))


def _probes_disjoint_and_monotonic(p1: np.ndarray, p2: np.ndarray) -> bool:
    if not (_is_strictly_monotonic(p1) and _is_strictly_monotonic(p2)):
        return False
    return bool(float(np.max(p1)) < float(np.min(p2)) or float(np.max(p2)) < float(np.min(p1)))


def _resolve_split_calibrated_probe(
    folder: Path,
    n_detectors: int,
    pix_per_det: int,
) -> tuple[list[np.ndarray] | None, str | None, str | None]:
    """Load ``CalibratedProbe.csv`` from ``folder`` or parent and split per detector."""
    cands = (
        (folder / "CalibratedProbe.csv", "data folder", "datadir"),
        (folder.parent / "CalibratedProbe.csv", "root folder", "rootdir"),
    )
    for cal_path, cal_loc, cal_lvl in cands:
        if not cal_path.is_file():
            continue
        try:
            arr = np.loadtxt(cal_path, delimiter=",").ravel().astype(float)
            if n_detectors == 2 and len(arr) >= 2 * pix_per_det:
                return (
                    [arr[:pix_per_det], arr[pix_per_det : 2 * pix_per_det]],
                    f"CalibratedProbe.csv ({cal_loc})",
                    cal_lvl,
                )
            if n_detectors == 1 and len(arr) >= pix_per_det:
                return [arr[:pix_per_det]], f"CalibratedProbe.csv ({cal_loc})", cal_lvl
        except Exception:
            logger.debug("Could not read CalibratedProbe.csv at %s", cal_path, exc_info=True)
    return None, None, None


def _build_dual_detector_maps(
    cube_full: np.ndarray,
    probe1: np.ndarray,
    probe2: np.ndarray,
    *,
    allow_combined: bool,
    pix_per_det: int,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Construct ``(detector_names, detector_probes, detector_data)`` for dual-array setups."""
    d1 = cube_full[:, :pix_per_det, :]
    d2 = cube_full[:, pix_per_det : 2 * pix_per_det, :]
    p1 = np.asarray(probe1, dtype=float).ravel()
    p2 = np.asarray(probe2, dtype=float).ravel()

    names = ["DET1", "DET2"]
    probes = {"DET1": p1, "DET2": p2}
    data = {"DET1": d1, "DET2": d2}

    if allow_combined and _probes_disjoint_and_monotonic(p1, p2):
        p_comb = np.concatenate([p1, p2])
        order = np.argsort(p_comb)
        d_comb = cube_full[:, : 2 * pix_per_det, :][:, order, :]
        names.append("Combined (DET1+DET2)")
        probes["Combined (DET1+DET2)"] = p_comb[order]
        data["Combined (DET1+DET2)"] = d_comb

    return names, probes, data


# --------------------------------------------------------------------------- #
#                              UoS 2D-IR reader                               #
# --------------------------------------------------------------------------- #
_UOS_SLICE_RE = re.compile(
    r"_D(\d+)(?:_([+-]?\d+(?:\.\d+)?)_ps)?",
    re.IGNORECASE,
)


def _find_uos_2dir_slices(folder: Path) -> list[tuple[int, float | None, Path]]:
    """Return sorted ``(D_index, optional_filename_ps, path)`` for non-empty delay-slice ``.2D`` files."""
    entries: list[tuple[int, float | None, Path]] = []
    for f in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if not f.is_file() or f.suffix.lower() != ".2d" or f.stat().st_size <= 0:
            continue
        m = _UOS_SLICE_RE.search(f.stem)
        if not m:
            continue
        k_idx = int(m.group(1))
        t2_fn = float(m.group(2)) if m.group(2) is not None else None
        entries.append((k_idx, t2_fn, f))
    return entries


@register_map_dataset_detector("UoS_2DIR")
def _uos_2dir_is_dataset(folder: Path) -> bool:
    """Detect whether ``folder`` is a University of Sheffield 2D-IR dataset directory."""
    folder = Path(folder)
    if not folder.is_dir():
        return False
    slices = _find_uos_2dir_slices(folder)
    if not slices:
        return False

    all_files = [f for f in folder.iterdir() if f.is_file()]
    dt_files = [f for f in all_files if f.suffix.lower() == ".dt" and f.stat().st_size > 0]
    if dt_files:
        dt_stems = {f.stem.lower() for f in dt_files}
        main_twod = [f for f in all_files if f.suffix.lower() == ".2d" and f.stem.lower() in dt_stems]
        if any(f.stat().st_size > 0 for f in main_twod):
            return False
        return True

    # Support partial / aborted UoS 2D-IR scans missing .DT (e.g. *-meh folders)
    has_ps_slices = any(t2 is not None for _, t2, _ in slices)
    has_w0 = any(f.name.lower() == "w0.csv" for f in all_files)
    return has_ps_slices and (has_w0 or len(slices) >= 1)


def _parse_uos_lg_sample_info(lg_file: Path | None) -> tuple[str, str]:
    """Extract compact and full metadata strings from a UoS ``.LG`` log file."""
    if lg_file is None or not lg_file.is_file():
        return "", ""
    try:
        lines = lg_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return "", ""

    compact_keys = (
        "acquisition mode",
        "data mode",
        "number of scans",
        "data normalised on reference?",
        "grating grooves/mm",
        "central wavelength nm",
        "entrance slit width mm",
    )
    skip_prefixes = ("background ", "trim values", "gain values", "........")
    compact_lines: list[str] = []
    full_lines: list[str] = []
    for raw_ln in lines:
        ln = raw_ln.strip()
        if not ln or ln.startswith("*"):
            continue
        low = ln.lower()
        if low.startswith(skip_prefixes):
            continue
        if ":" in ln:
            key, val = ln.split(":", 1)
            val_clean = val.strip()
            if not val_clean:
                continue
            pretty = f"{key.strip()}: {val_clean}"
            full_lines.append(pretty)
            if any(key.strip().lower().startswith(ck) for ck in compact_keys):
                compact_lines.append(pretty)
    return "\n".join(compact_lines), "\n".join(full_lines)


@register_map_loader("UoS_2DIR")
def read_UoS_2DIR(path: str, progress_tracker=None):
    """Read a University of Sheffield 2D-IR dataset (``.DT`` + ``*_D*_*.2D`` slices)."""
    from ..oneD.load import _estimate_uos_probe, _parse_uos_lg

    p = Path(path)
    folder = p if p.is_dir() else p.parent
    slices = _find_uos_2dir_slices(folder)
    if not slices:
        raise FileNotFoundError(f"No UoS 2D-IR delay-slice (*_D*_*.2D) files found in {folder}")

    all_files = sorted((f for f in folder.iterdir() if f.is_file()), key=lambda f: f.name.lower())
    dt_files = [f for f in all_files if f.suffix.lower() == ".dt" and f.stat().st_size > 0]
    lg_files = [f for f in all_files if f.suffix.lower() == ".lg"]

    dt_delays: np.ndarray | None = None
    if dt_files:
        try:
            dt_delays = np.loadtxt(dt_files[0], dtype=float).ravel()
        except Exception:
            dt_delays = None

    in_progress = dt_delays is None or len(slices) < len(dt_delays)

    aligned: list[tuple[float, int, Path]] = []
    for k_idx, fn_t2, fpath in slices:
        if dt_delays is not None and 0 <= k_idx < len(dt_delays):
            t2_val = float(dt_delays[k_idx])
        elif fn_t2 is not None:
            t2_val = float(fn_t2)
        else:
            t2_val = float(k_idx)
        aligned.append((t2_val, k_idx, fpath))

    aligned.sort(key=lambda item: (item[0], item[1]))
    delays_ps = np.asarray([item[0] for item in aligned], dtype=float)

    n_total = len(aligned)
    if progress_tracker is not None:
        progress_tracker.set_total(max(1, n_total))

    matrices: list[np.ndarray] = []
    for idx, (_t2, _k, fpath) in enumerate(aligned):
        if progress_tracker is not None:
            if hasattr(progress_tracker, "update_split"):
                progress_tracker.update_split(idx + 1, n_total, "loading")
            else:
                progress_tracker.update(idx + 1, label=f"Loading UoS 2D slice {idx + 1}/{n_total}...")
        arr = np.loadtxt(fpath, dtype=float)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        matrices.append(arr)

    min_rows = min(m.shape[0] for m in matrices)
    if any(m.shape[0] != min_rows for m in matrices):
        in_progress = True
        matrices = [m[:min_rows, :] for m in matrices]

    min_cols = min(m.shape[1] for m in matrices)
    matrices = [m[:, :min_cols] for m in matrices]
    cube_full = np.stack(matrices, axis=-1)  # (Nt1, Ncols, Nt2)

    nt1 = cube_full.shape[0]
    w0_cm1, dt1_fs = _read_w0_csv(folder)
    raw_t1delays = np.arange(nt1, dtype=float) * dt1_fs

    if min_cols >= 192:
        n_det = 2
        pix_per_det = 96
    else:
        n_det = 1
        pix_per_det = min_cols

    cal_probes, cal_source, cal_level = _resolve_split_calibrated_probe(folder, n_det, pix_per_det)
    lg_file = lg_files[0] if lg_files else None
    if cal_probes is None:
        gratings, cwls = _parse_uos_lg(lg_file) if lg_file is not None else ([], [])
        if (
            len(gratings) >= n_det
            and len(cwls) >= n_det
            and all(c > 0 for c in cwls[:n_det])
        ):
            cal_probes = [_estimate_uos_probe(gratings[d], cwls[d], pix_per_det) for d in range(n_det)]
            cal_source = f"Estimated from log file ({lg_file.name})"
        else:
            pix = np.arange(1, pix_per_det + 1, dtype=float)
            cal_probes = [pix.copy() for _ in range(n_det)]
            cal_source = "Pixel indices (uncalibrated)"

    units = _units_2d("ps", "cm-1")
    units["w0_cm1"] = w0_cm1
    units["dt1_fs"] = dt1_fs
    units["is_frequency_domain"] = False
    units["interferogram_from_signal"] = True
    units["cal_source"] = cal_source
    if cal_level is not None:
        units["cal_level"] = cal_level

    compact_info, full_info = _parse_uos_lg_sample_info(lg_file)
    if compact_info:
        units["sample_info"] = compact_info
    if full_info:
        units["sample_info_full"] = full_info

    if n_det == 2:
        det_names, det_probes, det_data = _build_dual_detector_maps(
            cube_full, cal_probes[0], cal_probes[1], allow_combined=False, pix_per_det=pix_per_det
        )
        units["detector_names"] = det_names
        units["detector_probes"] = det_probes
        units["detector_data"] = det_data
        units["active_detector"] = "DET1"
        raw_signal = det_data["DET1"].copy()
        probe = det_probes["DET1"].copy()
    else:
        raw_signal = cube_full[:, :pix_per_det, :].copy()
        probe = np.asarray(cal_probes[0], dtype=float).copy()

    raw_interferogram = np.nanmean(raw_signal, axis=1)
    Z = raw_signal.copy()
    pump = raw_t1delays.copy()

    return (
        Z,
        pump,
        probe,
        delays_ps,
        units,
        "cm-1",
        "shaper",
        in_progress,
        raw_signal,
        raw_interferogram,
        raw_t1delays,
    )


# --------------------------------------------------------------------------- #
#                            RAL 2D-IR (Raw & Proc)                           #
# --------------------------------------------------------------------------- #
def _find_ral_raw_avgdiff(folder: Path) -> Path | None:
    """Locate the non-empty ``*AvgDiff.csv`` file in ``folder`` (excluding 1D ``Avg Diff spec``)."""
    cands = sorted(
        [
            f
            for f in folder.glob("*.csv")
            if f.is_file()
            and f.stat().st_size > 0
            and "avgdiff" in f.name.lower()
            and "avg diff spec" not in f.name.lower()
        ],
        key=lambda f: f.name.lower(),
    )
    return cands[0] if cands else None


def _ral_header_has_encoded_t1(avg_file: Path) -> bool:
    """Check whether the first row of ``avg_file`` carries ``1e7``-encoded ``t1_idx`` columns."""
    try:
        with open(avg_file, encoding="utf-8", errors="ignore") as fh:
            first = fh.readline()
        parts = [p.strip() for p in first.split(",")[1:12] if p.strip()]
        vals = [float(p) for p in parts]
        return any(abs(v) >= 1e6 for v in vals)
    except Exception:
        return False


@register_map_dataset_detector("RAL_RAW")
def _ral_raw_is_dataset(folder: Path) -> bool:
    """Detect whether ``folder`` is a RAL raw 2D-IR dataset directory."""
    folder = Path(folder)
    if not folder.is_dir():
        return False
    avg_file = _find_ral_raw_avgdiff(folder)
    if avg_file is None:
        return False
    return (folder / "w0.csv").is_file() or _ral_header_has_encoded_t1(avg_file)


def _parse_ral_info_txt(folder: Path) -> tuple[dict[str, str], str, str]:
    """Parse ``Run * info.txt`` into ``(units_kv, compact_info, full_info)``."""
    info_files = sorted(
        (f for f in folder.iterdir() if f.is_file() and f.name.lower().endswith("info.txt")),
        key=lambda f: f.name.lower(),
    )
    if not info_files:
        return {}, "", ""
    units_kv: dict[str, str] = {}
    full_lines: list[str] = []
    compact_lines: list[str] = []
    compact_keys = {
        "date",
        "time",
        "sample",
        "solvent",
        "concentration",
        "polarisations",
        "cycles (if not stopped early)",
        "single cycle average time",
    }
    try:
        for raw_ln in info_files[0].read_text(encoding="utf-8", errors="ignore").splitlines():
            ln = raw_ln.strip()
            if not ln or ":" not in ln:
                continue
            key, val = ln.split(":", 1)
            k_clean = key.strip()
            v_clean = val.strip()
            if not v_clean:
                continue
            units_kv[f"info_{k_clean.lower().replace(' ', '_')}"] = v_clean
            pretty = f"{k_clean}: {v_clean}"
            full_lines.append(pretty)
            if k_clean.lower() in compact_keys:
                compact_lines.append(pretty)
    except OSError:
        pass
    return units_kv, "\n".join(compact_lines), "\n".join(full_lines)


@register_map_loader("RAL_RAW")
def read_RAL_RAW(path: str, progress_tracker=None, load_single_scans: bool | None = None):
    """Read a RAL raw 2D-IR dataset (``Run *_AvgDiff.csv`` + ``w0.csv`` + ``CalibratedProbe.csv``)."""
    import pandas as pd

    p = Path(path)
    if p.is_file() and "avgdiff" in p.name.lower() and "avg diff spec" not in p.name.lower():
        avg_file = p
        folder = p.parent
    else:
        folder = p if p.is_dir() else p.parent
        avg_file = _find_ral_raw_avgdiff(folder)
        if avg_file is None:
            raise FileNotFoundError(f"No '*AvgDiff.csv' file found in RAL 2D folder {folder}")

    if progress_tracker is not None:
        progress_tracker.set_total(2)
        if hasattr(progress_tracker, "update_split"):
            progress_tracker.update_split(1, 2, "loading")
        else:
            progress_tracker.update(1, label=f"Reading {avg_file.name}...")

    df_raw = pd.read_csv(avg_file, header=None, engine="c")
    if df_raw.shape[0] < 2 or df_raw.shape[1] < 3:
        raise ValueError(f"Invalid RAL_RAW matrix dimensions {df_raw.shape} in {avg_file}")

    hdr = pd.to_numeric(df_raw.iloc[0, 1:], errors="coerce").to_numpy(dtype=float)
    valid_cols = np.isfinite(hdr)
    hdr = hdr[valid_cols]
    mat = (
        df_raw.iloc[1:, 1:]
        .loc[:, valid_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    mat = np.where(np.isfinite(mat), mat, 0.0)

    t1_idx = np.round(hdr / 1e7).astype(int)
    t2_rem = hdr - t1_idx * 1e7

    # Partition columns into contiguous t2 blocks while keeping ALL columns per block
    # (including both leading t1_idx == 0 entries).
    blocks: list[tuple[float, np.ndarray]] = []
    b_start = 0
    for idx in range(1, len(hdr)):
        same_t2 = abs(t2_rem[idx] - t2_rem[b_start]) <= 1e-4
        wrapped_t1 = (idx > b_start + 1) and (t1_idx[idx] < t1_idx[idx - 1])
        if not same_t2 or wrapped_t1:
            t2_val = float(np.mean(t2_rem[b_start:idx]))
            blocks.append((t2_val, mat[:, b_start:idx].T))  # (Nt1_block, Npix)
            b_start = idx
    t2_val = float(np.mean(t2_rem[b_start:]))
    blocks.append((t2_val, mat[:, b_start:].T))

    if not blocks:
        raise ValueError(f"Could not decode any t2 delay blocks from {avg_file}")

    block_lens = [b[1].shape[0] for b in blocks]
    nt1_expected = max(set(block_lens), key=block_lens.count)
    in_progress = any(bl != nt1_expected for bl in block_lens)
    complete_blocks = [b for b in blocks if b[1].shape[0] == nt1_expected]
    if not complete_blocks:
        # Single partial block: keep whatever length was recorded
        nt1_expected = block_lens[0]
        complete_blocks = blocks

    complete_blocks.sort(key=lambda b: b[0])
    delays_ps = np.asarray([b[0] for b in complete_blocks], dtype=float)
    cube_full = np.stack([b[1] for b in complete_blocks], axis=-1)  # (Nt1, Npix_total, Nt2)

    if progress_tracker is not None:
        if hasattr(progress_tracker, "update_split"):
            progress_tracker.update_split(2, 2, "loading")
        else:
            progress_tracker.update(2, label="Finalizing RAL 2D-IR dataset...")

    n_pix_total = cube_full.shape[1]
    w0_cm1, dt1_fs = _read_w0_csv(folder)
    raw_t1delays = np.arange(nt1_expected, dtype=float) * dt1_fs

    if n_pix_total >= 256:
        n_det = 2
        pix_per_det = 128
    elif n_pix_total >= 192:
        n_det = 2
        pix_per_det = 96
    else:
        n_det = 1
        pix_per_det = n_pix_total

    cal_probes, cal_source, cal_level = _resolve_split_calibrated_probe(folder, n_det, pix_per_det)
    if cal_probes is None:
        pix = np.arange(1, pix_per_det + 1, dtype=float)
        cal_probes = [pix.copy() for _ in range(n_det)]
        cal_source = "Pixel indices (uncalibrated)"

    units = _units_2d("ps", "cm-1")
    units["w0_cm1"] = w0_cm1
    units["dt1_fs"] = dt1_fs
    units["is_frequency_domain"] = False
    units["interferogram_from_signal"] = True
    units["cal_source"] = cal_source
    if cal_level is not None:
        units["cal_level"] = cal_level

    info_kv, compact_info, full_info = _parse_ral_info_txt(folder)
    units.update(info_kv)
    if compact_info:
        units["sample_info"] = compact_info
    if full_info:
        units["sample_info_full"] = full_info

    if n_det == 2:
        det_names, det_probes, det_data = _build_dual_detector_maps(
            cube_full, cal_probes[0], cal_probes[1], allow_combined=True, pix_per_det=pix_per_det
        )
        units["detector_names"] = det_names
        units["detector_probes"] = det_probes
        units["detector_data"] = det_data
        units["active_detector"] = "DET1"
        raw_signal = det_data["DET1"].copy()
        probe = det_probes["DET1"].copy()
    else:
        raw_signal = cube_full[:, :pix_per_det, :].copy()
        probe = np.asarray(cal_probes[0], dtype=float).copy()

    raw_interferogram = np.nanmean(raw_signal, axis=1)
    Z = raw_signal.copy()
    pump = raw_t1delays.copy()

    return (
        Z,
        pump,
        probe,
        delays_ps,
        units,
        "cm-1",
        "shaper",
        in_progress,
        raw_signal,
        raw_interferogram,
        raw_t1delays,
    )


_RAL_PROC_FILE_RE = re.compile(
    r"([+-]?\d+(?:\.\d+)?)\s*(fs|ps|ns)(?=_2dir\.csv$)",
    re.IGNORECASE,
)


def _find_ral_proc_files(folder: Path) -> list[tuple[float, Path]]:
    """Find and sort ``*_2DIR.csv`` delay files in ``folder`` by population time ``t2`` (ps)."""
    entries: list[tuple[float, Path]] = []
    for f in sorted(folder.glob("*_2DIR.csv"), key=lambda p: p.name.lower()):
        if not f.is_file() or f.stat().st_size <= 0:
            continue
        m = _RAL_PROC_FILE_RE.search(f.name)
        if not m:
            continue
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit == "fs":
            val /= 1000.0
        elif unit == "ns":
            val *= 1000.0
        entries.append((val, f))
    entries.sort(key=lambda item: item[0])
    return entries


@register_map_dataset_detector("RAL_Proc")
def _ral_proc_is_dataset(folder: Path) -> bool:
    """Detect whether ``folder`` is a RAL processed 2D-IR directory (``*_2DIR.csv``)."""
    folder = Path(folder)
    if not folder.is_dir():
        return False
    return len(_find_ral_proc_files(folder)) > 0


@register_map_loader("RAL_Proc")
def read_RAL_Proc(path: str, progress_tracker=None):
    """Read a RAL processed 2D-IR dataset (directory of ``*_2DIR.csv`` delay files)."""
    import pandas as pd

    p = Path(path)
    folder = p if p.is_dir() else p.parent
    entries = _find_ral_proc_files(folder)
    if not entries:
        raise FileNotFoundError(f"No '*_2DIR.csv' processed delay files found in {folder}")

    n_files = len(entries)
    if progress_tracker is not None:
        progress_tracker.set_total(max(1, n_files))

    delays_list: list[float] = []
    slices: list[np.ndarray] = []
    pump_ref: np.ndarray | None = None
    probe_ref: np.ndarray | None = None

    for idx, (t2_ps, fpath) in enumerate(entries):
        if progress_tracker is not None:
            if hasattr(progress_tracker, "update_split"):
                progress_tracker.update_split(idx + 1, n_files, "loading")
            else:
                progress_tracker.update(idx + 1, label=f"Loading {fpath.name}...")
        with open(fpath, encoding="utf-8", errors="ignore") as fh:
            first_line = fh.readline()
        sep = "\t" if "\t" in first_line else ","
        arr = pd.read_csv(fpath, sep=sep, header=None, engine="c").to_numpy(dtype=float)
        if arr.shape[0] < 2 or arr.shape[1] < 2:
            raise ValueError(f"Invalid RAL_Proc CSV dimensions {arr.shape} in {fpath}")
        if pump_ref is None:
            pump_ref = arr[0, 1:].astype(float)
            probe_ref = arr[1:, 0].astype(float)
        delays_list.append(t2_ps)
        slices.append(arr[1:, 1:].T.astype(float))  # (Npump, Nprobe_total)

    assert pump_ref is not None and probe_ref is not None
    pump = pump_ref
    probe_all = probe_ref
    delays_ps = np.asarray(delays_list, dtype=float)
    cube_full = np.stack(slices, axis=-1)  # (Npump, Nprobe_total, Nt2)

    # Collapse redundant Hermitian mirror around Nyquist if present
    n_pump = len(pump)
    if (
        n_pump >= 4
        and n_pump % 2 == 0
        and np.allclose(
            cube_full[1 : n_pump // 2, :, :],
            cube_full[-1 : n_pump // 2 : -1, :, :],
            rtol=1e-4,
            atol=1e-6,
            equal_nan=True,
        )
    ):
        n_half = n_pump // 2
        pump = pump[:n_half]
        cube_full = cube_full[:n_half, :, :]

    n_pix_total = len(probe_all)
    if n_pix_total >= 256:
        n_det = 2
        pix_per_det = 128
    elif n_pix_total >= 192:
        n_det = 2
        pix_per_det = 96
    else:
        n_det = 1
        pix_per_det = n_pix_total

    units = _units_2d("ps", "cm-1")
    units["is_frequency_domain"] = True
    units["cal_source"] = "Dataset probe axis (RAL_Proc CSV)"

    # Only fall back to CalibratedProbe.csv if row headers are uncalibrated pixel indices
    if np.allclose(probe_all, np.arange(1, n_pix_total + 1, dtype=float)):
        cal_probes, cal_source, cal_level = _resolve_split_calibrated_probe(folder, n_det, pix_per_det)
        if cal_probes is not None:
            p1 = cal_probes[0]
            p2 = cal_probes[1] if n_det == 2 else None
            units["cal_source"] = cal_source
            if cal_level is not None:
                units["cal_level"] = cal_level
        else:
            p1 = probe_all[:pix_per_det]
            p2 = probe_all[pix_per_det : 2 * pix_per_det] if n_det == 2 else None
    else:
        p1 = probe_all[:pix_per_det]
        p2 = probe_all[pix_per_det : 2 * pix_per_det] if n_det == 2 else None

    if n_det == 2 and p2 is not None:
        det_names, det_probes, det_data = _build_dual_detector_maps(
            cube_full, p1, p2, allow_combined=True, pix_per_det=pix_per_det
        )
        units["detector_names"] = det_names
        units["detector_probes"] = det_probes
        units["detector_data"] = det_data
        units["active_detector"] = "DET1"
        Z = det_data["DET1"].copy()
        probe = det_probes["DET1"].copy()
    else:
        Z = cube_full[:, :pix_per_det, :].copy()
        probe = np.asarray(p1, dtype=float).copy()

    return (
        Z,
        pump,
        probe,
        delays_ps,
        units,
        "cm-1",
        "processed",
        False,
        Z.copy(),
        None,
        None,
    )


# --------------------------------------------------------------------------- #
#                         FAIR 2D HDF5 (HDF5_2D)                              #
# --------------------------------------------------------------------------- #
register_map_dataset_glob("HDF5_2D", "*.h5")


@register_map_dataset_file_filter("HDF5_2D")
def _pymorgan_2d_hdf5_is_file(path: Path | str) -> bool:
    """Return True if ``path`` is a PyMORGAN FAIR 2D HDF5 archive."""
    import h5py

    p = Path(path)
    if not p.is_file() or p.suffix.lower() not in {".h5", ".hdf5"}:
        return False
    try:
        if not h5py.is_hdf5(p):
            return False
        with h5py.File(p, "r") as f:
            fmt = f.attrs.get("pymorgan_format", "")
            if isinstance(fmt, (bytes, np.bytes_)):
                fmt = fmt.decode("utf-8", errors="replace")
            if fmt == "2D":
                return True
            if "pump" in f and "probe" in f and "delays" in f and "Z_R" in f:
                return True
    except Exception:
        return False
    return False


def _save_mapping_group(parent, group_name: str, mapping: dict[str, np.ndarray] | None, *, units: str | None = None) -> None:
    from ..hdf5_meta import create_compressed_dataset

    if not mapping:
        return
    grp = parent.create_group(group_name)
    order: list[str] = []
    for key, arr in mapping.items():
        safe_key = str(key).replace("/", "_")
        dset = create_compressed_dataset(grp, safe_key, np.asarray(arr, dtype=float), units=units)
        dset.attrs["original_key"] = str(key)
        order.append(str(key))
    import json

    grp.attrs["keys_json"] = json.dumps(order)


def _load_mapping_group(parent, group_name: str) -> dict[str, np.ndarray] | None:
    import json

    if group_name not in parent:
        return None
    grp = parent[group_name]
    out: dict[str, np.ndarray] = {}
    ordered_keys: list[str] = []
    if "keys_json" in grp.attrs:
        try:
            ordered_keys = list(json.loads(grp.attrs["keys_json"]))
        except Exception:
            ordered_keys = []
    by_orig: dict[str, np.ndarray] = {}
    for k in grp.keys():
        dset = grp[k]
        orig_k = dset.attrs.get("original_key", k)
        if isinstance(orig_k, (bytes, np.bytes_)):
            orig_k = orig_k.decode("utf-8", errors="replace")
        by_orig[str(orig_k)] = np.asarray(dset[:], dtype=float)
    for k in ordered_keys:
        if k in by_orig:
            out[k] = by_orig.pop(k)
    out.update(by_orig)
    return out or None


def save_2D_hdf5(dataset, path: str | PathLike) -> Path:
    """Save a :class:`~pymorgan.twoD.Dataset2D` instance to a FAIR/NeXus HDF5 archive."""
    import json

    import h5py

    from ..hdf5_meta import create_compressed_dataset, json_safe_units, write_fair_root_attrs

    target = Path(path)
    if target.suffix.lower() not in (".h5", ".hdf5"):
        target = target.with_suffix(".h5")
    target.parent.mkdir(parents=True, exist_ok=True)

    units_dict = dataset.units if isinstance(dataset.units, dict) else {}
    freq_u = str(getattr(dataset, "freq_units", "cm-1") or "cm-1")
    t2_u = str(units_dict.get("unitsT_ltx", "ps")).replace("$", "")
    z_u = str(units_dict.get("unitsZ_ltx", "mOD")).replace("$", "")

    with h5py.File(target, "w") as f:
        write_fair_root_attrs(
            f,
            pymorgan_format="2D",
            source=getattr(dataset, "source", None),
            original_data_type=getattr(dataset, "data_type", None),
            units=units_dict,
            dataset=dataset,
            format_version="2.0",
        )

        create_compressed_dataset(f, "pump", np.asarray(dataset.pump, dtype=float), units=freq_u, long_name="Pump axis")
        create_compressed_dataset(f, "probe", np.asarray(dataset.probe, dtype=float), units=freq_u, long_name="Probe axis")
        create_compressed_dataset(f, "delays", np.asarray(dataset.delays, dtype=float), units=t2_u, long_name="Population time t2")
        create_compressed_dataset(f, "Z_R", np.asarray(dataset.Z_R, dtype=float), units=z_u, long_name="Raw 2D signal cube")

        if getattr(dataset, "Z_C", None) is not None:
            create_compressed_dataset(f, "Z_C", np.asarray(dataset.Z_C, dtype=float), units=z_u, long_name="Corrected 2D signal cube")
        if getattr(dataset, "raw_signal", None) is not None:
            create_compressed_dataset(f, "raw_signal", np.asarray(dataset.raw_signal, dtype=float), long_name="Raw time-domain/frequency signal")
        if getattr(dataset, "raw_interferogram", None) is not None:
            create_compressed_dataset(f, "raw_interferogram", np.asarray(dataset.raw_interferogram, dtype=float), long_name="Raw pump interferogram")
        if getattr(dataset, "raw_t1delays", None) is not None:
            create_compressed_dataset(f, "raw_t1delays", np.asarray(dataset.raw_t1delays, dtype=float), units="fs", long_name="t1 delay axis")
        if getattr(dataset, "raw_signal_ss", None) is not None:
            create_compressed_dataset(f, "raw_signal_ss", np.asarray(dataset.raw_signal_ss, dtype=float), long_name="Single-scan 2D signal cube")
        if getattr(dataset, "raw_interferogram_ss", None) is not None:
            create_compressed_dataset(f, "raw_interferogram_ss", np.asarray(dataset.raw_interferogram_ss, dtype=float), long_name="Single-scan interferograms")

        _save_mapping_group(f, "states_data", getattr(dataset, "states_data", None), units=z_u)
        _save_mapping_group(f, "states_signal_ss", getattr(dataset, "states_signal_ss", None), units=z_u)
        _save_mapping_group(f, "detector_probes", getattr(dataset, "detector_probes", None), units=freq_u)
        _save_mapping_group(f, "detector_data", getattr(dataset, "detector_data", None), units=z_u)
        _save_mapping_group(f, "detector_signal_ss", getattr(dataset, "detector_signal_ss", None), units=z_u)

        f.attrs["data_type"] = str(getattr(dataset, "data_type", None) or "HDF5_2D")
        f.attrs["datatype"] = str(getattr(dataset, "datatype", None) or "processed")
        f.attrs["freq_units"] = freq_u
        f.attrs["w0_cm1"] = float(getattr(dataset, "w0_cm1", 0.0) or 0.0)
        f.attrs["dt1_fs"] = float(getattr(dataset, "dt1_fs", 0.0) or 0.0)
        f.attrs["shot_rotation"] = int(getattr(dataset, "shot_rotation", 0) or 0)
        f.attrs["is_frequency_domain"] = bool(getattr(dataset, "is_frequency_domain", False))
        f.attrs["in_progress"] = bool(getattr(dataset, "in_progress", False))
        f.attrs["nscans"] = int(getattr(dataset, "nscans", 1) or 1)
        if getattr(dataset, "cal_level", None) is not None:
            f.attrs["cal_level"] = str(dataset.cal_level)
        if getattr(dataset, "cal_source", None) is not None:
            f.attrs["cal_source"] = str(dataset.cal_source)
        if getattr(dataset, "active_state", None) is not None:
            f.attrs["active_state"] = str(dataset.active_state)
        if getattr(dataset, "active_detector", None) is not None:
            f.attrs["active_detector"] = str(dataset.active_detector)
        if getattr(dataset, "state_names", None):
            f.attrs["state_names_json"] = json.dumps(list(dataset.state_names))
        if getattr(dataset, "detector_names", None):
            f.attrs["detector_names_json"] = json.dumps(list(dataset.detector_names))
        if getattr(dataset, "active_scan_indices", None) is not None:
            f.attrs["active_scan_indices_json"] = json.dumps([int(i) for i in dataset.active_scan_indices])

        units_clean = json_safe_units(
            units_dict,
            drop_keys={
                "states_data",
                "states_signal_ss",
                "detector_probes",
                "detector_data",
                "detector_signal_ss",
                "raw_signal_ss",
                "raw_interferogram_ss",
            },
        )
        f.attrs["units_json"] = json.dumps(units_clean)

    return target


def _decode_attr(val: Any, default: str = "") -> str:
    if val is None:
        return default
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8", errors="replace")
    return str(val)


@register_map_loader("HDF5_2D")
def read_2D_hdf5(path: str | PathLike):
    """Read a FAIR 2D HDF5 archive into the 11-tuple expected by :meth:`Dataset2D.from_file`."""
    import json

    import h5py

    p = Path(path)
    with h5py.File(p, "r") as f:
        fmt = _decode_attr(f.attrs.get("pymorgan_format"), "")
        if fmt == "1D" or ("Zavg_R" in f and "pump" not in f and "Z_R" not in f):
            raise ValueError(
                f"File {p.name} is a 1D PyMORGAN HDF5 archive; use load_1D_hdf5."
            )
        pump = np.asarray(f["pump"][:], dtype=float)
        probe = np.asarray(f["probe"][:], dtype=float)
        delays = np.asarray(f["delays"][:], dtype=float)
        Z_R = np.asarray(f["Z_R"][:], dtype=float)
        raw_signal = np.asarray(f["raw_signal"][:], dtype=float) if "raw_signal" in f else None
        raw_interferogram = np.asarray(f["raw_interferogram"][:], dtype=float) if "raw_interferogram" in f else None
        raw_t1delays = np.asarray(f["raw_t1delays"][:], dtype=float) if "raw_t1delays" in f else None

        units_raw = _decode_attr(f.attrs.get("units_json"), "{}")
        try:
            units = json.loads(units_raw)
        except Exception:
            units = _units_2d("ps", "cm-1")

        freq_units = _decode_attr(f.attrs.get("freq_units"), "cm-1")
        datatype = _decode_attr(f.attrs.get("datatype"), "processed")
        in_progress = bool(f.attrs.get("in_progress", False))
        units["w0_cm1"] = float(f.attrs.get("w0_cm1", units.get("w0_cm1", 0.0)))
        units["dt1_fs"] = float(f.attrs.get("dt1_fs", units.get("dt1_fs", 0.0)))
        units["is_frequency_domain"] = bool(f.attrs.get("is_frequency_domain", units.get("is_frequency_domain", False)))

        orig_dt = _decode_attr(f.attrs.get("original_data_type"), "")
        if orig_dt and "original_data_type" not in units:
            units["original_data_type"] = orig_dt
        if "cal_level" in f.attrs:
            units["cal_level"] = _decode_attr(f.attrs["cal_level"])
        if "cal_source" in f.attrs:
            units["cal_source"] = _decode_attr(f.attrs["cal_source"])

        states_data = _load_mapping_group(f, "states_data")
        if states_data:
            units["states_data"] = states_data
        states_ss = _load_mapping_group(f, "states_signal_ss")
        if states_ss:
            units["states_signal_ss"] = states_ss
        det_probes = _load_mapping_group(f, "detector_probes")
        if det_probes:
            units["detector_probes"] = det_probes
        det_data = _load_mapping_group(f, "detector_data")
        if det_data:
            units["detector_data"] = det_data
        det_ss = _load_mapping_group(f, "detector_signal_ss")
        if det_ss:
            units["detector_signal_ss"] = det_ss

        if "state_names_json" in f.attrs:
            units["state_names"] = json.loads(_decode_attr(f.attrs["state_names_json"], "[]"))
        if "detector_names_json" in f.attrs:
            units["detector_names"] = json.loads(_decode_attr(f.attrs["detector_names_json"], "[]"))
        if "active_state" in f.attrs:
            units["active_state"] = _decode_attr(f.attrs["active_state"])
        if "active_detector" in f.attrs:
            units["active_detector"] = _decode_attr(f.attrs["active_detector"])
        if "raw_signal_ss" in f:
            units["raw_signal_ss"] = np.asarray(f["raw_signal_ss"][:], dtype=float)
        if "raw_interferogram_ss" in f:
            units["raw_interferogram_ss"] = np.asarray(f["raw_interferogram_ss"][:], dtype=float)
        if "nscans" in f.attrs:
            units["nscans"] = int(f.attrs["nscans"])

    return (
        Z_R,
        pump,
        probe,
        delays,
        units,
        freq_units,
        datatype,
        in_progress,
        raw_signal,
        raw_interferogram,
        raw_t1delays,
    )


def load_2D_hdf5(path: str | PathLike):
    """Load a FAIR 2D HDF5 archive into a fully restored :class:`~pymorgan.twoD.Dataset2D`."""
    import json

    import h5py

    from ..hdf5_meta import restore_dataset_history
    from .dataset import Dataset2D

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"2D HDF5 archive not found: {path}")

    (
        Z_R,
        pump,
        probe,
        delays,
        units,
        freq_units,
        datatype,
        in_progress,
        raw_signal,
        raw_interferogram,
        raw_t1delays,
    ) = read_2D_hdf5(p)

    with h5py.File(p, "r") as f:
        data_type = _decode_attr(f.attrs.get("data_type"), "HDF5_2D")
        source = _decode_attr(f.attrs.get("source"), str(p)) or str(p)
        Z_C = np.asarray(f["Z_C"][:], dtype=float) if "Z_C" in f else None
        shot_rotation = int(f.attrs.get("shot_rotation", 0) or 0)
        nscans = int(f.attrs.get("nscans", units.get("nscans", 1)) or 1)
        active_scan_indices = None
        if "active_scan_indices_json" in f.attrs:
            try:
                active_scan_indices = [int(i) for i in json.loads(_decode_attr(f.attrs["active_scan_indices_json"], "[]"))]
            except Exception:
                active_scan_indices = None

        ds = Dataset2D(
            Z_R,
            pump,
            probe,
            delays,
            units,
            freq_units=freq_units,
            source=source,
            data_type=data_type,
            datatype=datatype,
            in_progress=in_progress,
            raw_signal=raw_signal,
            raw_interferogram=raw_interferogram,
            raw_t1delays=raw_t1delays,
            cal_level=units.get("cal_level"),
            cal_source=units.get("cal_source"),
            is_frequency_domain=units.get("is_frequency_domain"),
            w0_cm1=units.get("w0_cm1"),
            dt1_fs=units.get("dt1_fs"),
            state_names=units.get("state_names"),
            states_data=units.get("states_data"),
            states_signal_ss=units.get("states_signal_ss"),
            active_state=units.get("active_state"),
            detector_names=units.get("detector_names"),
            detector_probes=units.get("detector_probes"),
            detector_data=units.get("detector_data"),
            detector_signal_ss=units.get("detector_signal_ss"),
            active_detector=units.get("active_detector"),
            raw_signal_ss=units.get("raw_signal_ss"),
            raw_interferogram_ss=units.get("raw_interferogram_ss"),
            nscans=nscans,
            active_scan_indices=active_scan_indices,
        )
        ds.shot_rotation = shot_rotation
        ds.Z_C = Z_C
        ds._is_corrected = Z_C is not None
        restore_dataset_history(f, ds)

    return ds



