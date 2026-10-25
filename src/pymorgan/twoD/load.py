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

from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

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


def _stub_loader(name: str) -> Loader:
    """Build a placeholder loader that fails loudly until implemented."""

    def loader(path: str) -> LoaderResult:
        raise NotImplementedError(
            f"The {name!r} 2-D loader is not implemented yet (requested file: {path}). "
            f"Implement it here and register it via @register_map_loader({name!r})."
        )

    loader.__name__ = f"load_{name}"
    return loader


# Instrument-specific formats are recognised but not yet implemented.
for _name in ("UoS_2DIR", "RAL_RAW", "RAL_Proc"):
    register_map_loader(_name)(_stub_loader(_name))



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


def is_map_dataset_file(name: str, filepath: str | PathLike) -> bool:
    """Whether ``filepath`` is a valid file dataset for map format ``name``."""
    fn = _MAP_DATASET_FILE_FILTERS.get(name)
    if fn is None:
        return True
    try:
        return bool(fn(Path(filepath)))
    except OSError:
        return False


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
        _pymess_attr_str,
        _resolve_pymess_omega3_axis,
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
        meta_dict: dict[str, str] = {}
        if "metadata" in f:
            for k, v in f["metadata"].attrs.items():
                meta_dict[str(k)] = _pymess_attr_str(v)

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
                    if k.startswith("state_") and isinstance(sm_grp[k], h5py.Dataset)
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

        n_t1_or_w1, n_pix, n_t2 = raw_cube.shape
        n_t2_orig = n_t2

        # Extract optional per-scan 4D cube (N_t1_or_w1, N_pix, N_t2_orig, N_scans)
        raw_signal_ss: np.ndarray | None = None
        raw_interferogram_ss: np.ndarray | None = None
        if "data/individual_scans" in f and isinstance(f["data/individual_scans"], h5py.Dataset):
            indiv_arr = np.asarray(f["data/individual_scans"][:], dtype=np.float64)
            if indiv_arr.ndim == 4:
                # Commonly (N_scans, N_t2, N_t1_or_w1, N_pix) or (N_t1_or_w1, N_pix, N_t2, N_scans)
                if indiv_arr.shape[1:] == (n_t2_orig, n_t1_or_w1, n_pix):
                    raw_signal_ss = np.transpose(indiv_arr, (2, 3, 1, 0))
                elif indiv_arr.shape[:3] == (n_t1_or_w1, n_pix, n_t2_orig):
                    raw_signal_ss = indiv_arr
                elif indiv_arr.shape[2:] == (n_t1_or_w1, n_pix) and indiv_arr.shape[0] == n_t2_orig:
                    raw_signal_ss = np.transpose(indiv_arr, (2, 3, 0, 1))
            elif indiv_arr.ndim == 3 and n_t2_orig == 1:
                if indiv_arr.shape[1:] == (n_t1_or_w1, n_pix):
                    raw_signal_ss = np.transpose(indiv_arr, (1, 2, 0))[:, :, np.newaxis, :]
                elif indiv_arr.shape[:2] == (n_t1_or_w1, n_pix):
                    raw_signal_ss = indiv_arr[:, :, np.newaxis, :]

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
        probe, cal_kind = _resolve_pymess_omega3_axis(f, p, n_pix)
        if cal_kind == "dataset":
            cal_source = "PyMESS spectral calibration (cm-1)"
        elif cal_kind == "sibling":
            cal_source = "PyMESS sibling HDF5 calibration (cm-1)"
        elif cal_kind == "csv":
            cal_source = "CalibratedProbe.csv (cm-1)"
        else:
            cal_source = "Uncalibrated pixel indices"

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

            c_par = states_data[par_key]
            c_perp = states_data[perp_key]
            c_ma = (c_par + 2.0 * g_factor * c_perp) / 3.0
            states_data["Magic Angle"] = c_ma
            if is_freq_domain:
                denom = c_par + 2.0 * g_factor * c_perp
                with np.errstate(divide="ignore", invalid="ignore"):
                    c_aniso = np.where(np.abs(denom) > 1e-6, (c_par - g_factor * c_perp) / denom, np.nan)
                states_data["Anisotropy"] = c_aniso

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

        units = _units_2d("ps", "cm-1")
        units["cal_source"] = cal_source
        units["is_frequency_domain"] = is_freq_domain
        units["w0_cm1"] = w0_cm1
        units["dt1_fs"] = dt1_fs
        units["pymess_metadata"] = meta_dict
        units["pymess_subtype"] = subtype
        if raw_signal_ss is not None:
            units["raw_signal_ss"] = raw_signal_ss
            units["raw_interferogram_ss"] = raw_interferogram_ss
            units["nscans"] = int(raw_signal_ss.shape[-1])
        if states_data:
            units["state_names"] = list(states_data.keys())
            units["states_data"] = states_data
            units["active_state"] = list(states_data.keys())[0]

        info_lines = [
            f"Mode: {'Frequency-Domain 2D' if is_freq_domain else f'Time-Domain 2D (w0 = {w0_cm1:.1f} cm-1)'}"
        ]
        if "sample_id" in meta_dict and meta_dict["sample_id"]:
            info_lines.append(f"Sample: {meta_dict['sample_id']}")
        if "solvent" in meta_dict and meta_dict["solvent"]:
            info_lines.append(f"Solvent: {meta_dict['solvent']}")
        if "fast_modulation" in meta_dict and meta_dict["fast_modulation"]:
            info_lines.append(f"Fast Modulation: {meta_dict['fast_modulation']}")
        if "pump_polarization" in meta_dict and meta_dict["pump_polarization"]:
            info_lines.append(f"Pump Polarization: {meta_dict['pump_polarization']}")
        if "probe_polarization" in meta_dict and meta_dict["probe_polarization"]:
            info_lines.append(f"Probe Polarization: {meta_dict['probe_polarization']}")
        if "waveplate_positions" in meta_dict and meta_dict["waveplate_positions"]:
            info_lines.append(f"Waveplates: {meta_dict['waveplate_positions']}")
        if states_data:
            info_lines.append(f"Polarization States: {', '.join(states_data.keys())}")
        if "comments" in meta_dict and meta_dict["comments"]:
            info_lines.append(f"Comments: {meta_dict['comments']}")
        units["sample_info"] = "\n".join(info_lines)

    return (
        Z,
        pump,
        probe,
        t2delays_ps,
        units,
        "cm-1",
        datatype_str,
        in_progress,
        raw_signal,
        raw_interferogram,
        raw_t1delays,
    )


