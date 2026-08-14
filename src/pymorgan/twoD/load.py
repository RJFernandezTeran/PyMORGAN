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

    Z = np.zeros((len(pump), len(probe), len(delays)))
    for i in range(len(delays)):
        Z[:, :, i] = np.reshape(Zflat[:, i], (len(pump), len(probe)), order="F")

    return Z, pump, probe, delays, _units_2d("ps", "cm-1"), "cm-1"


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

    units = _units_2d("ps", "cm-1")
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


def register_map_dataset_glob(name: str, pattern: str) -> None:
    """Declare ``name`` as a file-based map format whose datasets match ``pattern``."""
    _MAP_DATATYPE_GLOBS[name] = pattern


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

