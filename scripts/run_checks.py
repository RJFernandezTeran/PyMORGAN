"""Quick end-to-end smoke check for PyMORGAN.

Run::

    python scripts/run_checks.py

Exercises the whole package on self-contained synthetic data (so it does not
depend on files under testData/): the 1-D load/process/plot/analyse pipeline,
steady-state spectra, and the settings round-trip. Prints a PASS/FAIL line per
check and exits non-zero if any fail.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
# Put the repo root first so ``import pymorgan`` resolves to the in-tree
# source regardless of how the script is invoked or what is installed in
# the active environment (a stale site-packages copy otherwise shadows it).
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
import synthetic  # noqa: E402

import pymorgan as pm  # noqa: E402


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    truth = synthetic.make_synthetic_pdat(tmp / "syn.pdat")
    csv = synthetic.make_spectrum_csv(tmp / "abs.csv")
    p2 = synthetic.make_synthetic_p2dat(tmp / "syn.p2dat")

    pm.set_settings(pm.Settings())
    pm.apply_style()

    results: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        results.append((name, bool(cond)))
        print(f"[{'PASS' if cond else 'FAIL'}] {name}")

    import matplotlib.pyplot as plt

    # load
    d = pm.load_1D(truth["path"], data_type="PDAT")
    check("load: PDAT shapes", d.Zavg_R.ndim == 3 and d.delays.size > 0)

    # process
    d.background_correct(tmin=-20, tmax=-1)
    check("process: baseline removed", np.allclose(d.bkg_avg, truth["baseline"], atol=1e-6))

    # plot (all five)
    d.plot_contour(Zscale=80)
    d.plot_surface()
    d.plot_spectra([0.5, 1, 5, 20], doSmooth=1)
    _, t, Y = d.plot_kinetics([truth["centre"]], plotStyle="-")
    Sfit = np.random.RandomState(0).randn(d.probe.size, 2)
    d.plot_species_spectra(Sfit, [12.0, 120.0], [0.5, 5.0], [False, False], "Sequential")
    plt.close("all")
    check("plot: five plotters", Y.shape[0] == t.shape[0])

    # steady-state
    sp = pm.load_spectrum(csv["path"], kind="absorption")
    check("steady-state: auto-delimiter + units", sp.x_units == "cm^{-1}")
    d.plot_spectra([1, 5], doSmooth=1, Abs=sp.as_overlay_dict())
    plt.close("all")

    # settings
    p = tmp / "s.toml"
    pm.update_settings(profile="poster")
    pm.save_settings(p)
    check("settings: TOML round-trip", pm.Settings.load(p).profile.value == "poster")

    # 2-D pipeline
    d2 = pm.load_2D(p2["path"], data_type="P2DAT")
    d2.background_correct(reference=-1)
    out = d2.plot_map(0.5, show_colorbar=True)
    plt.close("all")
    check("2-D: load + process + map", d2.Z.shape[2] == p2["delays"].size and out.cbar is not None)

    check("no PumpProbe import", not any(m.startswith("PumpProbe") for m in sys.modules))

    n_fail = sum(1 for _, ok in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
