"""Unit tests for UniGE nsTA data loader and plot_counts functionality."""

from pathlib import Path
import numpy as np
import pytest
import scipy.io as sio
import synthetic

import pymorgan as pm
from pymorgan.oneD.load import is_dataset_file


def test_unige_nsta_file_detection(tmp_path):
    dat_path = tmp_path / "sample.dat"
    synthetic.make_synthetic_unige_nsta(dat_path)
    assert is_dataset_file("UniGE_nsTA", dat_path)
    assert not is_dataset_file("UniGE_nsTA", tmp_path / "nonexistent.dat")

    # Probe-only / calibration dataset
    probe_only = tmp_path / "probe_only.dat"
    probe_only.write_text("% pixel\tsignal\trms signal\treference\trms reference\n0\t800\t200\t3000\t900")
    assert not is_dataset_file("UniGE_nsTA", probe_only)


def test_unige_nsta_loader(tmp_path):
    dat_path = tmp_path / "sample.dat"
    synthetic.make_synthetic_unige_nsta(dat_path, n_delays=8, npixels=15)

    ds = pm.load_1D(dat_path, data_type="UniGE_nsTA")
    assert ds.Zavg_R.shape == (8, 15, 1)
    assert ds.delays.shape == (8,)
    assert ds.probe.shape == (15,)
    assert ds.counts is not None
    assert ds.counts.shape == (8,)
    assert np.all(ds.counts == 1000)
    assert ds.units["unitsT_ltx"] == "ns"
    assert ds.rms is not None
    assert ds.rms.shape == (8, 15)


def test_unige_nsta_zero_counts_filtering(tmp_path):
    dat_path = tmp_path / "sample_zero.dat"
    synthetic.make_synthetic_unige_nsta(dat_path, n_delays=8, npixels=15, zero_counts=True)

    ds = pm.load_1D(dat_path, data_type="UniGE_nsTA")
    assert ds.Zavg_R.shape == (7, 15, 1)
    assert ds.delays.shape == (7,)
    assert ds.counts.shape == (7,)


def test_unige_nsta_pix2lam_calibration(tmp_path):
    dat_path = tmp_path / "sample.dat"
    synthetic.make_synthetic_unige_nsta(dat_path, n_delays=5, npixels=10)
    lam = np.linspace(400.0, 700.0, 10)
    sio.savemat(tmp_path / "pix2lam.mat", {"lam": lam})

    ds = pm.load_1D(dat_path, data_type="UniGE_nsTA")
    assert np.allclose(ds.probe, lam)


def test_plot_counts(tmp_path):
    import matplotlib.pyplot as plt

    dat_path = tmp_path / "sample.dat"
    synthetic.make_synthetic_unige_nsta(dat_path, n_delays=6, npixels=10)
    ds = pm.load_1D(dat_path, data_type="UniGE_nsTA")

    fig, ax = plt.subplots()
    out_ax = ds.plot_counts(ax=ax)
    assert out_ax.get_yscale() == "log"
    assert "N_{total}" in out_ax.get_title()
    plt.close(fig)


def test_plot_counts_no_counts_raises(tmp_path):
    ds = pm.Dataset1D(
        Zavg_R=np.zeros((3, 4, 1)),
        delays=np.array([0.0, 1.0, 2.0]),
        probe=np.array([10.0, 20.0, 30.0, 40.0]),
        units={},
    )
    with pytest.raises(ValueError, match="No accumulation counts available"):
        ds.plot_counts()


def test_unige_nsta_edge_trimming(tmp_path):
    dat_path = tmp_path / "sample_noisy_edges.dat"
    delays_sec = np.linspace(-5e-7, 1e-4, 5)
    lines = [
        "% program = ps-ta 3.0.1",
        "% format = dl-mach-xo-delay-1 Delay, pixel, TA signal, rms, error, n samples",
    ]
    npixels = 10
    for i, t in enumerate(delays_sec):
        for p in range(npixels):
            if p == 0:
                sig, err = 0.0, 0.0  # empty
            elif p == 1:
                sig, err = -600.0, 500.0  # nonsense signal & noise (>500 mOD)
            elif p == npixels - 1:
                sig, err = 10.0, 50.0  # high noise edge (>20 mOD)
            else:
                sig, err = -5.0, 1.0  # valid signal
            lines.append(f"{t:.6e}\t{p}\t{sig:.6e}\t10.0\t{err:.6e}\t1000")

    dat_path.write_text("\n".join(lines), encoding="utf-8")
    ds = pm.load_1D(dat_path, data_type="UniGE_nsTA")
    assert ds.Zavg_R.shape[1] == npixels - 3  # pixels 0, 1, and N-1 trimmed


def test_shift_t0(tmp_path):
    dat_path = tmp_path / "sample_shift.dat"
    synthetic.make_synthetic_unige_nsta(dat_path, n_delays=5, npixels=10)
    ds = pm.load_1D(dat_path, data_type="UniGE_nsTA")

    orig_delays = ds.delays.copy()
    ds.shift_t0(5.0)
    assert np.allclose(ds.delays, orig_delays - 5.0)
    assert ds.Zavg_C is None

