"""Unit tests for all data loaders: PDAT, UniGE fsTA, HARPIA-TA, Helios-TA, MESS TRIR/TRUVIS, and calibration loading."""

from pathlib import Path
import numpy as np
import pytest
import synthetic

import pymorgan as pm
from pymorgan.oneD.load import (
    dataset_glob,
    describe_dataset,
    is_dataset_dir,
    is_dataset_file,
    is_directory_format,
    mess_recalc_average,
    parse_scan_selection,
)
from pymorgan.cal.load import load_experimental_spectrum


# --------------------------------------------------------------------------- #
#                             Registry & PDAT                                  #
# --------------------------------------------------------------------------- #
def test_registry_lists_pdat():
    assert "PDAT" in pm.available_loaders()


def test_read_pdat_shapes(dataset, pdat):
    assert dataset.Zavg_R.ndim == 3
    assert dataset.delays.shape == pdat["delays"].shape
    assert dataset.probe.shape == pdat["probe"].shape
    assert dataset.n_detectors == 1
    assert dataset.units["unitsL_lbl"] == "Wavenumber"
    assert dataset.units["unitsT_ltx"] == "ps"


def test_unknown_type_raises():
    with pytest.raises(KeyError):
        pm.load_1D("x.pdat", data_type="DOES_NOT_EXIST")


@pytest.mark.parametrize("name", ["UniGE_FLUPSold", "UniGE_FLUPSnew"])
def test_stub_loaders_raise(name, tmp_path):
    f = tmp_path / "x.dat"
    f.write_text("0")
    with pytest.raises(NotImplementedError):
        pm.load_1D(f, data_type=name)


def test_registry_shim_matches():
    from pymorgan.oneD import registry

    assert registry.available_loaders() == pm.available_loaders()


def test_register_custom_loader():
    @pm.register_loader("PYTEST_TMP")
    def _reader(path):
        z = np.zeros((2, 3, 1))
        return (
            z,
            np.array([0.0, 1.0]),
            np.array([1.0, 2.0, 3.0]),
            {"unitsZ": "mOD"},
            np.nan,
            None,
            None,
        )

    assert "PYTEST_TMP" in pm.available_loaders()
    d = pm.load_1D("ignored", data_type="PYTEST_TMP")
    assert d.Zavg_R.shape == (2, 3, 1)


def test_sample_info_without_noise(tmp_path):
    info = synthetic.make_synthetic_pdat(tmp_path / "a.pdat")
    ds = pm.load_1D(info["path"], data_type="PDAT")
    import re

    text_html = ds.sample_info().replace("<br>", "\n")
    text = re.sub("<[^<]+?>", "", text_html).replace("&nbsp;", " ")
    lines = text.split("\n")
    line1, line2, line3, line4 = lines[0], lines[1], lines[2], lines[3]

    assert line1 == "No single scan data available"
    assert "SNR: Inf" in line2
    assert "Res.:" in line2 and "cm⁻¹" in line2
    assert f"Delays: {ds.delays.size}, range:" in line3
    assert line4 == "Sample Info: No additional sample information available."


def test_sample_info_with_pdatn(tmp_path):
    info = synthetic.make_synthetic_pdat(tmp_path / "a.pdat")
    synthetic.make_synthetic_pdatn(info["path"], sigma=0.5)

    ds = pm.load_1D(info["path"], data_type="PDAT")
    assert np.allclose(np.asarray(ds.Zstdv, dtype=float), 0.5)

    import re

    text_html = ds.sample_info().replace("<br>", "\n")
    text = re.sub("<[^<]+?>", "", text_html).replace("&nbsp;", " ")
    lines = text.split("\n")
    line1, line2 = lines[0], lines[1]
    assert "Avg. Noise: 0.5000 mOD" in line1
    assert "Max Noise: 0.5000 mOD" in line1
    assert "SNR:" in line2 and "SNR: Inf" not in line2


def test_mess_sample_info_reading(tmp_path):
    dataset_dir = tmp_path / "Test_Dataset_100000"
    dataset_dir.mkdir()
    np.savetxt(dataset_dir / "Test_Dataset_100000_delays.csv", np.array([0.0, 1.0, 5.0]), delimiter=",")
    np.savetxt(dataset_dir / "Test_Dataset_100000_wavenumbers.csv", np.array([1000.0, 1010.0]), delimiter=",")
    np.savetxt(dataset_dir / "Test_Dataset_100000_Nspectra.csv", np.array([1]), fmt="%d")
    np.savetxt(dataset_dir / "Test_Dataset_100000_slowModulation.csv", np.array([1]), fmt="%d")

    sig = np.zeros((3, 2))
    np.savetxt(dataset_dir / "Test_Dataset_100000_signal_sp0_sm0_du0.csv", sig, delimiter=",")

    # Without sample info file
    ds_no_info = pm.load_1D(str(dataset_dir), data_type="MESS_TRIR")
    assert "No additional sample information available." in ds_no_info.sample_info()
    assert "No additional sample information available." in ds_no_info.sample_info_console()

    # With sample info file
    sample_info_content = (
        "Sample ID:\tTestSample\n\n"
        "Solvent: Methanol\n"
        "Pump WL: 400 nm\n\n"
        "Additional comments:\n"
        "First comment line\n"
        "Second comment line\n"
    )
    (dataset_dir / "Test_Dataset_100000_SampleInfo.txt").write_text(sample_info_content, encoding="utf-8")

    ds = pm.load_1D(str(dataset_dir), data_type="MESS_TRIR")
    info_html = ds.sample_info()
    info_console = ds.sample_info_console()

    assert "Sample Info:" in info_html
    assert "Sample Info:" in info_console
    assert "Sample ID: TestSample" in info_console
    assert "Solvent: Methanol" in info_console
    assert "Pump WL: 400 nm" in info_console
    assert "Additional comments: First comment line, Second comment line" in info_console


def test_noise_array_from_pdatn(tmp_path):
    info = synthetic.make_synthetic_pdat(tmp_path / "a.pdat")
    synthetic.make_synthetic_pdatn(info["path"], sigma=0.5)

    ds = pm.load_1D(info["path"], data_type="PDAT")
    noise = ds.noise_array()
    assert noise is not None
    assert np.allclose(np.asarray(noise, dtype=float), 0.5)
    assert np.asarray(noise).shape == np.asarray(ds.Z).shape


def test_noise_array_is_none_without_noise_information(tmp_path):
    info = synthetic.make_synthetic_pdat(tmp_path / "b.pdat")
    ds = pm.load_1D(info["path"], data_type="PDAT")
    assert ds.noise_array() is None


def test_private_noise_alias_still_works(tmp_path):
    info = synthetic.make_synthetic_pdat(tmp_path / "c.pdat")
    synthetic.make_synthetic_pdatn(info["path"], sigma=0.25)

    ds = pm.load_1D(info["path"], data_type="PDAT")
    assert np.allclose(ds._noise_array(), ds.noise_array())


# --------------------------------------------------------------------------- #
#                               UniGE fsTA                                    #
# --------------------------------------------------------------------------- #
def test_unige_fsta_loader_registered():
    loaders = pm.available_loaders()
    assert "UniGE_fsTA" in loaders


def test_unige_fsta_file_filter(tmp_path):
    ts_info = synthetic.make_synthetic_unige_fsta(tmp_path / "dataset_sample.dat")
    assert is_dataset_file("UniGE_fsTA", ts_info["path"])

    cal_info = synthetic.make_synthetic_unige_cal(tmp_path / "cal_dir")
    assert not is_dataset_file("UniGE_fsTA", cal_info["sample_path"])
    assert not is_dataset_file("UniGE_fsTA", cal_info["blank_path"])


def test_load_unige_fsta_synthetic_file(tmp_path):
    info = synthetic.make_synthetic_unige_fsta(tmp_path / "test_data.dat", nscans=2, npixels=60)
    ds = pm.load_1D(info["path"], data_type="UniGE_fsTA")

    assert ds.data_type == "UniGE_fsTA"
    assert ds.Zavg_R.ndim == 3
    assert ds.delays.shape == (5,)
    assert ds.nscans == 2
    assert ds.Zss_R is not None
    assert ds.Zss_R.ndim == 4
    assert ds.Zstdv is not None
    assert ds.Zstdv.ndim == 3

    assert ds.units["unitsL_lbl"] == "Wavelength"
    assert ds.units["unitsL_ltx"] == "nm"
    assert ds.units["unitsT_ltx"] == "ps"
    assert ds.units["unitsZ"] == "x1E3"

    assert np.isfinite(ds.Zavg_R).any()
    assert not np.isnan(ds.Zavg_R).all()


def test_unige_fsta_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        pm.load_1D("non_existent_unige_fsta_file.dat", data_type="UniGE_fsTA")


def test_unige_fsta_demo_directory_discovery():
    demo_dir = Path(r"C:\Users\ricar\switchdrive\Ambizione UniGE\Scripts\testData\UniGE_TA\demo")
    if not demo_dir.is_dir():
        pytest.skip("Demo dataset directory not available on local path")

    dat_files = sorted(demo_dir.glob("*.dat"))
    assert len(dat_files) > 0

    for fpath in dat_files:
        if is_dataset_file("UniGE_fsTA", fpath):
            ds = pm.load_1D(fpath, data_type="UniGE_fsTA")
            assert ds.data_type == "UniGE_fsTA"
            assert ds.Zavg_R.ndim == 3
            assert len(ds.delays) > 0
            assert len(ds.probe) > 0
            assert np.isfinite(ds.Zavg_R).any()


def test_unige_fsta_calibrated_probe_precedence(tmp_path):
    info = synthetic.make_synthetic_unige_fsta(tmp_path / "test_prec.dat", nscans=2, npixels=60, create_mat=True)

    csv_probe = np.linspace(400.0, 800.0, 60)
    np.savetxt(tmp_path / "CalibratedProbe.csv", csv_probe, delimiter=",", fmt="%.6f")

    ds = pm.load_1D(info["path"], data_type="UniGE_fsTA")
    expected_trimmed = csv_probe[25:-11]
    assert np.allclose(ds.probe, expected_trimmed)


def test_unige_fsta_no_calib_folder_fallback(tmp_path):
    standalone_dir = tmp_path / "standalone"
    info = synthetic.make_synthetic_unige_fsta(
        standalone_dir / "isolated.dat", nscans=2, npixels=60, create_mat=False
    )

    ds = pm.load_1D(info["path"], data_type="UniGE_fsTA")
    assert ds.data_type == "UniGE_fsTA"
    assert len(ds.probe) == 24
    assert np.allclose(ds.probe, np.arange(26, 50, dtype=float))


# --------------------------------------------------------------------------- #
#                                HARPIA-TA                                    #
# --------------------------------------------------------------------------- #
def test_harpia_loaders_registered():
    loaders = pm.available_loaders()
    assert "HARPIA_TA" in loaders


def test_load_harpia_synthetic_file(tmp_path):
    info = synthetic.make_synthetic_harpia_dataset(tmp_path / "sample_harpia.dat", nscans=2)
    ds = pm.load_1D(info["path"], data_type="HARPIA_TA")

    assert ds.data_type == "HARPIA_TA"
    assert ds.Zavg_R.ndim == 3
    assert ds.Zavg_R.shape == (5, 10, 1)
    assert ds.delays.shape == (5,)
    assert ds.probe.shape == (10,)
    assert ds.nscans == 2
    assert ds.Zss_R is not None
    assert ds.Zss_R.shape == (5, 10, 1, 2)
    assert ds.Zstdv is not None
    assert ds.Zstdv.shape == (5, 10, 1)

    assert ds.units["unitsL_lbl"] == "Wavelength"
    assert ds.units["unitsL_ltx"] == "nm"
    assert ds.units["unitsT_ltx"] == "ps"
    assert ds.units["unitsZ"] == "x1E3"

    assert np.isclose(ds.probe[0], 300.0)
    assert np.isclose(ds.probe[-1], 500.0)
    assert np.isclose(ds.delays[0], -2.0)
    assert np.isclose(ds.delays[-1], 10.0)

    assert np.isfinite(ds.Zavg_R).any()
    assert not np.isnan(ds.Zavg_R).all()


def test_harpia_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        pm.load_1D("non_existent_harpia_file.dat", data_type="HARPIA_TA")


def test_harpia_invalid_file_raises(tmp_path):
    invalid_file = tmp_path / "bad.dat"
    invalid_file.write_text("Not a HARPIA file\nHeader line 1\nHeader line 2\nLine 3\n")
    with pytest.raises(ValueError, match="valid HARPIA"):
        pm.load_1D(invalid_file, data_type="HARPIA_TA")


# --------------------------------------------------------------------------- #
#                                Helios TA                                    #
# --------------------------------------------------------------------------- #
def test_helios_loaders_registered():
    loaders = pm.available_loaders()
    assert "Helios_TA" in loaders


def test_helios_file_filter(tmp_path):
    info = synthetic.make_synthetic_helios_dataset(tmp_path / "exp_folder", main_name="run_a")
    folder = info["folder"]

    main_file = folder / "run_a.csv"
    scan_file = folder / "run_a_scan1.csv"
    bg_file = folder / "run_a-bg.csv"
    chirp_file = folder / "run_a-chirp.csv"
    pdat_file = folder / "run_a-CORRECTED.pdat"

    assert is_dataset_file("Helios_TA", main_file) is True
    assert is_dataset_file("Helios_TA", scan_file) is False
    assert is_dataset_file("Helios_TA", bg_file) is False
    assert is_dataset_file("Helios_TA", chirp_file) is False
    assert is_dataset_file("Helios_TA", pdat_file) is False


def test_load_helios_by_file_path(tmp_path):
    info = synthetic.make_synthetic_helios_dataset(
        tmp_path / "exp_dir", main_name="run_data", nscans=3, ndel=15, npix=20
    )
    ds = pm.load_1D(info["main_file"], data_type="Helios_TA")

    assert ds.data_type == "Helios_TA"
    assert ds.Zavg_R.ndim == 3
    assert ds.Zavg_R.shape == (15, 20, 1)
    assert ds.delays.shape == (15,)
    assert ds.probe.shape == (20,)
    assert ds.nscans == 3
    assert ds.Zss_R is not None
    assert ds.Zss_R.shape == (15, 20, 1, 3)
    assert ds.Zstdv is not None
    assert ds.Zstdv.shape == (15, 20, 1)

    assert ds.units["unitsL_lbl"] == "Wavelength"
    assert ds.units["unitsL_ltx"] == "nm"
    assert ds.units["unitsT_ltx"] == "ps"
    assert ds.units["unitsZ"] == "x1E3"

    assert np.isfinite(ds.Zavg_R).all()


def test_load_helios_by_folder_path(tmp_path):
    info = synthetic.make_synthetic_helios_dataset(
        tmp_path / "exp_dir", main_name="exp_dir", nscans=3, ndel=15, npix=20
    )
    ds = pm.load_1D(info["folder"], data_type="Helios_TA")

    assert ds.Zavg_R.shape == (15, 20, 1)
    assert ds.nscans == 3
    assert ds.scan_ids == [1, 2, 3]


def test_load_helios_multi_dataset_folder(tmp_path):
    folder = tmp_path / "multi_exp"
    info_a = synthetic.make_synthetic_helios_dataset(folder, main_name="sample_a", nscans=2)
    info_b = synthetic.make_synthetic_helios_dataset(folder, main_name="sample_b", nscans=4)

    ds_b = pm.load_1D(info_b["main_file"], data_type="Helios_TA")
    assert ds_b.nscans == 4
    assert ds_b.scan_ids == [1, 2, 3, 4]

    ds_a = pm.load_1D(info_a["main_file"], data_type="Helios_TA")
    assert ds_a.nscans == 2
    assert ds_a.scan_ids == [1, 2]


def test_helios_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        pm.load_1D("non_existent_helios_file.csv", data_type="Helios_TA")


def test_helios_background_correction_with_nans(tmp_path):
    info = synthetic.make_synthetic_helios_dataset(
        tmp_path / "exp_nan", main_name="exp_nan", nscans=2, ndel=10, npix=15
    )
    ds = pm.load_1D(info["main_file"], data_type="Helios_TA")

    pre_zero_mask = ds.delays <= 0.0
    ds.Zavg_R[pre_zero_mask, 0, 0] = np.nan
    if ds.Zss_R is not None:
        ds.Zss_R[pre_zero_mask, 0, 0, :] = np.nan

    ds.background_correct(tmin=ds.delays[0], tmax=0.0)

    assert np.isclose(ds.bkg_avg[0, 0], 0.0)
    assert not np.isnan(ds.bkg_avg[0, 0])

    pos_mask = ds.delays > 0.0
    assert np.isfinite(ds.Zavg_C[pos_mask, 0, 0]).all()


# --------------------------------------------------------------------------- #
#                                 MESS                                        #
# --------------------------------------------------------------------------- #
@pytest.fixture
def mess_trir(tmp_path):
    return synthetic.make_synthetic_mess(
        tmp_path / "SYN_TRIR_120000", kind="TRIR", nscans=4, n_slowmod=3
    )


@pytest.fixture
def mess_truvis(tmp_path):
    return synthetic.make_synthetic_mess(
        tmp_path / "SYN_TA_130000", kind="TRUVIS", nscans=4, n_slowmod=1
    )


def test_registry_lists_mess():
    loaders = pm.available_loaders()
    assert "MESS_TRIR" in loaders
    assert "MESS_TRUVIS" in loaders


def test_mess_are_directory_formats():
    assert is_directory_format("MESS_TRIR")
    assert is_directory_format("MESS_TRUVIS")
    assert not is_directory_format("PDAT")
    assert dataset_glob("MESS_TRIR") is None
    assert dataset_glob("PDAT") == "*.pdat"


def test_detector_and_describe(mess_trir):
    folder = mess_trir["folder"]
    assert is_dataset_dir("MESS_TRIR", folder)
    assert not is_dataset_dir("MESS_TRIR", folder + "/temp")
    info = describe_dataset("MESS_TRIR", folder)
    assert info["n_spectra"] == 1
    assert info["n_slowmod"] == 3
    assert "NONE" in info["anisotropy_modes"]


def test_load_unige_cal_synthetic(tmp_path):
    info = synthetic.make_synthetic_unige_cal(tmp_path / "cal_folder")

    exp_data = load_experimental_spectrum(info["sample_path"], cal_type_code=2)
    assert exp_data.file_path == info["sample_path"]
    assert len(exp_data.detector_data) == 1
    assert len(exp_data.detector_data[0]) == info["npixels"]
    assert np.isfinite(exp_data.detector_data[0]).all()


def test_load_unige_cal_demo_directory_discovery():
    demo_calib = Path(r"C:\Users\ricar\switchdrive\Ambizione UniGE\Scripts\testData\UniGE_TA\demo\calib")
    if not demo_calib.is_dir():
        pytest.skip("Demo calibration directory not available on local path")

    dat_files = sorted(demo_calib.glob("*.dat"))
    assert len(dat_files) > 0

    for fpath in dat_files:
        exp_data = load_experimental_spectrum(fpath, cal_type_code=2)
        assert len(exp_data.detector_data) == 1
        assert len(exp_data.detector_data[0]) > 0
        assert np.isfinite(exp_data.detector_data[0]).all()


def test_read_uos_irpp_synthetic_two_detectors(tmp_path):
    info = synthetic.make_synthetic_uos_irpp(tmp_path / "uos_dataset", n_detectors=2, n_scans=2, n_delays=10)
    ds = pm.load_1D(info["folder"], data_type="UoS_IRpp")

    assert ds.data_type == "UoS_IRpp"
    assert ds.n_detectors == 2
    assert ds.Z.shape == (10, 96, 2)
    assert ds.delays.shape == (10,)
    assert ds.probe.shape == (96, 2)
    assert ds.nscans == 2


def test_read_uos_irpp_synthetic_one_detector(tmp_path):
    info = synthetic.make_synthetic_uos_irpp(tmp_path / "uos_single_det", n_detectors=1, n_scans=1, n_delays=10)
    ds = pm.load_1D(info["folder"], data_type="UoS_IRpp")

    assert ds.data_type == "UoS_IRpp"
    assert ds.n_detectors == 1
    assert ds.Z.shape == (10, 96, 1)
    assert ds.delays.shape == (10,)
    assert ds.probe.shape == (96,)


def test_read_uos_irpp_real_dataset_discovery():
    uos_root = Path(r"c:\Users\ricar\switchdrive\Ambizione UniGE\Scripts\testData\UoS")
    if not uos_root.is_dir():
        pytest.skip("UoS test directory not present")

    candidates = [
        f.parent for f in uos_root.rglob("*.2D")
        if f.stat().st_size > 0 and (f.parent / (f.stem + ".DT")).is_file()
    ]
    if not candidates:
        pytest.skip("No valid UoS test datasets found")

    ds = pm.load_1D(candidates[0], data_type="UoS_IRpp")
    assert ds.data_type == "UoS_IRpp"
    assert ds.Z.ndim == 3
    assert ds.delays.ndim == 1
    assert ds.n_detectors in (1, 2)


# --------------------------------------------------------------------------- #
#                               Exported TXT                                  #
# --------------------------------------------------------------------------- #
def test_registry_lists_exported_txt():
    assert "Exported_TXT" in pm.available_loaders()


def test_read_exported_txt_synthetic(tmp_path):
    folder = tmp_path / "exp_txt_dataset"
    folder.mkdir()

    time_file = folder / "time.txt"
    time_file.write_text("# time / ps\n-1.0\n0.0\n1.0\n10.0\n")

    probe_file = folder / "wavelength.txt"
    probe_file.write_text("# wavelength / nm\n500.0\n510.0\n520.0\n")

    ta_file = folder / "TA.txt"
    ta_file.write_text(
        "# Transient Absorption / mOD\n"
        "0.1 0.2 0.3\n"
        "0.4 0.5 0.6\n"
        "0.7 0.8 0.9\n"
        "1.0 1.1 1.2\n"
    )

    assert is_dataset_dir("Exported_TXT", folder) is True

    ds = pm.load_1D(folder, data_type="Exported_TXT")
    assert ds.data_type == "Exported_TXT"
    assert ds.delays.shape == (4,)
    assert ds.probe.shape == (3,)
    assert ds.Zavg_R.shape == (4, 3, 1)
    assert ds.units["unitsT_ltx"] == "ps"
    assert ds.units["unitsL_ltx"] == "nm"

    # Also test loading via single file path inside folder
    ds_file = pm.load_1D(ta_file, data_type="Exported_TXT")
    assert ds_file.Zavg_R.shape == (4, 3, 1)


def test_read_exported_txt_fig4_dataset():
    fig4_path = Path(r"C:\Users\ricar\Downloads\DATA_FIG4\4CN")
    if not fig4_path.is_dir():
        pytest.skip("FIG4 4CN dataset not found on disk")

    assert is_dataset_dir("Exported_TXT", fig4_path) is True
    ds = pm.load_1D(fig4_path, data_type="Exported_TXT")
    assert ds.data_type == "Exported_TXT"
    assert ds.delays.shape == (425,)
    assert ds.probe.shape == (520,)
    assert ds.Zavg_R.shape == (425, 520, 1)


