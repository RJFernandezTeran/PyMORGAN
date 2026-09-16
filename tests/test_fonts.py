"""Tests for pymorgan.fonts module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from pymorgan.fonts import (
    _find_miktex_heros,
    _find_rinoh_heros,
    available_preferred_fonts,
    find_heros_fonts,
    install_fonts,
)


def test_find_miktex_heros():
    """Verify _find_miktex_heros runs without error and returns a list."""
    fonts = _find_miktex_heros()
    assert isinstance(fonts, list)


def test_find_rinoh_heros():
    """Verify _find_rinoh_heros locates font files when package is installed."""
    fonts = _find_rinoh_heros()
    assert len(fonts) > 0
    assert any("texgyreheros" in f.name for f in fonts)


def test_find_rinoh_heros_import_error():
    """Verify _find_rinoh_heros returns empty list on ImportError."""
    with patch.dict("sys.modules", {"rinoh_typeface_texgyreheros": None}):
        assert _find_rinoh_heros() == []


def test_find_heros_fonts_fallback():
    """Test that find_heros_fonts falls back to rinoh when miktex finds nothing."""
    fake_rinoh_fonts = [Path("/fake/texgyreheros-regular.otf")]
    with patch("pymorgan.fonts._find_miktex_heros", return_value=[]), patch(
        "pymorgan.fonts._find_rinoh_heros", return_value=fake_rinoh_fonts
    ):
        fonts = find_heros_fonts()
        assert fonts == fake_rinoh_fonts


def test_find_heros_fonts_prefer_miktex():
    """Test that find_heros_fonts prefers MiKTeX/TeX Live fonts if found."""
    fake_miktex_fonts = [Path("/miktex/texgyreheros-regular.otf")]
    fake_rinoh_fonts = [Path("/rinoh/texgyreheros-regular.otf")]
    with patch("pymorgan.fonts._find_miktex_heros", return_value=fake_miktex_fonts), patch(
        "pymorgan.fonts._find_rinoh_heros", return_value=fake_rinoh_fonts
    ):
        fonts = find_heros_fonts()
        assert fonts == fake_miktex_fonts


def test_install_fonts(tmp_path):
    """Test install_fonts copies files to target matplotlib font directory."""
    font_file = tmp_path / "texgyreheros-regular.otf"
    font_file.write_text("dummy font data")

    mpl_data_dir = tmp_path / "mpl_data"
    mpl_cache_dir = tmp_path / "mpl_cache"
    mpl_data_dir.mkdir()
    mpl_cache_dir.mkdir()
    (mpl_cache_dir / "fontlist-v330.json").write_text("{}")

    mock_fm = MagicMock()
    mock_fm.ttflist = []

    with (
        patch("pymorgan.fonts._find_miktex_heros", return_value=[]),
        patch("pymorgan.fonts._find_rinoh_heros", return_value=[font_file]),
        patch("matplotlib.get_data_path", return_value=str(mpl_data_dir)),
        patch("matplotlib.get_cachedir", return_value=str(mpl_cache_dir)),
        patch("matplotlib.font_manager._load_fontmanager", return_value=mock_fm),
    ):
        install_fonts()

        copied_font = mpl_data_dir / "fonts" / "ttf" / "texgyreheros-regular.otf"
        assert copied_font.exists()
        assert not (mpl_cache_dir / "fontlist-v330.json").exists()


def test_available_preferred_fonts():
    """Test available_preferred_fonts returns a list."""
    res = available_preferred_fonts()
    assert isinstance(res, list)


def test_are_fonts_installed_receipt_flag(tmp_path):
    """Test are_fonts_installed returns True when receipt flag exists."""
    from pymorgan.fonts import are_fonts_installed

    cache_dir = tmp_path / "mpl_cache"
    cache_dir.mkdir()
    flag = cache_dir / "pymorgan_fonts_installed"
    flag.touch()

    with patch("matplotlib.get_cachedir", return_value=str(cache_dir)):
        assert are_fonts_installed() is True


def test_are_fonts_installed_filesystem_match(tmp_path):
    """Test are_fonts_installed detects fonts on disk and writes receipt flag."""
    from pymorgan.fonts import are_fonts_installed

    cache_dir = tmp_path / "mpl_cache"
    cache_dir.mkdir()
    font_dir = tmp_path / "mpl_data" / "fonts" / "ttf"
    font_dir.mkdir(parents=True)
    (font_dir / "texgyreheros-regular.otf").touch()

    with (
        patch("matplotlib.get_cachedir", return_value=str(cache_dir)),
        patch("matplotlib.get_data_path", return_value=str(tmp_path / "mpl_data")),
    ):
        assert are_fonts_installed() is True
        assert (cache_dir / "pymorgan_fonts_installed").exists()


def test_are_fonts_installed_not_found(tmp_path):
    """Test are_fonts_installed returns False when fonts and flag are absent."""
    from pymorgan.fonts import are_fonts_installed

    cache_dir = tmp_path / "mpl_cache"
    cache_dir.mkdir()
    font_dir = tmp_path / "mpl_data" / "fonts" / "ttf"
    font_dir.mkdir(parents=True)

    mock_fm = MagicMock()
    mock_fm.ttflist = []

    with (
        patch("matplotlib.get_cachedir", return_value=str(cache_dir)),
        patch("matplotlib.get_data_path", return_value=str(tmp_path / "mpl_data")),
        patch("matplotlib.font_manager.fontManager", mock_fm),
    ):
        assert are_fonts_installed() is False


def test_ensure_fonts_installed_skips_when_installed():
    """Test ensure_fonts_installed skips installation if already installed."""
    from pymorgan.fonts import ensure_fonts_installed

    with patch("pymorgan.fonts.are_fonts_installed", return_value=True), patch(
        "pymorgan.fonts.install_fonts"
    ) as mock_install:
        result = ensure_fonts_installed()
        assert result is True
        mock_install.assert_not_called()


def test_launcher_main_skips_install_when_fonts_present():
    """Test launcher.main launches GUI directly without installing fonts if already present."""
    from pymorgan.launcher import main as launcher_main

    with (
        patch("pymorgan.launcher.are_fonts_installed", return_value=True),
        patch("pymorgan.launcher.install_fonts") as mock_install,
        patch("pymorgan.gui.app.main", return_value=0) as mock_gui,
    ):
        code = launcher_main(["prog"])
        assert code == 0
        mock_install.assert_not_called()
        mock_gui.assert_called_once_with(["prog"])


def test_launcher_main_installs_on_first_run():
    """Test launcher.main triggers font install on first run when fonts missing."""
    from pymorgan.launcher import main as launcher_main

    with (
        patch("pymorgan.launcher.are_fonts_installed", return_value=False),
        patch("pymorgan.launcher.install_fonts") as mock_install,
        patch("pymorgan.gui.app.main", return_value=0) as mock_gui,
    ):
        code = launcher_main(["prog"])
        assert code == 0
        mock_install.assert_called_once_with(quiet=False)
        mock_gui.assert_called_once_with(["prog"])


def test_launcher_main_handles_install_error_gracefully():
    """Test launcher.main continues to launch GUI even if font installation encounters an error."""
    from pymorgan.launcher import main as launcher_main

    with (
        patch("pymorgan.launcher.are_fonts_installed", return_value=False),
        patch("pymorgan.launcher.install_fonts", side_effect=OSError("Permission denied")),
        patch("pymorgan.gui.app.main", return_value=0) as mock_gui,
    ):
        code = launcher_main(["prog"])
        assert code == 0
        mock_gui.assert_called_once_with(["prog"])

