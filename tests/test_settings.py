"""Unit tests for settings backend, field schemas, coercion, and TOML persistence."""

from __future__ import annotations

from dataclasses import fields
from enum import Enum
from pathlib import Path

import pytest

import pymorgan as pm
from pymorgan.settings import Settings


def test_get_settings_returns_defaults():
    s = pm.get_settings()
    assert isinstance(s, Settings)
    assert s.n_contours == 40
    assert s.quick_plots is False


def test_update_settings_returns_new_instance():
    s1 = pm.update_settings(n_contours=16)
    assert s1.n_contours == 16
    assert pm.get_settings().n_contours == 16


def test_update_settings_validates_field_names():
    with pytest.raises(TypeError, match="unknown setting"):
        pm.update_settings(not_a_real_setting_name=42)


def test_every_field_round_trips_through_to_dict():
    original = pm.get_settings()
    restored = Settings.from_dict(original.to_dict())
    assert restored == original


def test_to_dict_covers_every_field_except_unset_optionals():
    s = pm.get_settings()
    data = s.to_dict()
    missing = {f.name for f in fields(s)} - set(data)
    assert all(getattr(s, name) is None for name in missing)


def test_to_dict_is_toml_friendly():
    data = pm.get_settings().to_dict()
    for key, value in data.items():
        assert not isinstance(value, Enum), key
        assert not isinstance(value, tuple | Path), key


def test_every_enum_field_is_coerced_from_its_string_value():
    for f in fields(Settings):
        default = f.default
        if not isinstance(default, Enum):
            continue
        updated = pm.update_settings(**{f.name: default.value})
        assert getattr(updated, f.name) is default, f.name


def test_save_and_load_toml(tmp_path):
    f = tmp_path / "settings.toml"
    pm.update_settings(n_contours=64, cmap="DkRd/Wh/DkBu")
    pm.save_settings(f)
    assert f.is_file()

    s_loaded = pm.load_settings(f)
    assert s_loaded.n_contours == 64
    assert s_loaded.cmap == "DkRd/Wh/DkBu"


def test_traces_cmap_setting_coercion():
    s = pm.update_settings(traces_cmap="red-purple-blue")
    assert s.traces_cmap == pm.settings.TracesCmap.RED_PURPLE_BLUE

    s2 = pm.update_settings(traces_cmap="rainbow reversed")
    assert s2.traces_cmap == pm.settings.TracesCmap.RAINBOW_REVERSED


def test_field_specs_contains_traces_cmap_in_common_tab():
    specs = Settings.field_specs()
    assert "traces_cmap" in specs
    assert specs["traces_cmap"]["tab"] == "common"
    assert specs["traces_cmap"]["choices"] == [
        "rainbow",
        "rainbow reversed",
        "red-purple-blue",
    ]


def test_sd_intensity_threshold_description():
    specs = Settings.field_specs()
    assert "sd_intensity_threshold" in specs
    assert specs["sd_intensity_threshold"]["label"] == "Relative intensity threshold"
    assert "fraction of maximum 2D intensity" in specs["sd_intensity_threshold"]["tooltip"]


def test_ensure_settings_file_generates_default_template(tmp_path):
    target = tmp_path / "settings.toml"
    assert not target.exists()

    created_path = pm.ensure_settings_file(target)
    assert created_path == target
    assert target.is_file()

    content = target.read_text(encoding="utf-8")
    assert "PyMORGAN Settings Configuration" in content
    assert 'default_datadir = ""' in content
    assert "[spectral_diffusion]" in content
    assert "sd_intensity_threshold = 0.5" in content

    loaded = pm.Settings.load(target)
    assert loaded.default_datadir == ""
    assert loaded.sd_intensity_threshold == 0.5


def test_ensure_settings_file_merges_new_settings_and_preserves_user_values(tmp_path):
    target = tmp_path / "settings.toml"
    # Create an existing user settings file with custom values and missing sections/keys
    user_content = """# My custom user comments
[plot]
n_contours = 75
profile = "poster"

[gui]
gui_theme = "Fusion Dark"
default_datadir = "/my/custom/datadir"
"""
    target.write_text(user_content, encoding="utf-8")

    pm.ensure_settings_file(target)

    content = target.read_text(encoding="utf-8")
    # User-modified values must be preserved
    assert 'gui_theme = "Fusion Dark"' in content
    assert 'default_datadir = "/my/custom/datadir"' in content
    assert 'profile = "poster"' in content
    assert "n_contours = 75" in content

    # Missing sections/keys with comments from template must be merged in
    assert "[chirp]" in content
    assert "parallel_fitting" in content
    assert "[spectral_diffusion]" in content
    assert "sd_intensity_threshold" in content

    loaded = pm.Settings.load(target)
    assert loaded.n_contours == 75
    assert loaded.gui_theme == "Fusion Dark"
def test_ensure_settings_file_issues_warning_on_creation(tmp_path):
    import warnings as _w

    import pytest

    target = tmp_path / "settings.toml"
    with pytest.warns(UserWarning, match="pymorgan-settings"):
        pm.ensure_settings_file(target)

    # Calling again on existing file should not issue a warning
    with _w.catch_warnings(record=True) as record:
        _w.simplefilter("always")
        pm.ensure_settings_file(target)
    assert len(record) == 0



