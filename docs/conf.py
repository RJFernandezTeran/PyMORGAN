"""Sphinx configuration for PyMORGAN documentation (PyData Sphinx Theme)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pymorgan  # noqa: E402

project = "PyMORGAN"
author = "Dr. Ricardo J. Fernández-Terán"
copyright = "2026, Dr. Ricardo J. Fernández-Terán (University of Geneva)"
version = pymorgan.__version__
release = pymorgan.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Bundle the compiled LaTeX PDF manual directly into the HTML root so it is
# downloadable at /main.pdf alongside the interactive web documentation.
html_extra_path = ["main.pdf"] if (Path(__file__).parent / "main.pdf").exists() else []

html_theme = "pydata_sphinx_theme"
html_title = f"PyMORGAN {version}"
html_logo = "../src/pymorgan/gui/icons/pirate-hat.png"
html_favicon = "../src/pymorgan/gui/icons/pymorgan.ico"

html_theme_options = {
    "github_url": "https://github.com/RJFernandezTeran/PyMORGAN",
    "navbar_align": "left",
    "show_nav_level": 2,
    "navigation_depth": 3,
    "show_toc_level": 2,
    "external_links": [
        {
            "name": "PDF Manual",
            "url": "https://rjfernandezteran.github.io/PyMORGAN/main.pdf",
        },
        {
            "name": "PyRATE-TA Docs",
            "url": "https://rjfernandezteran.github.io/PyRATE-TA/",
        },
    ],
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pymorgan/",
            "icon": "fa-solid fa-box",
        },
    ],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}
