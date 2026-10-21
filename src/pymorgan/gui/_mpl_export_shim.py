"""Self-healing loader for Matplotlib vector/raster export backends in frozen builds.

If an older standalone installation (e.g. v1.261020.1) is updated via the
Bleeding-Edge source sync without running a new full installer, PyInstaller's
frozen archive may lack the pure-Python backend modules (``backend_pdf``,
``backend_svg``, ``backend_ps``, ``backend_pgf``) and ``fontTools.subset``.
All compiled C extensions (``ft2font``, ``_backend_agg``, ``_path``, ``PIL``)
are already present in the frozen binary, so caching the missing pure-Python
modules in ``~/.config/pymorgan/mpl_backends/`` makes figure saving work
seamlessly without reinstalling.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import sys
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

_SHIM_INITIALIZED = False

# Ordered from leaf dependencies up to top-level export backends
_MPL_FALLBACK_FILES: tuple[tuple[str, str], ...] = (
    ("matplotlib._tight_bbox", "_tight_bbox.py"),
    ("matplotlib._type1font", "_type1font.py"),
    ("matplotlib._text_helpers", "_text_helpers.py"),
    ("matplotlib._mathtext_data", "_mathtext_data.py"),
    ("matplotlib._afm", "_afm.py"),
    ("matplotlib.dviread", "dviread.py"),
    ("matplotlib.texmanager", "texmanager.py"),
    ("matplotlib.backends.backend_mixed", "backends/backend_mixed.py"),
    ("matplotlib.backends._backend_pdf_ps", "backends/_backend_pdf_ps.py"),
    ("matplotlib.backends.backend_pdf", "backends/backend_pdf.py"),
    ("matplotlib.backends.backend_ps", "backends/backend_ps.py"),
    ("matplotlib.backends.backend_svg", "backends/backend_svg.py"),
    ("matplotlib.backends.backend_pgf", "backends/backend_pgf.py"),
)

_EXPORT_BACKEND_MODULES: tuple[str, ...] = (
    "matplotlib.backends.backend_agg",
    "matplotlib.backends.backend_pdf",
    "matplotlib.backends.backend_ps",
    "matplotlib.backends.backend_svg",
    "matplotlib.backends.backend_pgf",
)


class _MplShimFinder(importlib.abc.MetaPathFinder):
    """MetaPathFinder that serves cached pure-Python Matplotlib/fontTools modules."""

    def __init__(self, search_roots: list[Path]):
        self.search_roots = [Path(p) for p in search_roots]

    def find_spec(self, fullname: str, path=None, target=None):
        if fullname == "fontTools" or fullname.startswith(("matplotlib.", "fontTools.")):
            rel = fullname.replace(".", "/")
            for root_p in self.search_roots:
                py_file = root_p / f"{rel}.py"
                pkg_init = root_p / rel / "__init__.py"
                if py_file.is_file() or pkg_init.is_file():
                    parent_dir = str((root_p / rel).parent)
                    return importlib.machinery.PathFinder.find_spec(
                        fullname, [parent_dir], target
                    )
        return None


def _all_backends_importable() -> bool:
    for mod_name in _EXPORT_BACKEND_MODULES:
        try:
            importlib.import_module(mod_name)
        except Exception:
            return False
    return True


def _install_meta_finder(search_roots: list[Path]) -> None:
    for finder in sys.meta_path:
        if type(finder).__name__ == "_MplShimFinder":
            finder.search_roots = [Path(p) for p in search_roots]
            return
    sys.meta_path.insert(0, _MplShimFinder(search_roots))


def _ensure_fonttools_subset(cache_root: Path, timeout: float = 10.0) -> bool:
    """Ensure ``fontTools.subset`` and ``fontTools.ttLib`` are importable for Type 42 PDF/PS fonts."""
    try:
        import fontTools.subset  # noqa: F401
        import fontTools.ttLib  # noqa: F401

        return True
    except Exception:
        pass

    ft_extract_dir = cache_root / "fonttools_pkg"
    ft_pkg_dir = ft_extract_dir / "fontTools"

    try:
        import fontTools

        ft_ver = getattr(fontTools, "__version__", "")
    except Exception:
        ft_ver = ""

    if not (ft_pkg_dir / "subset" / "__init__.py").is_file():
        try:
            api_url = (
                f"https://pypi.org/pypi/fonttools/{ft_ver}/json"
                if ft_ver
                else "https://pypi.org/pypi/fonttools/json"
            )
            req = urllib.request.Request(
                api_url, headers={"User-Agent": "PyMORGAN-MplShim"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                meta = json.loads(resp.read().decode("utf-8"))
            wheel_url = None
            for entry in meta.get("urls", []):
                if entry.get("filename", "").endswith("none-any.whl"):
                    wheel_url = entry.get("url")
                    break
            if wheel_url:
                w_req = urllib.request.Request(
                    wheel_url, headers={"User-Agent": "PyMORGAN-MplShim"}
                )
                with urllib.request.urlopen(w_req, timeout=timeout * 2) as w_resp:
                    whl_bytes = w_resp.read()
                ft_extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(BytesIO(whl_bytes), "r") as zf:
                    for member in zf.namelist():
                        if member.startswith("fontTools/"):
                            zf.extract(member, ft_extract_dir)
        except Exception:
            pass

    if (ft_pkg_dir / "subset" / "__init__.py").is_file():
        # Purge partial frozen fontTools modules so the complete pure-Python package loads cleanly
        for k in [
            m for m in list(sys.modules) if m == "fontTools" or m.startswith("fontTools.")
        ]:
            sys.modules.pop(k, None)
        importlib.invalidate_caches()
        try:
            import fontTools.subset  # noqa: F401
            import fontTools.ttLib  # noqa: F401

            return True
        except Exception:
            return False
    return False


def _patch_mpl_source(rel_path: str, src: str) -> str:
    """Apply minimal compatibility guards to fetched Matplotlib backend files."""
    if rel_path in ("backends/backend_pdf.py", "backends/backend_svg.py"):
        # Avoid importing matplotlib.dates (which pulls in dateutil.rrule) just for UTC.
        src = src.replace(
            "from matplotlib.dates import UTC",
            "from datetime import timezone as _mpl_tz\nUTC = _mpl_tz.utc",
        )
    elif rel_path == "backends/_backend_pdf_ps.py":
        src = src.replace(
            "from fontTools import subset",
            "try:\n    from fontTools import subset\nexcept ImportError:\n    subset = None",
        )
    return src


def _load_module_with_fallback(mod_name: str, target_file: Path) -> bool:
    """Import ``mod_name``, falling back to direct file execution if needed."""
    try:
        importlib.import_module(mod_name)
        return True
    except Exception:
        if not target_file.is_file():
            return False
        try:
            spec = importlib.util.spec_from_file_location(mod_name, str(target_file))
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = mod
                spec.loader.exec_module(mod)
                parent_name, _, attr_name = mod_name.rpartition(".")
                if parent_name and parent_name in sys.modules:
                    setattr(sys.modules[parent_name], attr_name, mod)
                return True
        except Exception:
            sys.modules.pop(mod_name, None)
            return False
    return False


def ensure_mpl_export_backends(timeout: float = 8.0) -> bool:
    """Ensure Matplotlib PDF, SVG, PS/EPS, PGF, and Agg export backends are loaded."""
    global _SHIM_INITIALIZED
    if _SHIM_INITIALIZED and _all_backends_importable():
        return True

    if _all_backends_importable():
        _SHIM_INITIALIZED = True
        return True

    try:
        import matplotlib as mpl
        import matplotlib.backends as mpl_backends
    except Exception:
        return False

    mpl_ver = getattr(mpl, "__version__", "")
    if not mpl_ver:
        return False

    cache_root = Path.home() / ".config" / "pymorgan" / "mpl_backends"
    ver_root = cache_root / f"v{mpl_ver}"
    ver_dir = ver_root / "matplotlib"
    backends_dir = ver_dir / "backends"
    backends_dir.mkdir(parents=True, exist_ok=True)
    ft_extract_dir = cache_root / "fonttools_pkg"

    # Install MetaPathFinder and register fallback directories on package __path__
    _install_meta_finder([ver_root, ft_extract_dir])
    if hasattr(mpl, "__path__") and str(ver_dir) not in mpl.__path__:
        mpl.__path__.append(str(ver_dir))
    if hasattr(mpl_backends, "__path__") and str(backends_dir) not in mpl_backends.__path__:
        mpl_backends.__path__.append(str(backends_dir))

    # Ensure fontTools.subset is present for Type 42 font embedding
    has_subset = _ensure_fonttools_subset(cache_root, timeout=timeout)

    # Fetch any missing pure-Python Matplotlib modules for this exact Matplotlib version
    for mod_name, rel_path in _MPL_FALLBACK_FILES:
        target_file = ver_dir / rel_path
        if not target_file.is_file():
            try:
                importlib.import_module(mod_name)
                continue
            except Exception:
                pass
            url = (
                f"https://raw.githubusercontent.com/matplotlib/matplotlib/"
                f"v{mpl_ver}/lib/matplotlib/{rel_path}"
            )
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "PyMORGAN-MplShim"}
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw_src = resp.read().decode("utf-8")
                patched_src = _patch_mpl_source(rel_path, raw_src)
                target_file.parent.mkdir(parents=True, exist_ok=True)
                target_file.write_text(patched_src, encoding="utf-8")
            except Exception:
                pass

    importlib.invalidate_caches()

    # Load fallback files in dependency order
    for mod_name, rel_path in _MPL_FALLBACK_FILES:
        _load_module_with_fallback(mod_name, ver_dir / rel_path)

    if not has_subset:
        try:
            mpl.rcParams["pdf.fonttype"] = 3
            mpl.rcParams["ps.fonttype"] = 3
        except Exception:
            pass

    ok = True
    for mod_name in _EXPORT_BACKEND_MODULES:
        try:
            importlib.import_module(mod_name)
        except Exception:
            ok = False

    _SHIM_INITIALIZED = ok
    return ok
