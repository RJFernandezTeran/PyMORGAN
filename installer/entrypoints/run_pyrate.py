"""Entrypoint for standalone PyRATE-TA executable."""

import importlib.abc
import importlib.machinery
import json
import sys
from pathlib import Path


def _activate_bleeding_edge_overlay() -> None:
    """If a synced bleeding-edge source overlay exists and matches or exceeds bundled version, load it."""
    overlay_dir = Path.home() / ".config" / "pymorgan" / "bleeding_edge_src"
    meta_file = overlay_dir / "sync_meta.json"
    if not meta_file.is_file() or not (overlay_dir / "pyrate_ta" / "__init__.py").is_file():
        return
    try:
        import pymorgan as _bundled_pm

        bundled_ver = getattr(_bundled_pm, "__version__", "0.0.0")
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("bundled_version_at_sync") != bundled_ver:
            return
        for k in [
            m
            for m in sys.modules
            if m in ("pymorgan", "pyrate_ta") or m.startswith(("pymorgan.", "pyrate_ta."))
        ]:
            del sys.modules[k]
    except Exception:
        return

    class _BleedingEdgeFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname in ("pymorgan", "pyrate_ta") or fullname.startswith(
                ("pymorgan.", "pyrate_ta.")
            ):
                search_path = [str(overlay_dir)] if path is None else list(path)
                return importlib.machinery.PathFinder.find_spec(fullname, search_path, target)
            return None

    sys.path.insert(0, str(overlay_dir))
    sys.meta_path.insert(0, _BleedingEdgeFinder())


if __name__ == "__main__":
    import multiprocessing
    import os

    multiprocessing.freeze_support()
    if len(sys.argv) > 1 and sys.argv[1] in ("-m", "-c"):
        sys.exit(2)
    try:
        _mpl_cfg = Path.home() / ".config" / "pymorgan" / "mplconfig"
        _mpl_cfg.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(_mpl_cfg)
    except Exception:
        pass
    _activate_bleeding_edge_overlay()
    from pyrate_ta.launcher import main

    sys.exit(main())
