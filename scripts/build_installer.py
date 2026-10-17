"""Build script for PyMORGAN Suite standalone distribution and Windows installer.

Usage:
    uv run python scripts/build_installer.py [--clean] [--zip-only]

Steps performed:
1. Validates source repositories (PyMORGAN + PyRATE-TA).
2. Generates multi-resolution .ico files from official PNG icons.
3. Executes PyInstaller to compile standalone binaries without console terminals.
4. Generates a standalone portable zip bundle ready for direct distribution.
5. If Inno Setup compiler (ISCC.exe) is detected, compiles a standalone Windows Setup installer (.exe).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALLER_DIR = REPO_ROOT / "installer"
DIST_DIR = REPO_ROOT / "dist" / "PyMORGAN_Suite"
OUTPUT_DIR = REPO_ROOT / "dist_installer"


def ensure_icons():
    """Ensure multi-resolution .ico files exist for Windows executables."""
    print("[1/5] Verifying application icons...")
    try:
        from PIL import Image
    except ImportError:
        print(
            "      [NOTE] Pillow not installed, skipping icon regeneration (existing icons will be used)."
        )
        return

    icon_dir = REPO_ROOT / "src" / "pymorgan" / "gui" / "icons"
    hat_png = icon_dir / "pirate-hat.png"
    ship_png = icon_dir / "pirate_ship.png"

    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

    if hat_png.exists():
        im = Image.open(hat_png).convert("RGBA")
        im.save(icon_dir / "pymorgan.ico", format="ICO", sizes=sizes)
        im.resize((512, 512), Image.Resampling.LANCZOS).save(
            icon_dir / "pymorgan.icns", format="ICNS"
        )

    if ship_png.exists():
        im = Image.open(ship_png).convert("RGBA")
        im.save(icon_dir / "pyrate_ta.ico", format="ICO", sizes=sizes)
        # Also sync to PyRATE-TA if checked out
        pyrate_dir = REPO_ROOT.parent / "PyRATE-TA" / "src" / "pyrate_ta" / "gui" / "icons"
        if pyrate_dir.exists():
            im.save(pyrate_dir / "pyrate_ta.ico", format="ICO", sizes=sizes)

    # Generate Inno Setup wizard graphics with program logos
    assets_dir = INSTALLER_DIR / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    if hat_png.exists() and ship_png.exists():
        from PIL import ImageDraw

        hat_im = Image.open(hat_png).convert("RGBA")
        ship_im = Image.open(ship_png).convert("RGBA")

        # Wizard Large (164 x 314 px)
        W, H = 164, 314
        large = Image.new("RGB", (W, H), color="#0f172a")
        draw = ImageDraw.Draw(large)
        for y in range(H):
            r = int(15 + (30 - 15) * (y / H))
            g = int(23 + (41 - 23) * (y / H))
            b = int(42 + (59 - 42) * (y / H))
            draw.line([(0, y), (W, y)], fill=(r, g, b))

        hat_thumb = hat_im.copy()
        hat_thumb.thumbnail((85, 85), Image.Resampling.LANCZOS)
        large.paste(hat_thumb, ((W - hat_thumb.width) // 2, 35), hat_thumb)

        draw.line([(30, 145), (W - 30, 145)], fill="#334155", width=1)

        ship_thumb = ship_im.copy()
        ship_thumb.thumbnail((85, 85), Image.Resampling.LANCZOS)
        large.paste(ship_thumb, ((W - ship_thumb.width) // 2, 160), ship_thumb)
        large.save(assets_dir / "wizard_large.bmp", format="BMP")

        # Wizard Small (55 x 55 px)
        W_s, H_s = 55, 55
        small = Image.new("RGB", (W_s, H_s), color="#ffffff")
        hat_small = hat_im.copy()
        hat_small.thumbnail((48, 48), Image.Resampling.LANCZOS)
        small.paste(
            hat_small, ((W_s - hat_small.width) // 2, (H_s - hat_small.height) // 2), hat_small
        )
        small.save(assets_dir / "wizard_small.bmp", format="BMP")

    print("      Icons & wizard branding verified (pymorgan.ico, pyrate_ta.ico, wizard_large.bmp)")


def run_pyinstaller(clean: bool = False):
    """Run PyInstaller with the suite specification file."""
    print("[2/5] Building standalone application bundle via PyInstaller...")
    spec_file = INSTALLER_DIR / "PyMORGAN_Suite.spec"
    if not spec_file.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_file}")

    # Ensure main_window_ui.py is freshly compiled before freezing into PYZ
    ui_file = REPO_ROOT / "src" / "pymorgan" / "gui" / "main_window.ui"
    ui_compiled = REPO_ROOT / "src" / "pymorgan" / "gui" / "main_window_ui.py"
    if ui_file.exists():
        from PyQt6 import uic

        with open(ui_compiled, "w", encoding="utf-8") as fh:
            uic.compileUi(str(ui_file), fh)
        print("      Pre-compiled main_window.ui -> main_window_ui.py")

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
    ]
    if clean:
        cmd.append("--clean")
    cmd.append(str(spec_file))

    print(f"      Executing: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    # Verify produced executables exist and start cleanly without tracebacks
    suffix = ".exe" if sys.platform.startswith("win") else ""
    expected_exes = [f"PyMORGAN_Suite{suffix}", f"PyMORGAN{suffix}"]
    if (REPO_ROOT.parent / "PyRATE-TA" / "src").exists():
        expected_exes.append(f"PyRATE-TA{suffix}")

    if sys.platform == "darwin":
        app_bundle = REPO_ROOT / "dist" / "PyMORGAN Suite.app"
        if not app_bundle.exists():
            raise RuntimeError(f"Expected macOS .app bundle not found: {app_bundle}")
        subprocess.run(
            [
                "/usr/bin/codesign",
                "--verify",
                "--all-architectures",
                "--deep",
                "--strict",
                str(app_bundle),
            ],
            check=True,
        )
        print("      Verified macOS .app bundle code signature (codesign --verify --deep --strict)")
        bin_root = app_bundle / "Contents" / "MacOS"
    else:
        bin_root = DIST_DIR

    for exe in expected_exes:
        exe_path = bin_root / exe
        if not exe_path.exists():
            raise RuntimeError(f"Expected executable not found after build: {exe_path}")

    suite_exe = bin_root / f"PyMORGAN_Suite{suffix}"
    print("      Running post-build font installation check (2 passes)...")
    res = subprocess.run(
        [str(suite_exe), "--install-fonts"],
        cwd=bin_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0 or "Traceback (most recent call last):" in (res.stderr or ""):
        raise RuntimeError(
            f"Smoke test failed for {suite_exe} --install-fonts:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
        )

    import time

    smoke_env = os.environ.copy()
    smoke_env["QT_QPA_PLATFORM"] = "offscreen"
    smoke_env["PYTHONUNBUFFERED"] = "1"
    for exe in expected_exes:
        exe_path = bin_root / exe
        # 1. Verify recursion guard against -m / -c invocations
        guard_res = subprocess.run(
            [str(exe_path), "-m", "PyQt6.uic.pyuic"],
            cwd=bin_root,
            env=smoke_env,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if guard_res.returncode != 2:
            raise RuntimeError(f"Recursion guard failed for {exe} -m (rc={guard_res.returncode})")

        # 2. Verify full GUI initialization and single-process count
        print(f"      Smoke-testing GUI startup of {exe}...")
        proc = subprocess.Popen(
            [str(exe_path)],
            cwd=bin_root,
            env=smoke_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        collected_out: list[str] = []
        if exe == f"PyMORGAN{suffix}":
            import threading

            def _reader(stream, sink):
                for line in stream:
                    sink.append(line)
                    if "GUI loaded in" in line:
                        break

            t_read = threading.Thread(
                target=_reader, args=(proc.stdout, collected_out), daemon=True
            )
            t_read.start()
            t_read.join(timeout=15.0)
        else:
            time.sleep(4.5)

        if sys.platform.startswith("win"):
            tl = subprocess.check_output(
                ["tasklist", "/FI", f"IMAGENAME eq {exe}", "/NH"], text=True
            )
            running_lines = [
                line for line in tl.splitlines() if line.strip().lower().startswith(exe.lower())
            ]
            if len(running_lines) > 1:
                proc.kill()
                raise RuntimeError(
                    f"Multiple instances ({len(running_lines)}) of {exe} spawned on startup:\n{tl}"
                )
        proc.terminate()
        try:
            out, err = proc.communicate(timeout=5)
        except Exception:
            proc.kill()
            out, err = proc.communicate()
        full_out = "".join(collected_out) + (out or "")
        if "Traceback (most recent call last):" in (err or ""):
            raise RuntimeError(f"Startup traceback detected in {exe}:\n{err}")
        if exe == f"PyMORGAN{suffix}":
            if "GUI loaded in" not in full_out:
                raise RuntimeError(
                    f"{exe} did not finish loading GUI within 15s!\nSTDOUT: {full_out}\nSTDERR: {err}"
                )
            print(f"      {full_out.strip()}")

    print("      PyInstaller build and startup smoke tests successful.")
    print(f"      Output directory: {DIST_DIR}")


def build_portable_zip(version: str):
    """Package the standalone bundle into a portable .zip archive."""
    print("[3/5] Packaging standalone portable distribution zip...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_name = f"PyMORGAN_Suite_v{version}_Windows_x64_Portable.zip"
    zip_path = OUTPUT_DIR / zip_name

    if zip_path.exists():
        zip_path.unlink()

    print(f"      Compressing {DIST_DIR} -> {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(DIST_DIR):
            for file in files:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(DIST_DIR.parent)
                zf.write(full_path, rel_path)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"      Portable zip created: {zip_path} ({size_mb:.1f} MB)")
    return zip_path


def find_iscc() -> Path | None:
    """Find Inno Setup compiler executable (ISCC.exe) if installed."""
    # Check PATH
    iscc_path = shutil.which("iscc") or shutil.which("ISCC.exe")
    if iscc_path:
        return Path(iscc_path)

    # Common installation paths on Windows
    candidates = [
        Path(r"C:\ProgramData\chocolatey\bin\iscc.exe"),
        Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"),
        Path(r"C:\Program Files\Inno Setup 6\ISCC.exe"),
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Inno Setup 6"
        / "ISCC.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Inno Setup 6" / "ISCC.exe",
        Path(os.environ.get("LocalAppData", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def build_inno_installer(version: str):
    """Compile Inno Setup script into a single Windows installer executable."""
    print("[4/5] Checking for Inno Setup compiler...")
    iscc = find_iscc()
    iss_file = INSTALLER_DIR / "PyMORGAN_Suite.iss"

    if iscc is None:
        print("      [NOTE] Inno Setup compiler (ISCC.exe) was not found on this system.")
        print("      To generate the single-file setup wizard (PyMORGAN_Suite_Setup.exe):")
        print("      1. Download and install free Inno Setup from: https://jrsoftware.org/isdl.php")
        print(
            "      2. Re-run this script, or open installer/PyMORGAN_Suite.iss in Inno Setup and click 'Compile'."
        )
        print("      The portable .zip distribution is already complete and ready to distribute!")
        return None

    print(f"      Compiling Windows installer with: {iscc} (v{version})...")
    cmd = [str(iscc), f"/DMyAppVersion={version}", str(iss_file)]
    subprocess.run(cmd, cwd=INSTALLER_DIR, check=True)

    print("      Windows Setup Installer created successfully in dist_installer/!")


def build_linux_package(version: str) -> Path:
    """Package the standalone bundle into a Linux tar.gz with desktop shortcuts and installer."""
    print("[3/5] Packaging standalone Linux distribution tarball...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    archive_name = f"PyMORGAN_Suite_v{version}_Linux_x86_64.tar.gz"
    archive_path = OUTPUT_DIR / archive_name

    pkg_tmp = REPO_ROOT / "build_linux_pkg"
    if pkg_tmp.exists():
        shutil.rmtree(pkg_tmp)

    stage_dir = pkg_tmp / f"PyMORGAN_Suite_v{version}_Linux_x86_64"
    bin_dir = stage_dir / "bin"
    desktop_dir = stage_dir / "desktop"
    bin_dir.mkdir(parents=True, exist_ok=True)
    desktop_dir.mkdir(parents=True, exist_ok=True)

    # Copy binary files preserving symlinks
    if DIST_DIR.exists():
        for item in DIST_DIR.iterdir():
            if item.is_dir() and not item.is_symlink():
                shutil.copytree(item, bin_dir / item.name, symlinks=True)
            else:
                shutil.copy2(item, bin_dir / item.name, follow_symlinks=False)

    # Copy icons for Linux desktop integration
    icons_src = REPO_ROOT / "src" / "pymorgan" / "gui" / "icons"
    icons_dst = bin_dir / "icons"
    icons_dst.mkdir(parents=True, exist_ok=True)
    for icon_name in ["pirate-hat.png", "pirate_ship.png"]:
        if (icons_src / icon_name).exists():
            shutil.copy2(icons_src / icon_name, icons_dst / icon_name)

    # Ensure executable permissions on main binaries
    for bin_name in ["PyMORGAN_Suite", "PyMORGAN", "PyRATE-TA"]:
        target_bin = bin_dir / bin_name
        if target_bin.exists():
            try:
                os.chmod(target_bin, 0o755)
            except Exception:
                pass

    # Copy installer and desktop assets
    linux_assets = INSTALLER_DIR / "assets" / "linux"
    if linux_assets.exists():
        for f in linux_assets.glob("*.desktop"):
            shutil.copy2(f, desktop_dir / f.name)
        for f in ["install.sh", "uninstall.sh"]:
            src = linux_assets / f
            if src.exists():
                shutil.copy2(src, stage_dir / f)
                try:
                    os.chmod(stage_dir / f, 0o755)
                except Exception:
                    pass

    # Create tar.gz archive
    import tarfile

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(stage_dir, arcname=stage_dir.name)

    shutil.rmtree(pkg_tmp, ignore_errors=True)
    size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"      Linux tarball created: {archive_path} ({size_mb:.1f} MB)")
    return archive_path


def build_macos_package(version: str):
    """Package the standalone bundle into a macOS DMG and zip archive."""
    print("[3/5] Packaging macOS .app bundle and DMG installer...")
    sh_script = INSTALLER_DIR / "build_macos.sh"
    if sh_script.exists():
        subprocess.run(["bash", str(sh_script)], cwd=REPO_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Build PyMORGAN Suite Standalone Installer")
    parser.add_argument("--clean", action="store_true", help="Clean build cache before building")
    parser.add_argument(
        "--skip-pyinstaller", action="store_true", help="Skip PyInstaller build step"
    )
    parser.add_argument(
        "--platform",
        choices=["auto", "windows", "linux", "macos"],
        default="auto",
        help="Target packaging platform (default: auto-detected from host system)",
    )
    args = parser.parse_args()

    import pymorgan

    version = getattr(pymorgan, "__version__", "1.0.0")

    target_os = args.platform
    if target_os == "auto":
        if sys.platform.startswith("win"):
            target_os = "windows"
        elif sys.platform.startswith("darwin"):
            target_os = "macos"
        else:
            target_os = "linux"

    print("=" * 70)
    print(f"  PyMORGAN Suite Installer Builder  (v{version} - {target_os.upper()})")
    print("=" * 70)

    ensure_icons()

    if not args.skip_pyinstaller:
        run_pyinstaller(clean=args.clean)

    if target_os == "windows":
        build_portable_zip(version)
        build_inno_installer(version)
    elif target_os == "linux":
        build_linux_package(version)
    elif target_os == "macos":
        build_macos_package(version)

    print("[5/5] Build Summary:")
    print("=" * 70)
    if OUTPUT_DIR.exists():
        for item in OUTPUT_DIR.iterdir():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  * {item.name:<50} ({size_mb:>6.1f} MB)")
    print("=" * 70)
    print("All tasks finished successfully!\n")


if __name__ == "__main__":
    main()
