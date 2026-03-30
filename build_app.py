#!/usr/bin/env python3
"""
Build script for MRG LongPlay Studio V5.5 macOS .app distribution.

Story 5.6 — Epic 5: Polish & Production.

Usage:
    python3 build_app.py                # full build
    python3 build_app.py --rust-only    # rebuild Rust backend only
    python3 build_app.py --app-only     # skip Rust, build .app only
    python3 build_app.py --clean        # clean build artifacts
    python3 build_app.py --check        # verify build environment

Features:
    - Compiles Rust backend via maturin
    - Bundles Python + Rust + FFmpeg into .app
    - Apple Silicon native (arm64) support
    - Code signing preparation
    - Build verification and smoke test
"""

from __future__ import annotations

import glob
import os
import platform
import shutil
import subprocess
import sys
import time
from typing import List, Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
APP_NAME = "LongPlay Studio"
APP_VERSION = "5.10.0"
BUNDLE_ID = "com.mrg.longplay-studio"
MIN_MACOS = "13.0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUST_DIR = os.path.join(BASE_DIR, "rust")
DIST_DIR = os.path.join(BASE_DIR, "dist")
BUILD_DIR = os.path.join(BASE_DIR, "build")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[BUILD] {msg}")


def run(cmd: List[str], cwd: Optional[str] = None, check: bool = True) -> subprocess.CompletedProcess:
    log(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=False)


def check_tool(name: str) -> bool:
    return shutil.which(name) is not None


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def check_environment() -> bool:
    """Verify all required tools are available."""
    log("Checking build environment...")
    ok = True

    checks = {
        "python3": "Python 3.12+",
        "pip3": "pip",
        "cargo": "Rust toolchain",
        "maturin": "Maturin (PyO3 build)",
        "pyinstaller": "PyInstaller",
        "ffmpeg": "FFmpeg",
    }

    for tool, desc in checks.items():
        found = check_tool(tool)
        status = "OK" if found else "MISSING"
        log(f"  {desc:<30s} [{status}]")
        if not found:
            ok = False

    # check architecture
    arch = platform.machine()
    log(f"  Architecture: {arch}")
    if arch == "arm64":
        log("  Apple Silicon native build")
    else:
        log("  Intel build (Rosetta compatible)")

    # check Python version
    ver = sys.version_info
    log(f"  Python: {ver.major}.{ver.minor}.{ver.micro}")
    if ver.major < 3 or ver.minor < 11:
        log("  WARNING: Python 3.11+ recommended")

    return ok


def build_rust() -> bool:
    """Compile Rust backend via maturin."""
    log("=" * 50)
    log("Building Rust backend...")
    log("=" * 50)

    if not os.path.exists(RUST_DIR):
        log("ERROR: rust/ directory not found")
        return False

    try:
        run(["maturin", "develop", "--release"], cwd=RUST_DIR)
        log("Rust backend compiled successfully")

        # verify import
        result = subprocess.run(
            [sys.executable, "-c", "import longplay; print('OK:', dir(longplay))"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            log(f"Rust module verified: {result.stdout.strip()}")
            return True
        else:
            log(f"WARNING: Rust module import failed: {result.stderr.strip()}")
            log("Continuing with Python fallback")
            return True  # not fatal

    except subprocess.CalledProcessError as e:
        log(f"ERROR: Rust build failed: {e}")
        log("Continuing with Python fallback")
        return True  # not fatal


def install_dependencies() -> bool:
    """Install Python dependencies."""
    log("Installing Python dependencies...")
    req = os.path.join(BASE_DIR, "requirements.txt")
    if os.path.exists(req):
        try:
            run([sys.executable, "-m", "pip", "install", "-r", req, "-q"])
            return True
        except subprocess.CalledProcessError:
            log("WARNING: Some dependencies failed to install")
            return True
    return True


def build_app() -> bool:
    """Build .app bundle via PyInstaller."""
    log("=" * 50)
    log("Building macOS .app bundle...")
    log("=" * 50)

    spec_file = os.path.join(BASE_DIR, "longplay_v55.spec")

    # generate updated spec file
    _generate_spec(spec_file)

    try:
        run(["pyinstaller", "--clean", "--noconfirm", spec_file], cwd=BASE_DIR)
        app_path = os.path.join(DIST_DIR, f"{APP_NAME}.app")
        if os.path.exists(app_path):
            log(f"SUCCESS: {app_path}")
            size_mb = _dir_size_mb(app_path)
            log(f"App size: {size_mb:.1f} MB")
            return True
        else:
            log("ERROR: .app bundle not found after build")
            return False
    except subprocess.CalledProcessError as e:
        log(f"ERROR: PyInstaller build failed: {e}")
        return False


def _generate_spec(spec_path: str) -> None:
    """Generate updated PyInstaller spec file for V5.5."""
    # collect native modules
    so_files = glob.glob(os.path.join(BASE_DIR, "*.so"))
    so_files += glob.glob(os.path.join(BASE_DIR, "*.dylib"))

    native_binaries = ", ".join(f"('{f}', '.')" for f in so_files) if so_files else ""

    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-
# Auto-generated by build_app.py for LongPlay Studio V5.5
import os, glob

block_cipher = None
BASE = os.path.dirname(os.path.abspath(SPEC))

# Native modules
native_modules = [{native_binaries}]
for so in glob.glob(os.path.join(BASE, '*.so')):
    native_modules.append((so, '.'))
for dylib in glob.glob(os.path.join(BASE, '*.dylib')):
    native_modules.append((dylib, '.'))

a = Analysis(
    ['gui.py'],
    pathex=[BASE],
    binaries=native_modules,
    datas=[
        ('Logo.jpg', '.'),
        ('modules', 'modules'),
        ('gui', 'gui'),
    ],
    hiddenimports=[
        'longplay',
        'longplay_cpp',
        'modules.master',
        'modules.master.chain',
        'modules.master.rust_chain',
        'modules.master.equalizer',
        'modules.master.dynamics',
        'modules.master.imager',
        'modules.master.maximizer',
        'modules.master.limiter',
        'modules.master.loudness',
        'modules.master.analyzer',
        'modules.master.ai_assist',
        'modules.master.genre_profiles',
        'modules.master.match_eq',
        'modules.master.realtime_monitor',
        'modules.master.ab_compare',
        'modules.master.loudness_report',
        'modules.master.audio_io',
        'gui.models.track',
        'gui.models.commands',
        'gui.models.keyframes',
        'gui.models.effects',
        'gui.models.transitions',
        'gui.models.export_presets',
        'gui.models.autosave',
        'gui.widgets.spectrum_analyzer',
        'gui.widgets.wlm_meter',
        'gui.video.gpu_preview',
        'gui.styles_vintage',
        'gui.utils.profiler',
        'numpy',
        'scipy',
        'scipy.signal',
        'soundfile',
        'pedalboard',
        'pyloudnorm',
        'PyQt6',
        'PyQt6.QtWidgets',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtMultimedia',
        'PyQt6.QtMultimediaWidgets',
    ],
    hookspath=[],
    excludes=['tkinter', 'matplotlib', 'IPython', 'jupyter', 'test', 'tests', 'PySide6', 'PySide2'],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='{APP_NAME}',
    debug=False,
    strip=False,
    upx=False,
    console=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='{APP_NAME}',
)

app = BUNDLE(
    coll,
    name='{APP_NAME}.app',
    icon=None,
    bundle_identifier='{BUNDLE_ID}',
    info_plist={{
        'CFBundleName': '{APP_NAME}',
        'CFBundleDisplayName': '{APP_NAME} V{APP_VERSION}',
        'CFBundleVersion': '{APP_VERSION}',
        'CFBundleShortVersionString': '{APP_VERSION}',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '{MIN_MACOS}',
        'NSMicrophoneUsageDescription': 'LongPlay Studio needs microphone access for audio monitoring.',
    }},
)
"""

    with open(spec_path, "w") as f:
        f.write(spec_content)
    log(f"Generated spec file: {spec_path}")


def verify_build() -> bool:
    """Smoke test the built application."""
    log("Verifying build...")
    app_path = os.path.join(DIST_DIR, f"{APP_NAME}.app")

    if not os.path.exists(app_path):
        log("ERROR: .app not found")
        return False

    # check binary exists
    exe_path = os.path.join(app_path, "Contents", "MacOS", APP_NAME)
    if os.path.exists(exe_path):
        log(f"  Executable: OK ({exe_path})")
    else:
        log(f"  Executable: MISSING")
        return False

    # check Info.plist
    plist = os.path.join(app_path, "Contents", "Info.plist")
    if os.path.exists(plist):
        log(f"  Info.plist: OK")
    else:
        log(f"  Info.plist: MISSING")

    # check frameworks/libs
    frameworks = os.path.join(app_path, "Contents", "Frameworks")
    if os.path.exists(frameworks):
        count = len(os.listdir(frameworks))
        log(f"  Frameworks: {count} items")

    log("Build verification complete")
    return True


def clean() -> None:
    """Clean build artifacts."""
    log("Cleaning build artifacts...")
    for d in [BUILD_DIR, DIST_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
            log(f"  Removed: {d}")

    spec_gen = os.path.join(BASE_DIR, "longplay_v55.spec")
    if os.path.exists(spec_gen):
        os.remove(spec_gen)

    log("Clean complete")


def _dir_size_mb(path: str) -> float:
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = sys.argv[1:]

    if "--clean" in args:
        clean()
        return 0

    if "--check" in args:
        ok = check_environment()
        return 0 if ok else 1

    log("=" * 60)
    log(f"MRG LongPlay Studio V{APP_VERSION} — macOS Build")
    log("=" * 60)

    t0 = time.time()

    if not check_environment():
        log("WARNING: Some tools are missing. Build may be incomplete.")

    if "--app-only" not in args:
        if not build_rust():
            log("WARNING: Rust build skipped or failed")

    if "--rust-only" in args:
        log("Rust-only build complete")
        return 0

    install_dependencies()

    if not build_app():
        log("BUILD FAILED")
        return 1

    verify_build()

    elapsed = time.time() - t0
    log(f"\nBuild completed in {elapsed:.1f} seconds")
    log(f"Output: {os.path.join(DIST_DIR, APP_NAME + '.app')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
