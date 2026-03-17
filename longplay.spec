# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for LongPlay Studio V5
# Builds .app bundle with Rust + C++ audio engines

import os
import sys
import glob

block_cipher = None

BASE = os.path.dirname(os.path.abspath(SPEC))

# Collect native modules (.so files)
native_modules = []
for so in glob.glob(os.path.join(BASE, '*.so')):
    native_modules.append((so, '.'))

# Collect module files
module_files = []
for root, dirs, files in os.walk(os.path.join(BASE, 'modules')):
    for f in files:
        if f.endswith('.py'):
            src = os.path.join(root, f)
            dst = os.path.relpath(root, BASE)
            module_files.append((src, dst))

a = Analysis(
    ['gui.py'],
    pathex=[BASE],
    binaries=native_modules,
    datas=[
        ('Logo.jpg', '.'),
        ('modules', 'modules'),
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
        'numpy',
        'scipy',
        'scipy.signal',
        'soundfile',
        'pedalboard',
        'pyloudnorm',
        'pydub',
        'PyQt6',
        'PyQt6.QtWidgets',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtMultimedia',
        'PyQt6.QtMultimediaWidgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'IPython', 'jupyter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LongPlay Studio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='LongPlay Studio',
)

app = BUNDLE(
    coll,
    name='LongPlay Studio.app',
    icon=None,
    bundle_identifier='com.mrg.longplay-studio',
    info_plist={
        'CFBundleName': 'LongPlay Studio',
        'CFBundleDisplayName': 'LongPlay Studio V5',
        'CFBundleVersion': '5.9.0',
        'CFBundleShortVersionString': '5.9',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '11.0',
    },
)
