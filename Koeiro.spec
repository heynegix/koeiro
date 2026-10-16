# -*- mode: python ; coding: utf-8 -*-
import os
import sys

# Female DSP is a Windows x64 binary (the loader verifies the dll against the manifest
# beside it), so it only travels in a Windows build. The icon file is a Windows .ico.
DATA_FILES = [
    ('assets', 'assets'),
    ('src/gui/assets', 'src/gui/assets'),
    ('models/user_voices/user_3b11387911244c6e9a013b42fa88a458',
     'models/user_voices/user_3b11387911244c6e9a013b42fa88a458'),
]
# The folder this build was made in: dist/Koeiro/ holds no vc_models, so the executable
# reads this to find its AI environment. tools/write_build_root.py writes it before a
# local PyInstaller run; a clean checkout (CI) has none, and listing a missing file
# makes Analysis fail outright. Either way the distribution builders delete it.
if os.path.exists(os.path.join(SPECPATH, 'koeiro-root.txt')):
    DATA_FILES.append(('koeiro-root.txt', '.'))
# The nonvoice gate's models (tools/setup_release.py --nonvoice-models fetches them,
# CI runs that before freezing). A clean checkout has none, and Analysis fails on a
# missing path, so list the folder only when it is really there. Without the models
# the gate stays off; the AI voice route never depends on them.
if os.path.isdir(os.path.join(SPECPATH, 'models', 'nonvoice')):
    DATA_FILES.append(('models/nonvoice', 'models/nonvoice'))
if sys.platform == 'win32':
    DATA_FILES += [
        ('src/processors/native/female_dsp_x64.dll', 'src/processors/native'),
        ('src/processors/native/female_dsp_x64.json', 'src/processors/native'),
    ]

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=DATA_FILES,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Koeiro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets/icon.ico'] if sys.platform == 'win32' else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Koeiro',
)
