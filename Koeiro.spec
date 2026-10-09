# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[('assets', 'assets'), ('src/gui/assets', 'src/gui/assets'),
           # Female DSP: the loader reads the manifest beside the DLL before
           # LoadLibrary, so both must ship as data files in the same folder.
           ('src/processors/native/female_dsp_x64.dll', 'src/processors/native'),
           ('src/processors/native/female_dsp_x64.json', 'src/processors/native'),
           ('models/user_voices/user_3b11387911244c6e9a013b42fa88a458', 'models/user_voices/user_3b11387911244c6e9a013b42fa88a458'),
           # The folder this build was made in: dist/Koeiro/ holds no vc_models, so
           # the executable reads this to find its AI environment. See
           # tools/write_build_root.py, which writes it before PyInstaller runs.
           ('koeiro-root.txt', '.')],
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
    icon=['assets/icon.ico'],
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
