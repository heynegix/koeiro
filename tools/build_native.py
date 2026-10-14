"""Build the small Female DSP native library (Windows x64 DLL / macOS ARM64 dylib)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform as _platform
import subprocess
import sys


def _sources(root):
    return [root / "src/processors/native/female_dsp.cpp",
            *sorted((root / "src/processors/native").glob("*.h")),
            *sorted((root / "third_party").rglob("*.h"))]


def _manifest(output, platform_label, compiler, sources, root, key):
    manifest = {"abi": 1, "platform": platform_label, "compiler": compiler,
                key: hashlib.sha256(output.read_bytes()).hexdigest(),
                "source_hash_normalization": "CRLF to LF",
                "sources": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in sources}}
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_windows(root):
    vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    install = subprocess.check_output([str(vswhere), "-latest", "-products", "*", "-requires",
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], text=True).strip()
    if not install:
        raise SystemExit("Install Visual Studio C++ Build Tools first")
    output = root / "src/processors/native/female_dsp_x64.dll"
    build = root / ".build/native"
    build.mkdir(parents=True, exist_ok=True)
    script = build / "compile.cmd"
    script.write_text(f'@echo off\ncall "{install}\\VC\\Auxiliary\\Build\\vcvars64.bat" >nul\n'
        f'cl /nologo /O2 /EHsc /MT /LD /std:c++17 /DNOMINMAX /I"{root / "third_party"}" '
        f'"{root / "src/processors/native/female_dsp.cpp"}" /Fo"{build / "female_dsp.obj"}" '
        f'/Fe"{output}" /link /IMPLIB:"{build / "female_dsp.lib"}"\nexit /b %errorlevel%\n', encoding="utf-8")
    subprocess.run(["cmd.exe", "/d", "/c", str(script)], cwd=build, check=True)
    sources = _sources(root)
    _manifest(output, "Windows x64", "MSVC /O2 /MT C++17", sources, root, "dll_sha256")
    print(f"Built {output} ({output.stat().st_size} bytes)")
    return output


def build_macos(root, arch=None):
    arch = arch or ("arm64" if _platform.machine().lower() in ("arm64", "aarch64") else "x64")
    output = root / f"src/processors/native/female_dsp_{arch}.dylib"
    build = root / ".build/native"
    build.mkdir(parents=True, exist_ok=True)
    cxx = os.environ.get("CXX", "clang++")
    # arch flag keeps local builds honest on Apple Silicon vs Intel runners.
    arch_flag = "-arch arm64" if arch == "arm64" else "-arch x86_64"
    cmd = (f"{cxx} -O2 -std=c++17 -shared -fPIC -DNOMINMAX "
           f"{arch_flag} -I\"{root / 'third_party'}\" "
           f"\"{root / 'src/processors/native/female_dsp.cpp'}\" -o \"{output}\"")
    print("$ " + cmd)
    subprocess.run(cmd, shell=True, cwd=build, check=True)
    sources = _sources(root)
    _manifest(output, f"macOS {arch}", f"{cxx} -O2 -shared -fPIC C++17", sources, root, "lib_sha256")
    print(f"Built {output} ({output.stat().st_size} bytes)")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("windows-x64", "mac-arm64", "mac-x64", "auto"),
                        default="auto", help="Target DSP library (default: host)")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    target = args.platform
    if target == "auto":
        if sys.platform == "darwin":
            target = "mac-arm64" if _platform.machine().lower() in ("arm64", "aarch64") else "mac-x64"
        elif os.name == "nt":
            target = "windows-x64"
        else:
            # Linux keeps using the Windows DLL only for tests; native Linux .so
            # is a follow-up. Default to the Windows instructions for now.
            target = "windows-x64"
    if target == "windows-x64":
        build_windows(root)
    elif target == "mac-arm64":
        build_macos(root, "arm64")
    elif target == "mac-x64":
        build_macos(root, "x64")
    else:
        raise SystemExit(f"Unknown platform: {target}")


if __name__ == "__main__":
    main()
