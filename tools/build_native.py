"""Build the small Windows x64 DSP DLL from the vendored MIT headers (MSVC)."""
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
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
    sources = [root / "src/processors/native/female_dsp.cpp", *sorted((root / "src/processors/native").glob("*.h")),
               *sorted((root / "third_party").rglob("*.h"))]
    manifest = {"abi": 1, "platform": "Windows x64", "compiler": "MSVC /O2 /MT C++17",
                "dll_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "source_hash_normalization": "CRLF to LF",
                "sources": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in sources}}
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Built {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
