"""Record the folder this build was made in, for the bundled executable (stdlib only).

The packaged GUI is built into ``dist/Koeiro/``, which holds no ``vc_models``: the AI
environment is installed beside the source tree instead. Recording that folder here
lets the executable find its worker without the user setting an environment variable,
and ``KOEIRO_ROOT`` still overrides it when the executable is moved.

Run before PyInstaller; ``Koeiro.spec`` bundles the file it writes.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.runtime_paths import ROOT_FILE  # noqa: E402  (needs the path above)


def main():
    target = ROOT / ROOT_FILE
    target.write_text(str(ROOT), encoding='utf-8')
    print(f'wrote {target.name}: {ROOT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
