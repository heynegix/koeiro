"""Explicit LavaSR setup: repo clone, official weights, isolated vendor tree.

Recreates vc_models/post_lavasr/{repo,weights,vendor} exactly as the app
expects it, then verifies every weight against src/vc/lavasr_runtime.json.
No app-time downloads, no training, no audio device. Idempotent: steps that
are already done (matching SHA-256, existing clone, matching vendor stamp)
are skipped. Run with any Python 3.11+; downloads use the standard library.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/'vc_models/post_lavasr'
REPO_URL = 'https://github.com/ysharma3501/LavaSR.git'
HF_REPO = 'https://huggingface.co/YatharthS/LavaSR/resolve/main'
VENDOR_PACKAGES = ['git+https://github.com/langtech-bsc/vocos.git@matcha', 'encodec']


def venv_python(folder):
    if os.name == 'nt':
        return folder/'.venv/Scripts/python.exe'
    return folder/'.venv/bin/python'


def runtime_info():
    with (ROOT/'src/vc/lavasr_runtime.json').open(encoding='utf-8') as stream:
        return json.load(stream)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def ensure_repo():
    if (FOLDER/'repo/.git').is_dir():
        print('repo: already cloned, skipping')
        return
    FOLDER.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, str(FOLDER/'repo')], check=True)
    print('repo: cloned', REPO_URL)


def download_weights(info):
    weights = (ROOT/info['weights_dir']).resolve()
    if not weights.is_relative_to(FOLDER.resolve()):
        raise ValueError('Invalid LavaSR weights path')
    for name in info['sha256']:
        if '/' in name.replace('\\', '/') and '..' in Path(name).parts:
            raise ValueError('Unsafe weights path: ' + name)
        target = weights/name
        expected = info['sha256'][name]
        if target.is_file() and sha256(target) == expected:
            print('weights: present', name)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = HF_REPO + '/' + name
        print('weights: downloading', url)
        urllib.request.urlretrieve(url, str(target))
        if sha256(target) != expected:
            target.unlink(missing_ok=True)
            raise RuntimeError('Checksum mismatch after download: ' + name)
    print('weights: all %d files verified' % len(info['sha256']))


def ensure_worker_venv():
    base = venv_python(ROOT/'vc_models/meanvc2')
    if not base.is_file():
        raise RuntimeError('MeanVC2 worker venv missing: build vc_models/meanvc2/.venv first')
    ours = venv_python(FOLDER)
    if not ours.is_file():
        subprocess.run([str(base), '-m', 'venv', '--without-pip', str(FOLDER/'.venv')], check=True)
    out = subprocess.run([str(ours), '-c', 'import site,json;print(json.dumps(site.getsitepackages()))'],
                         capture_output=True, text=True, check=True)
    site_packages = json.loads(out.stdout)[0]
    (FOLDER/'.venv/Lib/site-packages' if os.name == 'nt' else Path(site_packages)).mkdir(parents=True, exist_ok=True)
    pointer = FOLDER/'.venv/Lib/site-packages/shared_cpu_runtime.pth' if os.name == 'nt' else Path(site_packages)/'shared_cpu_runtime.pth'
    pointer.write_text('import sys; sys.path.append(' + ascii(str(Path(site_packages))) + ')\n', encoding='ascii')
    print('worker venv: ready at', ours)
    return ours


def ensure_vendor(worker_python):
    marker = FOLDER/'vendor/.complete'
    stamp = '\n'.join(VENDOR_PACKAGES)
    if marker.is_file() and marker.read_text('utf-8') == stamp:
        print('vendor: already installed, skipping')
        return
    uv = shutil.which('uv')
    if not uv:
        raise RuntimeError('uv executable unavailable')
    subprocess.run([uv, 'pip', 'install', '--python', str(worker_python),
                    '--no-deps', '--target', str(FOLDER/'vendor'), *VENDOR_PACKAGES], check=True)
    marker.write_text(stamp, encoding='utf-8')
    print('vendor: installed', ', '.join(VENDOR_PACKAGES))


def check_only():
    info = runtime_info()
    weights = (ROOT/info['weights_dir']).resolve()
    missing = [name for name in info['sha256'] if not (weights/name).is_file()]
    mismatched = [name for name in info['sha256']
                  if (weights/name).is_file() and sha256(weights/name) != info['sha256'][name]]
    marker = FOLDER/'vendor/.complete'
    vendor_ok = marker.is_file() and marker.read_text('utf-8') == '\n'.join(VENDOR_PACKAGES)
    print(json.dumps(dict(repo=bool((FOLDER/'repo/.git').is_dir()), missing=missing,
                          mismatched=mismatched, vendor_ok=vendor_ok), indent=2))
    if missing or mismatched or not (FOLDER/'repo/.git').is_dir() or not vendor_ok:
        raise SystemExit(1)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true',
                        help='verify layout and checksums without downloading anything')
    args = parser.parse_args()
    if args.check_only:
        check_only()
        return
    info = runtime_info()
    ensure_repo()
    download_weights(info)
    worker = ensure_worker_venv()
    ensure_vendor(worker)
    print('LavaSR setup complete at', FOLDER)


if __name__ == '__main__':
    main()
