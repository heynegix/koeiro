"""Explicit developer setup: one pinned official LLVC checkpoint, verified SHA256."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'ebfe8c0fdeb974a7eeb463b3abbc8ce42a0e3851'
SHA256 = 'cceb7ab9621f84d62d283725ae3281cacb04f762d6a088fe3e97c1a29d4b8c0e'
URL = f'https://huggingface.co/KoeAI/llvc/resolve/{REVISION}/models/checkpoints/llvc/G_500000.pth'


def main():
    folder = ROOT/'models/research_llvc'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder/'model.pth'
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != SHA256:
        temporary = target.with_suffix('.download')
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open('wb') as output:
            while chunk := response.read(1024*1024):
                output.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError('LLVC checkpoint checksum mismatch; no model was installed')
        temporary.replace(target)
    metadata = dict(name='Girl 01 / LLVC 8312', backend='llvc', sample_rate=16000,
                    license='MIT', source_url=URL, revision=REVISION, sha256=SHA256,
                    bytes=target.stat().st_size, target='LibriSpeech speaker 8312',
                    note='Japanese perceptual quality requires listening; no model training is performed')
    (folder/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
