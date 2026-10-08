"""Explicit official model setup, no app-time downloads. Preserves source audio."""
from pathlib import Path
import tarfile
import urllib.request
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
URL='https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2'
FILES={'encoder-epoch-99-avg-1.int8.onnx','decoder-epoch-99-avg-1.onnx','joiner-epoch-99-avg-1.onnx','tokens.txt'}

def main():
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--vosk',action='store_true'); args=parser.parse_args()
    if args.vosk:
        import zipfile
        target=ROOT/'models/asr-vosk'; target.mkdir(parents=True,exist_ok=True)
        archive=ROOT/'models/asr-reazon/vosk-ja.zip'
        archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.is_file():
            urllib.request.urlretrieve('https://alphacephei.com/vosk/models/vosk-model-small-ja-0.22.zip',archive)
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                dest=(target/member.filename).resolve()
                if not dest.is_relative_to(target.resolve()): raise ValueError('Unsafe model path')
            z.extractall(target)
        print('Installed official Vosk Japanese small'); return
    target=ROOT/'models/asr-reazon'; target.mkdir(parents=True,exist_ok=True)
    archive=target/'source.tar.bz2'
    if not archive.exists(): urllib.request.urlretrieve(URL,archive)
    hashes={}
    with tarfile.open(archive) as tar:
        for member in tar:
            name=Path(member.name).name
            if name not in FILES or not member.isfile(): continue
            with tar.extractfile(member) as source, (target/name).open('wb') as output:
                digest=hashlib.sha256()
                while chunk:=source.read(1024*1024): output.write(chunk); digest.update(chunk)
            hashes[name]=digest.hexdigest()
    if set(hashes)!=FILES: raise ValueError('Official archive missing required model files')
    (target/'metadata.json').write_text(json.dumps(dict(source=URL,sha256=hashes,
        license='Apache-2.0 (ReazonSpeech k2-v2)',mode='offline rolling partial',sample_rate=16000),indent=2),encoding='utf-8')
    print('Installed required INT8 encoder and decoder/joiner files only')

if __name__=='__main__': main()
