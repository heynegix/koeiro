"""Package accepted, immutable source WAVs for the official Beatrice trainer."""
import argparse
import hashlib
import json
from pathlib import Path
import wave
import zipfile


def package(source, archive, evaluation):
    source, archive = Path(source).resolve(), Path(archive).resolve()
    records = [json.loads(line) for line in (source / 'metadata.jsonl').read_text(encoding='utf8').splitlines()]
    accepted = [r for r in records if r['status'] == 'accepted']
    if not accepted or not json.loads((source / 'validation_report.json').read_text())['passed']:
        raise ValueError('Dataset is empty or validator failed')
    manifest = []
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for record in accepted:
            path = (source / record['wav_path']).resolve()
            if not path.is_relative_to(source / 'wav'):
                raise ValueError('Accepted path is outside wav directory')
            with wave.open(str(path)) as w:
                duration = w.getnframes() / w.getframerate()
                if w.getnchannels() != 1 or w.getsampwidth() != 2 or not 2 <= duration <= 10:
                    raise ValueError(f'Invalid training WAV: {path.name}')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            name = f'train/own_female/{path.name}'
            z.write(path, name)
            manifest.append({'id':record['id'], 'file':name, 'sha256':digest, 'duration':duration})
        for index, path in enumerate(map(Path, evaluation), 1):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in {item['sha256'] for item in manifest}:
                raise ValueError('Evaluation audio duplicates a training clip')
            z.write(path, f'evaluation/input_{index:02d}.wav')
        z.writestr('accepted_manifest.json', json.dumps(manifest, indent=2))
    return {'clips':len(manifest), 'duration':sum(m['duration'] for m in manifest),
            'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(), 'evaluation_count':len(evaluation)}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--archive', required=True)
    p.add_argument('--evaluation', action='append', default=[])
    a = p.parse_args()
    print(json.dumps(package(a.source, a.archive, a.evaluation), indent=2))
