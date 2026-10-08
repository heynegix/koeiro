"""Publish only completed experimental reference packages; no callback file reads."""
import json
from pathlib import Path
import re

from ..runtime_paths import asset_root

ROOT = asset_root()


def discover(root=ROOT):
    result = {}
    folder = Path(root)/'models/naturalness_candidates'
    for path in folder.glob('*/profile.json'):
        try:
            if not re.fullmatch(r'natural_(calm|bright|lower|centroid)', path.parent.name):
                continue
            if path.stat().st_size > 4096:
                continue
            info = json.loads(path.read_text('utf-8'))
            if info.get('schema') != 1 or info.get('status') != 'COMPLETE':
                continue
            if not all((path.parent/p).is_file() for p in ('reference.wav', 'fixed_embedding.npy', 'runtime.json')):
                continue
            for mode in ('none', 'energy', 'combined'):
                key = path.parent.name+'_'+mode
                label = {'none':'補完なし', 'energy':'強弱のみ', 'combined':'強弱＋微小ピッチ'}[mode]
                result[key] = dict(folder='naturalness_candidates/'+path.parent.name,
                    name='比較用 · '+info['name']+' · '+label, backend='meanvc2',
                    naturalness_candidate=True, phrase_repair=mode != 'none', repair_mode=mode,
                    model_buffer_ms=1600, grid_delay_ms=40, extra_delay_ms=1600 if mode != 'none' else 0,
                    startup_chunks=4, queue_chunks=10, mute_during_startup=True,
                    voice=0, pitch=0., formant=0.)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return result
