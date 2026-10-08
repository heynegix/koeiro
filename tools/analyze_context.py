"""Replay the recorded analysis inputs, comparing causal context window sizes."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.prosody.controller import ProsodyController
from src.prosody.parameters import ProsodyParameters,load_preset


def main():
    source=Path('recordings/v08/prosody_comparison/comparison.json')
    data=json.loads(source.read_text(encoding='utf-8'))
    frames=data['comparisons']['OFF']['trace']
    result={}
    for window in (100,200,300):
        c=ProsodyController(); p=replace(load_preset('Anime Light',ProsodyParameters(enabled=True)),history_ms=window)
        controls=[c.update(t['input_f0'],t['input_voiced'],t['input_energy'],.03,p) for t in frames]
        result[str(window)]=dict(pitch_std=float(np.std([c.quantized_pitch_delta for c in controls])),
            endings=c.tracker.ending_count,transitions=c.tracker.transitions,
            slope_std=float(np.std([c.pitch_slope for c in controls])))
    Path('validation/v08-history-comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
