"""Same human WAV, delayed actual ASR partials, offline VC previews; no future text."""
import argparse
import json
from pathlib import Path
import sys
from dataclasses import replace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.compare_presets import read_wav,write_wav
from tools.compare_prosody import render
from src.vc.beatrice_vst import BeatriceVSTBackend
from src.prosody.parameters import ProsodyParameters,load_preset


def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',default='recordings/v03/beatrice_comparison/Original.wav')
    p.add_argument('--asr',default='validation/v09/asr-reazon-300-1000.json')
    p.add_argument('--output',default='recordings/v09/text_comparison'); args=p.parse_args()
    data,rate=read_wav(Path(args.input)); output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    results=json.loads(Path(args.asr).read_text(encoding='utf-8'))
    events=[]
    for item in results['partials']:
        ctx=dict(item['context']); ctx['source_seconds']=item['audio_ms']/1000
        events.append(dict(ready_seconds=item['audio_ms']/1000+item['processing_ms']/1000,context=ctx))
    backend=BeatriceVSTBackend(); backend.load(Path('models/girl_01')); backend.warmup()
    report=dict(input=args.input,asr=args.asr,kind='Offline same-WAV, causal delayed actual ASR results',
        listening='Unverified',model='Beatrice JVS002 +4',comparisons={})
    try:
        choices=[('prosody_off',ProsodyParameters(enabled=False)),
            ('rule_v2',load_preset('Anime Light',ProsodyParameters(enabled=True)))]
        choices.extend((name.replace(' ','_'),load_preset(name,ProsodyParameters(enabled=True,engine='text_v3')))
            for name in ('Natural','Anime Light','Anime Expressive'))
        for label,parameters in choices:
            audio,stats=render(data,parameters,backend,events if parameters.engine=='text_v3' else None)
            write_wav(output/(label+'.wav'),audio,rate)
            controls=stats.pop('trace'); pitch=np.array([c['text_pitch_delta'] for c in controls]); gain=np.array([c['text_gain_db'] for c in controls])
            stats.update(text_pitch_mean=float(pitch.mean()),text_pitch_p95_abs=float(np.percentile(abs(pitch),95)),
                text_pitch_max_abs=float(abs(pitch).max()),text_gain_mean=float(gain.mean()),
                text_active_frames=int(np.count_nonzero(abs(pitch)>.001)),frames=len(controls),
                peak=float(abs(audio).max()),finite=bool(np.isfinite(audio).all()))
            report['comparisons'][label]=stats
        (output/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2))
    finally: backend.unload()

if __name__=='__main__': main()
