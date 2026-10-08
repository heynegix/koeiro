"""Native same-WAV A/B/C and quantization attribution. No hardware audio opened."""
import argparse,json,sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.compare_prosody import render
from tools.compare_presets import read_wav,write_wav
from src.vc.beatrice_vst import BeatriceVSTBackend
from src.prosody.parameters import ProsodyParameters,load_preset


def distribution(values):
    a=np.asarray(values,dtype=float)
    return dict(mean=float(a.mean()),std=float(a.std()),p95_abs=float(np.percentile(abs(a),95)),max_abs=float(abs(a).max()))


def events(partials):
    result=[]
    for item in partials:
        ctx=dict(item['context']); ctx.pop('phrase_id',None); ctx['source_seconds']=item['audio_ms']/1000
        result.append(dict(ready_seconds=item.get('ready_seconds',item['audio_ms']/1000+item['processing_ms']/1000),context=ctx))
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--windows',default='test-results/v091/windows_raw.json')
    parser.add_argument('--legacy',default='validation/v09/asr-reazon-300-1000.json',help='Frozen v0.9 ASR events; preserved local baseline')
    parser.add_argument('--output',default='recordings/v091/calibration_comparison'); args=parser.parse_args()
    data=json.loads(Path(args.windows).read_text(encoding='utf-8')); source,rate=read_wav(Path(data['source']))
    latest=events(data['windows']['400']['partials'])
    old=json.loads(Path(args.legacy).read_text(encoding='utf-8'))
    legacy=events(old['partials'])
    base=ProsodyParameters(enabled=True,engine='text_v3',asr_window_ms=400)
    choices=[('Rule_v2',replace(base,engine='rule_v2'),None),
        ('v09_current',replace(base,text_strategy='legacy',asr_window_ms=1000),legacy)]
    choices.extend((name.replace(' ','_'),load_preset(name,base),latest) for name in ('Natural','Anime Light','Anime Expressive'))
    choices.extend((name.title(),replace(base,text_strategy=name),latest) for name in ('direct','modulation'))
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    report=dict(source=data['source'],listening='Unverified',kind='Offline causal same-WAV native Beatrice JVS002 +4',
        effective_definition='final native automation grid minus shadow audio-only grid; not measured acoustic F0',comparisons={})
    backend=BeatriceVSTBackend(); backend.load(Path('models/girl_01')); backend.warmup()
    try:
        for label,p,contexts in choices:
            audio,stats=render(source,p,backend,contexts); trace=stats.pop('trace')
            write_wav(out/(label+'.wav'),audio,rate)
            requested=np.array([v['text_pitch_delta'] for v in trace]); effective=np.array([v['effective_text_pitch'] for v in trace])
            active=(abs(requested)>.001)&np.array([v['f0']>0 for v in trace])
            stats.update(requested=distribution(requested),effective=distribution(effective),
                effective_on_requested=distribution(effective[active]) if active.any() else {},
                survival_rate=float(np.count_nonzero(active&(abs(effective)>=.1249))/max(1,np.count_nonzero(active))),
                active_ratio=float(active.mean()),effective_active_ratio=float(np.mean(abs(effective)>=.1249)),
                requested_frames=int(active.sum()),frames=len(trace),
                gain_requested=distribution([v['text_gain_db'] for v in trace]),
                gain_effective=distribution([v['effective_text_gain'] for v in trace]),
                max_native_grid_step=float(np.max(abs(np.diff([v['quantized_pitch_delta'] for v in trace])))),
                peak=float(abs(audio).max()),finite=bool(np.isfinite(audio).all()))
            (out/(label+'.trace.json')).write_text(json.dumps(trace,ensure_ascii=False),encoding='utf-8')
            report['comparisons'][label]=stats
            print(label,json.dumps({k:stats[k] for k in ('requested','effective','survival_rate','active_ratio')}),flush=True)
    finally: backend.unload()
    (out/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__': main()
