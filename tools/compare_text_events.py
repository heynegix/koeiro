"""Native same-WAV event comparisons and bounded transcript-free metadata."""
import argparse,json,sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.compare_prosody import render
from tools.compare_presets import read_wav,write_wav
from tools.compare_text_calibration import events,distribution
from src.prosody.parameters import ProsodyParameters,load_preset
from src.vc.beatrice_vst import BeatriceVSTBackend


def summarize(trace):
    records={}; metadata=[]
    for frame in trace:
        snapshot=frame.get('text_events',{})
        for event in snapshot.get('completed',[])+snapshot.get('events',[]): records[event['event_id']]=event
        if snapshot.get('waterfall',{}).get('event_id') is not None:
            metadata.append(dict(timestamp=frame['seconds'],phrase_id=frame['phrase_id'],
                waterfall=snapshot['waterfall'],effective_pitch=frame['effective_text_pitch'],
                active_events=snapshot.get('active_events',[])))
    groups={}
    for kind in ('QUESTION','FAREWELL','EXCLAMATION','SHORT_RESPONSE','CALLOUT'):
        group=[v for v in records.values() if v['event_type']==kind]
        groups[kind]=dict(count=len(group),survived=sum(v['event_survived'] for v in group),
            survival=sum(v['event_survived'] for v in group)/max(1,len(group)),
            mean_effective_duration_ms=float(np.mean([v['effective_duration_ms'] for v in group])) if group else None,
            mean_pitch=float(np.mean([v['mean_effective_pitch'] for v in group])) if group else None,
            max_pitch=max((v['max_effective_pitch'] for v in group),default=None),
            mean_gain=float(np.mean([v['mean_effective_gain'] for v in group])) if group else None,
            activation_latency_ms=[(v['activated_at']-v['created_at'])*1000 for v in group if v['activated_at'] is not None],
            context_to_effect_ms=[(v['first_effective_at']-v['created_at'])*1000 for v in group if v['first_effective_at'] is not None])
    requested=np.array([v['text_pitch_delta'] for v in trace]); effective=np.array([v['effective_text_pitch'] for v in trace])
    active=(abs(requested)>.001)&np.array([v['f0']>0 for v in trace])
    return dict(event_types=groups,events=list(records.values()),requested=distribution(requested),effective=distribution(effective),
        frame_survival=float(np.mean(abs(effective[active])>=.1249)) if active.any() else 0.,
        active_ratio=float(active.mean()),max_native_step=float(np.max(abs(np.diff([v['quantized_pitch_delta'] for v in trace]))))),metadata


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--asr',default='test-results/v092/asr_raw.json')
    parser.add_argument('--output',default='recordings/v092/event_comparison'); args=parser.parse_args()
    data=json.loads(Path(args.asr).read_text(encoding='utf-8')); source,rate=read_wav(Path(data['source']))
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    base=ProsodyParameters(enabled=True,engine='text_v3',text_strategy='events')
    single=events(data['conditions']['single']['partials']); dual=events(data['conditions']['dual']['partials'])
    choices=[('Rule_v2',replace(base,engine='rule_v2',text_strategy='hybrid'),None),
        ('v091',replace(base,text_strategy='hybrid'),single),('v092_Natural',load_preset('Natural',base),single),
        ('v092_Light',base,single),('v092_Expressive',load_preset('Anime Expressive',base),single),('v092_Dual_Light',base,dual)]
    backend=BeatriceVSTBackend(); backend.load(Path('models/girl_01')); backend.warmup()
    report=dict(source=data['source'],kind='Offline causal human WAV -> native JVS002 +4; no human listening',
        effective_definition='applied native control minus shadow audio-only control, not acoustic F0',comparisons={})
    try:
        for label,p,partials in choices:
            audio,stats=render(source,p,backend,partials); trace=stats.pop('trace'); summary,metadata=summarize(trace)
            summary.update(parameters=stats['parameters'],finite=bool(np.isfinite(audio).all()),peak=float(abs(audio).max()))
            report['comparisons'][label]=summary
            write_wav(out/(label+'.wav'),audio,rate)
            (out/(label+'.events.jsonl')).write_text('\n'.join(json.dumps(v,ensure_ascii=False) for v in metadata),encoding='utf-8')
            (out/(label+'.trace.json')).write_text(json.dumps(trace,ensure_ascii=False),encoding='utf-8')
            print(label,json.dumps({k:v for k,v in summary.items() if k not in ('events','parameters')}),flush=True)
    finally: backend.unload()
    (out/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
