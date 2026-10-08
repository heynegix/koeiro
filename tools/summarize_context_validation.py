"""Small auditable v0.8 report; raw WAV replay is never labelled human listening."""
import json
from pathlib import Path


def main():
    comparisons=json.loads(Path('recordings/v08/prosody_comparison/comparison.json').read_text(encoding='utf-8'))
    report={'presets':{},'trials':{},'quality':'PARTIAL: user could not clearly distinguish presets',
        'discord':'Live cuts reported; 78 ms ON subjectively good, AI drops remain in logs',
        'prosody_analysis_added_buffering_ms':0,'quality_priority_latency_budget_ms':500}
    for name,data in comparisons['comparisons'].items():
        trace=data['trace']
        report['presets'][name]=dict(parameters=data['parameters'],distribution=data['distribution'],
            finite=data['finite'],voiced_fraction=sum(t['f0']>0 for t in trace)/len(trace),
            invalid_f0=data['invalid_f0_count'],states=sorted(set(t['phrase_state'] for t in trace)))
    for path in sorted(Path('validation').glob('v08-*.json')):
        if path.name.endswith('.progress.json') or path.name.endswith('.settings.json') or path.name=='v08-context-summary.json':
            continue
        data=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data.get('cycles'),list): continue
        trials=[]
        for row in data['cycles']:
            ai=row['ai']; prosody=ai['prosody']
            trials.append(dict(rtf=ai.get('rtf'),inference=ai.get('inference'),callback=row['callback'],
                prosody_analysis=prosody.get('analysis'),prosody_total=prosody.get('process_total'),
                ai_underrun=ai['ai_underrun'],ai_overrun=ai['ai_overrun'],ai_drop=ai['dropped_chunks'],
                underflow=row['underflow'],overflow=row['overflow'],prosody_errors=prosody['errors'],
                prosody_queue_drop=prosody['queue_drop_chunks'],control=prosody.get('control'),
                worker_stopped=row.get('worker_stopped'),prosody_worker_stopped=row.get('prosody_worker_stopped')))
            populated=[o for o in row.get('observations',[]) if 'cpu_seconds' in o['ai'].get('prosody',{})]
            if len(populated)>1:
                first,last=populated[0],populated[-1]; duration=last['seconds']-first['seconds']
                trials[-1]['cpu_four_core_percent']={name:100*(b-a)/duration/4 for name,a,b in (
                    ('ai',first['ai']['cpu_seconds'],last['ai']['cpu_seconds']),
                    ('prosody',first['ai']['prosody']['cpu_seconds'],last['ai']['prosody']['cpu_seconds']),
                    ('parent',first['parent_cpu_seconds'],last['parent_cpu_seconds']))}
                trials[-1]['ram_mb']={name:[first['ai'][key]['ram_bytes']/1e6,last['ai'][key]['ram_bytes']/1e6]
                    for name,key in [('prosody','prosody')]}
                trials[-1]['ram_mb']['ai']=[first['ai']['ram_bytes']/1e6,last['ai']['ram_bytes']/1e6]
        report['trials'][path.name]=dict(input_kind=data.get('input_kind'),seconds=data.get('seconds'),
            cycles=trials,passed=data.get('passed'),crash=data.get('crash'),prosody_crash=data.get('prosody_crash'))
    Path('validation/v08-context-summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    # Windows consoles may use CP932; the saved UTF-8 report retains names.
    print(json.dumps(report,ensure_ascii=True,indent=2))


if __name__=='__main__': main()
