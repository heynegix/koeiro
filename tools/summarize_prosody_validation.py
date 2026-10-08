"""Generate an auditable compact v0.5 summary from raw reports."""
import json
from pathlib import Path


def main():
    folder=Path('validation/v0.5')
    result={}
    for path in sorted(folder.glob('*.json')):
        if path.name in ('summary.json',) or path.name.endswith('.progress.json'):
            continue
        data=json.loads(path.read_text(encoding='utf-8'))
        if 'cycles' not in data:
            continue
        cycles=data['cycles']
        if not isinstance(cycles,list) or not cycles:
            continue
        rows=[]
        for cycle in cycles:
            ai=cycle.get('ai',{})
            prosody=ai.get('prosody',{})
            rows.append(dict(inference=ai.get('inference'),rtf=ai.get('rtf'),callback=cycle.get('callback'),
                portaudio_underflow=cycle.get('underflow'),portaudio_overflow=cycle.get('overflow'),
                ai_underrun=ai.get('ai_underrun'),ai_overrun=ai.get('ai_overrun'),ai_drop=ai.get('dropped_chunks'),
                prosody_f0=prosody.get('analysis'),prosody_total=prosody.get('process_total'),
                prosody_error=prosody.get('errors'),prosody_queue_drop=prosody.get('queue_drop_chunks'),
                prosody_ram_mb=prosody.get('ram_bytes',0)/1e6,
                ai_ram_mb=ai.get('ram_bytes',0)/1e6,input_queue=ai.get('input_queue'),output_queue=ai.get('output_queue'),
                switches=cycle.get('switches'),worker_stopped=cycle.get('worker_stopped')))
            observations=cycle.get('observations',[])
            if len(observations)>1:
                # First fully populated streaming sample, excludes startup CPU.
                populated=[o for o in observations if 'cpu_seconds' in o.get('ai',{}).get('prosody',{})]
                if len(populated)>1:
                    first,last=populated[0],populated[-1]; duration=last['seconds']-first['seconds']
                    def cpu(a,b): return max(0,b-a)/duration/4*100
                    rows[-1]['cpu_four_core_percent']=dict(
                        ai=cpu(first['ai']['cpu_seconds'],last['ai']['cpu_seconds']),
                        prosody=cpu(first['ai']['prosody']['cpu_seconds'],last['ai']['prosody']['cpu_seconds']),
                        parent=cpu(first['parent_cpu_seconds'],last['parent_cpu_seconds']))
                    rows[-1]['ram_start_end_mb']=dict(
                        ai=[first['ai']['ram_bytes']/1e6,last['ai']['ram_bytes']/1e6],
                        prosody=[first['ai']['prosody']['ram_bytes']/1e6,last['ai']['prosody']['ram_bytes']/1e6])
        result[path.name]=dict(input_kind=data.get('input_kind'),seconds=data.get('seconds'),cycles=rows,
            error=data.get('error'),crash=data.get('crash'),prosody_crash=data.get('prosody_crash'),passed=data.get('passed'))
    (folder/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
