"""Synthetic feature/control proof, not recognition or listening verification."""
import json,sys,time
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.prosody.controller import ProsodyController
from src.prosody.context import PhraseContext
from src.prosody.parameters import ProsodyParameters,load_preset
from tools.compare_text_events import summarize


def metrics(values):
    return dict(avg=float(np.mean(values)),**{key:float(np.percentile(values,p)) for key,p in (('p50',50),('p95',95),('p99',99))},max=float(np.max(values)))


def main():
    result=dict(kind='Synthetic strong context + controlled causal F0; no microphone/ASR/audio listening',presets={})
    for preset in ('Natural','Anime Light','Anime Expressive'):
        combined=[]; latency=[]
        for kind,slope in [('QUESTION',3),('FAREWELL',-3),('EXCLAMATION',3),('SHORT_RESPONSE',0)]:
            c=ProsodyController(); c.phrase_id=1
            c.tracker.update=lambda *args:PhraseContext(state='ENDING_CANDIDATE',pitch_slope=slope,ending_probability=.9)
            p=load_preset(preset,ProsodyParameters(enabled=True,engine='text_v3',text_strategy='events'))
            trace=[]
            for i in range(50):
                text=dict(phrase_id=1,context_confidence=1,age_ms=0,phrase_type=kind)
                if kind=='SHORT_RESPONSE': text['short_response_type']='SURPRISE'
                else: text[dict(QUESTION='question_probability',FAREWELL='farewell_probability',EXCLAMATION='exclamation_probability')[kind]]=1
                source_f0=160*2**(slope*i*.02/12)
                start=time.perf_counter_ns(); control=c.update(source_f0,1,-25,.02,p,text); latency.append((time.perf_counter_ns()-start)/1e6)
                trace.append(dict(seconds=i*.02,**asdict(control)))
            summary,_=summarize(trace); combined.append(dict(type=kind,summary=summary))
        result['presets'][preset]=dict(cases=combined,processing_ms=metrics(latency))
    out=Path('validation/v092/synthetic-events.json'); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({p:{v['type']:v['summary']['event_types'][v['type']] for v in r['cases']} for p,r in result['presets'].items()}))


if __name__=='__main__':main()
