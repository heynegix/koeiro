"""Single vs conditional dual-window, offline CPU timings and simulated lag only."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import psutil
from tools.compare_presets import read_wav
from tools.benchmark_asr_windows import metrics
from src.asr.backend import ReazonBackend
from src.asr.context import TextContextAnalyzer
from src.asr.stability import ClassificationCache
from src.asr.confirmation import needs_confirmation
from src.vc.resampler import StreamingResampler


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--wav',default='recordings/v03/beatrice_comparison/Original.wav')
    parser.add_argument('--out',default='test-results/v092/asr_raw.json')
    parser.add_argument('--reverse',action='store_true',help='Check execution-order sensitivity once'); args=parser.parse_args()
    audio,rate=read_wav(Path(args.wav)); reduced=StreamingResampler(rate,16000).process(audio)
    backend=ReazonBackend('models/asr-reazon',1); start=time.perf_counter(); backend.load(); load=time.perf_counter()-start
    backend.warmup(); process=psutil.Process()
    report=dict(source=args.wav,input_seconds=len(reduced)/16000,load_seconds=load,
        kind='Offline inference timing; lag simulated, no microphone/Discord',conditions={})
    try:
        for dual in ((True,False) if args.reverse else (False,True)):
            cache=ClassificationCache(); analyzer=TextContextAnalyzer(); events=[]; costs=[]; lags=[]
            available=0.; drops=0; last_confirm=-100.; confirmations=0; cpu=time.process_time(); wall=time.perf_counter()
            for end in range(4800,len(reduced)+1,4800):
                stamp=end/16000
                if stamp<available-.6: drops+=1; continue
                start=time.perf_counter(); text=backend.transcribe(reduced[max(0,end-6400):end])
                # Confirmation probe must not advance the live partial history twice.
                primary=TextContextAnalyzer().snapshot(text,stamp,stamp) if text else {}
                confirm=False
                if dual and needs_confirmation(primary,stamp,last_confirm):
                    text=backend.transcribe(reduced[max(0,end-12800):end]); last_confirm=stamp; confirmations+=1; confirm=True
                cost=time.perf_counter()-start; ready=max(stamp,available)+cost; available=ready
                context=analyzer.snapshot(text,stamp,ready) if text else {}
                context=cache.update(context,stamp,0) if context else {}; context.pop('phrase_id',None)
                costs.append(cost*1000); lags.append((ready-stamp)*1000)
                events.append(dict(audio_ms=stamp*1000,processing_ms=cost*1000,ready_seconds=ready,
                    confirmed=confirm,context=context))
            duration=time.perf_counter()-wall
            report['conditions']['dual' if dual else 'single']=dict(processing=metrics(costs),lag=metrics(lags),
                rtf=sum(costs)/1000/report['input_seconds'],drops=drops,confirmations=confirmations,
                queue_max_hops=max(lags,default=0)/300,cpu_machine_percent=(time.process_time()-cpu)/duration*100/(psutil.cpu_count() or 1),
                ram_mb=process.memory_info().rss/1e6,classification_transitions=sum(
                    events[i]['context'].get('phrase_type')!=events[i-1]['context'].get('phrase_type') for i in range(1,len(events))),
                strong_question_frames=sum(e['context'].get('question_probability',0)>=.7 for e in events),
                strong_farewell_frames=sum(e['context'].get('farewell_probability',0)>=.7 for e in events),partials=events)
            print('dual' if dual else 'single',json.dumps({k:v for k,v in report['conditions']['dual' if dual else 'single'].items() if k!='partials'}),flush=True)
    finally: backend.unload()
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__': main()
