"""Five-window local WAV comparison, one model load; no audio device opened."""
import argparse,json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
import psutil
from src.asr.backend import ReazonBackend
from src.asr.context import TextContextAnalyzer
from src.asr.stability import ClassificationCache
from src.vc.resampler import StreamingResampler


def metrics(values):
    a=np.asarray(values,dtype=float)
    return dict(avg=float(a.mean()),p50=float(np.percentile(a,50)),p95=float(np.percentile(a,95)),
        p99=float(np.percentile(a,99)),max=float(a.max())) if len(a) else {}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--wav',default='recordings/v03/beatrice_comparison/Original.wav')
    p.add_argument('--out',default='test-results/v091/windows_raw.json'); args=p.parse_args()
    audio,rate=sf.read(args.wav,dtype='float32'); audio=audio.mean(axis=1) if audio.ndim==2 else audio
    if rate!=48000: raise ValueError('Use 48 kHz input')
    audio=audio[:45*48000]; reduced=StreamingResampler(48000,16000).process(audio)
    process=psutil.Process(); ram=process.memory_info().rss
    backend=ReazonBackend('models/asr-reazon',1)
    start=time.perf_counter(); backend.load(); load=time.perf_counter()-start
    # Shared speech warmup prevents window order from selectively paying first-speech costs.
    warm=time.perf_counter(); backend.transcribe(reduced[48000:64000]); warm=time.perf_counter()-warm
    report=dict(source=args.wav,kind='Offline sequential replay, no physical audio; lag simulation includes queue',
        load_seconds=load,warmup_seconds=warm,input_seconds=len(reduced)/16000,windows={})
    # These are provisional regions from the existing four-sentence protocol,
    # not manual word timestamps. Accuracy is a coverage proxy, not a WER/intent benchmark.
    regions=[(3.2,5.9,'STATEMENT'),(6.1,7.95,'QUESTION'),(8.,9.45,'CALLOUT'),(9.5,11.5,'FAREWELL')]
    hop=4800
    try:
        for window in (250,400,600,800,1000):
            analyzer=TextContextAnalyzer(); cache=ClassificationCache(); durations=[]; lag=[]; events=[]
            available_at=0.; drops=0; cpu=time.process_time(); started=time.perf_counter(); hits={}; latencies={}
            labeled=correct=0
            for end in range(hop,len(reduced)+1,hop):
                stamp=end/16000
                if stamp<available_at-.6: drops+=1; continue
                begin=time.perf_counter(); text=backend.transcribe(reduced[max(0,end-window*16):end]); cost=time.perf_counter()-begin
                received=max(stamp,available_at)+cost; available_at=received
                durations.append(cost*1000); lag.append((received-stamp)*1000)
                ctx=analyzer.snapshot(text,stamp,received) if text else {}
                ctx=cache.update(ctx,stamp,0) if ctx else {}
                ctx.pop('phrase_id',None)  # renderer attaches audio-derived phrase boundaries
                for index,(onset,offset,label) in enumerate(regions):
                    if onset<=stamp<=offset:
                        labeled+=1
                        if ctx.get('phrase_type')==label:
                            correct+=1; hits[index]=True; latencies.setdefault(label,(received-onset)*1000)
                events.append(dict(audio_ms=stamp*1000,processing_ms=cost*1000,ready_seconds=received,context=ctx))
            wall=time.perf_counter()-started
            report['windows'][str(window)]=dict(processing=metrics(durations),lag=metrics(lag),
                rtf=sum(durations)/1000/(len(reduced)/16000),drops=drops,
                simulated_queue_max_hops=max(lag,default=0)/300,
                provisional_type_coverage=len(hits)/len(regions),context_latency_ms=latencies,
                provisional_frame_accuracy=correct/max(1,labeled),labeled_frames=labeled,
                cpu_machine_percent=(time.process_time()-cpu)/wall*100/(psutil.cpu_count() or 1),
                ram_mb=process.memory_info().rss/1e6,additional_ram_mb=(process.memory_info().rss-ram)/1e6,partials=events)
            print(window,json.dumps({k:v for k,v in report['windows'][str(window)].items() if k!='partials'}),flush=True)
    finally: backend.unload()
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__': main()
