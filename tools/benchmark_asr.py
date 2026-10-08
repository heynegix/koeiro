"""Offline WAV replay benchmark. No audio device, no transcript in ordinary logs."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import time
import numpy as np
import soundfile as sf
import psutil
from src.asr.backend import ReazonBackend, VoskBackend
from src.asr.context import TextContextAnalyzer
from src.vc.resampler import StreamingResampler


def main():
    p=argparse.ArgumentParser(); p.add_argument('wav'); p.add_argument('--threads',type=int,default=1)
    p.add_argument('--update-ms',type=int,default=300); p.add_argument('--window-ms',type=int,default=3000)
    p.add_argument('--backend',choices=('reazon','vosk'),default='reazon')
    p.add_argument('--out',default='validation/v09/asr-benchmark.json'); args=p.parse_args()
    process=psutil.Process(); ram0=process.memory_info().rss
    root=Path(__file__).resolve().parents[1]
    backend=(VoskBackend(root/'models/asr-vosk/vosk-model-small-ja-0.22',args.threads) if args.backend=='vosk' else ReazonBackend(root/'models/asr-reazon',args.threads))
    t=time.perf_counter(); backend.load(); load=time.perf_counter()-t
    data,rate=sf.read(args.wav,dtype='float32'); data=data.mean(axis=1) if data.ndim==2 else data
    if rate!=48000: raise ValueError('Benchmark input must be 48 kHz')
    data=data[:48000*45]; resampler=StreamingResampler(48000,16000)
    samples=resampler.process(data); analyzer=TextContextAnalyzer()
    t=time.perf_counter(); backend.transcribe(samples[:16000]); warmup=time.perf_counter()-t
    backend.reset()
    hop=round(args.update_ms*16); window=round(args.window_ms*16)
    durations=[]; partials=[]; cpu=time.process_time(); start=time.perf_counter(); first=None
    for end in range(hop,len(samples)+1,hop):
        audio=samples[end-hop:end] if args.backend=='vosk' else samples[max(0,end-window):end]
        if args.backend=='reazon' and len(audio)<6400: continue
        t=time.perf_counter()
        text,final=backend.push(audio) if args.backend=='vosk' else (backend.transcribe(audio),False)
        elapsed=time.perf_counter()-t
        durations.append(elapsed*1000)
        if text and first is None: first=end/16+elapsed*1000
        # This explicit benchmark artifact contains text for accuracy inspection.
        partials.append(dict(audio_ms=end/16,processing_ms=elapsed*1000,text=text,
            final=final,context=analyzer.snapshot(text,end/16000,end/16000+elapsed,final)))
    wall=time.perf_counter()-start; duration=len(samples)/16000
    values=np.array(durations)
    result=dict(method='Vosk incremental Japanese' if args.backend=='vosk' else 'ReazonSpeech INT8 rolling-prefix offline recognizer',threads=args.threads,
        update_ms=args.update_ms,window_ms=args.window_ms,load_seconds=load,warmup_seconds=warmup,
        model_bytes=sum(f.stat().st_size for f in backend.path.rglob('*') if f.is_file() and f.suffix not in ('.zip','.bz2')),
        ram_mb=process.memory_info().rss/1e6,additional_ram_mb=(process.memory_info().rss-ram0)/1e6,
        cpu_one_core_percent=(time.process_time()-cpu)/wall*100,cpu_machine_percent=(time.process_time()-cpu)/wall*100/(psutil.cpu_count() or 1),
        input_seconds=duration,inference_seconds=float(values.sum()/1000),rtf=float(values.sum()/1000/duration),
        projected_realtime_cpu_machine_percent=(time.process_time()-cpu)/duration*100/(psutil.cpu_count() or 1),
        processing=dict(avg=float(values.mean()),p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95)),p99=float(np.percentile(values,99)),max=float(values.max())),
        first_nonempty_from_file_start_ms=first,partial_latency_note='File-start emission time; semantic context latency needs annotated word timestamps.',
        projected_processing_lag_ms=float(values.max()),partials=partials)
    target=Path(args.out); target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='partials'},indent=2))

if __name__=='__main__': main()
