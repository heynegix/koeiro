"""ASR subprocess, CPU low priority; rolling windows never block VC."""
from dataclasses import asdict
import sys
import time
import argparse
import numpy as np
from .protocol import send, receive
from src.vc.resampler import StreamingResampler
from .performance import ASRTimingStats as TimingStats
from .backend import ReazonBackend
from .context import TextContextAnalyzer
from .stability import ClassificationCache


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--model',required=True)
    parser.add_argument('--threads',type=int,default=1); args=parser.parse_args()
    out=sys.stdout.buffer
    try:
        import psutil
        process=psutil.Process()
        if sys.platform=='win32': process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        backend=ReazonBackend(args.model,args.threads)
        start=time.perf_counter(); backend.load(); load=time.perf_counter()-start
        resampler=StreamingResampler(48000,16000); analyzer=TextContextAnalyzer()
        cache=ClassificationCache()
        quiet=0.; timing=TimingStats(); resample_timing=TimingStats(); classify_timing=TimingStats()
        cpu=time.process_time(); processed_seconds=0.; phrase_seconds=0.
        history=np.empty(0,dtype=np.float32)
        t=time.perf_counter(); backend.warmup(); warmup=time.perf_counter()-t
        send(out,dict(status='Ready',load_seconds=load,warmup_seconds=warmup,ram_bytes=process.memory_info().rss))
        while True:
            h,audio=receive(sys.stdin.buffer)
            if h['op']=='stop': break
            if h['op']=='reset':
                quiet=0.; phrase_seconds=0.; history=np.empty(0,dtype=np.float32); resampler.reset(); analyzer.reset(); backend.reset(); cache.reset()
                send(out,dict(status='Ready',context={})); continue
            if h['op']!='process' or not 0<len(audio)<=48000:
                raise ValueError('Invalid ASR request')
            begin=time.perf_counter_ns(); reduced=resampler.process(audio); seconds=len(audio)/48000
            resample_ms=(time.perf_counter_ns()-begin)/1e6
            resample_timing.record(round(resample_ms*1e6),len(audio),48000)
            energy=float(np.sqrt(np.mean(reduced**2)))
            quiet=quiet+seconds if energy<.004 else 0.
            phrase_seconds+=seconds; processed_seconds+=seconds
            window=h.get('window_ms',1000)
            if window not in (250,400,600,800,1000,1500,2000): raise ValueError('Invalid ASR window')
            history=np.concatenate((history,reduced))[-round(window*16):]
            final=quiet>=.45
            t=time.perf_counter_ns()
            text=backend.transcribe(history) if energy>=.004 or final and analyzer.previous else ''
            elapsed=time.perf_counter_ns()-t; timing.record(elapsed,len(audio),48000)
            begin=time.perf_counter_ns()
            context=analyzer.snapshot(text,h['audio_timestamp'],time.monotonic(),final) if text else {}
            context=cache.update(context,h['audio_timestamp'],h.get('phrase_id',0)) if context else {}
            classify_ms=(time.perf_counter_ns()-begin)/1e6
            classify_timing.record(round(classify_ms*1e6),len(audio),48000)
            send(out,dict(status='Listening',context=context,processing=asdict(timing.snapshot()),
                rtf=timing.total_ns/1e9/max(.001,processed_seconds),
                cpu_seconds=time.process_time()-cpu,ram_bytes=process.memory_info().rss,
                final=final,processed_timestamp=h['audio_timestamp'],resample_ms=resample_ms,
                classification_ms=classify_ms,collected_audio_ms=len(history)/16,phrase_id=h.get('phrase_id',0),
                resample=asdict(resample_timing.snapshot()),classification=asdict(classify_timing.snapshot())))
            if final or quiet>=.45 or phrase_seconds>=8:
                quiet=0.; phrase_seconds=0.; history=np.empty(0,dtype=np.float32); analyzer.reset(); backend.reset(); cache.reset()
    except Exception as error:
        send(out,dict(status='Error',error=str(error)))
        raise

if __name__=='__main__': main()
