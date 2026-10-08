"""Offline analysis benchmark: fixed context, explicit update deadline, no I/O."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import psutil
from scipy.io import wavfile
from scipy.signal import resample_poly
from src.audio.performance import TimingStats
from src.prosody.f0 import FCPEEstimator, YinEstimator


def load_wave(path):
    rate, audio = wavfile.read(path)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32)/float(2**(np.iinfo(audio.dtype).bits-1))
    else:
        audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Invalid human WAV')
    from math import gcd
    g=gcd(rate,16000)
    return resample_poly(audio,16000//g,rate//g).astype(np.float32)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wav',type=Path,default=Path('recordings/v03/input.wav'))
    parser.add_argument('--report',type=Path,default=Path('validation/v0.5/f0-benchmark.json'))
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--method',choices=['all','fcpe','yin'],default='all')
    args=parser.parse_args()
    if not 1<=args.repeats<=5:
        parser.error('repeats must be 1..5')
    audio=load_wave(args.wav)
    report={'input':str(args.wav),'input_duration':len(audio)/16000,'methods':[]}
    for method,context in [('fcpe',80),('fcpe',160),('yin',80)]:
        if args.method!='all' and args.method!=method:
            continue
        started=time.perf_counter()
        estimator=FCPEEstimator() if method=='fcpe' else YinEstimator()
        load=time.perf_counter()-started
        length=round(context*16)
        warm=time.perf_counter()
        for _ in range(5):
            estimator.extract(audio[:length]) if method=='fcpe' else estimator.estimate(audio[:length])
        warm=time.perf_counter()-warm
        timing=TimingStats()
        durations=[]
        process=psutil.Process()
        cpu=time.process_time(); wall=time.perf_counter()
        for _ in range(args.repeats):
            for end in range(length,len(audio),480):
                t=time.perf_counter_ns()
                estimator.extract(audio[end-length:end]) if method=='fcpe' else estimator.estimate(audio[end-length:end])
                elapsed_ns=time.perf_counter_ns()-t
                timing.record(elapsed_ns,480,16000)
                durations.append(elapsed_ns/1e6)
        elapsed=time.perf_counter()-wall
        stats=asdict(timing.snapshot())
        if durations:
            stats.update({f'p{q}_ms':float(np.percentile(durations,q)) for q in (50,95,99)})
        item=dict(method=method,context_ms=context,update_ms=30,load_seconds=load,warmup_seconds=warm,
            processing=stats,rtf=stats['average_ms']/30,cpu_seconds=time.process_time()-cpu,
            wall_seconds=elapsed,ram_mb=process.memory_info().rss/1e6)
        item['cpu_percent_four_cores_audio_timeline']=item['cpu_seconds']/max(.001,timing.count*.030)/4*100
        item['percentiles']='exact offline percentiles; elapsed wall time, includes scheduling'
        report['methods'].append(item)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(item),flush=True)

if __name__=='__main__':
    main()
