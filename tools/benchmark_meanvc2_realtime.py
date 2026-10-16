"""Short, device-free replay through the actual stateful MeanVC2 backend.

``--devices cpu cuda`` replays the same audio through both routes in one process
and writes the difference between them into the report. On a GPU machine this is
how the app's CUDA route is checked against the recorded Kaggle T4 runs (see
notes/kaggle_gpu_verification.md), and the report is what to paste into an issue.
Offline replay only: no audio devices are opened and nothing is listened to.
"""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from src.vc.meanvc2 import MeanVC2Backend
from src.vc.models import DEFAULT_VOICE_ID,VOICE_PROFILES,profile,is_meanvc2
def statistics_mean(values,key):return float(np.mean([v[key] for v in values]))


def read_input(path):
    """One float32 mono 16 kHz block of `path`, as the backend consumes it."""
    audio,sr=sf.read(path,dtype='float32')
    if audio.ndim>1:audio=audio.mean(axis=1)
    return resample_poly(audio,16000,sr).astype(np.float32)


def replay(device,args,audio,actual_samples):
    """Warm up one route, stream `audio` through it, return its output and numbers.

    The tail flush and the delay trim are the production ones: this is the route a
    listener would hear, not a batch inference.
    """
    import psutil
    import torch
    if args.audition_vc720:
        from src.vc.meanvc2_continuity import MeanVC2ContinuityBackend
        backend=MeanVC2ContinuityBackend(args.threads,6,1,device=device)
    else:backend=MeanVC2Backend(threads=args.threads,device=device)
    backend.load(ROOT/'models'/profile(args.model)['folder'])
    # Honour the profile's own inference grouping so --model measures the real
    # production condition; explicit flags still win for offline comparison.
    backend.select_profile(profile(args.model))
    if args.vc_group_chunks is not None:
        backend.vc_group_chunks=args.vc_group_chunks;backend.reset()
        backend.stats.update(vc_group_chunks=backend.vc_group_chunks,algorithmic_buffer_ms=backend.algorithmic_buffer_ms,
                             offline_vc_override=True)
    if args.vocoder_batch_frames is not None:
        backend.vocoder_batch_frames=args.vocoder_batch_frames;backend.reset()
        backend.stats.update(vocoder_batch_frames=backend.vocoder_batch_frames,algorithmic_buffer_ms=backend.algorithmic_buffer_ms,
                             offline_vocoder_override=True)
    backend.warmup()
    n=(len(audio)+2559)//2560
    times=[];chunks=[];stages=[];saved=backend.get_stats()
    for i in range(n):
        start=time.perf_counter();chunks.append(backend.process_chunk(audio[i*2560:(i+1)*2560]));times.append(time.perf_counter()-start)
        stages.append(backend.stats['last_stage_ms'])
    # Flush finite stream tail, then remove the explicitly reported model buffer.
    delay=round((saved['algorithmic_buffer_ms']+saved.get('interpolation_grid_delay_ms',0))*16)
    for _ in range(math.ceil(delay/2560)+8):chunks.append(backend.process_chunk(np.zeros(2560,dtype=np.float32)))
    result=np.concatenate(chunks)[delay:delay+actual_samples]
    if len(result)!=actual_samples:raise RuntimeError('Incomplete finite-stream tail')
    record=dict(saved,device=str(device),torch=str(torch.__version__),
        cuda_available=bool(torch.cuda.is_available()),
        cuda_name=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
        audio_seconds=actual_samples/16000,padded_audio_seconds=len(audio)/16000,
        generation_seconds=sum(times),rtf=sum(times)/(actual_samples/16000),
        chunk_mean_ms=float(np.mean(times)*1000),chunk_p95_ms=float(np.percentile(times,95)*1000),
        chunk_max_ms=max(times)*1000,process_ram_bytes=psutil.Process().memory_info().rss,
        output_rms=float(np.sqrt(np.mean(result.astype('float64')**2))),
        output_finite=bool(np.isfinite(result).all()),
        stage_mean_ms={key:statistics_mean(stages,key) for key in ['features','vc','vocoder']})
    backend.unload()
    if device=='cuda':torch.cuda.empty_cache()
    return result,record


def compare(left,right):
    """What one route's samples say about the other's, in plain units."""
    difference=np.abs(left.astype('float64')-right.astype('float64'))
    signal=max(float(np.sqrt(np.mean(left.astype('float64')**2))),1e-12)
    rms=float(np.sqrt(np.mean(difference**2)))
    return dict(samples=int(len(left)),identical=bool(np.array_equal(left,right)),
        max_abs_diff=float(difference.max()),rms_diff=rms,
        diff_to_signal_db=20*math.log10(max(rms,1e-12)/signal))


def main():
    p=argparse.ArgumentParser();p.add_argument('--threads',type=int,default=2);p.add_argument('--seconds',type=float,default=4.8)
    p.add_argument('--source',type=Path,default=None,help='WAV to replay; default: the voice reference')
    p.add_argument('--model',choices=[key for key in VOICE_PROFILES if is_meanvc2(key)],default=DEFAULT_VOICE_ID)
    p.add_argument('--devices',nargs='+',choices=('cpu','cuda'),default=['cpu'],
                   help='One for a plain measurement, both to subtract one route from the other')
    p.add_argument('--vocoder-batch-frames',type=int,choices=(1,36),help='Offline comparison override; runtime file is never edited')
    p.add_argument('--vc-group-chunks',type=int,choices=(1,3,6),help='Offline comparison override; runtime file is never edited')
    p.add_argument('--audition-vc720',action='store_true',help='Validation only: replay original audition implementation')
    p.add_argument('--report',type=Path,required=True);args=p.parse_args()
    args.report.parent.mkdir(parents=True,exist_ok=True)
    folder=ROOT/'models'/profile(args.model)['folder']
    source=args.source or folder/'reference.wav'
    audio=read_input(source)[:round(args.seconds*16000)]
    actual_samples=len(audio)
    audio=np.pad(audio,(0,((actual_samples+2559)//2560)*2560-actual_samples))
    routes={};results={};errors={}
    for device in args.devices:
        print(f'--- {device} ---',flush=True)
        try:
            results[device],routes[device]=replay(device,args,audio,actual_samples)
        except Exception as error:
            errors[device]=f'{type(error).__name__}: {error}'
            routes[device]=dict(device=str(device),error=errors[device])
        print(json.dumps(routes[device],indent=2,ensure_ascii=False),flush=True)
    report=dict(stage='complete',profile=args.model,source=str(source),
        input_kind='Offline WAV replay; no audio devices',
        platform=sys.platform,python=sys.version.split()[0],
        routes=[routes[key] for key in args.devices])
    if len(args.devices)>1 and len(results)==len(args.devices):
        first,second=args.devices[0],args.devices[1]
        report['comparison']=dict(baseline=first,against=second,**compare(results[first],results[second]))
    # A single route keeps its numbers flat at the top level: the older readers of
    # this report look for the keys there, not under routes.
    if len(args.devices)==1:report.update(routes[args.devices[0]])
    for device,out in results.items():
        suffix='.wav' if len(args.devices)==1 else f'.{device}.wav'
        sf.write(args.report.with_suffix(suffix),out,16000,subtype='FLOAT')
    args.report.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    if 'comparison' in report:print(json.dumps(report['comparison'],indent=2,ensure_ascii=False))
    if errors:raise SystemExit('Failed: '+', '.join(f'{key}: {value}' for key,value in errors.items()))
if __name__=='__main__':main()
