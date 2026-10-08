"""Short, device-free replay through the actual stateful MeanVC2 backend."""
import argparse,json,sys,time
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from src.vc.meanvc2 import MeanVC2Backend
from src.vc.models import VOICE_PROFILES,profile,is_meanvc2
def statistics_mean(values,key):return float(np.mean([v[key] for v in values]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--threads',type=int,default=2);p.add_argument('--seconds',type=float,default=4.8)
    p.add_argument('--source',type=Path,default=ROOT/'recordings/v011_mega_tournament/source/source_normal.wav')
    p.add_argument('--model',choices=[key for key in VOICE_PROFILES if is_meanvc2(key)],default='meanvc2_120')
    p.add_argument('--vocoder-batch-frames',type=int,choices=(1,36),help='Offline comparison override; runtime file is never edited')
    p.add_argument('--vc-group-chunks',type=int,choices=(1,3,6),help='Offline comparison override; runtime file is never edited')
    p.add_argument('--audition-vc720',action='store_true',help='Validation only: replay original audition implementation')
    p.add_argument('--report',type=Path,required=True);args=p.parse_args()
    args.report.parent.mkdir(parents=True,exist_ok=True)
    def save(value):args.report.write_text(json.dumps(value,indent=2),encoding='utf-8')
    save(dict(stage='loading',input_kind='Offline WAV replay; no audio devices',threads=args.threads))
    if args.audition_vc720:
        from src.vc.meanvc2_continuity import MeanVC2ContinuityBackend
        backend=MeanVC2ContinuityBackend(args.threads,6,1)
    else:backend=MeanVC2Backend(threads=args.threads)
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
    save(dict(stage='warmup',model=backend.get_stats()));backend.warmup()
    audio,sr=sf.read(args.source,dtype='float32');audio=resample_poly(audio,16000,sr).astype(np.float32)
    audio=audio[:round(args.seconds*16000)];actual_samples=len(audio);n=(len(audio)+2559)//2560
    audio=np.pad(audio,(0,n*2560-len(audio)))
    times=[];chunks=[];stages=[];saved=backend.get_stats();save(dict(stage='inference',model=saved))
    for i in range(n):
        start=time.perf_counter();chunks.append(backend.process_chunk(audio[i*2560:(i+1)*2560]));times.append(time.perf_counter()-start)
        stages.append(backend.stats['last_stage_ms'])
    # Flush finite stream tail, then remove the explicitly reported model buffer.
    delay=round((saved['algorithmic_buffer_ms']+saved.get('interpolation_grid_delay_ms',0))*16)
    for _ in range(math.ceil(delay/2560)+8):chunks.append(backend.process_chunk(np.zeros(2560,dtype=np.float32)))
    result=np.concatenate(chunks)[delay:delay+actual_samples]
    if len(result)!=actual_samples:raise RuntimeError('Incomplete finite-stream tail')
    sf.write(args.report.with_suffix('.wav'),result,16000,subtype='FLOAT')
    import psutil
    saved.update(stage='complete',profile=args.model,source=str(args.source),input_kind='Offline WAV replay; no audio devices',
        audio_seconds=actual_samples/16000,padded_audio_seconds=len(audio)/16000,generation_seconds=sum(times),rtf=sum(times)/(actual_samples/16000),
        chunk_mean_ms=float(np.mean(times)*1000),chunk_p95_ms=float(np.percentile(times,95)*1000),
        chunk_max_ms=max(times)*1000,process_ram_bytes=psutil.Process().memory_info().rss,
        output_rms=float(np.sqrt(np.mean(result.astype('float64')**2))),
        output_finite=bool(np.isfinite(result).all()),stage_mean_ms={k:statistics_mean(stages,k) for k in ['features','vc','vocoder']})
    save(saved);print(json.dumps(saved,indent=2));backend.unload()
if __name__=='__main__':main()
