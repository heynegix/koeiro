"""Device-free, sequential N150 comparison of the selected unchanged voice."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'recordings/v011_phrase_repair'


def main():
    import numpy as np
    import soundfile as sf
    import psutil
    from src.vc.meanvc2 import MeanVC2Backend, checksum
    from src.vc.meanvc2_phrase import MeanVC2PhraseBackend
    from src.vc.phrase_prosody import HOP, HISTORY, LOOKAHEAD, repair_window
    from tools.compare_meanvc2_continuity import require_idle_voice_worker
    from src.vc.resampler import StreamingResampler
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['meanvc2_ref20','meanvc2_ref60'], default='meanvc2_ref20')
    p.add_argument('--threads',type=int,choices=(1,2,4),default=2)
    p.add_argument('--stream-only',action='store_true')
    p.add_argument('--parity-only',action='store_true')
    args = p.parse_args()
    require_idle_voice_worker()
    if psutil.virtual_memory().available < 4*1024**3:
        raise RuntimeError('Need 4GiB available RAM before loading one model')
    OUT.mkdir(parents=True, exist_ok=True)
    folder = ROOT / 'models' / args.model
    if args.parity_only:
        b=MeanVC2Backend(args.threads);b.load(folder)
        source=ROOT/'recordings/v011_meanvc2_continuity/source/source_long.wav'
        wave,rate=sf.read(source,dtype='float32')
        if rate!=48000:raise ValueError('Expected 48k source')
        down=StreamingResampler(48000,16000)
        padded=np.pad(wave,(0,(-len(wave))%7680+30*7680))
        audio=np.concatenate([down.process(padded[i:i+7680]) for i in range(0,len(padded),7680)])
        outputs=[]
        for batch in (1,36):
            b.vocoder_batch_frames=batch;b.reset();parts=[]
            trim=b.algorithmic_buffer_ms*16+(640 if b.bn_interpolation=='fixed_linear' else 0)
            for i in range(0,len(audio),HOP):parts.append(b.process_chunk(audio[i:i+HOP]))
            outputs.append(np.concatenate(parts)[trim:trim+round(len(wave)/3)])
        error=float(np.max(abs(outputs[0]-outputs[1])))
        result=dict(kind='Same threads, checkpoint, source and embedding; FLOAT32; decode batch 1 vs 36',
                    threads=args.threads,max_absolute_error=error,
                    rmse=float(np.sqrt(np.mean((outputs[0]-outputs[1])**2))),tolerance=1e-5,passed=error<=1e-5)
        (OUT/'decode_parity.json').write_text(json.dumps(result,indent=2),'utf-8')
        b.unload();print(json.dumps(result),flush=True)
        if not result['passed']:raise RuntimeError('Decode parity failed')
        return
    report = dict(kind='OFFLINE WAV; no audio devices; listening not performed',
                  model=args.model, runtime_sha256=checksum(folder/'runtime.json'),
                  embedding_sha256=checksum(folder/'fixed_embedding.npy'), rows=[])
    b = MeanVC2Backend(args.threads)
    if not args.stream_only:
        b.load(folder); b.warmup()
    alignment = (b.algorithmic_buffer_ms + (40 if b.bn_interpolation=='fixed_linear' else 0)) if not args.stream_only else 0
    samples = {}
    for name in (('long',) if args.stream_only else ('normal','low','bright','long')):
        source = (ROOT/'recordings/v011_meanvc2_continuity/source/source_long.wav' if name=='long' else
                  ROOT/f'recordings/v011_mega_tournament/source/source_{name}.wav')
        wave, rate = sf.read(source, dtype='float32')
        if wave.ndim == 2: wave = wave.mean(axis=1)
        if rate != 48000: raise ValueError('Expected existing 48kHz source')
        down = StreamingResampler(48000,16000)
        padded = np.pad(wave, (0, (-len(wave)) % 7680 + 30*7680))
        audio = np.concatenate([down.process(padded[i:i+7680]) for i in range(0,len(padded),7680)])
        if args.stream_only:
            samples[name]=(audio,None)
            continue
        b.reset(); parts=[]; times=[]
        for i in range(0,len(audio),HOP):
            start=time.perf_counter(); parts.append(b.process_chunk(audio[i:i+HOP]))
            times.append(time.perf_counter()-start)
        baseline=np.concatenate(parts)
        source_aligned=np.pad(audio,(alignment*16,0))[:len(baseline)]
        samples[name]=(audio,baseline)
        n=round(len(wave)/3)
        trim=alignment*16
        sf.write(OUT/f'{name}_baseline.wav',baseline[trim:trim+n],16000,subtype='PCM_16')
        sf.write(OUT/f'{name}_source.wav',audio[:n],16000,subtype='PCM_16')
        for variant,energy,pitch in [('energy',True,False),('pitch',False,True),('combined',True,True)]:
            result=[]; diagnostics=[]
            for at in range(0,len(baseline),HOP):
                lo,hi=at-HISTORY,at+LOOKAHEAD
                def window(x):
                    z=np.zeros(hi-lo,np.float32)
                    begin,end=max(lo,0),min(hi,len(x))
                    z[begin-lo:end-lo]=x[begin:end]
                    return z
                edited,stats=repair_window(window(source_aligned),window(baseline),HISTORY,energy,pitch)
                result.append(edited);diagnostics.append(stats)
            converted=np.concatenate(result)[trim:trim+n]
            sf.write(OUT/f'{name}_{variant}.wav',converted,16000,subtype='PCM_16')
            report['rows'].append(dict(source=name,variant=variant,audio_seconds=n/16000,
                 base_rtf=sum(times)/(len(audio)/16000),
                 analysis_rtf=sum(s['analysis_ms'] for s in diagnostics)/1000/(len(audio)/16000),
                 analysis_p95_ms=float(np.percentile([s['analysis_ms'] for s in diagnostics],95)),
                 pitch_regions=sum(s['pitch_regions'] for s in diagnostics),
                 max_shift_st=max(s['max_shift_st'] for s in diagnostics),
                 max_gain_db=max(s['max_gain_db'] for s in diagnostics),
                 clipping_fraction=float(np.mean(abs(converted)>=1)),
                 waveform_rmse=float(np.sqrt(np.mean((converted-baseline[trim:trim+n])**2)))))
        print(name,report['rows'][-1],flush=True)
    b.unload()
    # Same production wrapper, with input paced like a running capture device.
    b=MeanVC2PhraseBackend(args.threads);b.load(folder);b.warmup()
    alignment=b.repair.alignment//16
    audio,_=samples['long']; parts=[]; times=[]
    neural=[]
    original_process=b.repair.process
    def capture_neural(source,voice):
        neural.append(voice.copy())
        return original_process(source,voice)
    b.repair.process=capture_neural
    start=time.perf_counter()
    for i in range(0,len(audio),HOP):
        elapsed=time.perf_counter()-start
        remaining=i/16000-elapsed
        if remaining>0: time.sleep(remaining)
        at=time.perf_counter();parts.append(b.process_chunk(audio[i:i+HOP]))
        times.append(time.perf_counter()-at)
        if psutil.virtual_memory().available<2*1024**3: raise RuntimeError('Available RAM below 2GiB')
    report['paced']=dict(kind='Device-free paced replay through actual production backend',
        audio_seconds=len(audio)/16000,rtf=sum(times)/(len(audio)/16000),
        p95_ms=float(np.percentile(times,95)*1000),max_ms=max(times)*1000,
        model=b.get_stats(),ram_bytes=psutil.Process().memory_info().rss)
    output=np.concatenate(parts)
    raw=np.concatenate(neural)
    sf.write(OUT/'long_fast_neural.wav',raw[alignment*16:alignment*16+round(len(wave)/3)],16000,subtype='PCM_16')
    baseline,base_rate=sf.read(OUT/'long_baseline.wav',dtype='float32')
    comparison=raw[alignment*16:alignment*16+len(baseline)]
    report['neural_parity']=dict(max_absolute_error=float(np.max(abs(comparison-baseline))),
        rmse=float(np.sqrt(np.mean((comparison-baseline)**2))),tolerance=4e-5,
        passed=bool(np.max(abs(comparison-baseline))<=4e-5),
        note='Optimized unedited stream vs saved PCM16 current voice; tolerance includes PCM quantization')
    trim=alignment*16+LOOKAHEAD
    sf.write(OUT/'long_actual_stream.wav',output[trim:trim+round(len(wave)/3)],16000,subtype='PCM_16')
    b.unload()
    (OUT/('stream_measurements.json' if args.stream_only else 'measurements.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
    if args.stream_only:
        print(json.dumps(report,ensure_ascii=False),flush=True)
        return
    cards=[]
    for name in samples:
        cards.append(f'<h2>{name}</h2>')
        for variant in ('baseline','energy','pitch','combined'):
            cards.append(f'<p>{variant}</p><audio controls src="{name}_{variant}.wav"></audio>')
    cards.append('<h2>long actual stream</h2><audio controls src="long_actual_stream.wav"></audio>')
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>今の声・抑揚補完比較</title><style>body{background:#151920;color:#eee;font:16px sans-serif;max-width:900px;margin:30px auto}audio{width:100%}</style><h1>現在声と抑揚補完</h1><p>Offline比較。baselineが現在声。聴感評価は未実施。</p>'+''.join(cards),'utf-8')
    print(json.dumps(report['paced'],ensure_ascii=False),flush=True)


if __name__=='__main__':main()
