import hashlib
import math
import shutil
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('Empty or non-finite audio')
    return x.mean(axis=1), sr

def stats(path):
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('Empty or non-finite output')
    return dict(duration=len(x)/sr, sample_rate=sr, channels=x.shape[1],
                rms=float(np.sqrt(np.mean(x.astype('float64')**2))),
                peak=float(np.max(np.abs(x))), dc=float(np.mean(x)),
                clipping_samples=int(np.sum(np.abs(x) >= 0.999)), sha256=digest(path))

def resample(x, sr, target):
    g = math.gcd(sr, target)
    return resample_poly(x, target//g, sr//g) if sr != target else x

def normalize(raw, destination):
    x, sr = read(raw)
    before = stats(raw)
    dc = float(x.mean())
    x = x - dc
    rms = float(np.sqrt(np.mean(x.astype('float64')**2)))
    if rms < 1e-7:
        raise ValueError('Silent output: cannot audition')
    gain = 0.1/rms
    peak = float(np.max(np.abs(x)))
    gain = min(float(gain), 0.98/peak)
    sf.write(destination, x*gain, sr, subtype='PCM_16')
    return dict(raw=before, dc_removed=dc, scalar_gain=gain, after=stats(destination),
                method='DC subtraction + one constant RMS scalar to 0.1 RMS, reduced only for 0.98 peak safety')

def prepare_reference(original, folder, exclusions=()):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    x, sr = read(original)
    if len(x)/sr < 60:
        raise ValueError('Reference must contain at least 60 seconds')
    # Acoustic proxies only: not a gender classifier or verified noise measurement.
    # Frame RMS, hard clipping, low-energy occupancy, and level variation.
    frame = max(1, int(sr*.02))
    frames = x[:len(x)//frame*frame].reshape(-1, frame)
    rms = np.sqrt(np.mean(frames.astype('float64')**2, axis=1))
    clips = np.mean(np.abs(frames) >= .999, axis=1)
    silence = rms < max(.002, float(np.percentile(rms, 80))*.08)
    result = {'original': str(original), 'original_stats': stats(original), 'segments': {},
              'excluded_source_intervals': list(exclusions),
              'selection': 'Contiguous acoustic-proxy ranking; female voice and low noise NOT listening-verified',
              'female_stability_verified': False}
    for seconds in (5, 10, 20, 60):
        width = seconds*50
        scores = []
        for start in range(0, len(rms)-width+1, 25):
            a=start*frame/sr; b=a+seconds
            if any(a < e['end']+.25 and b > e['start']-.25 for e in exclusions):
                continue
            r = rms[start:start+width]
            clip = float(clips[start:start+width].sum())
            quiet = float(silence[start:start+width].mean())
            variation = float(np.std(np.log(np.maximum(r, 1e-6))))
            score = clip*10000 + quiet*10 + variation
            scores.append((score, start))
        score, start = min(scores)
        start_sample = start*frame
        path = folder/f'reference_{seconds:02d}s.wav'
        sf.write(path, x[start_sample:start_sample+seconds*sr], sr, subtype='PCM_16')
        result['segments'][f'{seconds:02d}s'] = dict(path=str(path), start_seconds=start_sample/sr,
                                                     score=score, stats=stats(path))
    full = folder/'reference_full.wav'
    if exclusions:
        keep=np.ones(len(x),dtype=bool)
        for e in exclusions:
            keep[max(0,int((e['start']-.25)*sr)):min(len(x),int((e['end']+.25)*sr))]=False
        sf.write(full,x[keep],sr,subtype='PCM_16')
        result['full_note']='All remaining target audio concatenated after held-out source exclusion + 250ms margins; original preserved'
    else:
        shutil.copy2(original, full)
    result['segments']['full'] = dict(path=str(full), stats=stats(full))
    return result

def extract_sources(original, folder):
    """User-authorized target-derived sources; do not label as male recordings."""
    x,sr=read(original);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    options=[]
    for start in np.arange(0,len(x)/sr-7,3.5):
        a=x[int(start*sr):int((start+7)*sr)]
        r=float(np.sqrt(np.mean(a.astype('float64')**2)))
        if r<.005 or np.max(np.abs(a))>=.999:continue
        f=a[:len(a)//(sr//10)*(sr//10)].reshape(-1,sr//10)
        energy=np.sqrt(np.mean(f.astype('float64')**2,axis=1))
        if np.mean(energy<.003)>.35:continue
        # Spectral centroid is a brightness proxy, not perceived vocal gender.
        spec=np.abs(np.fft.rfft(f*np.hanning(f.shape[1]),axis=1))
        hz=np.fft.rfftfreq(f.shape[1],1/sr)
        centroid=float(np.sum(spec*hz)/max(1e-9,np.sum(spec)))
        # Low-band energy proxy; no artificial pitch shifting.
        low=float(np.sum(spec[:,hz<400])/max(1e-9,np.sum(spec)))
        options.append(dict(start=float(start),end=float(start+7),centroid=centroid,low_band_ratio=low,rms=r))
    selected=[]
    for name in ('normal','low','bright'):
        eligible=[e for e in options if all(e['end']+.5<=s['start'] or e['start']>=s['end']+.5 for s in selected)]
        if not eligible:raise ValueError('Insufficient clean nonoverlapping source segments')
        if name=='low':choice=max(eligible,key=lambda e:e['low_band_ratio'])
        elif name=='bright':choice=max(eligible,key=lambda e:e['centroid'])
        else:
            median=np.median([e['centroid'] for e in eligible]);choice=min(eligible,key=lambda e:abs(e['centroid']-median))
        choice=dict(choice,label=name);selected.append(choice)
        path=folder/f'source_{name}.wav'
        sf.write(path,x[int(choice['start']*sr):int(choice['end']*sr)],sr,subtype='PCM_16')
        choice['path']=str(path);choice['sha256']=digest(path)
    return dict(origin=str(original),origin_sha256=digest(original),source_kind='target_female_voice_heldout',
                category_verified=False,male_source_residue_evaluable=False,intervals=selected,
                authorization='User explicitly allowed extracting sources from 67clips; spectral labels are proxies')
