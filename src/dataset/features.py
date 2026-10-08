from pathlib import Path
import hashlib
import math
import numpy as np


def read_wave(path):
    from scipy.io import wavfile
    rate,audio=wavfile.read(path)
    if rate<8000 or rate>192000 or audio.ndim not in (1,2) or not len(audio):
        raise ValueError('Invalid WAV shape/rate/duration')
    if np.issubdtype(audio.dtype,np.integer):
        if audio.dtype==np.uint8:
            audio=(audio.astype(np.float32)-128)/128
        else:
            audio=audio.astype(np.float32)/(2**(np.iinfo(audio.dtype).bits-1))
    else:
        audio=audio.astype(np.float32)
    if audio.ndim==2:
        audio=audio.mean(axis=1)
    if not np.isfinite(audio).all():
        raise ValueError('Non-finite WAV')
    if np.max(np.abs(audio))<1e-5:
        raise ValueError('Silent WAV')
    return rate,audio


def validate_pairs(root, expected_ids=None):
    root=Path(root)
    files={style:{} for style in ('neutral','anime')}
    problems=[]
    duplicates=[]; digests={}
    for style in files:
        for path in sorted((root/'generated'/style).rglob('*.wav')):
            identifier=path.stem
            if identifier in files[style]:
                problems.append(dict(id=identifier,style=style,reason='Duplicate ID',path=str(path)))
                files[style][identifier]=None
            else:
                files[style][identifier]=path
    ids=set(expected_ids) if expected_ids is not None else set(files['neutral'])|set(files['anime'])
    pairs=[]
    for identifier in sorted(ids):
        pair={}; valid=True
        for style in files:
            path=files[style].get(identifier)
            try:
                if path is None:
                    raise ValueError('Pair file missing/duplicate')
                rate,audio=read_wave(path)
                pair[style]=dict(path=str(path),sample_rate=rate,duration=len(audio)/rate,
                    rms=float(np.sqrt(np.mean(audio.astype(np.float64)**2))),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                digest=pair[style]['sha256']
                if digest in digests:
                    duplicates.append(dict(id=identifier,style=style,reason='Identical WAV content',
                        matches=digests[digest]))
                else:
                    digests[digest]=dict(id=identifier,style=style)
            except (OSError,ValueError,TypeError) as error:
                problems.append(dict(id=identifier,style=style,reason=str(error),path=str(path)))
                valid=False
        if valid:
            pairs.append(dict(id=identifier,**pair))
    for style in files:
        for identifier in files[style].keys()-ids:
            problems.append(dict(id=identifier,style=style,reason='Unexpected ID'))
    return dict(expected_pairs=len(ids),valid_pairs=pairs,problems=problems,
        duplicate_audio=duplicates,
        wav_counts={style:len(paths) for style,paths in files.items()})


def extract_features(path,estimator):
    from scipy.signal import resample_poly
    rate,audio=read_wave(path)
    divisor=math.gcd(rate,16000)
    reduced=resample_poly(audio,16000//divisor,rate//divisor).astype(np.float32)
    f0,voiced=estimator.extract(reduced)
    times=np.arange(len(f0),dtype=np.float64)*.010
    # Centered 25 ms RMS window, computed independently of pitch extraction.
    half=200
    squared=np.pad(reduced.astype(np.float64)**2,(half,half))
    cumulative=np.concatenate(([0.],np.cumsum(squared)))
    indices=np.minimum(np.arange(len(f0))*160,len(reduced))
    rms=np.sqrt(np.maximum(0,(cumulative[indices+2*half]-cumulative[indices])/(2*half)))
    energy=(20*np.log10(np.maximum(rms,1e-6))).astype(np.float32)
    silence=energy<-60
    voiced=np.asarray(voiced,dtype=np.float32)
    voiced[silence]=0
    f0=np.asarray(f0,dtype=np.float32)
    f0[(voiced<=0)|~np.isfinite(f0)|(f0<65)|(f0>650)]=0
    voiced[f0==0]=0
    return dict(f0=f0,voiced=voiced,energy=energy,rms=rms.astype(np.float32),time=times,
        silence_mask=silence,sample_rate=np.int32(rate),analysis_sample_rate=np.int32(16000),
        duration=np.float64(len(audio)/rate),hop_seconds=np.float64(.010),
        voiced_kind=np.array('FCPE binary decision, not calibrated probability'))


def relative_f0(f0):
    f0=np.asarray(f0)
    valid=np.isfinite(f0)&(f0>0)
    output=np.zeros(len(f0),dtype=np.float32)
    baseline=float(np.median(f0[valid])) if valid.any() else 0.
    if baseline:
        output[valid]=12*np.log2(f0[valid]/baseline)
    return output


def voiced_delta(values,voiced):
    delta=np.zeros(len(values),dtype=np.float32)
    valid=(voiced[1:]>0)&(voiced[:-1]>0)
    delta[1:]=np.where(valid,np.diff(values),0)
    return delta


def metrics(features):
    f0=features['f0']; voiced=features['voiced']>0
    pitch=f0[voiced]; relative=relative_f0(f0)
    energy=features['energy'][~features['silence_mask']]
    deltas=voiced_delta(relative,features['voiced'])
    def distribution(x):
        if not len(x):
            return dict(median=0.,mean=0.,std=0.,iqr=0.,range=0.)
        return dict(median=float(np.median(x)),mean=float(np.mean(x)),std=float(np.std(x)),
            iqr=float(np.percentile(x,75)-np.percentile(x,25)),range=float(np.max(x)-np.min(x)))
    indices=np.flatnonzero(voiced)
    slope=float(np.polyfit(features['time'][indices],relative[indices],1)[0]) if len(indices)>2 else 0.
    return dict(f0=distribution(pitch),pitch_st=distribution(relative[voiced]),
        f0_slope_st_s=slope,f0_velocity_abs_mean_st_s=float(np.mean(np.abs(deltas))/.010),
        energy_db=distribution(energy),duration=float(features['duration']),voiced_ratio=float(voiced.mean()))
