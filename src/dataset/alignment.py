"""Banded monotonic DTW. No equal-duration/frame-index assumption."""
import numpy as np
from .features import relative_f0, voiced_delta


def _dynamic(cost,band):
    n,m=cost.shape
    dp=np.full((n+1,m+1),np.inf,dtype=np.float64)
    move=np.zeros((n,m),dtype=np.uint8)
    dp[0,0]=0
    for i in range(n):
        center=i*m/max(1,n)
        for j in range(max(0,int(center-band)),min(m,int(center+band)+2)):
            diag=dp[i,j]; up=dp[i,j+1]+.08; left=dp[i+1,j]+.08
            if diag<=up and diag<=left:
                best=diag; step=0
            elif up<=left:
                best=up; step=1
            else:
                best=left; step=2
            dp[i+1,j+1]=cost[i,j]+best
            move[i,j]=step
    if not np.isfinite(dp[n,m]):
        return np.empty((0,2),dtype=np.int32),np.inf
    reverse=[]; i=n-1; j=m-1
    while i>=0 and j>=0:
        reverse.append((i,j))
        step=move[i,j]
        if step==0:
            i-=1; j-=1
        elif step==1:
            i-=1
        else:
            j-=1
    return np.asarray(reverse[::-1],dtype=np.int32),dp[n,m]/len(reverse)


_accelerated=None


def align(neutral,anime):
    global _accelerated
    if not len(neutral['f0']) or not len(anime['f0']):
        raise ValueError('Empty alignment features')
    def descriptors(f):
        energy=f['energy'][::2]
        energy=np.clip((energy-np.median(energy))/max(3.,float(np.std(energy))),-3,3)
        return np.column_stack((energy,relative_f0(f['f0'])[::2]/6,f['voiced'][::2]))
    a,b=descriptors(neutral),descriptors(anime)
    if not len(a) or not len(b):
        raise ValueError('Empty alignment features')
    cost=np.abs(a[:,None,0]-b[None,:,0])+.5*np.abs(a[:,None,2]-b[None,:,2])
    cost+=.5*np.abs(a[:,None,1]-b[None,:,1])*(a[:,None,2]>0)*(b[None,:,2]>0)
    if _accelerated is None:
        try:
            from numba import njit
            _accelerated=njit(cache=True)(_dynamic)
        except ImportError:
            _accelerated=_dynamic
    path,score=_accelerated(cost,max(5,int(max(len(a),len(b))*.25)))
    if not len(path):
        raise ValueError('No DTW path within band')
    grouped=np.bincount(path[:,0],weights=path[:,1])/np.maximum(1,np.bincount(path[:,0]))
    mapping=np.rint(np.interp(np.arange(len(neutral['f0'])),np.arange(len(grouped))*2,grouped*2)).astype(np.int32)
    mapping=np.clip(mapping,0,len(anime['f0'])-1)
    repeats=0; longest=0
    for step in np.diff(mapping):
        repeats=repeats+1 if step==0 else 0
        longest=max(longest,repeats)
    return mapping,dict(mean_cost=float(score),path_frames=len(path),
        longest_flat_seconds=longest*.01,monotonic=bool(np.all(np.diff(mapping)>=0)),
        normalized_endpoint_error=abs(float(mapping[-1]-(len(anime['f0'])-1)))/max(1,len(anime['f0'])))


def training_sample(neutral,anime,mapping):
    mapping=np.asarray(mapping,dtype=np.int32)
    if len(mapping)!=len(neutral['f0']) or np.any(mapping<0) or np.any(mapping>=len(anime['f0'])):
        raise ValueError('Invalid alignment mapping')
    nf=neutral['f0']; tf=anime['f0'][mapping]
    nv=neutral['voiced']; tv=anime['voiced'][mapping]
    nr=relative_f0(nf); tr=relative_f0(anime['f0'])[mapping]
    ne=neutral['energy']; te=anime['energy'][mapping]
    paired=(nv>0)&(tv>0)
    ne_relative=ne-(float(np.median(ne[nv>0])) if np.any(nv>0) else float(np.median(ne)))
    ae=anime['energy']; av=anime['voiced']
    te_relative=te-(float(np.median(ae[av>0])) if np.any(av>0) else float(np.median(ae)))
    # Register-normalized targets prevent the teacher's speaker identity/base
    # pitch from being mistaken for prosody. Keep raw Hz for later research.
    pitch_offset=np.where(paired,tr-nr,0).astype(np.float32)
    return dict(neutral_f0=nf,neutral_f0_delta=voiced_delta(nr,nv),neutral_relative_st=nr,
        neutral_energy=ne,neutral_energy_delta=np.diff(ne,prepend=ne[0]).astype(np.float32),neutral_voiced=nv,
        target_f0=tf,target_f0_delta=voiced_delta(tr,tv),target_relative_st=tr,target_energy=te,
        target_pitch_offset_st=pitch_offset,target_energy_offset_db=(te_relative-ne_relative).astype(np.float32),
        neutral_relative_energy_db=ne_relative.astype(np.float32),target_relative_energy_db=te_relative.astype(np.float32),
        target_energy_raw_offset_db=(te-ne).astype(np.float32),
        target_voiced=tv,valid_pitch_target=paired,alignment=mapping,time=neutral['time'],
        hop_seconds=neutral['hop_seconds'],schema_version=np.int32(1),
        delta_units=np.array('adjacent-frame semitones; energy dBFS; offset targets register-normalized'))


def split_ids(ids,seed=500):
    ids=list(ids)
    if len(set(ids))!=len(ids):
        raise ValueError('Duplicate sentence ID')
    ordered=sorted(ids)
    np.random.default_rng(seed).shuffle(ordered)
    n=len(ordered); train=int(n*.8); val=int(n*.1)
    return dict(train=ordered[:train],validation=ordered[train:train+val],test=ordered[train+val:])
