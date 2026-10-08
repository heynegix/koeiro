"""Pinned local MossFormer2_SR_48K post-VC band restoration (quality mode).

Loads the same official ClearVoice checkpoints measured in the offline post-VC
comparison (RTF 21.7-23.0 on this N150, peak RAM ~1.6 GiB). Runs only inside
the MeanVC2 worker via the isolated vendor environment; no audio devices,
downloads or training. The finished MeanVC2 48 kHz utterance waveform is fed
exactly like the measured adapter's DataReader path (48 kHz in, no input
normalization); the model restores the 16-48 kHz band in place at 48 kHz.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from ..runtime_paths import asset_root


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


class MossFormerSR:
    def __init__(self, threads=1):
        root=Path(__file__).resolve().parents[2]
        info=json.loads(Path(__file__).with_name('mossformer_runtime.json').read_text('utf-8'))
        folder=(root/'vc_models/post_mossformer_sr').resolve()
        checkpoint=root/info['checkpoint_dir']
        if not checkpoint.resolve().is_relative_to(root):
            raise ValueError('Invalid MossFormer checkpoint path')
        for name,expected in info['sha256'].items():
            path=checkpoint/name
            if not path.is_file() or sha256(path)!=expected:
                raise ValueError('MossFormer checkpoint checksum mismatch: '+name)
        for filename in (checkpoint/'last_best_checkpoint').read_text('utf-8').splitlines():
            if filename and (Path(filename).name!=filename or not (checkpoint/filename).is_file()):
                raise FileNotFoundError('Required pretrained weights missing: '+filename)
        # Vendor tree holds clearvoice plus its pure-python dependencies. torch,
        # torchaudio, numpy, scipy, librosa and soundfile must keep resolving to
        # the worker environment; the vendor must not shadow any of them.
        banned=('numpy','scipy','torch','torchaudio','soundfile','librosa')
        vendor=folder/'vendor'
        for entry in vendor.iterdir():
            lowered=entry.name.lower()
            if any(lowered==name or lowered.startswith(name+'-') for name in banned):
                raise RuntimeError('Vendor shadows a worker dependency: '+entry.name)
        import torch
        torch.set_num_threads(max(1,int(threads)))
        try:torch.set_num_interop_threads(1)
        except RuntimeError:pass
        sys.path.insert(0,str(vendor))
        previous=os.getcwd()
        try:
            from clearvoice import ClearVoice
            # decode_one_audio is the function the measured post-VC adapter
            # reached through model(input_path=...); decode_one_audio_batch
            # has a b==1 squeeze/index bug, so it is not used.
            from clearvoice.utils.decode import decode_one_audio
            # ClearVoice resolves its relative checkpoint_dir against the cwd.
            os.chdir(folder)
            self.model=ClearVoice(task='speech_super_resolution',model_names=['MossFormer2_SR_48K'])
        finally:
            os.chdir(previous)
            try:sys.path.remove(str(vendor))
            except ValueError:pass
        self.network=self.model.models[0]
        self.decode=decode_one_audio
        self.sha256=dict(info['sha256'])

    def process(self,audio):
        """48 kHz float32 mono -> 48 kHz float32 mono, same length."""
        import torch
        x=np.asarray(audio,dtype=np.float32)
        if x.ndim!=1 or not np.isfinite(x).all() or len(x)<480:
            raise RuntimeError('Invalid MossFormer input')
        # Real-model diagnostic: an entirely near-silent input makes the model
        # hallucinate audio after ~0.7 s (RMS up to ~0.17). Utterances from the
        # collector always contain voiced frames, so this guard only fires on
        # degenerate input; genuine speech peaks exceed this level by far.
        if float(np.max(abs(x)))<1e-3:
            return np.zeros_like(x)
        # Same 48 kHz domain the measured adapter used: no resample, no
        # normalization. >20 s utterances take the official sliding-window path.
        # decode_one_audio expects a numpy (1, N) row and returns 1-D audio.
        with torch.inference_mode():
            output=self.decode(self.network.model,'cpu',x[None,:].copy(),self.network.args)
        output=np.asarray(output,dtype=np.float32).squeeze()
        if output.ndim!=1 or not np.isfinite(output).all() or np.max(abs(output),initial=0)<1e-5:
            raise RuntimeError('Invalid/silent MossFormer output')
        if abs(len(output)/48000-len(audio)/48000)>.05:
            raise RuntimeError('MossFormer output length changed beyond 50ms')
        result=output[:len(audio)].astype(np.float32,copy=True)
        if len(result)<len(audio):result=np.pad(result,(0,len(audio)-len(result)))
        peak=float(np.max(abs(result)))
        if peak>.999:result*=.999/peak
        return result
