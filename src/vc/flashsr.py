"""Pinned local FlashSR post-VC inference matching the selected P009 audition."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

from ..runtime_paths import asset_root


class FlashSR:
    def __init__(self, intra_op_threads=1):
        root=asset_root()
        info=json.loads(Path(__file__).with_name('flashsr_runtime.json').read_text('utf-8'))
        checkpoint=root/info['checkpoint']
        if checkpoint.resolve().is_relative_to(root):pass
        else:raise ValueError('Invalid FlashSR checkpoint path')
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=info['sha256']:
            raise ValueError('FlashSR checkpoint checksum mismatch')
        if type(intra_op_threads) is not int or not 1<=intra_op_threads<=4:
            raise ValueError('FlashSR ORT thread count must be 1..4')
        # Dedicated vendor contains ORT only; the app environment is unchanged.
        sys.path.insert(0,str(root/'vc_models/post_flashsr/vendor'))
        import onnxruntime as ort
        options=ort.SessionOptions()
        options.intra_op_num_threads=intra_op_threads
        options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(checkpoint),options,providers=['CPUExecutionProvider'])
        self.sha256=info['sha256']
        self.intra_op_threads=intra_op_threads

    def process(self,audio):
        from scipy.signal import resample_poly
        x=resample_poly(audio,1,3).astype(np.float32)
        if np.max(abs(x),initial=0)<1e-6:return np.zeros_like(audio)
        result=self.session.run(None,{'x':x[None,None,:]})[0].reshape(-1)
        if not np.isfinite(result).all() or abs(len(result)-len(audio))>2:
            raise RuntimeError('Invalid FlashSR output')
        result=result[:len(audio)].astype(np.float32)
        if len(result)<len(audio):result=np.pad(result,(0,len(audio)-len(result)))
        # Same official peak normalization followed by the blind package's
        # DC subtraction + constant RMS gain. No EQ or compressor.
        result=result/(np.max(abs(result))+1e-7)*.999
        result-=float(np.mean(result,dtype=np.float64))
        rms=float(np.sqrt(np.mean(result.astype(np.float64)**2)))
        gain=min(.1/max(rms,1e-12),.98/max(float(np.max(abs(result))),1e-12))
        return (result*gain).astype(np.float32)
