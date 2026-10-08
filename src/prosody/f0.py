"""F0 adapters. Torch/FCPE are imported only by offline jobs, never the app."""
import math
import numpy as np


class YinEstimator:
    """FFT YIN difference/CMND on a trailing 80 ms analysis window at 16 kHz.

    Confidence is periodicity, not a calibrated speech probability. No network,
    model, or stateful audio resampling occurs in the PortAudio callback.
    """
    sample_rate = 16000
    window_samples = 1280

    def estimate(self, audio):
        x = np.asarray(audio, dtype=np.float64)
        if x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError('Invalid F0 input')
        if len(x) < self.window_samples:
            x = np.pad(x, (self.window_samples-len(x), 0))
        x = x[-self.window_samples:]
        rms = float(np.sqrt(np.mean(x*x)))
        db = max(-120., 20*math.log10(max(rms, 1e-6)))
        if db < -60:
            return 0., 0., db
        x = x-x.mean()
        # Equal-length comparisons avoid biased lag normalization.
        n = len(x)//2
        max_lag = min(n-1, int(self.sample_rate/65))
        min_lag = int(self.sample_rate/650)
        a = np.pad(x[:n], (0, len(x)-n))
        size = 1 << (2*len(x)-1).bit_length()
        corr = np.fft.irfft(np.conj(np.fft.rfft(a, size))*np.fft.rfft(x, size), size)
        power = np.concatenate(([0.], np.cumsum(x*x)))
        lag = np.arange(max_lag+1)
        diff = np.maximum(0., power[n]+power[n+lag]-power[lag]-2*corr[:max_lag+1])
        cmnd = np.ones(max_lag+1)
        cmnd[1:] = diff[1:]*lag[1:]/np.maximum(np.cumsum(diff[1:]), 1e-12)
        candidates = np.flatnonzero(cmnd[min_lag:] < .15)
        if not len(candidates):
            return 0., 0., db
        peak = int(candidates[0]+min_lag)
        while peak < max_lag and cmnd[peak+1] < cmnd[peak]:
            peak += 1
        confidence = float(np.clip(1-cmnd[peak], 0, 1))
        refined = float(peak)
        if 0 < peak < max_lag:
            left, mid, right = cmnd[peak-1:peak+2]
            denom = left-2*mid+right
            if abs(denom) > 1e-12:
                refined += float(np.clip(.5*(left-right)/denom, -.5, .5))
        return self.sample_rate/refined, confidence, db


class FCPEEstimator:
    """Official bundled torchfcpe 0.0.4 CPU model; isolated/offline dependency."""
    sample_rate = 16000
    def __init__(self, threads=1):
        import torch
        from torchfcpe import spawn_bundled_infer_model
        torch.set_num_threads(threads)
        self.torch = torch
        self.model = spawn_bundled_infer_model(device='cpu').eval()

    def extract(self, audio):
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
            raise ValueError('Invalid FCPE input')
        target = len(x)//160+1
        with self.torch.inference_mode():
            result = self.model.infer(self.torch.from_numpy(x).reshape(1,-1,1), sr=16000,
                decoder_mode='local_argmax', threshold=.006, f0_min=65, f0_max=650,
                interp_uv=False, output_interp_target_length=target)
        f0 = result.detach().cpu().numpy().reshape(-1).astype(np.float32)
        # FCPE public infer returns F0 only: this is a binary voiced decision,
        # deliberately not labelled calibrated confidence/probability.
        f0[~np.isfinite(f0)] = 0
        return f0, (f0 > 0).astype(np.float32)
