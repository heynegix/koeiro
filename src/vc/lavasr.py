"""Pinned local LavaSR v2 bandwidth restoration for the utterance delivery mode.

Uses the byte-identical official YatharthS/LavaSR checkpoint measured offline in
validation/v011/post_vc_comparison_report.md (LavaSR v1 condition, denoise=False,
FastLRMerge cutoff 8 kHz, RTF 0.078-0.101, peak RAM 0.47 GiB on the 25 s source).
Measured here it is the only already-installed restoration model on this CPU that
is cheap enough for live utterance delivery: FlashSR costs RTF ~3.1 and 3.9 GiB
at 30 s, so it cannot keep up with conversation.

Runs only inside the MeanVC2 worker, which already provides torch/torchaudio. The
repo and vendor trees are removed from sys.path again after import so they cannot
shadow the worker's numpy/scipy/torch. Weights are local and SHA-256 pinned; no
download, no training, no audio device.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from ..runtime_paths import asset_root


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


class LavaSR:
    def __init__(self, cutoff_hz=8000, transition_bins=1024, denoise=False):
        root = asset_root()
        info = json.loads(Path(__file__).with_name('lavasr_runtime.json').read_text('utf-8'))
        folder = (root / 'vc_models/post_lavasr').resolve()
        weights = (root / info['weights_dir']).resolve()
        if not weights.is_relative_to(folder):
            raise ValueError('Invalid LavaSR weights path')
        for name, expected in info['sha256'].items():
            path = weights / name
            if not path.is_file() or sha256(path) != expected:
                raise ValueError('LavaSR checkpoint checksum mismatch: ' + name)
        if type(cutoff_hz) is not int or not 1000 <= cutoff_hz <= 16000:
            raise ValueError('LavaSR cutoff must be 1000..16000 Hz')
        if denoise not in (True, False):
            raise ValueError('LavaSR denoise must be a boolean')
        banned = ('numpy', 'scipy', 'torch', 'torchaudio', 'soundfile', 'librosa', 'onnxruntime')
        vendor = folder / 'vendor'
        for entry in vendor.iterdir():
            lowered = entry.name.lower()
            if any(lowered == name or lowered.startswith(name + '-') for name in banned):
                raise RuntimeError('Vendor shadows a worker dependency: ' + entry.name)
        repo, vendor_path = folder / 'repo', str(vendor)
        sys.path.insert(0, vendor_path)
        sys.path.insert(0, str(repo))
        try:
            # A local path is passed so huggingface_hub never reaches the network.
            from LavaSR.model import LavaEnhance2
            from LavaSR.enhancer.linkwitz_merge import FastLRMerge
            self.model = LavaEnhance2(str(weights), 'cpu')
            self.model.bwe_model.lr_refiner = FastLRMerge(device='cpu', cutoff=cutoff_hz,
                                                         transition_bins=transition_bins)
        finally:
            for entry in (str(repo), vendor_path):
                try:
                    sys.path.remove(entry)
                except ValueError:
                    pass
        import torch
        self.torch = torch
        self.sha256 = dict(info['sha256'])
        self.cutoff_hz = cutoff_hz
        self.transition_bins = transition_bins
        # Denoise runs a separate upstream stage inside the same checkpoint. It is off by
        # default because the measured 4-8 kHz comparison found no benefit from it on this
        # material and it costs extra time; exposed so the choice can be re-tested without
        # touching the vendored model.
        self.denoise = bool(denoise)
        self.provenance = dict(checkpoint='YatharthS/LavaSR enhancer_v2 (local, SHA-256 pinned)',
                               denoise=self.denoise, cutoff_hz=cutoff_hz,
                               transition_bins=transition_bins)

    def process(self, audio):
        """48 kHz float32 mono -> 48 kHz float32 mono, same duration."""
        from scipy.signal import resample_poly
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim != 1 or not np.isfinite(x).all() or len(x) < 480:
            raise RuntimeError('Invalid LavaSR input')
        if float(np.max(abs(x), initial=0)) < 1e-6:
            return np.zeros_like(x)
        reduced = resample_poly(x, 1, 3).astype(np.float32)
        with self.torch.inference_mode():
            enhanced = self.model.enhance(self.torch.from_numpy(reduced.copy()).unsqueeze(0),
                                          denoise=self.denoise, batch=False)
        result = np.asarray(enhanced.detach().float().cpu().numpy(), dtype=np.float32).reshape(-1)
        if not np.isfinite(result).all() or float(np.max(abs(result), initial=0)) < 1e-6:
            raise RuntimeError('Invalid/silent LavaSR output')
        if abs(len(result) - len(x)) > 480:
            raise RuntimeError('LavaSR output duration changed beyond 10ms')
        result = result[:len(x)].astype(np.float32, copy=True)
        if len(result) < len(x):
            result = np.pad(result, (0, len(x) - len(result)))
        # Same DC removal and constant RMS gain as the blind comparison package.
        # No peak normalisation, matching the measured LavaSR condition.
        result = result - float(np.mean(result, dtype=np.float64))
        rms = float(np.sqrt(np.mean(result.astype(np.float64) ** 2)))
        gain = min(.1 / max(rms, 1e-12), .98 / max(float(np.max(abs(result))), 1e-12))
        return (result * gain).astype(np.float32)