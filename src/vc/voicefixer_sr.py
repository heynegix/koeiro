"""Pinned local VoiceFixer2 post-VC restoration (fifth delivery mode).

Uses the same Render-AI-Team voicefixer2 package and the byte-identical official
checkpoints measured offline in validation/v011/post_vc2_comparison_report.md
(Q012, RTF 6.4-7.4, peak RAM up to ~4.0 GiB on the 25 s source). The upstream
hf:// checkpoint endpoints are unavailable, so the official Zenodo mirror is
pinned here with SHA-256. Runs only inside the MeanVC2 worker; the vendor tree
must not shadow torch/numpy/scipy/librosa/soundfile. No audio devices, no
downloads, no training.

The upstream model restores at 44.1 kHz; this wrapper converts the finished
MeanVC2 48 kHz utterance to 44.1 kHz, restores, and resamples the result back
to 48 kHz while preserving duration.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from ..runtime_paths import asset_root


def _resample(x, src_rate, dst_rate):
    """Linear-interpolation resampling on numpy only (worker has no scipy).

    Utterance post-processing resamples complete buffers, so a simple linear
    grid at the target rate is sufficient and deterministic. The real-model RPC
    verification checks length, finiteness and listening-equivalent energy.
    """
    if src_rate == dst_rate:
        return np.asarray(x, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    n_out = int(round(len(x) * dst_rate / src_rate))
    src_grid = np.arange(n_out, dtype=np.float64) * src_rate / dst_rate
    base = np.floor(src_grid).astype(np.int64)
    frac = src_grid - base
    lo = x[np.clip(base, 0, len(x) - 1)]
    hi = x[np.clip(base + 1, 0, len(x) - 1)]
    return (lo * (1 - frac) + hi * frac).astype(np.float32)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


class VoiceFixerSR:
    def __init__(self, threads=1):
        root = asset_root()
        info = json.loads(Path(__file__).with_name('voicefixer_runtime.json').read_text('utf-8'))
        folder = (root / 'vc_models/post_voicefixer2').resolve()
        weights = root / info['weights_dir']
        for name, expected in info['sha256'].items():
            path = weights / name
            if not path.is_file() or sha256(path) != expected:
                raise ValueError('VoiceFixer checkpoint checksum mismatch: ' + name)
        banned = ('numpy', 'scipy', 'torch', 'torchaudio', 'soundfile', 'librosa')
        vendor = folder / 'vendor'
        for entry in vendor.iterdir():
            lowered = entry.name.lower()
            if any(lowered == name or lowered.startswith(name + '-') for name in banned):
                raise RuntimeError('Vendor shadows a worker dependency: ' + entry.name)
        import torch
        torch.set_num_threads(max(1, int(threads)))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        sys.path.insert(0, str(vendor))
        sys.path.insert(0, str(folder / 'repo'))
        try:
            import cached_path as cached_path_module
            expected_paths = {'hf://voicefixer/voicefixer/vf.ckpt': weights / 'vf.ckpt',
                              'hf://voicefixer/vocoder/model.ckpt-1490000_trimed.pt': weights / 'vocoder_44100.pt'}
            def local_cached_path(url, *args, **kwargs):
                pinned = expected_paths.get(str(url))
                if pinned is None:
                    return cached_path_module.cached_path(url, *args, **kwargs)
                if not pinned.is_file() or sha256(pinned) != info['sha256'][pinned.name]:
                    raise ValueError('VoiceFixer checkpoint checksum mismatch: ' + pinned.name)
                return str(pinned)
            cached_path_module.cached_path = local_cached_path
            from voicefixer import VoiceFixer
            self.restorer = VoiceFixer()
        finally:
            for entry in (str(vendor), str(folder / 'repo')):
                try:
                    sys.path.remove(entry)
                except ValueError:
                    pass
        self.restore = self.restorer.restore
        self.sha256 = dict(info['sha256'])

    def process(self, audio):
        """48 kHz float32 mono -> 48 kHz float32 mono, same duration."""
        import wave
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim != 1 or not np.isfinite(x).all() or len(x) < 480:
            raise RuntimeError('Invalid VoiceFixer input')
        if float(np.max(abs(x))) < 1e-3:
            return np.zeros_like(x)
        up44 = _resample(x, 48000, 44100)
        temp = Path(tempfile.mkdtemp(prefix='vfr_', dir=os.environ.get('TMP')))
        try:
            work, out = temp / 'in.wav', temp / 'out.wav'
            samples = (np.clip(up44, -1.0, 1.0) * 32767.0).astype('<i2')
            with wave.open(str(work), 'wb') as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(44100)
                stream.writeframes(samples.tobytes())
            self.restore(str(work), str(out), cuda=False, mode=0)
            with wave.open(str(out), 'rb') as stream:
                rate = stream.getframerate()
                width = stream.getsampwidth()
                channels = stream.getnchannels()
                frames = stream.readframes(stream.getnframes())
            if rate != 44100 or width != 2:
                raise RuntimeError('VoiceFixer output format changed: rate=%d width=%d' % (rate, width))
            restored = np.frombuffer(frames, dtype='<i2').astype(np.float32) / 32768.0
            if channels == 2:
                restored = restored.reshape(-1, 2).mean(axis=1)
        finally:
            for item in (work, out):
                try:
                    item.unlink()
                except OSError:
                    pass
            try:
                temp.rmdir()
            except OSError:
                pass
        if not np.isfinite(restored).all() or float(np.max(abs(restored), initial=0)) < 1e-5:
            raise RuntimeError('Invalid/silent VoiceFixer output')
        back48 = _resample(restored, 44100, 48000)
        if abs(len(back48) / 48000.0 - len(audio) / 48000.0) > .05:
            raise RuntimeError('VoiceFixer output duration changed beyond 50ms')
        result = back48[:len(audio)].astype(np.float32, copy=True)
        if len(result) < len(audio):
            result = np.pad(result, (0, len(audio) - len(result)))
        peak = float(np.max(abs(result)))
        if peak > .999:
            result *= .999 / peak
        return result
