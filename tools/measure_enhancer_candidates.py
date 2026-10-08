"""Measure every affordable post-VC restoration candidate on one real utterance.

Runs the VC stage once, then applies each enhancer to the same MeanVC2 output so
the comparison isolates the enhancer. Reports RTF per stage, peak RAM, and the
4-8 kHz share that tracks intelligibility on a 16 kHz-bandlimited signal. Offline
WAV only; no audio device, no listening.
"""
import argparse, json, sys, time, wave
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vc.voice_library import DEFAULT_VOICE_ID


def read48(path):
    with wave.open(str(path), 'rb') as handle:
        return np.frombuffer(handle.readframes(handle.getnframes()),
                             dtype='<i2').astype(np.float32) / 32768


def measure(x):
    x = x.astype('float64')
    x = x - x.mean()
    n = len(x)
    power = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1 / 48000)

    def band(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return float(power[mask].sum()) if mask.any() else 0.0

    low, high, very = band(100, 1000), band(4000, 8000), band(8000, 20000)
    total = low + band(1000, 4000) + high + very
    return {'tilt_4_8k_db': round(10 * np.log10(max(high, 1e-12) / max(low, 1e-12)), 2),
            'share_4_8k': round(high / total, 5),
            'share_8k_plus': round(very / total, 5)}


def normalise(x, target=0.1):
    """Same DC + constant RMS contract the app enhancers use."""
    x = x - float(np.mean(x, dtype=np.float64))
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(x)))
    gain = min(target / max(rms, 1e-12), .98 / max(peak, 1e-12))
    return (x * gain).astype(np.float32)


def write_wav(folder, label, audio):
    if folder is None:
        return None
    import soundfile as sf
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f'{label}.wav'
    sf.write(destination, audio, 48000, subtype='FLOAT')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=ROOT / 'recordings/v011_meanvc2_continuity/source/source_long.wav')
    parser.add_argument('--seconds', type=float, default=25.0)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--write-dir', type=Path, default=None,
                        help='Write each candidate as a 48 kHz WAV for blind listening.')
    parser.add_argument('--report', type=Path,
                        default=ROOT / 'validation/v011/enhancer_candidates.json')
    args = parser.parse_args()

    from src.vc.utterance import convert_utterance, utterance_limit
    from src.vc.meanvc2_phrase import MeanVC2PhraseBackend
    from src.vc.models import profile
    from src.vc.resampler import StreamingResampler

    process = psutil.Process()
    audio = read48(args.source)[:round(args.seconds * 48000)]
    seconds = len(audio) / 48000
    selected = profile(DEFAULT_VOICE_ID)
    down = StreamingResampler(48000, 16000)
    up = StreamingResampler(16000, 48000)

    backend = MeanVC2PhraseBackend(threads=args.threads)
    backend.load(ROOT / 'models' / selected['folder'])
    backend.select_profile(selected)
    backend.warmup()
    base_ram = process.memory_info().rss
    try:
        started = time.perf_counter()
        voice = convert_utterance(backend, down, up, audio,
                                  max_seconds=utterance_limit('lavasr'))
        vc_seconds = time.perf_counter() - started
    finally:
        backend.unload()
    vc_ram = process.memory_info().rss
    print(f'VC stage: {vc_seconds:.2f}s rtf={vc_seconds / seconds:.4f} '
          f'ram={vc_ram / 1024 ** 2:.0f} MiB', flush=True)

    write_wav(args.write_dir, 'meanvc2_only', voice)
    rows = [dict(candidate='meanvc2_only', enhancer_rtf=0.0,
                 total_rtf=vc_seconds / seconds, exact_length=True,
                 finite=bool(np.isfinite(voice).all()),
                 peak_ram_bytes=vc_ram, **measure(voice))]

    from scipy.signal import resample_poly
    import torch

    reduced = resample_poly(voice, 1, 3).astype(np.float32)

    def timed(label, function, total_rtf_base):
        peak_before = process.memory_info().rss
        began = time.perf_counter()
        try:
            result = function()
            elapsed = time.perf_counter() - began
            peak = max(process.memory_info().rss, peak_before)
            row = dict(candidate=label, enhancer_seconds=round(elapsed, 3),
                       enhancer_rtf=round(elapsed / seconds, 4),
                       total_rtf=round(total_rtf_base + elapsed / seconds, 4),
                       exact_length=len(result) == len(voice),
                       finite=bool(np.isfinite(result).all()),
                       peak_ram_bytes=peak, **measure(result))
        except Exception as error:
            row = dict(candidate=label, error=repr(error)[:200])
        if 'error' not in row:
            write_wav(args.write_dir, label, result)
            row['output'] = str(args.write_dir / f'{label}.wav')
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in
                          ('candidate', 'enhancer_rtf', 'total_rtf', 'share_4_8k',
                           'share_8k_plus', 'tilt_4_8k_db', 'exact_length', 'finite', 'error')},
                         ensure_ascii=False), flush=True)

    lava_root = ROOT / 'vc_models/post_lavasr'
    sys.path.insert(0, str(lava_root / 'vendor'))
    sys.path.insert(0, str(lava_root / 'repo'))
    from LavaSR.model import LavaEnhance2
    from LavaSR.enhancer.linkwitz_merge import FastLRMerge
    lava = LavaEnhance2(str(lava_root / 'weights'), 'cpu')
    torch.set_num_threads(args.threads)

    def make_lava(cutoff, denoise):
        model = LavaEnhance2(str(lava_root / 'weights'), 'cpu')
        model.bwe_model.lr_refiner = FastLRMerge(device='cpu', cutoff=cutoff,
                                                 transition_bins=1024)
        def run():
            with torch.inference_mode():
                out = model.enhance(torch.from_numpy(reduced.copy()).unsqueeze(0),
                                    denoise=denoise, batch=False)
            array = np.asarray(out.detach().float().cpu().numpy(),
                               dtype=np.float32).reshape(-1)
            array = array[:len(voice)]
            if len(array) < len(voice):
                array = np.pad(array, (0, len(voice) - len(array)))
            return normalise(array)
        return run

    for cutoff in (8000, 6000, 10000):
        for denoise in ((False, True) if cutoff == 8000 else (False,)):
            label = f'lavasr_cutoff{cutoff}' + ('_denoise' if denoise else '')
            timed(label, make_lava(cutoff, denoise), vc_seconds / seconds)

    nova_root = ROOT / 'vc_models/post_novasr'
    for entry in (str(nova_root / 'repo'), str(nova_root / 'vendor')):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    def make_nova():
        from NovaSR import FastSR
        model = FastSR(str(nova_root / 'weights/pytorch_model_v1.bin'), half=False)

        def run():
            with torch.inference_mode():
                out = model.infer(torch.from_numpy(reduced.copy()).unsqueeze(0).unsqueeze(0))
            array = np.asarray(out.detach().float().cpu().numpy(),
                               dtype=np.float32).reshape(-1)
            array = array[:len(voice)]
            if len(array) < len(voice):
                array = np.pad(array, (0, len(voice) - len(array)))
            return normalise(array)
        return run

    timed('novasr', make_nova(), vc_seconds / seconds)

    summary = dict(input_kind='Offline WAV; VC run once then each enhancer on the same output',
                   source=str(args.source), seconds=seconds, threads=args.threads,
                   vc_rtf=round(vc_seconds / seconds, 4), rows=rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('saved', args.report)


if __name__ == '__main__':
    main()