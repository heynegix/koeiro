"""What the ASR trace does to CUDA graph recording, with the whole traceback.

Loads MeanVC2 on CUDA and runs the offline pass's attempts: a control capture
that proves graphs work on this machine at all, the ASR stage with its host
scalar recipe and offset checks, every VC shape the buffered route builds, and
the same ASR call under device scalars, host scalars and each capture error
mode. The stage's own recipe is the host scalars, because the trace reads the
offset back to the host to index the position table; the device scalars and the
modes record what this torch accepts. No audio device is opened and nothing is
listened to.

    python tools/probe_meanvc2_capture.py
"""
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def attempt(label, run):
    try:
        value = run()
        print(f'{label}: OK {value}', flush=True)
    except Exception:
        print(f'{label}: FAILED', flush=True)
        traceback.print_exc()
        print('', flush=True)


def capture_with_mode(torch, run, mode):
    """Record one call exactly like MeanVC2Backend._capture, with a known mode."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.inference_mode():
        for _ in range(2):
            run()
    side.synchronize()
    graph = torch.cuda.CUDAGraph()
    options = {} if mode is None else dict(capture_error_mode=mode)
    with torch.inference_mode(), torch.cuda.graph(graph, stream=side, **options):
        run()
    graph.replay()
    return graph


def main():
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is not available')
    print(f'torch {torch.__version__} cuda {torch.version.cuda} '
          f'{torch.cuda.get_device_name(0)}', flush=True)

    import numpy as np
    import soundfile as sf
    from src.runtime_paths import asset_root
    from src.vc.meanvc2 import MeanVC2Backend, OfflineAsrStage, OfflineVcStage
    from src.vc.models import DEFAULT_VOICE_ID, profile

    folder = asset_root() / 'models' / profile(DEFAULT_VOICE_ID)['folder']
    audio, rate = sf.read(folder / 'reference.wav', dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    backend = MeanVC2Backend(threads=4, device='cuda')
    backend.load(folder)
    backend.select_profile(profile(DEFAULT_VOICE_ID))
    backend.warmup()
    print(f'profile vc_group_chunks {backend.vc_group_chunks} steps {backend.steps}', flush=True)
    print(f'retargeted device constants: {backend.traced_devices_moved}', flush=True)

    def control():
        value = torch.zeros(8, 8, device='cuda')

        def run():
            value.add_(1.0)
        graph = capture_with_mode(torch, run, None)
        graph.replay()
        return float(value[0, 0].item())

    def asr_stage():
        stage = OfflineAsrStage(backend, 19)
        return f'baked window {stage.baked_window} at offset {int(stage.offset)}'

    attempt('control capture (8x8 add)', control)
    attempt('ASR stage (19 frames)', asr_stage)

    frames = backend.vc_group_chunks * 12 + 4
    kv = [tuple(torch.zeros(1, 2, backend.max_kv_frames, 64, device='cuda') for _ in range(2))
          for _ in range(4)]
    for step in range(backend.steps):
        for name, cache in (('no cache', None), ('24-frame cache', kv)):
            attempt(f'VC stage step {step} frames {frames} ({name})',
                    lambda step=step, cache=cache: OfflineVcStage(backend, frames, step, cache))

    window = torch.zeros(1, 19, 80, device='cuda')
    att = torch.zeros_like(backend.att_cache)
    cnn = torch.zeros_like(backend.cnn_cache)
    device_offset = torch.full((), 8, dtype=torch.int64, device='cuda')
    host_offset = torch.full((), 8, dtype=torch.int64)
    host_cache = torch.full((), 8, dtype=torch.int64)

    def asr_device():
        backend.asr(window, device_offset, backend.cache_size_tensor, att, cnn)

    def asr_host():
        backend.asr(window, host_offset, host_cache, att, cnn)

    attempt('ASR capture, device scalars',
            lambda: bool(capture_with_mode(torch, asr_device, None)))
    attempt('ASR capture, host scalars (the stage recipe)',
            lambda: bool(capture_with_mode(torch, asr_host, None)))

    for mode in ('global', 'thread_local', 'relaxed'):
        attempt(f'ASR capture mode {mode} (host scalars)',
                lambda mode=mode: bool(capture_with_mode(torch, asr_host, mode)))


if __name__ == '__main__':
    main()
