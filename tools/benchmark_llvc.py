"""Offline human-WAV gate before any live AI integration; correct time/audio RTF."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import psutil
from src.vc.llvc import LLVCBackend
from src.vc.resampler import StreamingResampler
from tools.compare_presets import read_wav, write_wav


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--backend', choices=['torch', 'onnx'], default='torch')
    parser.add_argument('--output-dir', type=Path, default=Path('recordings/v03/llvc'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source, rate = read_wav(args.input)
    if rate != 48000:
        raise ValueError('Benchmark input must be 48 kHz')
    if args.backend == 'onnx':
        from src.vc.llvc_onnx import LLVCOnnxBackend
        backend = LLVCOnnxBackend(args.factor, args.threads)
    else:
        backend = LLVCBackend(args.factor, args.threads)
    backend.load(Path('models/research_llvc'))
    backend.warmup()
    down, up = StreamingResampler(48000, 16000), StreamingResampler(16000, 48000)
    n = backend.chunk_samples*3
    # Flush causal FIR + model lookahead alignment for same-length files.
    delay = 144  # two FIRs (96 samples) + model alignment (48 samples) @48k.
    padded = np.pad(source, (0, delay + (-len(source)-delay) % n))
    output = np.empty_like(padded)
    times, resample_times = [], []
    process = psutil.Process()
    start, cpu = time.perf_counter(), time.process_time()
    max_ram = process.memory_info().rss
    for offset in range(0, len(padded), n):
        begin = time.perf_counter()
        audio = down.process(padded[offset:offset+n])
        middle = time.perf_counter()
        converted = backend.process_chunk(audio)
        end = time.perf_counter()
        output[offset:offset+n] = up.process(converted)
        finish = time.perf_counter()
        times.append((end-middle)*1000)
        resample_times.append((middle-begin+finish-end)*1000)
        if offset % (n*50) == 0:
            max_ram = max(max_ram, process.memory_info().rss)
    duration = time.perf_counter()-start
    cpu_seconds = time.process_time()-cpu
    converted = output[delay:delay+len(source)]
    name = f'llvc-f{args.factor}-t{args.threads}' + ('-onnx' if args.backend == 'onnx' else '')
    write_wav(args.output_dir/(name+'.wav'), converted, rate)
    def stats(values):
        return dict(average_ms=float(np.mean(values)), p95_ms=float(np.percentile(values,95)),
                    maximum_ms=float(np.max(values)))
    report = dict(**backend.get_stats(), input=str(args.input), input_seconds=len(source)/rate,
                  output_rate=rate, output_frames=len(converted), inference_seconds=sum(times)/1000,
                  total_processing_seconds=duration, rtf=duration/(len(source)/rate),
                  inference=stats(times), resample=stats(resample_times),
                  cpu_one_core_percent=100*cpu_seconds/duration, ram_peak_bytes=max_ram,
                  output_peak=float(np.max(np.abs(converted))), finite=bool(np.isfinite(converted).all()),
                  offline_alignment_crop_ms=delay/rate*1000,
                  output_quality='Human listening required; objective metrics do not establish female voice quality')
    (args.output_dir/(name+'.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    backend.unload()


if __name__ == '__main__':
    main()
