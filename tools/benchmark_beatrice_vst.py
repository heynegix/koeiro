"""Optional local-only alternative benchmark via the official, unmodified VST.

Does not link/extract Beatrice's private inference API. No model is downloaded
or redistributed by this tool. Supply a legally usable model configuration.
"""
import argparse
import json
from pathlib import Path
import struct
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import psutil
from tools.compare_presets import read_wav, write_wav


from src.vc.vst_state import set_model_preset
from src.utils.windows import audio_scheduling


def main():
    with audio_scheduling():
        run()


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plugin',type=Path,required=True,help='Windows VST3 binary or bundle')
    parser.add_argument('--model',type=Path,required=True,help='Model TOML used only locally')
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--voice',type=int,default=1)
    parser.add_argument('--pitch',type=float,default=0)
    parser.add_argument('--chunk',type=int,default=624)
    parser.add_argument('--output',type=Path,default=Path('recordings/v03/beatrice/Beatrice.wav'))
    args = parser.parse_args()
    if not args.model.is_file() or not 0<=args.voice<=255 or not -12<=args.pitch<=12 or not 64<=args.chunk<=4096:
        parser.error('Invalid model/voice/pitch')
    from pedalboard import load_plugin
    source, rate = read_wav(args.input)
    if rate!=48000 or not len(source):
        parser.error('48 kHz human input required')
    start=time.perf_counter()
    plugin = load_plugin(str(args.plugin.resolve()))
    plugin.preset_data = set_model_preset(plugin.preset_data,args.model,args.voice,args.pitch)
    load_seconds = time.perf_counter()-start
    start=time.perf_counter()
    warm=(.005*np.sin(np.arange(args.chunk)*.071)).astype(np.float32)
    for _ in range(12):
        plugin.process(warm,rate,buffer_size=args.chunk,reset=False)
    warmup_seconds = time.perf_counter()-start
    plugin.reset()
    audio = np.pad(source,(0,4800))
    output = []
    timings=[]
    cpu=time.process_time(); start=time.perf_counter()
    for i in range(0,len(audio),args.chunk):
        begin=time.perf_counter()
        result=plugin.process(audio[i:i+args.chunk],rate,buffer_size=args.chunk,reset=False)
        timings.append(1000*(time.perf_counter()-begin))
        output.append(result.reshape(-1))
    elapsed=time.perf_counter()-start
    report = {'backend':'Official Beatrice VST3 via Pedalboard', 'model':str(args.model.resolve()),
              'voice':args.voice,'pitch':args.pitch,'input_seconds':len(source)/rate,
              'chunk_frames':args.chunk,'output_rate':rate,'output_frames':len(audio),
              'load_seconds':load_seconds,'warmup_seconds':warmup_seconds,'rtf':elapsed/(len(source)/rate),
              'inference':{'average_ms':float(np.mean(timings)),'p95_ms':float(np.percentile(timings,95)),
                           'maximum_ms':float(np.max(timings))},
              'cpu_one_core_percent':100*(time.process_time()-cpu)/elapsed,
              'ram_bytes':psutil.Process().memory_info().rss,
              'reported_latency_samples':plugin.reported_latency_samples,
              'alignment':'No compensation of unreported model latency; output includes 100 ms tail',
              'warmup':'Nonzero signal / 12 chunks; zero input skips the native network',
              'quality':'Human listening required', 'finite':bool(np.isfinite(np.concatenate(output)).all())}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_wav(args.output,np.concatenate(output),rate)
    args.output.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)


if __name__=='__main__':
    main()
