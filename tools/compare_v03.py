"""Four same-human-input comparisons; AI model delay is not auto-compensated."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.compare_presets import read_wav, write_wav, render
from src.presets.female_presets import load_preset
from src.vc.llvc_onnx import LLVCOnnxBackend
from src.vc.beatrice_vst import BeatriceVSTBackend
from src.vc.resampler import StreamingResampler
from src.vc.post_fx import LightPostFX


def convert(source, gain_db=0):
    folder=Path(__file__).resolve().parents[1]/'models/girl_01'
    runtime=json.loads((folder/'runtime.json').read_text(encoding='utf-8'))
    if runtime['backend']=='beatrice_vst':
        backend=BeatriceVSTBackend()
        delay=0  # unknown native delay is retained, not guessed away
        down=up=None
    elif runtime['backend']=='llvc':
        backend=LLVCOnnxBackend(1,1)
        folder=folder.parent/'research_llvc'
        delay=144
        down,up=StreamingResampler(48000,16000),StreamingResampler(16000,48000)
    else:
        raise ValueError('Unsupported comparison backend')
    backend.load(folder)
    backend.warmup()
    n=624
    amplified = np.clip(source*10**(gain_db/20), -1,1).astype(np.float32)
    padded = np.pad(amplified,(0,delay+(-len(amplified)-delay)%n))
    output = np.empty_like(padded)
    started = time.perf_counter()
    for offset in range(0,len(padded),n):
        chunk=padded[offset:offset+n]
        result=backend.process_chunk(down.process(chunk) if down else chunk)
        output[offset:offset+n] = up.process(result) if up else result
    duration = time.perf_counter()-started
    model=backend.get_stats()
    backend.unload()
    result = output[delay:delay+len(source)]
    return result, {'gain_db':gain_db,'rtf':duration/(len(source)/48000),'model':model,
                    'alignment_crop_ms':delay/48,
                    'input_clipped_samples':int(np.count_nonzero(abs(source)*10**(gain_db/20)>1))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=Path('recordings/v03/beatrice_comparison'))
    parser.add_argument('--input-gain-experiment',action='store_true',help='Also compare +6 dB model input (no default change)')
    args = parser.parse_args()
    source, rate = read_wav(args.input)
    if rate != 48000 or len(source) == 0:
        parser.error('A non-empty 48 kHz human PCM16 WAV is required')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    # Equal-length 100 ms tail on every file preserves delayed final phonemes.
    padded=np.pad(source,(0,4800))
    dsp, delay, _ = render(padded,rate,load_preset('Anime Test'))
    converted, stats = convert(padded)
    fx = LightPostFX()
    with_fx = np.empty_like(converted)
    for offset in range(0,len(converted),624):
        with_fx[offset:offset+624] = fx.process(converted[offset:offset+624],brightness=50)
    samples = {'Original':padded,'Anime_Test':dsp,'AI_Voice':converted,'AI_Voice_PostFX':with_fx}
    report = {'source':str(args.input.resolve()),'sample_rate':rate,'frames':len(source),
              'seconds':len(source)/rate,'tail_ms':100,'output_frames':len(padded),
              'dsp_alignment_ms':delay/rate*1000,'ai_alignment_crop_ms':stats['alignment_crop_ms'],
              'alignment':'DSP delay compensated offline; native AI model delay unmeasured and retained',
              'ai':stats,'post_fx_brightness':50,'human_quality':'Not automatically evaluated', 'files':{}}
    if args.input_gain_experiment:
        samples['AI_Voice_InputGain6dB'], report['gain_experiment'] = convert(padded,6)
    for name,audio in samples.items():
        path = args.output_dir/(name+'.wav')
        write_wav(path,audio,rate)
        report['files'][name] = {'path':str(path.resolve()),'peak':float(np.max(abs(audio))),
                               'rms_dbfs':float(20*np.log10(max(1e-12,np.sqrt(np.mean(audio*audio))))),
                               'finite':bool(np.isfinite(audio).all())}
        print(path,flush=True)
    (args.output_dir/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
