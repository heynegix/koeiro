"""Offline voice audition. Never modifies installed runtime or live settings."""
import json
import time
from pathlib import Path
import sys
import argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from pedalboard import load_plugin
from tools.compare_presets import read_wav,write_wav
from src.vc.vst_state import set_model_preset
from src.utils.windows import audio_scheduling


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refine',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    folder=root/('recordings/voice_quality/refined' if args.refine else 'recordings/voice_quality')
    folder.mkdir(parents=True,exist_ok=True)
    runtime=json.loads((root/'models/girl_01/runtime.json').read_text())
    plugin_path=root/'models/girl_01'/runtime['plugin']
    jvs=root/'models/girl_01'/runtime['model']
    character_runtime=json.loads((root/'models/character_01/runtime.json').read_text(encoding='utf-8'))
    official=root/'models/character_01'/character_runtime['model']
    candidates=[
        ('01_Current_JVS002',jvs,1,4.,0.),
        ('02_Tsukuyomi_UTAU_P6',official,0,6.,0.),
        ('03_Tsukuyomi_Corpus_P6',official,1,6.,0.),
        ('04_Tsukuyomi_UTAU_P8_F1',official,0,8.,1.),
        ('05_Tsukuyomi_Corpus_P8_F1',official,1,8.,1.),
        ('06_JVS010_P6_F1',jvs,9,6.,1.),
    ]
    if args.refine:
        candidates=[('01_JVS010_P6_F1',jvs,9,6.,1.),
            ('02_Anime_Cute_P7_F05',official,0,7.,.5),
            ('03_Anime_Soft_P7_F0',official,1,7.,0.),
            ('04_UTAU_P7_F0',official,0,7.,0.),
            ('05_JVS010_P7_F05',jvs,9,7.,.5)]
    source,rate=read_wav(root/'recordings/v03/input.wav')
    if rate!=48000: raise ValueError('48 kHz input required')
    source=np.pad(source,(0,4800));chunk=1248
    reports=[];montage=[]
    with audio_scheduling():
        for name,model,voice,pitch,formant in candidates:
            if not model.is_file(): raise ValueError('Missing official model: '+str(model))
            plugin=load_plugin(str(plugin_path.resolve()))
            plugin.preset_data=set_model_preset(plugin.preset_data,model,voice,pitch)
            plugin.formant_shift_st=formant
            warm=(.005*np.sin(np.arange(chunk)*.071)).astype(np.float32)
            for _ in range(12): plugin.process(warm,rate,buffer_size=chunk,reset=False)
            plugin.reset();result=[];timings=[];begin=time.perf_counter()
            for i in range(0,len(source),chunk):
                frame=np.zeros(chunk,dtype=np.float32)
                n=min(chunk,len(source)-i);frame[:n]=source[i:i+n]
                t=time.perf_counter()
                y=plugin.process(frame,rate,buffer_size=chunk,reset=False).reshape(-1)
                timings.append((time.perf_counter()-t)*1000)
                if len(y)!=chunk or not np.isfinite(y).all(): raise ValueError('Invalid native audio')
                result.append(y[:n].copy())
            audio=np.concatenate(result).astype(np.float32)
            elapsed=time.perf_counter()-begin
            np.clip(audio,-.98,.98,out=audio)
            write_wav(folder/(name+'.wav'),audio,rate)
            reports.append(dict(name=name,model=str(model),voice=voice,pitch=pitch,formant=formant,
                avg_ms=float(np.mean(timings)),p95_ms=float(np.percentile(timings,95)),
                p99_ms=float(np.percentile(timings,99)),max_ms=max(timings),rtf=elapsed/(len(source)/rate),
                finite=True,frames=len(audio),reported_latency_samples=plugin.reported_latency_samples))
            # Shared scalar for the whole utterance; no pumping or individual
            # peak normalization concealing voice dynamics.
            rms=float(np.sqrt(np.mean(audio.astype(np.float64)**2)))
            factor=min(.98/max(float(np.max(np.abs(audio))),1e-8),.08/max(rms,1e-8))
            montage.extend((audio*factor,np.zeros(rate,dtype=np.float32)))
            print(name,round(reports[-1]['rtf'],3),flush=True)
    write_wav(folder/'All_Candidates.wav',np.concatenate(montage),rate)
    (folder/'comparison.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
    (folder/'README.txt').write_text('\n'.join(f'{name}: voice={voice}, pitch={pitch}, formant={formant}' for name,_,voice,pitch,formant in candidates)+'\nSame human input; no prosody controller. Native model change only.\nCharacter model terms: https://prj-beatrice.com/2.0.0-rc.0-official-model-1-terms\n',encoding='utf-8')


if __name__=='__main__':main()
