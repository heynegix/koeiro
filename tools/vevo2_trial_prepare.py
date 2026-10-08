"""Reuse Japanese ASR and preserve the source and liked voice as comparison."""
import json, math, shutil, sys
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.asr.backend import ReazonBackend
OUT=ROOT/'recordings/v011_vevo2_style_trial'
OUT.mkdir(parents=True,exist_ok=True)
if '--verify-output' in sys.argv:
    asr=ReazonBackend(ROOT/'models/asr-reazon',threads=2);asr.load()
    data,sr=sf.read(OUT/'vevo2_style_raw.wav',dtype='float32')
    g=math.gcd(sr,16000);audio=resample_poly(data.reshape(-1),16000//g,sr//g).astype(np.float32)
    stream=asr.recognizer.create_stream();stream.accept_waveform(16000,audio)
    asr.recognizer.decode_stream(stream)
    result={'output_asr_text':stream.result.text.strip(),
        'expected_text':json.loads((OUT/'inputs.json').read_text('utf-8'))['target_text'],
        'note':'Automatic ASR check only; human listening has not been performed.'}
    (OUT/'output_asr.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False));sys.exit(0)
if not (OUT/'inputs.json').exists():
    sources={'source':ROOT/'recordings/v011_mega_tournament/source/source_normal.wav',
             'timbre':ROOT/'validation/v011/meanvc2_adopted720_normal.wav',
             'style':ROOT/'recordings/v011_mega_tournament/reference/reference_05s.wav'}
    asr=ReazonBackend(ROOT/'models/asr-reazon',threads=2);asr.load()
    texts={}
    for key,original in sources.items():
        dest=OUT/(key+'.wav');shutil.copy2(original,dest)
        data,sr=sf.read(dest,dtype='float32');data=data.mean(axis=1) if data.ndim>1 else data
        if key=='timbre':continue
        g=math.gcd(sr,16000);audio=resample_poly(data,16000//g,sr//g).astype(np.float32)
        # This standalone offline trial transcribes the complete short WAV.
        stream=asr.recognizer.create_stream();stream.accept_waveform(16000,audio)
        asr.recognizer.decode_stream(stream);texts[key]=stream.result.text.strip()
        if not texts[key]:raise RuntimeError('Empty transcript: '+key)
    inputs={k:str(OUT/(k+'.wav')) for k in sources}
    inputs.update(target_text=texts['source'],style_text=texts['style'],
        original_files={k:str(p) for k,p in sources.items()},
        task='style-converted speech VC; latency ignored; fixed current-voice timbre reference',
        transcript_method='ReazonSpeech INT8 complete short WAV, unverified ASR transcript')
    (OUT/'inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(inputs,ensure_ascii=False,indent=2))
