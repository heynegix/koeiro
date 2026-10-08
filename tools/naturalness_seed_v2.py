"""Official Seed-VC V2 offline ceiling trial, CPU FP32, no app integration."""
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'recordings/v011_naturalness'


def run():
    import soundfile as sf
    source=OUT/'source/normal.wav';original_target=ROOT/'models/meanvc2_ref60/reference.wav'
    reference,reference_rate=sf.read(original_target,dtype='float32',frames=sf.info(original_target).samplerate*5)
    target=OUT/'reference/seed_v2_reference05.wav'
    sf.write(target,reference,reference_rate,subtype='PCM_16')
    from src.vc.voice_library import digest
    result_path=OUT/'metadata/seed_v2_result.json'
    repo=ROOT/'vc_models/seedvc/repo'
    if result_path.exists():
        cached=json.loads(result_path.read_text('utf-8'))
        if (cached.get('source_sha256')==digest(source) and cached.get('reference_sha256')==digest(target)
                and cached.get('official_inference_sha256')==digest(repo/'inference_v2.py')
                and Path(cached['output']).exists() and digest(cached['output'])==cached.get('output_sha256')):
            print('Seed-VC V2 CACHE',flush=True);return
    import torch
    import soundfile as sf
    import numpy as np
    from types import SimpleNamespace
    sys.path.insert(0,str(repo))
    # Official recipe has relative config paths; shared HF caches stay on D:.
    import os
    os.chdir(repo)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(51005)
    spec=importlib.util.spec_from_file_location('natural_seed_v2_official',repo/'inference_v2.py')
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    official.device=torch.device('cpu');official.dtype=torch.float32
    args=SimpleNamespace(ar_checkpoint_path=None,cfm_checkpoint_path=None,compile=False,diffusion_steps=25,
        length_adjust=1.,intelligibility_cfg_rate=.7,similarity_cfg_rate=.7,top_p=.9,temperature=1.,
        repetition_penalty=1.,convert_style=False,anonymization_only=False)
    started=time.perf_counter()
    # Transformers 4.44 and Torch 2.5 otherwise disagree on the pretrained
    # HuBERT positional convolution's weight_g/weight_v key names. Preserve
    # the checkpoint's legacy weight-norm parameterization instead of leaving
    # that convolution randomly initialized. Restore after model creation.
    native_weight_norm=torch.nn.utils.parametrizations.weight_norm
    torch.nn.utils.parametrizations.weight_norm=torch.nn.utils.weight_norm
    try:
        official.vc_wrapper_v2=official.load_v2_models(args)
    finally:
        torch.nn.utils.parametrizations.weight_norm=native_weight_norm
    loaded=time.perf_counter()-started
    started=time.perf_counter()
    rate,converted=official.convert_voice_v2(str(source),str(target),args)
    elapsed=time.perf_counter()-started
    converted=np.asarray(converted,dtype=np.float32).squeeze()
    if converted.ndim!=1 or not np.isfinite(converted).all() or np.max(abs(converted))<.001:raise ValueError('Invalid V2 output')
    dest=OUT/'raw/seed_v2_normal.wav';sf.write(dest,converted,rate,subtype='PCM_16')
    duration=len(sf.read(source)[0])/sf.info(source).samplerate
    row=dict(status='SUCCESS',model='Seed-VC V2',precision='CPU FP32 AR cache/inference; weights unchanged',
        output=str(dest),output_sha256=digest(dest),load_seconds=loaded,generation_seconds=elapsed,
        source_seconds=duration,output_seconds=len(converted)/rate,rtf=elapsed/duration,
        realtime_suitability='UNVERIFIED' if elapsed/duration<=1 else 'OFFLINE ONLY',
        source_sha256=digest(source),reference_sha256=digest(target),official_inference_sha256=digest(repo/'inference_v2.py'),
        reference_original_sha256=digest(original_target),reference_selection='first5s of unchanged current reference; bounded to avoid long-reference attention RAM',
        voice_preservation='UNRATED; same reference does not guarantee same MeanVC2 voice',
        settings=vars(args),kind='OFFLINE WAV; no microphone')
    (OUT/'metadata/seed_v2_result.json').write_text(json.dumps(row,indent=2),'utf-8')
    print(json.dumps(row),flush=True)
