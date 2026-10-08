"""C1 offline whole-utterance MeanVC2 inference with a selectable step count.

Mirrors tools/vc_tournament/infer.py meanvc2(): same 120ms preset, same fixed
speaker embedding cache key, seed 110, same official vc_inference(). The only
difference is the step count (2 or 3) passed on the command line, so the
offline-vs-streaming comparison is not confounded by step count. Offline only;
no audio devices; runtime.json and the app are untouched.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
SOURCE,REFERENCE,OUTPUT,META,STEPS=sys.argv[1:6]
STEPS=int(STEPS)
assert STEPS in (2,3), STEPS
SOURCE=Path(SOURCE);REFERENCE=Path(REFERENCE);OUTPUT=Path(OUTPUT);META=Path(META)
REPO=ROOT/'vc_models/meanvc2/repo'
BASE=REPO.parent
os.environ['HF_HUB_CACHE']=str(ROOT/'vc_models/cache/hf')
sys.path[:]=[p for p in sys.path if p and Path(p).resolve()!=ROOT]
sys.path.insert(0,str(REPO))
metrics={'stage':'imports','device':'cpu','steps':STEPS}

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def mark(stage):
    metrics['stage']=stage
    tmp=META.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    tmp.replace(META)
    print('[stage] '+stage,flush=True)

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def write(x,sr):
    import soundfile as sf
    if hasattr(x,'detach'):x=x.detach().cpu().float().squeeze().numpy()
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    sf.write(OUTPUT,x,sr,subtype='FLOAT')
    metrics['output']=str(OUTPUT)

def main():
    import numpy as np
    import torch
    # Windows libtorch's filename fopen can reject Japanese workspace paths.
    native_jit_load=torch.jit.load
    def unicode_jit_load(value,*args,**kwargs):
        if isinstance(value,(str,os.PathLike)):
            with open(value,'rb') as stream:return native_jit_load(stream,*args,**kwargs)
        return native_jit_load(value,*args,**kwargs)
    torch.jit.load=unicode_jit_load
    code=REPO/'src/infer/infer_e2e.py'
    sys.path.insert(0,str(code.parent))
    import types
    package=types.ModuleType('src.model')
    package.__path__=[str(REPO/'src/model')]
    sys.modules['src.model']=package
    e=module(code,'meanvc2_e2e')
    preset=e.MODEL_PRESETS['120ms']
    mark('model_load');t=time.perf_counter()
    asr=e.load_asr_model(preset['asr_ckpt'],'cpu')
    with open(preset['vc_config']) as f:config=json.load(f)
    vc=e.DiT(**config['model']).to('cpu')
    vc=e.load_checkpoint(vc,preset['vc_ckpt'],device='cpu',use_ema=True).float().eval()
    vocos=torch.jit.load(str(REPO/'ckpts/vocos/vocos.pt'),map_location='cpu').eval()
    metrics['model_load_seconds']=time.perf_counter()-t
    mark('reference_embedding');t=time.perf_counter()
    key=sha(REFERENCE)+sha(REPO/'preprocess/ckpts/wavlm_large_finetune.pth')+sha(REPO/'preprocess/ckpts/wavlm_large.pt')+sha(REPO/'preprocess/models/ecapa_tdnn.py')+sha(BASE/'requirements.lock.txt')
    cache=BASE/'reference_embeddings';cache.mkdir(exist_ok=True)
    embfile=cache/(hashlib.sha256(key.encode()).hexdigest()+'.npy')
    metrics['fixed_embedding_cache_hit']=embfile.exists()
    if embfile.exists():spk=np.load(embfile,allow_pickle=False)
    else:
        encoder=e.load_spk_model('cpu');spk=e.extract_spk_emb(str(REFERENCE),encoder,'cpu')
        np.save(embfile,spk);del encoder
    metrics['reference_seconds']=time.perf_counter()-t
    mark('generation');t=time.perf_counter()
    torch.manual_seed(110)
    bn=e.extract_bn(str(SOURCE),asr,window=preset['bn_window'],stride=preset['bn_stride'],offset_init=preset['offset_init'],offset_step=preset['offset_step'],req_cache=preset['req_cache'])
    _,audio=e.vc_inference(vc,vocos,bn,spk,preset['chunk_size'],preset['block_size'],STEPS,'cpu')
    metrics['generation_seconds']=time.perf_counter()-t
    write(audio,16000)
    samples=int(np.asarray(audio).reshape(-1).shape[0])
    metrics['input_sha256']=sha(SOURCE)
    metrics['reference_sha256']=sha(REFERENCE)
    metrics['output_seconds']=samples/16000
    metrics['rtf']=metrics['generation_seconds']/max(metrics['output_seconds'],1e-9)
    mark('complete')

if __name__=='__main__':main()
