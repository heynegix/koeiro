"""Curated inference-only environments; source revision and dependency lock recorded."""
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from .process import run

COMMON=['numpy<2','soundfile','scipy','librosa>=0.10','einops','huggingface-hub<1','PyYAML','tqdm','setuptools','packaging']
EXTRAS={
 'meanvc2':['accelerate==1.7.0','ema-pytorch==0.7.7','x-transformers==2.2.11','torchdiffeq','prefigure','omegaconf','s3prl','matplotlib'],
 'meanvc':['x-transformers','omegaconf','pyaudio','accelerate','ema-pytorch','prefigure','funasr','s3prl'],
 'conan':['torchdyn==1.0.6','pandas','pyloudnorm','praat-parselmouth','g2p_en','tensorboard','scikit-learn','textgrid','webrtcvad-wheels','h5py','chardet','scikit-image'],
 'seedvc':['transformers==4.44.2','accelerate','munch','hydra-core','omegaconf','einops','descript-audio-codec','descript-audiotools','pydub','jiwer','matplotlib'],
 'xvc':['transformers==4.44.1','x-transformers==1.40.2','hydra-core','julius','lightning==2.2.4','ema-pytorch','torchmetrics','descript-audiotools','modelscope','einx','wandb'],
 'amphion':['transformers==4.41.2','accelerate==0.24.1','encodec','json5','ruamel.yaml','safetensors','g2p_en','unidecode','easydict','omegaconf','matplotlib','openai-whisper','pyworld','ipython'],
 'knnvc':[],
 'freevc':['tensorboard','webrtcvad-wheels','resampy'],
 'fragmentvc':['fairseq @ git+https://github.com/pytorch/fairseq.git@1a709b2a401ac8bd6d805c8a6a5f4d7f03b923ff','editdistance','jsonargparse==2.32.2','sox','matplotlib'],
 'ezvc':['espnet @ git+https://github.com/wanchichen/espnet.git@ssl'],
 'openvoice':['eng_to_ipa','inflect','unidecode','pypinyin','cn2an','jieba','langid'],
}

def prepare(model,root,out,timeout, retry=False):
    base=root/'vc_models'/model.family;repo=base/'repo';base.mkdir(parents=True,exist_ok=True)
    statusfile=base/'setup.json'
    spec=dict(repo=model.repo,python=model.python,packages=COMMON+EXTRAS.get(model.family,[]),torch='1.6.0+cpu' if model.family=='fragmentvc' else '2.5.1 CPU')
    signature=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    previous=json.loads(statusfile.read_text('utf-8')) if statusfile.exists() else {}
    operations=[]
    def op(stage,argv,cwd=base):
        r=run(argv,cwd,out/'logs'/f'{model.family}_{stage}_{time.time_ns()}.log',timeout=timeout)
        r['stage']=stage;operations.append(r)
        statusfile.write_text(json.dumps(dict(signature=signature,spec=spec,revision=revision,operations=operations,status='IN_PROGRESS'),indent=2,ensure_ascii=False),encoding='utf-8')
        return r['status']=='SUCCESS'
    revision=None
    if not (repo/'.git').exists():
        op('clone',['git','clone','--depth','1','https://github.com/'+model.repo+'.git',str(repo)])
    if (repo/'.git').exists():
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    valid_previous=(previous.get('signature')==signature and previous.get('revision')==revision)
    if valid_previous and previous.get('status')=='FAILED' and not retry:
        # Do not repeatedly rebuild known-broken environments or revisit gated
        # downloads on the default one-command resume. Explicit retry or a
        # changed setup specification/revision starts a new attempt.
        return dict(previous,cache_hit=True)
    if model.family!='audiocpp':
        python=base/'.venv/Scripts/python.exe'
        if not python.exists(): op('venv',['uv','venv','--managed-python','--python',model.python,str(base/'.venv')])
        deps_previous=next((s for s in previous.get('operations',[]) if s['stage']=='dependencies'),{})
        if python.exists() and (not valid_previous or deps_previous.get('status')!='SUCCESS'):
            torch_version='1.6.0' if model.family=='fragmentvc' else '2.5.1'
            torch_packages=['torch=='+torch_version+('+cpu' if model.family=='fragmentvc' else '')]+([] if model.family=='fragmentvc' else ['torchaudio==2.5.1','torchvision==0.20.1'])
            torch_options=['--extra-index-url','https://pypi.org/simple','--index-strategy','unsafe-best-match'] if model.family=='fragmentvc' else []
            torch_ok=op('torch',['uv','pip','install','--python',str(python),*torch_packages,'--index-url','https://download.pytorch.org/whl/cpu',*torch_options])
            constraints=base/'torch-constraints.txt'
            constraints.write_text('torch=='+torch_version+'\n'+('' if model.family=='fragmentvc' else 'torchaudio==2.5.1\ntorchvision==0.20.1\n'),encoding='utf-8')
            if torch_ok:
                op('dependencies',['uv','pip','install','--python',str(python),'-c',str(constraints),*spec['packages']])
            else:
                operations.append(dict(stage='dependencies',status='FAILED',reason='CPU_TORCH_INSTALL_FAILED',last_error='Dependency installation skipped to avoid unintended non-CPU torch download.'))
            if model.family=='ezvc':
                op('submodules',['git','submodule','update','--init','--recursive'],repo)
                op('package',['uv','pip','install','--python',str(python),'-e',str(repo)])
        elif valid_previous: operations+=previous.get('operations',[])
        if python.exists():
            lock=subprocess.run(['uv','pip','freeze','--python',str(python)],capture_output=True,text=True)
            (base/'requirements.lock.txt').write_text(lock.stdout,encoding='utf-8')
    # Checkpoints attempted even if dependency install fails; download uses runner env.
    download_previous=next((s for s in previous.get('operations',[]) if s['stage']=='download'),{})
    if not valid_previous or download_previous.get('status')!='SUCCESS' or retry:
        op('download',[str(root/'vc_models/runner/.venv/Scripts/python.exe'),str(root/'tools/vc_tournament/download.py'),model.family],root)
    elif not any(s['stage']=='download' for s in operations): operations.append(download_previous)
    latest={s['stage']:s for s in operations}
    value=dict(signature=signature,revision=revision,spec=spec,operations=list(latest.values()),
               status='FAILED' if any(s['status']=='FAILED' for s in latest.values()) else 'SUCCESS')
    records=base/'download.json'
    if records.exists():value['download_records']=json.loads(records.read_text('utf-8'))
    value['history']=previous.get('history',[])+([{k:v for k,v in previous.items() if k!='history'}] if previous.get('operations') else [])
    statusfile.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    return value
