"""Download only public inference artifacts. Never invokes training scripts."""
import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path
import requests
from filelock import FileLock
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT/'vc_models/cache'

def disk_budget(required=0):
    free=shutil.disk_usage(CACHE.resolve()).free
    if free < required + 2.5*1024**3:
        raise RuntimeError(f'DISK_LIMIT: required {required/1024**3:.2f} GiB + 2.5 GiB reserve; available {free/1024**3:.2f} GiB')

def url_download(url, dest):
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(dest)+'.lock', timeout=1200):
        return _url_download(url,dest)

def _url_download(url,dest):
    if dest.exists() and dest.stat().st_size:
        return dest
    temp = dest.with_suffix(dest.suffix+'.partial')
    offset=temp.stat().st_size if temp.exists() else 0
    headers={'Range':f'bytes={offset}-'} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 60)) as r:
        r.raise_for_status()
        if 'text/html' in r.headers.get('Content-Type','').lower():
            raise RuntimeError('Download returned HTML instead of an inference artifact')
        disk_budget(int(r.headers.get('Content-Length','0')))
        with temp.open('ab' if offset and r.status_code==206 else 'wb') as f:
            for block in r.iter_content(1024*1024):
                disk_budget()
                f.write(block)
    temp.replace(dest)
    return dest

def hf(repo, filename, dest,repo_type=None):
    probe=hf_hub_download(repo_id=repo,filename=filename,repo_type=repo_type,cache_dir=str(CACHE/'hf'),dry_run=True)
    disk_budget(probe.file_size if probe.will_download else 0)
    path = hf_hub_download(repo_id=repo, filename=filename,repo_type=repo_type, cache_dir=str(CACHE/'hf'))
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        try:
            os.link(path, dest)
        except OSError:
            shutil.copy2(path, dest)
    return dest

def shared_file(source,dest):
    source=Path(source);dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():return dest
    try:os.link(source,dest)
    except OSError:shutil.copy2(source,dest)
    return dest

def snapshot(repo, dest, patterns):
    probe=snapshot_download(repo_id=repo,local_dir=str(dest),allow_patterns=patterns,dry_run=True,max_workers=2)
    disk_budget(sum(f.file_size for f in probe if f.will_download))
    return snapshot_download(repo_id=repo, local_dir=str(dest), allow_patterns=patterns,
                             max_workers=2)

def unzip(path, dest):
    dest = Path(dest).resolve()
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            target = (dest/name).resolve()
            if not target.is_relative_to(dest):
                raise ValueError('Unsafe archive path')
        z.extractall(dest)

def main(family):
    base = ROOT/'vc_models'/family; repo = base/'repo'
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = []
    def job(name, action):
        jobs.append(dict(name=name,status='IN_PROGRESS'))
        print('Artifact stage: '+name,flush=True)
        (base/'download.json').write_text(json.dumps(jobs,indent=2,ensure_ascii=False),encoding='utf-8')
        try:
            value = action()
            jobs[-1]=dict(name=name, status='SUCCESS', path=str(value))
        except Exception as e:
            jobs[-1]=dict(name=name, status='FAILED', error=f'{type(e).__name__}: {e}')
        print(json.dumps(jobs[-1], ensure_ascii=False), flush=True)
        (base/'download.json').write_text(json.dumps(jobs,indent=2,ensure_ascii=False),encoding='utf-8')
    if family == 'meanvc2':
        for name, dest in [('fastu2pp_80ms.pt','preprocess/ckpts/fastu2pp_80ms.pt'),
                           ('fastu2pp_160ms.pt','preprocess/ckpts/fastu2pp_160ms.pt'),
                           ('meanvc2_40ms_40ms.safetensors','ckpts/pretrained_models/meanvc2_40ms_40ms.safetensors'),
                           ('meanvc2_120ms_40ms.safetensors','ckpts/pretrained_models/meanvc2_120ms_40ms.safetensors'),
                           ('vocos.pt','ckpts/vocos/vocos.pt')]:
            job(name, lambda n=name,d=dest: hf('ASLP-lab/MeanVC2', n, repo/d))
    if family == 'meanvc':
        for name in ('model_200ms.safetensors','meanvc_200ms.pt','fastu2++.pt','vocos.pt'):
            job(name, lambda n=name: hf('ASLP-lab/MeanVC', n, repo/'src/ckpt'/n))
    if family in ('meanvc','meanvc2','knnvc','freevc'):
        wavlm = CACHE/'torch/hub/checkpoints/WavLM-Large.pt'
        # Microsoft's historical release URL currently returns 404. kNN-VC authors
        # publish the original frozen Microsoft checkpoint as their official mirror.
        job('WavLM-Large', lambda: url_download('https://github.com/bshall/knn-vc/releases/download/v0.1/WavLM-Large.pt',wavlm))
        if family in ('meanvc','meanvc2'):
            sp = repo/('preprocess/ckpts' if family == 'meanvc2' else 'src/runtime/speaker_verification/ckpt')
            def speaker():
                import gdown
                dest = sp/'wavlm_large_finetune.pth'; sp.mkdir(parents=True,exist_ok=True)
                shared=CACHE/'speaker/wavlm_large_finetune.pth';shared.parent.mkdir(parents=True,exist_ok=True)
                with FileLock(str(shared)+'.lock',timeout=1200):
                    if not shared.exists() and dest.exists():
                        try:os.link(dest,shared)
                        except OSError:shutil.copy2(dest,shared)
                    if not shared.exists():
                        temp=shared.with_suffix('.partial');disk_budget(1400*1024**2)
                        value=gdown.download(id='1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP',output=str(temp),quiet=False,resume=True,use_cookies=False)
                        if not value:raise RuntimeError('Official ECAPA Google Drive download unavailable')
                        temp.replace(shared)
                    if not dest.exists():
                        try:os.link(shared,dest)
                        except OSError:shutil.copy2(shared,dest)
                if wavlm.exists() and not (sp/'wavlm_large.pt').exists():
                    try: os.link(wavlm, sp/'wavlm_large.pt')
                    except OSError: shutil.copy2(wavlm, sp/'wavlm_large.pt')
                return dest
            job('ECAPA speaker encoder', speaker)
    if family == 'knnvc':
        job('prematched HiFiGAN', lambda: url_download('https://github.com/bshall/knn-vc/releases/download/v0.1/prematch_g_02500000.pt',CACHE/'torch/hub/checkpoints/prematch_g_02500000.pt'))
    if family == 'seedvc':
        for n in ('DiT_uvit_tat_xlsr_ema.pth','config_dit_mel_seed_uvit_xlsr_tiny.yml',
                  'DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth','config_dit_mel_seed_uvit_whisper_small_wavenet.yml'):
            job(n, lambda n=n: hf('Plachta/Seed-VC',n,repo/'checkpoints'/n))
        job('CAMPPlus',lambda: hf('funasr/campplus','campplus_cn_common.bin',repo/'checkpoints/campplus_cn_common.bin'))
        job('HiFT',lambda: hf('FunAudioLLM/CosyVoice-300M','hift.pt',repo/'checkpoints/hift.pt'))
        for n in ('facebook/wav2vec2-xls-r-300m','openai/whisper-small','nvidia/bigvgan_v2_22khz_80band_256x'):
            patterns=['config.json','bigvgan_generator.pt'] if n.startswith('nvidia/') else ['*.json','*.bin','*.safetensors','*.model']
            job(n,lambda n=n,patterns=patterns: snapshot_download(n,cache_dir=str(CACHE/'hf'),allow_patterns=patterns,max_workers=2))
    if family == 'amphion':
        job('HuBERT-Large shared encoder',lambda:url_download('https://download.pytorch.org/torchaudio/models/hubert_fairseq_large_ll60k.pth',CACHE/'torch/hub/checkpoints/hubert_fairseq_large_ll60k.pth'))
        job('Vevo-Timbre',lambda: snapshot('amphion/Vevo',repo/'ckpts/Vevo-local',['tokenizer/vq8192/*','acoustic_modeling/Vq8192ToMels/*','acoustic_modeling/Vocoder/*']))
        for n in ('ns3_facodec_encoder_v2.bin','ns3_facodec_decoder_v2.bin'):
            job(n,lambda n=n: hf('amphion/naturalspeech3_facodec',n,repo/'ckpts/facodec'/n))
    if family in ('amphion','conan'):
        def drive_folder():
            import gdown
            fid = '1NPzSIuSKO8o87g5ImNzpw_BgbhsZaxNg' if family=='amphion' else '1QhnECo2L4xfXDgdrnM6L1xpsH7u3iRvj'
            dest = base/('noro-checkpoints' if family=='amphion' else 'checkpoints')
            files=gdown.download_folder(id=fid,output=str(dest),quiet=True,skip_download=True,timeout=(30,60),use_cookies=False)
            needed=[f for f in files if not {'codes','.git','logs','lightning_logs','tensorboard','tb'}.intersection(Path(f.path).parts)
                    and not any(word in Path(f.path).name.lower() for word in ('optimizer','scheduler','discriminator'))
                    and Path(f.path).suffix.lower() in ('.ckpt','.pt','.pth','.bin','.safetensors','.npy','.json','.yaml','.yml')]
            if not needed:raise RuntimeError('Official checkpoint folder has no inference artifacts')
            for item in needed:
                local=Path(item.local_path).resolve()
                if not local.is_relative_to(dest.resolve()):raise ValueError('Unsafe Google Drive file path')
                local.parent.mkdir(parents=True,exist_ok=True)
                print('Inference checkpoint: '+str(local),flush=True)
                if local.exists() and local.stat().st_size:
                    continue
                disk_budget()
                try:
                    url_download('https://drive.usercontent.google.com/download?id='+item.id+'&export=download&confirm=t',local)
                except Exception as direct_error:
                    print('Direct public download failed: '+str(direct_error),flush=True)
                    if not gdown.download(id=item.id,output=str(local),quiet=False,resume=True,timeout=(30,60),use_cookies=False):
                        raise RuntimeError('Checkpoint unavailable: '+str(item.path))
            return dest
        job('Noro' if family=='amphion' else 'Conan + Fast + Emformer + HiFiGAN',drive_folder)
        if family=='conan':
            def fast_checkpoint():
                files=list((base/'checkpoints/Conan_fast').glob('model_ckpt_steps_*.ckpt'))
                if not files:raise RuntimeError('Official Conan_fast folder exposes config/logs but no main VC checkpoint. Fast Emformer alone is insufficient; normal Conan is not substituted.')
                return files[0]
            job('Conan Fast main VC checkpoint',fast_checkpoint)
    if family == 'xvc':
        job('X-VC',lambda: snapshot('chenxie95/X-VC',repo/'ckpts',['*.pt','*.yaml','*.json']))
        job('GLM voice tokenizer',lambda: snapshot('zai-org/glm-4-voice-tokenizer',repo/'pretrained/glm',['*.json','*.bin','*.safetensors','*.py']))
        job('ERes2Net',lambda: url_download('https://modelscope.cn/api/v1/models/iic/speech_eres2net_sv_en_voxceleb_16k/repo?Revision=master&FilePath=pretrained_eres2net.ckpt',repo/'pretrained/eres2net/pretrained_eres2net.ckpt'))
        job('ERes2Net configuration',lambda: url_download('https://modelscope.cn/api/v1/models/iic/speech_eres2net_sv_en_voxceleb_16k/repo?Revision=master&FilePath=configuration.json',repo/'pretrained/eres2net/configuration.json'))
    if family == 'freevc':
        for filename in ('checkpoints/freevc.pth','speaker_encoder/ckpt/pretrained_bak_5805000.pt'):
            job('Official FreeVC '+filename,lambda filename=filename: hf('OlaWod/FreeVC',filename,repo/filename,repo_type='space'))
        job('WavLM shared link',lambda: shared_file(CACHE/'torch/hub/checkpoints/WavLM-Large.pt',repo/'wavlm/WavLM-Large.pt'))
    if family == 'fragmentvc':
        def release():
            r = requests.get('https://api.github.com/repos/yistLin/FragmentVC/releases',timeout=30);r.raise_for_status()
            paths=[]
            for rel in r.json():
                for a in rel['assets']:
                    if a['name'].endswith(('.pt','.pth','.zip')):
                        p=url_download(a['browser_download_url'],base/'checkpoints'/a['name']);paths.append(p)
                        if p.suffix=='.zip': unzip(p,base/'checkpoints')
            if not paths: raise RuntimeError('No public release checkpoint assets found')
            return paths
        job('FragmentVC + universal vocoder release',release)
        job('wav2vec2 Base',lambda: url_download('https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt',base/'checkpoints/wav2vec_small.pt'))
    if family == 'openvoice':
        for filename in ('config.json','checkpoint.pth'):
            job('OpenVoice V2 converter '+filename,lambda filename=filename: hf('myshell-ai/OpenVoiceV2','converter/'+filename,base/'checkpoints_v2/converter'/filename))
    if family == 'ezvc':
        job('EZ-VC inference weights',lambda: snapshot('SPRINGLab/EZ-VC',base/'checkpoints',['*.pt','*.safetensors','*.json','*.yaml','vocab.txt']))
    if family == 'audiocpp':
        def binary():
            r=requests.get('https://api.github.com/repos/0xShug0/audio.cpp/releases/latest',timeout=30);r.raise_for_status()
            (base/'release.json').write_text(json.dumps(r.json(),indent=2),encoding='utf-8')
            assets=[a for a in r.json()['assets'] if re.search(r'win.*(cpu|portable).*\.zip$',a['name'],re.I)]
            if not assets: raise RuntimeError('Windows CPU release asset not found')
            a=sorted(assets,key=lambda a:'portable' not in a['name'])[0]
            z=url_download(a['browser_download_url'],base/a['name']);unzip(z,base/'bin');return base/'bin'
        job('Windows standalone CPU',binary)
        def gguf(quantized=False):
            from huggingface_hub import HfApi
            files=HfApi().list_repo_files('audio-cpp/audio.cpp-gguf')
            matches=[f for f in files if 'meanvc2' in f.lower() and '120' in f and f.endswith('.gguf') and ('q4_k' in f if quantized else ('f32' in f or 'fp32' in f))]
            if not matches: raise RuntimeError('Official MeanVC2 fp32 GGUF not found')
            return hf('audio-cpp/audio.cpp-gguf',matches[0],base/'models'/Path(matches[0]).name)
        job('MeanVC2 120ms Q4_K self-contained GGUF',lambda:gguf(True))
        job('MeanVC2 120ms FP32 self-contained GGUF',gguf)
    (base/'download.json').write_text(json.dumps(jobs,indent=2,ensure_ascii=False),encoding='utf-8')
    return 1 if any(j['status']=='FAILED' for j in jobs) else 0

if __name__=='__main__':
    def watchdog():
        while True:
            try:disk_budget()
            except RuntimeError as e:
                print(str(e),flush=True);os._exit(75)
            time.sleep(.5)
    threading.Thread(target=watchdog,daemon=True).start()
    sys.exit(main(sys.argv[1]))
