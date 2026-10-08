"""Inference-only Vevo2 assets, pinned and resumable; everything stays on D:."""
import hashlib, json, os, shutil, urllib.request, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'vc_models/vevo2'
REV = '2674843cbaa50aa89ee7ccaf5bb15d6ccf46c6c8'
def fetch(url, target, expected=None, sha=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (expected is None or target.stat().st_size == expected):
        if sha is None or digest(target) == sha: return
    part = target.with_suffix(target.suffix+'.part')
    for attempt in range(80):
        offset = part.stat().st_size if part.exists() else 0
        if expected and offset == expected: break
        # Avoid a cached redirect signed for a different byte range.
        request_url=url+('?' if '?' not in url else '&')+f'resume={offset}'
        # Bounded ranges avoid long-lived CDN streams being cut midway.
        end=min(offset+64*2**20-1,expected-1) if expected else None
        req = urllib.request.Request(request_url, headers={'Range':f'bytes={offset}-{end if end is not None else ""}'} if offset or expected else {})
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                if expected is None:
                    content_range=response.headers.get('Content-Range','')
                    if '/' in content_range:expected=int(content_range.rsplit('/',1)[1])
                    elif response.headers.get('Content-Length'):expected=int(response.headers['Content-Length'])
                mode = 'ab' if offset and response.status == 206 else 'wb'
                with part.open(mode) as stream:
                    while chunk := response.read(1024*1024):
                        if shutil.disk_usage(BASE).free < 2.5*2**30: raise RuntimeError('D disk reserve reached')
                        stream.write(chunk)
            if not expected or part.stat().st_size == expected: break
        except (OSError, urllib.error.URLError) as error:
            print('Download retry:',type(error).__name__,flush=True)
        print('Resume bytes:',part.stat().st_size if part.exists() else 0,flush=True)
        time.sleep(2)
    if expected and part.stat().st_size != expected: raise RuntimeError('Incomplete download: '+str(target))
    if sha and digest(part) != sha: raise RuntimeError('SHA mismatch: '+str(target))
    part.replace(target)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8*1024*1024): h.update(chunk)
    return h.hexdigest()
def main():
    manifest = json.loads((BASE/'remote_manifest.json').read_text())
    prefixes=['acoustic_modeling/fm_emilia101k_singnet7k_repa/',
              'contentstyle_modeling/posttrained/', 'tokenizer/', 'vocoder/']
    files=[s for s in manifest['siblings'] if any(s['rfilename'].startswith(p) for p in prefixes)
           and not any(x in s['rfilename'] for x in ['optimizer','rng_state','scheduler','trainer_state','training_args'])
           and s['rfilename'] not in ['vocoder/model_1.safetensors','vocoder/model_2.safetensors']
           and not s['rfilename'].startswith('tokenizer/prosody_')]
    print('Selected bytes:',sum(s['size'] for s in files),flush=True)
    for item in files:
        name=item['rfilename']; target=BASE/'checkpoints'/name; sha=item.get('lfs',{}).get('sha256')
        # Vevo and Vevo2 publish identical vocoder files. Reuse only after SHA verification.
        old=ROOT/'vc_models/amphion/repo/ckpts/Vevo-local/acoustic_modeling/Vocoder'/Path(name).name
        if name.startswith('vocoder/') and sha and old.exists() and digest(old)==sha and not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True); os.link(old,target)
            print('Verified hardlink:',name,flush=True)
        print('Asset:',name,flush=True)
        fetch(f'https://huggingface.co/RMSnow/Vevo2/resolve/{REV}/{name}',target,item['size'],sha)
    # Whisper medium encoder is the model's prescribed feature frontend, not a replaceable ASR option.
    import ast
    tree=ast.parse((ROOT/'vc_models/amphion/.venv/Lib/site-packages/whisper/__init__.py').read_text('utf-8'))
    urls=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_MODELS' for t in n.targets))
    url=urls['medium']; sha=url.split('/')[-2]
    fetch(url,BASE/'whisper/medium.pt',sha=sha)
    (BASE/'assets_ready.json').write_text(json.dumps({'revision':REV,'files':[s['rfilename'] for s in files]},indent=2))
if __name__=='__main__':main()
