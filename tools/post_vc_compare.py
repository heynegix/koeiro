"""Guarded, resumable offline post-VC comparison. No audio devices or app edits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'recordings/v011_post_vc'
BASE=ROOT/'vc_models/meanvc2/.venv/Scripts/python.exe'
MODELS={
 'lavasr':('https://github.com/ysharma3501/LavaSR.git',['git+https://github.com/langtech-bsc/vocos.git@matcha','encodec']),
 'flashsr':('https://github.com/ysharma3501/FlashSR.git',['onnxruntime==1.20.1','coloredlogs','humanfriendly','flatbuffers']),
 'novasr':('https://github.com/ysharma3501/NovaSR.git',[]),
 'apbwe':('https://github.com/yxlu-0102/AP-BWE.git',['gdown','beautifulsoup4','soupsieve','filelock']),
 'resemble':('https://github.com/resemble-ai/resemble-enhance.git',['omegaconf','hydra-core','antlr4-python3-runtime==4.9.3','rich','markdown-it-py','mdurl','tabulate']),
 'mossformer_sr':(None,['clearvoice','omegaconf','hydra-core','antlr4-python3-runtime==4.9.3','gdown','beautifulsoup4','soupsieve','filelock','rotary-embedding-torch','torchinfo','modelscope','addict','simplejson','yamlargparse','python-speech-features','scenedetect==0.6.6','opencv-python==4.10.0.84','easydict']),
}

def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),'utf-8');tmp.replace(path)

def env_for(model=None):
 env=os.environ.copy();cache=ROOT/'vc_models/cache';temp=cache/'temp/post_vc';temp.mkdir(parents=True,exist_ok=True)
 env.update(TEMP=str(temp),TMP=str(temp),TMPDIR=str(temp),HF_HOME=str(cache/'hf-home'),HF_HUB_CACHE=str(cache/'hf'),TORCH_HOME=str(cache/'torch'),PIP_CACHE_DIR=str(cache/'pip'),UV_CACHE_DIR=str(cache/'uv'),XDG_CACHE_HOME=str(cache/'xdg'),NUMBA_CACHE_DIR=str(cache/'numba'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HF_HUB_DISABLE_XET='1',PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='')
 if model:
  folder=ROOT/'vc_models'/('post_'+model)
  env['PYTHONPATH']=os.pathsep.join(map(str,[folder/'vendor',folder/'repo',ROOT]))
 return env

def guarded(command,log,env,timeout=600):
 import psutil
 log.parent.mkdir(parents=True,exist_ok=True);start=time.monotonic();peak=0;reason=None;cpu=0
 if psutil.virtual_memory().available<1.5*1024**3 or shutil.disk_usage(ROOT).free<3*1024**3:raise RuntimeError('Insufficient RAM/disk before launch')
 if log.exists():
  history=log.parent/'attempt_history';history.mkdir(exist_ok=True);shutil.copy2(log,history/(log.stem+'_'+str(time.time_ns())+'.log'))
 with log.open('w',encoding='utf-8') as stream:
  process=subprocess.Popen([str(c) for c in command],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
  root=psutil.Process(process.pid)
  while process.poll() is None:
   try:
    family=[root]+root.children(recursive=True)
    rss=sum(p.memory_info().rss for p in family if p.is_running());peak=max(peak,rss)
    cpu=max(cpu,sum(p.cpu_times().user+p.cpu_times().system for p in family if p.is_running()))
    if rss>4.5*1024**3 or psutil.virtual_memory().available<1.5*1024**3:reason='RAM guard'
    elif shutil.disk_usage(ROOT).free<3*1024**3:reason='Disk guard'
    elif time.monotonic()-start>timeout:reason='Time guard'
    if reason:
     for p in reversed(family):
      try:p.kill()
      except psutil.Error:pass
     break
   except psutil.Error:pass
   time.sleep(.25)
  process.wait()
 elapsed=time.monotonic()-start
 return dict(status='SUCCESS' if process.returncode==0 and not reason else 'FAILED',exit_code=process.returncode,reason=reason,wall_seconds=elapsed,peak_ram_bytes=peak,cpu_seconds=cpu,average_cpu_percent_one_core=100*cpu/elapsed,log=str(log),last_error=log.read_text('utf-8',errors='replace')[-4000:] if process.returncode or reason else '')

def setup(model):
 folder=ROOT/'vc_models'/('post_'+model);folder.mkdir(parents=True,exist_ok=True)
 repo,packages=MODELS[model];env=env_for(model);phases=[]
 # Own venv/package tree; shared installed CPU torch is read-only via a .pth.
 python=folder/'.venv/Scripts/python.exe'
 commands=[]
 if not python.exists():commands.append([BASE,'-m','venv','--without-pip',folder/'.venv'])
 if repo and not (folder/'repo/.git').exists():commands.append(['git','clone','--depth','1',repo,folder/'repo'])
 for i,cmd in enumerate(commands):
  r=guarded(cmd,OUT/f'logs/{model}_setup_{i}.log',env);phases.append(r)
  if r['status']!='SUCCESS':return r
 site=folder/'.venv/Lib/site-packages';site.mkdir(parents=True,exist_ok=True)
 (site/'shared_cpu_runtime.pth').write_text('import sys; sys.path.append('+ascii(str(ROOT/'vc_models/meanvc2/.venv/Lib/site-packages'))+')\n','ascii')
 marker=folder/'vendor/.complete';package_stamp='\n'.join(packages)
 if packages and (not marker.exists() or marker.read_text('utf-8')!=package_stamp):
  uv=shutil.which('uv')
  if not uv:raise RuntimeError('uv executable unavailable')
  r=guarded([uv,'pip','install','--python',python,'--no-deps','--target',folder/'vendor',*packages],OUT/f'logs/{model}_packages.log',env)
  phases.append(r)
  if r['status']!='SUCCESS':return r
  marker.write_text(package_stamp,'utf-8')
 if model=='resemble':
  package=folder/'repo/resemble_enhance'
  # Bypass imports of the training engine, not neural inference computations.
  for sub,name in [('enhancer','Enhancer'),('denoiser','Denoiser')]:
   source=package/sub/'inference.py';code=source.read_text('utf-8')
   code=code.replace(f'from .train import {name}, HParams',f'from .{sub} import {name}\nfrom .hparams import HParams')
   source.write_text(code,'utf-8')
  source=package/'enhancer/enhancer.py';code=source.read_text('utf-8')
  code=code.replace('from ..utils.distributed import global_leader_only\n','').replace('from ..utils.train_loop import TrainLoop\n','').replace('    @global_leader_only\n','')
  if 'def _visualize(self, original_mel, denoised_mel):\n        return' not in code:
   code=code.replace('def _visualize(self, original_mel, denoised_mel):','def _visualize(self, original_mel, denoised_mel):\n        return  # Offline inference: training visualizer is disabled.')
  source.write_text(code,'utf-8')
  (package/'utils/__init__.py').write_text('from .utils import save_mels, tree_map\n','utf-8')
  source=package/'hparams.py';code=source.read_text('utf-8')
  code=code.replace('OmegaConf.load(path)',"OmegaConf.create(path.read_text('utf-8').replace('pathlib.PosixPath', 'pathlib.Path'))")
  source.write_text(code,'utf-8')
  save(folder/'inference_only_patch.json',dict(reason='Avoid training-only DeepSpeed imports and map YAML PosixPath tags to native Path on Windows CPU; model forward/checkpoint untouched',files=['enhancer/inference.py','denoiser/inference.py','enhancer/enhancer.py','utils/__init__.py','hparams.py']))
 return dict(status='SUCCESS',phases=phases,python=str(python))

def prepare():
 import soundfile as sf
 from tools.vc_tournament.audio import stats
 selected=json.loads((ROOT/'settings.json').read_text('utf-8'))['ai_model']
 from src.vc.models import profile
 p=profile(selected)
 if p.get('backend')!='meanvc2':raise ValueError('Expected current MeanVC2 selection')
 cases={'normal':(ROOT/'recordings/v011_mega_tournament/source/source_normal.wav',8),
        'low':(ROOT/'recordings/v011_mega_tournament/source/source_low.wav',8),
        'bright':(ROOT/'recordings/v011_mega_tournament/source/source_bright.wav',8),
        'long':(ROOT/'recordings/v011_meanvc2_continuity/source/source_long.wav',25)}
 inputs={}
 for name,(source,seconds) in cases.items():
  stamp=OUT/f'metadata/base_{name}.json';baseline=OUT/f'input/{name}.wav'
  fingerprint=hashlib.sha256((selected+stats(source)['sha256']+str(seconds)).encode()).hexdigest()
  if stamp.exists() and baseline.exists() and json.loads(stamp.read_text('utf-8')).get('fingerprint')==fingerprint:
   inputs[name]=baseline;continue
  label='post_vc_'+name
  r=guarded([ROOT/'.venv/Scripts/python.exe',ROOT/'tools/verify_voice_library.py','--profile',selected,'--source',source,'--seconds',seconds,'--output-label',label],OUT/f'logs/base_{name}.log',env_for(),240)
  if r['status']!='SUCCESS':save(stamp,r);raise RuntimeError('Baseline failed: '+str(r))
  result=json.loads((ROOT/f'validation/v011/{label}_verification.json').read_text('utf-8'))
  x,sr=sf.read(result['output'],dtype='float32')
  model=result['worker_model'];delay=(model['algorithmic_buffer_ms']+model.get('interpolation_grid_delay_ms',0)+model.get('phrase_extra_delay_ms',0))/1000
  # Fixed algorithmic leading buffer only; no content/pitch/EQ editing.
  x=x[round(delay*sr):round((delay+result['input_seconds'])*sr)]
  if len(x)!=round(result['input_seconds']*sr):raise ValueError('Baseline flush too short')
  baseline.parent.mkdir(parents=True,exist_ok=True);sf.write(baseline,x,sr,subtype='FLOAT')
  source_audio,source_sr=sf.read(source,dtype='float32');(OUT/'source').mkdir(exist_ok=True)
  sf.write(OUT/f'source/{name}.wav',source_audio[:round(result['input_seconds']*source_sr)],source_sr,subtype='PCM_16')
  save(stamp,dict(fingerprint=fingerprint,profile=selected,leading_buffer_trim_seconds=delay,verification=result,input=stats(baseline)))
  inputs[name]=baseline
 (OUT/'reference').mkdir(exist_ok=True);shutil.copy2(ROOT/'models'/p['folder']/'reference.wav',OUT/'reference/target.wav')
 return selected,inputs

def render(rows):
 from tools.vc_tournament.audio import normalize,stats
 (OUT/'blind').mkdir(parents=True,exist_ok=True);mapping=[];public=[]
 good=sorted([r for r in rows if r['status']=='SUCCESS'],key=lambda r:(r['source'],r['model']))
 random.Random(1105).shuffle(good)
 for i,r in enumerate(good,1):
  bid=f'P{i:03d}';dest=OUT/f'blind/{bid}.wav';correction=normalize(r['output'],dest)
  mapping.append(dict(id=bid,**r,normalization=correction))
  public.append(dict(id=bid,audio=dest.name,source='../source/'+r['source']+'.wav',reference='../input/'+r['source']+'.wav',case=r['source']))
 data=dict(package=hashlib.sha256(json.dumps(public,sort_keys=True).encode()+''.join(stats(OUT/'blind'/r['audio'])['sha256'] for r in public).encode()).hexdigest(),candidates=public)
 save(OUT/'metadata/blind_manifest.json',mapping)
 template=(ROOT/'tools/vc_tournament/listening.html').read_text('utf-8')
 template=template.replace('v0.11 Blind Voice Comparison','変換後の自然さ · Blind比較').replace('Target Reference','現在の変換音声（後段なし）').replace('Targetへの似具合','現在の声の維持').replace('Targetへの類似','現在の声の維持').replace('元声残り','幼さ・無理に高くした感じ').replace('元声の少なさ','幼さの少なさ').replace('v011-ratings','post-vc-ratings')
 template=template.replace("['reference','Target Reference']","['reference','現在の変換音声（後段なし）']")
 template=template.replace("title.textContent=c.id", "title.textContent=c.id+' · '+({normal:'通常声',low:'低め',bright:'明るめ',long:'長文'}[c.case]||c.case)")
 template=template.replace('<main id="candidates">','<label>録音の種類 <select id="case-filter"><option value="long">長文</option><option value="normal">通常声</option><option value="low">低め</option><option value="bright">明るめ</option><option value="all">全て</option></select></label><main id="candidates">')
 template=template.replace('for(const c of data.candidates){',"for(const c of data.candidates.filter(c=>document.getElementById('case-filter').value==='all'||c.case===document.getElementById('case-filter').value)){")
 template=template.replace('render();\n</script>',"document.getElementById('case-filter').onchange=render;\nrender();\n</script>")
 template=template.replace('<main id="candidates">','<p>基準は現在のVC出力です。基準そのものも匿名候補に含みます。後段による声の変化・自然さ・機械感を評価してください。追加モデルはoffline試験で、ライブ動作は未確認です。</p><p>登録Reference: <audio controls preload="none" src="../reference/target.wav"></audio></p><main id="candidates">')
 (OUT/'blind/index.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c')),'utf-8')
 lines=['# 後段自然さ比較','',f'成功WAV {len(good)} / 失敗条件 {sum(r["status"]=="FAILED" for r in rows)}。READY FOR HUMAN LISTENING。Winner未決定。',
 '全てoffline WAV。学習、音声デバイス、アプリ設定変更なし。元WAV保持。BlindはDC除去と一定音量ゲインのみ。','',
 '| モデル | Source | 状態 | 推論RTF | Peak RAM GiB | 理由 |','|---|---|---|---:|---:|---|']
 for r in rows:
  rtf=f'{r["rtf"]:.3f}' if 'rtf' in r else '—'
  ram=f'{r["peak_ram_bytes"]/1024**3:.2f}' if 'peak_ram_bytes' in r else '—'
  reason=str(r.get('reason') or r.get('last_error') or '').replace(chr(10),' ').replace('|','/')[-180:]
  lines.append(f'| {r["model"]} | {r["source"]} | {r["status"]} | {rtf} | {ram} | {reason} |')
 lines+=['','速度は後段単体のoffline値。VCを含むライブ遅延や全経路RTFの保証ではない。未成功モデルのstage/log/最後のエラーはmetadata/results.jsonに保持。',
 '各モデルの環境はvc_models/post_*。CPU PyTorchの既存パッケージをread-only共有し、モデル専用venv/vendor/repoへ依存を隔離。',
 '比較ページ: recordings/v011_post_vc/blind/index.html。評価JSON名: post-vc-ratings.json。',
 'Context7 MCPはこのセッションでは利用不可。モデルAPIは取得した公式README/ソースを確認。']
 lines+=['','## モデルと比較条件','','現在選択中のMeanVC2出力を固定入力として使用。通常声・低め・明るめ・25秒の長文を同じ条件で比較。登録Referenceと後段なし音声をページから再生できる。声の維持・自然さ・女声らしさは高いほど良く、幼さと機械感は低いほど良い。人間の試聴は未実施。',
 '','| モデル | 公式配布元 | 配置済み推論資産 MiB |','|---|---|---:|']
 for model,(repo,_) in MODELS.items():
  folder=ROOT/'vc_models'/('post_'+model)
  assets=[p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in {'.pt','.pth','.bin','.onnx','.safetensors'} and not any(part in {'.venv','vendor','.cache','.git'} for part in p.parts)]
  size=sum(p.stat().st_size for p in assets)/1024**2
  url=repo or 'https://huggingface.co/alibabasglab/MossFormer2_SR_48K'
  lines.append(f'| {model} | [{model}]({url.removesuffix(".git")}) | {size:.1f} |')
 lines+=['','Resemble EnhanceはWindows CPU推論のため学習用DeepSpeedのimportと学習可視化を切り離し、YAMLのPosixPathをWindowsのPathへ読み替えた。モデルのforwardと配布重みは変更していない。',
 'MossFormerは必要な推論checkpointを事前取得・存在確認する。ClearVoiceが重み未取得でも処理を継続した旧試行は失敗として除外し、履歴を保持。',
 'モデル名と速度の対応はmetadata/blind_manifest.jsonに分離。評価はブラウザ内保存に加えpost-vc-ratings.jsonへ書き出せる。']
 (ROOT/'validation/v011/post_vc_comparison_report.md').write_text('\n'.join(lines)+'\n','utf-8')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='*',default=list(MODELS));parser.add_argument('--render-only',action='store_true');args=parser.parse_args()
 path=OUT/'metadata/results.json';rows=json.loads(path.read_text('utf-8')) if path.exists() else []
 if args.render_only:render(rows);return
 if rows:save(OUT/f'metadata/attempt_history/results_{time.time_ns()}.json',rows)
 from tools.compare_meanvc2_continuity import require_idle_voice_worker
 require_idle_voice_worker();selected,inputs=prepare()
 from tools.vc_tournament.audio import stats
 for row in rows:
  if row['model']!='baseline' and row.get('input_sha256')!=stats(inputs[row['source']])['sha256']:
   row.update(status='STALE',reason='Selected baseline changed; retained previous output is excluded from listening')
 for name,source in inputs.items():
  rows=[r for r in rows if not(r['model']=='baseline' and r['source']==name)]
  rows.append(dict(model='baseline',source=name,status='SUCCESS',output=str(source),profile=selected))
 save(path,rows);render(rows)
 for model in args.models:
  if model not in MODELS:raise ValueError(model)
  print('SETUP '+model,flush=True);setup_result=setup(model);save(OUT/f'metadata/setup_{model}.json',setup_result)
  for name,source in inputs.items():
   from tools.vc_tournament.audio import stats
   fingerprint=stats(source)['sha256']
   adapter_revision='moss_prefetch_v2' if model=='mossformer_sr' else '2d4c0b484747c0f92fb1de782d736f450b5edca50e79a62a307b803ad34bea33'
   cached=next((r for r in rows if r['model']==model and r['source']==name and r.get('input_sha256')==fingerprint and r.get('adapter_revision',r.get('adapter_sha256'))==adapter_revision and r['status']=='SUCCESS' and Path(r['output']).exists()),None)
   if cached:continue
   if setup_result['status']!='SUCCESS':r=dict(setup_result,stage='setup')
   else:
    worker_meta=OUT/f'metadata/{model}_{name}_worker.json';output=OUT/f'raw/{model}/{name}.wav'
    print('INFER '+model+' '+name,flush=True)
    r=guarded([setup_result['python'],ROOT/'tools/post_vc_worker.py','--model',model,'--input',source,'--output',output,'--metadata',worker_meta],OUT/f'logs/{model}_{name}.log',env_for(model),600)
    r['stage']='load/download/inference'
    if r['status']=='SUCCESS':r.update(json.loads(worker_meta.read_text('utf-8')))
    else:r['reason']=r.get('reason') or (r['last_error'].splitlines()[-1] if r['last_error'] else 'Process failed')
   rows=[old for old in rows if not(old['model']==model and old['source']==name)]
   rows.append(dict(r,model=model,source=name,input_sha256=fingerprint))
   save(path,rows);render(rows)
 print(str(OUT/'blind/index.html'),flush=True)

if __name__=='__main__':main()
