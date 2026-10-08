"""Guarded, resumable offline post-VC comparison #2 (new conditions + C1 grid).

New conditions: lavasr2 (denoise on), voicefixer2, flashsr_voicefixer2,
lavasrden_flashsr, plus the C1 meanvc2 offline 2-step/3-step grid on the current
profile inputs. Baselines are carried over from recordings/v011_post_vc.
No audio devices, no app edits; implementation work later must go into a NEW
separate mode, not the existing ones.
"""
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
OUT=ROOT/'recordings/v011_post_vc2'
BASE=ROOT/'vc_models/meanvc2/.venv/Scripts/python.exe'
CHAINS=['flashsr','lavasr2','voicefixer2','flashsr_voicefixer2','lavasrden_flashsr']

def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),'utf-8');tmp.replace(path)

def env_for():
 env=os.environ.copy();cache=ROOT/'vc_models/cache';temp=cache/'temp/post_vc2';temp.mkdir(parents=True,exist_ok=True)
 env.update(TEMP=str(temp),TMP=str(temp),TMPDIR=str(temp),HF_HOME=str(cache/'hf-home'),HF_HUB_CACHE=str(cache/'hf'),TORCH_HOME=str(cache/'torch'),PIP_CACHE_DIR=str(cache/'pip'),UV_CACHE_DIR=str(cache/'uv'),XDG_CACHE_HOME=str(cache/'xdg'),NUMBA_CACHE_DIR=str(cache/'numba'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HF_HUB_DISABLE_XET='1',PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='')
 paths=[ROOT/'vc_models/post_lavasr/vendor',ROOT/'vc_models/post_lavasr/repo',ROOT/'vc_models/post_flashsr/vendor',ROOT/'vc_models/post_flashsr/repo',ROOT/'vc_models/post_voicefixer2/vendor',ROOT/'vc_models/post_voicefixer2/repo',ROOT]
 env['PYTHONPATH']=os.pathsep.join(map(str,paths))
 return env

def guarded(command,log,env,timeout=900):
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
 return dict(status='SUCCESS' if process.returncode==0 and not reason else 'FAILED',exit_code=process.returncode,reason=reason,wall_seconds=elapsed,peak_ram_bytes=peak,cpu_seconds=cpu,log=str(log),last_error=log.read_text('utf-8',errors='replace')[-4000:] if process.returncode or reason else '')

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
  label='postvc2_'+name
  r=guarded([ROOT/'.venv/Scripts/python.exe',ROOT/'tools/verify_voice_library.py','--profile',selected,'--source',source,'--seconds',seconds,'--output-label',label],OUT/f'logs/base_{name}.log',env_for(),600)
  if r['status']!='SUCCESS':save(stamp,r);raise RuntimeError('Baseline failed: '+str(r))
  result=json.loads((ROOT/f'validation/v011/{label}_verification.json').read_text('utf-8'))
  x,sr=sf.read(result['output'],dtype='float32')
  model=result['worker_model']
  delay=(model['algorithmic_buffer_ms']+model.get('interpolation_grid_delay_ms',0)+model.get('phrase_extra_delay_ms',0))/1000
  x=x[round(delay*sr):round((delay+result['input_seconds'])*sr)]
  if len(x)!=round(result['input_seconds']*sr):raise ValueError('Baseline flush too short')
  baseline.parent.mkdir(parents=True,exist_ok=True);sf.write(baseline,x,sr,subtype='FLOAT')
  source_audio,source_sr=sf.read(source,dtype='float32');(OUT/'source').mkdir(exist_ok=True)
  sf.write(OUT/f'source/{name}.wav',source_audio[:round(result['input_seconds']*source_sr)],source_sr,subtype='PCM_16')
  save(stamp,dict(fingerprint=fingerprint,profile=selected,leading_buffer_trim_seconds=delay,verification=result,input=stats(baseline)))
  inputs[name]=baseline
 (OUT/'reference').mkdir(exist_ok=True);shutil.copy2(ROOT/'models'/p['folder']/'reference.wav',OUT/'reference/target.wav')
 return selected,inputs

C1_RAW_SOURCES={'normal':ROOT/'recordings/v011_mega_tournament/source/source_normal.wav',
 'low':ROOT/'recordings/v011_mega_tournament/source/source_low.wav',
 'bright':ROOT/'recordings/v011_mega_tournament/source/source_bright.wav',
 'long':ROOT/'recordings/v011_meanvc2_continuity/source/source_long.wav'}
C1_REFERENCE_SHA='453589ccbbcb9472d3e02b739fbe92b2861696fc14dc3e8b63d746be2682869b'

def c1_rows(rows):
 from tools.vc_tournament.audio import stats
 reference=ROOT/'models/meanvc2_120/reference.wav'
 for name,source in C1_RAW_SOURCES.items():
  fingerprint=stats(source)['sha256']
  rows=[r for r in rows if not(r['model'].startswith('c1_') and r['source']==name)]
  for variant in ('offline2','offline3'):
   output=OUT/f'raw/{variant}/{name}.wav'
   meta=OUT/f'metadata/{variant}_{name}_worker.json'
   steps=variant[-1]
   if meta.exists() and output.exists() and json.loads(meta.read_text('utf-8')).get('input_sha256')==fingerprint and json.loads(meta.read_text('utf-8')).get('stage')=='complete':
    row=dict(json.loads(meta.read_text('utf-8')));row['model']='c1_'+variant;row['source']=name;row['status']='SUCCESS';rows.append(row);continue
   r=guarded([BASE,str(ROOT/'tools/vc_tournament/infer_c1.py'),str(source),str(reference),str(output),str(meta),steps],OUT/f'logs/{variant}_{name}.log',env_for(),900)
   r['stage']='c1_'+variant
   if r['status']=='SUCCESS':
    data=json.loads(meta.read_text('utf-8'))
    if data.get('steps')!=int(steps) or data.get('reference_sha256')!=C1_REFERENCE_SHA:
     r=dict(status='FAILED',reason='C1 condition guard failed: steps/reference mismatch',stage='c1_'+variant)
    else:r.update(data)
   else:r['reason']=r.get('reason') or (r['last_error'].splitlines()[-1] if r['last_error'] else 'Process failed')
   rows.append(dict(r,model='c1_'+variant,source=name,input_sha256=fingerprint))
   save(OUT/'metadata/results.json',rows)
  # Streaming 2-step via the production MeanVC2Backend offline replay (the same
  # tool that produced the user-listened B-series WAVs). Reference is fixed by
  # the meanvc2_120 profile folder and asserted here.
  if stats(reference)['sha256']!=C1_REFERENCE_SHA:raise RuntimeError('meanvc2_120 reference changed; C1 streaming comparison would be invalid')
  stem=OUT/f'metadata/c1_stream2_{name}'
  # The benchmark tool writes its report to the exact --report path (no .json
  # suffix) and the WAV next to it via with_suffix('.wav').
  if not (stem.with_suffix('.wav').exists() and stem.exists()
          and json.loads(stem.read_text('utf-8')).get('stage')=='complete'):
   r=guarded([BASE,str(ROOT/'tools/benchmark_meanvc2_realtime.py'),'--model','meanvc2_120','--source',str(source),'--seconds','60','--threads','4','--report',str(stem)],OUT/f'logs/stream2_{name}.log',env_for(),900)
   r['stage']='c1_stream2'
   if r['status']=='SUCCESS':
    data=json.loads(stem.read_text('utf-8'))
    if data.get('profile')!='meanvc2_120' or not data.get('output_finite',True):
     r=dict(status='FAILED',reason='C1 streaming guard failed: profile/finiteness mismatch',stage='c1_stream2')
    else:r['rtf']=data.get('rtf');r['reused_from']=str(stem)
   else:r['reason']=r.get('reason') or (r['last_error'].splitlines()[-1] if r['last_error'] else 'Process failed')
  else:
   data=json.loads(stem.read_text('utf-8'))
   r=dict(status='SUCCESS',rtf=data.get('rtf'),reused_from=str(stem),stage='c1_stream2')
  rows.append(dict(r,model='c1_stream2',source=name,input_sha256=fingerprint,output=str(stem.with_suffix('.wav'))))
  save(OUT/'metadata/results.json',rows)
  # Streaming 3-step has no steps override in the production tool; only the
  # user-listened cached B-series WAV (normal) participates.
  stem3=ROOT/'validation/v011/meanvc2_stream_sdpa3'
  try:
   j=json.loads((stem3.with_suffix('.json')).read_text('utf-8'))
   ok=(j.get('stage')=='complete' and j.get('steps')==3 and j.get('reference_sha256')==C1_REFERENCE_SHA
       and 'source_normal' in j.get('source','').replace('\\','/'))
  except FileNotFoundError:
   ok=False
  if ok and name=='normal':
   rows.append(dict(model='c1_stream3',source=name,status='SUCCESS',output=str(stem3.with_suffix('.wav')),input_sha256=fingerprint,reused_from='meanvc2_stream_sdpa3',
                    provenance='User-listened B-series cache reused; steps/reference/source recorded in meanvc2_stream_sdpa3.json'))
  else:
   rows.append(dict(model='c1_stream3',source=name,status='FAILED',reason='Streaming 3-step has no runtime override; only the cached B-series normal source participates',stage='cache_check'))
  save(OUT/'metadata/results.json',rows)
 return rows

def guarded_with_retry(command,log,env,timeout=900):
 import psutil
 for _ in range(18):
  if psutil.virtual_memory().available>2.5*1024**3:break
  time.sleep(10)
 first=guarded(command,log,env,timeout)
 if first['status']=='SUCCESS' or first.get('reason')!='RAM guard':return first
 # Transient memory pressure (user apps fluctuate); retry once after a pause.
 for _ in range(12):
  time.sleep(10)
  if psutil.virtual_memory().available>2.5*1024**3:break
 print('RETRY after RAM guard: '+log.name,flush=True)
 second=guarded(command,log,env,timeout)
 second['retried_after_ram_guard']=True
 return second

def infer_chains(rows,inputs):
 from tools.vc_tournament.audio import stats
 COMPOSITES={'flashsr_voicefixer2':['flashsr','voicefixer2'],'lavasrden_flashsr':['lavasrden','flashsr']}
 for chain in CHAINS:
  stages=COMPOSITES.get(chain,[chain])
  for name,source in inputs.items():
   fingerprint=stats(source)['sha256']
   adapter_revision='postvc2_conditions_v1'
   cached=next((r for r in rows if r['model']==chain and r['source']==name and r.get('input_sha256')==fingerprint and r.get('adapter_revision')==adapter_revision and r['status']=='SUCCESS' and Path(r['output']).exists()),None)
   if cached:continue
   output=OUT/f'raw/{chain}/{name}.wav';meta=OUT/f'metadata/{chain}_{name}_worker.json'
   runner=None
   row=dict(model=chain,source=name,input_sha256=fingerprint)
   if len(stages)==1:
    r=guarded_with_retry([BASE,str(ROOT/'tools/post_vc_worker2.py'),'--chain',chain,'--input',str(source),'--output',str(output),'--metadata',str(meta)],OUT/f'logs/{chain}_{name}.log',env_for(),900)
   else:
    staging=OUT/'raw'/'_staging';staging.mkdir(parents=True,exist_ok=True)
    current=str(source);rtf=0;prov=[];failure=None
    for k,stage in enumerate(stages):
     stage_meta=OUT/f'metadata/{chain}_{name}_stage{k}.json'
     stage_out=staging/f'{chain}_{name}_stage{k}.wav'
     r=guarded_with_retry([BASE,str(ROOT/'tools/post_vc_worker2.py'),'--chain',stage,'--input',current,'--output',str(stage_out),'--metadata',str(stage_meta)],OUT/f'logs/{chain}_{name}_stage{k}.log',env_for(),900)
     if r['status']!='SUCCESS':
      failure=dict(r,stage=f'{chain}:stage{k}:{stage}');break
     data=json.loads(stage_meta.read_text('utf-8'));rtf+=data['rtf'];prov.append(data['provenance']);current=str(stage_out)
    if failure:r=failure
    else:
     output.parent.mkdir(parents=True,exist_ok=True);shutil.move(current,output)
     r=dict(status='SUCCESS',rtf=rtf,stages=stages,provenance=prov,output=str(output))
   r['stage']='load/download/inference'
   if r['status']=='SUCCESS' and len(stages)==1:r.update(json.loads(meta.read_text('utf-8')))
   elif r['status']!='SUCCESS':r['reason']=r.get('reason') or (r.get('last_error','').splitlines()[-1] if r.get('last_error') else 'Process failed')
   rows=[old for old in rows if not(old['model']==chain and old['source']==name)]
   row.update(r)
   rows.append(row)
   save(OUT/'metadata/results.json',rows)
 return rows

def render(rows,selected,inputs):
 from tools.vc_tournament.audio import normalize,stats
 import numpy as np
 import soundfile as sf
 (OUT/'blind').mkdir(parents=True,exist_ok=True);mapping=[];public=[]
 good=sorted([r for r in rows if r['status']=='SUCCESS' and r.get('model') not in ('baseline',)],key=lambda r:(r['source'],r['model']))
 random.Random(1106).shuffle(good)
 good=[r for r in good if r.get('output') and Path(r['output']).exists()]
 for i,r in enumerate(good,1):
  bid=f'Q{i:03d}';dest=OUT/f'blind/{bid}.wav';correction=normalize(r['output'],dest)
  mapping.append(dict(id=bid,**r,normalization=correction))
  refdir='source' if r['model'].startswith('c1_') else 'input'
  public.append(dict(id=bid,audio=dest.name,source='../source/'+r['source']+'.wav',reference='../'+refdir+'/'+r['source']+'.wav',case=r['source']))
 data=dict(package=hashlib.sha256(json.dumps(public,sort_keys=True).encode()+''.join(stats(OUT/'blind'/r['audio'])['sha256'] for r in public).encode()).hexdigest(),candidates=public)
 save(OUT/'metadata/blind_manifest.json',mapping)
 template=(ROOT/'tools/vc_tournament/listening.html').read_text('utf-8')
 template=template.replace('v0.11 Blind Voice Comparison','後段改善 第2比較 · Blind').replace('Target Reference','現在の変換音声（後段なし・今回基準）').replace('Targetへの似具合','現在の声の維持').replace('Targetへの類似','現在の声の維持').replace('元声残り','幼さ・無理に高くした感じ').replace('元声の少なさ','幼さの少なさ').replace('v011-ratings','post-vc2-ratings')
 template=template.replace("['reference','Target Reference']","['reference','現在の変換音声（後段なし・今回基準）']")
 template=template.replace("title.textContent=c.id","title.textContent=c.id+' · '+({normal:'通常声',low:'低め',bright:'明るめ',long:'長文'}[c.case]||c.case)")
 template=template.replace('<main id="candidates">','<label>録音の種類 <select id="case-filter"><option value="long">長文</option><option value="normal">通常声</option><option value="low">低め</option><option value="bright">明るめ</option><option value="all">全て</option></select></label><main id="candidates">')
 template=template.replace('for(const c of data.candidates){',"for(const c of data.candidates.filter(c=>document.getElementById('case-filter').value==='all'||c.case===document.getElementById('case-filter').value)){")
 template=template.replace('render();\\n</script>',"document.getElementById('case-filter').onchange=render;\\nrender();\\n</script>")
 template=template.replace('<main id="candidates">','<p>基準は今回の現在の出力（後段なし）です。VoiceFixer2条件は44.1kHz出力をそのまま比較します。モデル対応表はmetadata/blind_manifest.jsonへ分離。offline WAV比較で、ライブ動作・実機遅延は未確認。</p><main id="candidates">')
 (OUT/'blind/index.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c')),'utf-8')
 lines=['# 後段改善 第2比較（新条件+発話推論方式の切り分け）','',f'成功 {len(good)} / 全行 {len(rows)}。READY FOR HUMAN LISTENING。Winner未決定。','',
 '全てoffline WAV。学習、音声デバイス、アプリ設定変更なし。元WAV保持。BlindはDC除去と一定音量ゲインのみ。','',
 '| 条件 | Source | 状態 | 推論RTF | Peak RAM GiB | 理由 |','|---|---|---|---:|---:|---|']
 for r in rows:
  rtf=f'{r["rtf"]:.3f}' if 'rtf' in r else '—'
  ram=f'{r["peak_ram_bytes"]/1024**3:.2f}' if 'peak_ram_bytes' in r else '—'
  reason=str(r.get('reason') or r.get('last_error') or '').replace(chr(10),' ').replace('|','/')[-180:]
  lines.append(f'| {r["model"]} | {r["source"]} | {r["status"]} | {rtf} | {ram} | {reason} |')
 lines+=['','速度は条件単体のoffline値。VCを含むライブ遅延や全経路RTFの保証ではない。','',
 '## 条件','',f'現在の選択プロファイル: `{selected}`。4ソース(通常/低め/明るめ/25秒長文)は現在の出力(後段なし)に固定アルゴリズム遅延分の先頭を除いたもの。','',
 '- lavasr2: LavaSR v2のdenoise=True(重みはv011_post_vcのlavasrと同じローカル資産。v1比較はdenoise=Falseだった差分条件)','',
 '- voicefixer2: Render-AI-Team voicefixer2(vf.ckpt+vocoder)。44.1kHz出力。numpy2系互換フォーク','',
 '- flashsr_voicefixer2: FlashSR(48k)→VoiceFixer2の2段','',
 '- lavasrden_flashsr: 16kHzでLavaSR denoiser→FlashSRの2段(BWEはFlashSRのみ)','',
 '- c1_offline2 / c1_offline3: 同一Reference(453589cc…)・seed110でオフライン全体attention推論を2step/3stepで実行。ストリーミング2step(現行)との差分が「発話一括推論方式」そのものの効果',
 '- 基準(後段なし)・既存flashsr/lavasr/apbwe/novasr条件は第1比較(recordings/v011_post_vc)のWAVをそのまま使用','',
 '比較ページ: recordings/v011_post_vc2/blind/index.html。評価JSON名: post-vc2-ratings.json。','',
 '隔離: venv共有はvc_models/meanvc2/.venv(既存パッケージを書き換えない)+post_voicefixer2/vendorへ新規pipパッケージを分離。PyTorchは既存のtorch 2.5.1+cpuを使用。モデルのforwardと配布重みは変更していない。','',
 '実装方針: 採用時は既存モードを壊さず必ず別モード(第5の処理方法)として追加する。この比較はoffline WAV検証のみ。']
 (ROOT/'validation/v011/post_vc2_comparison_report.md').write_text('\n'.join(lines)+'\n','utf-8')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--render-only',action='store_true');args=parser.parse_args()
 path=OUT/'metadata/results.json';rows=json.loads(path.read_text('utf-8')) if path.exists() else []
 if args.render_only:
  selected=json.loads((ROOT/'settings.json').read_text('utf-8'))['ai_model']
  render(rows,selected,{});return
 from tools.compare_meanvc2_continuity import require_idle_voice_worker
 require_idle_voice_worker();selected,inputs=prepare()
 rows=c1_rows(rows);save(OUT/'metadata/results.json',rows)
 rows=infer_chains(rows,inputs);save(OUT/'metadata/results.json',rows)
 rows=[r for r in rows if r['model']!='baseline']+[dict(model='baseline',source=name,status='SUCCESS',output=str(source),profile=selected) for name,source in inputs.items()]
 save(path,rows);render(rows,selected,inputs)
 print(str(OUT/'blind/index.html'),flush=True)

if __name__=='__main__':main()
