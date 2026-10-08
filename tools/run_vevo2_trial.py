"""Bounded resumable offline trial; no application or audio-device changes."""
import hashlib, html, json, os, shutil, subprocess, sys, time
from pathlib import Path
import psutil
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'vc_models/vevo2'
OUT=ROOT/'recordings/v011_vevo2_style_trial'
OUT.mkdir(parents=True,exist_ok=True)
CACHE=ROOT/'vc_models/cache'
ENV=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',
         OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',HF_HUB_DISABLE_XET='1')
for key,folder in {'TEMP':'temp','TMP':'temp','UV_CACHE_DIR':'uv','HF_HOME':'hf-home',
 'HF_HUB_CACHE':'hf','TRANSFORMERS_CACHE':'hf','TORCH_HOME':'torch','XDG_CACHE_HOME':'xdg',
 'MPLCONFIGDIR':'matplotlib','NUMBA_CACHE_DIR':'numba','NLTK_DATA':'nltk'}.items():
    dest=CACHE/folder;dest.mkdir(exist_ok=True,parents=True);ENV[key]=str(dest)
ENV['PYTHONPATH']=str(ROOT/'vc_models/amphion/repo')
def run(stage,command,timeout):
    log=OUT/(stage+'.log');started=time.perf_counter();peak=0;cpu=0;reason=None
    previous=OUT/(stage+'_process.json')
    if previous.exists():
        history=OUT/'attempt_history';history.mkdir(exist_ok=True)
        stamp=str(time.time_ns());shutil.copy2(previous,history/(stage+'_'+stamp+'.json'))
        if log.exists():shutil.copy2(log,history/(stage+'_'+stamp+'.log'))
    with log.open('w',encoding='utf-8') as stream:
        child=subprocess.Popen([str(c) for c in command],cwd=ROOT/'vc_models/amphion/repo' if stage.startswith('inference') else ROOT,
             env=ENV,stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        parent=psutil.Process(child.pid);seen={}
        try:parent.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        except psutil.Error:pass
        while child.poll() is None:
            rss=0
            try:
                for p in [parent]+parent.children(recursive=True):
                    rss+=p.memory_info().rss
                    if p.pid not in seen:
                        seen[p.pid]=p;p.cpu_percent(None)
                        try:p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                        except psutil.Error:pass
                    cpu=max(cpu,seen[p.pid].cpu_percent(None))
            except psutil.Error:pass
            peak=max(peak,rss)
            available=psutil.virtual_memory().available
            if available<2.0*2**30:reason='SYSTEM_AVAILABLE_RAM_BELOW_2_GIB'
            if peak>4.5*2**30:reason='PROCESS_RAM_ABOVE_4_5_GIB'
            if psutil.disk_usage(str(ROOT)).free<2.5*2**30:reason='D_DISK_RESERVE'
            if time.perf_counter()-started>timeout:reason='OFFLINE_TIMEOUT'
            if reason:
                try:
                    for p in parent.children(recursive=True):p.kill()
                    parent.kill()
                except psutil.Error:pass
                break
            time.sleep(.25)
        child.wait()
    result=dict(stage=stage,status='SUCCESS' if child.returncode==0 and not reason else 'FAILED',
        reason=reason,exit_code=child.returncode,elapsed_seconds=time.perf_counter()-started,
        peak_ram_bytes=peak,cpu_process_percent_max=cpu,log=str(log),
        last_error=log.read_text('utf-8',errors='replace')[-2500:] if child.returncode else None)
    (OUT/(stage+'_process.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    return result
def finish(result):
    report=ROOT/'validation/v011/vevo2_style_trial_report.md'
    inputs=json.loads((OUT/'inputs.json').read_text('utf-8')) if (OUT/'inputs.json').exists() else {}
    lines=['# Vevo2 style-converted VC offline trial','',
           'Current MeanVC2 app configuration and audio path are unchanged. No training or physical audio trial.',
           'Latency is not a selection criterion. Human listening has not been performed.',
           '',f'Status: {result["status"]}',f'Stage: {result["stage"]}',
           f'Reason: {result.get("reason")}',f'Peak process RAM: {result["peak_ram_bytes"]/2**30:.2f} GiB',
           f'Elapsed: {result["elapsed_seconds"]:.1f} seconds','',
           'Reported inference time covers successful AR/FM/vocoder processes with cached reference tokens; setup, earlier feature extraction and failed attempts are excluded.',
           'Target text (unverified ASR): '+inputs.get('target_text',''),
           'Style text (unverified ASR): '+inputs.get('style_text',''),'',
           'Timbre: current adopted MeanVC2 VC720 normal output. Style: selected 5-second original female target reference.',
           'Official AR+FM model; source-prosody copying disabled for style conversion. No pitch/formant/EQ processing.',
           'Inference weights pinned to RMSnow/Vevo2 2674843cbaa50aa89ee7ccaf5bb15d6ccf46c6c8.',
           'Official recipe: https://github.com/open-mmlab/Amphion/blob/main/models/svc/vevo2/README.md',
           'Official model: https://huggingface.co/RMSnow/Vevo2',
           'Staged loading uses the original Whisper medium encoder and official 32-step FM. Cached tokens permit resume.',
           'AR uses official BF16 weights; no quantization or finetuning. Each inference stage runs in a separate process.',
           f'Artifacts: {OUT}']
    failures={}
    for path in list((OUT/'attempt_history').glob('*.json'))+list(OUT.glob('inference_attempt_*.json')):
        item=json.loads(path.read_text('utf-8'))
        if item.get('status')=='FAILED':
            reason=item.get('reason') or (item.get('last_error') or '').strip().split('\n')[-1]
            failures[(item.get('stage'),reason)]=item
    if failures:
        lines+=['','## Retained failed attempts (resolved by later stages when status is SUCCESS)','']
        for (stage,reason),item in failures.items():
            lines.append(f'- {stage}: {reason}; peak RAM {item["peak_ram_bytes"]/2**30:.2f} GiB')
    raw=OUT/'vevo2_style_raw.wav'
    if raw.exists() and result['status']=='SUCCESS':
        if (OUT/'output_asr.json').exists():
            check=json.loads((OUT/'output_asr.json').read_text('utf-8'))
            lines+=['','Output ASR transcript (not human verification): '+check['output_asr_text']]
        blind=OUT/'comparison';blind.mkdir(exist_ok=True)
        metadata={}
        for name,path in [('baseline',OUT/'timbre.wav'),('vevo2',raw),('source',OUT/'source.wav'),('style',OUT/'style.wav')]:
            x,sr=sf.read(path,dtype='float32');x=x.reshape(-1)
            clipping=float(np.mean(np.abs(x)>=1));x=x-np.mean(x)
            window=max(1,round(sr*.02));frames=x[:len(x)//window*window].reshape(-1,window)
            energies=np.sqrt(np.mean(frames*frames,axis=1))
            active=frames[energies>max(float(np.max(energies))*.1,1e-5)]
            rms=float(np.sqrt(np.mean(active*active))) if active.size else float(np.sqrt(np.mean(x*x)))
            gain=min(.1/max(rms,1e-9),.98/max(float(np.max(np.abs(x))),1e-9))
            sf.write(blind/(name+'.wav'),x*gain,sr,subtype='PCM_16')
            metadata[name]={'duration_seconds':len(x)/sr,'raw_clipping_fraction':clipping,'scalar_gain':gain,'active_speech_rms_before':rms}
        (OUT/'output_metrics.json').write_text(json.dumps(metadata,indent=2))
        lines+=['','READY FOR HUMAN LISTENING',f'Comparison: {blind/"index.html"}',
                'Volume matching: DC removal and scalar gain calibrated on active 20-ms frames, with peak headroom only; no compressor or EQ.',
                'Inference wall time includes staged model loading; it is not pure steady-state RTF.']
        text='<!doctype html><meta charset="utf-8"><title>Vevo2 話し方比較</title><style>body{font:18px sans-serif;background:#171923;color:#eee;max-width:850px;margin:40px auto}section{padding:20px;background:#242735;margin:15px 0}textarea{width:95%;height:100px}audio{width:100%}</style><h1>今の声 × Vevo2 話し方変換</h1><p>声質が保たれたか、発音が正しいか、抑揚が良くなったかを確認してください。</p>'
        for name,label in [('baseline','現在採用している声'),('vevo2','Vevo2 style-converted'),('style','話し方Reference（自分で出した女声）'),('source','通常声Source')]:
            text+=f'<section><h2>{label}</h2><audio controls src="{name}.wav"></audio></section>'
        text+='<p>内容：'+html.escape(inputs['target_text'])+'</p><textarea id="comments" placeholder="声質・発音・抑揚の感想"></textarea><p><button onclick="save()">感想をJSONで保存</button></p><script>function save(){let b=new Blob([JSON.stringify({comments:document.getElementById("comments").value,trial:"vevo2_style_converted",time:new Date().toISOString()},null,2)],{type:"application/json"});let a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="vevo2-style-ratings.json";a.click()}</script>'
        (blind/'index.html').write_text(text,encoding='utf-8')
    else:lines+=['','Last error:', '```',result.get('last_error') or '', '```']
    report.write_text('\n'.join(lines),encoding='utf-8')
def main():
    if not (OUT/'inputs.json').exists():
        r=run('prepare',[ROOT/'.venv-asr/Scripts/python.exe',ROOT/'tools/vevo2_trial_prepare.py'],600)
        if r['status']!='SUCCESS':finish(r);return 1
    if not (BASE/'assets_ready.json').exists():
        r=run('setup',[BASE/'.venv310/Scripts/python.exe',ROOT/'tools/vevo2_trial_setup.py'],14400)
        if r['status']!='SUCCESS':finish(r);return 1
    if (OUT/'vevo2_style_raw.wav').exists():
        r=json.loads((OUT/'inference_process.json').read_text())
    else:
        old=OUT/'inference_process.json'
        if old.exists():shutil.copy2(old,OUT/f'inference_attempt_{time.time_ns()}.json')
        results=[]
        for step,artifact in [('tokens','tokens.pt'),('ar','generated_codes.pt'),('fm','generated_mel.pt'),('vocoder','vevo2_style_raw.wav')]:
            profile=OUT/('inference_'+step+'_process.json')
            if (OUT/artifact).exists():
                if profile.exists():results.append(json.loads(profile.read_text()))
                continue
            r=run('inference_'+step,[BASE/'.venv310/Scripts/python.exe',ROOT/'tools/vevo2_trial_infer.py','--stage',step],7200)
            results.append(r)
            if r['status']!='SUCCESS':finish(r);return 1
        r=dict(stage='inference',status='SUCCESS',reason=None,last_error=None,
               elapsed_seconds=sum(v['elapsed_seconds'] for v in results),
               peak_ram_bytes=max((v['peak_ram_bytes'] for v in results),default=0),
               cpu_process_percent_max=max((v['cpu_process_percent_max'] for v in results),default=0),stages=results)
        old.write_text(json.dumps(r,indent=2))
    if r['status']=='SUCCESS' and not (OUT/'output_asr.json').exists():
        run('output_asr',[ROOT/'.venv-asr/Scripts/python.exe',ROOT/'tools/vevo2_trial_prepare.py','--verify-output'],600)
    finish(r)
    return 0 if r['status']=='SUCCESS' else 1
if __name__=='__main__':sys.exit(main())
