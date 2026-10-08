"""Resumable, device-free reference/repair campaign. All storage stays on D:."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'recordings/v011_naturalness'
PACK = ROOT/'models/naturalness_candidates'
from src.vc.voice_library import digest


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), 'utf-8')
    tmp.replace(path)


def sources():
    return {**{k: ROOT/f'recordings/v011_mega_tournament/source/source_{k}.wav'
               for k in ('normal','low','bright')},
            'long': ROOT/'recordings/v011_meanvc2_continuity/source/source_long.wav'}


def prepare():
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from src.prosody.f0 import YinEstimator
    original = ROOT/'recordings/v011_mega_tournament/reference/reference_full.wav'
    audio, rate = sf.read(original, dtype='float32')
    if audio.ndim != 1 or rate != 48000: raise ValueError('Expected original mono48k reference')
    mono = resample_poly(audio, 1, 3).astype(np.float32)
    estimator = YinEstimator()
    features = np.array([estimator.estimate(mono[p-1280:p]) for p in range(1280,len(mono)+1,640)])
    rows = []
    for sec in range(0, int(len(mono)/16000)-4):
        x = mono[sec*16000:(sec+5)*16000]
        f = features[max(0,sec*25):min(len(features),(sec+5)*25-1)]
        valid = (f[:,1] >= .88) & (f[:,0] >= 80) & (f[:,0] <= 550) & (f[:,2] > -48)
        frames = x.reshape(-1,320)
        rms = np.sqrt(np.mean(frames.astype('float64')**2,axis=1))
        active = float(np.mean(rms > .003))
        if active < .6 or valid.sum() < 35 or np.mean(abs(x)>=.999) > .001: continue
        pitch = float(np.median(f[valid,0]))
        variation = float(np.std(12*np.log2(f[valid,0])))
        rows.append(dict(offset=sec, pitch_hz=pitch, pitch_spread_st=variation,
                         rms=float(np.sqrt(np.mean(x*x))), active=active))
    if len(rows) < 3: raise ValueError('Insufficient usable reference windows')
    pitches = np.array([r['pitch_hz'] for r in rows])
    levels = np.array([r['rms'] for r in rows])
    selected = []
    for name, quantile in [('calm',.5),('lower',.2),('bright',.75)]:
        target = float(np.quantile(pitches,quantile))
        ranked = sorted(rows,key=lambda r: abs(np.log2(r['pitch_hz']/target))*3
                        + r['pitch_spread_st']*.1 + abs(np.log(r['rms']/np.median(levels)))*.25
                        + (1-r['active']))
        row = next((r for r in ranked if all(abs(r['offset']-old['offset'])>=5 for old in selected)),None)
        if row is None: raise ValueError('Cannot select three nonoverlapping windows')
        row = dict(row,name=name)
        selected.append(row)
        sf.write(OUT/'reference'/f'{name}.wav',mono[row['offset']*16000:(row['offset']+5)*16000],16000,subtype='PCM_16')
    save(OUT/'metadata/reference_selection.json',dict(original=str(original),original_sha256=digest(original),
        selection='Acoustic proxies, NOT verified naturalness, femininity, adultness, or absence of strain',windows=selected))
    for name, source in sources().items(): shutil.copy2(source,OUT/'source'/f'{name}.wav')
    print('PREPARED',selected,flush=True)


def embeddings():
    import importlib.util
    import types
    import numpy as np
    import soundfile as sf
    import torch
    from tools.register_voice import centroid
    repo = ROOT/'vc_models/meanvc2/repo'
    signature = digest(ROOT/'models/meanvc2_ref60/runtime.json') + 'naturalness-embedding-v1' + ''.join(
        digest(OUT/'reference'/f'{name}.wav') for name in ('calm','lower','bright'))
    stamp=OUT/'metadata/embedding_package.json'
    if stamp.exists():
        cached=json.loads(stamp.read_text('utf-8'))
        if cached.get('signature')==signature and all(Path(p).is_file() and digest(p)==h for p,h in cached.get('files',{}).items()):
            print('EMBEDDINGS CACHE',flush=True);return
    sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'src/infer'))
    package = types.ModuleType('src.model');package.__path__=[str(repo/'src/model')];sys.modules['src.model']=package
    spec = importlib.util.spec_from_file_location('natural_ref_upstream',repo/'src/infer/infer_e2e.py')
    official = importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    torch.set_num_threads(1)
    encoder = official.load_spk_model('cpu')
    vectors = []; parts = []
    selection = json.loads((OUT/'metadata/reference_selection.json').read_text('utf-8'))
    for name in ('calm','lower','bright'):
        wave, _ = sf.read(OUT/'reference'/f'{name}.wav',dtype='float32')
        wave -= wave.mean()
        with torch.inference_mode(): vector = encoder(torch.from_numpy(wave)[None]).squeeze().numpy()
        if vector.shape != (256,) or not np.isfinite(vector).all(): raise ValueError('Invalid embedding')
        vectors.append(vector);parts.append(wave)
    combined, keep = centroid(vectors)
    for name, vector, wave in list(zip(('calm','lower','bright'),vectors,parts))+[('centroid',combined,np.concatenate(parts))]:
        folder = PACK/('natural_'+name);folder.mkdir(parents=True,exist_ok=True)
        sf.write(folder/'reference.wav',wave,16000,subtype='PCM_16')
        np.save(folder/'fixed_embedding.npy',vector,allow_pickle=False)
        runtime = json.loads((ROOT/'models/meanvc2_ref60/runtime.json').read_text('utf-8'))
        runtime['assets'] = [a for a in runtime['assets'] if a['path'].startswith('vc_models/')]
        runtime['assets'] += [dict(path=(folder/p).relative_to(ROOT).as_posix(),sha256=digest(folder/p))
                              for p in ('reference.wav','fixed_embedding.npy')]
        runtime.update(vocoder_batch_frames=36,display_name='自然さ比較 '+name,human_approved_offline=False,
                       reference_origin=selection,reference_embedding_method='single bounded5s' if name!='centroid' else 'trimmed centroid',
                       human_approved_blind_ids=[],approved_variant='UNRATED')
        save(folder/'runtime.json',runtime)
        save(folder/'profile.json',dict(schema=1,status='COMPLETE',name={'calm':'中間域Reference','lower':'低め域Reference','bright':'高め域Reference','centroid':'近傍平均Reference'}[name],
                                       embedding_sha256=digest(folder/'fixed_embedding.npy'),retained_indices=keep if name=='centroid' else None))
    save(stamp,dict(signature=signature,files={str(p):digest(p) for p in PACK.glob('*/*') if p.is_file()}))
    print('EMBEDDINGS COMPLETE',flush=True)


def convert():
    import numpy as np
    import soundfile as sf
    from src.vc.meanvc2 import MeanVC2Backend
    from src.vc.phrase_prosody import HOP,HISTORY,LOOKAHEAD,repair_window
    from src.vc.resampler import StreamingResampler
    from src.prosody.f0 import YinEstimator
    refs = {'current':ROOT/'models/meanvc2_ref60',**{k:PACK/('natural_'+k) for k in ('calm','lower','bright','centroid')}}
    cached=[]
    for ref,folder in refs.items():
        for name,path in sources().items():
            stamp=OUT/'metadata'/f'{ref}_{name}.json'
            signature=digest(path)+digest(folder/'fixed_embedding.npy')+digest(ROOT/'src/vc/phrase_prosody.py')+digest(ROOT/'src/vc/meanvc2.py')
            if not stamp.exists():continue
            old=json.loads(stamp.read_text('utf-8'))
            if old.get('signature')==signature and len(old.get('rows',[]))==3 and all(
                    Path(r['output']).exists() and digest(r['output'])==r['sha256'] for r in old['rows']):
                cached.extend(old['rows'])
    if len(cached)==60:
        save(OUT/'metadata/results.json',dict(status='READY FOR HUMAN LISTENING',rows=cached))
        print('CACHE ALL 60; no model loaded',flush=True);return
    load_start=time.perf_counter()
    b = MeanVC2Backend(1);b.load(ROOT/'models/meanvc2_ref60');b.vocoder_batch_frames=36
    model_load_seconds=time.perf_counter()-load_start
    estimator=YinEstimator()
    refs = {'current':ROOT/'models/meanvc2_ref60',**{k:PACK/('natural_'+k) for k in ('calm','lower','bright','centroid')}}
    rows = []
    for ref,folder in refs.items():
        vector = np.load(folder/'fixed_embedding.npy',allow_pickle=False)
        b.speaker=b.torch.from_numpy(vector.copy())[None].float()
        with b.torch.inference_mode(): memory=b.speaker_memory_forward(b.speaker)
        b.vc.gtm.forward=lambda _, value=memory:value
        for name,path in sources().items():
            stamp=OUT/'metadata'/f'{ref}_{name}.json'
            signature=digest(path)+digest(folder/'fixed_embedding.npy')+digest(ROOT/'src/vc/phrase_prosody.py')+digest(ROOT/'src/vc/meanvc2.py')
            if stamp.exists():
                old=json.loads(stamp.read_text('utf-8'))
                if old.get('signature')==signature and old.get('rows') and all(Path(r['output']).exists() and digest(r['output'])==r['sha256'] for r in old['rows']):
                    rows.extend(old['rows']);print('CACHE',ref,name,flush=True);continue
            wave,rate=sf.read(path,dtype='float32');assert rate==48000
            down=StreamingResampler(48000,16000)
            padded=np.pad(wave,(0,(-len(wave))%7680+30*7680))
            audio=np.concatenate([down.process(padded[i:i+7680]) for i in range(0,len(padded),7680)])
            b.reset();parts=[];times=[]
            for at in range(0,len(audio),HOP):
                start=time.perf_counter();parts.append(b.process_chunk(audio[at:at+HOP]));times.append(time.perf_counter()-start)
            base=np.concatenate(parts);alignment=b.algorithmic_buffer_ms*16+640
            aligned=np.pad(audio,(alignment,0))[:len(base)];n=round(len(wave)/3)
            local=[]
            for mode in ('none','energy','combined'):
                result=[];stats=[]
                if mode=='none':converted=base[alignment:alignment+n]
                else:
                    for at in range(0,len(base),HOP):
                        def window(x):
                            lo,hi=at-HISTORY,at+LOOKAHEAD
                            z=np.zeros(hi-lo,np.float32);begin,end=max(lo,0),min(hi,len(x))
                            if end>begin:z[begin-lo:end-lo]=x[begin:end]
                            return z
                        edited,info=repair_window(window(aligned),window(base),HISTORY,True,mode=='combined')
                        result.append(edited);stats.append(info)
                    converted=np.concatenate(result)[alignment:alignment+n]
                dest=OUT/'raw'/f'{ref}_{name}_{mode}.wav'
                sf.write(dest,converted,16000,subtype='PCM_16')
                # Boundary derivative ratio is diagnostic, not a naturalness score.
                derivative=abs(np.diff(converted));edges=np.arange(HOP-alignment%HOP,len(converted)-1,HOP)
                f=np.array([estimator.estimate(converted[p-1280:p]) for p in range(1280,len(converted)+1,640)])
                voiced=(f[:,1]>=.88)&(f[:,0]>=80)&(f[:,0]<=550)&(f[:,2]>-48)
                consecutive=voiced[1:]&voiced[:-1]
                jumps=abs(12*np.log2(np.maximum(f[1:,0],1)/np.maximum(f[:-1,0],1)))
                row=dict(reference=ref,source=name,mode=mode,output=str(dest),sha256=digest(dest),
                         audio_seconds=n/16000,generation_seconds=sum(times),rtf_including_flush=sum(times)/(len(audio)/16000),
                         model_load_seconds=model_load_seconds,
                         voiced_f0_quantiles_hz=np.quantile(f[voiced,0],[.1,.5,.9]).tolist() if voiced.any() else [],
                         voiced_pitch_jumps_over_5st=int(np.sum(consecutive&(jumps>5))),
                         analysis_seconds=sum(s['analysis_ms'] for s in stats)/1000,
                         peak=float(np.max(abs(converted))),clipping=float(np.mean(abs(converted)>=1)),
                         boundary_derivative_p95=float(np.percentile(derivative[edges],95)) if len(edges) else 0,
                         all_derivative_p95=float(np.percentile(derivative,95)),
                         pitch_regions=sum(s['pitch_regions'] for s in stats),
                         model_buffer_ms=b.algorithmic_buffer_ms,grid_delay_ms=40,
                         extra_delay_ms=1600 if mode!='none' else 0,
                         quality='UNRATED',kind='OFFLINE synchronous repair; realtime deadline behavior tested separately')
                local.append(row);rows.append(row)
            save(stamp,dict(signature=signature,rows=local));print('CONVERTED',ref,name,flush=True)
    b.unload();save(OUT/'metadata/results.json',dict(status='READY FOR HUMAN LISTENING',rows=rows))


def blind():
    import numpy as np
    import soundfile as sf
    from src.prosody.f0 import YinEstimator
    from tools.vc_tournament.audio import normalize
    result=json.loads((OUT/'metadata/results.json').read_text('utf-8'))
    seed_path=OUT/'metadata/seed_v2_result.json'
    if seed_path.exists() and not any(r.get('mode')=='seed_v2' for r in result['rows']):
        seed=json.loads(seed_path.read_text('utf-8'))
        if seed.get('status')=='SUCCESS' and Path(seed['output']).is_file() and digest(seed['output'])==seed['output_sha256']:
            wave,rate=sf.read(seed['output'],dtype='float32')
            result['rows'].append(dict(reference='current',source='normal',mode='seed_v2',output=seed['output'],
                sha256=seed['output_sha256'],audio_seconds=len(wave)/rate,rtf_including_flush=seed['rtf'],
                generation_seconds=seed['generation_seconds'],analysis_seconds=0,
                peak=float(np.max(abs(wave))),clipping=float(np.mean(abs(wave)>=1)),
                model_buffer_ms=None,grid_delay_ms=None,extra_delay_ms=None,quality='UNRATED',
                model='Seed-VC V2',kind='OFFLINE quality ceiling; not integrated in realtime app'))
    estimator=YinEstimator()
    for row in result['rows']:
        if 'voiced_f0_quantiles_hz' in row:continue
        audio,rate=sf.read(row['output'],dtype='float32')
        if rate!=16000:
            from scipy.signal import resample_poly
            import math
            divisor=math.gcd(rate,16000);audio=resample_poly(audio,16000//divisor,rate//divisor)
        f=np.array([estimator.estimate(audio[p-1280:p]) for p in range(1280,len(audio)+1,640)])
        voiced=(f[:,1]>=.88)&(f[:,0]>=80)&(f[:,0]<=550)&(f[:,2]>-48)
        jumps=abs(12*np.log2(np.maximum(f[1:,0],1)/np.maximum(f[:-1,0],1)))
        row.update(voiced_f0_quantiles_hz=np.quantile(f[voiced,0],[.1,.5,.9]).tolist() if voiced.any() else [],
                   voiced_pitch_jumps_over_5st=int(np.sum(voiced[1:]&voiced[:-1]&(jumps>5))))
    save(OUT/'metadata/results.json',result)
    for name in sources():
        normalize(OUT/'raw'/f'current_{name}_combined.wav', OUT/'blind'/f'current_{name}.wav')
    rows=list(result['rows'])
    manifest_path=OUT/'metadata/blind_manifest.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text('utf-8'))['candidates']
        order={(r['reference'],r['source'],r['mode']):i for i,r in enumerate(previous)}
        rows.sort(key=lambda r:order.get((r['reference'],r['source'],r['mode']),len(order)))
    else:
        random.Random(51005).shuffle(rows)
    public=[];private=[]
    for i,row in enumerate(rows,1):
        ident=f'N{i:03}';dest=OUT/'blind'/f'{ident}.wav'
        correction=normalize(row['output'],dest)
        reference=ROOT/'models/meanvc2_ref60/reference.wav' if row['reference']=='current' else PACK/('natural_'+row['reference'])/'reference.wav'
        target=OUT/'reference'/f"target_{row['reference']}.wav"
        if row['mode']=='seed_v2':
            reference=OUT/'reference/seed_v2_reference05.wav'
            target=OUT/'reference'/f'candidate_{ident}.wav'
        shutil.copy2(reference,target)
        public.append(dict(id=ident,audio=dest.name,source=f"../source/{row['source']}.wav",reference='../reference/target_current.wav',
                           candidate_reference='../reference/'+target.name,source_kind=row['source'],
                           audio_sha256=digest(dest),source_sha256=digest(sources()[row['source']]),reference_sha256=digest(target)))
        private.append(dict(id=ident,**row,normalization=correction))
    package=hashlib.sha256(json.dumps(public,sort_keys=True).encode()).hexdigest()
    save(OUT/'metadata/blind_manifest.json',dict(package=package,candidates=private))
    template=(ROOT/'tools/vc_tournament/listening.html').read_text('utf-8')
    template=template.replace('__DATA__',json.dumps(dict(package=package,candidates=public),ensure_ascii=False).replace('<','\\u003c'))
    template=template.replace('v0.11 Blind Voice Comparison','人間らしさ・幼さの比較')
    template=template.replace('優先順位：Targetへの類似 → 自然さ → 元声の少なさ → 女声らしさ → 機械感の少なさ。速度は別評価です。',
        '優先順位：普通に人が喋っている自然さ → 幼さ・無理な高音感の少なさ → 好きな今の声の維持。速度は別評価です。')
    template=template.replace("['mechanical','機械感']","['mechanical','機械感'],['childlike','幼さ・無理な高音感'],['voice_changed','好きな今の声からの変化']")
    template=template.replace('元声残り・機械感：5が多い','元声残り・機械感・幼さ・声の変化：5が多い')
    template=template.replace('v011-ratings.json','naturalness-ratings.json')
    template=template.replace("[['Candidate',c.audio],['Target Reference',c.reference],['Source',c.source]]",
        "[['Candidate',c.audio],['現在の声','current_'+c.source_kind+'.wav'],['目標Reference',c.reference],['候補Reference',c.candidate_reference],['Source',c.source]]")
    template=template.replace('<main id="candidates">','<label>録音 <select id="sourcefilter"><option value="long">長文</option><option value="normal">通常</option><option value="low">低め</option><option value="bright">明るめ</option><option value="">全て</option></select></label><main id="candidates">')
    template=template.replace('for(const c of data.candidates){',"for(const c of data.candidates.filter(c=>!document.getElementById('sourcefilter').value||c.source_kind===document.getElementById('sourcefilter').value)){")
    template=template.replace('render();\n</script>',"document.getElementById('sourcefilter').onchange=render;\nrender();\n</script>")
    (OUT/'blind/index.html').write_text(template,'utf-8')
    save(OUT/'metadata/listening_focus.json',dict(
        instructions=['語頭の子音から母音への移り変わり','弱い語尾の抜け','長文の途中で音色が急に変わらないか',
                      '幼さ・無理に高くした感じ','今の好きな声を保っているか'],
        numeric_boundary_diagnostics='Not a quality ranking; use to locate candidate boundaries for listening',
        synthetic_emotions_added=False,source_limit='Uses existing four recordings; no new small-voice/laughter performance was recorded'))
    print('BLIND',len(public),flush=True)


def guarded(stage, python):
    import psutil
    from tools.compare_meanvc2_continuity import require_idle_voice_worker
    require_idle_voice_worker()
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',HF_HUB_DISABLE_XET='1')
    for key,sub in {'TEMP':'temp','TMP':'temp','HF_HOME':'hf-home','HF_HUB_CACHE':'hf','TORCH_HOME':'torch','XDG_CACHE_HOME':'xdg','NUMBA_CACHE_DIR':'numba','MPLCONFIGDIR':'matplotlib'}.items():
        env[key]=str(ROOT/'vc_models/cache'/sub)
    env['HF_HUB_OFFLINE']='0' if stage=='seed' else '1'
    log=OUT/'logs'/f'{stage}_{time.time_ns()}.log'
    started=time.monotonic();peak=0;reason=None
    with log.open('w',encoding='utf-8') as stream:
        p=subprocess.Popen([str(python),'-u',str(Path(__file__).resolve()),'--stage',stage],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        proc=psutil.Process(p.pid)
        while p.poll() is None:
            try:
                tree=[proc]+proc.children(recursive=True)
                peak=max(peak,sum(q.memory_info().rss for q in tree if q.is_running()))
                if peak>4.5*1024**3:reason='PROCESS_TREE_RAM_LIMIT_4.5GiB'
                elif psutil.virtual_memory().available<1.5*1024**3:reason='AVAILABLE_RAM_BELOW_1.5GiB'
                elif shutil.disk_usage(ROOT).free<3*1024**3:reason='DISK_RESERVE_3GiB'
                elif time.monotonic()-started>(1800 if stage=='seed' else 1200):reason='TIMEOUT'
                if reason:
                    for q in reversed(tree):
                        try:q.kill()
                        except psutil.Error:pass
                    break
            except psutil.NoSuchProcess:pass
            time.sleep(.2)
        p.wait()
    row=dict(stage=stage,status='SUCCESS' if p.returncode==0 and not reason else 'FAILED',reason=reason,
             peak_ram_bytes=peak,wall_seconds=time.monotonic()-started,returncode=p.returncode,log=str(log),
             last_error=log.read_text('utf-8',errors='replace')[-3000:] if reason or p.returncode else None)
    stamp=OUT/'metadata'/f'{stage}_attempt.json'
    if stamp.exists():
        archive=OUT/'metadata/attempt_history'/f'{stage}_{time.time_ns()}.json'
        archive.parent.mkdir(exist_ok=True)
        shutil.copy2(stamp,archive)
    save(stamp,row)
    print(stage,row['status'],row['reason'],flush=True)
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['prepare','embeddings','convert','blind','seed'])
    parser.add_argument('--skip-seed',action='store_true');args=parser.parse_args()
    for folder in ('reference','source','raw','blind','metadata','logs'):(OUT/folder).mkdir(parents=True,exist_ok=True)
    if args.stage:
        import psutil
        if os.name=='nt':
            p=psutil.Process();p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);p.cpu_affinity(p.cpu_affinity()[:1])
        if args.stage=='seed':
            from tools.naturalness_seed_v2 import run
            run()
        else:globals()[args.stage]()
        return
    stages=[]
    python=ROOT/'vc_models/meanvc2/.venv/Scripts/python.exe'
    for stage in ('prepare','embeddings','convert'):
        row=guarded(stage,python);stages.append(row)
        if row['status']!='SUCCESS':break
    if not args.skip_seed:stages.append(guarded('seed',ROOT/'vc_models/seedvc/.venv/Scripts/python.exe'))
    if any(r['stage']=='convert' and r['status']=='SUCCESS' for r in stages):
        stages.append(guarded('blind',python))
    save(OUT/'metadata/campaign.json',dict(stages=stages,status='READY FOR HUMAN LISTENING' if any(r['stage']=='blind' and r['status']=='SUCCESS' for r in stages) else 'INCOMPLETE'))
    if any(r['stage']=='blind' and r['status']=='SUCCESS' for r in stages):
        from tools.report_naturalness_campaign import main as report
        report()


if __name__=='__main__':main()
