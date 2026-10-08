"""Resumable offline MeanVC2 quality comparison. No devices, training or downloads."""
import argparse,hashlib,importlib.util,json,os,random,shutil,sys,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'recordings/v011_meanvc2_improvements'

THREADS=1
VARIANT=None

def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def bound_worker():
    import psutil
    process=psutil.Process()
    if os.name=='nt':
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        process.cpu_affinity(process.cpu_affinity()[:THREADS])
    os.environ['OMP_NUM_THREADS']=str(THREADS)
    os.environ['MKL_NUM_THREADS']=str(THREADS)
def save(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and data.get('status')=='RUNNING':
        previous=read(p)
        if previous.get('status') in ('FAILED','RUNNING'):
            history=OUT/'metadata/attempt_history';history.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,history/(p.parent.name+'_'+p.stem+'_'+str(time.time_ns())+'.json'))
    temporary=p.with_suffix(p.suffix+'.tmp');temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8');temporary.replace(p)
def read(p):return json.loads(Path(p).read_text('utf-8'))

def verify():
    import numpy as np,torch
    from src.vc.meanvc2 import MeanVC2Backend
    b=MeanVC2Backend(THREADS);b.load(ROOT/'models/meanvc2_120')
    torch.manual_seed(110)
    audio=np.random.default_rng(110).normal(0,.05,2560*8).astype(np.float32)
    full=b.kaldi.fbank(torch.from_numpy(audio*32768)[None],frame_length=25,frame_shift=10,snip_edges=True,
                       num_mel_bins=80,energy_floor=0.,dither=0.,sample_frequency=16000)
    b.feature_frontend='aligned';b.reset()
    pieces=[b._extract_streaming_fbank(audio[i:i+2560]) for i in range(0,len(audio),2560)]
    streaming=torch.cat(pieces)
    fbank_error=float((streaming-full).abs().max())
    assert streaming.shape==full.shape and torch.allclose(streaming,full,atol=1e-5,rtol=1e-5)
    seen=[]
    def fake_asr(window,*args):
        seen.append(window[0].clone());return torch.zeros(1,4,256),args[-2],args[-1]
    b.asr=fake_asr;b.reset()
    for i in range(0,len(audio),2560):b._append_features(audio[i:i+2560])
    window_errors=[float((w-full[i*16:i*16+19]).abs().max()) for i,w in enumerate(seen)]
    assert max(window_errors)<1e-4
    b.bn_interpolation='fixed_linear';b.reset()
    b._append_bn(torch.arange(4).float()[None,:,None].expand(1,4,256))
    b._append_bn(torch.arange(4,8).float()[None,:,None].expand(1,4,256))
    expected=torch.arange(3,7,.25)
    assert torch.equal(b.cond[0,16:,0],expected)
    # Verify cropped rolling decode against full decode with finite temporal
    # receptive field. No reduction of context is accepted without this check.
    mel=torch.rand(1,80,240)*.8+.1
    with torch.inference_mode():
        full_audio=b.vocos.decode(mel).squeeze().numpy()
        errors={};times={}
        for context in (36,32):
            pieces=[];start=0;begin=time.perf_counter()
            for total in range(12,241,12):
                end=total-context
                if end<=start:continue
                origin=max(0,start-context)
                decoded=b.vocos.decode(mel[:,:,origin:total]).squeeze().numpy()
                pieces.append(decoded[(start-origin)*160:(end-origin)*160]);start=end
            result=np.concatenate(pieces)
            errors[str(context)]=float(np.max(np.abs(result-full_audio[:len(result)])))
            times[str(context)]=time.perf_counter()-begin
    accept32=errors['32']<1e-5 and times['32']<times['36']
    result=dict(status='SUCCESS',backend_sha256=sha(ROOT/'src/vc/meanvc2.py'),input_kind='Synthetic tensors; no devices',fbank_max_error=fbank_error,
                asr_window_max_error=max(window_errors),asr_windows=len(seen),fixed_grid_verified=True,
                vocoder_full_decode_max_errors=errors,vocoder_decode_seconds=times,
                accepted_vocoder_context=32 if accept32 else 36)
    save(OUT/'metadata/verification.json',result);print(json.dumps(result),flush=True)

def embeddings():
    import numpy as np,soundfile as sf,torch
    from scipy.signal import resample_poly
    repo=ROOT/'vc_models/meanvc2/repo'
    sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'src/infer'))
    package=types.ModuleType('src.model');package.__path__=[str(repo/'src/model')];sys.modules['src.model']=package
    spec=importlib.util.spec_from_file_location('meanvc2_quality_official',repo/'src/infer/infer_e2e.py')
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    torch.set_num_threads(THREADS)
    refs=ROOT/'recordings/v011_mega_tournament/reference'
    folder=OUT/'reference';folder.mkdir(parents=True,exist_ok=True)
    source_ids={'ref05':('reference_05s.wav',[0]),'ref20':('reference_20s.wav',[0,7.5,15]),
                'ref60':('reference_60s.wav',[0,27.5,55])}
    encoder=None;rows=[]
    model_hashes=sha(repo/'preprocess/ckpts/wavlm_large_finetune.pth')+sha(repo/'preprocess/ckpts/wavlm_large.pt')
    provenance=OUT/'metadata/resume_provenance.json'
    legacy_tool=read(provenance)['legacy_tool_sha256'] if provenance.exists() else ''
    for name,(filename,starts) in source_ids.items():
        reference=refs/filename
        signature=sha(reference)+model_hashes+'segments5s-v1'
        legacy_signature=sha(reference)+model_hashes+legacy_tool+'segments5s-v1'
        stamp=folder/(name+'.json');dest=folder/(name+'.npy')
        if stamp.exists() and dest.exists() and read(stamp).get('signature') in (signature,legacy_signature) and read(stamp).get('embedding_sha256')==sha(dest):
            rows.append(read(stamp));print('EMBEDDING CACHE '+name,flush=True);continue
        if encoder is None:
            print('LOADING FIXED SPEAKER ENCODER',flush=True);encoder=official.load_spk_model('cpu')
        audio,sr=sf.read(reference,dtype='int16');audio=audio.astype(np.float32)/32767
        parts=[];cache=folder/'segments';cache.mkdir(exist_ok=True)
        for offset in starts:
            segment=audio[round(offset*sr):round((offset+5)*sr)]
            wave=resample_poly(segment,1,3) if sr==48000 else __import__('librosa').resample(segment,orig_sr=sr,target_sr=16000)
            key=hashlib.sha256(wave.tobytes()+signature.encode()).hexdigest()
            piece=cache/(key+'.npy')
            legacy_piece=cache/(hashlib.sha256(wave.tobytes()+legacy_signature.encode()).hexdigest()+'.npy')
            if not piece.exists() and legacy_piece.exists():shutil.copy2(legacy_piece,piece)
            if piece.exists():v=np.load(piece,allow_pickle=False)
            else:
                print(f'EMBEDDING {name} {offset}s',flush=True)
                with torch.inference_mode():v=encoder(torch.from_numpy(wave.copy())[None].float()).squeeze().numpy()
                if v.shape!=(256,) or not np.isfinite(v).all():raise ValueError('Invalid speaker vector')
                np.save(piece,v)
            if v.shape!=(256,) or not np.isfinite(v).all():raise ValueError('Invalid cached speaker vector')
            parts.append(v)
        # Direction averaging of one speaker, then restore the median original
        # norm. This avoids changing GTM input scale merely through averaging.
        vectors=np.stack(parts);norms=np.linalg.norm(vectors,axis=1)
        unit=vectors/np.maximum(norms[:,None],1e-8)
        centre=unit.mean(axis=0);centre/=max(np.linalg.norm(centre),1e-8)
        distances=unit@centre
        # With three segments retain the two most mutually central directions.
        keep=np.argsort(distances)[-min(2,len(parts)):]
        centre=unit[keep].mean(axis=0);centre/=max(np.linalg.norm(centre),1e-8)
        result=(centre*np.median(norms[keep])).astype(np.float32)
        np.save(dest,result);shutil.copy2(reference,folder/(name+'.wav'))
        row=dict(name=name,signature=signature,reference_sha256=sha(reference),embedding_sha256=sha(dest),
                 offsets_seconds=starts,retained_indices=keep.tolist(),segment_cosines=distances.tolist(),
                 method='5s bounded windows; trimmed unit-vector centroid; median norm restored; no training')
        save(stamp,row);rows.append(row)
        del encoder;encoder=None
    save(OUT/'metadata/embeddings.json',dict(status='SUCCESS',references=rows))

def convert():
    import numpy as np,soundfile as sf,torch
    from scipy.signal import resample_poly
    from src.vc.meanvc2 import MeanVC2Backend
    verification=read(OUT/'metadata/verification.json')
    context=verification['accepted_vocoder_context']
    b=MeanVC2Backend(THREADS);b.load(ROOT/'models/meanvc2_120')
    original=np.load(ROOT/'models/meanvc2_120/fixed_embedding.npy',allow_pickle=False)
    variants=[('baseline','baseline','legacy','legacy'),('fixed_grid','baseline','legacy','fixed_linear'),
              ('aligned_asr','baseline','aligned','legacy'),('aligned_fixed','baseline','aligned','fixed_linear'),
              ('ref05','ref05','legacy','legacy'),('ref20','ref20','legacy','legacy'),('ref60','ref60','legacy','legacy'),
              ('ref20_aligned','ref20','aligned','fixed_linear'),('ref60_aligned','ref60','aligned','fixed_linear')]
    for ident,ref,frontend,interpolation in variants:
        if VARIANT and ident!=VARIANT:continue
        vector=original if ref=='baseline' else np.load(OUT/'reference'/(ref+'.npy'),allow_pickle=False)
        b.speaker=torch.from_numpy(vector.copy())[None].float()
        with torch.inference_mode():memory=b.speaker_memory_forward(b.speaker)
        b.vc.gtm.forward=lambda _,value=memory:value
        b.feature_frontend=frontend;b.bn_interpolation=interpolation
        b.vocoder_left=b.vocoder_right=context
        for name in ('normal','low','bright'):
            source=ROOT/f'recordings/v011_mega_tournament/source/source_{name}.wav'
            folder=OUT/'raw'/ident;folder.mkdir(parents=True,exist_ok=True)
            dest=folder/f'source_{name}.wav';stamp=dest.with_suffix('.json')
            config=dict(measurement_schema=2,variant=ident,reference=ref,frontend=frontend,interpolation=interpolation,
                        vocoder_context=context,steps=2,threads=THREADS,source_sha256=sha(source),
                        embedding_sha256=hashlib.sha256(vector.tobytes()).hexdigest(),
                        backend_sha256=sha(ROOT/'src/vc/meanvc2.py'),conversion_algorithm='quality-grid-v1')
            signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
            if stamp.exists() and dest.exists() and read(stamp).get('signature')==signature and read(stamp).get('output_sha256')==sha(dest):
                print('CACHE '+ident+' '+name,flush=True);continue
            save(stamp,dict(status='RUNNING',signature=signature,config=config))
            try:
                print('CONVERT '+ident+' '+name,flush=True)
                # A variant change must reset the old mode's pending buffer
                # before warmup; warmup itself resets only after its replay.
                b.reset();b.warmup();audio,sr=sf.read(source,dtype='float32')
                if sr!=48000 or audio.ndim!=1:raise ValueError('Prepared source must be 48kHz mono')
                audio=resample_poly(audio,1,3).astype(np.float32);actual_samples=len(audio)
                n=(len(audio)+2559)//2560;audio=np.pad(audio,(0,n*2560-len(audio)))
                outputs=[];times=[];started_cpu=time.process_time();started=time.perf_counter()
                for i in range(n):
                    start=time.perf_counter();outputs.append(b.process_chunk(audio[i*2560:(i+1)*2560]));times.append(time.perf_counter()-start)
                for _ in range(6):outputs.append(b.process_chunk(np.zeros(2560,dtype=np.float32)))
                delay=b.algorithmic_buffer_ms*16+(640 if interpolation=='fixed_linear' else 0)
                elapsed=time.perf_counter()-started
                result=np.concatenate(outputs)[delay:delay+actual_samples]
                assert len(result)==actual_samples and np.isfinite(result).all()
                sf.write(dest,result,16000,subtype='FLOAT')
                import psutil
                report=dict(status='SUCCESS',signature=signature,config=config,tool_sha256=sha(Path(__file__)),source=str(source),output=str(dest),output_sha256=sha(dest),
                            audio_seconds=actual_samples/16000,output_seconds=len(result)/16000,
                            generation_seconds=sum(times),rtf=sum(times)/(actual_samples/16000),
                            generation_with_flush_seconds=elapsed,rtf_with_flush=elapsed/(actual_samples/16000),
                            model_load_seconds=b.stats['load_seconds'],
                            chunk_p95_ms=float(np.percentile(times,95)*1000),chunk_max_ms=max(times)*1000,
                            model_buffer_ms=b.algorithmic_buffer_ms,interpolation_grid_delay_ms=40 if interpolation=='fixed_linear' else 0,
                            rms=float(np.sqrt(np.mean(result.astype('float64')**2))),clipping_samples=int(np.sum(np.abs(result)>=.999)),
                            process_ram_bytes=psutil.Process().memory_info().rss,cpu_seconds=time.process_time()-started_cpu,
                            cpu_percent=(time.process_time()-started_cpu)/elapsed*100,
                            input_kind='Offline source WAV; no audio devices',human_quality_accepted=False)
                save(stamp,report)
            except Exception as error:
                save(stamp,dict(status='FAILED',signature=signature,config=config,error=str(error),
                                calls=b.calls,pending_audio_samples=len(b.pending_audio)))
                print('FAILED '+ident+' '+name+': '+str(error),flush=True)

def build():
    from tools.vc_tournament.audio import normalize
    rows=[]
    for p in sorted((OUT/'raw').glob('*/*.json')):
        row=read(p)
        if row.get('status')=='SUCCESS' and Path(row['output']).is_file() and sha(row['output'])==row['output_sha256']:rows.append(row)
    shuffled=rows.copy();random.Random(111).shuffle(shuffled)
    public=[];private=[];blind=OUT/'blind';blind.mkdir(parents=True,exist_ok=True)
    for i,row in enumerate(shuffled,1):
        ident=f'C{i:03}';dest=blind/(ident+'.wav');correction=normalize(row['output'],dest)
        private.append(dict(blind_id=ident,variant=row['config']['variant'],source=row['source'],normalization=correction))
        public.append(dict(id=ident,audio=dest.name,source='../source/'+Path(row['source']).name,
                           reference='../reference/reference_10s.wav',sha256=sha(dest)))
    for folder in ('source','reference'):(OUT/folder).mkdir(exist_ok=True)
    for source in (ROOT/'recordings/v011_mega_tournament/source').glob('source_*.wav'):shutil.copy2(source,OUT/'source'/source.name)
    shutil.copy2(ROOT/'models/meanvc2_120/reference.wav',OUT/'reference/reference_10s.wav')
    package=hashlib.sha256(json.dumps(public,sort_keys=True).encode()).hexdigest()
    save(OUT/'metadata/blind_manifest.json',dict(package=package,candidates=private))
    template=(ROOT/'tools/vc_tournament/listening.html').read_text('utf-8')
    data=json.dumps(dict(package=package,candidates=public),ensure_ascii=False).replace('<','\\u003c')
    template=template.replace('__DATA__',data).replace('v0.11 Voice Conversion Blind Tournament','MeanVC2 B003 Improvement Comparison')
    template=template.replace('v0.11 Blind Voice Comparison','MeanVC2 B003 Improvement Comparison').replace("['mechanical','機械感']","['mechanical','機械感'],['childlike','幼い声の感じ']")
    template=template.replace('元声残り・機械感：5が多い','元声残り・機械感・幼い声の感じ：5が多い').replace('v011-ratings.json','meanvc2-improvement-ratings.json')
    template=template.replace('<main id="candidates">','<label>Sourceで絞り込み <select id="sourcefilter"><option value="">全て</option><option value="normal">通常声</option><option value="low">低め</option><option value="bright">明るめ</option></select></label><main id="candidates">')
    template=template.replace('for(const c of data.candidates){',"for(const c of data.candidates.filter(c=>!document.getElementById('sourcefilter').value||c.source.endsWith('source_'+document.getElementById('sourcefilter').value+'.wav'))){")
    template=template.replace('render();\n</script>',"document.getElementById('sourcefilter').onchange=render;\nrender();\n</script>")
    (blind/'index.html').write_text(template,encoding='utf-8')
    save(OUT/'metadata/results.json',dict(candidates=len(rows),rows=rows,package=package))
    failures=[dict(job=p.stem,**read(p)) for p in (OUT/'metadata').glob('*_process.json') if read(p).get('status')!='SUCCESS']
    failures += [dict(job=str(p.relative_to(OUT)),**read(p)) for p in (OUT/'raw').glob('*/*.json') if read(p).get('status')!='SUCCESS']
    history=[dict(archive=p.name,**read(p)) for p in (OUT/'metadata/attempt_history').glob('*.json')]
    lines=['# MeanVC2 B003 improvement comparison','',
           'Status: '+('READY FOR HUMAN LISTENING' if rows else 'PREPARATION INCOMPLETE'),'',
           'No training, downloads, audio-device trial, pitch/formant/EQ or prosody processing. Production B003 reference/runtime remains unchanged. Quality improvements are unverified until human listening.', '',
           'After the user reported another PC reboot, preserved embeddings ref05/ref20 and synthetic verification were resumed. Kernel-Power 41 identifies an unexpected restart, not its cause. Hashing now streams 1 MiB blocks instead of reading entire checkpoints into RAM. Offline workers use one CPU core, below-normal priority, a 4.5 GiB sampled RAM limit and separate processes per variant. These controls do not prove that further system resets are impossible.', '',
           'References: original fixed 10s embedding; 5s/20s/60s references with bounded 5s windows, up to two central normalized vectors averaged with original norm restored. No subjective gender verification was automated.', '',
           'Sources: independently recorded source_normal.wav / source_low.wav / source_bright.wav. Originals preserved. Blind volume correction is DC removal and scalar gain only.', '',
           'Experimental changes: aligned 25ms/10ms fbank framing and ASR windows; fixed 40ms-to-10ms linear feature grid; alternative fixed speaker embeddings. Synthetic framing agrees with whole-wave features within 1.91e-6. Vocoder context stays 36: the 32-frame alternative did not pass the strict numerical check.', '',
           'RTF below is offline generation excluding padding flush; the JSON also records total time with flush. One-core experiment RTF is not the app’s four-core realtime performance. Aligned mode uses 640ms model buffering vs 480ms legacy; fixed interpolation adds 40ms feature delay. Neither end-to-end latency nor live quality was measured here.', '',
           '| Blind ID | Variant | Source | RTF (1 core) | RTF incl. flush | Process RAM MiB |',
           '|---|---|---|---:|---:|---:|']
    ids={(r['variant'],Path(r['source']).name):r['blind_id'] for r in private}
    for row in rows:
        c=row['config'];lines.append(f"| {ids[(c['variant'],Path(row['source']).name)]} | {c['variant']} | {Path(row['source']).stem} | {row['rtf']:.3f} | {row.get('rtf_with_flush',0):.3f} | {row['process_ram_bytes']/1024**2:.1f} |")
    lines+=['','## Current failed / interrupted jobs','',json.dumps(failures,ensure_ascii=False,indent=2),'',
            '## Earlier failed / interrupted attempts (retained after recovery)','',json.dumps(history,ensure_ascii=False,indent=2),'',
            'Blind page: `recordings/v011_meanvc2_improvements/blind/index.html`. Private mapping: `metadata/blind_manifest.json`. Ratings include childlike timbre (5 = more childlike / worse). No automatic winner.', '']
    report=ROOT/'validation/v011/meanvc2_improvement_report.md';report.parent.mkdir(parents=True,exist_ok=True);report.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(candidates=len(rows),page=str(blind/'index.html'))),flush=True)

def main():
    global THREADS,VARIANT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',choices=('verify','embeddings','convert'));parser.add_argument('--report-only',action='store_true');parser.add_argument('--threads',type=int,choices=(1,2),default=1);parser.add_argument('--variant');args=parser.parse_args()
    THREADS=args.threads;VARIANT=args.variant
    if args.worker:
        bound_worker()
        return dict(verify=verify,embeddings=embeddings,convert=convert)[args.worker]()
    if not args.report_only:
        from tools.vc_tournament.process import run
        import psutil
        jobs=[('verify',None),('embeddings',None)]+[('convert',variant) for variant in ('baseline','fixed_grid','aligned_asr','aligned_fixed','ref05','ref20','ref60','ref20_aligned','ref60_aligned')]
        for phase,variant in jobs:
            job=phase+('_'+variant if variant else '')
            stamp=OUT/'metadata'/(job+'_process.json')
            if phase=='verify' and (OUT/'metadata/verification.json').exists() and (OUT/'metadata/verify_process.json').exists() and read(OUT/'metadata/verify_process.json')['status']=='SUCCESS' and read(OUT/'metadata/verification.json').get('backend_sha256')==sha(ROOT/'src/vc/meanvc2.py'):
                print('VERIFICATION CACHE (same backend)',flush=True);continue
            if phase=='convert' and stamp.exists() and read(stamp).get('status')=='SUCCESS':
                import numpy as np
                reference_name=variant.split('_')[0] if variant.startswith('ref') else 'baseline'
                vector_file=ROOT/'models/meanvc2_120/fixed_embedding.npy' if reference_name=='baseline' else OUT/'reference'/(reference_name+'.npy')
                expected_vector=hashlib.sha256(np.load(vector_file,allow_pickle=False).tobytes()).hexdigest() if vector_file.exists() else None
                cases=[OUT/'raw'/variant/f'source_{source}.json' for source in ('normal','low','bright')]
                if all(p.exists() and read(p).get('status')=='SUCCESS' and read(p).get('config',{}).get('measurement_schema')==2 and read(p).get('config',{}).get('conversion_algorithm')=='quality-grid-v1' and read(p).get('config',{}).get('embedding_sha256')==expected_vector and read(p).get('config',{}).get('threads')==THREADS and read(p).get('config',{}).get('backend_sha256')==sha(ROOT/'src/vc/meanvc2.py') and Path(read(p)['output']).exists() and read(p)['output_sha256']==sha(read(p)['output']) and read(p)['config']['source_sha256']==sha(read(p)['source']) for p in cases):
                    print('PROCESS CACHE '+job,flush=True);continue
            if psutil.virtual_memory().available<4*1024**3:
                save(stamp,dict(status='FAILED',reason='FREE_RAM_BELOW_4_GIB'));print(job+' insufficient free RAM',flush=True);continue
            command=[ROOT/'vc_models/meanvc2/.venv/Scripts/python.exe',Path(__file__),'--worker',phase,'--threads',str(THREADS)]
            if variant:command+=['--variant',variant]
            save(stamp,dict(status='RUNNING',threads=THREADS,variant=variant))
            log=OUT/'logs'/(job+'.log')
            if log.exists():
                history=OUT/'logs/history';history.mkdir(parents=True,exist_ok=True)
                shutil.copy2(log,history/(job+'_'+str(time.time_ns())+'.log'))
            result=run(command,ROOT,log,timeout=1800,ram_limit_gb=4.5)
            result['threads']=THREADS;result['variant']=variant;save(stamp,result)
            print(job+' '+result['status'],flush=True)
            time.sleep(2)
    build()

if __name__=='__main__':main()
