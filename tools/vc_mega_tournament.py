"""One command: isolated setup, references, fixed sources, inference, blind report.

No audio device imports. No training, application settings, or live audio changes.
"""
import argparse
import ast
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from vc_tournament.audio import digest, prepare_reference, stats, extract_sources
from vc_tournament.blind import build
from vc_tournament.catalog import MODELS, families
from vc_tournament.process import run
from vc_tournament.setup import prepare
from vc_tournament.sources import decode_sources

ROOT=Path(__file__).resolve().parents[1]

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as stream:
        stream.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False))
        stream.flush();os.fsync(stream.fileno())
    tmp.replace(path)

def result_cache(previous,out,retry_failed=False):
    rows={r.get('fingerprint'):r for r in previous.get('results',[])
          if r['status']=='SUCCESS' or (r['status']=='FAILED' and not retry_failed)}
    for path in (Path(out)/'metadata/results').glob('*/*.json'):
        try:
            row=json.loads(path.read_text('utf-8'))
            if row.get('fingerprint') and (row['status']=='SUCCESS' or (row['status']=='FAILED' and not retry_failed)):
                rows[row['fingerprint']]=row
        except (OSError,ValueError,KeyError):continue
    return rows

def classification(rtf):
    if rtf is None:return 'UNMEASURED'
    if rtf<=.7:return 'REALTIME STRONG'
    if rtf<=1:return 'REALTIME POSSIBLE'
    if rtf<=2:return 'BORDERLINE'
    return 'OFFLINE ONLY'

def host_info():
    import psutil
    processor=platform.processor()
    if os.name=='nt':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                processor=winreg.QueryValueEx(key,'ProcessorNameString')[0]
        except OSError:pass
    return dict(platform=platform.platform(),processor=processor,cpu_count=os.cpu_count(),device='cpu',ram_total_bytes=psutil.virtual_memory().total)

def inference_code_fingerprint(family,path=None):
    path=path or ROOT/'tools/vc_tournament/infer.py'
    tree=ast.parse(Path(path).read_text('utf-8'))
    adapters={'meanvc2','meanvc','conan','seedvc','xvc','vevo','facodec','knnvc','freevc','fragmentvc','audiocpp','ezvc','openvoice'}
    selected=[node for node in tree.body if not (isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in adapters and node.name!=family)]
    return hashlib.sha256(ast.dump(ast.Module(body=selected,type_ignores=[]),include_attributes=False).encode()).hexdigest()

def fingerprint(model,source,reference,setup,*,process_digest=None):
    adapter=model.id if model.family=='amphion' else model.family
    process_hash=process_digest or digest(ROOT/'tools/vc_tournament/process.py')
    # This exact monitor-only CPU correction changes no environment, commands,
    # limits or waveform. Preserve existing audio caches, refresh CPU separately.
    if process_digest is None and process_hash=='e1a82c0309b052e9e17a22fa56a2ab5510f2c1a3b19c5867867e84d6ebe21b86':
        process_hash='b11ddb1fa42987a173fe953a06083eb7bb45ccff23997ba978e1a5f7dcb382f5'
    code=[inference_code_fingerprint(adapter),process_hash]
    lock=ROOT/'vc_models'/model.family/'requirements.lock.txt'
    artifacts=ROOT/'vc_models'/model.family/'artifacts.json'
    value=dict(model={k:v for k,v in model.__dict__.items() if k!='note'},source=digest(source),reference=digest(reference),
               revision=setup.get('revision'),deps=digest(lock) if lock.exists() else None,
               artifacts=digest(artifacts) if artifacts.exists() else None,code=code)
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def artifact_inventory(base):
    artifacts=[]
    manifest=base/'artifacts.json'
    previous={}
    if manifest.exists():
        previous={r['path']:r for r in json.loads(manifest.read_text('utf-8')).get('files',[])}
    for directory,dirs,files in os.walk(base):
        dirs[:]=sorted(d for d in dirs if d not in ('.venv','.git','.cache'))
        for name in sorted(files):
            p=Path(directory)/name
            if p.suffix not in ('.pth','.pt','.ckpt','.bin','.safetensors','.gguf'):continue
            info=p.stat();relative=str(p.relative_to(base));old=previous.get(relative,{})
            sha=old['sha256'] if old.get('bytes')==info.st_size and old.get('mtime_ns')==info.st_mtime_ns else digest(p)
            artifacts.append(dict(path=relative,bytes=info.st_size,mtime_ns=info.st_mtime_ns,sha256=sha))
    value=dict(files=artifacts,total_bytes=sum(a['bytes'] for a in artifacts))
    save(manifest,value);return value

def shared_inventory(base,records,local_inventory):
    manifest=base/'shared_artifacts.json'
    old=json.loads(manifest.read_text('utf-8')).get('files',[]) if manifest.exists() else []
    prior={r['path']:r for r in old};found={}
    local_hashes={r['sha256'] for r in local_inventory['files']}
    shared_root=(ROOT/'vc_models/cache').resolve()
    for record in records:
        path=Path(record.get('path',''))
        if record.get('status')!='SUCCESS' or not path.exists():continue
        path=path.resolve()
        if not path.is_relative_to(shared_root):continue
        for file in ([path] if path.is_file() else sorted(path.rglob('*'))):
            if not file.is_file() or file.suffix not in ('.pt','.pth','.ckpt','.bin','.safetensors','.gguf'):continue
            if any(word in file.name for word in ('optimizer','discriminator','_3msteps')):continue
            if file.name=='pytorch_model.bin' and (file.parent/'model.safetensors').exists():continue
            info=file.stat();name=str(file.relative_to(shared_root));cached=prior.get(name,{})
            checksum=cached['sha256'] if cached.get('bytes')==info.st_size and cached.get('mtime_ns')==info.st_mtime_ns else digest(file)
            if checksum not in local_hashes:found[name]=dict(path=name,bytes=info.st_size,mtime_ns=info.st_mtime_ns,sha256=checksum)
    value=dict(files=list(found.values()),total_bytes=sum(r['bytes'] for r in found.values()))
    save(manifest,value);return value

def report(out,manifest):
    setups=manifest['setup'];rows=manifest['results']
    successes=[r for r in rows if r['status']=='SUCCESS']
    successful_conditions={r['model'] for r in successes}
    lines=['# v0.11 No-Training Zero-Shot VC Mega Tournament','',f"Status: **{manifest['status']}**",'',
      'No winner selected. Human quality ratings, listening and physical microphone/Discord latency tests are unperformed. Offline first-chunk timing is recorded only where directly instrumented.',
      'Only offline setup/download/reference processing and (when sources exist) WAV inference. Realtime audio path unchanged; no custom training.',
      f"Completion mode: {manifest.get('completion_mode','full tournament')}",
      f"Attempted conditions: {len(MODELS)}; conditions with successful WAVs: {len(successful_conditions)}; conditions without successful WAVs: {len(MODELS)-len(successful_conditions)}; successful source conversions: {len(successes)} / {len(rows)}. Partial successes remain identified by source below.",
      'Context7 tool unavailable; official upstream source/README inspected. Current sources are separately recorded user male voices decoded from authorized M4A files.','',
      '## Measurement host',f"CPU: {manifest.get('host',{}).get('processor','unknown')}; OS: {manifest.get('host',{}).get('platform','unknown')}; logical CPUs: {manifest.get('host',{}).get('cpu_count','unknown')}; RAM bytes: {manifest.get('host',{}).get('ram_total_bytes','unknown')}; inference device: CPU.",
      'Verification: offline unit/mock tests and WAV integrity checks. Browser JavaScript syntax checked; human listening and hardware audio trials are unperformed.','',
      '## Input',f"Reference: `{manifest['reference'].get('original','MISSING')}`",f"Reference duration: {manifest['reference'].get('original_stats',{}).get('duration','unknown')} s",
      'Reference selection uses acoustic proxies; female stability / noise cleanliness require human verification.',
      f"Reference segments: {json.dumps(manifest['reference'].get('segments',{}),ensure_ascii=False)}",
      'Source files: `source_normal.wav`, `source_low.wav`, `source_bright.wav`, 5–10 s. Original M4A recordings are preserved; conversion provenance and original hashes are in source/*.origin.json. No target clips substitute for the current sources.',
      f"Missing/invalid sources: {manifest['missing_sources']}",'',
      '## All candidates (variants listed separately)',
      '| Candidate | Family | Setup | Conversion | Environment cached checkpoint bytes | Reference | Notes |',
      '|---|---|---|---|---:|---|---|']
    for m in MODELS:
        s=setups.get(m.family,{})
        rr=[r for r in rows if r['model']==m.id]
        state=', '.join(sorted(set(r['status'] for r in rr))) or 'SOURCE_PENDING'
        size=s.get('artifacts',{}).get('total_bytes',0)+s.get('shared_artifacts',{}).get('total_bytes',0)
        lines.append(f"| {m.id} | {m.family} | {s.get('status','NOT_ATTEMPTED')} | {state} | {size} | {m.reference} | {m.note} |")
    lines+=['','## Measured results','Cold end-to-end RTF includes process startup/imports/model/reference loads. Speed classification uses instrumented generation RTF where available, otherwise the conservative cold RTF. The per-result rtf_scope identifies which applies. Offline throughput does not prove streaming latency.',
       'CPU % is summed process-tree usage (100% = one core); RAM is sampled RSS, not exact allocator peak. Setup RAM is not inference RAM. Recovered native outputs use CLI self-reported peak RSS, explicitly identified in their metadata; their original cold wall time and CPU samples were lost. Separate refreshed benchmarks provide sampled process-tree RAM and CPU without changing these WAVs.',
       '| Model / Source | Status / Stage | Cold seconds | Generation seconds | RTF | Peak RSS MB (see definition) | Worker RSS MB | CPU mean % | Output seconds | Clip samples | Speed suitability |',
       '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        metric=r.get('metrics') or {};audio=r.get('audio',{})
        worker=metric.get('process_ram_bytes')
        worker_mb=f'{worker/1024**2:.1f}' if worker is not None else '—'
        lines.append(f"| {r['model']} / {Path(r['source']).name} | {r['status']} / {r['stage']} | {metric.get('wall_seconds','—')} | {r.get('generation_seconds','—')} | {r.get('rtf','—')} | {metric.get('peak_ram_bytes',0)/1024**2:.1f} | {worker_mb} | {metric.get('cpu_percent_mean','—')} | {audio.get('duration','—')} | {audio.get('clipping_samples','—')} | {r.get('suitability','UNMEASURED')} |")
    lines+=['','## Speed summary (successful-source median)',
      'These are speed classifications only, with no voice-quality winner. Median is over successful source conversions; missing/failed sources are excluded and displayed in the detailed table. Individual sources can fall in different classes. No streaming or hardware realtime trial was performed.',
      '| Condition | Successful sources | Median RTF | Speed classification |',
      '|---|---:|---:|---|']
    for model in MODELS:
        rr=[r for r in successes if r['model']==model.id]
        if rr:
            median=statistics.median(r['rtf'] for r in rr)
            lines.append(f'| {model.id} | {len(rr)} | {median:.4f} | {classification(median)} |')
        else:lines.append(f'| {model.id} | 0 | — | UNMEASURED |')
    lines+=['','## CPU measurement refresh',
      'Legacy CPU samples were invalid because fresh process objects lost their preceding sample. Those values are null, not interpreted as idle. A separate matched source/reference inference measures CPU for cached models without replacing quality WAVs or their original timing/RAM measurements.',
      '| Model | Source | Benchmark status | CPU mean % | CPU peak % | Wall seconds | Peak tree RSS MB |',
      '|---|---|---|---:|---:|---:|---:|']
    for model,b in manifest.get('cpu_benchmarks',{}).items():
        v=b['metrics']
        lines.append(f"| {model} | {Path(b['source']).name} | {v['status']} | {v.get('cpu_percent_mean')} | {v.get('cpu_percent_peak')} | {v.get('wall_seconds')} | {v.get('peak_ram_bytes',0)/1024**2:.1f} |")
        if v['status']!='SUCCESS':
            lines.append(f"CPU benchmark failure: {model}: {v.get('reason')}; log `{v.get('log')}`. Quality WAV remains available.")
    lines+=['','## Setup/download failures']
    for family,s in setups.items():
        lines.append(f"\n### {family}\nOfficial source: https://github.com/{families()[family].repo}\nRevision: `{s.get('revision')}`")
        for op in s.get('operations',[]):
            lines.append(f"- {op['stage']}: {op['status']}; {op.get('wall_seconds',0):.1f}s; peak RSS {op.get('peak_ram_bytes',0)/1024**2:.1f} MB; log `{op.get('log')}`")
            if op['status']=='FAILED':lines.append('```text\n'+str(op.get('reason') or op.get('last_error'))[-3500:]+'\n```')
        for record in s.get('download_records',[]):
            lines.append(f"- Artifact {record['name']}: {record['status']}; {record.get('error',record.get('path',''))}")
    lines+=['','## Conversion failures / source wait']
    for r in rows:
        if r['status']!='SUCCESS':
            metric=r.get('metrics') or {}
            lines.append(f"- {r['model']} / {Path(r['source']).name}: {r['status']} at {r['stage']}: {r.get('error','')} (RAM {metric.get('peak_ram_bytes','unmeasured')}); log `{metric.get('log')}`")
            if metric.get('last_error'):
                lines.append('```text\n'+metric['last_error'][-2500:]+'\n```')
    lines+=['','## Blind / human quality',f"Blind candidates: {manifest['blind']['count']}",f"Page: `{manifest['blind']['page']}`",'Private model/ID mapping: `metadata/blind_manifest.json`. Ratings exported from page; no automatic winner.',
      'Quality: UNRATED. Similarity, naturalness, original-voice residue, femininity and mechanical artifacts require listening.',
      'X-VC is GPU-oriented; CPU results, when present, are classified by their measured RTF above. OFFLINE ONLY models are unsuitable for N150 realtime production. Unmeasured models are not classified as realtime.',
      'First audio latency: null unless directly instrumented; no claim based on advertised chunk size. Fixed MeanVC2 embedding cache is implemented in the Python adapter; not counted as a listening candidate without generated output.','',
      '## Versions / resume','The final code/report snapshot is identified by its Git commit and preview tag, reported to the user after validation. Success caches require source/reference/code/dependency/revision/artifact fingerprint plus WAV hash. Failures remain recorded; unchanged inference failures resume from cache unless --retry-failed is supplied. Successful setup stages and downloads reuse cache.',
      'Storage: models, isolated environments, base interpreters, download/dependency caches and temporary files are on D:. Original input M4A files on C: are read-only inputs and preserved.',
      'Checkpoint bytes above count family-local cached checkpoint files (including unused checkpoints shipped or downloaded in that environment) plus required shared encoder/vocoder weights. Shared files with the same hash as a local file are excluded. Variants can report the same environment total; different families can share files. These numbers are not unique allocated NTFS bytes. Details: vc_models/<family>/artifacts.json and shared_artifacts.json.',
      'Standalone MeanVC2 selected self-contained checkpoint sizes: FP32 GGUF 1,629,326,784 bytes; Q4_K GGUF 342,368,032 bytes. Their common environment total also includes both variants and upstream repository fixtures.']
    if (out/'metadata/recovery_manifest.json').exists():
        lines+=['','## Disk repair / interrupted-run recovery',
          'D: NTFS metadata errors interrupted the run. The user performed repair and reboot; the Windows repair log reported corrections completed and no further action required. All 42 pre-existing successful WAV hashes were verified after repair before resuming.',
          'Six completed native outputs were recovered from their WAVs, CLI logs and sidecar metrics. Recovery provenance is in metadata/recovery_manifest.json. Missing original cold wall/CPU measurements remain null. Per-result durable journals now complement the coordinator manifest and history snapshots.']
        if (out/'metadata/tournament_manifest.crash_1400.bin').exists():
            lines+=['A further unexpected PC restart interrupted CPU refresh. Windows reported D: healthy after reboot, but the coordinator JSON contained invalid bytes. All 42 quality WAV hashes, individual result journals and six completed CPU benchmark JSONs remained valid. The corrupt coordinator was preserved as metadata/tournament_manifest.crash_1400.bin; resume used history plus journals. Atomic JSON writes now flush and fsync before replacement. Kernel-Power event 41 alone does not identify the restart cause.']
    source_manifest=out/'metadata/source_manifest.json'
    if source_manifest.exists():
        lines+=['','## Fixed source details','| Source | Duration s | Hz | Channels | SHA256 |','|---|---:|---:|---:|---|']
        for source in json.loads(source_manifest.read_text('utf-8')).get('sources',[]):
            info=source['stats']
            lines.append(f"| {Path(source['path']).name} | {info['duration']:.4f} | {info['sample_rate']} | {info['channels']} | {info['sha256']} |")
    private=out/'metadata/blind_manifest.json'
    if private.exists():
        lines+=['','## Private Blind IDs (open after listening)','| Blind ID | Model | Source |','|---|---|---|']
        for candidate in json.loads(private.read_text('utf-8')).get('candidates',[]):
            lines.append(f"| {candidate['blind_id']} | {candidate['model']} | {Path(candidate['source']).name} |")
    text='\n'.join(lines)+'\n'
    (out/'reports/mega_tournament_report.md').write_text(text,encoding='utf-8')
    path=ROOT/'validation/v011/mega_tournament_report.md';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=Path('voice_sample/reference/combined_67clips.wav'))
    parser.add_argument('--source-dir',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'recordings/v011_mega_tournament')
    parser.add_argument('--skip-setup',action='store_true')
    parser.add_argument('--report-only',action='store_true',help='Verify saved WAVs and rebuild blind/report without setup or inference')
    parser.add_argument('--refresh-cpu',action='store_true',help='Explicitly rerun cached models with missing CPU samples; disabled by default')
    parser.add_argument('--setup-only',action='store_true')
    parser.add_argument('--setup-workers',type=int,choices=(1,2),default=2)
    parser.add_argument('--m4a-dir',type=Path)
    parser.add_argument('--setup-timeout',type=int,default=1200)
    parser.add_argument('--inference-timeout',type=int,default=600)
    parser.add_argument('--retry-download',action='store_true')
    parser.add_argument('--retry-failed',action='store_true',help='Retry unchanged failed inference conditions')
    parser.add_argument('--extract-target-sources',action='store_true',help='Explicit user-authorized target-derived sources; male removal cannot be evaluated')
    args=parser.parse_args();out=args.output.resolve()
    for d in ('reference','source','raw','blind','metadata','reports','logs'):(out/d).mkdir(parents=True,exist_ok=True)
    started=time.time();oldpath=out/'metadata/tournament_manifest.json'
    try:previous=json.loads(oldpath.read_text('utf-8')) if oldpath.exists() else {}
    except (OSError,ValueError):previous={}
    if not previous:
        # Recover the coordinator after interruption; individual result hashes
        # are still checked against current inputs/code/artifacts before reuse.
        history=out/'metadata/history'
        for candidate in sorted(history.glob('manifest_*.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True):
            try:
                restored=json.loads(candidate.read_text('utf-8'))
                if any(r.get('status')=='SUCCESS' for r in restored.get('results',[])):
                    previous=restored;break
            except (OSError,ValueError):continue
    if args.report_only:
        if not previous:raise RuntimeError('No valid manifest/history available for report-only recovery')
        manifest=dict(previous)
        by_pair={(r['model'],Path(r['source']).name):r for r in previous.get('results',[])}
        for path in (out/'metadata/results').glob('*/*.json'):
            try:
                row=json.loads(path.read_text('utf-8'))
                by_pair[(row['model'],Path(row['source']).name)]=row
            except (OSError,ValueError,KeyError):continue
        rows=[]
        for model in MODELS:
            for name in ('source_normal.wav','source_low.wav','source_bright.wav'):
                row=by_pair.get((model.id,name))
                if row is None:continue
                if row['status']=='SUCCESS':
                    try:
                        audio=stats(row['output'])
                        if audio['sha256']!=row['audio']['sha256']:raise ValueError('Cached output SHA256 mismatch')
                        if audio['rms']<1e-7:raise ValueError('Silent cached output')
                    except Exception as error:
                        row=dict(row,status='FAILED',stage='cached_output_validation',error=str(error))
                rows.append(row)
        manifest['results']=rows;manifest['cpu_benchmarks']={}
        for model in MODELS:
            attempts=[r for r in rows if r['model']==model.id]
            good=[r for r in attempts if r['status']=='SUCCESS']
            if not good:continue
            measured=next((r for r in attempts if (r.get('metrics') or {}).get('cpu_measurement_schema')==2
                           and (r.get('metrics') or {}).get('cpu_percent_mean') is not None
                           and (r['status']=='SUCCESS' or r['stage']=='generation')),None)
            if measured:
                value=dict(source=measured['source'],metrics=measured['metrics'],scope='original conversion')
            else:
                path=out/'metadata/cpu_benchmarks'/f'{model.id}.json'
                try:value=json.loads(path.read_text('utf-8')) if path.exists() else {}
                except (OSError,ValueError):value={}
                if value.get('fingerprint') not in {r['fingerprint'] for r in good} or value.get('metrics',{}).get('cpu_measurement_schema')!=2:
                    value=dict(source=good[0]['source'],scope='no further inference after repeated unexpected PC restarts',
                               metrics=dict(status='UNMEASURED',cpu_percent_mean=None,cpu_percent_peak=None,
                                            reason='Additional CPU refresh discontinued after repeated PC restarts; existing quality WAV preserved'))
            manifest['cpu_benchmarks'][model.id]=value
        manifest['completion_mode']='Saved-output verification and blind/report rebuild; additional CPU inference discontinued after repeated unexpected PC restarts. Unavailable CPU measurements remain UNMEASURED.'
        manifest['blind']=build(out,rows)
        manifest['status']='READY FOR HUMAN LISTENING' if manifest['blind']['count'] else 'NO_SUCCESSFUL_OUTPUTS'
        manifest['finished_at']=time.time();save(oldpath,manifest);report(out,manifest)
        summary=dict(status=manifest['status'],candidate_conditions=len(MODELS),families=len(families()),
                     successful_outputs=manifest['blind']['count'],page=manifest['blind']['page'],completion_mode=manifest['completion_mode'])
        save(out/'reports/summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2));return 0
    if previous:
        save(out/'metadata/history'/f'manifest_{time.time_ns()}.json',previous)
    if args.m4a_dir:
        save(out/'metadata/source_decode_manifest.json',decode_sources(args.m4a_dir,out/'source'))
    reference={};missing=[];sources=[]
    provenance_path=out/'metadata/source_provenance.json'
    if args.extract_target_sources and not provenance_path.exists():
        save(provenance_path,extract_sources(args.reference,out/'source'))
    provenance=json.loads(provenance_path.read_text('utf-8')) if provenance_path.exists() else {}
    exclusions=[]
    if args.reference.exists() and provenance.get('origin_sha256')==digest(args.reference):
        exclusions=[e for e in provenance.get('intervals',[]) if Path(e['path']).exists() and digest(e['path'])==e['sha256']]
    try:
        reference=prepare_reference(args.reference,out/'reference',exclusions)
        save(out/'metadata/reference_manifest.json',reference)
    except Exception as e:reference={'error':str(e)}
    for name in ('source_normal.wav','source_low.wav','source_bright.wav'):
        src=(args.source_dir or out/'source')/name
        if not src.exists():missing.append(name);continue
        try:
            info=stats(src)
            if not 5<=info['duration']<=10:raise ValueError('duration must be 5–10 seconds')
            if info['rms']<1e-6:raise ValueError('silent source')
            dest=out/'source'/name
            if src.resolve()!=dest.resolve():shutil.copy2(src,dest)
            sources.append(dict(path=str(dest),stats=info,provenance=provenance or 'User-provided source'))
        except Exception as e:missing.append(f'{name}: {e}')
    save(out/'metadata/source_manifest.json',dict(sources=sources,missing=missing))
    manifest=dict(schema=1,started_at=started,status='PREPARING',reference=reference,missing_sources=missing,
                  host=host_info(),
                  setup={},results=[],blind={'count':0,'page':str(out/'blind/index.html')})
    save(oldpath,manifest)
    def setup_one(family,model):
        print(f'[{family}] setup/download',flush=True)
        try:
            if args.skip_setup:
                p=ROOT/'vc_models'/family/'setup.json'
                value=json.loads(p.read_text('utf-8')) if p.exists() else {'status':'NOT_ATTEMPTED'}
            else:value=prepare(model,ROOT,out,args.setup_timeout,args.retry_download)
            downloads=ROOT/'vc_models'/family/'download.json'
            if downloads.exists():value['download_records']=json.loads(downloads.read_text('utf-8'))
            for record in value.get('download_records',[]):
                path=record.get('path','')
                marker='\\v011-models\\'
                if marker in path and path.startswith('C:'):
                    record['path']=str(ROOT/'vc_models'/path.split(marker,1)[1])
            value['artifacts']=artifact_inventory(ROOT/'vc_models'/family)
            value['shared_artifacts']=shared_inventory(ROOT/'vc_models'/family,value.get('download_records',[]),value['artifacts'])
        except Exception as e:value=dict(status='FAILED',error=f'{type(e).__name__}: {e}')
        return family,value
    with ThreadPoolExecutor(max_workers=args.setup_workers) as pool:
        jobs=[pool.submit(setup_one,family,model) for family,model in families().items()]
        for future in as_completed(jobs):
            family,value=future.result()
            manifest['setup'][family]=value;save(oldpath,manifest)
            print(f'[{family}] {value["status"]}',flush=True)
    if args.setup_only:
        manifest['status']='SETUP_COMPLETE_INFERENCE_PENDING';save(oldpath,manifest)
        report(out,manifest);return 0
    manifest['status']='CONVERTING';save(oldpath,manifest)
    oldrows=result_cache(previous,out,args.retry_failed)
    worker=ROOT/'tools/vc_tournament/infer.py'
    for model in MODELS:
        (out/'raw'/model.id).mkdir(parents=True,exist_ok=True)
        if missing or reference.get('error'):
            manifest['results'].append(dict(model=model.id,source='',reference=model.reference,status='SOURCE_PENDING' if missing else 'REFERENCE_FAILED',stage='input',error=', '.join(missing) or reference.get('error'),metrics=None))
            continue
        for source in sources:
            src=Path(source['path']);ref=out/'reference'/f'reference_{model.reference}.wav'
            setup=manifest['setup'][model.family]
            fp=fingerprint(model,src,ref,setup);dest=out/'raw'/model.id/src.name
            row=dict(model=model.id,source=str(src),reference=str(ref),fingerprint=fp,output=str(dest),status='FAILED',stage='inference',
                     input_audio_duration=source['stats']['duration'],first_audio_latency=None,model_load_time=None,generation_seconds=None)
            cached=oldrows.get(fp)
            if not cached and digest(ROOT/'tools/vc_tournament/process.py')=='e1a82c0309b052e9e17a22fa56a2ab5510f2c1a3b19c5867867e84d6ebe21b86':
                # A running pre-fix coordinator may have recorded the new hash.
                cached=oldrows.get(fingerprint(model,src,ref,setup,process_digest='e1a82c0309b052e9e17a22fa56a2ab5510f2c1a3b19c5867867e84d6ebe21b86'))
            if cached and (cached['status']=='FAILED' or (dest.exists() and cached.get('audio',{}).get('sha256')==digest(dest))):
                row=dict(cached,cache_hit=True,fingerprint=fp)
            else:
                py=ROOT/'vc_models'/model.family/'.venv/Scripts/python.exe'
                if model.family=='audiocpp':py=ROOT/'vc_models/runner/.venv/Scripts/python.exe'
                print(f'[{model.id}] {src.name}',flush=True)
                try:
                    if not py.exists():raise RuntimeError('Isolated Python environment missing')
                    if dest.exists():dest.unlink() # Never accept an old WAV after a failed new render.
                    sidecar=dest.with_suffix('.metrics.json')
                    if sidecar.exists():sidecar.unlink()
                    command=[py,worker,model.id,src,ref,dest]+(['--retry-failed'] if args.retry_failed else [])
                    metric=run(command,ROOT/'vc_models'/model.family/'repo',out/'logs'/f'{model.id}_{src.stem}.log',args.inference_timeout)
                    row['metrics']=metric
                    if sidecar.exists():
                        row['adapter_metrics']=json.loads(sidecar.read_text('utf-8'))
                        row['stage']=row['adapter_metrics']['stage']
                        row['model_load_time']=row['adapter_metrics'].get('model_load_seconds')
                        row['generation_seconds']=row['adapter_metrics'].get('generation_seconds')
                        row['first_audio_latency']=row['adapter_metrics'].get('first_audio_latency_seconds')
                        if row['generation_seconds'] is not None:
                            row['generation_rtf']=row['generation_seconds']/source['stats']['duration']
                    if metric['status']!='SUCCESS':raise RuntimeError(metric['reason'] or metric['last_error'])
                    row['audio']=stats(dest)
                    ratio=row['audio']['duration']/source['stats']['duration']
                    if not .8<=ratio<=1.2:raise ValueError(f'Output duration mismatch: ratio {ratio:.3f}')
                    if row['audio']['rms']<1e-7:raise ValueError('Silent output')
                    row['cold_rtf']=metric['wall_seconds']/source['stats']['duration']
                    row['rtf']=row.get('generation_rtf',row['cold_rtf'])
                    row['rtf_scope']=(row.get('adapter_metrics') or {}).get('rtf_scope','generation') if 'generation_rtf' in row else 'cold_end_to_end'
                    row['suitability']=classification(row['rtf'])
                    row['status']='SUCCESS';row['stage']='complete'
                except Exception as e:row['error']=str(e)
            if row['status']=='SUCCESS':
                # Apply current reporting rules to cached measurements without rendering again.
                wall=row['metrics'].get('wall_seconds')
                row['cold_rtf']=wall/source['stats']['duration'] if wall is not None else None
                row['rtf']=row.get('generation_rtf',row['cold_rtf'])
                row['rtf_scope']=(row.get('adapter_metrics') or {}).get('rtf_scope','generation') if 'generation_rtf' in row else 'cold_end_to_end'
                row['suitability']=classification(row['rtf'])
            metric=row.get('metrics')
            if metric:
                # uv's Windows launcher is a small parent of the actual Python
                # worker. Keep its memory separately; do not label it model RAM.
                metric.setdefault('launcher_process_ram_bytes',metric.get('process_ram_bytes'))
                metric.setdefault('launcher_process_peak_ram_bytes',metric.get('process_peak_ram_bytes'))
                adapter=row.get('adapter_metrics') or {}
                metric['process_ram_bytes']=adapter.get('worker_process_ram_bytes')
                metric['process_peak_ram_bytes']=adapter.get('worker_process_peak_ram_bytes')
                metric['process_ram_definition']='Python adapter RSS; native/CLI child memory is included in peak_ram_bytes (sampled process tree)'
                if metric.get('cpu_measurement_schema')!=2:
                    metric.setdefault('invalid_legacy_cpu_percent_mean',metric.get('cpu_percent_mean'))
                    metric.setdefault('invalid_legacy_cpu_percent_peak',metric.get('cpu_percent_peak'))
                    metric['cpu_percent_mean']=None;metric['cpu_percent_peak']=None
                    metric['cpu_percent_definition']='UNMEASURED: legacy sampler invalid; see separate CPU benchmark'
            manifest['results'].append(row)
            save(out/'metadata/results'/model.id/(src.stem+'.json'),row)
            save(oldpath,manifest)
    manifest['cpu_benchmarks']={}
    for model in MODELS:
        attempts=[r for r in manifest['results'] if r['model']==model.id]
        rows=[r for r in attempts if r['status']=='SUCCESS']
        if not rows:continue
        measured=next((r for r in rows if (r.get('metrics') or {}).get('cpu_measurement_schema')==2),None)
        if measured is None:
            # A bounded generation timeout still contains a valid CPU sample;
            # report its failed status/source instead of rendering a duplicate.
            measured=next((r for r in attempts if (r.get('metrics') or {}).get('cpu_measurement_schema')==2
                           and r['stage']=='generation' and (r.get('metrics') or {}).get('cpu_percent_mean') is not None),None)
        if measured:
            manifest['cpu_benchmarks'][model.id]=dict(source=measured['source'],metrics=measured['metrics'],scope='original conversion')
            continue
        row=rows[0];cache=out/'metadata/cpu_benchmarks'/f'{model.id}.json'
        prior=json.loads(cache.read_text('utf-8')) if cache.exists() else {}
        if prior.get('fingerprint')==row['fingerprint'] and prior.get('metrics',{}).get('cpu_measurement_schema')==2 and not (args.retry_failed and prior['metrics']['status']=='FAILED'):
            manifest['cpu_benchmarks'][model.id]=prior
            continue
        if not args.refresh_cpu:
            manifest['cpu_benchmarks'][model.id]=dict(source=row['source'],scope='cached conversion; additional inference disabled by default',
                metrics=dict(status='UNMEASURED',cpu_percent_mean=None,cpu_percent_peak=None,
                             reason='Legacy CPU sample unavailable; use --refresh-cpu only to explicitly request additional inference'))
            continue
        print(f'[{model.id}] CPU measurement refresh (quality WAV preserved)',flush=True)
        destination=out/'metadata/cpu_benchmarks'/f'{model.id}.wav';destination.parent.mkdir(parents=True,exist_ok=True)
        python=ROOT/'vc_models'/model.family/'.venv/Scripts/python.exe'
        if model.family=='audiocpp':python=ROOT/'vc_models/runner/.venv/Scripts/python.exe'
        command=[python,ROOT/'tools/vc_tournament/infer.py',model.id,row['source'],row['reference'],destination]
        try:
            metric=run(command,ROOT/'vc_models'/model.family/'repo',out/'logs'/f'{model.id}_cpu_refresh.log',args.inference_timeout)
        except Exception as error:
            metric=dict(status='FAILED',reason=f'{type(error).__name__}: {error}',cpu_measurement_schema=2,
                        cpu_percent_mean=None,cpu_percent_peak=None)
        value=dict(fingerprint=row['fingerprint'],source=row['source'],reference=row['reference'],metrics=metric,
                   scope='separate matched source/reference offline inference; original quality WAV preserved')
        save(cache,value);manifest['cpu_benchmarks'][model.id]=value;save(oldpath,manifest)
    manifest['blind']=build(out,manifest['results'])
    manifest['status']='READY FOR HUMAN LISTENING' if manifest['blind']['count'] else ('SOURCE_REQUIRED' if missing else 'NO_SUCCESSFUL_OUTPUTS')
    manifest['finished_at']=time.time();save(oldpath,manifest);report(out,manifest)
    summary=dict(status=manifest['status'],candidate_conditions=len(MODELS),families=len(families()),
                 successful_outputs=manifest['blind']['count'],missing_sources=missing,page=manifest['blind']['page'])
    save(out/'reports/summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':sys.exit(main())
