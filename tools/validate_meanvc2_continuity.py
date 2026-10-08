"""Bounded four-thread production replay and waveform parity; no audio devices."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.compare_meanvc2_continuity import OUT,read,save,sha,require_idle_voice_worker
from tools.vc_tournament.process import run


def main():
    import numpy as np
    import psutil
    import soundfile as sf
    require_idle_voice_worker()
    if not read(OUT/'metadata/decode_equivalence.json')['passed']:
        raise RuntimeError('Four-source decode equivalence must pass before production validation')
    if psutil.virtual_memory().available<4*1024**3:
        raise RuntimeError('Need at least 4 GiB free RAM')
    python=ROOT/'vc_models/meanvc2/.venv/Scripts/python.exe'
    signature=dict(backend_sha256=sha(ROOT/'src/vc/meanvc2.py'),vc_group_chunks=1,
                   runtime_sha256=sha(ROOT/'models/meanvc2_ref60/runtime.json'),
                   source_sha256=sha(OUT/'source/source_long.wav'),threads=4)
    reports=[]
    for frames in (1,36):
        require_idle_voice_worker()
        report=ROOT/f'validation/v011/meanvc2_continuity_production_{frames}.json'
        process_report=report.with_name(report.stem+'_process.json')
        if report.exists() and process_report.exists() and report.with_suffix('.wav').exists():
            saved=read(process_report)
            if saved.get('signature')==signature and saved.get('output_sha256')==sha(report.with_suffix('.wav')) and saved.get('status')=='SUCCESS':
                reports.append(read(report));continue
        arguments=['tools/benchmark_meanvc2_realtime.py','--model','meanvc2_ref60','--threads','4',
                   '--seconds','40','--source',str(OUT/'source/source_long.wav'),
                   '--vc-group-chunks','1','--vocoder-batch-frames',str(frames),'--report',str(report)]
        # Existing isolated interpreter; lower priority and no model downloads.
        bootstrap='import psutil,runpy,sys; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS); sys.argv='+repr(arguments)+'; runpy.run_path(sys.argv[0],run_name="__main__")'
        process=run([python,'-c',bootstrap],ROOT,OUT/'logs'/f'production_{frames}.log',timeout=180,ram_limit_gb=4.5)
        process['signature']=signature
        if process['status']=='SUCCESS':
            process['output_sha256']=sha(report.with_suffix('.wav'))
        save(process_report,process)
        if process['status']!='SUCCESS':raise RuntimeError('Production replay failed: '+str(process['last_error']))
        reports.append(read(report))
    paths=[ROOT/f'validation/v011/meanvc2_continuity_production_{n}.wav' for n in (1,36)]
    a,sr=sf.read(paths[0],dtype='float32');b,_=sf.read(paths[1],dtype='float32')
    error=float(np.max(np.abs(a-b))) if len(a)==len(b) else None
    voice=read(OUT/'metadata/preserved_voice.json')
    unchanged=all(sha(path)==voice[key] for key,path in (
        ('settings_sha256',ROOT/'settings.json'),('reference_sha256',ROOT/'models/meanvc2_ref60/reference.wav'),
        ('embedding_sha256',ROOT/'models/meanvc2_ref60/fixed_embedding.npy')))
    result=dict(input_kind='Offline WAV replay; four threads; no audio devices',signature=signature,
                waveform_max_absolute_error=error,parity_passed=error is not None and error<=1e-5,
                voice_and_settings_unchanged=unchanged,baseline_rtf=reports[0]['rtf'],optimized_rtf=reports[1]['rtf'],
                waveform_reconstruction_group_frames=36,model_buffer_ms=reports[1]['algorithmic_buffer_ms'],
                nominal_buffer_and_grid_and_preroll_ms=reports[1]['algorithmic_buffer_ms']+40+1280,
                end_to_end_latency_measured=False,quality_blur_improvement_verified=False)
    result['passed']=result['parity_passed'] and unchanged and reports[1]['rtf']<1
    save(ROOT/'validation/v011/meanvc2_continuity_production_validation.json',result)
    print(json.dumps(result),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':sys.exit(main())
