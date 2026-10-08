"""Bounded production replay against the user's approved long-phrase candidate."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.compare_meanvc2_continuity import OUT,read,save,sha,require_idle_voice_worker
from tools.vc_tournament.process import run
from src.vc.config import AIParameters


def main():
    import numpy as np
    import psutil
    import soundfile as sf
    require_idle_voice_worker()
    if psutil.virtual_memory().available<4*1024**3:
        raise RuntimeError('Need at least 4 GiB free RAM')
    preserved={str(p):sha(p) for p in [ROOT/'settings.json',ROOT/'models/meanvc2_ref60/reference.wav',
                                    ROOT/'models/meanvc2_ref60/fixed_embedding.npy']}
    rows=[]
    for name in ('long','normal','low','bright'):
        require_idle_voice_worker()
        source=OUT/f'source/source_{name}.wav'
        audition=OUT/f'raw/vc720/source_{name}.wav'
        report=ROOT/f'validation/v011/meanvc2_adopted720_{name}.json'
        signature=dict(backend=sha(ROOT/'src/vc/meanvc2.py'),runtime=sha(ROOT/'models/meanvc2_ref60/runtime.json'),
                       source=sha(source),audition=sha(audition),threads=4,
                       benchmark=sha(ROOT/'tools/benchmark_meanvc2_realtime.py'))
        metric=report.with_name(report.stem+'_process.json')
        cached=read(metric) if metric.exists() else {}
        if cached.get('status')!='SUCCESS' or cached.get('signature')!=signature or not report.with_suffix('.wav').exists() or cached.get('output_sha256')!=sha(report.with_suffix('.wav')):
            arguments=['tools/benchmark_meanvc2_realtime.py','--model','meanvc2_ref60','--threads','4',
                       '--seconds','40','--source',str(source),'--report',str(report)]
            bootstrap='import psutil,runpy,sys; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS); sys.argv='+repr(arguments)+'; runpy.run_path(sys.argv[0],run_name="__main__")'
            process=run([ROOT/'vc_models/meanvc2/.venv/Scripts/python.exe','-c',bootstrap],ROOT,
                        OUT/'logs'/f'adopted720_{name}.log',timeout=180,ram_limit_gb=4.5)
            process['signature']=signature
            if process['status']=='SUCCESS':process['output_sha256']=sha(report.with_suffix('.wav'))
            save(metric,process)
            if process['status']!='SUCCESS':raise RuntimeError('Replay failed: '+str(process['last_error']))
        result=read(report)
        # Saved auditions use two threads; compare the original implementation
        # at four threads to separate FP32 reduction drift from porting errors.
        control=report.with_name(report.stem+'_audition4.json')
        control_metric=control.with_name(control.stem+'_process.json')
        control_signature=dict(signature,experiment=sha(ROOT/'src/vc/meanvc2_continuity.py'))
        cached_control=read(control_metric) if control_metric.exists() else {}
        if cached_control.get('status')!='SUCCESS' or cached_control.get('signature')!=control_signature or not control.with_suffix('.wav').exists() or cached_control.get('output_sha256')!=sha(control.with_suffix('.wav')):
            arguments=['tools/benchmark_meanvc2_realtime.py','--model','meanvc2_ref60','--threads','4',
                       '--seconds','40','--source',str(source),'--audition-vc720','--report',str(control)]
            bootstrap='import psutil,runpy,sys; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS); sys.argv='+repr(arguments)+'; runpy.run_path(sys.argv[0],run_name="__main__")'
            process=run([ROOT/'vc_models/meanvc2/.venv/Scripts/python.exe','-c',bootstrap],ROOT,
                        OUT/'logs'/f'audition4_{name}.log',timeout=180,ram_limit_gb=4.5)
            process['signature']=control_signature
            if process['status']=='SUCCESS':process['output_sha256']=sha(control.with_suffix('.wav'))
            save(control_metric,process)
            if process['status']!='SUCCESS':raise RuntimeError('Audition control failed')
        original,_=sf.read(audition,dtype='float32')
        a,sr=sf.read(control.with_suffix('.wav'),dtype='float32');b,br=sf.read(report.with_suffix('.wav'),dtype='float32')
        error=float(np.max(np.abs(a-b))) if sr==br and a.shape==b.shape else None
        rows.append(dict(source=name,rtf=result['rtf'],chunk_max_ms=result['chunk_max_ms'],
                         two_vs_four_threads_max_error=float(np.max(np.abs(original-b))),
                         waveform_max_error=error,matched_approved_candidate=error is not None and error<=1e-5,
                         model_buffer_ms=result['algorithmic_buffer_ms'],report=str(report)))
    unchanged=all(sha(path)==digest for path,digest in preserved.items())
    result=dict(variant='vc720',user_approval='All five long candidates sound good, little perceived difference; choose by data.',
                candidate_package=read(OUT/'metadata/blind_manifest.json')['package'],
                input_kind='Offline WAV replay, four threads, no audio devices',rows=rows,
                voice_assets_and_settings_unchanged=unchanged,
                nominal_total_buffer_ms=rows[0]['model_buffer_ms']+40+AIParameters(model='meanvc2_ref60').startup_frames/48,
                actual_end_to_end_latency_measured=False,
                passed=unchanged and all(r['matched_approved_candidate'] and r['rtf']<1 for r in rows))
    save(ROOT/'validation/v011/meanvc2_adoption_validation.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':sys.exit(main())
