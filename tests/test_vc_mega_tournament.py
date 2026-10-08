"""Offline unit/mock verification only. Never hardware/listening measurements."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pytest
# soundfile is a benchmark-only dependency; without it this module is skipped the
# same way the other isolated-environment tests are.
sf = pytest.importorskip('soundfile')

TOOLS=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
from vc_tournament.audio import normalize,prepare_reference,stats,digest
from vc_tournament.blind import build
from vc_tournament.catalog import MODELS
from vc_tournament.process import run
from vc_tournament.sources import decode_sources
from vc_mega_tournament import classification,fingerprint,inference_code_fingerprint

def test_result_journal_recovers_after_coordinator_loss(tmp_path):
    from vc_mega_tournament import result_cache
    folder=tmp_path/'metadata/results/mock';folder.mkdir(parents=True)
    (folder/'normal.json').write_text(json.dumps(dict(fingerprint='complete',status='SUCCESS')))
    (folder/'low.json').write_text(json.dumps(dict(fingerprint='failed',status='FAILED')))
    (folder/'broken.json').write_text('{interrupted')
    assert set(result_cache({},tmp_path))=={'complete','failed'}
    assert set(result_cache({},tmp_path,True))=={'complete'}

def test_adapter_fix_does_not_invalidate_other_models(tmp_path):
    worker=tmp_path/'worker.py'
    worker.write_text('import os\ndef meanvc2(): return 1\ndef conan(): return 1\n')
    before=inference_code_fingerprint('meanvc2',worker)
    conan_before=inference_code_fingerprint('conan',worker)
    worker.write_text('import os\ndef meanvc2(): return 1\ndef conan(): return 2\n')
    assert inference_code_fingerprint('meanvc2',worker)==before
    assert inference_code_fingerprint('conan',worker)!=conan_before

def test_requested_inventory_and_speed_boundaries():
    assert {'meanvc2_40','meanvc2_120','meanvc','conan','conan_fast','seedvc_tiny','xvc','vevo','facodec','noro','knnvc_20','knnvc_full','freevc','fragmentvc'}<=set(m.id for m in MODELS)
    assert [classification(x) for x in [None,.7,1,2,2.01]]==['UNMEASURED','REALTIME STRONG','REALTIME POSSIBLE','BORDERLINE','OFFLINE ONLY']

def test_shared_amphion_environment_keeps_adapter_cache_separate(tmp_path,monkeypatch):
    import vc_mega_tournament as runner
    from vc_tournament.catalog import Model
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    code=tmp_path/'tools/vc_tournament';code.mkdir(parents=True)
    (code/'process.py').write_text('# shared helper')
    worker=code/'infer.py'
    worker.write_text('def vevo(): return 1\ndef facodec(): return 1\n')
    audio=tmp_path/'input.wav';audio.write_bytes(b'fixed test bytes')
    vevo=Model('vevo','amphion','open-mmlab/Amphion')
    facodec=Model('facodec','amphion','open-mmlab/Amphion')
    prior=[fingerprint(m,audio,audio,{}) for m in (vevo,facodec)]
    worker.write_text('def vevo(): return 2\ndef facodec(): return 1\n')
    current=[fingerprint(m,audio,audio,{}) for m in (vevo,facodec)]
    assert current[0]!=prior[0]
    assert current[1]==prior[1]

def test_blind_normalization_is_one_scalar(tmp_path):
    t=np.arange(16000)/16000
    x=.22*np.sin(2*np.pi*233*t)+.14
    raw=tmp_path/'raw.wav';out=tmp_path/'blind.wav';sf.write(raw,x,16000,subtype='FLOAT')
    meta=normalize(raw,out);y,_=sf.read(out)
    assert abs(y.mean())<1e-4
    assert np.max(np.abs(y-(x-x.mean())*meta['scalar_gain']))<4e-5
    assert meta['after']['peak']<=.981
    assert digest(raw)==meta['raw']['sha256']

def test_clipping_safety_and_silence_rejected(tmp_path):
    p=tmp_path/'float.wav';sf.write(p,np.array([0,2,-2,0]*1000,dtype='float32'),16000,subtype='FLOAT')
    meta=normalize(p,tmp_path/'out.wav');assert meta['raw']['clipping_samples']==2000
    assert meta['after']['clipping_samples']==0
    sf.write(p,np.zeros(500),16000)
    with pytest.raises(ValueError,match='Silent'):normalize(p,tmp_path/'silent.wav')

def test_reference_preserved_and_holdout_excluded(tmp_path):
    sr=8000;t=np.arange(sr*100)/sr;x=.08*np.sin(2*np.pi*220*t)
    original=tmp_path/'original.wav';sf.write(original,x,sr);before=digest(original)
    e=[dict(start=30,end=37)]
    info=prepare_reference(original,tmp_path/'refs',e)
    assert digest(original)==before
    assert info['segments']['full']['stats']['duration']==pytest.approx(92.5)
    for key in ['05s','10s','20s','60s']:
        seg=info['segments'][key];a=seg['start_seconds'];b=a+seg['stats']['duration']
        assert b<=29.75 or a>=37.25

def test_blind_does_not_disclose_models_and_package_changes(tmp_path):
    for d in ['metadata','blind','reference','source','raw']:(tmp_path/d).mkdir()
    raw=tmp_path/'raw/secret.wav';sf.write(raw,.1*np.sin(np.arange(16000)*.1),16000)
    row=dict(model='SECRET_MODEL',source=str(tmp_path/'source/source_normal.wav'),reference='PRIVATE_REF',fingerprint='PRIVATE_KEY',status='SUCCESS',output=str(raw))
    first=build(tmp_path,[row]);text=(tmp_path/'blind/index.html').read_text('utf-8')
    assert 'SECRET_MODEL' not in text and 'PRIVATE_KEY' not in text and 'PRIVATE_REF' not in text
    assert 'A001.wav' in text and first['count']==1
    assert json.loads((tmp_path/'metadata/blind_manifest.json').read_text())['candidates'][0]['model']=='SECRET_MODEL'
    second=build(tmp_path,[]);assert second['count']==0 and first['package']!=second['package']

def test_process_failure_does_not_prevent_next_process(tmp_path):
    fail=run([sys.executable,'-c','raise RuntimeError("mock failure")'],tmp_path,tmp_path/'fail.log',timeout=10)
    success=run([sys.executable,'-c','print("next candidate")'],tmp_path,tmp_path/'success.log',timeout=10)
    assert fail['status']=='FAILED' and 'mock failure' in fail['last_error']
    assert success['status']=='SUCCESS'

def test_process_timeout(tmp_path):
    value=run([sys.executable,'-c','import time; time.sleep(60)'],tmp_path,tmp_path/'timeout.log',timeout=1)
    assert value['reason']=='TIMEOUT' and value['wall_seconds']<10

def test_cpu_samples_keep_the_previous_process_sample(tmp_path):
    code='import time\nend=time.perf_counter()+1.5\nx=0\nwhile time.perf_counter()<end: x+=1'
    value=run([sys.executable,'-c',code],tmp_path,tmp_path/'cpu.log',timeout=10)
    assert value['status']=='SUCCESS' and value['cpu_measurement_schema']==2
    assert value['cpu_percent_mean']>5
    assert value['cpu_percent_peak']>10

def test_resume_retries_download_without_preserving_old_failure_as_current(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from vc_tournament import setup
    from vc_tournament.catalog import Model
    root=tmp_path;base=root/'vc_models/knnvc'
    (base/'repo/.git').mkdir(parents=True)
    (base/'.venv/Scripts').mkdir(parents=True)
    (base/'.venv/Scripts/python.exe').write_bytes(b'mock')
    monkeypatch.setattr(setup.subprocess,'check_output',lambda *a,**k:'mock-revision\n')
    monkeypatch.setattr(setup.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='mock-lock',returncode=0))
    attempts=[];download_success=False
    def fake_run(argv,*a,**k):
        attempts.append(argv)
        is_download=any(str(v).endswith('download.py') for v in argv)
        return {'status':'FAILED' if is_download and not download_success else 'SUCCESS'}
    monkeypatch.setattr(setup,'run',fake_run)
    m=Model('knnvc_20','knnvc','bshall/knn-vc')
    first=setup.prepare(m,root,root/'output',10)
    assert first['status']=='FAILED'
    attempts.clear();download_success=True
    cached=setup.prepare(m,root,root/'output',10)
    assert cached['status']=='FAILED' and cached['cache_hit']
    assert attempts==[]
    second=setup.prepare(m,root,root/'output',10,retry=True)
    assert second['status']=='SUCCESS'
    assert len(attempts)==1 and any(str(v).endswith('download.py') for v in attempts[0])
    assert first['operations'][-1]['status']=='FAILED'
    assert second['history'] # Historical failure retained separately.

def test_report_only_recovers_journal_without_setup_or_inference(tmp_path,monkeypatch):
    import vc_mega_tournament as runner
    from vc_tournament.catalog import Model
    out=tmp_path/'output';metadata=out/'metadata';(metadata/'history').mkdir(parents=True)
    raw=tmp_path/'raw.wav';sf.write(raw,.1*np.sin(np.arange(16000)*.1),16000)
    from vc_tournament.audio import stats
    row=dict(model='meanvc',source='source_normal.wav',reference='reference_10s.wav',
             fingerprint='cache-key',output=str(raw),status='SUCCESS',stage='complete',audio=stats(raw),metrics={})
    previous=dict(results=[],setup={},reference={},missing_sources=[])
    (metadata/'tournament_manifest.json').write_bytes(b'corrupt\x83')
    # History recovery requires at least one recorded successful output.
    previous['results']=[dict(row,status='SUCCESS')]
    (metadata/'history/manifest_1.json').write_text(json.dumps(previous))
    journal=metadata/'results/meanvc';journal.mkdir(parents=True)
    (journal/'source_normal.json').write_text(json.dumps(row))
    monkeypatch.setattr(runner,'MODELS',[Model('meanvc','meanvc','ASLP-lab/MeanVC')])
    def forbidden(*a,**k):raise AssertionError('Report-only must not setup or infer')
    monkeypatch.setattr(runner,'prepare',forbidden);monkeypatch.setattr(runner,'run',forbidden)
    monkeypatch.setattr(runner,'report',lambda *a:None)
    monkeypatch.setattr(sys,'argv',['runner','--report-only','--output',str(out)])
    assert runner.main()==0
    result=json.loads((metadata/'tournament_manifest.json').read_text())
    assert result['status']=='READY FOR HUMAN LISTENING' and result['blind']['count']==1
    assert digest(raw)==row['audio']['sha256']
    assert result['cpu_benchmarks']['meanvc']['metrics']['status']=='UNMEASURED'
