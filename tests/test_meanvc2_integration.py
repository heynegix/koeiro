import io,time
import pytest
import numpy as np
from src.vc.config import AIParameters
from src.vc.models import DEFAULT_VOICE_ID, profile
from src.vc.protocol import send,receive
from src.vc.bridge import AIBridge
from src.processors.ai_voice import AIVoiceProcessor
from tests.test_ai_voice import FakeClient,wait
from tests.test_ai_gui import ai_window,pump

# One voice ships, so the tuning values live in that profile rather than being spread
# across the several variants this test used to parameterise over.
VOICE=DEFAULT_VOICE_ID
PROFILE=profile(VOICE)

def test_meanvc_fixed_hop_bounded_queues_and_neutral_controls():
    p=AIParameters(model=VOICE,threads=4,pitch=7,post_fx=True,low_cut=True)
    assert p.chunk_frames==7680 and p.queue_chunks==PROFILE['queue_chunks']
    assert p.startup_frames==PROFILE['startup_chunks']*p.chunk_frames
    # The route has no native pitch or post-FX stage, so those requests are dropped
    # rather than silently accepted and ignored.
    assert p.pitch==0 and not p.post_fx and not p.low_cut
    stream=io.BytesIO();send(stream,{'op':'process'},np.zeros(p.chunk_frames,dtype=np.float32));stream.seek(0)
    assert len(receive(stream)[1])==p.chunk_frames

def test_meanvc_never_starts_or_submits_prosody(monkeypatch):
    bridge=AIBridge(AIParameters(model=VOICE),FakeClient)
    p=AIVoiceProcessor(bridge=bridge);p.prepare(48000,256)
    def forbidden(*a,**k):raise AssertionError('MeanVC2 must not run prosody')
    monkeypatch.setattr(bridge.prosody,'load',forbidden)
    monkeypatch.setattr(bridge.prosody,'submit',forbidden)
    monkeypatch.setattr(bridge.prosody,'control',forbidden)
    try:
        bridge.load();wait(lambda:bridge.status=='Ready')
        bridge.set_active(True);wait(lambda:bridge.ack_generation==bridge.generation)
        for _ in range(31):p.process(np.ones((256,1),dtype=np.float32),48000)
        bridge.input_ready.signal();wait(lambda:bridge.client.calls>=1)
        assert not bridge.prosody.active
    finally:bridge.stop()

def test_meanvc_underrun_rebuffers_instead_of_repeated_tiny_dropout():
    bridge=AIBridge(AIParameters(model=VOICE),FakeClient)
    p=AIVoiceProcessor(bridge=bridge);p.prepare(48000,256)
    bridge.status='Ready';bridge.ack_generation=bridge.generation
    p._epoch=bridge.generation;p._primed=True
    p.process(np.zeros((256,1),dtype=np.float32),48000)
    assert bridge.underruns==1 and not p._primed
    p.process(np.zeros((256,1),dtype=np.float32),48000)
    assert bridge.underruns==1

def test_long_inference_percentiles_do_not_saturate_at_maximum():
    from src.audio.performance import TimingStats
    s=TimingStats()
    for _ in range(99):s.record(160000000,7680,48000)
    s.record(700000000,7680,48000)
    snapshot=s.snapshot()
    assert 160<=snapshot.p50_ms<=161
    assert 160<=snapshot.p95_ms<=161
    assert snapshot.maximum_ms==700

def test_meanvc_gui_disables_processing_controls_the_route_does_not_have(ai_window):
    app,window,backend,bridge=ai_window
    # The shipped voice runs without prosody and has no native pitch or post-FX stage,
    # so those controls stay disabled rather than accepting an ignored request.
    assert not window.prosody_on.isChecked()
    assert window.prosody_engine.currentData()=='off'
    for widget in (window.prosody_on,window.prosody_engine,window.text_influence,window.text_strategy,
                   window.log_transcripts,window.ai_pitch,window.ai_post_fx):
        assert not widget.isEnabled()
    pump(app,lambda:bridge.parameters.model==VOICE and not window._pending)

def test_compute_scheduling_retains_timer_without_mmcss_registration():
    from types import SimpleNamespace
    from src.utils.windows import audio_scheduling
    calls=[]
    def forbidden(*args):raise AssertionError('Compute thread must not join MMCSS')
    avrt=SimpleNamespace(AvSetMmThreadCharacteristicsW=forbidden,AvRevertMmThreadCharacteristics=forbidden)
    winmm=SimpleNamespace(timeBeginPeriod=lambda n:calls.append(('begin',n)) or 0,
                         timeEndPeriod=lambda n:calls.append(('end',n)))
    with audio_scheduling(avrt,winmm,register_mmcss=False) as result:
        assert result==dict(mmcss=False,timer_1ms=True)
    assert calls==[('begin',1),('end',1)]


def test_meanvc_profile_uses_isolated_worker_and_survives_settings_roundtrip():
    from pathlib import Path
    from src.vc.client import worker_python
    from src.settings.manager import AppSettings
    root=Path(__file__).resolve().parents[1]
    if not (root/'vc_models/meanvc2/.venv/Scripts/python.exe').is_file():
        pytest.skip('Optional isolated MeanVC2 Python not installed')
    executable,environment=worker_python(root,VOICE)
    assert Path(environment.get('__PYVENV_LAUNCHER__',str(executable)))==root/'vc_models/meanvc2/.venv/Scripts/python.exe'
    saved=AppSettings.from_dict({'ai_model':VOICE,'ai_pitch':9,'ai_post_fx':True})
    assert saved.ai_parameters().model==VOICE and saved.ai_parameters().pitch==0
    assert not saved.ai_parameters().post_fx


def test_restarting_the_voice_discards_the_previous_queue(ai_window):
    app,window,backend,bridge=ai_window
    expected=bridge.chunk_frames*PROFILE['queue_chunks']
    assert bridge.input.capacity==expected and bridge.output.capacity==expected
    bridge.output.write(np.ones(256,dtype=np.float32))
    window.stop_button.click()
    pump(app,lambda:window.start_button.isEnabled())
    window.start_button.click()
    pump(app,lambda:bridge.status=='Ready' and not window._pending)
    assert bridge.output.available==0
    assert not window.prosody_on.isEnabled() and not window.ai_pitch.isEnabled()
