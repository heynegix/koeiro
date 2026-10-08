from dataclasses import replace,asdict
import math
import time
from types import SimpleNamespace
import numpy as np
import pytest
from src.prosody.parameters import ProsodyParameters,PRESETS,load_preset
from src.prosody.controller import ProsodyController
from src.prosody.f0 import YinEstimator
from src.prosody.bridge import ProsodyBridge
from src.settings.manager import AppSettings,SettingsManager
from .test_ai_gui import ai_window


@pytest.mark.parametrize('name',list(PRESETS))
def test_presets(name):
    p=load_preset(name,ProsodyParameters(enabled=True,amount=42))
    assert p.preset==name and p.amount==42 and p.enabled
    assert p.min_pitch==PRESETS[name]['min_pitch'] and p.max_pitch==PRESETS[name]['max_pitch']


@pytest.mark.parametrize('name,value',[('amount',-1),('pitch_range',101),('energy',math.nan),
    ('min_pitch',-2),('max_pitch',2),('update_ms',25),('enabled',1),('ending_lift',1),
    ('preset','bad'),('confidence',0),('attack_ms',1),('release_ms',999),('onset_lift',1),('max_gain_db',2)])
def test_invalid_parameters(name,value):
    with pytest.raises(ValueError):
        replace(ProsodyParameters(),**{name:value})


def test_settings_v04_migration_and_roundtrip(tmp_path):
    old=AppSettings.from_dict(dict(voice_mode='ai_voice',ai_pitch=4))
    assert not old.prosody_parameters().enabled
    p=load_preset('Natural',ProsodyParameters(enabled=True,amount=30))
    manager=SettingsManager(tmp_path/'settings.json')
    assert manager.save(replace(old,prosody=asdict(p)))
    assert manager.load().prosody_parameters()==p
    manager.path.write_text('{broken',encoding='utf-8')
    assert not manager.load().prosody_parameters().enabled


@pytest.mark.parametrize('bad',[None,[],42,dict(amount=math.nan),dict(update_ms=25),dict(enabled='true')])
def test_invalid_migration(bad):
    p=AppSettings.from_dict(dict(prosody=bad)).prosody_parameters()
    assert not p.enabled and math.isfinite(p.amount) and p.update_ms==30


def test_baseline_relative_rise_energy_and_clamps():
    c=ProsodyController(); p=ProsodyParameters(enabled=True)
    for _ in range(30):
        c.update(160,.99,-30,parameters=p)
    result=c.update(180,.99,-24,parameters=p)
    assert 160<=result.baseline_f0<180
    assert result.relative_pitch_st>0 and result.velocity_st_s>0 and result.pitch_delta>0
    assert result.gain_db>0
    for f in np.linspace(180,320,30):
        result=c.update(float(f),.99,-10,parameters=p)
        assert -.8<=result.pitch_delta<=1.2 and abs(result.gain_db)<=1


def test_onset_requires_silence_and_smoothing():
    c=ProsodyController(); p=ProsodyParameters(enabled=True)
    first=c.update(150,.99,-30,parameters=p)
    assert first.onset and 0<first.pitch_delta<=.12
    second=c.update(155,.99,-30,parameters=p)
    assert not second.onset and abs(second.pitch_delta-first.pitch_delta)<=.12
    for _ in range(3):
        c.update(0,0,-80,parameters=p)
    assert not c.update(155,.99,-30,parameters=p).onset
    for _ in range(10):
        c.update(0,0,-80,parameters=p)
    assert c.update(155,.99,-30,parameters=p).onset


def test_unvoiced_decay_and_no_constant_modulation():
    c=ProsodyController(); p=ProsodyParameters(enabled=True)
    for _ in range(100):
        result=c.update(160,.99,-30,parameters=p)
    assert abs(result.pitch_delta)<.001 and abs(result.gain_db)<.001
    for _ in range(100):
        result=c.update(0,0,-80,parameters=p)
    assert result.f0==0 and abs(result.pitch_delta)<.001


@pytest.mark.parametrize('f0',[math.nan,math.inf,-math.inf])
def test_nonfinite_controller_rejected(f0):
    c=ProsodyController()
    with pytest.raises(ValueError):
        c.update(f0,1,-30)
    assert c.invalid_f0_count==1


def test_octave_error_rejected_and_reset():
    c=ProsodyController()
    for _ in range(20):
        c.update(140,.99,-30)
    assert c.update(280,.99,-30).f0==0
    assert c.invalid_f0_count==1
    c.reset()
    assert c.invalid_f0_count==0 and c.pitch==0


@pytest.mark.parametrize('frequency',[80,110,160,220,320,500])
def test_yin_sine_frequency(frequency):
    x=(.1*np.sin(2*np.pi*frequency*np.arange(1280)/16000)).astype(np.float32)
    f0,confidence,energy=YinEstimator().estimate(x)
    assert abs(f0-frequency)/frequency<.02 and confidence>.9 and energy>-30


def test_yin_unvoiced_and_invalid():
    e=YinEstimator()
    assert e.estimate(np.zeros(1280,dtype=np.float32))[0]==0
    assert e.estimate(np.random.default_rng(1).normal(0,.05,1280))[0]==0
    with pytest.raises(ValueError):
        e.estimate(np.array([math.nan]))


def test_amount_zero_and_disabled_are_zero():
    for p in (ProsodyParameters(),ProsodyParameters(enabled=True,amount=0)):
        c=ProsodyController()
        for f in (100,110,130,180):
            r=c.update(f,1,-20,parameters=p)
            assert r.pitch_delta==0 and r.gain_db==0


def test_ending_experimental_default_off():
    assert not ProsodyParameters().ending_lift
    c=ProsodyController(); p=ProsodyParameters(enabled=True,ending_lift=True)
    for _ in range(30):
        c.update(160,1,-20,parameters=p)
    r=c.update(158,1,-30,parameters=p)
    assert 0<=r.ending_probability<=1 and -.8<=r.pitch_delta<=1.2


class FakeProsodyClient:
    def __init__(self,p):
        self.parameters=p; self.process=None; self.closed=False
    def start(self): pass
    def read(self): return dict(status='Ready'),np.empty(0,dtype=np.float32)
    def request(self,header,audio=None):
        if header['op']=='reset': return self.read()
        return dict(status='Ready',control=dict(pitch_delta=.3,gain_db=.2)),np.empty(0,dtype=np.float32)
    def interrupt(self): self.closed=True
    def close(self): self.closed=True


def wait(predicate):
    until=time.monotonic()+2
    while time.monotonic()<until:
        if predicate(): return
        time.sleep(.005)
    raise AssertionError('Prosody worker timeout')


def test_worker_lifecycle_latest_queue_and_off_control():
    b=ProsodyBridge(ProsodyParameters(enabled=True),FakeProsodyClient)
    b.load(); b.load()
    try:
        wait(lambda:b.status=='Ready')
        b.set_active(True)
        time.sleep(.03)
        b.submit(np.ones(1440,dtype=np.float32)*.1)
        wait(lambda:b.status=='Running')
        assert b.control()==(.3,.2)
        assert b.control(time.monotonic()+1)==(0.,0.)
        assert b.input.capacity==7680
        b.configure(replace(b.parameters,enabled=False))
        assert b.control(time.monotonic()+1)==(0.,0.)  # normal OFF fades; faults are immediate
    finally:
        b.stop()
    assert not b.alive and b.client is None


def test_queue_bounded_and_consumer_owns_drop():
    b=ProsodyBridge(ProsodyParameters(enabled=True),FakeProsodyClient)
    b.active=True; b.status='Ready'
    for _ in range(8):
        b.submit(np.ones(1440,dtype=np.float32))
    assert b.input.available<=b.input.capacity and b.dropped_frames>0
    assert b.input.read_position==0


def test_failure_fallback_does_not_change_ai():
    from src.vc.bridge import AIBridge
    ai=AIBridge(); ai.status='Ready'
    b=ai.prosody
    b.parameters=ProsodyParameters(enabled=True); b.active=True; b.status='Running'
    b._control=(time.monotonic(),.3,.2)
    before=ai.parameters
    b._fail('test crash')
    assert b.status=='Error' and b.control()==(0.,0.) and b.errors==1
    # A prosody failure must not disturb the voice route, which runs at native pitch.
    assert ai.status=='Ready' and ai.parameters==before and ai.parameters.pitch==0
    b._fail('again'); assert b.errors==1


def test_watchdog_stall():
    b=ProsodyBridge(ProsodyParameters(enabled=True),FakeProsodyClient)
    b.status='Running'; b._request_started=time.monotonic()-2
    b._watch()
    assert b.status=='Error' and 'stalled' in b.error


def test_safe_control_while_loading_and_inactive():
    b=ProsodyBridge()
    for status in ('Stopped','Starting','Ready','Error'):
        b.status=status
        assert b.control()==(0.,0.)


def test_prosody_settings_persist_but_stay_off_on_the_voice_route(ai_window):
    app,window,backend,ai=ai_window
    ai.prosody.client_factory=FakeProsodyClient
    window.prosody_preset.setCurrentText('Natural')
    window.anime_amount.setValue(35)
    window.prosody_range.setValue(70)
    window._capture_settings()
    saved=window.settings.prosody_parameters()
    # The shipped route runs without prosody, so the amount is stored but enabled stays
    # False and the toggle cannot be switched on.
    assert not saved.enabled and saved.preset=='Natural'
    assert saved.amount==35 and saved.range_expansion==.1
    assert not window.prosody_on.isEnabled()
    window.prosody_on.setChecked(True)
    assert not window.settings.prosody_parameters().enabled
    # A failure inside the analysis path still leaves the voice route running.
    ai.status='Ready'
    ai.prosody._fail('fixture crash')
    assert ai.status=='Ready'


def test_rise_boost_and_baseline_median():
    controls=[]
    for boost in (0.,.35):
        c=ProsodyController(); p=ProsodyParameters(enabled=True,rise_boost=boost,onset_lift=0)
        for _ in range(25): c.update(160,1,-30,parameters=p)
        result=c.update(170,1,-30,parameters=p)
        assert result.baseline_f0==pytest.approx(160)
        controls.append(result.pitch_delta)
    assert controls[1]>controls[0]


def test_analysis_crash_does_not_block_callback():
    from src.vc.bridge import AIBridge
    from src.processors.ai_voice import AIVoiceProcessor
    ai=AIBridge(); processor=AIVoiceProcessor(bridge=ai)
    processor.prepare(48000,256)
    ai.status='Ready'; ai.ack_generation=ai.generation; ai.active=True
    ai.prosody.parameters=ProsodyParameters(enabled=True)
    ai.prosody._fail('test')
    block=np.ones((256,1),dtype=np.float32)*.05
    result=processor.process(block,48000)
    assert result is block and np.isfinite(result).all() and ai.status=='Ready'


def test_device_disconnect_stops_the_voice_worker_and_reconnects(ai_window):
    from .test_ai_gui import pump
    app,window,backend,ai=ai_window
    ai.prosody.client_factory=FakeProsodyClient
    window.mode.setCurrentIndex(2)
    pump(app,lambda:ai.status=='Ready')
    window.start_button.click()
    pump(app,lambda:window.stop_button.isEnabled())
    backend.streams[-1].active=False
    pump(app,lambda:window.controller.snapshot.state=='Error' and not ai.alive)
    assert window.controller.alive and not window.controller.engine.running
    window.refresh.click()
    pump(app,lambda:window.start_button.isEnabled())
    window.start_button.click()
    pump(app,lambda:window.stop_button.isEnabled() and ai.status=='Ready')
    assert ai.alive
