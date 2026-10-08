from dataclasses import asdict,replace
import math
import time
import numpy as np
import pytest
from src.asr.context import TextContextAnalyzer
from src.asr.bridge import ASRBridge
from src.prosody.context import PhraseContext
from src.prosody.parameters import ProsodyParameters,load_preset
from src.prosody.text_modifier import TextModifier
from src.prosody.controller import ProsodyController
from src.settings.manager import AppSettings
from .test_ai_gui import ai_window


@pytest.mark.parametrize('text,expected',[('ほんとに？','QUESTION'),('これでいいの','QUESTION'),
    ('今日はテストをしています','STATEMENT'),('やった！','EXCLAMATION'),('すごい','EXCLAMATION'),
    ('ねえ、ちょっと聞いて','CALLOUT'),('じゃあ、またね','FAREWELL'),('うん','SHORT_RESPONSE'),
    ('ううん','SHORT_RESPONSE'),('えっ','SHORT_RESPONSE'),('なるほど','SHORT_RESPONSE')])
def test_classification(text,expected):
    ctx=TextContextAnalyzer().update(text,1,1)
    assert ctx.phrase_type==expected
    assert 0<=ctx.context_confidence<=1


@pytest.mark.parametrize('text',['','\ufffd','hello','a'*513,'\x00ほんと'])
def test_garbage_safe(text):
    ctx=TextContextAnalyzer().update(text,1,1)
    assert ctx.context_confidence==0


def test_stable_prefix_and_reset():
    a=TextContextAnalyzer()
    values=[a.update(t,1,1) for t in ('ほ','ほん','ほんと','ほんとに','ほんとに？')]
    assert values[-1].stable_prefix=='ほんとに'
    assert values[-1].question_probability>.9
    a.reset(); assert a.update('うん',2,2).stable_prefix==''


def run_modifier(text,context=None,p=None,audio_pitch=.3):
    p=p or ProsodyParameters(enabled=True,engine='text_v3',amount=100)
    ctx=context or PhraseContext(state='ENDING_CANDIDATE',ending_probability=1,pitch_slope=5)
    m=TextModifier()
    for _ in range(30): result=m.update(audio_pitch,0,.5,ctx,text,p,.03)
    return result


def context(**kw):
    return dict(context_confidence=1,age_ms=0,stable_age_ms=200,**kw)


def test_question_direction_and_late_result():
    assert run_modifier(context(question_probability=1))[0]>0
    falling=PhraseContext(state='ENDING_CANDIDATE',ending_probability=1,pitch_slope=-5)
    assert run_modifier(context(question_probability=1),falling)[0]==0
    assert run_modifier(dict(context(question_probability=1),stable_age_ms=20))[0]==0


def test_statement_confidence_stale_empty():
    for t in ({},context(),dict(context(question_probability=1),context_confidence=0),
              dict(context(question_probability=1),age_ms=900)):
        assert run_modifier(t)==(0,0)


def test_exclamation_callout_farewell():
    assert run_modifier(context(exclamation_probability=1))[1]>0
    onset=PhraseContext(state='EARLY')
    assert run_modifier(context(callout_probability=1),onset)[0]>0
    assert run_modifier(context(farewell_probability=1))[0]<0


def test_modifier_strength_order_and_limits():
    results=[]
    for name in ('Natural','Anime Light','Anime Expressive'):
        p=load_preset(name,ProsodyParameters(enabled=True,engine='text_v3',amount=100))
        results.append(run_modifier(context(question_probability=1),p=p)[0])
    assert 0<results[0]<results[1]<results[2]<=.6


def test_controller_no_context_matches_v2_and_clamps():
    a,b=ProsodyController(),ProsodyController()
    p=ProsodyParameters(enabled=True)
    for i in range(100):
        v2=a.update(160+i*.3,1,-25,.03,p)
        v3=b.update(160+i*.3,1,-25,.03,replace(p,engine='text_v3'),{})
        assert v2.pitch_delta==v3.pitch_delta and v2.gain_db==v3.gain_db
    assert p.min_pitch<=v3.pitch_delta<=p.max_pitch


def test_off_silence_fades():
    c=ProsodyController(); p=ProsodyParameters(enabled=True,engine='text_v3')
    for i in range(100): c.update(160+i,1,-20,.03,p,context(exclamation_probability=1))
    for i in range(80): result=c.update(0,0,-100,.03,replace(p,engine='off'))
    assert abs(result.pitch_delta)<.01 and abs(result.gain_db)<.01


@pytest.mark.parametrize('data',[{}, {'engine':'unknown','asr_threads':999,'text_influence':math.nan},
    {'engine':'text_v3','transcript_logging':'true'}])
def test_settings_migration(data):
    p=AppSettings.from_dict(dict(prosody=data)).prosody_parameters()
    assert p.engine in ('rule_v2','text_v3') and not p.transcript_logging
    assert 1<=p.asr_threads<=4 and math.isfinite(p.text_influence)


def test_bounded_queue_no_callback_wait():
    b=ASRBridge(ProsodyParameters(enabled=True,engine='text_v3')); b.active=True; b.status='Ready'
    chunk=np.ones(256,dtype=np.float32)
    for _ in range(400): b.submit(chunk)
    assert b.input.available<=b.input.capacity and b.drops>0
    b._fail('fixture crash')
    assert b.current()=={} and b.status=='Error' and b.fallbacks==1


def test_stale_context_and_privacy():
    b=ASRBridge(ProsodyParameters(enabled=True,engine='text_v3')); b.active=True; b.status='Listening'
    b.context=dict(partial_text='秘密の発話',stable_prefix='秘密',audio_timestamp=time.monotonic())
    assert b.current()['partial_text']
    assert 'partial_text' not in b.snapshot()['context']
    assert not b.current(time.monotonic()+1)


class FakeClient:
    def __init__(self,p): self.process=None
    def start(self): pass
    def read(self): return dict(status='Ready'),None
    def request(self,h,a=None):
        if h['op']=='reset': return dict(status='Ready'),None
        return dict(status='Listening',context=dict(audio_timestamp=h['audio_timestamp'],partial_text='うん')),None
    def interrupt(self): pass
    def close(self): pass


def wait(predicate):
    until=time.monotonic()+2
    while not predicate() and time.monotonic()<until: time.sleep(.005)
    assert predicate()


def test_worker_lifecycle_restart_and_no_double_start():
    p=ProsodyParameters(enabled=True,engine='text_v3')
    b=ASRBridge(p,FakeClient); b.load(); b.load(); wait(lambda:b.status=='Ready')
    thread=b.thread; b.load(); assert b.thread is thread
    b.set_active(True); time.sleep(.03)
    b.submit(np.ones(14400,dtype=np.float32)); wait(lambda:b.status=='Listening')
    b.stop(); assert not b.alive and b.status=='Stopped'
    b.load(); wait(lambda:b.status=='Ready'); b.stop()


def test_missing_worker_environment_and_model():
    from src.asr.backend import ReazonBackend
    with pytest.raises((ValueError,ImportError)):
        ReazonBackend('missing-model').load()


def test_watchdog_stall():
    b=ASRBridge(ProsodyParameters()); b.request_timeout=.001
    b._request_started=time.monotonic()-1; b._watch()
    assert b.status=='Error' and not b.current()


@pytest.mark.parametrize('frames',[14400,28800,43200,48000])
def test_asr_protocol_large_bounded_hops(frames):
    from io import BytesIO
    from src.asr.protocol import send,receive
    stream=BytesIO(); source=np.zeros(frames,dtype=np.float32)
    send(stream,{'op':'process'},source); stream.seek(0)
    header,audio=receive(stream)
    assert header['op']=='process' and len(audio)==frames


def test_asr_protocol_rejects_unbounded():
    from io import BytesIO
    from src.asr.protocol import send
    with pytest.raises(ValueError): send(BytesIO(),{},np.zeros(48001,dtype=np.float32))


def test_asr_percentiles_do_not_saturate_callback_histogram():
    from src.asr.performance import ASRTimingStats
    s=ASRTimingStats()
    for ms in range(100,200): s.record(ms*1000000,14400,48000)
    result=s.snapshot()
    assert result.p50_ms==pytest.approx(149.1) and result.p95_ms==pytest.approx(194.1)
    assert result.p99_ms==pytest.approx(198.1) and result.maximum_ms==199


def test_text_ui_settings_round_trip(ai_window):
    app,window,backend,ai=ai_window
    window.prosody_engine.setCurrentIndex(window.prosody_engine.findData('text_v3'))
    window.text_influence.setValue(42)
    window.asr_threads.setValue(2)
    window.log_transcripts.setChecked(True)
    p=window._prosody_parameters()
    assert p.engine=='text_v3' and p.text_influence==42 and p.asr_threads==2 and p.transcript_logging
    migrated=AppSettings.from_dict(dict(prosody=asdict(p))).prosody_parameters()
    assert migrated==p


def test_asr_cleanup_failure_does_not_prevent_prosody_stop(monkeypatch):
    from src.prosody.bridge import ProsodyBridge
    bridge=ProsodyBridge()
    bridge.active=True; bridge._control=(time.monotonic(),.3,.2)
    def broken_stop(): raise RuntimeError('fixture ASR cleanup error')
    monkeypatch.setattr(bridge.asr,'stop',broken_stop)
    bridge.stop()
    assert bridge.status=='Stopped' and not bridge.active and bridge._stop.is_set()
    assert bridge.control()==(0.,0.)
