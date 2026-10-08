from dataclasses import replace
import time
import numpy as np
import pytest
from src.prosody.parameters import ProsodyParameters,load_preset
from src.prosody.calibration import CalibratedTextModifier,context_freshness
from src.prosody.context import PhraseContext
from src.prosody.controller import ProsodyController
from src.asr.stability import ClassificationCache
from src.asr.bridge import ASRBridge


@pytest.mark.parametrize('window',[250,400,600,800,1000])
def test_windows_migrate(window):
    p=ProsodyParameters.from_dict(dict(asr_window_ms=window,text_strategy='hybrid'))
    assert p.asr_window_ms==window


@pytest.mark.parametrize('age,expected',[(0,1),(300,1),(500,.8),(800,.5),(1000,0),(2000,0)])
def test_age_attenuation(age,expected): assert context_freshness(age)==pytest.approx(expected)


def text(**kwargs):
    return dict(context_confidence=1,age_ms=0,stable_age_ms=200,question_probability=1,**kwargs)


@pytest.mark.parametrize('strategy',['direct','modulation','hybrid'])
def test_effective_question_survives(strategy):
    c=ProsodyController(); p=load_preset('Anime Light',ProsodyParameters(enabled=True,engine='text_v3',text_strategy=strategy))
    # Controlled audio-derived ending fixture; F0 remains a gently rising source.
    c.tracker.update=lambda *args:PhraseContext(state='ENDING_CANDIDATE',ending_probability=1,pitch_slope=3)
    controls=[c.update(160+i*.30,1,-25,.03,p,text()) for i in range(80)]
    assert max(v.effective_text_pitch for v in controls)>=.125
    assert all(p.min_pitch<=v.quantized_pitch_delta<=p.max_pitch for v in controls)
    if strategy=='hybrid': assert controls[-1].text_effect_survival_rate>.5


def test_expressive_larger_than_light_and_safe_multi_context():
    peaks=[]
    for preset in ('Natural','Anime Light','Anime Expressive'):
        c=ProsodyController(); p=load_preset(preset,ProsodyParameters(enabled=True,engine='text_v3'))
        c.tracker.update=lambda *args:PhraseContext(state='ENDING_CANDIDATE',ending_probability=1,pitch_slope=3)
        controls=[c.update(160+i*.10,1,-25,.03,p,text(exclamation_probability=1,callout_probability=1)) for i in range(100)]
        peaks.append(max(v.effective_text_pitch for v in controls))
        assert all(p.min_pitch<=v.quantized_pitch_delta<=p.max_pitch and abs(v.gain_db)<=1.5 for v in controls)
    assert peaks[2]>peaks[1]>=.125


@pytest.mark.parametrize('ctx',[{},dict(context_confidence=.1,age_ms=0,question_probability=1),dict(context_confidence=1,age_ms=1500,question_probability=1)])
def test_ineffective_context_zero(ctx):
    m=CalibratedTextModifier(); p=ProsodyParameters(enabled=True,engine='text_v3')
    audio=PhraseContext(state='ENDING_CANDIDATE',pitch_slope=4,ending_probability=1)
    for _ in range(50): changed,bias,gain=m.prepare(audio,ctx,p,.03,.5,3)
    assert bias==gain==0 and changed.range_expansion==p.range_expansion


def test_no_forced_rise_or_statement_acting():
    m=CalibratedTextModifier(); p=ProsodyParameters(enabled=True,engine='text_v3')
    audio=PhraseContext(state='ENDING_CANDIDATE',pitch_slope=-4,ending_probability=1)
    for _ in range(50): _,bias,_=m.prepare(audio,text(),p,.03,-.5,-3)
    assert bias==0


def test_phrase_id_rejects_previous_question():
    c=ProsodyController(); p=ProsodyParameters(enabled=True,engine='text_v3')
    c.phrase_id=5
    c.tracker.update=lambda *args:PhraseContext(state='ENDING_CANDIDATE',pitch_slope=4,ending_probability=1)
    for i in range(100): out=c.update(150+i*.2,1,-25,.03,p,text(phrase_id=4))
    assert out.effective_text_pitch==0 and out.text_pitch_delta==0


def test_probability_hold_and_phrase_reset():
    cache=ClassificationCache()
    a=cache.update({'question_probability':.95},0,1)
    b=cache.update({},.1,1)
    c=cache.update({},.2,2)
    assert a['question_probability']>.9 and b['question_probability']>.65 and c['question_probability']==0


def test_medium_question_survives_one_unknown_hop_but_ack_clears():
    cache=ClassificationCache()
    cache.update(dict(question_probability=.65,context_confidence=.5),0,1)
    assert cache.update({},.3,1)['question_probability']>=.65
    assert cache.update(dict(short_response_type='ACK'),.4,1)['question_probability']==0


def test_asr_phrase_and_restart_context_reset():
    b=ASRBridge(ProsodyParameters(enabled=True,engine='text_v3'))
    b.context=dict(phrase_id=1,audio_timestamp=time.monotonic(),partial_text='ほんとに')
    b.active=True; b.status='Listening'
    b.set_phrase(2)
    assert not b.context
    b.context=dict(phrase_id=1,audio_timestamp=time.monotonic())
    assert not b.current()
    b.stop(); assert not b.context


def test_bounded_latest_only_short_window():
    b=ASRBridge(ProsodyParameters(enabled=True,engine='text_v3',asr_window_ms=250))
    b.active=True; b.status='Ready'
    for _ in range(400): b.submit(np.zeros(256,dtype=np.float32))
    assert b.input.available<=57600 and b.drops>0


def test_preset_interpolation_limits():
    m=CalibratedTextModifier(); ctx=PhraseContext(state='ENDING_CANDIDATE',pitch_slope=4,ending_probability=1)
    p=load_preset('Natural',ProsodyParameters(enabled=True,engine='text_v3'))
    for _ in range(40): m.prepare(ctx,text(),p,.03,1,3)
    last=m.bias
    p=load_preset('Anime Expressive',p)
    m.prepare(ctx,text(),p,.03,1,3)
    assert abs(m.bias-last)<.125


def test_flat_question_does_not_generate_melody():
    c=ProsodyController(); p=ProsodyParameters(enabled=True,engine='text_v3')
    c.tracker.update=lambda *args:PhraseContext(state='ENDING_CANDIDATE',ending_probability=1,pitch_slope=0)
    for _ in range(80): out=c.update(160,1,-25,.03,p,text())
    assert out.effective_text_pitch==0


def test_modulation_exclamation_and_farewell_direction():
    p=ProsodyParameters(enabled=True,engine='text_v3')
    m=CalibratedTextModifier(); ctx=PhraseContext(state='ENDING_CANDIDATE',ending_probability=1,pitch_slope=-4)
    data=dict(context_confidence=1,age_ms=0,stable_age_ms=200,exclamation_probability=1,farewell_probability=1)
    for _ in range(80): changed,bias,gain=m.prepare(ctx,data,p,.03,-.4,-4)
    assert changed.range_expansion>p.range_expansion and changed.fall_boost>p.fall_boost
    assert bias==0 and abs(gain)<=.5


def test_restart_does_not_restore_old_context():
    from .test_text_prosody import FakeClient,wait
    b=ASRBridge(ProsodyParameters(enabled=True,engine='text_v3'),FakeClient)
    b.load(); wait(lambda:b.status=='Ready')
    b.context=dict(question_probability=1,phrase_id=0,audio_timestamp=time.monotonic())
    b.restart(); wait(lambda:b.status=='Ready' and b._restart_thread is not None and not b._restart_thread.is_alive())
    assert b.context=={}
    b.stop(); assert not b.alive


@pytest.mark.parametrize('window,expected',[(250,300),(400,400),(600,600),(800,600),(1000,600)])
def test_window_scheduler(window,expected):
    from src.asr.scheduler import update_interval
    assert update_interval(300,window,300)==expected
    assert update_interval(300,window,100)==300


def test_analysis_discontinuity_never_reuses_phrase_id():
    c=ProsodyController()
    c.phrase_id=7
    c.reset(preserve_phrase=True)
    assert c.phrase_id==8
    p=ProsodyParameters(enabled=True,engine='text_v3')
    c.tracker.update=lambda *args:PhraseContext(state='ENDING_CANDIDATE',ending_probability=1,pitch_slope=3)
    for _ in range(80): out=c.update(180,1,-25,.03,p,text(phrase_id=7))
    assert out.text_pitch_delta==0 and out.effective_text_pitch==0
    c.reset(preserve_phrase=True)
    assert c.phrase_id==9
