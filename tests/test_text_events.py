from dataclasses import replace
import pytest
from src.prosody.events import TextProsodyEvent,TextProsodyEventEngine,ENVELOPES
from src.prosody.context import PhraseContext
from src.prosody.parameters import ProsodyParameters,load_preset
from src.prosody.controller import ProsodyController
from src.asr.confirmation import needs_confirmation


def parameters(preset='Anime Light'):
    return load_preset(preset,ProsodyParameters(enabled=True,engine='text_v3',text_strategy='events'))


def text(kind='QUESTION',**values):
    result=dict(phrase_id=1,age_ms=0,context_confidence=1,phrase_type=kind)
    if kind in ('QUESTION','FAREWELL','EXCLAMATION','CALLOUT'):
        result[dict(QUESTION='question_probability',FAREWELL='farewell_probability',
            EXCLAMATION='exclamation_probability',CALLOUT='callout_probability')[kind]]=1
    result.update(values); return result


@pytest.mark.parametrize('kind',list(ENVELOPES))
def test_event_envelope_lifecycle(kind):
    attack,hold,release,maximum=ENVELOPES[kind]
    e=TextProsodyEvent(1,1,kind,1,0,None,attack,hold,release,.25,0,'PENDING',maximum/1000,0)
    e.advance(0); assert e.state=='ATTACK' and e.envelope==0
    e.advance(attack/2000); assert e.envelope==pytest.approx(.5)
    e.advance(attack/1000+.01); assert e.state=='HOLD' and e.envelope==1
    e.advance((attack+hold+release/2)/1000); assert e.state=='RELEASE' and e.envelope==pytest.approx(.5)
    e.advance(maximum/1000+.01); assert e.state=='FINISHED' and e.envelope==0


@pytest.mark.parametrize('preset,minimum',[('Anime Light',.125),('Anime Expressive',.25)])
@pytest.mark.parametrize('kind,slope',[('QUESTION',2),('FAREWELL',-2)])
def test_quantum_survives_for_two_hundred_ms(preset,minimum,kind,slope):
    c=ProsodyController(); c.phrase_id=1
    c.tracker.update=lambda *args:PhraseContext(state='ENDING_CANDIDATE',pitch_slope=slope,ending_probability=1)
    p=parameters(preset)
    controls=[c.update(160*2**(slope*i*.02/12),1,-25,.02,p,text(kind)) for i in range(45)]
    durations=sum(.02 for out in controls if abs(out.effective_text_pitch)>=minimum-1e-5)
    assert durations>=.20
    assert all(p.min_pitch<=v.quantized_pitch_delta<=p.max_pitch for v in controls)
    assert max(abs(v.quantized_pitch_delta-controls[i-1].quantized_pitch_delta) for i,v in enumerate(controls) if i)<.251


def test_cancel_is_release_not_abrupt_zero():
    e=TextProsodyEventEngine(); p=parameters(); ctx=PhraseContext(state='LATE')
    for i in range(10): e.update(text(),ctx,p,i*.03,.03,1,True)
    event=e.events['QUESTION']; level=event.envelope
    e.update(text('STATEMENT'),ctx,p,.30,.03,1,True)
    assert event.state=='RELEASE' and event.envelope==level
    e.update({},ctx,p,.40,.03,1,True); assert 0<event.envelope<level
    e.update({},ctx,p,.60,.03,1,True); assert event.state=='CANCELLED'


def test_debounce_refresh_and_max_duration():
    e=TextProsodyEventEngine(); p=parameters(); ctx=PhraseContext(state='LATE')
    for i in range(100): e.update(text(),ctx,p,i*.03,.03,1,True)
    assert e.count==1 and e.activations==1 and e.events['QUESTION'].state=='FINISHED'
    assert e.events['QUESTION'].observed_duration_ms<=900


@pytest.mark.parametrize('silence,created',[(.20,True),(.25,True),(.26,False),(.50,False)])
def test_current_phrase_grace(silence,created):
    e=TextProsodyEventEngine()
    e.update(text(),PhraseContext(state='SILENCE',silence_duration=silence),parameters(),0,.03,1,False)
    assert bool(e.events)==created


def test_new_phrase_cancels_previous_event():
    e=TextProsodyEventEngine(); ctx=PhraseContext(state='LATE'); p=parameters()
    e.update(text(),ctx,p,0,.03,1,True); e.update(text(),ctx,p,.1,.03,1,True)
    e.update(text(),ctx,p,.15,.03,2,True)
    assert not e.events and e.records[-1]['state']=='CANCELLED'


@pytest.mark.parametrize('values',[dict(age_ms=1001),dict(phrase_id=0),dict(context_confidence=.1),dict(age_ms=float('nan'))])
def test_invalid_context_never_triggers(values):
    e=TextProsodyEventEngine(); e.update(text(**values),PhraseContext(state='LATE'),parameters(),0,.03,1,True)
    assert e.count==0


def test_strong_falling_question_does_not_get_floor():
    e=TextProsodyEventEngine(); p=parameters()
    for i in range(15): e.update(text(),PhraseContext(state='LATE',pitch_slope=-10),p,i*.02,.02,1,True)
    assert e.minimum==0 and e.waterfall['phrase_state_multiplier']==.35


def test_farewell_does_not_reverse_rise():
    e=TextProsodyEventEngine(); p=parameters()
    for i in range(15): result=e.update(text('FAREWELL'),PhraseContext(state='LATE',pitch_slope=4),p,i*.02,.02,1,True)
    assert result[0]==0 and result[1]<0


def test_mixed_roles_waterfall_and_clamp():
    c=ProsodyController(); c.phrase_id=1; p=parameters('Anime Expressive')
    c.tracker.update=lambda *args:PhraseContext(state='LATE',pitch_slope=2)
    for _ in range(15): out=c.update(170,1,-25,.03,p,text(exclamation_probability=1))
    assert {v['event_type'] for v in out.text_events['events']}=={'QUESTION','EXCLAMATION'}
    assert out.text_events['waterfall']['base_event_strength']==.25
    assert p.min_pitch<=out.quantized_pitch_delta<=p.max_pitch and abs(out.gain_db)<=1.5


@pytest.mark.parametrize('subtype',['ACK','NEGATIVE','SURPRISE'])
def test_short_response(subtype):
    e=TextProsodyEventEngine(); p=parameters()
    e.update(text('SHORT_RESPONSE',short_response_type=subtype),PhraseContext(state='EARLY'),p,0,.03,1,True)
    assert e.events['SHORT_RESPONSE'].subtype==subtype
    if subtype!='SURPRISE': assert e.events['SHORT_RESPONSE'].base_pitch_strength==0


def test_late_callout_skipped():
    e=TextProsodyEventEngine(); e.update(text('CALLOUT'),PhraseContext(state='MIDDLE',phrase_duration=1),parameters(),0,.03,1,True)
    assert e.count==0


def test_crash_release_and_restart_clear():
    e=TextProsodyEventEngine(); ctx=PhraseContext(state='LATE'); p=parameters()
    for i in range(8): e.update(text(),ctx,p,i*.03,.03,1,True)
    e.update({'unavailable':True},ctx,p,.24,.03,1,True)
    assert e.events['QUESTION'].state=='RELEASE'
    e.update({},ctx,p,.55,.03,1,True); assert e.events['QUESTION'].state=='CANCELLED'
    c=ProsodyController(); c.events=e; c.reset(preserve_phrase=True)
    assert c.events.count==0 and not c.events.events


@pytest.mark.parametrize('context,elapsed,expected',[
    ({'phrase_type':'QUESTION','question_probability':.65,'context_confidence':.5},1,True),
    ({'phrase_type':'UNKNOWN'},1,True),
    ({'phrase_type':'QUESTION','question_probability':1,'context_confidence':1},1,False),
    ({'phrase_type':'UNKNOWN'},.2,False),
    ({'phrase_type':'STATEMENT','context_confidence':.5},1,False)])
def test_conditional_single_flight_confirmation(context,elapsed,expected):
    assert needs_confirmation(context,elapsed,0)==expected


def test_minimum_yields_to_clamp_and_invalid_f0():
    e=TextProsodyEventEngine(); p=parameters()
    for i in range(15): e.update(text(),PhraseContext(state='LATE'),p,i*.02,.02,1,True)
    assert e.enforce(1.125,1.125,p)==1.125
    e.update(text(),PhraseContext(state='LATE'),p,.31,.02,1,False)
    assert e.minimum==0


def test_controller_crash_event_release_is_smooth():
    c=ProsodyController(); c.phrase_id=1; p=parameters()
    c.tracker.update=lambda *args:PhraseContext(state='LATE',pitch_slope=3)
    for _ in range(8): before=c.update(160,1,-25,.02,p,text())
    after=c.update(160,1,-25,.02,p,{'unavailable':True})
    assert after.text_events['events'][0]['state']=='RELEASE'
    assert abs(after.quantized_pitch_delta-before.quantized_pitch_delta)<=.125
    for _ in range(40): final=c.update(160,1,-25,.02,p,{'unavailable':True})
    assert final.quantized_pitch_delta==final.audio_quantized_pitch


def test_controller_allows_only_current_phrase_grace():
    c=ProsodyController(); c.phrase_id=1; p=parameters()
    c.tracker.update=lambda *args:PhraseContext(state='SILENCE',silence_duration=.2)
    out=c.update(0,0,-80,.02,p,text())
    assert out.text_events['event_count']==1 and out.effective_text_pitch==0
    c.phrase_id=2
    out=c.update(0,0,-80,.02,p,text())
    assert not out.text_events['active_events']


def test_asr_restart_epoch_releases_and_never_revives_old_event():
    e=TextProsodyEventEngine(); ctx=PhraseContext(state='LATE'); p=parameters()
    for i in range(8): e.update(text(source_generation=1),ctx,p,i*.03,.03,1,True)
    old=e.events['QUESTION'].event_id
    e.update({'unavailable':True,'source_generation':2},ctx,p,.24,.03,1,True)
    assert e.events['QUESTION'].state=='RELEASE'
    for i in range(10): e.update({'source_generation':2},ctx,p,.27+i*.03,.03,1,True)
    e.update(text(source_generation=2),ctx,p,.65,.03,1,True)
    assert e.events['QUESTION'].event_id>old
    assert e.records[-1]['state']=='CANCELLED'


def test_event_preset_multiplier_morphs_before_output_smoothing():
    e=TextProsodyEventEngine(); ctx=PhraseContext(state='LATE'); p=parameters('Natural')
    e.update(text(),ctx,p,0,.01,1,True)
    before=e.update(text(),ctx,p,.1,.01,1,True)[0]
    after=e.update(text(),ctx,parameters('Anime Expressive'),.11,.01,1,True)[0]
    assert .5<e.waterfall['preset_multiplier']<2
    assert abs(after-before)<.125
