"""Causal acoustic scenarios, including endings with opposite directions."""
from dataclasses import replace, asdict
import math
import numpy as np
import pytest
from src.prosody.context import PhraseStateTracker, EndingProbability
from src.prosody.controller import ProsodyController
from src.prosody.parameters import ProsodyParameters, load_preset
from src.prosody.output import OutputControl


def simulate(p, pitches, energies=None):
    c=ProsodyController()
    energies=energies if energies is not None else np.full(len(pitches),-25.)
    return c,[c.update(float(f),1 if f else 0,float(e),.03,p) for f,e in zip(pitches,energies)]


@pytest.mark.parametrize('direction',[-1,1])
def test_presets_distinct_and_keep_original_direction(direction):
    curve=160*2**(direction*np.linspace(0,5,150)/12)
    strength=[]
    for name in ('Natural','Anime Light','Anime Expressive'):
        _,results=simulate(load_preset(name,ProsodyParameters(enabled=True)),curve)
        offsets=np.array([r.quantized_pitch_delta for r in results[30:]])
        assert direction*offsets.mean()>=0
        assert direction*np.mean([r.pitch_delta for r in results[30:]])>0
        strength.append(abs(offsets).mean())
    assert strength[0]<strength[1]<strength[2]


@pytest.mark.parametrize('direction',[-1,0,1])
def test_ending_direction_and_flat_never_lift(direction):
    tracker=PhraseStateTracker()
    for i in range(40):
        tracker.update(60,1,-20,True,.03)
    for i in range(12):
        context=tracker.update(60+direction*i*.12,.82,-20-i*2,True,.03)
    assert context.ending_probability>.4
    assert direction==0 or direction*context.pitch_slope>0
    c,results=simulate(load_preset('Anime Expressive',ProsodyParameters(enabled=True)),
        160*2**(direction*np.linspace(0,3,100)/12),np.r_[np.full(75,-20.),np.linspace(-20,-55,25)])
    assert direction==0 and abs(results[-1].pitch_delta)<.01 or direction*results[-1].pitch_delta>0


def test_long_stable_phrase_not_forced_to_end():
    tracker=PhraseStateTracker()
    states=[tracker.update(60,1,-25,True,.03).state for _ in range(200)]
    assert states[-1]=='MIDDLE' and 'ENDING_CANDIDATE' not in states
    assert {'ONSET','EARLY','MIDDLE'}<=set(states)


def test_short_pause_continuity_and_long_reset():
    tracker=PhraseStateTracker()
    for _ in range(30): tracker.update(60,1,-25,True,.03)
    for _ in range(8): tracker.update(0,0,-90,False,.03)
    resumed=tracker.update(60,1,-25,True,.03)
    assert resumed.phrase_duration>.9 and resumed.short_pause_count==1
    for _ in range(45): final=tracker.update(0,0,-90,False,.03)
    assert final.state=='SILENCE' and tracker.reset_count==1
    assert len(tracker.pitch_history)==0 and final.ending_probability==0
    assert tracker.update(60,1,-25,True,.03).state=='ONSET'


def test_hysteresis_limits_state_chatter():
    tracker=PhraseStateTracker()
    for _ in range(30): tracker.update(60,1,-25,True,.03)
    changes=tracker.transitions
    for i in range(50): tracker.update(60,.8 if i%2 else 1,-25 if i%2 else -26,True,.03)
    assert tracker.transitions-changes<=12


def test_noise_and_short_utterance():
    c,results=simulate(ProsodyParameters(enabled=True),np.zeros(30),np.full(30,-75))
    assert all(r.phrase_state=='SILENCE' and r.pitch_delta==0 for r in results)
    _,results=simulate(ProsodyParameters(enabled=True),np.full(7,160.))
    assert results[-1].phrase_state in ('ONSET','EARLY')


def test_unvoiced_consonants_with_energy_do_not_end_phrase():
    tracker=PhraseStateTracker()
    for _ in range(30): tracker.update(60,1,-25,True,.03)
    for _ in range(15): context=tracker.update(0,0,-25,False,.03)
    assert context.state=='MIDDLE' and tracker.reset_count==0
    assert not tracker.update(60,1,-25,True,.03).onset


def test_initial_unvoiced_energy_cannot_prevent_first_onset():
    tracker=PhraseStateTracker()
    for _ in range(20):
        assert tracker.update(0,0,-25,False,.03).state=='SILENCE'
    assert tracker.update(60,1,-25,True,.03).state=='ONSET'


def test_pitch_grid_deadband_and_dwell():
    output=OutputControl(); p=ProsodyParameters(enabled=True)
    for _ in range(100): output.update(.125,0,.03,p)
    before=output.changes
    for _ in range(60): output.update(.19,0,.03,p)
    assert output.quantized==.125 and output.changes==before
    for _ in range(30): output.update(.24,0,.03,p)
    assert output.quantized==.25


def test_switch_and_off_slew_no_reset_jump():
    c=ProsodyController(); p=load_preset('Natural',ProsodyParameters(enabled=True))
    outputs=[]
    for i in range(160):
        if i==60: p=load_preset('Anime Expressive',p)
        if i==100: p=replace(p,enabled=False)
        outputs.append(c.update(160*2**(i*.02/12),1,-25,.03,p))
    assert max(abs(b.pitch_delta-a.pitch_delta) for a,b in zip(outputs,outputs[1:]))<=.120001
    assert abs(outputs[101].pitch_delta)>0  # OFF fades instead of jumping to zero
    assert abs(outputs[-1].pitch_delta)<.01 and outputs[-1].quantized_pitch_delta==0


def test_octave_does_not_spike_or_corrupt_context():
    c,results=simulate(ProsodyParameters(enabled=True),np.r_[np.full(30,150),300,np.full(30,150)])
    assert c.invalid_f0_count==1
    assert max(abs(r.pitch_delta) for r in results)<.4
    assert results[-1].phrase_state=='MIDDLE'


@pytest.mark.parametrize('ms',[30,40,50])
def test_update_rates_bounded_and_finite(ms):
    p=load_preset('Anime Expressive',ProsodyParameters(enabled=True,update_ms=ms))
    c=ProsodyController()
    for i in range(100):
        r=c.update(160*2**(math.sin(i*.1)*3/12),1,-25+math.sin(i*.1)*5,ms/1000,p)
        assert -1.2<=r.quantized_pitch_delta<=1.5 and abs(r.gain_db)<=1.5
        assert r.quantized_pitch_delta*8==round(r.quantized_pitch_delta*8)


def test_v1_named_migration_and_custom_preservation():
    p=ProsodyParameters.from_dict(dict(preset='Anime Light',range_expansion=.3))
    assert p.rule_version==2 and p.range_expansion==.4
    assert ProsodyParameters.from_dict(dict(preset='Custom',range_expansion=.3)).range_expansion==.3
    assert ProsodyParameters.from_dict(asdict(p))==p


def test_invalid_state_isolation():
    from src.prosody.bridge import ProsodyBridge
    from src.vc.bridge import AIBridge
    ai=AIBridge(); ai.status='Ready'
    ai.prosody._fail('Invalid phrase state')
    assert ai.status=='Ready' and not ai.prosody.parameters.enabled
    assert ai.prosody.control()==(0,0)


def test_future_frames_cannot_change_past_control():
    prefix=np.linspace(140,180,40)
    p=ProsodyParameters(enabled=True)
    _,a=simulate(p,np.r_[prefix,np.full(30,200)])
    _,b=simulate(p,np.r_[prefix,np.full(30,100)])
    assert a[:40]==b[:40]


@pytest.mark.parametrize('direction',[-1,0,1])
def test_ending_emphasis_observes_direction_only(direction):
    from src.prosody.rules import ContextRuleEngine
    from src.prosody.context import PhraseContext
    p=ProsodyParameters(enabled=True,onset_lift=0)
    context=PhraseContext(state='ENDING_CANDIDATE',pitch_slope=direction*8,ending_probability=.9)
    delta,_=ContextRuleEngine().targets(0,0,0,0,context,p)
    assert delta==0 if direction==0 else direction*delta>0
    disabled,_=ContextRuleEngine().targets(0,0,0,0,context,replace(p,ending_emphasis=False))
    assert disabled==0


def test_zero_energy_amount_disables_context_gain():
    from src.prosody.rules import ContextRuleEngine
    from src.prosody.context import PhraseContext
    p=ProsodyParameters(enabled=True,energy=0)
    for state in ('ONSET','MIDDLE','ENDING_CANDIDATE'):
        _,gain=ContextRuleEngine().targets(1,1,10,.1,
            PhraseContext(state=state,pitch_slope=8,energy_slope=-20,ending_probability=.9),p)
        assert gain==0
