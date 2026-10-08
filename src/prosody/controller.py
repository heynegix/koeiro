"""Causal rule control. Produces small deltas; does not synthesize speech/F0."""
from collections import deque
from dataclasses import dataclass, asdict, replace, field
import math
import statistics
from .parameters import ProsodyParameters
from .context import PhraseStateTracker, ProsodyContext, PhraseContext
from .output import OutputControl
from .rules import ContextRuleEngine
from .text_modifier import TextModifier
from .calibration import CalibratedTextModifier
from .events import TextProsodyEventEngine


@dataclass(frozen=True)
class ProsodyControl:
    f0: float = 0.
    baseline_f0: float = 0.
    voiced: float = 0.
    energy_db: float = -120.
    energy_deviation_db: float = 0.
    relative_pitch_st: float = 0.
    velocity_st_s: float = 0.
    pitch_delta: float = 0.
    gain_db: float = 0.
    onset: bool = False
    ending_probability: float = 0.
    phrase_state: str = 'SILENCE'
    phrase_duration: float = 0.
    silence_duration: float = 0.
    pitch_slope: float = 0.
    energy_slope: float = 0.
    raw_pitch_delta: float = 0.
    quantized_pitch_delta: float = 0.
    state_transition_count: int = 0
    ending_trigger_count: int = 0
    state_reset_count: int = 0
    quantization_change_count: int = 0
    hysteresis_hold_count: int = 0
    ending_emphasis_frames: int = 0
    audio_pitch_delta: float = 0.
    text_pitch_delta: float = 0.
    audio_gain_db: float = 0.
    text_gain_db: float = 0.
    phrase_id: int = 0
    text_pitch_bias: float = 0.
    text_range_multiplier: float = 1.
    text_rise_multiplier: float = 1.
    text_fall_multiplier: float = 1.
    effective_text_pitch: float = 0.
    audio_quantized_pitch: float = 0.
    pre_hysteresis_pitch: float = 0.
    text_effect_survival_rate: float = 0.
    text_effect_active_ratio: float = 0.
    smoothed_text_pitch: float = 0.
    effective_text_gain: float = 0.
    text_events: dict = field(default_factory=dict)


class ProsodyController:
    def __init__(self):
        self.reset()

    def reset(self, preserve_phrase=False):
        next_phrase = getattr(self, 'phrase_id', 0) + 1 if preserve_phrase else 0
        self.history=deque(maxlen=80)
        self.energy_history=deque(maxlen=80)
        self.seconds=0.
        self.silence=.5
        self.onset_remaining=0.
        self.smoothed_log_f0=None
        self.previous_relative=0.
        self.previous_energy=-120.
        self.pitch=self.gain=0.
        self.invalid_f0_count=0
        self.speech_duration=0.
        self.tracker=PhraseStateTracker()
        self.output=OutputControl()
        self.rules=ContextRuleEngine()
        self.text_modifier=TextModifier()
        self.calibration=CalibratedTextModifier()
        self.events=TextProsodyEventEngine()
        self.audio_output=OutputControl()
        self.phrase_id=next_phrase; self.text_requested_frames=self.text_survived_frames=self.control_frames=0
        self.context=ProsodyContext(PhraseContext())
        self._effective=None
        self._enable_mix=0.
        self.ending_emphasis_frames=0

    def update(self,f0,voiced,energy_db,dt=.030,parameters=None,text_context=None):
        p=parameters or ProsodyParameters(enabled=True)
        if p.engine=='off': p=replace(p,enabled=False)
        if not all(math.isfinite(float(v)) for v in (f0,voiced,energy_db,dt)) or not 0<dt<=1 or not 0<=voiced<=1:
            self.invalid_f0_count+=1
            raise ValueError('Invalid F0/control frame')
        # Coefficients morph on preset/manual changes; no stream restart.
        numeric=('range_expansion','rise_boost','fall_boost','ending_strength','onset_lift',
            'energy_dynamics','attack_ms','release_ms','min_pitch','max_pitch')
        if self._effective is None:
            self._effective=p
        else:
            alpha=1-math.exp(-dt/.08)  # ~240 ms to 95% of the new coefficients
            self._effective=replace(p,**{key:getattr(self._effective,key)+alpha*
                (getattr(p,key)-getattr(self._effective,key)) for key in numeric})
        p=self._effective
        self.seconds+=dt
        good=65<=f0<=650 and voiced>=p.confidence and energy_db>-60
        if f0 < 0 or (f0>0 and not 65<=f0<=650):
            self.invalid_f0_count+=1
        # Reject abrupt octave-scale jumps within voiced speech. Reset the
        # reference after silence so a new phrase can have a different register.
        if good and self.smoothed_log_f0 is not None and self.silence<.15:
            if abs(12*(math.log2(f0)-self.smoothed_log_f0))>9:
                good=False
                self.invalid_f0_count+=1
        onset=good and (not self.history or self.silence>=p.min_silence_ms/1000 and
            self.tracker.time-self.tracker.last_energy_time>=p.min_silence_ms/1000)
        if onset:
            self.smoothed_log_f0=None
            self.onset_remaining=.150
            self.speech_duration=0.
        relative=velocity=baseline=deviation=0.
        target_pitch=target_gain=0.
        ending=0.
        while self.history and self.seconds-self.history[0][0]>p.baseline_ms/1000:
            self.history.popleft()
        while self.energy_history and self.seconds-self.energy_history[0][0]>p.baseline_ms/1000:
            self.energy_history.popleft()
        if good:
            self.silence=0.
            self.speech_duration+=dt
            logf=math.log2(f0)
            a=1-math.exp(-dt/.050)
            self.smoothed_log_f0=logf if self.smoothed_log_f0 is None else self.smoothed_log_f0+a*(logf-self.smoothed_log_f0)
            smooth_f0=2**self.smoothed_log_f0
            self.history.append((self.seconds,smooth_f0))
            self.energy_history.append((self.seconds,energy_db))
            baseline=statistics.median(v for _,v in self.history)
            relative=max(-6.,min(6.,12*math.log2(smooth_f0/baseline)))
            velocity=max(-36.,min(36.,(relative-self.previous_relative)/dt))
            deviation=energy_db-statistics.median(v for _,v in self.energy_history)
            self.previous_relative=relative
        else:
            self.silence+=dt
            if self.silence>.15:
                self.smoothed_log_f0=None
                self.previous_relative=0.
        self.previous_energy=energy_db
        # Trajectory is absolute smoothed semitone, not a moving-baseline slope.
        trajectory=12*self.smoothed_log_f0 if good else 0.
        context=self.tracker.update(trajectory,voiced if good else 0.,energy_db,good,dt,
            p.history_ms,p.min_silence_ms)
        text=text_context if isinstance(text_context,dict) else {}
        if context.long_reset or context.onset:
            self.text_modifier.reset()
            self.calibration.reset()
        if context.onset: self.phrase_id+=1
        if 'phrase_id' in text and text['phrase_id']!=self.phrase_id:
            text={}
        if context.long_reset or context.onset or (context.state=='SILENCE' and (p.text_strategy!='events' or context.silence_duration>.25)):
            text={}
        self.context=ProsodyContext(context,text)
        ending=context.ending_probability
        if good:
            target_pitch,target_gain=self.rules.targets(relative,velocity,deviation,self.onset_remaining,context,p)
            if p.enabled and p.amount>0 and p.ending_emphasis and context.state=='ENDING_CANDIDATE' and abs(context.pitch_slope)>.6:
                self.ending_emphasis_frames+=1
        self.onset_remaining=max(0.,self.onset_remaining-dt)
        audio_pitch,audio_gain=target_pitch,target_gain
        bias=0.; multipliers=(1.,1.,1.)
        if p.text_strategy=='events':
            bias,text_gain,range_factor,rise_factor,fall_factor=self.events.update(
                text,context,p,self.seconds,dt,self.phrase_id,good)
            multipliers=(range_factor,rise_factor,fall_factor)
            modulated=replace(p,range_expansion=min(.7,p.range_expansion+(range_factor-1)),
                rise_boost=min(.5,p.rise_boost+(rise_factor-1)),fall_boost=min(.3,p.fall_boost+(fall_factor-1)))
            changed_pitch,changed_gain=self.rules.targets(relative,velocity,deviation,self.onset_remaining,context,modulated)
            original_pitch,original_gain=self.rules.targets(relative,velocity,deviation,self.onset_remaining,context,p)
            text_pitch=max(-.5,min(.5,bias+changed_pitch-original_pitch))
            text_gain=max(-.5,min(.5,text_gain+changed_gain-original_gain))
        elif p.text_strategy=='legacy':
            text_pitch,text_gain=self.text_modifier.update(audio_pitch,audio_gain,relative,context,text,p,dt)
        else:
            modulated,bias,text_gain=self.calibration.prepare(context,text,p,dt,relative,velocity)
            multipliers=(self.calibration.range,self.calibration.rise,self.calibration.fall)
            changed_pitch,changed_gain=self.rules.targets(relative,velocity,deviation,self.onset_remaining,context,modulated)
            # Preserve the same onset clock as the audio-only path.
            original_pitch,original_gain=self.rules.targets(relative,velocity,deviation,self.onset_remaining,context,p)
            correction=changed_pitch-original_pitch
            text_pitch=(0. if p.text_strategy=='direct' else correction)+(0. if p.text_strategy=='modulation' else bias)
            text_gain+=changed_gain-original_gain if p.text_strategy!='direct' else 0.
            text_pitch=max(-self.calibration.limit,min(self.calibration.limit,text_pitch))
        if good:
            target_pitch+=text_pitch; target_gain+=text_gain
        if context.long_reset:
            self.history.clear(); self.energy_history.clear()
            self.smoothed_log_f0=None; self.previous_relative=0.
        # Normal OFF is faded in analysis; failure still uses immediate safety fallback.
        desired=1. if p.enabled and p.amount>0 else 0.
        self._enable_mix+=(1-math.exp(-dt/.10))*(desired-self._enable_mix)
        target_pitch=max(p.min_pitch,min(p.max_pitch,target_pitch))*self._enable_mix
        target_gain=max(-p.max_gain_db,min(p.max_gain_db,target_gain))*self._enable_mix
        prior_quantized=self.output.quantized
        quantized,self.gain=self.output.update(target_pitch,target_gain,dt,p)
        audio_quantized,audio_final_gain=self.audio_output.update(
            max(p.min_pitch,min(p.max_pitch,audio_pitch))*self._enable_mix,
            max(-p.max_gain_db,min(p.max_gain_db,audio_gain))*self._enable_mix,dt,p)
        effective_text=quantized-audio_quantized
        if p.text_strategy=='events':
            quantized=self.events.enforce(audio_quantized,quantized,p)
            low,high=math.ceil(p.min_pitch*8)/8,math.floor(p.max_pitch*8)/8
            quantized=max(low,min(high,max(prior_quantized-.125,min(prior_quantized+.125,quantized))))
            self.output.quantized=quantized
            effective_text=quantized-audio_quantized
            self.events.observe(effective_text,self.gain-audio_final_gain,dt)
            if self.events.waterfall.get('event_id') is not None:
                self.events.waterfall.update(event_requested_pitch=bias,
                    rule_modulation_pitch=text_pitch-bias,combined_raw_result=target_pitch,
                    smoothed_result=self.output.pitch-self.audio_output.pitch,
                    pre_quantization_result=self.output.pitch-self.audio_output.pitch,
                    post_quantization_result=effective_text,final_applied_result=effective_text)
        self.control_frames+=1
        requested=good and abs(text_pitch)>.001
        self.text_requested_frames+=int(requested)
        self.text_survived_frames+=int(requested and abs(effective_text)>=.1249)
        self.pitch=self.output.pitch
        return ProsodyControl(f0 if good else 0.,baseline,voiced if good else 0.,energy_db,
            deviation,relative,velocity,self.pitch,self.gain,onset,ending,
            context.state,context.phrase_duration,context.silence_duration,context.pitch_slope,
            context.energy_slope,target_pitch,quantized,self.tracker.transitions,self.tracker.ending_count,
            self.tracker.reset_count,self.output.changes,self.output.holds,self.ending_emphasis_frames,
            audio_pitch,text_pitch,audio_gain,text_gain,self.phrase_id,bias,*multipliers,
            effective_text,audio_quantized,round(self.pitch*8)/8,
            self.text_survived_frames/max(1,self.text_requested_frames),
            self.text_requested_frames/self.control_frames,
            self.pitch-self.audio_output.pitch,self.gain-audio_final_gain,
            self.events.snapshot() if p.text_strategy=='events' else {})

    def snapshot(self,control):
        return dict(asdict(control),invalid_f0_count=self.invalid_f0_count)
