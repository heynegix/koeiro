"""Bounded causal text events; controls only, never owns an audio stream."""
from collections import deque
from dataclasses import dataclass, asdict
import math


ENVELOPES = {
    'QUESTION': (80, 300, 300, 800),
    'FAREWELL': (100, 300, 300, 900),
    'EXCLAMATION': (60, 200, 200, 600),
    'SHORT_RESPONSE': (40, 150, 150, 450),
    'CALLOUT': (40, 100, 120, 350),
}


@dataclass
class TextProsodyEvent:
    event_id: int
    phrase_id: int
    event_type: str
    confidence: float
    created_at: float
    activated_at: float | None
    attack_ms: float
    hold_ms: float
    release_ms: float
    base_pitch_strength: float
    base_energy_strength: float
    state: str
    expires_at: float
    source_context_age_ms: float
    subtype: str = 'UNKNOWN'
    envelope: float = 0.
    release_at: float | None = None
    release_level: float = 0.
    effective_duration_ms: float = 0.
    frames: int = 0
    effective_quantized_frames: int = 0
    pitch_sum: float = 0.
    gain_sum: float = 0.
    max_effective_pitch: float = 0.
    observed_duration_ms: float = 0.
    first_effective_at: float | None = None

    def cancel(self, now):
        if self.state in ('FINISHED', 'CANCELLED', 'RELEASE'): return
        self.release_at, self.release_level = now, self.envelope
        self.state = 'RELEASE'

    def advance(self, now):
        if self.state in ('FINISHED', 'CANCELLED'): self.envelope = 0.; return
        if self.activated_at is None: self.activated_at = now
        self.observed_duration_ms=(now-self.activated_at)*1000
        if self.release_at is not None:
            self.envelope = self.release_level * max(0., 1-(now-self.release_at)*1000/self.release_ms)
            if self.envelope == 0: self.state = 'CANCELLED'
            return
        elapsed = (now-self.activated_at)*1000
        if now >= self.expires_at: self.state='FINISHED'; self.envelope=0.; return
        if elapsed < self.attack_ms:
            self.state='ATTACK'; self.envelope=elapsed/self.attack_ms
        elif elapsed < self.attack_ms+self.hold_ms:
            self.state='HOLD'; self.envelope=1.
        else:
            self.state='RELEASE'
            self.envelope=max(0.,1-(elapsed-self.attack_ms-self.hold_ms)/self.release_ms)
            if self.envelope == 0: self.state='FINISHED'

    def metrics(self):
        return dict(asdict(self), event_duration_ms=self.observed_duration_ms,
            event_survived=self.effective_duration_ms>=200,
            mean_effective_pitch=self.pitch_sum/max(1,self.frames),
            mean_effective_gain=self.gain_sum/max(1,self.frames),
            survival_rate_per_event=self.effective_quantized_frames/max(1,self.frames))


class TextEventDetector:
    keys={'QUESTION':'question_probability','FAREWELL':'farewell_probability',
        'EXCLAMATION':'exclamation_probability','CALLOUT':'callout_probability'}

    def detect(self, text):
        def finite(key):
            value=text.get(key,0.)
            return float(value) if type(value) in (int,float) and math.isfinite(value) else 0.
        confidence=finite('context_confidence')
        if confidence<.5: return []
        result=[(kind,confidence,'UNKNOWN') for kind,key in self.keys.items() if finite(key)>=.65]
        subtype=text.get('short_response_type','UNKNOWN')
        if subtype in ('ACK','AGREEMENT','NEGATIVE','SURPRISE','REACTION'):
            result.append(('SHORT_RESPONSE',confidence,subtype))
        return result


class TextProsodyEventEngine:
    """At most five events per phrase, 64 completed records, no repeated attack."""
    def __init__(self):
        self.detector=TextEventDetector(); self.events={}; self.records=deque(maxlen=64)
        self.phrase_id=None; self.next_id=0; self.count=self.activations=self.cancels=0
        self.waterfall={}; self.direction=0.; self.minimum=0.; self.dominant=None
        self.last_floor=-100.; self.current_time=0.
        self.source_generation=None; self.source_changed=False
        self.preset_level=None

    def cancel_all(self, now):
        for event in self.events.values():
            if event.state not in ('FINISHED','CANCELLED','RELEASE'): self.cancels+=1
            event.cancel(now)

    def update(self, text, context, p, now, dt, phrase_id, good):
        self.current_time=now
        generation=text.get('source_generation',self.source_generation)
        if self.source_generation is not None and generation!=self.source_generation:
            self.cancel_all(now); self.source_changed=True
        self.source_generation=generation
        if phrase_id != self.phrase_id:
            for event in self.events.values():
                if event.state not in ('FINISHED','CANCELLED'):
                    self.cancels+=1; event.state='CANCELLED'
                event.envelope=0.; self.records.append(event.metrics())
            self.events={}; self.phrase_id=phrase_id
        valid=text.get('phrase_id',phrase_id)==phrase_id
        age=text.get('age_ms',10000.)
        age=float(age) if type(age) in (int,float) and math.isfinite(age) else 10000.
        enabled=p.enabled and p.engine=='text_v3' and p.amount>0 and p.text_influence>0
        # Grace only concerns current phrase; no output is created in silence.
        grace=context.silence_duration<=.25
        if self.source_changed and all(e.state in ('FINISHED','CANCELLED') for e in self.events.values()):
            self.records.extend(e.metrics() for e in self.events.values()); self.events={}; self.source_changed=False
        fresh=valid and 0<=age<=1000 and grace and not self.source_changed
        if enabled and fresh:
            for kind,confidence,subtype in self.detector.detect(text):
                if kind=='CALLOUT' and (context.phrase_duration>.3 or age>300): continue
                if kind not in self.events:
                    attack,hold,release,maximum=ENVELOPES[kind]
                    self.next_id+=1; self.count+=1
                    pitch={'QUESTION':.25,'FAREWELL':-.25,'EXCLAMATION':.10,
                        'SHORT_RESPONSE':.125 if subtype=='SURPRISE' else 0.,'CALLOUT':.125}[kind]
                    gain={'QUESTION':0.,'FAREWELL':-.25,'EXCLAMATION':.30,
                        'SHORT_RESPONSE':.15 if subtype=='SURPRISE' else 0.,'CALLOUT':.1}[kind]
                    self.events[kind]=TextProsodyEvent(self.next_id,phrase_id,kind,confidence,now,None,
                        attack,hold,release,pitch,gain,'PENDING',now+maximum/1000,age,subtype)
                elif self.events[kind].state in ('ATTACK','HOLD'):
                    # Reconfirmation may strengthen confidence, never restart or prolong indefinitely.
                    self.events[kind].confidence=max(self.events[kind].confidence,confidence)
            # Explicit contradiction withdraws an event; transient UNKNOWN is not contradiction.
            if text.get('phrase_type')=='STATEMENT' and text.get('context_confidence',0)>=.8:
                self.cancel_all(now)
        if not enabled or not grace or text.get('unavailable',False): self.cancel_all(now)
        for event in self.events.values():
            if event.activated_at is None: self.activations+=1
            event.advance(now)
        active=[e for e in self.events.values() if e.envelope>0 and e.state not in ('CANCELLED','FINISHED')]
        self.dominant=next((self.events[k] for k in ('QUESTION','FAREWELL','SHORT_RESPONSE','EXCLAMATION','CALLOUT')
            if k in self.events and self.events[k] in active),None)
        self.direction=self.minimum=0.; self.waterfall={}
        if not active or not good or not enabled: return 0.,0.,1.,1.,1.
        desired_level={'Natural':.5,'Anime Light':1.,'Anime Expressive':2.,'Custom':1.}[p.preset]
        self.preset_level=desired_level if self.preset_level is None else self.preset_level+(1-math.exp(-dt/.08))*(desired_level-self.preset_level)
        level=self.preset_level
        amount=min(1.5,p.amount/60)*p.text_influence/100
        values={}
        for event in active:
            event_age=event.source_context_age_ms+(now-event.created_at)*1000
            age_factor=1. if event_age<400 else .8 if event_age<700 else .5 if event_age<=1000 else 0.
            # Confidence qualified once: avoid repeatedly multiplying tiny probabilities.
            strength=level*amount*event.confidence*age_factor*event.envelope
            values[event.event_type]=(event,strength,age_factor)
        dominant=self.dominant
        event,strength,age_factor=values[dominant.event_type]
        direction=1.
        if event.event_type=='QUESTION': direction=.35 if context.pitch_slope<-2 else .75 if abs(context.pitch_slope)<.6 else 1.
        elif event.event_type=='FAREWELL': direction=0. if context.pitch_slope>.6 else 1.
        pitch=event.base_pitch_strength*strength*direction
        gain=max(-.5,min(.5,sum(e.base_energy_strength*s for e,s,_ in values.values())))
        ex=values.get('EXCLAMATION',(None,0.,0.))[1]
        q=values.get('QUESTION',(None,0.,0.))[1]
        fall=values.get('FAREWELL',(None,0.,0.))[1] if context.pitch_slope<=0 else 0.
        self.direction=1 if pitch>0 else -1 if pitch<0 else 0
        eligible=event.event_type in ('QUESTION','FAREWELL') and event.confidence>=.8 and age_factor>=.5
        eligible=eligible and event.state=='HOLD' and direction>=.75 and p.preset!='Natural' and amount>=.8
        self.minimum=(.25 if p.preset=='Anime Expressive' and level>=1.8 else .125)*min(1.,amount) if eligible else 0.
        self.waterfall=dict(event_id=event.event_id,base_event_strength=event.base_pitch_strength,
            preset_multiplier=level,confidence_multiplier=event.confidence,age_multiplier=age_factor,
            phrase_state_multiplier=direction,envelope_value=event.envelope,
            pre_quantization_result=pitch,minimum_quantum=self.minimum,post_quantization_result=0.,final_applied_result=0.,minimum_blocked_by_clamp=False)
        return pitch,gain,1+.15*ex,1+.20*q+.15*ex,1+.10*fall

    def enforce(self, audio_quantized, final, p):
        """Qualified HOLD reserves one native step, subject to headroom and bounded slew."""
        if self.minimum and self.direction and self.current_time-self.last_floor>=.06:
            target=audio_quantized+self.direction*self.minimum
            low,high=math.ceil(p.min_pitch*8)/8,math.floor(p.max_pitch*8)/8
            self.waterfall['minimum_blocked_by_clamp']=not low<=target<=high
            if low<=target<=high and self.direction*(final-audio_quantized)<self.minimum:
                final+=self.direction*.125; final=max(low,min(high,final)); self.last_floor=self.current_time
        self.waterfall.update(post_quantization_result=final-audio_quantized)
        return final

    def observe(self, pitch, gain, dt):
        if self.dominant is None: return
        event=self.dominant
        if event.envelope<=0: return
        event.frames+=1; event.pitch_sum+=pitch; event.gain_sum+=gain
        event.max_effective_pitch=max(event.max_effective_pitch,abs(pitch))
        if abs(pitch)>=.1249:
            if event.first_effective_at is None: event.first_effective_at=self.current_time
            event.effective_quantized_frames+=1; event.effective_duration_ms+=dt*1000
        self.waterfall['final_applied_result']=pitch

    def snapshot(self):
        return dict(event_count=self.count,event_activation_count=self.activations,
            event_cancel_count=self.cancels,active_events=[e.metrics() for e in self.events.values()
                if e.state not in ('FINISHED','CANCELLED')],events=[e.metrics() for e in self.events.values()],
            completed=list(self.records),waterfall=dict(self.waterfall))
