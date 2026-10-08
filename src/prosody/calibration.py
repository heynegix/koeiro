"""Context modulates observed rules. No melody generation, no audio wait."""
from dataclasses import replace
import math


def context_freshness(age_ms):
    if not math.isfinite(age_ms) or age_ms<0: return 0.
    if age_ms<=300: return 1.
    if age_ms<=500: return 1.-.2*(age_ms-300)/200
    if age_ms<=800: return .8-.3*(age_ms-500)/300
    return max(0.,.5*(1000-age_ms)/200)


class CalibratedTextModifier:
    envelopes={'SILENCE':0.,'ONSET':.05,'EARLY':.20,'MIDDLE':.45,'LATE':.75,'ENDING_CANDIDATE':1.}
    limits={'Natural':.125,'Anime Light':.250,'Anime Expressive':.500,'Custom':.250}

    def __init__(self): self.reset()

    def reset(self):
        self.effects={k:0. for k in ('q','ex','call','farewell','ack','surprise')}
        self.limit=0.; self.range=self.rise=self.fall=1.; self.bias=self.gain=0.

    def prepare(self,context,text,p,dt,relative,velocity):
        def number(key,default=0.):
            v=text.get(key,default)
            return float(v) if type(v) in (int,float) and math.isfinite(v) else default
        confidence=max(0.,min(1.,number('context_confidence')))
        strength=max(0.,(confidence-.2)/.8)*context_freshness(number('age_ms',10000.))
        strength*=p.text_influence/100*min(1.5,p.amount/60) if p.enabled and p.engine=='text_v3' else 0.
        strength*=self.envelopes.get(context.state,0.)
        # Preset strength is encoded in its cap; interpolation avoids preset jumps.
        cap=self.limits[p.preset]
        self.limit+=(1-math.exp(-dt/.08))*(cap-self.limit)
        wanted=dict(q=max(0.,min(1.,(number('question_probability')-.45)/.40)),
            ex=max(0.,min(1.,(number('exclamation_probability')-.45)/.40)),
            call=max(0.,min(1.,(number('callout_probability')-.45)/.40)),
            farewell=max(0.,min(1.,(number('farewell_probability')-.45)/.40)),
            ack=float(text.get('short_response_type') in ('ACK','AGREEMENT')),
            surprise=float(text.get('short_response_type')=='SURPRISE'))
        a=1-math.exp(-dt/.080)
        for key in self.effects:
            self.effects[key]+=a*(wanted[key]*strength-self.effects[key])
        q,ex,call,farewell,ack,surprise=(self.effects[k] for k in self.effects)
        level=self.limit/.25
        self.range=1.+.15*level*ex+.08*level*surprise+.05*level*q-.10*ack
        self.rise=1.+.18*level*ex+.20*level*q
        self.fall=1.+.08*level*ex+.10*level*farewell
        # Question influence only amplifies existing upward trajectory.
        rising=context.pitch_slope>.6 and velocity>=0
        ending=context.state in ('LATE','ENDING_CANDIDATE') and context.silence_duration<.09
        stable=number('stable_age_ms')>=90
        bias=0.
        if rising and stable and ending: bias+=self.limit*q*context.ending_probability
        if context.state in ('ONSET','EARLY') and context.pitch_slope>=0:
            bias+=self.limit*.5*(call+surprise)
        self.bias=max(-self.limit,min(self.limit,bias))
        self.gain=min(.5,.3*level*ex+.1*surprise)-.15*farewell*float(ending)
        coefficients=dict(range_expansion=p.range_expansion+(1+p.range_expansion)*(self.range-1),
            rise_boost=p.rise_boost+(1+p.rise_boost)*(self.rise-1),fall_boost=p.fall_boost+(1+p.fall_boost)*(self.fall-1),
            ending_strength=p.ending_strength*(1+.35*level*q if rising else 1+.15*farewell),
            onset_lift=p.onset_lift+min(.25,self.limit*.5)*(call+surprise),
            energy_dynamics=p.energy_dynamics*(1+.15*level*ex))
        # Temporary coefficients stay inside parameter validation bounds.
        bounds={k:(lo,hi) for k,lo,hi in p.bounds()}
        modulated=replace(p,**{k:max(bounds[k][0],min(bounds[k][1],v)) for k,v in coefficients.items()})
        return modulated,self.bias,self.gain
