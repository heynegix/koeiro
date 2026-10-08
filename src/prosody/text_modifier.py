"""Text is delayed evidence, never a replacement pitch contour."""
import math


class TextModifier:
    strengths={'Natural':(.20,.15,.15),'Anime Light':(.50,.45,.40),
               'Anime Expressive':(.80,.75,.70),'Custom':(.50,.45,.40)}

    def __init__(self): self.reset()

    def reset(self):
        self.pitch=self.gain=0.
        self.influence=0.

    def update(self,audio_pitch,audio_gain,relative,context,text,parameters,dt):
        pitch=gain=0.
        # Defensive boundary: invalid text falls back to audio, not worker Error.
        def number(name,low=0.,high=1.,default=0.):
            v=text.get(name,default) if isinstance(text,dict) else default
            return float(v) if type(v) in (int,float) and math.isfinite(v) and low<=v<=high else default
        age=number('age_ms',0.,100000.,100000.)
        fresh=max(0.,min(1.,(600-age)/300))
        confidence=number('context_confidence')
        enabled=parameters.enabled and parameters.engine=='text_v3'
        strength=confidence*fresh*(parameters.text_influence/100)*min(1.5,parameters.amount/60) if enabled else 0.
        if context.state=='SILENCE': strength=0.
        q,ex,call=self.strengths[parameters.preset]
        question=number('question_probability')*q
        exclaim=number('exclamation_probability')*ex
        callout=number('callout_probability')*call
        farewell=number('farewell_probability')
        # Delayed final or a newly recognized ending must not produce a last-frame jump.
        ending=context.state in ('LATE','ENDING_CANDIDATE') and context.silence_duration<.06
        has_window=number('stable_age_ms',0.,100000.)>=100
        if ending and has_window and context.pitch_slope>.6:
            pitch+=.8*question*context.ending_probability
        # Strong falling source contour is never reversed by question text.
        pitch+=audio_pitch*.20*exclaim
        gain+=.5*exclaim
        if context.state in ('ONSET','EARLY'):
            pitch+=.4*callout
            gain+=.2*callout
            if isinstance(text,dict) and text.get('short_response_type')=='SURPRISE':
                pitch+=.25*ex; gain+=.15*ex
            if isinstance(text,dict) and text.get('short_response_type')=='ACK':
                pitch-=audio_pitch*.10
        if ending and farewell:
            pitch-=audio_pitch*.20*farewell
            gain-=.15*farewell
        self.influence+=(1-math.exp(-dt/.15))*(strength-self.influence)
        a=1-math.exp(-dt/.12)
        self.pitch+=a*(pitch*self.influence-self.pitch)
        self.gain+=a*(gain*self.influence-self.gain)
        self.pitch=max(-.6,min(.6,self.pitch)); self.gain=max(-.5,min(.5,self.gain))
        return self.pitch,self.gain
