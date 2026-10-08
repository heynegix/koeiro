"""Bounded per-phrase probability hold for changing rolling partials."""
import math


class ClassificationCache:
    keys=('question_probability','exclamation_probability','callout_probability','farewell_probability')
    def __init__(self): self.reset()
    def reset(self):
        self.values={k:0. for k in self.keys}; self.time=None; self.phrase=None
        self.confirmed={k:-100. for k in self.keys}
        self.category=None; self.repeats=0

    def update(self,context,stamp,phrase_id):
        if self.phrase!=phrase_id: self.reset(); self.phrase=phrase_id
        dt=0. if self.time is None else max(0.,stamp-self.time)
        self.time=stamp
        result=dict(context)
        for key in self.keys:
            raw=context.get(key,0.)
            raw=float(raw) if type(raw) in (int,float) and math.isfinite(raw) else 0.
            # High activation .65, release .45; transient UNKNOWN fades rather than resets.
            old=self.values[key]*math.exp(-dt/.45)
            if raw>=.65: self.confirmed[key]=stamp
            # Short windows often lose the discriminating word on the next hop.
            # Retain its activation for at most 600 ms within this phrase only.
            if stamp-self.confirmed[key]<.6: old=max(old,.65)
            if context.get('short_response_type') in ('ACK','AGREEMENT','NEGATIVE'):
                old=0.; self.confirmed[key]=-100.
            self.values[key]=max(old,raw) if raw>=.65 or old>=.45 else raw
            result[key]=max(0.,min(1.,self.values[key]))
        result['phrase_id']=phrase_id
        labels=('QUESTION','EXCLAMATION','CALLOUT','FAREWELL')
        index=max(range(4),key=lambda i:result[self.keys[i]])
        if result[self.keys[index]]>=.65 and result.get('short_response_type','UNKNOWN')=='UNKNOWN':
            result['phrase_type']=labels[index]
        category=result.get('phrase_type','UNKNOWN')
        self.repeats=self.repeats+1 if category==self.category else 1; self.category=category
        if result[self.keys[index]]>=.65 and result.get('context_confidence',0)>.2:
            result['context_confidence']=max(result['context_confidence'],min(.85,.5+.1*(self.repeats-1)))
        result['classification_repeats']=self.repeats
        return result
