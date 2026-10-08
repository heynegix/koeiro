"""Bounded Japanese partial classification; no semantic model or transcript logs."""
from dataclasses import dataclass, asdict
import math
import re
import unicodedata


@dataclass(frozen=True)
class TextContext:
    partial_text: str = ''
    final_text: str = ''
    stable_prefix: str = ''
    unstable_suffix: str = ''
    context_confidence: float = 0.
    question_probability: float = 0.
    exclamation_probability: float = 0.
    callout_probability: float = 0.
    farewell_probability: float = 0.
    reaction_probability: float = 0.
    phrase_type: str = 'UNKNOWN'
    short_response_type: str = 'UNKNOWN'
    audio_timestamp: float = 0.
    created_at: float = 0.
    last_updated: float = 0.
    age_ms: float = 0.
    repeat_count: int = 0


class TextContextAnalyzer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = ''; self.repeats = 0; self.created = 0.

    def update(self, text, timestamp, now, final=False, confidence=None):
        if (not isinstance(text,str) or len(text)>512 or not math.isfinite(timestamp)
                or not math.isfinite(now) or timestamp>now+.1):
            self.reset(); return TextContext()
        text=unicodedata.normalize('NFKC',text).strip()
        if not text or '\ufffd' in text or any(unicodedata.category(c)=='Cc' for c in text):
            self.reset(); return TextContext()
        text=text[-160:]
        # Ignore decoder garbage rather than guessing a phrase type.
        if not re.search('[ぁ-んァ-ン一-龥]',text):
            self.reset(); return TextContext()
        prefix=''
        for a,b in zip(self.previous,text):
            if a!=b: break
            prefix+=a
        self.repeats=self.repeats+1 if text==self.previous else 1
        if not self.created: self.created=now
        stability=min(.90,.35+.15*self.repeats+.20*len(prefix)/max(1,len(text)))
        if final: stability=.95
        if confidence is not None:
            stability=min(stability,max(0.,min(1.,float(confidence)))) if math.isfinite(confidence) else 0.
        plain=re.sub(r'[\s。、!?]', '',text)
        q=.95 if '?' in text else .65 if re.search('(かな|なの|ですか|ますか|いいの|だろう|ほんとに|本当に)$',plain) else .45 if re.search('(どう|何|なに|なんで|なぜ|どこ|いつ|誰)',plain) else 0.
        ex=.9 if '!' in text else .7 if re.search('^(やった|すごい|えっ|うそ|わあ|わー)',plain) else .4 if plain in ('マジ','ほんと','本当') else 0.
        call=.8 if re.match('^(ねえ|ねぇ|ちょっと|あの|もしもし)',plain) else 0.
        farewell=.9 if re.search('(またね|じゃあね|ばいばい|バイバイ|おやすみ|またあとで)$',plain) else 0.
        short=({'うん':'ACK','はい':'ACK','ううん':'NEGATIVE','いいえ':'NEGATIVE',
                'え':'SURPRISE','えっ':'SURPRISE','マジ':'SURPRISE','ほんと':'SURPRISE',
                '本当':'SURPRISE','そう':'AGREEMENT','なるほど':'AGREEMENT','へえ':'REACTION'}.get(plain,'UNKNOWN'))
        phrase='SHORT_RESPONSE' if short!='UNKNOWN' else max(
            [('QUESTION',q),('EXCLAMATION',ex),('CALLOUT',call),('FAREWELL',farewell)],key=lambda v:v[1])[0]
        if max(q,ex,call,farewell)<.4 and short=='UNKNOWN': phrase='STATEMENT'
        self.previous=text
        return TextContext(text,text if final else '',prefix,text[len(prefix):],stability,q,ex,call,
            farewell,.8 if short!='UNKNOWN' else 0.,phrase,short,timestamp,self.created,now,
            max(0.,(now-timestamp)*1000),self.repeats)

    def snapshot(self, *args, **kwargs):
        return asdict(self.update(*args,**kwargs))
