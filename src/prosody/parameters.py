from dataclasses import dataclass, replace
import math

PRESETS = {
    'Natural': dict(range_expansion=.10, rise_boost=.05, fall_boost=.02, onset_lift=.15,
        ending_strength=.05, energy_dynamics=.05, min_pitch=-.5, max_pitch=.6, attack_ms=90., release_ms=140.),
    'Anime Light': dict(range_expansion=.40, rise_boost=.25, fall_boost=.10, onset_lift=.35,
        ending_strength=.25, energy_dynamics=.18, min_pitch=-.8, max_pitch=1.2, attack_ms=70., release_ms=110.),
    'Anime Expressive': dict(range_expansion=.65, rise_boost=.45, fall_boost=.20, onset_lift=.50,
        ending_strength=.45, energy_dynamics=.30, min_pitch=-1.2, max_pitch=1.5, attack_ms=50., release_ms=90.),
}


@dataclass(frozen=True)
class ProsodyParameters:
    enabled: bool = False
    preset: str = 'Anime Light'
    amount: float = 60.
    pitch_range: float = 50.
    energy: float = 40.
    ending_lift: bool = False
    ending_emphasis: bool = True
    rule_version: int = 2
    range_expansion: float = .40
    rise_boost: float = .25
    fall_boost: float = .10
    ending_strength: float = .25
    onset_lift: float = .35
    energy_dynamics: float = .18
    min_pitch: float = -.8
    max_pitch: float = 1.2
    max_gain_db: float = 1.0
    attack_ms: float = 70.
    release_ms: float = 110.
    update_ms: float = 30.
    baseline_ms: float = 700.
    confidence: float = .8
    min_silence_ms: float = 180.
    history_ms: float = 200.
    engine: str = 'rule_v2'
    asr_backend: str = 'sherpa_onnx'
    asr_model: str = 'reazon_int8'
    asr_threads: int = 1
    text_influence: float = 100.
    transcript_logging: bool = False
    asr_update_ms: float = 300.
    asr_window_ms: float = 400.
    text_strategy: str = 'hybrid'

    def __post_init__(self):
        if any(type(getattr(self,k)) is not bool for k in ('enabled','ending_lift','ending_emphasis')):
            raise ValueError('Prosody switches must be boolean')
        if self.preset not in (*PRESETS, 'Custom'):
            raise ValueError('Invalid prosody preset')
        for name, low, high in self.bounds():
            value=getattr(self,name)
            if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:
                raise ValueError(f'{name} must be {low}..{high}')
        if self.rule_version != 2:
            raise ValueError('Unsupported rule version')
        if self.engine not in ('off','rule_v2','text_v3') or self.asr_backend!='sherpa_onnx' or self.asr_model!='reazon_int8':
            raise ValueError('Unsupported text prosody engine/model')
        if type(self.asr_threads) is not int or not 1<=self.asr_threads<=4 or type(self.transcript_logging) is not bool:
            raise ValueError('Invalid ASR configuration')
        if self.text_strategy not in ('legacy','direct','modulation','hybrid','events'):
            raise ValueError('Unsupported text calibration strategy')
        if self.asr_update_ms not in (300,600,900) or self.asr_window_ms not in (250,400,600,800,1000,1500,2000):
            raise ValueError('Unsupported ASR window/update')
        if self.update_ms not in (20,30,40,50):
            raise ValueError('Update interval must be 20, 30, 40 or 50 ms')

    @staticmethod
    def bounds():
        return [('amount',0,100),('pitch_range',0,100),('energy',0,100),('text_influence',0,100),
            ('asr_update_ms',300,900),('asr_window_ms',250,2000),
            ('range_expansion',0,.7),('rise_boost',0,.5),('fall_boost',0,.3),
            ('ending_strength',0,.5),('onset_lift',0,.5),('history_ms',100,300),
            ('energy_dynamics',0,.3),('min_pitch',-1.2,0),('max_pitch',0,1.5),
            ('max_gain_db',0,1.5),('attack_ms',50,100),('release_ms',80,150),
            ('update_ms',20,50),('baseline_ms',500,800),('confidence',.7,.95),
            ('min_silence_ms',150,500)]

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data,dict):
            return cls()
        default=cls(); valid={}
        for name in ('enabled','ending_lift','ending_emphasis'):
            valid[name]=data.get(name) if type(data.get(name)) is bool else getattr(default,name)
        valid['preset']=data.get('preset') if data.get('preset') in (*PRESETS,'Custom') else default.preset
        for name,low,high in cls.bounds():
            value=data.get(name)
            valid[name]=value if type(value) in (int,float) and math.isfinite(value) and low<=value<=high else getattr(default,name)
        if valid['update_ms'] not in (20,30,40,50):
            valid['update_ms']=30.
        valid['engine']=data.get('engine') if data.get('engine') in ('off','rule_v2','text_v3') else 'rule_v2'
        valid['asr_threads']=data.get('asr_threads') if type(data.get('asr_threads')) is int and 1<=data['asr_threads']<=4 else 1
        valid['transcript_logging']=data.get('transcript_logging') if type(data.get('transcript_logging')) is bool else False
        if valid['asr_update_ms'] not in (300,600,900): valid['asr_update_ms']=300.
        if valid['asr_window_ms'] not in (250,400,600,800,1000,1500,2000): valid['asr_window_ms']=1000.
        valid['text_strategy']=data.get('text_strategy') if data.get('text_strategy') in ('legacy','direct','modulation','hybrid','events') else 'hybrid'
        # Named v1 presets adopt v2 coefficients; Custom keeps manual values.
        if data.get('rule_version') != 2 and valid['preset'] in PRESETS:
            valid.update(PRESETS[valid['preset']])
        return cls(**valid)


def load_preset(name, parameters=None):
    if name not in PRESETS:
        raise ValueError('Unknown prosody preset')
    return replace(parameters or ProsodyParameters(),preset=name,**PRESETS[name])
