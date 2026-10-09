from dataclasses import dataclass
import math
from .models import is_meanvc2
from .models import VOICE_PROFILES
from .voice_library import DEFAULT_VOICE_ID

QUALITY_FACTORS = {'low_latency': 1, 'balanced': 2, 'stable': 4}
VOICE_MODES = ('original', 'female_dsp', 'ai_voice')
# Compute device for neural inference. 'auto' uses CUDA when the worker torch
# can see a GPU, otherwise CPU. 'cuda' fails loudly without one; 'cpu' forces
# the validated CPU route and hides GPUs for deterministic behaviour.
DEVICES = ('auto', 'cpu', 'cuda')
# This build ships the LavaSR restoration only. The models for the others are still
# on disk, but nothing can select them. `delivery` is the internal value the worker
# understands; models.DELIVERY_MODES maps the UI labels onto it.
DELIVERY_MODES = ('streaming', 'utterance')
ENHANCERS = ('none', 'lavasr')
UTTERANCE_ENHANCERS = ('lavasr',)
# Listening-comparison variants. One integrated mode applies every experimental
# improvement at once (pause refresh, 3-block grouping, ending-focused energy
# repair, high-frequency blend, adaptive segmentation with overlap, input
# levelling) on the utterance+LavaSR route; a second natural-leaning variant
# keeps the tuned 720ms grouping while bundling floor preservation, combined
# micro-pitch repair, tail keeping and a gentle output finish; a third
# low-delay variant shortens the inference groups and repair lookahead for
# latency at listening-verified quality cost; 'none' is the
# shipped behaviour.
EXPERIMENTS = ('none', 'all', 'natural', 'lowdelay', 'fastest')


@dataclass(frozen=True)
class AIParameters:
    quality: str = 'low_latency'
    threads: int = 1
    device: str = 'auto'
    model: str = DEFAULT_VOICE_ID
    brightness: float = 50.0
    low_cut: bool = True
    limiter: bool = True
    post_fx: bool = True
    output_wait_ms: float = 39.0
    crossfade_ms: float = 20.0
    pitch: float = 4.0
    delivery: str = 'streaming'
    enhancer: str = 'none'
# Optional LavaSR upstream denoise. Default off: the measured comparison found no
    # benefit from it on this material, and it costs time inside an already RTF-sensitive path.
    lavasr_denoise: bool = False
    # Listening-comparison variant. Only the utterance+LavaSR route honours it; any
    # other combination falls back to 'none' rather than raising on a restored setting.
    experiment: str = 'none'
    # Voice tuning sliders. Only the natural comparison route honours them; the
    # shipped routes and the all-in comparison ignore them entirely. Defaults
    # reproduce the validated natural recipe exactly.
    tune_sib_db: float = 3.0
    tune_cons_db: float = 3.0
    tune_caps: float = 1.0
    tune_floor_db: float = 3.0
    tune_excess_db: float = 9.0
    tune_mid: float = 0.8
    tune_match: float = 0.25
    tune_ptrans: float = 0.20
    tune_pcap: float = 1.0
    tune_combined: bool = True
    tune_level_db: float = -20.0

    def __post_init__(self):
        if self.delivery not in DELIVERY_MODES or self.enhancer not in ENHANCERS:
            raise ValueError('Unsupported AI delivery/enhancer')
        if self.experiment not in EXPERIMENTS:
            raise ValueError('Unsupported AI experiment')
        if self.model is None:
            # No voice registered yet (fresh install, or every voice removed).
            # Stay constructible and neutral; starting AI reports a clean Error
            # with a registration prompt instead of crashing the app.
            object.__setattr__(self,'delivery','streaming')
            object.__setattr__(self,'enhancer','none')
            object.__setattr__(self,'experiment','none')
            object.__setattr__(self,'lavasr_denoise',False)
        elif not is_meanvc2(self.model):
            object.__setattr__(self,'delivery','streaming')
            object.__setattr__(self,'enhancer','none')
        elif self.enhancer in UTTERANCE_ENHANCERS:
            object.__setattr__(self,'delivery','utterance')
        elif self.delivery == 'utterance':
            # An utterance route with no restoration model is not something this
            # build can run; fall back rather than raising on a restored setting.
            object.__setattr__(self,'delivery','streaming')
            object.__setattr__(self,'enhancer','none')
        if self.experiment != 'none' and (self.delivery != 'utterance' or self.enhancer != 'lavasr'):
            # Experimental variants only exist on the utterance+LavaSR route.
            object.__setattr__(self,'experiment','none')
        if self.enhancer != 'lavasr':
            object.__setattr__(self,'lavasr_denoise',False)
        if self.quality not in QUALITY_FACTORS or (self.model is not None and self.model not in VOICE_PROFILES):
            raise ValueError('Unsupported AI model/quality')
        if self.device not in DEVICES:
            raise ValueError('Unsupported AI device')
        # Validate before the neutral override below, so an out-of-range request is
        # rejected rather than being silently replaced by the neutral value.
        if type(self.threads) is not int or not 1 <= self.threads <= 4:
            raise ValueError('AI CPU threads must be 1..4')
        if type(self.brightness) not in (int, float) or not math.isfinite(self.brightness) or not 0 <= self.brightness <= 100:
            raise ValueError('AI brightness must be 0..100')
        if any(type(v) is not bool for v in (self.low_cut, self.limiter, self.post_fx, self.lavasr_denoise,
                                                 self.tune_combined)):
            raise ValueError('AI FX switches must be boolean')
        for name, low, high in [('tune_sib_db', 0, 6), ('tune_cons_db', 0, 6),
                                ('tune_caps', 0.5, 1.5), ('tune_floor_db', 0, 6),
                                ('tune_excess_db', 3, 24), ('tune_mid', 0, 1),
                                ('tune_match', 0, 0.5), ('tune_ptrans', 0, 0.5),
                                ('tune_pcap', 0, 2), ('tune_level_db', -26, -14)]:
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) \
                    or not low <= value <= high:
                raise ValueError(f'{name} must be {low}..{high}')
        for name, low, high in [('output_wait_ms',20,156), ('crossfade_ms',20,100), ('pitch',-12,12)]:
            value = getattr(self, name)
            if type(value) not in (int,float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'{name} must be {low}..{high}')
        if is_meanvc2(self.model):
            for name,value in [('pitch',0.),('brightness',50.),('low_cut',False),('limiter',False),('post_fx',False)]:
                object.__setattr__(self,name,value)

    @property
    def chunk_frames(self):
        if is_meanvc2(self.model):return 7680 # ASR stride: 160ms, VC blocks: 120+40ms.
        return 624*QUALITY_FACTORS[self.quality]

    @property
    def queue_chunks(self):
        if is_meanvc2(self.model):return VOICE_PROFILES[self.model].get('queue_chunks',8)
        # Fixed capacity for the chosen pre-roll, plus one writable chunk.
        # Never grows during playback; legacy 39 ms remains four chunks.
        return max(4, math.ceil(self.startup_frames/self.chunk_frames)+1)

    @property
    def mute_during_startup(self):
        if self.delivery=='utterance':return True
        return VOICE_PROFILES.get(self.model, {}).get('mute_during_startup',False)

    @property
    def startup_chunks(self):
        if is_meanvc2(self.model):return VOICE_PROFILES[self.model].get('startup_chunks',6)
        # A 13 ms chunk needs headroom for Windows' bursty shared-mode delivery.
        # This is fixed startup buffering, never an expanding backlog.
        return 3 if self.quality == 'low_latency' else 2

    @property
    def startup_frames(self):
        # Exact sample target allows 25/30/35/39 ms comparisons. Longer quality
        # modes retain the established two-chunk safety margin.
        requested=round(self.output_wait_ms*48)
        if is_meanvc2(self.model):return max(requested,self.startup_chunks*self.chunk_frames)
        return requested if self.quality=='low_latency' else max(requested,2*self.chunk_frames)
