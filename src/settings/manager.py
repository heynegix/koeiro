from dataclasses import asdict, dataclass, field
import json
import logging
import math
from pathlib import Path

from src.processors.dsp_parameters import DSPParameters, MODES, QUALITIES
from src.presets.female_presets import PRESETS
from src.vc.config import AIParameters, EXPERIMENTS, QUALITY_FACTORS, VOICE_MODES, UTTERANCE_ENHANCERS
from src.prosody.parameters import ProsodyParameters
from src.vc.voice_library import DEFAULT_VOICE_ID

log = logging.getLogger(__name__)
BUFFERS = (64, 128, 256, 512, 1024)
SAMPLE_RATES = (44100, 48000)
# A settings file can name a profile this build no longer offers, so the fallback is
# the one voice this build ships rather than a removed character profile.
DEFAULT_AI_MODEL = DEFAULT_VOICE_ID
DEFAULT_DELIVERY = 'utterance'


def _number(value, default, low, high):
    if type(value) not in (int, float) or not low <= value <= high:
        return default
    return float(value) if math.isfinite(value) else default


def _device(value):
    if (isinstance(value, dict) and isinstance(value.get("name"), str)
            and isinstance(value.get("host_api"), str)
            and 0 < len(value["name"]) <= 512 and 0 < len(value["host_api"]) <= 128):
        return {"name": value["name"], "host_api": value["host_api"]}
    return None


@dataclass
class AppSettings:
    input_device: dict | None = None
    output_device: dict | None = None
    sample_rate: int = 48000
    buffer_size: int = 256
    gain_db: float = 0.0
    noise_gate_db: float = -45.0
    monitor: bool = False
    monitor_device: dict | None = None
    monitor_volume_db: float = 0.0
    tutorial_seen: bool = False
    window_size: tuple = (560, 750)
    window_position: tuple | None = None
    dsp_mode: str = "original"
    preset: str = "Original"
    pitch_semitones: float = 3.0
    formant_semitones: float = 2.0
    brightness: float = 60.0
    low_cut: bool = True
    limiter: bool = True
    wet: float = 1.0
    dsp_quality: str = "low_latency"
    voice_mode: str = "ai_voice"
    ai_model: str = DEFAULT_AI_MODEL
    ai_quality: str = "low_latency"
    ai_threads: int = 1
    ai_brightness: float = 50.0
    ai_low_cut: bool = True
    ai_limiter: bool = True
    ai_post_fx: bool = True
    ai_output_wait_ms: float = 39.0
    ai_crossfade_ms: float = 20.0
    ai_pitch: float = 4.0
    ai_delivery: str = DEFAULT_DELIVERY
    ai_enhancer: str = 'lavasr'
    ai_lavasr_denoise: bool = True
    ai_experiment: str = 'none'
    ai_tune_sib_db: float = 3.0
    ai_tune_cons_db: float = 3.0
    ai_tune_caps: float = 1.0
    ai_tune_floor_db: float = 3.0
    ai_tune_excess_db: float = 9.0
    ai_tune_mid: float = 0.8
    ai_tune_match: float = 0.25
    ai_tune_ptrans: float = 0.20
    ai_tune_pcap: float = 1.0
    ai_tune_combined: bool = True
    ai_tune_level_db: float = -20.0
    prosody: dict = field(default_factory=lambda: asdict(ProsodyParameters()))

    def prosody_parameters(self):
        return ProsodyParameters.from_dict(self.prosody)

    def ai_parameters(self):
        return AIParameters(quality=self.ai_quality, threads=self.ai_threads, model=self.ai_model,
                            brightness=self.ai_brightness, low_cut=self.ai_low_cut,
                            limiter=self.ai_limiter, post_fx=self.ai_post_fx,
                            output_wait_ms=self.ai_output_wait_ms,crossfade_ms=self.ai_crossfade_ms,pitch=self.ai_pitch,
                            delivery=self.ai_delivery,enhancer=self.ai_enhancer,
                            lavasr_denoise=self.ai_lavasr_denoise,
                            experiment=self.ai_experiment,
                            tune_sib_db=self.ai_tune_sib_db, tune_cons_db=self.ai_tune_cons_db,
                            tune_caps=self.ai_tune_caps, tune_floor_db=self.ai_tune_floor_db,
                            tune_excess_db=self.ai_tune_excess_db, tune_mid=self.ai_tune_mid,
                            tune_match=self.ai_tune_match, tune_ptrans=self.ai_tune_ptrans,
                            tune_pcap=self.ai_tune_pcap, tune_combined=self.ai_tune_combined,
                            tune_level_db=self.ai_tune_level_db)

    def dsp_parameters(self):
        return DSPParameters(self.dsp_mode, self.pitch_semitones, self.formant_semitones,
                             self.brightness, self.low_cut, self.limiter, self.wet, self.dsp_quality)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("Settings root must be a JSON object")
        defaults = cls()
        values = {}
        values['prosody']=asdict(ProsodyParameters.from_dict(data.get('prosody')))
        for field in ("input_device", "output_device"):
            values[field] = _device(data.get(field))
        for field, choices in (("sample_rate", SAMPLE_RATES), ("buffer_size", BUFFERS)):
            value = data.get(field)
            values[field] = value if type(value) is int and value in choices else getattr(defaults, field)
        values["gain_db"] = _number(data.get("gain_db"), 0.0, -20, 20)
        values["noise_gate_db"] = _number(data.get("noise_gate_db"), -45.0, -80, -10)
        monitor = data.get("monitor")
        values["monitor"] = monitor if type(monitor) is bool else False
        values["monitor_device"] = _device(data.get("monitor_device"))
        values["monitor_volume_db"] = _number(data.get("monitor_volume_db"), 0.0, -30, 6)
        seen = data.get("tutorial_seen")
        values["tutorial_seen"] = seen if type(seen) is bool else False
        for field, choices in (("dsp_mode", MODES), ("dsp_quality", QUALITIES),
                                ("preset", (*PRESETS, "Custom"))):
            value = data.get(field)
            values[field] = value if isinstance(value, str) and value in choices else getattr(defaults, field)
        for field, low, high in (("pitch_semitones", -12, 12), ("formant_semitones", -6, 6),
                                 ("brightness", 0, 100), ("wet", 0, 1)):
            values[field] = _number(data.get(field), getattr(defaults, field), low, high)
        for field in ("low_cut", "limiter"):
            value = data.get(field)
            values[field] = value if type(value) is bool else getattr(defaults, field)
        # A file written before voice_mode existed carries only dsp_mode, so that value is
# migrated. A file without either key gets the current default rather than dsp_mode's.
        if 'voice_mode' in data:
            mode = data.get('voice_mode')
        elif 'dsp_mode' in data:
            mode = values['dsp_mode']
        else:
            mode = defaults.voice_mode
        values['voice_mode'] = mode if isinstance(mode, str) and mode in VOICE_MODES else defaults.voice_mode
        from src.vc.models import VOICE_PROFILES, default_voice_id
        for field, choices in (('ai_model', tuple(VOICE_PROFILES)), ('ai_quality', tuple(QUALITY_FACTORS))):
            value = data.get(field)
            values[field] = value if isinstance(value, str) and value in choices else getattr(defaults, field)
        if values['ai_model'] not in VOICE_PROFILES:
            # A voice deleted after the settings were saved (or a fresh install
            # with no voices yet) falls back to the default, then any voice.
            # None means "no voice yet": AIParameters stays constructible and
            # the GUI shows a registration prompt instead of crashing.
            values['ai_model'] = default_voice_id() or next(iter(VOICE_PROFILES), None)
        threads = data.get('ai_threads')
        values['ai_threads'] = threads if type(threads) is int and 1 <= threads <= 4 else defaults.ai_threads
        values['ai_brightness'] = _number(data.get('ai_brightness'), 50.0, 0, 100)
        values['ai_output_wait_ms'] = _number(data.get('ai_output_wait_ms'),39.0,20,156)
        values['ai_crossfade_ms'] = _number(data.get('ai_crossfade_ms'),20.0,20,100)
        values['ai_pitch'] = _number(data.get('ai_pitch'),4.0,-12,12)
        for name,choices in [('ai_delivery',('streaming','utterance')),('ai_enhancer',('none','lavasr'))]:
            value=data.get(name)
            values[name]=value if isinstance(value,str) and value in choices else getattr(defaults,name)
        # This build ships the LavaSR route only, so the two fields cannot disagree.
        if values['ai_delivery'] == 'utterance':
            values['ai_enhancer'] = 'lavasr'
        elif values['ai_enhancer'] == 'lavasr':
            values['ai_delivery'] = 'utterance'
        experiment = data.get('ai_experiment')
        values['ai_experiment'] = experiment if isinstance(experiment, str) and experiment in EXPERIMENTS else 'none'
        if values['ai_experiment'] != 'none':
            # Experimental variants only exist on the utterance+LavaSR route.
            values['ai_delivery'] = 'utterance'
            values['ai_enhancer'] = 'lavasr'
        denoise = data.get('ai_lavasr_denoise')
        values['ai_lavasr_denoise'] = denoise if type(denoise) is bool else True
        for field_name, default, low, high in (
                ('ai_tune_sib_db', 3.0, 0, 6), ('ai_tune_cons_db', 3.0, 0, 6),
                ('ai_tune_caps', 1.0, 0.5, 1.5), ('ai_tune_floor_db', 3.0, 0, 6),
                ('ai_tune_excess_db', 9.0, 3, 24), ('ai_tune_mid', 0.8, 0, 1),
                ('ai_tune_match', 0.25, 0, 0.5), ('ai_tune_ptrans', 0.20, 0, 0.5),
                ('ai_tune_pcap', 1.0, 0, 2), ('ai_tune_level_db', -20.0, -26, -14)):
            value = data.get(field_name)
            values[field_name] = (value if type(value) in (int, float)
                                  and low <= value <= high else default)
        combined = data.get('ai_tune_combined')
        values['ai_tune_combined'] = combined if type(combined) is bool else True
        for field in ('ai_low_cut', 'ai_limiter', 'ai_post_fx'):
            value = data.get(field)
            values[field] = value if type(value) is bool else getattr(defaults, field)
        for field, bounds in (("window_size", ((480, 4096), (650, 4096))),
                              ("window_position", ((-32768, 32768), (-32768, 32768)))):
            value = data.get(field)
            if (isinstance(value, (list, tuple)) and len(value) == 2
                    and all(type(v) is int and low <= v <= high for v, (low, high) in zip(value, bounds))):
                values[field] = tuple(value)
        return cls(**values)


class SettingsManager:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self):
        try:
            if self.path.stat().st_size > 64 * 1024:
                raise ValueError("Settings file too large")
            return AppSettings.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            return AppSettings()
        except (OSError, ValueError, TypeError, RecursionError):
            log.warning("Invalid/unreadable settings; using defaults", exc_info=True)
            return AppSettings()

    def save(self, settings):
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            validated = AppSettings.from_dict(asdict(settings))
            temporary.write_text(json.dumps(asdict(validated), ensure_ascii=False,
                                            indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(self.path)
            return True
        except (OSError, ValueError, TypeError):
            log.exception("Failed to save settings: %s", self.path)
            return False
