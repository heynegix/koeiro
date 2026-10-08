import hashlib
import json
from pathlib import Path
import time

import numpy as np
from .base import VoiceConversionBackend
from .vst_state import set_model_preset


class BeatriceVSTBackend(VoiceConversionBackend):
    """Official unmodified VST in an isolated host; never links private APIs.

    Native model resampling and state continuity are owned by the VST. Asset
    installation is explicit and local; no network/model acquisition here.
    """
    sample_rate = 48000

    def __init__(self, chunk_factor=1):
        if type(chunk_factor) is not int or chunk_factor not in (1,2,4):
            raise ValueError('Invalid AI chunk factor')
        self.chunk_samples = 624*chunk_factor
        self.plugin = None
        self.stats = {}

    def load(self, model_path):
        started = time.perf_counter()
        folder = Path(model_path).resolve()
        runtime = json.loads((folder/'runtime.json').read_text(encoding='utf-8'))
        manifest = json.loads((folder/'beatrice-manifest.json').read_text(encoding='utf-8'))
        assets = folder/'beatrice'
        for name, expected in manifest['files'].items():
            path = (assets/name).resolve()
            if not path.is_relative_to(assets) or not path.is_file():
                raise ValueError('Beatrice assets missing; see README setup')
            if path.stat().st_size != expected['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=expected['sha256']:
                raise ValueError('Beatrice asset checksum mismatch')
        plugin_path, config = (folder/runtime['plugin']).resolve(), (folder/runtime['model']).resolve()
        if not plugin_path.is_relative_to(assets) or not config.is_relative_to(assets):
            raise ValueError('Invalid Beatrice asset path')
        voice, pitch = runtime['voice'], runtime['pitch']
        if type(voice) is not int or not 0<=voice<100 or type(pitch) not in (int,float) or not -12<=pitch<=12:
            raise ValueError('Invalid Beatrice voice parameters')
        from pedalboard import load_plugin
        self.plugin = load_plugin(str(plugin_path))
        self.plugin.preset_data = set_model_preset(self.plugin.preset_data, config, voice, pitch)
        self._config=config
        self.stats = {'backend':'Beatrice 2 / official VST3 CPU', 'name':runtime['name'],
                      'sample_rate':48000,'chunk_samples':self.chunk_samples,'voice':voice,'pitch':pitch,
                      'model_bytes':sum(v['bytes'] for k,v in manifest['files'].items() if k.endswith('.bin')),
                      'load_seconds':time.perf_counter()-started,'model_alignment_delay_ms':None,
                      'resampler_delay_ms':0,'internal_resampling':'Included in native inference timing',
                      'license':manifest['license'],'reported_latency_samples':self.plugin.reported_latency_samples}

    def select_profile(self, selected):
        """Load-time only: profile changes restart the isolated worker."""
        voice,pitch,formant=selected['voice'],selected['pitch'],selected['formant']
        if type(voice) is not int or not 0<=voice<100 or not np.isfinite([pitch,formant]).all() or not -12<=pitch<=12 or not -2<=formant<=2:
            raise ValueError('Invalid Beatrice voice profile')
        self.plugin.preset_data=set_model_preset(self.plugin.preset_data,self._config,voice,pitch)
        self.plugin.formant_shift_st=formant
        self.stats.update(name=selected['name'],voice=voice,pitch=pitch,formant=formant)

    def reset(self):
        if self.plugin is not None:
            self.plugin.reset()

    def set_pitch(self, target):
        """Worker-only native automation; retain model and neural stream state."""
        if type(target) not in (int,float) or not np.isfinite(target) or not -12 <= target <= 12:
            raise ValueError('Invalid Beatrice pitch')
        current = self.stats['pitch']
        # Official parameter is quantized to 1/8 semitone. One step per chunk
        # avoids abrupt slider jumps; no preset/model reload in the hot path.
        value = round((current+max(-.125,min(.125,target-current)))*8)/8
        if value != current:
            self.plugin.pitch_shift_st = value
            self.stats['pitch'] = value

    def warmup(self):
        started = time.perf_counter()
        # The official VST skips its network for all-zero input. Use a tiny
        # deterministic signal here to actually warm inference, then reset it.
        signal = (.005*np.sin(np.arange(self.chunk_samples)*.071)).astype(np.float32)
        for _ in range(12):
            self.process_chunk(signal)
        self.stats['warmup_seconds'] = time.perf_counter()-started
        self.reset()

    def process_chunk(self, audio):
        if self.plugin is None:
            raise RuntimeError('Beatrice is not loaded')
        if audio.dtype!=np.float32 or audio.ndim!=1 or len(audio)!=self.chunk_samples or not np.isfinite(audio).all():
            raise ValueError('Beatrice requires one exact finite float32 mono chunk')
        result = self.plugin.process(audio,48000,buffer_size=self.chunk_samples,reset=False).reshape(-1)
        if len(result)!=len(audio) or not np.isfinite(result).all():
            raise RuntimeError('Invalid Beatrice output')
        return np.clip(result,-1,1).astype(np.float32)

    def unload(self):
        self.plugin = None

    def get_stats(self):
        return dict(self.stats)
