"""Parallel voice slots, callback-owned pre-roll and switching-only crossfade."""
import numpy as np
from dataclasses import replace
from .base import AudioProcessor
from .female_dsp import FemaleDSPProcessor
from .ai_voice import AIVoiceProcessor
from src.vc.config import VOICE_MODES


class VoiceRouter(AudioProcessor):
    def __init__(self, dsp=None, ai=None, mode='original'):
        self.dsp = dsp or FemaleDSPProcessor()
        if isinstance(self.dsp,FemaleDSPProcessor):
            self.dsp.set_parameters(replace(self.dsp.parameters,mode='female_dsp'))
        self.ai = ai or AIVoiceProcessor()
        if mode not in VOICE_MODES:
            raise ValueError('Invalid voice mode')
        self.mode = mode
        self._current = mode
        self._target = mode
        self._last = 0.0
        self._dsp_preroll = 0

    def select(self, mode):
        if mode not in VOICE_MODES:
            raise ValueError('Invalid voice mode')
        self.mode = mode
        if mode == 'ai_voice':
            self.ai.bridge.load()
        # Leaving AI: callback continues consuming until the fade completes.
        # It alone clears output; never reset SPSC reader from the GUI.

    def configure_ai(self, parameters):
        self.ai.bridge.configure(parameters)
        if self.mode == 'ai_voice':
            self.select(self.mode)

    def restart_ai(self):
        self.ai.bridge.stop()
        if self.mode == 'ai_voice':
            self.select('ai_voice')

    def prepare(self, sample_rate, max_frames):
        self.dsp.prepare(sample_rate, max_frames)
        self.ai.prepare(sample_rate, max_frames)
        self._source = np.empty((max_frames,1),dtype=np.float32)
        self._old = np.empty_like(self._source)
        self._new = np.empty_like(self._source)
        self._mix = np.empty_like(self._source)
        self._ramp = np.empty_like(self._source)
        self._indices = np.arange(1,max_frames+1,dtype=np.float32).reshape(-1,1)
        self._fade_frames = round(sample_rate*self.ai.bridge.parameters.crossfade_ms/1000)
        self._position = self._fade_frames
        self._current = self._target = 'original'
        self._last = 0.0
        self._dsp_preroll = 0
        if self.mode == 'ai_voice':
            self.select(self.mode)
            # Complete the start epoch on the control thread before the first
            # microphone callback; never wait or reset from that callback.
            if self.ai.bridge.parameters.mute_during_startup and self.ai.bridge.alive and self.ai.bridge.status=='Ready':
                self.ai.bridge.activate_prepared()

    def reset(self):
        self.dsp.reset()
        self.ai.reset()

    def stop(self):
        self.ai.stop()

    @property
    def algorithmic_latency_samples(self):
        return self.dsp.algorithmic_latency_samples if self.mode == 'female_dsp' else 0

    def _render(self, mode, source, target, sample_rate):
        np.copyto(target, source)
        if mode == 'female_dsp':
            return self.dsp.process(target, sample_rate)
        if mode == 'ai_voice':
            return self.ai.process(target, sample_rate)
        return target

    def process(self, audio, sample_rate):
        bridge = self.ai.bridge
        desired = self.mode
        # The callback owns mode activity while streaming. A GUI request must
        # never race a departing callback that finishes an earlier fade and
        # disables AI. Reconcile here before generation/readiness checks.
        if desired == 'ai_voice' and not bridge.active and bridge.status != 'Error':
            bridge.set_active(True)
        muted_start=(desired=='ai_voice' and bridge.parameters.mute_during_startup and (bridge.status!='Error' or bridge.parameters.delivery=='utterance'))
        if desired == 'ai_voice' and (bridge.status != 'Ready' or not self.ai.supported
                                     or bridge.ack_generation != bridge.generation):
            if muted_start and (self.ai.supported or bridge.parameters.delivery=='utterance'):
                audio.fill(0);return audio
            desired = 'original'
        if (desired == 'ai_voice' and bridge.parameters.enhancer in ('mossformer','voicefixer')
                and not self.ai.supported):
            # Quality mode prefers silence over leaking original speech.
            audio.fill(0);return audio
        if desired != self._target:
            if self._position < self._fade_frames and self._position >= self._fade_frames/2:
                self._current = self._target
            self._target = desired
            self._position = 0
            if desired == 'female_dsp' and self._current != desired:
                self.dsp.reset()
                self._dsp_preroll = getattr(self.dsp,'algorithmic_latency_samples',0)
        n = len(audio)
        source, old, new = self._source[:n], self._old[:n], self._new[:n]
        np.copyto(source, audio)
        if self._current == self._target:
            result = self._render(self._current, source, new, sample_rate)
            np.copyto(audio, result)
        else:
            waiting_dsp = self._target == 'female_dsp' and self._dsp_preroll > 0
            target_audio = self._render(self._target, source, new, sample_rate)
            if waiting_dsp:
                self._dsp_preroll = max(0,self._dsp_preroll-n)
            if (self._target == 'ai_voice' and not self.ai._primed) or waiting_dsp:
                if muted_start and self._current=='original':audio.fill(0)
                else:np.copyto(audio, self._render(self._current, source, old, sample_rate))
                self._position = 0  # keep old route during either slot's pre-roll
            else:
                if muted_start and self._current=='original':old.fill(0)
                elif self._current == 'ai_voice' and bridge.status != 'Ready':
                    old.fill(self._last)
                else:
                    self._render(self._current, source, old, sample_rate)
                ramp, mix = self._ramp[:n], self._mix[:n]
                np.add(self._indices[:n], self._position, out=ramp)
                np.divide(ramp, self._fade_frames, out=ramp)
                np.minimum(ramp, 1, out=ramp)
                np.subtract(target_audio, old, out=mix)
                np.multiply(mix, ramp, out=mix)
                np.add(old, mix, out=audio)
                if self._position == 0:
                    bridge.crossfade_count += 1  # worker logs off callback
                self._position += n
                if self._position >= self._fade_frames:
                    self._current = self._target
                    if self._current != 'ai_voice':
                        bridge.set_active(False)
                        bridge.output.discard()
        self._last = float(audio[-1,0])
        # Also covers cancelling AI during pre-roll (no fade was started).
        if self._current == self._target != 'ai_voice' and self.mode != 'ai_voice' and bridge.active:
            bridge.set_active(False)
            bridge.output.discard()
        return audio
