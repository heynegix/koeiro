from src.vc.models import is_meanvc2
import numpy as np
from .base import AudioProcessor
from src.vc.bridge import AIBridge


class AIVoiceProcessor(AudioProcessor):
    """Only non-blocking ring copies and short dropout ramps run on callback."""
    def __init__(self, parameters=None, bridge=None):
        self.bridge = bridge or AIBridge(parameters)
        self._epoch = -1
        self._primed = False
        self._last = 0.0
        self._recovering = True

    def prepare(self, sample_rate, max_frames):
        if sample_rate != 48000:
            self.bridge.error = 'AI Voiceは48 kHz専用です。OriginalへFallback。'
            self.supported = False
            return
        self.supported = True
        self.bridge.prepare()
        self._fade = np.minimum(1, np.arange(1,max_frames+1,dtype=np.float32)/(sample_rate*.005))
        self._decay = np.exp(-np.arange(1,max_frames+1,dtype=np.float32)/(sample_rate*.001))
        self._fade_position = 0
        self._ramp = np.empty(max_frames, dtype=np.float32)
        self._dry = np.empty(max_frames, dtype=np.float32)
        self._mix = np.empty(max_frames, dtype=np.float32)
        self._reset_local()

    def reset(self):
        """Control-side reset; neural state is reset asynchronously by its owner."""
        self.bridge.generation+=1
        self.bridge.input_ready.signal()
        self._reset_local()

    def _reset_local(self):
        self._epoch = -1
        self._primed = False
        self._last = 0.0
        self._recovering = True
        self._fade_position = 0
        self._startup_fade=False
        self._has_played=False

    def stop(self):
        self.bridge.stop()

    def process(self, audio, sample_rate):
        bridge = self.bridge
        if not getattr(self, 'supported', False) or bridge.status != 'Ready' or bridge.ack_generation != bridge.generation:
            if getattr(self,'supported',False) and bridge.parameters.mute_during_startup and (bridge.status!='Error' or bridge.parameters.delivery=='utterance'):audio.fill(0)
            return audio  # Original while loading/error/reset; engine stays alive.
        if self._epoch != bridge.generation:
            bridge.output.discard()  # callback owns output consumer counter
            self._reset_local()
            self._epoch = bridge.generation
        if not is_meanvc2(bridge.parameters.model):
            bridge.prosody.set_active(bridge.active)
            bridge.prosody.submit(audio[:,0])  # independent non-blocking analysis copy
        if not bridge.input.write(audio[:,0]):
            bridge.overruns += 1
            bridge.dropped_frames += len(audio)
            bridge.record_incident(2)
        if bridge.input.available>=bridge.chunk_frames:
            bridge.input_ready.signal()
        if bridge.parameters.delivery=='utterance':
            # This route is ready even while intentionally silent. VoiceRouter
            # must enter AI instead of waiting for streaming pre-roll forever.
            self._primed=True
            if bridge.utterance_output_generation!=bridge.generation:
                bridge.output.discard();audio.fill(0);self._recovering=True;self._fade_position=0
                return audio
            available=min(len(audio),bridge.output.available)
            if available:
                bridge.output.read_into(audio[:available,0])
                if self._recovering:
                    ramp=self._ramp[:available]
                    np.add(self._fade[:available],self._fade_position/240,out=ramp)
                    np.minimum(ramp,1,out=ramp);np.multiply(audio[:available,0],ramp,out=audio[:available,0])
                    self._fade_position+=available
                    if self._fade_position>=240:self._recovering=False
            if available<len(audio):
                audio[available:,0].fill(0);self._recovering=True;self._fade_position=0
            return audio
        starting = not self._primed
        meanvc = is_meanvc2(bridge.parameters.model)
        if not starting:self._has_played=True
        if starting:
            target = bridge.parameters.startup_frames
            if meanvc and self._has_played:
                target = min(target, 2 * bridge.chunk_frames)
            if bridge.output.available >= target:
                self._primed = True
                bridge.preroll_observed_frames = bridge.output.available
                self._startup_fade=not self._has_played
            else:
                if meanvc and self._has_played:
                    np.multiply(self._decay[:len(audio)],self._last,out=audio[:,0])
                    self._last=float(audio[-1,0])
                    return audio
                if bridge.parameters.mute_during_startup:audio.fill(0)
                return audio  # Original continues during initial buffering.
        if self._startup_fade:
            if bridge.parameters.mute_during_startup:self._dry[:len(audio)].fill(0)
            else:np.copyto(self._dry[:len(audio)],audio[:,0])
        if not bridge.output.read_into(audio[:,0]):
            bridge.underruns += 1
            bridge.record_incident(1)
            # Retain playable partial output rather than silencing a whole
            # callback when only its tail is missing (624 vs 256 frame units).
            available=min(len(audio),bridge.output.available)
            if available:
                bridge.output.read_into(audio[:available,0])
                self._last=float(audio[available-1,0])
            missing=len(audio)-available
            bridge.missing_frames+=missing
            np.multiply(self._decay[:missing], self._last, out=audio[available:,0])
            self._last = float(audio[-1,0])
            self._recovering = True
            self._startup_fade=False
            self._fade_position = 0
            if is_meanvc2(bridge.parameters.model):self._primed=False
            return audio
        if self._recovering:
            ramp = self._ramp[:len(audio)]
            np.add(self._fade[:len(audio)], self._fade_position/240, out=ramp)
            np.minimum(ramp, 1, out=ramp)
            np.multiply(audio[:,0], ramp, out=audio[:,0])
            if self._startup_fade:
                mix = self._mix[:len(audio)]
                np.subtract(1, ramp, out=mix)
                np.multiply(mix, self._dry[:len(audio)], out=mix)
                np.add(audio[:,0], mix, out=audio[:,0])
            self._fade_position += len(audio)
            self._recovering = self._fade_position < 240
            if not self._recovering:self._startup_fade=False
        self._last = float(audio[-1,0])
        self._has_played=True
        return audio
