from src.vc.models import VOICE_PROFILES, is_meanvc2
from collections import deque
import logging
import math

import numpy as np
from .base import AudioProcessor
from .env_passthrough import DelayRing, TransientDetector
from src.nonvoice.bridge import NonVoiceBridge
from src.vc.bridge import AIBridge, fallback_is_silent

log = logging.getLogger(__name__)


class AIVoiceProcessor(AudioProcessor):
    """Only non-blocking ring copies and short dropout ramps run on callback."""

    ENV_MAX_DELAY_MS = 2500.0

    def __init__(self, parameters=None, bridge=None, nonvoice=None):
        self.bridge = bridge or AIBridge(parameters)
        self._epoch = -1
        self._primed = False
        self._last = 0.0
        self._recovering = True
        # Environmental passthrough state. The ring stays None on routes whose
        # output cannot be index-aligned with its input (utterance, DSP voices).
        self.env_passthrough = True
        self._env_ring = None
        self._env_detector = TransientDetector()
        self._nonvoice = nonvoice if nonvoice is not None else NonVoiceBridge()
        self._env_mix = 0.0
        self._env_accepted = 0
        self._env_read = 0
        self._env_foreign = 0
        self._env_drops = deque(maxlen=64)
        self._env_own_pending = 0
        self._env_dropped_seen = 0
        self._env_delay = 0
        self._env_delay_resolved = False
        self._env_max_frames = 0
        self._env_base = 0

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
        self._env_max_frames = max_frames
        self._env_index = np.arange(1.0, max_frames + 1, dtype=np.float32)
        self._env_raw = np.empty(max_frames, dtype=np.float32)
        self._env_moves = np.empty(max_frames, dtype=np.float32)
        self._env_gain = np.empty(max_frames, dtype=np.float32)
        self._env_blend = np.empty(max_frames, dtype=np.float32)
        self._env_fade_frames = max(1, round(sample_rate * self.bridge.parameters.crossfade_ms / 1000.0))
        self._env_delay_resolved = False
        if is_meanvc2(self.bridge.parameters.model) and self.bridge.parameters.delivery == 'streaming':
            self._env_delay = round(self._env_alignment_ms() * 48.0)
            self._env_ring = DelayRing(self._env_capacity())
        else:
            self._env_ring = None
        if self._env_ring is not None:
            self._nonvoice.load()
        else:
            self._nonvoice.stop()
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
        self._env_mix = 0.0
        self._env_accepted = 0
        self._env_read = 0
        self._env_foreign = 0
        self._env_drops.clear()
        self._env_own_pending = 0
        self._env_dropped_seen = self.bridge.dropped_frames
        self._env_delay_resolved = False
        self._env_base = self._nonvoice.mark_epoch()
        self._env_detector.reset()
        if self._env_ring is not None:
            self._env_ring.reset()

    def stop(self):
        try:
            self.bridge.stop()
        except Exception:
            # A worker that will not join is already reported by the bridge status,
            # which the AI line shows. Raising here aborted the rest of the teardown.
            log.exception('AI worker stop incomplete')
        try:
            self._nonvoice.stop()
        except Exception:
            log.exception('environment passthrough stop incomplete')

    def process(self, audio, sample_rate):
        bridge = self.bridge
        if not getattr(self, 'supported', False) or bridge.status != 'Ready' or bridge.ack_generation != bridge.generation:
            if getattr(self,'supported',False) and bridge.parameters.mute_during_startup and (bridge.status!='Error' or fallback_is_silent(bridge.parameters)):audio.fill(0)
            # Streaming falls back to the Original on a failure; the sentence route
            # keeps the microphone silent instead, because once a sentence has been
            # engaged its raw voice must never reach the listener.
            return audio
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
            if self._env_ring is not None:
                self._env_own_pending += len(audio)
            bridge.record_incident(2)
        elif self._env_ring is not None:
            self._env_ring.write(audio[:,0])
            self._env_accepted += len(audio)
            self._nonvoice.submit(audio[:,0])
        if self._env_ring is not None:
            self._env_account_drops()
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
                self._env_read += available
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
        self._env_read += len(audio)
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
        if self._env_usable():
            self._env_route(audio, len(audio))
        self._last = float(audio[-1,0])
        self._has_played=True
        return audio

    def set_env_passthrough(self, enabled):
        """Control-side switch; a live route fades between raw and converted.

        The bridge mirrors the setting for the utterance route, which masks
        pieces with the offline gate instead of this callback's ring.
        """
        self.env_passthrough = bool(enabled)
        self.bridge.set_env_passthrough(enabled)

    def _env_alignment_ms(self):
        """How far the output content lags the input content it describes."""
        for value in (self.bridge.model_stats.get('model_alignment_delay_ms'),
                      self.bridge.stats.get('algorithmic_buffer_ms')):
            if type(value) in (int, float) and math.isfinite(value) and 100.0 <= value <= self.ENV_MAX_DELAY_MS:
                return float(value)
        profile = VOICE_PROFILES.get(self.bridge.parameters.model) or {}
        value = profile.get('model_buffer_ms')
        return float(value) if type(value) in (int, float) else 1490.0

    def _env_capacity(self):
        bridge = self.bridge
        # Sized for the largest delay the resolver will ever accept, so resolving
        # the real value from the worker's stats can never outgrow the ring: a
        # mid-stream rebuild would restart its write counter and break the
        # absolute-index alignment. Plus everything the bridge may still hold,
        # and a block of slack on both ends of the read window.
        delay = round(self.ENV_MAX_DELAY_MS * 48.0)
        return delay + bridge.input.capacity + bridge.output.capacity + 4 * self._env_max_frames

    def _env_usable(self):
        """Only the streaming route has a ring; the rest always returns False."""
        if self._env_ring is None:
            return False
        if not (self.env_passthrough or self._env_mix > 0.0):
            return False
        if not self._env_delay_resolved and self.bridge.model_stats:
            self._env_delay = round(self._env_alignment_ms() * 48.0)
            self._env_delay_resolved = True
        return True

    def _env_account_drops(self):
        """Separate the bridge thread's input trims from our own overruns.

        Only frames dropped after they were accepted shift the content the
        model later emits; a callback overrun never entered the accepted
        stream and must not move the read window.
        """
        dropped = self.bridge.dropped_frames
        total = dropped - self._env_dropped_seen
        self._env_dropped_seen = dropped
        own = self._env_own_pending
        self._env_own_pending = 0
        foreign = total - own
        if foreign > 0:
            # The worker trims the pending head, so the discarded frames sit
            # just behind what is still queued. The position is approximate
            # within one queue depth; exact bookkeeping would need worker-side
            # indexes the callback must not wait for.
            position = self._env_accepted - self.bridge.input.available
            self._env_foreign += foreign
            self._env_drops.append((position, foreign))

    def _env_route(self, audio, frames):
        """Replace converted transients with their raw input, with a short fade.

        The converted block describes input content exactly ``_env_delay``
        samples older, plus every foreign drop that reached the model since
        that content was accepted. Drops newer than the content are excluded:
        they move later content, not this block.
        """
        base = (self._env_read - frames) - self._env_delay
        start = base + self._env_foreign
        for _ in range(2):
            shift = self._env_foreign
            for position, size in self._env_drops:
                if position > start:
                    shift -= size
            start = base + shift
        raw = self._env_raw[:frames]
        self._env_ring.read_into(raw, start)
        transient = self._env_detector.process(raw)
        routed = (self.env_passthrough
                  and self._nonvoice.allow(self._env_base, start, frames, transient))
        target = 1.0 if routed else 0.0
        mix = self._env_mix
        if mix == 0.0 and target == 0.0:
            return
        moves = self._env_moves[:frames]
        np.multiply(self._env_index[:frames], 1.0 / self._env_fade_frames, out=moves)
        if target >= mix:
            np.add(moves, mix, out=moves)
            np.minimum(moves, target, out=moves)
        else:
            np.subtract(mix, moves, out=moves)
            np.maximum(moves, target, out=moves)
        self._env_mix = float(moves[-1])
        gain = self._env_gain[:frames]
        blend = self._env_blend[:frames]
        out = audio[:, 0]
        np.subtract(1.0, moves, out=gain)
        np.multiply(out, gain, out=out)
        np.multiply(raw, moves, out=blend)
        np.add(out, blend, out=out)
