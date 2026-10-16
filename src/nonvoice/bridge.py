"""Analysis thread and decision rings for the environment passthrough gate.

The callback only copies input blocks into a ring and asks ``allow`` for a
routing verdict; a separate thread resamples 48 kHz to 16 kHz, runs Silero VAD
per 512-frame and YAMNet per 0.16 s hop, and publishes the results into
preallocated rings addressed by absolute 16 kHz index. Nothing blocking,
allocating per sample or model-related runs on the audio callback.

Decision for a content span (mirrors the offline probe, seed 11):

    route = (max event over covering windows >= A_THRESHOLD
             or (any transient hop in the span and max event >= B_THRESHOLD))
            and no eroded speech frame covering [span - 96 ms, span + 192 ms]

Unwritten speech frames count as speech and unwritten event windows as no
event, so the gate fails closed while it is loading and after a restart.
"""
import logging
import threading

import numpy as np

from src.vc.ring import AudioRing
from src.vc.wakeup import InputWakeup
from . import models as model_files

log = logging.getLogger(__name__)

RATE48 = 48000
RATE16 = 16000
DECIM = RATE48 // RATE16
SILERO_FRAME = 512          # 32 ms @16k
SILERO_CONTEXT = 64
YAM_WINDOW = 15360          # 0.96 s @16k
YAM_HOP = 2560              # 0.16 s @16k

A_THRESHOLD = 0.20
B_THRESHOLD = 0.05
SPEECH_THRESHOLD = 0.5
# A window that strongly looks like the speaker's own breath or throat sound
# vetoes routing: raw coughs and sniffles are voice leaking unchanged. Below
# this the sound may still route as an event (a distant TV cough sits lower
# than a close-mic one, and rejecting every quiet cough would silence half a
# room); the event list no longer contains these classes either, so the two
# gates must both fail.
VOCAL_THRESHOLD = 0.30
MARGIN_LO = 1536            # 96 ms before the span
MARGIN_HI = 3072            # 192 ms after the span
SPEECH_CAPACITY = 1024      # 32 s of speech frames
EVENT_CAPACITY = 512        # 82 s of event windows

CHUNK48 = 4800              # consumer read unit, 100 ms
INPUT_SECONDS = 2.0

EVENT_SLOT_EMPTY = np.float32(-1.0)


class Decimator:
    """48 kHz -> 16 kHz, 49-tap Blackman lowpass, output m holds input 3m."""

    TAPS = 49

    def __init__(self):
        half = (self.TAPS - 1) // 2
        index = np.arange(self.TAPS)
        cutoff = 7200.0 / RATE48
        kernel = np.sinc(cutoff * (index - half)) * np.blackman(self.TAPS)
        kernel = (kernel / kernel.sum()).astype(np.float32)
        self._reversed = kernel[::-1].copy()
        self._carry = np.zeros(self.TAPS - 1, dtype=np.float32)

    def reset(self):
        self._carry.fill(0)

    def process(self, chunk):
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if len(chunk) == 0:
            return chunk
        window = np.concatenate((self._carry, chunk))
        count = (len(window) - self.TAPS) // DECIM + 1
        if count <= 0:
            self._carry = window[-(self.TAPS - 1):].copy()
            return window[:0]
        frames = np.lib.stride_tricks.sliding_window_view(window, self.TAPS)[::DECIM]
        output = (frames[:count] @ self._reversed).astype(np.float32)
        self._carry = window[-(self.TAPS - 1):].copy()
        return output


class DecisionStore:
    """Lock-free single-writer rings of speech probabilities and event scores."""

    def __init__(self):
        self._speech = np.zeros(SPEECH_CAPACITY, dtype=np.float32)
        self._speech_written = 0
        self._events = np.full(EVENT_CAPACITY, EVENT_SLOT_EMPTY, dtype=np.float32)
        self._events_written = 0
        self._vocal = np.full(EVENT_CAPACITY, EVENT_SLOT_EMPTY, dtype=np.float32)
        self._vocal_written = 0

    @property
    def speech_frames(self):
        return self._speech_written

    @property
    def event_windows(self):
        return self._events_written

    @property
    def vocal_windows(self):
        return self._vocal_written

    def publish_speech(self, frame, probability):
        self._speech[frame % SPEECH_CAPACITY] = probability
        self._speech_written = frame + 1

    def publish_event(self, window, score):
        self._events[window % EVENT_CAPACITY] = score
        self._events_written = window + 1

    def publish_vocal(self, window, score):
        self._vocal[window % EVENT_CAPACITY] = score
        self._vocal_written = window + 1

    def _speech_at(self, frame):
        """Probability at an absolute frame; None when not available."""
        if frame < 0 or frame >= self._speech_written:
            return None
        if frame < self._speech_written - SPEECH_CAPACITY:
            return None
        return float(self._speech[frame % SPEECH_CAPACITY])

    def _event_at(self, window):
        """Score at an absolute window; None when not available."""
        if window < 0 or window >= self._events_written:
            return None
        if window < self._events_written - EVENT_CAPACITY:
            return None
        return float(self._events[window % EVENT_CAPACITY])

    def _vocal_at(self, window):
        """Voice-adjacent score at an absolute window; None when not available."""
        if window < 0 or window >= self._vocal_written:
            return None
        if window < self._vocal_written - EVENT_CAPACITY:
            return None
        return float(self._vocal[window % EVENT_CAPACITY])

    def _eroded(self, frame):
        """Three consecutive speech frames around ``frame``; missing counts as speech."""
        for offset in (-1, 0, 1):
            probability = self._speech_at(frame + offset)
            if probability is not None and probability < SPEECH_THRESHOLD:
                return False
        return True

    def allow_span(self, low, high, transient):
        """Routing verdict for the 16 kHz content span [low, high)."""
        first = max(0, low // SILERO_FRAME - 3)
        last = (high + MARGIN_HI) // SILERO_FRAME + 2
        for frame in range(first, last + 1):
            if self._eroded(frame):
                return False
        event = vocal = 0.0
        window_first = max(0, -(-(low - YAM_WINDOW) // YAM_HOP))
        window_last = high // YAM_HOP
        for window in range(window_first, window_last + 1):
            score = self._event_at(window)
            if score is not None and score > event:
                event = score
            score = self._vocal_at(window)
            if score is not None and score > vocal:
                vocal = score
        if vocal >= VOCAL_THRESHOLD:
            return False
        if event >= A_THRESHOLD:
            return True
        return transient and event >= B_THRESHOLD


class NonVoiceBridge:
    """Callback-safe facade over the decision store and the analysis thread."""

    def __init__(self):
        self._input = AudioRing(int(INPUT_SECONDS * RATE48))
        self._wakeup = InputWakeup()
        self._stop = threading.Event()
        self._thread = None
        self._store = DecisionStore()
        self._decimator = Decimator()
        self._written_total = 0
        self._ready = False
        self._failed = False
        self._unavailable = False
        self._error = None
        # Thread-owned analysis state.
        self._silero = None
        self._yamnet = None
        self._event_ids = None
        self._vocal_ids = None
        self._silero_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._silero_context = np.zeros(SILERO_CONTEXT, dtype=np.float32)
        self._frame = np.zeros(SILERO_FRAME, dtype=np.float32)
        self._frame_fill = 0
        self._frames_written = 0
        self._samples_written = 0
        self._yam_ring = np.zeros(YAM_WINDOW + YAM_HOP, dtype=np.float32)
        self._yam_next = 0
        self._yam_window = np.zeros(YAM_WINDOW, dtype=np.float32)
        self._block = np.empty(CHUNK48, dtype=np.float32)
        self._sr = np.array(RATE16, dtype=np.int64)

    @property
    def status(self):
        if self._failed:
            return 'error'
        if self._unavailable:
            return 'unavailable'
        if self._ready:
            return 'ready'
        return 'loading' if self._thread is not None else 'off'

    @property
    def error(self):
        return self._error

    @property
    def speech_frames(self):
        return self._store.speech_frames

    @property
    def event_windows(self):
        return self._store.event_windows

    @property
    def pending(self):
        """Samples submitted but not consumed by the analysis thread yet."""
        return self._input.available

    def load(self, directory=None):
        """Start the analysis thread; safe to call again while it runs."""
        if self._thread is not None:
            return self._ready or self._thread.is_alive()
        if self._failed:
            return False
        directory = directory or model_files.models_dir()
        if directory is None:
            self._unavailable = True
            self._error = 'モデルが見つかりません'
            log.info('environment passthrough disabled: nonvoice models are not installed')
            return False
        self._unavailable = False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(directory,),
                                        name='koeiro-nonvoice', daemon=True)
        self._thread.start()
        return True

    def stop(self):
        thread, self._thread = self._thread, None
        self._ready = False
        if thread is None:
            return
        self._stop.set()
        self._wakeup.signal()
        thread.join(5.0)

    def mark_epoch(self):
        """Absolute position of the next submitted sample; call before restarting."""
        return self._written_total

    def submit(self, audio):
        """Callback side: copy one accepted input block for analysis."""
        if self._failed or self._thread is None:
            return
        if not self._input.write(audio):
            self._fail('input ring overflow')
            return
        self._written_total += len(audio)
        self._wakeup.signal()

    def allow(self, base, start, frames, transient):
        """Routing verdict for content [start, start + frames) of the epoch at ``base``."""
        if not self._ready or self._failed:
            return False
        low = (base + start) // DECIM
        high = (base + start + frames + DECIM - 1) // DECIM
        return self._store.allow_span(low, high, bool(transient))

    def _fail(self, message):
        if not self._failed:
            self._failed = True
            self._error = message
            log.warning('environment passthrough disabled: %s', message)

    def _run(self, directory):
        try:
            self._silero, self._yamnet, self._event_ids, self._vocal_ids = \
                model_files.load_sessions(directory)
        except Exception:
            log.exception('nonvoice models failed to load')
            self._fail('モデルの読み込みに失敗')
            return
        self._wakeup.open()
        self._ready = True
        log.info('environment passthrough gate ready (%s)', directory)
        try:
            while not self._stop.is_set():
                self._drain()
                if self._failed:
                    break
                self._wakeup.wait(self._stop)
        except Exception:
            log.exception('nonvoice analysis thread failed')
            self._fail('解析スレッドが異常終了')
        finally:
            self._ready = False
            self._wakeup.close()

    def _drain(self):
        for _ in range(32):
            available = self._input.available
            take = min(available, CHUNK48)
            take -= take % DECIM
            if take < DECIM:
                break
            if not self._input.read_into(self._block[:take]):
                break
            output = self._decimator.process(self._block[:take])
            if len(output):
                self._consume(output)
        if self._input.available >= DECIM:
            self._wakeup.signal()

    def _consume(self, samples):
        offset = 0
        while offset < len(samples):
            take = min(SILERO_FRAME - self._frame_fill, len(samples) - offset)
            self._frame[self._frame_fill:self._frame_fill + take] = samples[offset:offset + take]
            self._frame_fill += take
            offset += take
            if self._frame_fill == SILERO_FRAME:
                self._run_silero(self._frame)
                self._frame_fill = 0
        capacity = len(self._yam_ring)
        position = self._samples_written % capacity
        first = min(len(samples), capacity - position)
        self._yam_ring[position:position + first] = samples[:first]
        if len(samples) > first:
            self._yam_ring[:len(samples) - first] = samples[first:]
        self._samples_written += len(samples)
        while self._yam_next * YAM_HOP + YAM_WINDOW <= self._samples_written:
            start = self._yam_next * YAM_HOP
            if start < self._samples_written - capacity:
                self._yam_next += 1  # overwritten while the thread was behind
                continue
            self._read_window(start, self._yam_window)
            scores = np.asarray(self._yamnet.run(
                None, {'waveform': self._yam_window})[0]).ravel()
            self._store.publish_event(self._yam_next, float(scores[self._event_ids].max()))
            if len(self._vocal_ids):
                self._store.publish_vocal(self._yam_next, float(scores[self._vocal_ids].max()))
            else:
                self._store.publish_vocal(self._yam_next, 0.0)
            self._yam_next += 1

    def _run_silero(self, frame512):
        x = np.concatenate((self._silero_context, frame512))[None, :]
        output = self._silero.run(None, {'input': x, 'state': self._silero_state, 'sr': self._sr})
        self._silero_state = np.asarray(output[1], dtype=np.float32)
        self._silero_context = x[0, -SILERO_CONTEXT:].copy()
        self._store.publish_speech(self._frames_written, float(np.ravel(output[0])[0]))
        self._frames_written += 1

    def _read_window(self, start, out):
        capacity = len(self._yam_ring)
        position = start % capacity
        first = min(len(out), capacity - position)
        out[:first] = self._yam_ring[position:position + first]
        if len(out) > first:
            out[first:] = self._yam_ring[:len(out) - first]
