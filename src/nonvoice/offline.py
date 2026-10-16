"""Piece-local gate for environment passthrough on the utterance route.

The streaming route lets the audio callback ask ``NonVoiceBridge.allow``
against rings the analysis thread keeps running. The utterance route has no
continuous stream: the bridge receives complete pieces and the silent gaps
between them only at conversion time, so the same decision table is applied
piece-locally here against real model sessions on the inference thread.

The span decision mirrors ``DecisionStore.allow_span`` exactly — thresholds,
Silero erosion, YAMNet window coverage and the -96 ms/+192 ms speech margin —
and the tests cross-check the two implementations on identical rings. It
fails closed the same way: frames after a buffer count as speech and event
windows that were never computed contribute nothing, while the frames before
a piece are its own recorded preroll, which the endpoint detector proved
subthreshold. A gate whose models are missing or broken routes nothing at
all, so a fault always preserves converted audio and never raw speech.
"""
import logging

import numpy as np

from . import models as model_files
from .bridge import (
    A_THRESHOLD, B_THRESHOLD, DECIM, MARGIN_HI, SILERO_CONTEXT, SILERO_FRAME,
    SPEECH_THRESHOLD, VOCAL_THRESHOLD, YAM_HOP, YAM_WINDOW, Decimator,
)
from src.processors.env_passthrough import TransientDetector

log = logging.getLogger(__name__)

RATE48 = 48000
RATE16 = 16000
MASK_HOP = 512          # 10.7 ms @48 kHz; the transient detector's own hop
GAP_SECONDS = 20.0      # bound on the never-converted gap audio a bridge keeps
SPLICE_FADE = 64        # 1.3 ms taper at every raw-into-converted handover
# Content quieter than this peak cannot reach any event threshold: measured, a
# broadband knock attenuated to this peak scores ~0.01 (A is 0.20) and fires
# no transient hop. Skipping the models there keeps silent gaps and pauses
# cheap, which matters because this gate shares the machine with the converter.
SKIP_PEAK = 0.02


def eroded_flags(probs):
    """``DecisionStore._eroded`` over one frame array under the offline edges.

    A frame stays speech when no available neighbour probability is below the
    speech threshold. The frame before the piece is its recorded preroll
    (speechless by construction); the frame after it is missing data, which
    the span check itself vetoes.
    """
    below = np.asarray(probs, dtype=np.float32) < SPEECH_THRESHOLD
    n = len(below)
    if not n:
        return np.zeros(0, dtype=bool)
    left = np.empty(n, dtype=bool)
    left[0] = False
    left[1:] = below[:-1]
    right = np.empty(n, dtype=bool)
    right[:n - 1] = below[1:]
    right[n - 1] = False
    return ~(left | below | right)


def span_allowed(flags, events, vocals, low, high, transient):
    """Mirror of ``DecisionStore.allow_span`` for 16 kHz content [low, high).

    ``flags`` is one ``eroded`` verdict per Silero frame and ``events`` /
    ``vocals`` one YAMNet score per fully computed window; both may end short
    of the margin the span needs, which counts exactly as the unwritten frames
    and windows of the streaming verdict would. A high vocal score vetoes the
    span the way speech does: raw coughs and sniffles are voice leaking.
    """
    first = max(0, low // SILERO_FRAME - 3)
    last = (high + MARGIN_HI) // SILERO_FRAME + 2
    if last >= len(flags) or flags[first:last + 1].any():
        return False
    window_first = max(0, -(-(low - YAM_WINDOW) // YAM_HOP))
    window_last = high // YAM_HOP
    event = vocal = 0.0
    if window_first <= window_last and window_first < len(events):
        end = min(window_last + 1, len(events))
        covered = events[window_first:end]
        if len(covered):
            event = float(covered.max())
        vocal = float(vocals[window_first:end].max())
    if vocal >= VOCAL_THRESHOLD:
        return False
    if event >= A_THRESHOLD:
        return True
    return bool(transient) and event >= B_THRESHOLD


def extract_spans(audio, mask, fade=SPLICE_FADE):
    """The routed spans of never-converted audio, concatenated.

    Unrouted audio is dropped rather than replaced, so the result is shorter
    than the input the way the utterance route already compresses time. A few
    ms of equal-power ramp keeps each span's ends from clicking.
    """
    picked = np.asarray(mask, dtype=bool)
    edge = np.diff(picked.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(edge == 1)
    ends = np.flatnonzero(edge == -1)
    parts = []
    for a, b in zip(starts, ends):
        part = audio[a:b].astype(np.float32).copy()
        length = min(fade, (b - a) // 2)
        if length:
            taper = np.sin(np.linspace(0, np.pi / 2, length, dtype=np.float32)) ** 2
            part[:length] *= taper
            part[-length:] *= taper[::-1]
        parts.append(part)
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def splice_raw(converted, raw, mask, fade=SPLICE_FADE):
    """Put raw event sound back into the converted piece it was taken from.

    Same length in and out: the converted timeline is preserved everywhere
    except the routed spans, which carry the exact raw samples, with a short
    equal-power taper at each handover so the seam cannot click.
    """
    converted = np.asarray(converted, dtype=np.float32)
    raw = np.asarray(raw, dtype=np.float32)
    picked = np.asarray(mask, dtype=np.float32).reshape(-1) > 0.5
    if converted.shape != raw.shape or picked.shape != converted.shape:
        raise ValueError('Splicing needs three equal-length mono signals')
    out = converted.copy()
    if not picked.any():
        return out
    edge = np.diff(picked.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(edge == 1)
    ends = np.flatnonzero(edge == -1)
    for a, b in zip(starts, ends):
        out[a:b] = raw[a:b]
        length = min(fade, (b - a) // 2)
        if length:
            taper = np.sin(np.linspace(0, np.pi / 2, length, dtype=np.float32)) ** 2
            out[a:a + length] = converted[a:a + length] * (1 - taper) + raw[a:a + length] * taper
            out[b - length:b] = raw[b - length:b] * taper[::-1] + converted[b - length:b] * taper
    return out.astype(np.float32)


class OfflineGate:
    """Real-model span decisions for one piece or gap at conversion time.

    Sessions are held for the bridge's lifetime, not per piece; a piece costs
    one Silero pass, one YAMNet pass per ~0.96 s and one transient pass on the
    inference thread, which is already offline in this route.
    """

    def __init__(self, directory=None):
        self._error = ''
        self._silero = self._yamnet = None
        self._event_ids = self._vocal_ids = None
        self._sr = np.array(RATE16, dtype=np.int64)
        try:
            directory = directory or model_files.models_dir()
            if directory is None:
                self._error = 'モデルが見つかりません'
            else:
                self._silero, self._yamnet, self._event_ids, self._vocal_ids = \
                    model_files.load_sessions(directory)
        except Exception:
            log.exception('environment passthrough offline gate failed to load')
            self._silero = self._yamnet = None
            self._error = 'モデルの読み込みに失敗'

    @property
    def available(self):
        return self._silero is not None and self._yamnet is not None

    @property
    def error(self):
        return self._error

    def route_mask(self, audio):
        """Raw-routing verdict per sample: 1.0 routes, 0.0 keeps converted audio."""
        return self._mask(audio).astype(np.float32)

    def ambient_only(self, audio):
        """Only the routed spans of never-converted gap audio."""
        return extract_spans(np.asarray(audio, dtype=np.float32), self._mask(audio) > 0)

    def _mask(self, audio):
        if not self.available:
            raise RuntimeError(self._error)
        raw = np.asarray(audio, dtype=np.float32).reshape(-1)
        hops = len(raw) // MASK_HOP
        if not hops:
            return np.zeros(len(raw), dtype=np.float32)
        transients = self._transients(raw, hops)
        if not transients.any() and float(np.max(np.abs(raw))) < SKIP_PEAK:
            # Nothing here could pass the A/B evidence gates; do not spend
            # Silero or YAMNet time on it.
            return np.zeros(len(raw), dtype=np.float32)
        reduced = Decimator().process(raw)
        flags = eroded_flags(self._silero_probs(reduced, len(reduced) // SILERO_FRAME))
        # YAMNet windows need ~0.96 s of content; a bare clap piece and a short
        # gap are shorter still. A continuous stream would have surrounded them
        # with silence, so the windows are computed over zero padding on both
        # sides and indexed back into piece coordinates.
        padded = np.concatenate((np.zeros(YAM_WINDOW, dtype=np.float32), reduced,
                                 np.zeros(YAM_WINDOW, dtype=np.float32)))
        events, vocals = self._yamnet_scores(padded)
        events = events[YAM_WINDOW // YAM_HOP:]
        vocals = vocals[YAM_WINDOW // YAM_HOP:]
        mask = np.zeros(len(raw), dtype=np.float32)
        want = np.zeros(hops, dtype=bool)
        for hop in range(hops):
            low = hop * MASK_HOP // DECIM
            high = (hop * MASK_HOP + MASK_HOP + DECIM - 1) // DECIM
            want[hop] = span_allowed(flags, events, vocals, low, high, bool(transients[hop]))
        mask[:hops * MASK_HOP] = np.repeat(want, MASK_HOP)
        return mask

    def _silero_probs(self, reduced, frames):
        """Same streamed state the analysis thread keeps: context plus RNN."""
        probs = np.zeros(frames, dtype=np.float32)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros(SILERO_CONTEXT, dtype=np.float32)
        for frame in range(frames):
            x = np.concatenate((context,
                                reduced[frame * SILERO_FRAME:(frame + 1) * SILERO_FRAME]))[None, :]
            output = self._silero.run(None, {'input': x, 'state': state, 'sr': self._sr})
            probs[frame] = float(np.ravel(output[0])[0])
            state = np.asarray(output[1], dtype=np.float32)
            context = x[0, -SILERO_CONTEXT:].copy()
        return probs

    def _yamnet_scores(self, reduced):
        """Whole-window scores only, as the streaming store publishes them.

        Two maxima per window: environmental events and voice-adjacent sounds,
        the latter vetoing routing. Windows that stay below ``SKIP_PEAK``
        everywhere score zero by measurement (A is 0.20 and such content
        reaches ~0.01), so their model time is not spent: real material is
        mostly silence between events and this is where the gate cost lives.
        """
        windows = max(0, 1 + (len(reduced) - YAM_WINDOW) // YAM_HOP)
        events = np.zeros(windows, dtype=np.float32)
        vocals = np.zeros(windows, dtype=np.float32)
        magnitude = np.abs(reduced)
        for window in range(windows):
            start = window * YAM_HOP
            if float(magnitude[start:start + YAM_WINDOW].max()) < SKIP_PEAK:
                continue
            output = np.asarray(self._yamnet.run(
                None, {'waveform': reduced[start:start + YAM_WINDOW]})[0]).ravel()
            events[window] = float(output[self._event_ids].max())
            if len(self._vocal_ids):
                vocals[window] = float(output[self._vocal_ids].max())
        return events, vocals

    def _transients(self, raw, hops):
        """Fresh detector per buffer: each buffer is its own epoch here."""
        detector = TransientDetector()
        flags = np.zeros(hops, dtype=bool)
        for hop in range(hops):
            flags[hop] = detector.process(raw[hop * MASK_HOP:(hop + 1) * MASK_HOP])
        return flags
