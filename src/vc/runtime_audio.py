"""Runtime audio guards shared by the enhancer and the conversion worker.

Both decisions here trade a little time for throughput, so the measurements they rely
on must be explicit: every guard reports what it did and why, and each one is a no-op
unless the audio clearly qualifies.
"""
import numpy as np

# A frame this far below the clip's own speech level counts as silence.
SILENCE_MARGIN_DB = 40.0
# Below this peak the enhancer's output would be inaudible anyway.
MIN_ENHANCER_PEAK = 1e-4
# An absolute ceiling too, because a clip of pure digital silence has zero dynamic
# range and the percentile test above cannot see a difference inside it.
ABSOLUTE_SILENCE_DB = -90.0
# Silence shorter than this in total is not worth a special path: the crossfade back to
# speech costs more than processing the silence would.
MIN_SILENCE_SECONDS = 0.05
MAX_SILENCE_SECONDS = 10.0


def frame_levels(audio, rate, frame_ms=20.0):
    frame = max(1, int(rate*frame_ms/1000))
    hop = frame
    count = 1 + max(0, (len(audio)-frame))//hop
    if count <= 0 or len(audio) < frame:
        return np.zeros(0), frame, hop
    audio = np.asarray(audio, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(audio**2)))
    starts = np.arange(count)*hop
    rms = np.sqrt(np.maximum((cumulative[starts+frame]-cumulative[starts])/frame, 1e-12))
    return 20.0*np.log10(np.maximum(rms, 1e-9)), frame, hop


def silence_mask(audio, rate, margin_db=SILENCE_MARGIN_DB, frame_ms=20.0):
    """Per-frame silence decision relative to the clip's own loudest frames.

    Absolute thresholds do not survive a quiet recording or a hot one; the peak of the
    same clip is the only reference available without knowing the capture gain.
    """
    levels, frame, hop = frame_levels(audio, rate, frame_ms)
    if not levels.size:
        return np.zeros(0, dtype=bool), frame, hop
    reference = float(np.percentile(levels, 95))
    return levels < reference-margin_db, frame, hop


def silence_bounds(audio, rate, margin_db=SILENCE_MARGIN_DB, frame_ms=20.0):
    """Contiguous (start, end) sample ranges that are silent, at least MIN long."""
    mask, frame, hop = silence_mask(audio, rate, margin_db, frame_ms)
    if not mask.size or not mask.any():
        return []
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)))
    bounds = []
    if mask[0]:
        bounds.append([0, int(edges[0]) if edges.size else len(mask)])
    for index in range(0, len(edges), 2):
        start = int(edges[index])+1
        end = int(edges[index+1])+1 if index+1 < len(edges) else len(mask)
        bounds.append([start, end])
    if mask[-1]:
        if bounds and bounds[-1][1] == len(mask):
            bounds[-1][1] = len(mask)
        else:
            bounds.append([len(mask)-1, len(mask)])
    return [(start*hop, min(end*hop+hop, len(audio)))
            for start, end in bounds
            if (min(end*hop+hop, len(audio))-start*hop)/rate >= MIN_SILENCE_SECONDS]


def total_silence_seconds(audio, rate, margin_db=SILENCE_MARGIN_DB, frame_ms=20.0):
    mask, _, _ = silence_mask(audio, rate, margin_db, frame_ms)
    if not mask.size:
        return 0.0
    return float(np.count_nonzero(mask))*frame_ms/1000.0


def skip_enhancement(audio, rate, margin_db=SILENCE_MARGIN_DB, min_seconds=MIN_SILENCE_SECONDS):
    """Report whether an enhancer can be skipped, and how much silence justified it.

    A clip is only skippable when it is entirely inaudible. A partial skip would need a
    splice back into the processed result, and a seam is worse than the RTF saved.
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1 or not np.isfinite(x).all():
        return {'skip': False, 'reason': 'invalid audio', 'silence_seconds': 0.0}
    peak = float(np.max(np.abs(x), initial=0.0))
    if peak < MIN_ENHANCER_PEAK:
        return {'skip': True, 'reason': 'below audibility', 'silence_seconds': len(x)/rate}
    levels, frame, _ = frame_levels(x, rate)
    if not levels.size:
        return {'skip': False, 'reason': 'too short to analyse', 'silence_seconds': 0.0}
    # Digital silence has no dynamic range at all, so the percentile spread is zero and
    # the "loud content" test cannot see a difference. Compare against an absolute floor
    # as well, otherwise an all-zero clip looks like a flat but valid signal.
    spread = float(np.max(levels))-float(np.percentile(levels, 95))
    absolute_peak_db = 20.0*np.log10(max(peak, 1e-12))
    if spread > 3.0 or absolute_peak_db > ABSOLUTE_SILENCE_DB:
        return {'skip': False, 'reason': 'loud content present',
                'silence_seconds': total_silence_seconds(x, rate, margin_db)}
    silence = total_silence_seconds(x, rate, margin_db)
    if silence >= min_seconds and silence >= 0.9*len(x)/rate:
        return {'skip': True, 'reason': 'entirely silent',
                'silence_seconds': silence}
    return {'skip': False, 'reason': 'speech present', 'silence_seconds': silence}


def speech_segments(audio, rate, margin_db=SILENCE_MARGIN_DB, frame_ms=20.0):
    """Sample ranges worth converting: everything that is not silence.

    The caller must still run the model across the surrounding silence to keep the
    recurrent state and the attention mask continuous; this only reports where the
    audible content sits.
    """
    x = np.asarray(audio)
    bounds = silence_bounds(x, rate, margin_db, frame_ms)
    if not bounds:
        return [(0, len(x))]
    segments = []
    cursor = 0
    for start, end in bounds:
        if start > cursor:
            segments.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < len(x):
        segments.append((cursor, len(x)))
    return segments or [(0, len(x))]