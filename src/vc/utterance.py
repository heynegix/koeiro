"""Bounded endpoint detection on the bridge thread, never on the audio callback."""
from collections import deque
import numpy as np

MAX_SECONDS = 30
QUALITY_MAX_SECONDS = 60  # MossFormer quality mode only; other modes stay at 30 s.

# Enhancers measured cheap and memory-bounded enough to accept the 60 s
# utterance: MossFormer2_SR and VoiceFixer2 on RTF/RAM grounds, and LavaSR at
# RTF 0.043-0.064 with 1.15 GiB peak on this CPU. FlashSR is deliberately not
# here: at RTF ~2.4-3.2 and 3.6-3.9 GiB peak it cannot serve a 60 s utterance.
LONG_FORM_ENHANCERS = ('mossformer', 'voicefixer', 'lavasr')

# Pause length that justifies a neural-state refresh in the experimental refresh
# route. Shorter pauses keep the running state; only a real break resynchronises.
REFRESH_PAUSE_SECONDS = 1.2
REFRESH_OVERLAP_SAMPLES = int(0.16*48000)
# Refresh only pays off on long utterances: each extra pass adds flush padding,
# so short utterances below this length take the single pass instead.
REFRESH_MIN_SECONDS = 10
# Input levelling target for the experimental levelled route. LavaSR normalises
# the output anyway, so this only steadies what the converter itself receives.
LEVEL_TARGET_DB = -20.0
LEVEL_MAX_GAIN_DB = 18.0
# Crossover for the experimental high-frequency blend: the converter's own band
# below, LavaSR's restoration above, with a raised-cosine handover between.
BLEND_CUTOFF_HZ = 8000.0
BLEND_WIDTH_HZ = 2000.0


def _framed_matrix(audio, frame, hop):
    """Contiguous (count, frame) window matrix over `audio`.

    Rows match the historical per-frame loop exactly (same samples in the
    same order); only the Python loop is gone. The caller keeps every
    downstream formula untouched, so outputs are unchanged while the
    per-frame FFTs run as one batched call.
    """
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    count = 1 + (len(audio) - frame) // hop
    shape = (count, frame)
    strides = (audio.strides[0] * hop, audio.strides[0])
    return np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)


def _framed_matrix(audio, frame, hop):
    """Contiguous (count, frame) window matrix over `audio`.

    Rows match the historical per-frame loop exactly (same samples in the
    same order); only the Python loop is gone. The caller keeps every
    downstream formula untouched, so outputs are unchanged while the
    per-frame FFTs run as one batched call.
    """
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    count = 1 + (len(audio) - frame) // hop
    shape = (count, frame)
    strides = (audio.strides[0] * hop, audio.strides[0])
    return np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)


def _fade_pair(length):
    """Equal-power out/in ramps over `length` samples."""
    angle = np.linspace(0, np.pi/2, length)
    return np.cos(angle)**2, np.sin(angle)**2


def crossfade_blend(previous_tail, current, overlap):
    """Blend the tail of the previous result into the head of the current one.

    Returns the full current utterance with its first `overlap` samples mixed.
    Length is unchanged, so the bridge's exact-length check still holds.
    """
    previous_tail = np.asarray(previous_tail, dtype=np.float64)
    current = np.asarray(current, dtype=np.float32)
    if overlap <= 0 or len(previous_tail) < overlap or len(current) < overlap:
        raise ValueError('Overlap needs at least that many samples on both sides')
    fade_out, fade_in = _fade_pair(overlap)
    head = previous_tail[-overlap:]*fade_out + current[:overlap].astype(np.float64)*fade_in
    return np.concatenate((head, current[overlap:])).astype(np.float32)


def crossfade_join(parts, overlap):
    """Join converted segments with an equal-power crossfade at each boundary."""
    chunks = [np.asarray(part, dtype=np.float32) for part in parts]
    if not chunks:
        raise ValueError('Nothing to join')
    if len(chunks) == 1:
        return chunks[0].copy()
    if overlap <= 0 or any(len(chunk) < overlap for chunk in chunks):
        raise ValueError('Overlap needs at least that many samples in every part')
    fade_out, fade_in = _fade_pair(overlap)
    joined = chunks[0][:-overlap].copy()
    for index in range(1, len(chunks)):
        previous, current = chunks[index-1], chunks[index]
        mixed = previous[-overlap:].astype(np.float64)*fade_out + current[:overlap].astype(np.float64)*fade_in
        # Middle parts give up both sides; only the last part keeps its tail.
        tail = current[overlap:] if index == len(chunks)-1 else current[overlap:-overlap]
        joined = np.concatenate((joined, mixed, tail))
    return joined.astype(np.float32)


def split_at_pauses(audio, rate=48000, min_pause_s=REFRESH_PAUSE_SECONDS):
    """Split points at long silences; returns [(start, end)] covering all audio."""
    from .runtime_audio import silence_bounds
    audio = np.asarray(audio, dtype=np.float32)
    bounds = silence_bounds(audio, rate)
    cuts = [0]
    for start, end in bounds:
        if (end-start)/rate >= min_pause_s:
            cuts.append((start+end)//2)
    cuts.append(len(audio))
    cuts = sorted(set(cuts))
    return [(cuts[index], cuts[index+1]) for index in range(len(cuts)-1)]


def level_utterance(audio, target_db=LEVEL_TARGET_DB, max_gain_db=LEVEL_MAX_GAIN_DB):
    """Scale one utterance to a fixed RMS so the converter sees a steady level.

    Quieter than digital silence stays untouched; the gain cap keeps a whisper
    from becoming a noise blast. Length and finite-ness are unchanged.
    """
    audio = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(audio.astype(np.float64)**2)))
    if rms < 1e-9:
        return audio.copy()
    gain = 10**(target_db/20)/rms
    gain = min(gain, 10**(max_gain_db/20))
    return (audio*gain).astype(np.float32)


def tame_sibilance(audio, rate=48000, max_cut_db=3.0, ratio=0.6):
    """Gentle pre-conversion taming of harsh sibilant frames.

    Uses the same 6-12 kHz energy-ratio detector as the high-frequency blend
    guard: frames dominated by sibilance are pulled down by at most
    `max_cut_db` with 50 ms ramps, everything else is untouched. Applied to
    the converter input so the voice model never sees a harsh burst it would
    amplify; same length in, same length out. No training, no fixed threshold
    on absolute level.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all() or not len(audio):
        raise ValueError('Sibilance taming needs finite mono audio')
    if not 0 <= max_cut_db <= 12:
        raise ValueError('Sibilance cut must be 0..12 dB')
    if not 0 < ratio < 1:
        raise ValueError('Sibilance ratio must be 0..1')
    if max_cut_db == 0:
        return audio.copy()
    frame, hop = int(rate*0.02), int(rate*0.01)
    if len(audio) < frame:
        return audio.copy()
    count = 1+(len(audio)-frame)//hop
    windows = _framed_matrix(audio, frame, hop).astype(np.float64)*np.hanning(frame)
    spectrum = np.abs(np.fft.rfft(windows, axis=1))**2+1e-12
    bins = np.fft.rfftfreq(frame, 1.0/rate)
    high = spectrum[:, (bins >= 6000)&(bins <= 12000)].sum(axis=1)
    voice = spectrum[:, (bins >= 300)&(bins <= 12000)].sum(axis=1)
    share = np.where(voice > 0, high/np.maximum(voice, 1e-12), 0.0)
    depth = np.clip((share-ratio)/0.2, 0.0, 1.0)*max_cut_db
    # Smooth over ~50 ms so the taming never chatters at boundaries.
    kernel = np.ones(5)/5
    smooth = np.convolve(np.pad(depth, 2, mode='edge'), kernel, mode='valid')
    centres = (np.arange(count)*hop+frame//2).astype(np.float64)
    envelope = np.interp(np.arange(len(audio)), centres, smooth, left=0.0, right=0.0)
    out = (audio.astype(np.float64)*10**(-envelope/20)).astype(np.float32)
    if len(out) != len(audio) or not np.isfinite(out).all():
        raise RuntimeError('Invalid sibilance taming output')
    return out


def lift_consonants(audio, rate=48000, max_lift_db=3.0, band=(2000.0, 6000.0)):
    """Gentle pre-conversion lift of consonant frames in a fixed band.

    Frames that look like consonants (spectral flux plus mid-high band share,
    above the utterance's own floor) get up to `max_lift_db` in `band`, so a
    fast consonant reaches the converter before it is smeared. Steady vowels,
    silence and breaths stay untouched. The split is a tapered whole-utterance
    FFT filter (zero-phase, no delay); only the time-varying mix can move.
    Same length in, same length out. No training.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all() or not len(audio):
        raise ValueError('Consonant lift needs finite mono audio')
    if not 0 <= max_lift_db <= 12:
        raise ValueError('Consonant lift must be 0..12 dB')
    if max_lift_db == 0:
        return audio.copy()
    lo, hi = band
    if not 300 <= lo < hi <= 12000:
        raise ValueError('Consonant band must sit inside 300..12000 Hz')
    if len(audio) < int(rate*0.02):
        return audio.copy()
    # Tapered band split: raised-cosine skirts keep ringing negligible.
    spectrum = np.fft.rfft(audio.astype(np.float64))
    freqs = np.fft.rfftfreq(len(audio), 1.0/rate)
    skirt = 1000.0
    up = np.clip((freqs-(lo-skirt))/max(skirt, 1e-9), 0.0, 1.0)
    down = np.clip(((hi+skirt)-freqs)/max(skirt, 1e-9), 0.0, 1.0)
    weight = (0.5*(1-np.cos(np.pi*np.clip(up, 0, 1))))*(0.5*(1-np.cos(np.pi*np.clip(down, 0, 1))))
    band_part = np.fft.irfft(spectrum*weight, n=len(audio)).astype(np.float32)
    # Detector on 20 ms frames / 10 ms hop.
    frame, hop = int(rate*0.02), int(rate*0.01)
    count = 1+(len(audio)-frame)//hop
    windows = _framed_matrix(audio, frame, hop).astype(np.float64)*np.hanning(frame)
    mag = np.abs(np.fft.rfft(windows, axis=1))+1e-12
    bins = np.fft.rfftfreq(frame, 1.0/rate)
    band_total = mag[:, (bins >= 300)&(bins <= 12000)].sum(axis=1)
    share = mag[:, (bins >= lo)&(bins <= hi)].sum(axis=1)/np.maximum(band_total, 1e-12)
    level = 20*np.log10(np.maximum(np.sqrt((windows**2).mean(axis=1)), 1e-9))
    logm = np.log(mag)
    flux = np.empty(count)
    flux[0] = 0.0
    flux[1:] = np.maximum(logm[1:]-logm[:-1], 0.0).mean(axis=1)
    floor = float(np.percentile(level, 5))
    flux_n = np.clip((flux-float(np.median(flux)))/max(float(np.percentile(flux, 90))
                      - float(np.median(flux)), 1e-9), 0.0, 1.0)
    score = flux_n*np.clip((share-0.25)/0.25, 0.0, 1.0)
    score[level < floor+6.0] = 0.0
    lift = score*max_lift_db
    # Smooth over ~50 ms so the lift never chatters at boundaries.
    kernel = np.ones(5)/5
    smooth = np.convolve(np.pad(lift, 2, mode='edge'), kernel, mode='valid')
    centres = (np.arange(count)*hop+frame//2).astype(np.float64)
    envelope = np.interp(np.arange(len(audio)), centres, smooth, left=0.0, right=0.0)
    out = (audio.astype(np.float64)
           + band_part.astype(np.float64)*(10**(envelope/20)-1.0)).astype(np.float32)
    if len(out) != len(audio) or not np.isfinite(out).all():
        raise RuntimeError('Invalid consonant lift output')
    return out


def clean_input(audio, rate=48000, cutoff_hz=80.0):
    """Pre-conversion input cleanup: tapered high-pass plus DC removal.

    Removes mic rumble, breath thumps and DC that otherwise reach the voice
    model as low-frequency mud and turn into warbly artefacts. Whole-utterance
    zero-phase FFT filter (no delay, no state); speech above the cutoff is
    untouched. Same length in, same length out. No training.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all() or not len(audio):
        raise ValueError('Input cleanup needs finite mono audio')
    if not 20 <= cutoff_hz <= 300:
        raise ValueError('Cleanup cutoff must be 20..300 Hz')
    spectrum = np.fft.rfft(audio.astype(np.float64) - float(np.mean(audio.astype(np.float64))))
    freqs = np.fft.rfftfreq(len(audio), 1.0/rate)
    skirt = cutoff_hz/2
    ramp = np.clip((freqs-(cutoff_hz-skirt))/max(skirt, 1e-9), 0.0, 1.0)
    weight = 0.5*(1-np.cos(np.pi*ramp))
    out = np.fft.irfft(spectrum*weight, n=len(audio)).astype(np.float32)
    if len(out) != len(audio) or not np.isfinite(out).all():
        raise RuntimeError('Invalid input cleanup output')
    return out


def tame_plosives(audio, rate=48000, max_cut_db=6.0):
    """Gentle pre-conversion taming of plosive thumps and mic bursts.

    Frames with a sudden low-band (100-500 Hz) surge get pulled down by at
    most `max_cut_db` with fast ramps; steady speech, vowels and sibilance
    are untouched. Keeps pops from hitting the converter as broadband mud.
    Same length in, same length out. No training.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all() or not len(audio):
        raise ValueError('Plosive taming needs finite mono audio')
    if not 0 <= max_cut_db <= 12:
        raise ValueError('Plosive cut must be 0..12 dB')
    if max_cut_db == 0:
        return audio.copy()
    frame, hop = int(rate*0.02), int(rate*0.01)
    if len(audio) < frame:
        return audio.copy()
    count = 1+(len(audio)-frame)//hop
    windows = _framed_matrix(audio, frame, hop).astype(np.float64)*np.hanning(frame)
    mag = np.abs(np.fft.rfft(windows, axis=1))+1e-12
    bins = np.fft.rfftfreq(frame, 1.0/rate)
    low = mag[:, (bins >= 100)&(bins <= 500)].sum(axis=1)
    total = mag[:, (bins >= 100)&(bins <= 12000)].sum(axis=1)
    level = 10*np.log10(low+1e-12)
    rise = np.empty(count)
    rise[0] = 0.0
    rise[1:] = level[1:]-level[:-1]
    surge = np.maximum(rise, 0.0)*(low/np.maximum(total, 1e-12))
    # Real close-mic pops surge by 10 dB or more; the threshold stays low so
    # milder thumps still catch a proportional cut, while steady vowels (no
    # rise) and flat noise (no share) score near zero.
    depth = np.clip((surge-1.0)/8.0, 0.0, 1.0)*max_cut_db
    # Fast ~20 ms smoothing: pops are brief, smearing them helps nothing.
    kernel = np.ones(3)/3
    smooth = np.convolve(np.pad(depth, 1, mode='edge'), kernel, mode='valid')
    centres = (np.arange(count)*hop+frame//2).astype(np.float64)
    envelope = np.interp(np.arange(len(audio)), centres, smooth, left=0.0, right=0.0)
    out = (audio.astype(np.float64)*10**(-envelope/20)).astype(np.float32)
    if len(out) != len(audio) or not np.isfinite(out).all():
        raise RuntimeError('Invalid plosive taming output')
    return out


def fry_fraction(audio, rate=48000, frame_s=0.04, hop_s=0.02):
    """Roughness proxy of the input itself: creak-like frame share.

    A frame counts as creaky when its best lag sits in the fry band
    (40-130 Hz) with moderate periodicity, or when periodicity is weak
    while pitched low. Returns a 0..1 fraction plus the median periodicity.
    Measurement only; never changes audio. Deterministic.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all() or not len(audio):
        raise ValueError('Roughness measurement needs finite mono audio')
    frame, hop = int(rate*frame_s), int(rate*hop_s)
    if len(audio) < frame:
        return dict(fry=0.0, periodicity=0.0, frames=0)
    count = 1+(len(audio)-frame)//hop
    windows = _framed_matrix(audio, frame, hop).astype(np.float64)
    windows -= windows.mean(axis=1, keepdims=True)
    energy = np.einsum('ij,ij->i', windows, windows)
    # Batched linear autocorrelation through the power spectrum: same raw lag
    # sums as the historical per-frame correlate, without the Python loop.
    padded = np.zeros((count, 2*frame), dtype=np.float64)
    padded[:, :frame] = windows
    full = np.fft.irfft(np.abs(np.fft.rfft(padded, axis=1))**2, axis=1)[:, :frame]
    full /= np.maximum(full[:, :1], 1e-12)
    lo, hi = int(rate/600), int(rate/40)
    band = full[:, lo:hi]
    left = band[:, :-2]
    middle = band[:, 1:-1]
    right = band[:, 2:]
    is_peak = (middle >= left) & (middle > right)
    masked = np.where(is_peak, middle, -np.inf)
    has_peak = is_peak.any(axis=1)
    at = np.argmax(masked, axis=1)+1
    strength = np.where(has_peak, band[np.arange(count), at], 0.0)
    f0 = np.where(has_peak, rate/(lo+at), 0.0)
    strengths = np.where(energy <= 1e-12, 0.0, strength)
    creak = int((((40 <= f0) & (f0 <= 130) & (strengths > 0.5)) |
                 ((strengths >= 0.35) & (strengths < 0.55) & (f0 < 200))).sum())
    return dict(fry=round(creak/max(count, 1), 4),
                periodicity=round(float(np.median(strengths)), 3),
                frames=count)


def suppress_floor(audio, rate=48000, margin_db=3.0, max_cut_db=6.0, protect=None):
    """Gentle downward expansion of the utterance's own noise floor.

    The reference is the quietest 5% of this utterance, so no fixed threshold
    can misfire on a hot or quiet take. Frames below floor+margin are pulled
    down by at most `max_cut_db` with 50 ms ramps; speech frames are untouched.
    `protect` is an optional same-length 0..1 mask (e.g. breath frames) that
    proportionally exempts frames from suppression. Same length in, same
    length out.
    """
    audio = np.asarray(audio, dtype=np.float32)
    frame, hop = int(rate*0.02), int(rate*0.01)
    if len(audio) < frame or not np.isfinite(audio).all():
        raise ValueError('Floor suppression needs finite mono audio of 20 ms or more')
    if protect is not None:
        protect = np.asarray(protect, dtype=np.float64)
        if protect.shape != audio.shape or not np.isfinite(protect).all() \
                or float(protect.min()) < 0.0 or float(protect.max()) > 1.0:
            raise ValueError('Floor protection must be a 0..1 mask of equal length')
    count = 1+(len(audio)-frame)//hop
    windows = _framed_matrix(audio, frame, hop).astype(np.float64)
    energy = 20*np.log10(np.maximum(np.sqrt((windows**2).mean(axis=1)), 1e-9))
    floor = float(np.percentile(energy, 5))
    depth = np.clip((floor+margin_db-energy)/margin_db, 0.0, 1.0)*max_cut_db
    if protect is not None:
        centres = (np.arange(count)*hop+frame//2).astype(np.float64)
        shield = np.interp(centres, np.arange(len(audio)), protect, left=0.0, right=0.0)
        depth = depth*(1.0-np.clip(shield, 0.0, 1.0))
    # Smooth over ~50 ms so the expansion never chatters at boundaries.
    kernel = np.ones(5)/5
    smooth = np.convolve(np.pad(depth, 2, mode='edge'), kernel, mode='valid')
    centres = (np.arange(count)*hop+frame//2).astype(np.float64)
    envelope = np.interp(np.arange(len(audio)), centres, smooth, left=0.0, right=0.0)
    out = (audio.astype(np.float64)*10**(-envelope/20)).astype(np.float32)
    if len(out) != len(audio) or not np.isfinite(out).all():
        raise RuntimeError('Invalid floor suppression output')
    return out


def fade_edges(audio, rate=48000, ms=10.0):
    """Short equal-power fades at both ends of one utterance.

    Removes hard boundary steps left by delay trimming or block joins without
    changing length. Same length in, same length out.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all() or not len(audio):
        raise ValueError('Edge fade needs finite mono audio')
    if not 0 < ms <= 50:
        raise ValueError('Edge fade must be 0..50 ms')
    length = min(int(rate*ms/1000), len(audio)//2)
    if length <= 0:
        return audio.copy()
    angle = np.linspace(0, np.pi/2, length)
    ramp = np.sin(angle)**2
    out = audio.copy()
    out[:length] = (out[:length].astype(np.float64)*ramp).astype(np.float32)
    out[-length:] = (out[-length:].astype(np.float64)*ramp[::-1]).astype(np.float32)
    if len(out) != len(audio) or not np.isfinite(out).all():
        raise RuntimeError('Invalid edge fade output')
    return out


def detect_clicks(audio, rate=48000, jump=0.25, quiet_rms=0.1):
    """Count sample discontinuities typical of splice clicks.

    A click is a jump larger than `jump` between adjacent samples while the
    surrounding 5 ms stays quiet; loud transients (plosives, onsets) are not
    clicks. Returns the count; the worker reports it, listening judges it.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all() or len(audio) < 240:
        raise ValueError('Click detection needs finite mono audio of 5 ms or more')
    steps = np.abs(np.diff(audio.astype(np.float64)))
    candidates = np.flatnonzero(steps > jump)
    clicks = 0
    last = -10**9
    for position in candidates:
        if position-last < 32:
            continue  # one impulse makes two adjacent steps; count it once
        lo, hi = max(0, position-120), min(len(audio), position+121)
        neighbourhood = audio[lo:position].tolist()+audio[position+1:hi].tolist()
        rms = float(np.sqrt(np.mean(np.square(neighbourhood)))) if neighbourhood else 0.0
        if rms < quiet_rms:
            clicks += 1
            last = position
    return clicks

def blend_highs(base, enhanced, rate=48000, cutoff=BLEND_CUTOFF_HZ, width=BLEND_WIDTH_HZ,
                sibilance_guard=True, mid_cutoff=4000.0, mid_weight=0.5, mid_width=2000.0,
                sib_ratio=0.6, sib_mix=0.5, guard_mode='base', excess_db=9.0):
    """Keep the converter's own band, take only the highs from the restoration.

    A two-stage handover: up to `mid_weight` across [mid_cutoff±mid_width/2],
    then up to full restoration across [cutoff±width/2]. The sibilance guard
    then pulls harsh frames partway back toward `base`. In `base` mode, frames
    whose 6-12 kHz share in `base` exceeds `sib_ratio` are mixed back to
    `sib_mix` restoration. In `excess` mode, frames where the restoration
    itself added more than `excess_db` of 6-12 kHz energy over `base` while
    looking sibilant in `enhanced` are mixed back instead: the converter
    output is dull up there by construction, so a base-side detector can
    barely fire where it matters most. Same length in, same length out.
    """
    if not 0 < sib_ratio < 1:
        raise ValueError('Sibilance ratio must be 0..1')
    if not 0 <= sib_mix <= 1:
        raise ValueError('Sibilance mix must be 0..1')
    if guard_mode not in ('base', 'excess'):
        raise ValueError('Guard mode must be base or excess')
    if not 0 < excess_db <= 24:
        raise ValueError('Excess margin must be 0..24 dB')
    base = np.asarray(base, dtype=np.float32)
    enhanced = np.asarray(enhanced, dtype=np.float32)
    if base.shape != enhanced.shape or base.ndim != 1 or not len(base):
        raise ValueError('Blend needs two finite mono signals of equal length')
    if not np.isfinite(base).all() or not np.isfinite(enhanced).all():
        raise ValueError('Blend needs two finite mono signals of equal length')
    frequencies = np.fft.rfftfreq(len(base), 1.0/rate)

    def ramp(low, high, level):
        position = np.clip((frequencies-low)/max(high-low, 1e-9), 0.0, 1.0)
        return (0.5*(1-np.cos(np.pi*position)))*level

    weight = np.minimum(ramp(mid_cutoff-mid_width/2, mid_cutoff+mid_width/2, mid_weight)
                        + ramp(cutoff-width/2, cutoff+width/2, 1.0), 1.0)
    spectrum = np.fft.rfft(base)*(1-weight)+np.fft.rfft(enhanced)*weight
    result = np.fft.irfft(spectrum, n=len(base)).astype(np.float32)
    if sibilance_guard and len(base) >= 960:
        frame, hop = 960, 480
        count = 1+(len(base)-frame)//hop
        windows = _framed_matrix(base, frame, hop).astype(np.float64)*np.hanning(frame)
        spectrum = np.abs(np.fft.rfft(windows, axis=1))**2+1e-12
        bins = np.fft.rfftfreq(frame, 1.0/rate)
        high = spectrum[:, (bins >= 6000)&(bins <= 12000)].sum(axis=1)
        voice = spectrum[:, (bins >= 300)&(bins <= 12000)].sum(axis=1)
        guard = np.ones(count, dtype=np.float64)
        if guard_mode == 'base':
            ratio = np.where(voice > 0, high/np.maximum(voice, 1e-12), 0.0)
            guard[ratio > sib_ratio] = sib_mix
        else:
            glance = _framed_matrix(enhanced, frame, hop).astype(np.float64)*np.hanning(frame)
            restored = np.abs(np.fft.rfft(glance, axis=1))**2+1e-12
            renewed = restored[:, (bins >= 6000)&(bins <= 12000)].sum(axis=1)
            body = restored[:, (bins >= 300)&(bins <= 12000)].sum(axis=1)
            added = 10*np.log10(np.maximum(renewed, 1e-12)/np.maximum(high, 1e-12))
            share = np.where(body > 0, renewed/np.maximum(body, 1e-12), 0.0)
            guard[(body > 0) & (share > sib_ratio) & (added > excess_db)] = sib_mix
        centres = (np.arange(count)*hop+frame//2).astype(np.float64)
        mask = np.interp(np.arange(len(base)), centres, guard, left=1.0, right=1.0)
        result = (base*(1-mask)+result*mask).astype(np.float32)
    if len(result) != len(base) or not np.isfinite(result).all():
        raise RuntimeError('Invalid blended output')
    return result


def utterance_limit(enhancer):
    return QUALITY_MAX_SECONDS if enhancer in LONG_FORM_ENHANCERS else MAX_SECONDS


class UtteranceCollector:
    sample_rate = 48000
    frame_samples = 960

    def __init__(self, silence_ms=800, threshold_db=-48, max_seconds=MAX_SECONDS, keep_ms=200):
        if type(max_seconds) is not int or not 1 <= max_seconds <= QUALITY_MAX_SECONDS:
            raise ValueError('Utterance limit must be an integer of at most %d seconds' % QUALITY_MAX_SECONDS)
        self.max_seconds = max_seconds
        self.silence_frames = round(silence_ms / 20)
        self.threshold = 10 ** (threshold_db / 20)
        self.keep_frames = max(0, round(keep_ms / 20))
        self.reset()

    def reset(self):
        self.preroll = deque(maxlen=10)
        self.parts = []
        self.samples = self.silent = self.voiced = 0
        self.pending = np.empty(0, dtype=np.float32)
        self.limit_splits = 0
        self._result_flags = deque()

    def take_result_flags(self):
        """Drain one continuation flag per completed result, oldest first.

        ``feed``/``flush`` append exactly one flag per returned utterance:
        True when the utterance hit the length cap (more speech follows in the
        same stream), False when it ended on silence or manual finish. The
        caller drains in lockstep with the returned results; ``reset`` clears
        any undrained remainder so a stale flag can never leak across epochs.
        """
        flags = list(self._result_flags)
        self._result_flags.clear()
        return flags

    def finish(self):
        if not self.parts:
            return None
        # Keep some of the final pause; retain the beginning and all voiced audio.
        trim = max(0, self.silent - self.keep_frames) * self.frame_samples
        audio = np.concatenate(self.parts)
        if trim:
            audio = audio[:-trim]
        valid = self.voiced >= 5
        self.parts = []; self.samples = self.silent = self.voiced = 0
        self.preroll.clear()
        return audio if valid and len(audio) else None

    def feed(self, audio):
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError('Invalid utterance input')
        self.pending = np.concatenate((self.pending, audio))
        results = []
        consumed = 0
        while len(self.pending) - consumed >= self.frame_samples:
            frame = self.pending[consumed:consumed+self.frame_samples].copy()
            consumed += self.frame_samples
            voiced = float(np.sqrt(np.mean(frame.astype(np.float64)**2))) >= self.threshold
            if not self.parts:
                if not voiced:
                    self.preroll.append(frame)
                    continue
                self.parts = list(self.preroll)
                self.samples = sum(map(len, self.parts))
                self.preroll.clear()
            self.parts.append(frame); self.samples += len(frame)
            self.voiced += int(voiced)
            self.silent = 0 if voiced else self.silent + 1
            at_limit = self.samples >= self.max_seconds*self.sample_rate
            if self.silent >= self.silence_frames or at_limit:
                if at_limit:self.limit_splits += 1
                result = self.finish()
                if result is not None:
                    results.append(result)
                    self._result_flags.append(bool(at_limit))
        self.pending = self.pending[consumed:].copy()
        return results

    def flush(self):
        if self.parts and len(self.pending):
            self.parts.append(self.pending.copy());self.samples+=len(self.pending)
        self.pending=np.empty(0,dtype=np.float32)
        result = self.finish()
        # A manual/endpoint finish never continues: any cap split already left
        # through feed() with its own flag.
        if result is not None:
            self._result_flags.append(False)
        return result


def _convert_core(backend, down, up, audio, delay, hop):
    """One reset-bounded conversion pass with the fixed delay trimmed.

    The caller decides how many passes an utterance gets: one for the shipped
    route, one per pause-bounded segment for the experimental refresh route.
    """
    backend.reset()
    if down:down.reset(); up.reset()
    padded_length = ((len(audio)+delay+48000+hop-1)//hop)*hop
    padded = np.zeros(padded_length,dtype=np.float32);padded[:len(audio)] = audio
    parts=[]
    for start in range(0,len(padded),hop):
        block=padded[start:start+hop]
        reduced=down.process(block) if down else block
        converted=backend.process_chunk(reduced)
        parts.append(up.process(converted) if up else converted)
    result=np.concatenate(parts)[delay:delay+len(audio)].copy()
    if len(result)!=len(audio) or not np.isfinite(result).all():
        raise RuntimeError('Invalid utterance conversion output')
    backend.reset()
    return result


def convert_utterance(backend, down, up, audio, enhance=None, max_seconds=MAX_SECONDS,
                      statistics=None, refresh_pauses=False):
    """Reset and drain the existing approved VC route, then remove its fixed delay.

    The whole waveform is available before conversion. MeanVC2 still uses its
    pretrained internal streaming hops; this does not pretend to add global attention.
    ``enhance`` (MossFormerSR/FlashSR) receives the delay-trimmed 48 kHz result and
    must return the same length; its failure is reported, never silently skipped.

    Silence-aware conversion is opt-in through ``statistics``. When supplied it is
    filled in with how much of the utterance was inaudible, which is the only thing
    that can be skipped safely: MeanVC2's vocoder is stateful per call, so dropping
    silent blocks would change the output around every gap. The reported figure is
    therefore an opportunity, not a saving.
    """
    if type(max_seconds) is not int or not 1 <= max_seconds <= QUALITY_MAX_SECONDS:
        raise ValueError('Utterance limit must be an integer of at most %d seconds' % QUALITY_MAX_SECONDS)
    if not 0 < len(audio) <= max_seconds*48000 or not np.isfinite(audio).all():
        raise ValueError('Utterance must be finite mono audio of at most %d seconds' % max_seconds)
    if statistics is not None:
        from .runtime_audio import total_silence_seconds
        silence = total_silence_seconds(audio, 48000)
        statistics['input_silence_seconds'] = round(silence, 3)
        statistics['input_seconds'] = round(len(audio)/48000, 3)
        statistics['silence_share'] = round(silence/max(len(audio)/48000, 1e-9), 4)
    backend.reset()
    if down:down.reset(); up.reset()
    stats = backend.get_stats()
    delay = round((stats['algorithmic_buffer_ms'] + stats.get('interpolation_grid_delay_ms',0)
                   + stats.get('phrase_extra_delay_ms',0))*48)
    hop = backend.chunk_samples * (48000//backend.sample_rate)
    if refresh_pauses:
        # Pause-aware refresh: a long silence resynchronises the recurrent state
        # instead of letting it drift across the whole utterance. Each segment is
        # converted with context on both sides and rejoined with a crossfade, so
        # the output length still matches the input exactly. Short utterances
        # take the single pass: drift is negligible there and each extra pass
        # would only add flush padding against the RTF budget.
        segments = split_at_pauses(audio) if len(audio) >= REFRESH_MIN_SECONDS*48000 else [(0, len(audio))]
        if len(segments) > 1:
            extension = REFRESH_OVERLAP_SAMPLES//2
            converted = []
            for start, end in segments:
                extended = audio[max(0,start-extension):min(len(audio),end+extension)]
                converted.append(_convert_core(backend, down, up, extended, delay, hop))
            result = crossfade_join(converted, REFRESH_OVERLAP_SAMPLES)
            if len(result) > len(audio):
                result = result[:len(audio)]
            elif len(result) < len(audio):
                result = np.concatenate((result, np.zeros(len(audio)-len(result), dtype=np.float32)))
        else:
            result = _convert_core(backend, down, up, audio, delay, hop)
        if statistics is not None:
            statistics['refresh_segments'] = len(segments)
            statistics['refresh_overlap_samples'] = REFRESH_OVERLAP_SAMPLES if len(segments) > 1 else 0
    else:
        result = _convert_core(backend, down, up, audio, delay, hop)
    if len(result)!=len(audio) or not np.isfinite(result).all():
        raise RuntimeError('Invalid utterance conversion output')
    if enhance is not None:
        from .runtime_audio import skip_enhancement
        decision = skip_enhancement(result, 48000)
        if statistics is not None:
            statistics['enhancer_skipped'] = bool(decision['skip'])
            statistics['enhancer_skip_reason'] = decision['reason']
            statistics['output_silence_seconds'] = round(decision['silence_seconds'], 3)
        if not decision['skip']:
            result=enhance(result)
            if len(result)!=len(audio) or not np.isfinite(result).all():
                raise RuntimeError('Invalid utterance enhancer output')
    return result
