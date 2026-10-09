"""Bounded, optional phrase repair. Analysis/editing never runs on audio callbacks.

The neural voice is untouched. One asynchronous job may be outstanding; late
or uncertain repairs bypass to the identically delayed neural waveform.
"""
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np

from src.prosody.f0 import YinEstimator


RATE = 16000
HOP = 2560
LOOKAHEAD = 10 * HOP  # 1600ms; jobs get one input hop before their deadline.
HISTORY = 2 * RATE
MAX_ALIGNMENT_SAMPLES = 2 * RATE  # Includes aligned/grouped VC's 1640ms delay.


def pitch_edit(audio, frequency, semitones, rate=RATE):
    """Small fixed-duration TD-PSOLA edit with local, waveform-refined marks.

    Only called for confidently periodic speech. No formant/EQ/global shift.
    An identity request is exact; inadequate marks bypass without resynthesis.
    `rate` lets the utterance retrospective reuse the same edit at 48 kHz.
    """
    x = np.asarray(audio, dtype=np.float32)
    if abs(semitones) < .015 or not 80 <= frequency <= 550:
        return x.copy()
    period = rate / frequency
    radius = max(2, round(period * .18))
    anchor = int(np.argmax(x[:min(len(x), round(period * 2))]))
    marks = [anchor]
    while marks[-1] + period + radius < len(x):
        predicted = round(marks[-1] + period)
        lo, hi = predicted - radius, predicted + radius + 1
        marks.append(lo + int(np.argmax(x[lo:hi])))
    marks = np.asarray(marks)
    width = round(period)
    marks = marks[(marks >= width) & (marks + width < len(x))]
    if len(marks) < 4:
        return x.copy()
    window = np.hanning(2 * width + 1)
    accum = np.zeros(len(x), dtype=np.float64)
    weight = np.zeros(len(x), dtype=np.float64)
    step = period / (2 ** (float(np.clip(semitones, -.6, .6)) / 12))
    for position in np.arange(marks[0], marks[-1] + .5, step):
        dest = round(position)
        mark = marks[int(np.argmin(abs(marks - position)))]
        lo, hi = dest - width, dest + width + 1
        if lo < 0 or hi > len(x):
            continue
        accum[lo:hi] += x[mark-width:mark+width+1] * window
        weight[lo:hi] += window
    result = x.copy()
    valid = weight > .2
    result[valid] = (accum[valid] / weight[valid]).astype(np.float32)
    return result


def repair_window(source, voice, center, energy=True, pitch=True, features=None):
    """Analyze matching clocks, edit only the central 160ms, return diagnostics.

    Relative contours remove each speaker's median pitch and loudness. The
    source's absolute male F0 is never the output target. No global alignment
    guessing or unbounded Viterbi decoding occurs in the live route.
    """
    started = time.perf_counter()
    estimator = YinEstimator()
    step = 640  # 40ms features; overlapping 80ms analysis windows.
    positions = np.arange(1280, len(voice) + 1, step)
    if features is None:
        a = np.array([estimator.estimate(source[p-1280:p]) for p in positions])
        b = np.array([estimator.estimate(voice[p-1280:p]) for p in positions])
    else:
        a, b = features
    # Both need confidently periodic audio and sensible speech pitch.
    valid = ((a[:, 1] >= .88) & (b[:, 1] >= .88) &
             (a[:, 0] >= 80) & (b[:, 0] >= 80) &
             (a[:, 0] <= 550) & (b[:, 0] <= 550) &
             (a[:, 2] > -48) & (b[:, 2] > -48))
    f0_delta = np.zeros(len(positions))
    gain_delta = np.zeros(len(positions))
    trusted = np.zeros(len(positions), dtype=bool)
    if valid.sum() >= 8:
        sa = 12 * np.log2(np.maximum(a[:, 0], 1))
        sb = 12 * np.log2(np.maximum(b[:, 0], 1))
        error = (sa - np.median(sa[valid])) - (sb - np.median(sb[valid]))
        # Octave errors / unrelated contours must not become large repairs.
        trusted = valid & (abs(error) <= 4)
        for k in range(1, len(valid)):
            if valid[k] and valid[k-1] and (abs(sa[k]-sa[k-1]) > 5 or abs(sb[k]-sb[k-1]) > 5):
                trusted[k] = False
        if pitch:
            f0_delta[trusted] = np.clip(error[trusted] * .20, -.6, .6)
        if energy:
            relative = (a[:, 2] - np.median(a[valid, 2])) - (b[:, 2] - np.median(b[valid, 2]))
            gain_delta[trusted] = np.clip(relative[trusted] * .25, -1.5, 1.5)
            gain_delta[abs(gain_delta) < .05] = 0  # Ignore window-phase noise.
    # Analysis windows are trailing: timestamp their middle, not their end.
    times = positions - 640
    indices = np.arange(center, center + HOP)
    gain = np.interp(indices, times, gain_delta, left=0, right=0)
    shift = np.interp(indices, times, f0_delta, left=0, right=0)
    out = voice[center:center+HOP].copy()
    edited = 0
    # Local grains use 40ms context on either side. Crossfade only edited
    # regions to the original, preserving unvoiced samples exactly.
    for start in range(0, HOP, step):
        absolute = center + start
        nearest = int(np.argmin(abs(times - (absolute + step // 2))))
        # Require both feature endpoints around this grain to be trustworthy.
        # Otherwise interpolation could carry a preceding vowel's correction
        # into a consonant, silence, or an octave-error region.
        endpoints = np.searchsorted(times, [absolute, absolute + step - 1])
        lo_feature = max(0, int(endpoints[0]) - 1)
        hi_feature = min(len(times) - 1, int(endpoints[1]))
        if not np.all(trusted[lo_feature:hi_feature+1]):
            gain[start:start+step] = 0
            shift[start:start+step] = 0
            continue
        delta = float(np.mean(shift[start:start+step]))
        lo, hi = absolute-step, absolute+2*step
        if pitch and abs(delta) >= .03 and lo >= 0 and hi <= len(voice):
            segment = pitch_edit(voice[lo:hi], float(b[nearest, 0]), delta)[step:2*step]
            blend = np.sin(np.linspace(0, np.pi, step)) ** 2
            out[start:start+step] += (segment-out[start:start+step]) * blend
            edited += 1
    # Gain also returns smoothly to identity at boundaries between chunks.
    envelope = np.ones(HOP)
    edge = 160
    envelope[:edge] = np.linspace(0, 1, edge)
    envelope[-edge:] = np.linspace(1, 0, edge)
    out *= np.power(10., gain * envelope / 20).astype(np.float32)
    # Only attenuate if a repair would introduce clipping. No compression.
    original_peak = float(np.max(abs(voice[center:center+HOP])))
    peak = float(np.max(abs(out)))
    if peak > max(.98, original_peak):
        out *= max(.98, original_peak) / peak
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite phrase repair')
    return out, dict(analysis_ms=(time.perf_counter()-started)*1000,
                    voiced_frames=int(valid.sum()), pitch_regions=edited,
                    max_shift_st=float(np.max(abs(shift))),
                    max_gain_db=float(np.max(abs(gain))))


class PhraseAnalyzer:
    """Analysis-thread-only feature cache, keyed by absolute sample clock/epoch."""
    def __init__(self):
        self.epoch = None
        self.cache = {}
        self.estimator = YinEstimator()

    def __call__(self, source, voice, center, energy, pitch, origin, epoch):
        started = time.perf_counter()
        if epoch != self.epoch:
            self.cache = {}
            self.epoch = epoch
        positions = np.arange(1280, len(voice)+1, 640)
        current = {}
        estimated = 0
        for p in positions:
            key = int(origin+p)
            feature = self.cache.get(key)
            if feature is None:
                feature = (self.estimator.estimate(source[p-1280:p]),
                           self.estimator.estimate(voice[p-1280:p]))
                estimated += 2
            current[key] = feature
        self.cache = current  # bounded by this finite window, never utterance length
        pairs = list(current.values())
        features = (np.array([v[0] for v in pairs]), np.array([v[1] for v in pairs]))
        result, stats = repair_window(source, voice, center, energy, pitch, features)
        stats.update(analysis_ms=(time.perf_counter()-started)*1000,
                     estimated_frames=estimated, cached_frames=len(current))
        return result, stats


class PhraseRepair:
    """Fixed-delay repair with bounded history and a single nonblocking job.

    The delay is ``lookahead_hops`` input hops of future context plus one
    output hop: analysis for an output position sees that much future audio
    before the correction is applied. The shipped 10-hop setting is the
    human-approved condition; shorter settings trade future context (and
    therefore correction quality, especially at endings) for latency and must
    be listening-verified per recipe, never silently adopted.
    """
    def __init__(self, alignment_samples, energy=True, pitch=True, focus=None,
                 lookahead_hops=10):
        if type(alignment_samples) is not int or not 0 <= alignment_samples <= MAX_ALIGNMENT_SAMPLES:
            raise ValueError(f'Invalid neural/source alignment: expected integer samples in 0..{MAX_ALIGNMENT_SAMPLES}')
        if type(lookahead_hops) is not int or not 1 <= lookahead_hops <= 10:
            raise ValueError('Repair lookahead must be 1..10 input hops')
        if focus not in (None, 'ending'):
            raise ValueError('Unsupported phrase repair focus')
        self.alignment = alignment_samples
        self.lookahead_hops = lookahead_hops
        self.energy, self.pitch = energy, pitch
        # 'ending' ramps corrections up over the utterance so the closing prosody
        # gets the strongest repair. Anything else behaves exactly as before.
        self.focus = focus
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='phrase-analysis')
        self.analyzer = PhraseAnalyzer()
        self.pending = None
        self.epoch = 0
        self.reset()

    def reset(self):
        self.epoch += 1
        if self.pending is not None:
            self.pending[2].cancel()
        self.source_delay = np.zeros(self.alignment, np.float32)
        self.source = np.empty(0, np.float32)
        self.voice = np.empty(0, np.float32)
        self.origin = self.end = 0
        self.completed = {}
        self.stats = dict(applied=0, late_bypass=0, busy_skip=0, failures=0,
                          disabled=False, last_error='', expired_results=0,
                          estimated_frames=0, cached_frames=0,
                          extra_delay_ms=self.lookahead_hops*160, analysis_ms=0., max_analysis_ms=0.,
                          pitch_regions=0, max_shift_st=0., max_gain_db=0.)

    def realign(self, alignment_samples):
        """Follow a changed algorithmic buffer; the source/voice offset must match it."""
        if type(alignment_samples) is not int or not 0 <= alignment_samples <= MAX_ALIGNMENT_SAMPLES:
            raise ValueError('Invalid neural/source alignment: expected integer samples in 0..%d' % MAX_ALIGNMENT_SAMPLES)
        if alignment_samples == self.alignment:
            return
        self.alignment = alignment_samples
        self.reset()

    def _read(self, array, start, end):
        result = np.zeros(end-start, np.float32)
        lo, hi = max(start, self.origin), min(end, self.end)
        if hi > lo:
            result[lo-start:hi-start] = array[lo-self.origin:hi-self.origin]
        return result

    def process(self, source, voice):
        if source.shape != (HOP,) or voice.shape != (HOP,) or not np.isfinite(source).all() or not np.isfinite(voice).all():
            raise ValueError('Expected finite mono 160ms chunks')
        joined = np.concatenate((self.source_delay, source))
        aligned = joined[:HOP]
        self.source_delay = joined[HOP:]
        self.source = np.concatenate((self.source, aligned))
        self.voice = np.concatenate((self.voice, voice))
        self.end += HOP
        lookahead = self.lookahead_hops*HOP
        target = self.end - lookahead - HOP
        if self.pending is not None and self.pending[2].done():
            epoch, at, future = self.pending
            self.pending = None
            if epoch == self.epoch:
                try:
                    completed = future.result()
                    if at >= target:
                        self.completed[at] = completed
                    else:
                        self.stats['expired_results'] += 1
                except Exception as error:
                    self.stats['failures'] += 1
                    self.stats['last_error'] = str(error)[:300]
                    self.stats['disabled'] = True
        result = self._read(self.voice, target, target+HOP)
        if target >= 0:
            repaired = self.completed.pop(target, None)
            if repaired is not None:
                fresh, diagnostics = repaired
                if self.focus == 'ending':
                    # Corrections start gentle and reach full strength ~3 s into
                    # the utterance, so the ending carries the repair instead of
                    # the onset.
                    scale = 0.3+0.7*min(1.0, self.end/48000.0)
                    result = (result+(fresh-result)*scale).astype(np.float32)
                else:
                    result = fresh
                self.stats['applied'] += 1
                self.stats['analysis_ms'] = diagnostics['analysis_ms']
                self.stats['max_analysis_ms'] = max(self.stats['max_analysis_ms'], diagnostics['analysis_ms'])
                self.stats['pitch_regions'] += diagnostics['pitch_regions']
                self.stats['max_shift_st'] = max(self.stats['max_shift_st'], diagnostics['max_shift_st'])
                self.stats['max_gain_db'] = max(self.stats['max_gain_db'], diagnostics['max_gain_db'])
                self.stats['estimated_frames'] += diagnostics.get('estimated_frames',0)
                self.stats['cached_frames'] = diagnostics.get('cached_frames',0)
            else:
                self.stats['late_bypass'] += 1
        self.completed = {k: v for k, v in self.completed.items() if k > target}
        next_target = target + HOP
        if next_target >= 0 and self.pending is None and not self.stats['disabled']:
            begin = next_target - HISTORY
            self.pending = (self.epoch, next_target, self.executor.submit(
                self.analyzer, self._read(self.source, begin, self.end),
                self._read(self.voice, begin, self.end), HISTORY, self.energy, self.pitch, begin, self.epoch))
        elif next_target >= 0 and self.pending is not None:
            self.stats['busy_skip'] += 1
        retain = max(0, self.end - lookahead - HISTORY - HOP)
        discard = retain-self.origin
        if discard > 0:
            self.source = self.source[discard:].copy()
            self.voice = self.voice[discard:].copy()
            self.origin = retain
        return result

    def close(self):
        self.epoch += 1
        self.executor.shutdown(wait=False, cancel_futures=True)


def _downsample_48_to_16(audio):
    """Offline analysis decimation, preferably through the streaming resampler.

    The main test environment has no scipy, so fall back to a short moving
    average plus decimation there. Analysis envelopes and F0 live far below
    the affected band, so the fallback only feeds measurements, never audio.
    """
    audio = np.asarray(audio, dtype=np.float32)
    try:
        from .resampler import StreamingResampler
        down = StreamingResampler(48000, 16000)
        down.reset()
        return np.asarray(down.process(audio), dtype=np.float32)
    except ImportError:
        kernel = np.ones(9, dtype=np.float64)/9
        smooth = np.convolve(np.pad(audio, (4, 4), mode='edge'), kernel, mode='valid')
        return smooth[::3].astype(np.float32)[:len(audio)//3]


def _frame_features(audio16):
    """Per-40ms voiced level/pitch/confidence triplets for whole-utterance analysis."""
    estimator = YinEstimator()
    step = 640
    positions = np.arange(1280, len(audio16)+1, step)
    rows = np.array([estimator.estimate(audio16[p-1280:p]) for p in positions])
    times = positions-640
    return times, rows


def analyze_utterance_pair(source48, voice48):
    """Shared whole-utterance analysis for the retrospective pass and metrics.

    Downsampling and F0 estimation run once here so the integrated route does
    not pay them twice per utterance.
    """
    source48 = np.asarray(source48, dtype=np.float32)
    voice48 = np.asarray(voice48, dtype=np.float32)
    if source48.shape != voice48.shape or source48.ndim != 1 or not len(source48):
        raise ValueError('Analysis needs two finite mono signals of equal length')
    if not np.isfinite(source48).all() or not np.isfinite(voice48).all():
        raise ValueError('Analysis needs two finite mono signals of equal length')
    source, voice = _downsample_48_to_16(source48), _downsample_48_to_16(voice48)
    times, a = _frame_features(source)
    _, b = _frame_features(voice)
    voiced = ((a[:, 1] >= .88) & (b[:, 1] >= .88) &
              (a[:, 0] >= 80) & (a[:, 0] <= 550) &
              (b[:, 0] >= 80) & (b[:, 0] <= 550) &
              (a[:, 2] > -48) & (b[:, 2] > -48))
    return dict(times=times, source_features=a, voice_features=b, voiced=voiced)


def utterance_metrics(source48, voice48, analysis=None):
    """Whole-utterance objective read-out: energy correlation, F0 spread, jumps.

    Same length in/out required. Returns a plain dict of floats the worker
    reports in its statistics; listening still decides.
    """
    if analysis is None:
        analysis = analyze_utterance_pair(source48, voice48)
    else:
        source48 = np.asarray(source48, dtype=np.float32)
        voice48 = np.asarray(voice48, dtype=np.float32)
        if source48.shape != voice48.shape or source48.ndim != 1 or not len(source48):
            raise ValueError('Metrics need two finite mono signals of equal length')
    times, a, b, voiced = (analysis['times'], analysis['source_features'],
                           analysis['voice_features'], analysis['voiced'])
    metrics = dict(voiced_frames=int(voiced.sum()), frames=len(a))
    if voiced.sum() >= 4:
        sa = a[voiced, 2].astype(np.float64)
        sb = b[voiced, 2].astype(np.float64)
        metrics['energy_correlation'] = float(np.corrcoef(sa, sb)[0, 1]) if sa.std() > 0 and sb.std() > 0 else 0.0
        fa = 12*np.log2(np.maximum(a[voiced, 0], 1).astype(np.float64))
        fb = 12*np.log2(np.maximum(b[voiced, 0], 1).astype(np.float64))
        metrics['source_f0_range_st'] = float(np.percentile(fa, 95)-np.percentile(fa, 5))
        metrics['voice_f0_range_st'] = float(np.percentile(fb, 95)-np.percentile(fb, 5))
        jumps = np.abs(np.diff(fb))
        metrics['octave_jump_rate'] = float((jumps > 10).sum()/max(len(jumps), 1))
        tail = slice(max(0, int(voiced.sum()*0.85)), voiced.sum())
        tail_n = tail.stop-tail.start
        metrics['ending_energy_slope_db'] = float(
            np.polyfit(np.arange(tail_n), sb[tail], 1)[0]) if tail_n >= 2 else 0.0
    else:
        metrics.update(energy_correlation=0.0, source_f0_range_st=0.0,
                       voice_f0_range_st=0.0, octave_jump_rate=0.0,
                       ending_energy_slope_db=0.0)
    return {key: round(value, 4) if isinstance(value, float) else value for key, value in metrics.items()}


def retrospective_repair(source48, voice48, analysis=None, adaptive=False, cap_boost=1.0,
                         match_rate=0.25):
    """Non-causal utterance shaping for the integrated comparison route.

    The whole utterance is known, so unlike the streaming repair this matches
    the converted energy contour against the source's relative dynamics, with
    extra room at the ending, while breaths, onsets and unvoiced frames are
    left alone. Same length in, same length out; never invents content.
    With `adaptive`, the gain caps scale with the source's own voiced dynamic
    range (bounded 2.0..3.5 dB, ending 3.0..4.0 dB): typical readings behave
    exactly as before, while dramatic passages are compressed less.
    `cap_boost` (0.5..1.5) scales those adaptive caps for the tuning GUI;
    `match_rate` (0..0.5) scales how much of the relative dynamics is applied.
    Defaults reproduce the validated recipe exactly.
    """
    source48 = np.asarray(source48, dtype=np.float32)
    voice48 = np.asarray(voice48, dtype=np.float32)
    if source48.shape != voice48.shape or source48.ndim != 1 or not len(source48):
        raise ValueError('Repair needs two finite mono signals of equal length')
    if not np.isfinite(source48).all() or not np.isfinite(voice48).all():
        raise ValueError('Repair needs two finite mono signals of equal length')
    if not np.isfinite(np.asarray(cap_boost, dtype=np.float64)).all() \
            or not 0.5 <= float(cap_boost) <= 1.5:
        raise ValueError('Cap boost must be 0.5..1.5')
    if not np.isfinite(np.asarray(match_rate, dtype=np.float64)).all() \
            or not 0 <= float(match_rate) <= 0.5:
        raise ValueError('Match rate must be 0..0.5')
    if analysis is None:
        analysis = analyze_utterance_pair(source48, voice48)
    times, a, b, voiced = (analysis['times'], analysis['source_features'],
                           analysis['voice_features'], analysis['voiced'])
    floor = float(np.percentile(a[:, 2], 5))
    breath = (~voiced) & (a[:, 1] < 0.4) & (a[:, 2] > floor+3.0) & (b[:, 2] > floor+3.0)
    gain = np.zeros(len(times))
    matched = 0
    base_cap, end_cap = 2.0, 3.0
    if adaptive and voiced.sum() >= 4:
        spread = float(np.std(a[voiced, 2].astype(np.float64)))
        scale = np.clip(spread/4.0, 1.0, 1.75)
        base_cap = min(max(2.0*scale*float(cap_boost), 1.0), 5.0)
        end_cap = min(max(3.0*scale*float(cap_boost), 1.0), 6.0)
    if voiced.sum() >= 4:
        sa = a[:, 2].astype(np.float64)
        sb = b[:, 2].astype(np.float64)
        relative = (sa-np.median(sa[voiced]))-(sb-np.median(sb[voiced]))
        last = int(len(times)*0.85)
        cap = np.full(len(times), base_cap)
        cap[last:] = end_cap  # the ending gets extra room, where flatness shows most
        gain[voiced] = np.clip(relative[voiced]*float(match_rate), -cap[voiced], cap[voiced])
        gain[np.abs(gain) < 0.05] = 0.0
        gain[breath] = 0.0
        matched = int((np.abs(gain) > 0).sum())
    # Smooth over ~200 ms so frame edges never click.
    kernel = np.ones(5)/5
    smooth = np.convolve(np.pad(gain, 2, mode='edge'), kernel, mode='valid')
    # Onset guard: no cuts in the first 300 ms, only gentle lifts.
    onset_frames = int(300/40)
    smooth[:onset_frames] = np.maximum(smooth[:onset_frames], -1.0)
    envelope = np.interp(np.arange(len(voice48)), times*3, smooth, left=0.0, right=0.0)
    out = (voice48.astype(np.float64)*10**(envelope/20)).astype(np.float32)
    original_peak = float(np.max(abs(voice48)))
    peak = float(np.max(abs(out)))
    if peak > max(.98, original_peak):
        out = (out*max(.98, original_peak)/peak).astype(np.float32)
    if len(out) != len(voice48) or not np.isfinite(out).all():
        raise RuntimeError('Invalid retrospective repair output')
    report = dict(matched_frames=matched,
                  max_gain_db=round(float(np.max(abs(smooth))), 3),
                  breath_frames=int(breath.sum()),
                  voiced_frames=int(voiced.sum()))
    if adaptive:
        report.update(adaptive_caps=[round(base_cap, 3), round(end_cap, 3)],
                      cap_boost=round(float(cap_boost), 3))
    return out, report


def retrospective_pitch(source48, voice48, analysis=None, transfer=0.20, max_shift_st=1.0):
    """Non-causal intonation residue repair for the comparison route.

    Mirrors the live repair's transfer rate (20%) and trust gates, but works
    on the whole utterance at 48 kHz with a wider ±1.0 st cap: it picks up the
    phrase-level residue the streaming repair left behind (late-bypassed
    windows, ending contours). Only confidently periodic vowel frames are
    touched, in 40 ms grains with sin^2 crossfades; breaths, onsets, unvoiced
    frames and octave jumps bypass exactly. Same length in, same length out.
    """
    source48 = np.asarray(source48, dtype=np.float32)
    voice48 = np.asarray(voice48, dtype=np.float32)
    if source48.shape != voice48.shape or source48.ndim != 1 or not len(source48):
        raise ValueError('Repair needs two finite mono signals of equal length')
    if not np.isfinite(source48).all() or not np.isfinite(voice48).all():
        raise ValueError('Repair needs two finite mono signals of equal length')
    if not 0 <= transfer <= 1:
        raise ValueError('Pitch transfer must be 0..1')
    if not 0 < max_shift_st <= 2:
        raise ValueError('Pitch cap must be 0..2 st')
    if analysis is None:
        analysis = analyze_utterance_pair(source48, voice48)
    times, a, b, voiced = (analysis['times'], analysis['source_features'],
                           analysis['voice_features'], analysis['voiced'])
    valid = ((a[:, 1] >= .88) & (b[:, 1] >= .88) &
             (a[:, 0] >= 80) & (b[:, 0] >= 80) &
             (a[:, 0] <= 550) & (b[:, 0] <= 550) &
             (a[:, 2] > -48) & (b[:, 2] > -48))
    shift = np.zeros(len(times))
    trusted = np.zeros(len(times), dtype=bool)
    if valid.sum() >= 8:
        sa = 12*np.log2(np.maximum(a[:, 0], 1).astype(np.float64))
        sb = 12*np.log2(np.maximum(b[:, 0], 1).astype(np.float64))
        error = (sa-np.median(sa[valid]))-(sb-np.median(sb[valid]))
        trusted = valid & (abs(error) <= 4)
        for k in range(1, len(valid)):
            if valid[k] and valid[k-1] and (abs(sa[k]-sa[k-1]) > 5 or abs(sb[k]-sb[k-1]) > 5):
                trusted[k] = False
        shift[trusted] = np.clip(error[trusted]*transfer, -max_shift_st, max_shift_st)
        shift[np.abs(shift) < .03] = 0.0
    # Smooth over ~200 ms so grain edges never click.
    kernel = np.ones(5)/5
    smooth = np.convolve(np.pad(shift, 2, mode='edge'), kernel, mode='valid')
    # Onset guard: halve corrections in the first 300 ms.
    onset_frames = int(300/40)
    smooth[:onset_frames] *= 0.5
    floor = float(np.percentile(a[:, 2], 5))
    breath = ((~voiced) & (a[:, 1] < 0.4) & (a[:, 2] > floor+3.0) & (b[:, 2] > floor+3.0))
    out = voice48.copy()
    edited = 0
    step = 1920  # 40 ms grains at 48 kHz, mirroring the live 40 ms path.
    blend = np.sin(np.linspace(0, np.pi, step)) ** 2
    for start in range(0, len(out), step):
        absolute = start
        endpoints = np.searchsorted(times*3, [absolute, absolute+step-1])
        lo_feature = max(0, int(endpoints[0])-1)
        hi_feature = min(len(times)-1, int(endpoints[1]))
        if not np.all(trusted[lo_feature:hi_feature+1]):
            continue
        delta = float(np.mean(smooth[lo_feature:hi_feature+1]))
        nearest = int(np.argmin(abs(times*3-(absolute+step//2))))
        lo, hi = absolute-step, absolute+2*step
        if abs(delta) < .03 or lo < 0 or hi > len(out):
            continue
        segment = pitch_edit(out[lo:hi], float(b[nearest, 0]), delta,
                             rate=48000)[step:2*step]
        stop = min(start+step, len(out))
        width = stop-start
        out[start:stop] += (segment[:width]-out[start:stop])*blend[:width]
        edited += 1
    if len(out) != len(voice48) or not np.isfinite(out).all():
        raise RuntimeError('Invalid retrospective pitch output')
    return out, dict(shifted_grains=edited, trusted_frames=int(trusted.sum()),
                     max_shift_st=round(float(np.max(abs(smooth))), 3),
                     voiced_frames=int(voiced.sum()))


def breath_sample_mask(analysis, length):
    """Sample-rate breath mask from a whole-utterance analysis.

    Returns a float32 0..1 mask of `length` samples at 48 kHz that is 1 where
    the retrospective repair saw breath frames, so later stages (e.g. floor
    suppression) can exempt breaths instead of pressing them down. Same
    breath definition as the repair; never invents content.
    """
    if type(length) is not int or length <= 0:
        raise ValueError('Breath mask needs a positive sample length')
    times = np.asarray(analysis['times'], dtype=np.float64)
    voiced = np.asarray(analysis['voiced'])
    a = np.asarray(analysis['source_features'], dtype=np.float64)
    b = np.asarray(analysis['voice_features'], dtype=np.float64)
    if times.ndim != 1 or voiced.ndim != 1 or not len(times) or len(times) != len(voiced) \
            or a.shape[0] != len(times) or b.shape[0] != len(times):
        raise ValueError('Breath mask needs a matching utterance analysis')
    floor = float(np.percentile(a[:, 2], 5))
    breath = ((~voiced) & (a[:, 1] < 0.4) & (a[:, 2] > floor+3.0) & (b[:, 2] > floor+3.0))
    mask = np.interp(np.arange(length), times*3, breath.astype(np.float64),
                     left=0.0, right=0.0)
    out = np.clip(mask, 0.0, 1.0).astype(np.float32)
    if len(out) != length or not np.isfinite(out).all():
        raise RuntimeError('Invalid breath mask output')
    return out
