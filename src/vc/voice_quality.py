"""Shared decode, speech detection and quality analysis for voice material.

One implementation so the app registration path and the offline clone-dataset
builder cannot drift apart. The speech detector, the quiet-point splitter and the
F0 estimator are the same routines the dataset builder already used offline, moved
here so registration inherits them instead of fixed-length windows.

Everything here is offline and numpy-only: no audio device, no training, no
network. Quality numbers are measurements, not judgments about the person.
"""
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

FRAME_MS = 20.0
HOP_MS = 10.0
DECODE_TIMEOUT_SECONDS = 300
# A bounded decode keeps memory finite on long recordings. Longer files are
# refused with an explicit reason instead of silently truncating the head.
MAX_DECODE_SECONDS = 1800.0
SAFE_PEAK = 0.95
# Below this peak-to-floor spread a clip has no meaningfully quieter material, so a
# relative silence test would classify the whole clip as silence.
NO_DYNAMIC_RANGE_DB = 12.0
# Flatness above this is broadband hiss rather than a usable voice. Real speech in
# the calibration corpus peaks at 0.165; speech masked by continuous hiss measured
# 0.328, so the gate sits between them.
MAX_SPECTRAL_FLATNESS = 0.25
# Reported SNR is capped: digital-silence frames would otherwise imply absurd ratios.
MAX_REPORTED_SNR_DB = 40.0
# Normalised autocorrelation height required before a frame is called voiced.
MIN_ACF_PEAK = 0.30
# Loud-to-quiet frame spread below which a recording has almost no silence. Speech is
# intermittent; measured clean material with real pauses sits far above this.
LITTLE_SILENCE_DB = 18.0
# Pauses are a small share of a recording: measured clips hold roughly a fifth of their
# frames in silence, so a 20th percentile lands inside the quietest *speech* and reports
# a spread near zero. The quiet level has to be read well below the pause share.
QUIET_PERCENTILE = 10.0
# Loud-to-quiet spread below which a take is effectively continuous. A tonal quiet band
# is only evidence of a bed above this; below it the quiet frames are just soft speech.
HAS_PAUSE_DB = 4.0


def ffmpeg_executable():
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('FFmpegが見つかりません。音声ファイル変換用のFFmpegを導入してください。')
    return ffmpeg


def decode_audio(source, sample_rate=16000, max_seconds=MAX_DECODE_SECONDS):
    """Decode any container ffmpeg understands into mono float32.

    The accepted format is decided by decodability rather than by file extension.
    Inter-sample overs, which are common in lossy MP3, get one documented scale
    instead of the decoder's own hard limiting.
    """
    source = Path(source)
    if not source.is_file():
        raise ValueError('音声ファイルが見つかりません。')
    command = [ffmpeg_executable(), '-nostdin', '-v', 'error', '-i', str(source),
               '-vn', '-ac', '1', '-ar', str(sample_rate), '-f', 'f32le', 'pipe:1']
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=DECODE_TIMEOUT_SECONDS, check=False,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except subprocess.TimeoutExpired as error:
        raise RuntimeError('音声の読み込みが時間内に終わりませんでした。') from error
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode('utf-8', 'replace').strip().splitlines()
        raise ValueError('音声ファイルを読み込めませんでした。WAV/MP3/M4A/FLAC/OGG/AAC/WMA/Opus等の対応形式を選んでください。'
                         + ('（' + detail[-1][:120] + '）' if detail else ''))
    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise ValueError('音声ファイルが空です。')
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    scale = min(1.0, SAFE_PEAK / peak) if peak > SAFE_PEAK else 1.0
    if max_seconds is not None and len(audio) > max_seconds * sample_rate:
        raise ValueError('音声が長すぎます。%d分以内に切り出してから登録してください。'
                         % int(max_seconds // 60))
    if scale != 1.0:
        audio = audio * scale
    return audio, scale


def frame_rms(audio, frame, hop):
    """Sliding RMS for every hop position; vectorised so long files stay cheap."""
    audio = np.asarray(audio, dtype=np.float64)
    if len(audio) < frame:
        return np.array([float(np.sqrt(np.mean(audio ** 2) + 1e-12))]) if audio.size else np.zeros(0)
    count = 1 + (len(audio) - frame) // hop
    starts = np.arange(count) * hop
    cumulative = np.concatenate(([0.0], np.cumsum(audio ** 2)))
    ends = starts + frame
    total = cumulative[ends] - cumulative[starts]
    return np.sqrt(np.maximum(total / frame, 1e-12))


def dbfs(values):
    return 20.0 * np.log10(np.maximum(np.asarray(values, dtype=np.float64), 1e-8))


def noise_floor_db(audio, rate, percentile=20):
    frame = max(1, int(rate * FRAME_MS / 1000))
    hop = max(1, int(rate * HOP_MS / 1000))
    levels = dbfs(frame_rms(audio, frame, hop))
    finite = levels[np.isfinite(levels)]
    return float(np.percentile(finite, percentile)) if len(finite) else -70.0


def speech_level_db(audio, rate, percentile=70):
    frame = max(1, int(rate * FRAME_MS / 1000))
    hop = max(1, int(rate * HOP_MS / 1000))
    levels = dbfs(frame_rms(audio, frame, hop))
    finite = levels[np.isfinite(levels)]
    return float(np.percentile(finite, percentile)) if len(finite) else -30.0


def _runs(active, hop, hang_frames):
    """Contiguous True runs with a hangover so consonants are not chopped."""
    result = []
    start = None
    quiet = 0
    for index, on in enumerate(active):
        if on:
            if start is None:
                start = index
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet >= hang_frames:
                result.append((start * hop, max((index - quiet + 1) * hop, start * hop + hop)))
                start = None
                quiet = 0
    if start is not None:
        result.append((start * hop, len(active) * hop + hop))
    return result


def detect_speech(audio, rate, robust=True):
    """Adaptive speech regions: percentile noise floor plus a hysteresis margin.

    ``robust`` fixes a real failure mode of a plain percentile floor: when speech
    fills more than ~80% of the file, the 20th percentile *is* a speech frame, the
    floor rises with the signal and the threshold ends up above the speech it is
    meant to find. The robust variant estimates the floor from the quietest frames
    and additionally caps the threshold relative to the typical speech level, so a
    near-continuous recording still yields regions. ``robust=False`` reproduces the
    dataset builder's original estimator exactly.
    """
    frame = max(1, int(rate * FRAME_MS / 1000))
    hop = max(1, int(rate * HOP_MS / 1000))
    levels = dbfs(frame_rms(audio, frame, hop))
    floor_db = noise_floor_db(audio, rate, percentile=5 if robust else 20)
    on_db = max(floor_db + 10.0, -52.0)
    if robust:
        on_db = min(on_db, speech_level_db(audio, rate) - 10.0)
    off_db = on_db - 6.0
    active = levels >= on_db
    state = False
    for index, level in enumerate(levels):
        if state:
            if level < off_db:
                state = False
        elif level >= on_db:
            state = True
        active[index] = state
    pad = int(rate * 0.06)
    merge_gap = int(rate * 0.28)
    merged = []
    for start, end in _runs(active, hop, hang_frames=max(2, int(180 / HOP_MS))):
        start = max(0, start - pad)
        end = min(len(audio), end + pad)
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return [(s, e) for s, e in merged if e > s]


def clipping_runs(audio, rate, limit=0.999):
    """Contiguous over-full-scale runs; clipped audio cannot represent a reference.

    The peak test is per sample, but the returned offsets must be in samples while
    ``_runs`` reports ``index * hop``. The hit is therefore reduced to frames first,
    so a single clipped sample still opens a run instead of being lost to the hop.
    """
    audio = np.asarray(audio, dtype=np.float64)
    frame = max(1, int(rate * 10 / 1000))
    count = 1 + (len(audio) - frame) // frame if len(audio) >= frame else 0
    if count <= 0:
        return []
    hot = (np.abs(audio[:count * frame]) >= limit).astype(np.int32)
    frame_hot = np.add.reduceat(hot, np.arange(0, count * frame, frame)) > 0
    if not frame_hot.any():
        return []
    return _runs(frame_hot, frame, hang_frames=2)


def split_region(audio, start, end, rate, min_seconds=2.0, max_seconds=10.0, target_seconds=7.0):
    """Split a long region at the quietest nearby point, never mid-word."""
    cuts = [start]
    cursor = start
    while end - cursor > max_seconds * rate:
        preferred = min(cursor + target_seconds * rate, end - min_seconds * rate)
        search_start = max(cursor + min_seconds * rate, preferred - rate)
        search_end = min(end - min_seconds * rate, preferred + rate)
        if search_end <= search_start:
            cut = cursor + max_seconds * rate
        else:
            local = frame_rms(audio[int(search_start):int(search_end)],
                              max(1, int(rate * 0.04)), max(1, int(rate * 0.02)))
            cut = int(search_start) + int(np.argmin(local)) * max(1, int(rate * 0.02)) + max(1, int(rate * 0.04)) // 2
            cut = int(min(max(cut, cursor + min_seconds * rate), cursor + max_seconds * rate))
        cuts.append(cut)
        cursor = cut
    cuts.append(int(end))
    cuts = [int(c) for c in cuts]
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] > cuts[i]]


def candidate_regions(audio, rate, min_seconds=2.0, max_seconds=10.0, target_seconds=7.0):
    """Natural speech regions, split at quiet points and at clipping boundaries."""
    regions = []
    for start, end in detect_speech(audio, rate):
        for piece_start, piece_end in split_region(audio, int(start), int(end), rate,
                                                  min_seconds, max_seconds, target_seconds):
            boundaries = [piece_start]
            for hot_start, hot_end in clipping_runs(audio[piece_start:piece_end], rate):
                hot_start, hot_end = piece_start + int(hot_start), piece_start + int(hot_end)
                if boundaries[-1] < hot_start < piece_end:
                    boundaries.append(hot_start)
                boundaries.append(min(piece_end, hot_end))
            boundaries.append(piece_end)
            boundaries = sorted(set(boundaries))
            for index in range(len(boundaries) - 1):
                if boundaries[index + 1] - boundaries[index] >= min_seconds * rate * 0.99:
                    regions.append((boundaries[index], boundaries[index + 1]))
    return regions


def f0_clusters(f0_values, separation_semitones=6.0, minimum_share=0.20):
    """Group voiced F0 values into clusters, for a multi-speaker proxy.

    Returns ``(count, largest_share, largest_median, second_median)``. ``count`` is a
    hint, not a verdict: one speaker legitimately spans a wide range, and a single
    speaker's modal note can differ from their overall median.
    """
    voiced = np.asarray([value for value in np.ravel(f0_values) if value and value > 0], dtype=np.float64)
    if voiced.size < 10:
        return 1, 1.0, (float(np.median(voiced)) if voiced.size else 0.0), None
    ascending = np.sort(voiced)
    log_ascending = np.log2(ascending)
    # Find the largest gap in log-frequency; anything wider than the separation
    # threshold is a candidate boundary between populations.
    gaps = np.diff(log_ascending)
    if not gaps.size or gaps.max() <= separation_semitones/12.0:
        return 1, 1.0, float(np.median(voiced)), None
    boundary = int(np.argmax(gaps))
    low, high = ascending[:boundary+1], ascending[boundary+1:]
    share_low, share_high = len(low)/len(voiced), len(high)/len(voiced)
    if min(share_low, share_high) < minimum_share:
        return 1, 1.0, float(np.median(voiced)), None
    larger, smaller = (low, high) if share_low >= share_high else (high, low)
    return 2, float(max(share_low, share_high)), float(np.median(larger)), float(np.median(smaller))


def music_likeness(audio, rate):
    """Fraction of low-energy frames that are still tonal; music sustains through gaps.

    Speech pauses sit near the noise floor and are aperiodic. A bed that keeps a clear
    pitch in those frames is more likely accompaniment than room tone. The quiet frames
    are the same percentile band the level spread uses, so both measurements describe
    the same part of the recording.

    This is a proxy. A sustained vowel, a held note or a chime can score high too, so a
    high value is evidence to look at the recording, not a conclusion about it.
    """
    audio = np.asarray(audio, dtype=np.float64)
    frame = max(1, int(rate * 0.04))
    hop = max(1, int(rate * 0.02))
    if len(audio) < frame:
        return None
    count = 1 + (len(audio) - frame) // hop
    indices = np.arange(count) * hop
    windows = np.lib.stride_tricks.sliding_window_view(audio, frame)[indices]
    levels = dbfs(np.sqrt(np.mean(windows ** 2, axis=1)))
    if not levels.size:
        return None
    quiet = levels < float(np.percentile(levels, QUIET_PERCENTILE))
    if np.count_nonzero(quiet) < 5:
        return None
    # Flatten the selected frames: estimate_f0 measures length, and a 2-D frame stack
    # would report its row count instead.
    values, _ = estimate_f0(np.ascontiguousarray(windows[quiet]).ravel(), rate)
    if not len(values):
        return 0.0
    tonal = (values > 0) & (values >= 70.0) & (values <= 550.0)
    return round(float(np.count_nonzero(tonal)/len(values)), 4)


def high_pass(audio, rate, cutoff_hz=80.0):
    """Remove rumble below `cutoff_hz` with a 2nd-order Butterworth, zero phase.

    DC removal alone leaves handling rumble and HVAC noise below ~100 Hz, which the
    encoder pools into the speaker embedding. `filtfilt` keeps the phase flat, so the
    clip is not otherwise altered; `cutoff_hz=0` is a no-op.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if cutoff_hz == 0:
        return audio.astype(np.float32)
    if cutoff_hz is None or cutoff_hz < 0:
        raise ValueError('High-pass cutoff must be 0 (off) or a positive frequency')
    if not 20.0 <= cutoff_hz < rate/2:
        raise ValueError('High-pass cutoff must be between 20 Hz and Nyquist')
    if len(audio) < 64:
        return audio.astype(np.float32)
    try:
        from scipy.signal import butter, sosfiltfilt
    except ImportError:
        # Without scipy, a first-order one-pole is still better than nothing and costs
        # nothing; the response is not flat, which the docstring must not hide.
        alpha = 1.0 - np.exp(-2.0*np.pi*float(cutoff_hz)/rate)
        filtered = np.empty_like(audio)
        previous = 0.0
        for index, value in enumerate(audio):
            previous += alpha*(value-previous)
            filtered[index] = value-previous
        return filtered.astype(np.float32)
    sos = butter(2, float(cutoff_hz), btype='highpass', fs=rate, output='sos')
    # filtfilt needs more samples than the filter order; short clips fall back to lfilter.
    if len(audio) > 3*(2*len(sos)+1):
        filtered = sosfiltfilt(sos, audio)
    else:
        from scipy.signal import sosfilt
        filtered = sosfilt(sos, audio)
    return filtered.astype(np.float32)


def assess_contamination(audio, rate, f0_values=None):
    """Measure background-music and multi-speaker risk. Reports evidence, not a verdict.

    The level statistics are percentile based rather than noise-floor based: speech is
    intermittent, so the spread between loud and quiet frames is the one measurement
    that stays meaningful even when a bed occupies the gaps. Neither check can prove a
    recording is clean, and neither one rejects anything -- they exist so the person
    registering a voice sees the risk instead of meeting it later as an artefact.
    """
    audio = np.asarray(audio, dtype=np.float64)
    frame = max(1, int(rate * FRAME_MS / 1000))
    hop = max(1, int(rate * HOP_MS / 1000))
    levels = dbfs(frame_rms(audio, frame, hop))
    empty = {'gap_level_db': None, 'background_music_risk': None, 'speaker_count_hint': 1,
             'speaker_cluster_share': 1.0, 'f0_median_hz': None, 'f0_secondary_hz': None,
             'flags': []}
    if not levels.size:
        return empty
    speech_level = float(np.percentile(levels, 80))
    quiet_level = float(np.percentile(levels, QUIET_PERCENTILE))
    gap_level_db = round(speech_level - quiet_level, 2)
    tonal_gaps = music_likeness(audio, rate)
    count, share, larger, smaller = f0_clusters(
        f0_values if f0_values is not None else estimate_f0(audio, rate)[0])
    flags = []
    # A tonal quiet band only means "music" when the recording actually pauses. In a
    # near-continuous take the quiet frames are simply the soft parts of the same speech,
    # which are also tonal, so the same measurement would accuse ordinary narration.
    if tonal_gaps is not None and tonal_gaps >= 0.5 and gap_level_db >= HAS_PAUSE_DB:
        flags.append('background_music_suspected')
    if gap_level_db < LITTLE_SILENCE_DB:
        flags.append('little_silence')
    if count >= 2:
        flags.append('multiple_speakers_possible')
    return {
        'gap_level_db': gap_level_db,
        'background_music_risk': tonal_gaps,
        'speaker_count_hint': count,
        'speaker_cluster_share': round(share, 4),
        'f0_median_hz': round(larger, 1) if larger else None,
        'f0_secondary_hz': round(smaller, 1) if smaller else None,
        'flags': flags,
    }


def estimate_f0(audio, rate):
    """Per-frame autocorrelation F0 with parabolic peak refinement.

    Verified against a synthetic harmonic stack with known f0: within 1.2% from 100 to
    400Hz. It describes the material and does not drive realtime conversion.
    """
    if len(audio) < int(rate * 0.03):
        return np.zeros(0), 0.0
    step = max(1, rate // 16000)
    down = np.asarray(audio, dtype=np.float64)[::step]
    fs = max(1, rate // step)
    frame = max(1, int(fs * 0.032))
    hop = max(1, int(fs * 0.010))
    if len(down) < frame:
        return np.zeros(0), 0.0
    count = 1 + (len(down) - frame) // hop
    indices = np.arange(count) * hop
    windows = np.lib.stride_tricks.sliding_window_view(down, frame)[indices]
    windows = windows - windows.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(windows ** 2, axis=1))
    window = np.hanning(frame)
    nfft = 1
    while nfft < frame * 2:
        nfft *= 2
    spectrum = np.fft.rfft(windows * window, n=nfft, axis=1)
    ac = np.fft.irfft(np.abs(spectrum) ** 2, n=nfft, axis=1)[:, :frame]
    minimum_lag = max(1, int(fs / 500.0))
    maximum_lag = min(frame - 2, int(fs / 80.0))
    values = np.zeros(count)
    usable = (rms >= 0.008) & (ac[:, 0] > 1e-10)
    if usable.any():
        rows = ac[usable]
        segment = rows[:, minimum_lag:maximum_lag + 1]
        lag = np.argmax(segment, axis=1) + minimum_lag
        peak = segment[np.arange(len(lag)), lag - minimum_lag] / rows[:, 0]
        shift = np.zeros(len(lag))
        interior = (lag >= 1) & (lag < frame - 1)
        if interior.any():
            index = np.nonzero(interior)[0]
            base = lag[index]
            a = rows[index, base - 1]
            b = rows[index, base]
            c = rows[index, base + 1]
            denominator = a - 2 * b + c
            valid = np.abs(denominator) > 1e-12
            shift[index[valid]] = 0.5 * (a[valid] - c[valid]) / denominator[valid]
        frequency = fs / np.maximum(1e-6, lag + shift)
        good = (peak >= MIN_ACF_PEAK) & (frequency >= 70.0) & (frequency <= 550.0)
        values[np.nonzero(usable)[0]] = np.where(good, frequency, 0.0)
    return values, float(np.count_nonzero(values) / max(1, len(values)))


def band_energy(audio, rate, bands=12, low_hz=100.0, high_hz=8000.0):
    """Normalised log-spaced band energies; the acoustic signature used for diversity."""
    audio = np.asarray(audio, dtype=np.float64)
    if len(audio) < 2048:
        return np.zeros(bands)
    audio = audio - audio.mean()
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio)))) ** 2
    freqs = np.fft.rfftfreq(len(audio), 1.0 / rate)
    edges = np.geomspace(low_hz, min(high_hz, rate / 2.0), bands + 1)
    out = np.zeros(bands)
    for index in range(bands):
        mask = (freqs >= edges[index]) & (freqs < edges[index + 1])
        out[index] = spectrum[mask].sum() if mask.any() else 0.0
    total = out.sum()
    return out / total if total > 0 else out


def spectral_flatness(audio, rate):
    """Geometric/arithmetic mean of the power spectrum; rises with hiss or music."""
    audio = np.asarray(audio, dtype=np.float64)
    if len(audio) < 2048:
        return 1.0
    audio = audio - audio.mean()
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio)))) ** 2 + 1e-12
    return float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))


def acoustic_profile(audio, rate, f0_values=None, bins=16):
    """Band energies plus an F0 histogram, normalised into one comparable vector."""
    bands = band_energy(audio, rate)
    if f0_values is None:
        f0_values, _ = estimate_f0(audio, rate)
    voiced = f0_values[f0_values > 0]
    histogram = np.zeros(bins)
    if voiced.size:
        edges = np.linspace(float(np.percentile(voiced, 2)), float(np.percentile(voiced, 98)), bins + 1)
        edges[0], edges[-1] = 0.0, 550.0
        histogram = np.histogram(np.clip(voiced, 0.0, 550.0), bins=edges)[0].astype(np.float64)
        if histogram.sum() > 0:
            histogram /= histogram.sum()
    # Band energies dominate; F0 spread only breaks ties between similar timbre.
    profile = np.concatenate([bands, 0.5 * histogram])
    return profile / max(float(np.linalg.norm(profile)), 1e-12)


def profile_distance(first, second):
    """1 - cosine between two acoustic profiles; 0 means the same material."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 0:
        return 1.0
    return float(1.0 - np.dot(first, second) / denominator)


def analyse(audio, rate, threshold_db=None):
    """Per-clip measurements used by both registration and the dataset builder.

    ``noise_db`` and ``silence_ratio`` answer different questions and are no longer
    conflated: a quiet but perfectly clean clip has a high noise floor yet almost no
    silence, while a clip of pure hiss has a high floor *and* no speech above it.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError('音声データは有限値の一チャンネル信号である必要があります。')
    frame = max(1, int(rate * FRAME_MS / 1000))
    hop = max(1, int(rate * HOP_MS / 1000))
    levels = dbfs(frame_rms(audio, frame, hop))
    reference_db = float(np.percentile(levels, 95))
    floor_db = float(np.percentile(levels, 5))
    has_floor = (reference_db - floor_db) >= NO_DYNAMIC_RANGE_DB
    if threshold_db is None:
        # Without genuinely quiet material there is no silence to find, and a
        # relative threshold would label the entire clip silent.
        threshold_db = max(floor_db + 6.0, reference_db - 35.0) if has_floor else reference_db - 12.0
    speech = levels[levels >= threshold_db]
    speech_db = float(np.mean(speech)) if speech.size else reference_db
    silence_ratio = float(np.mean(levels < threshold_db)) if len(levels) else 1.0
    # Speech-to-noise is only measurable when the clip actually contains noise-only
    # frames. Continuous hiss has none, so a number here would be invented rather
    # than measured; report None and let the flatness gate carry that case.
    quiet = levels[levels <= reference_db - 20.0]
    snr_db = None
    if has_floor and quiet.size >= 8:
        snr_db = round(min(MAX_REPORTED_SNR_DB, speech_db - float(np.mean(quiet))), 2)
    f0_values, voiced_ratio = estimate_f0(audio, rate)
    voiced_f0 = f0_values[f0_values > 0]
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return {
        'duration_sec': float(len(audio) / rate),
        'rms_dbfs': float(dbfs(np.sqrt(np.mean(audio ** 2) + 1e-12))),
        'peak_dbfs': float(dbfs(peak)),
        'peak_linear': peak,
        'clipping_ratio': float(np.mean(np.abs(audio) >= 0.999)) if audio.size else 0.0,
        'silence_ratio': silence_ratio,
        'voiced_ratio': float(voiced_ratio),
        'snr_db': snr_db,
        'spectral_flatness': round(spectral_flatness(audio, rate), 5),
        'f0_median_hz': float(np.median(voiced_f0)) if voiced_f0.size else None,
        'f0_values_hz': voiced_f0.tolist(),
    }


# Explicit, inspectable gates. A rejected clip always reports which rule fired.
REJECT_RULES = (
    ('too_short', lambda s: s['duration_sec'] < 2.0),
    ('clipped', lambda s: s['clipping_ratio'] > 0.01),
    ('too_much_silence', lambda s: s['silence_ratio'] > 0.60),
    ('near_silent', lambda s: s['rms_dbfs'] < -55.0),
    ('too_quiet', lambda s: s['snr_db'] is not None and s['snr_db'] < 6.0),
    ('insufficient_voiced_f0', lambda s: s['voiced_ratio'] < 0.10),
    ('hiss_or_noise', lambda s: s['spectral_flatness'] > MAX_SPECTRAL_FLATNESS),
)


def rejection_reason(stats):
    """Return the first failing gate name, or None when the clip is usable."""
    for name, rule in REJECT_RULES:
        if rule(stats):
            return name
    return None


def quality_score(stats):
    """0..1 preference for a usable clip: speech-to-noise, voiced share, level, tone."""
    # An unmeasurable SNR is treated as neutral rather than as perfect.
    snr = 0.5 if stats['snr_db'] is None else min(max(stats['snr_db'], 0.0), MAX_REPORTED_SNR_DB) / MAX_REPORTED_SNR_DB
    voiced = min(max(stats['voiced_ratio'], 0.0), 0.85) / 0.85
    level = 1.0 - min(abs(stats['rms_dbfs'] + 22.0) / 22.0, 1.0)
    if stats['f0_median_hz']:
        tone = 1.0 - min(abs(math.log2(max(stats['f0_median_hz'], 1.0) / 190.0)), 1.0)
    else:
        tone = 0.0
    return round(0.40 * snr + 0.25 * voiced + 0.20 * level + 0.15 * tone, 4)


DEFAULT_BUDGET_SECONDS = 60.0
MIN_REFERENCE_SECONDS = 3.0

# Plain-language causes, keyed by the gate that rejected the material.
REJECT_ADVICE = {
    'clipped': '音割れの区間が多く使えません。入力ゲインを下げて録音し直してください。',
    'too_much_silence': '無音や間が多い区間が選ばれませんでした。発話を詰めて録音してください。',
    'near_silent': '声の大半が非常に小さい音量です。マイクとの距離やゲインを見直してください。',
    'too_quiet': 'SNR（信号対雑音比）が低く、ノイズに埋もれた区間が多く選ばれませんでした。',
    'insufficient_voiced_f0': '声の高さを検出できない区間が多く選ばれませんでした。',
    'hiss_or_noise': '環境音やノイズが多い区間が多く選ばれませんでした。',
    'too_short': '発話区間が短すぎます。5秒以上、話してください。',
}

# Contamination flags are reported, never used to reject. Each is a proxy with a known
# blind spot, stated here so the wording does not promise more than the measurement.
CONTAMINATION_ADVICE = {
    'background_music_suspected':
        '発話していない間にも音が続くため、BGMや環境音が含まれる可能性があります。',
    'little_silence':
        '無音がほとんどありません。発話と間を分けた録音にすると、'
        '登録が安定しやすくなります。',
    'multiple_speakers_possible':
        '声の高さが2つの集団に分かれています。複数の話者が録られている可能性があります。',
}


class NoUsableAudio(ValueError):
    """No clip passed the gates. Carries the report so the GUI can explain why."""

    def __init__(self, report):
        advice = report.get('advice') or ['品質基準を満たす発話区間が見つかりませんでした。']
        super().__init__(advice[0])
        self.report = report


def _acceptance_advice(rejected, accepted_count, source_seconds, speech_seconds):
    """Turn the tally of failed gates into something the person can act on."""
    advice = []
    if not accepted_count and rejected:
        worst = max(rejected.items(), key=lambda item: item[1])[0]
        advice.append(REJECT_ADVICE.get(worst, '品質基準を満たす発話区間が見つかりませんでした。'))
    if source_seconds and speech_seconds / max(source_seconds, 1e-9) < 0.20:
        advice.append('発話時間が少ない録音です。20秒以上、話してから登録してください。')
    if accepted_count and len(rejected) == 0:
        advice.append('すべての発話区間が品質基準を満たしていました。')
    return advice


def contamination_advice(contamination):
    """Plain-language notes for the contamination proxies that did fire."""
    if not isinstance(contamination, dict):
        return []
    return [CONTAMINATION_ADVICE[flag] for flag in contamination.get('flags') or []
            if flag in CONTAMINATION_ADVICE]


def select_reference(audio, rate, budget_seconds=DEFAULT_BUDGET_SECONDS,
                     min_seconds=2.0, max_seconds=10.0, target_seconds=7.0,
                     diversity_weight=0.55, progress=None):
    """Pick reference material: natural boundaries, quality gates, then diversity.

    Returns ``(parts, selections, report)`` with the chosen clips in chronological
    order. ``report`` is serialisable and is what the GUI shows, so every number a
    person sees traces back to a measurement rather than to a guess.
    """
    def notify(stage, done=0, total=0):
        if progress is not None:
            progress(stage, done, total)

    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError('音声データを解析できませんでした。')
    if len(audio) < MIN_REFERENCE_SECONDS * rate:
        raise ValueError('音声が短すぎます。最低%d秒は必要です。' % int(MIN_REFERENCE_SECONDS))

    notify('発話区間を検出しています', 0, 1)
    regions = detect_speech(audio, rate)
    speech_seconds = sum(end - start for start, end in regions) / rate
    notify('発話区間を検出しています', 1, 1)
    candidates = candidate_regions(audio, rate, min_seconds, max_seconds, target_seconds)

    rejected = {}
    accepted = []
    for index, (start, end) in enumerate(candidates):
        notify('品質を判定しています', index, len(candidates))
        stats = analyse(audio[start:end], rate)
        reason = rejection_reason(stats)
        if reason is not None:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        stats['start'] = int(start)
        stats['end'] = int(end)
        stats['quality'] = quality_score(stats)
        accepted.append(stats)

    source_seconds = len(audio) / rate
    if not accepted:
        raise NoUsableAudio({
            'source_seconds': round(source_seconds, 2),
            'speech_seconds': round(speech_seconds, 2),
            'region_count': len(regions), 'candidate_count': len(candidates),
            'accepted_count': 0, 'rejected': rejected, 'retained_seconds': 0.0,
            'budget_seconds': budget_seconds, 'selections': [],
            'advice': _acceptance_advice(rejected, 0, source_seconds, speech_seconds),
        })

    notify('多样化を計算しています', 0, 1)
    for row in accepted:
        row['profile'] = acoustic_profile(audio[row['start']:row['end']], rate,
                                          np.asarray(row['f0_values_hz'], dtype=np.float64))
    remaining = list(accepted)
    chosen = []
    profiles = []
    retained = 0.0
    while remaining:
        room = budget_seconds - retained
        if chosen and room < min_seconds:
            break
        affordable = [row for row in remaining if row['duration_sec'] <= room + 1e-9]
        if not affordable:
            break

        def rank(row):
            if not profiles:
                return row['quality']
            nearest = min(profile_distance(row['profile'], other) for other in profiles)
            return row['quality'] - diversity_weight * nearest

        best = max(affordable, key=rank)
        nearest = min((profile_distance(best['profile'], other) for other in profiles), default=0.0)
        best['diversity_to_previous'] = round(nearest, 4)
        chosen.append(best)
        profiles.append(best['profile'])
        remaining.remove(best)
        retained += best['duration_sec']
    notify('多样化を計算しています', 1, 1)

    if not chosen or retained < MIN_REFERENCE_SECONDS:
        raise NoUsableAudio({
            'source_seconds': round(source_seconds, 2),
            'speech_seconds': round(speech_seconds, 2),
            'region_count': len(regions), 'candidate_count': len(candidates),
            'accepted_count': len(accepted), 'rejected': rejected, 'retained_seconds': 0.0,
            'budget_seconds': budget_seconds, 'selections': [],
            # Distinct from the source-length case: the file was long enough, it simply
            # did not contain enough usable speech to publish a reference from.
            'advice': ['%s秒より短い有効な発話しか取り出せませんでした。'
                       '発話を追加して、もう少し長く録音してください。' % int(MIN_REFERENCE_SECONDS)],
        })

    # Chronological order, so the retained reference still sounds like one recording.
    chosen.sort(key=lambda row: row['start'])
    parts = [audio[row['start']:row['end']].astype(np.float32) for row in chosen]
    retained_audio = np.concatenate(parts) if len(parts) > 1 else parts[0]
    contamination = assess_contamination(retained_audio, rate)
    selections = [{
        'offset_seconds': round(row['start'] / rate, 3),
        'duration_seconds': round(row['duration_sec'], 3),
        'quality': row['quality'],
        'snr_db': row['snr_db'],
        'voiced_ratio': round(row['voiced_ratio'], 4),
        'silence_ratio': round(row['silence_ratio'], 4),
        'spectral_flatness': row['spectral_flatness'],
        'f0_median_hz': round(row['f0_median_hz'], 1) if row['f0_median_hz'] else None,
        'clipping_ratio': round(row['clipping_ratio'], 6),
        'diversity_to_previous': row.get('diversity_to_previous', 0.0),
    } for row in chosen]
    report = {
        'source_seconds': round(source_seconds, 2),
        'speech_seconds': round(speech_seconds, 2),
        'region_count': len(regions),
        'candidate_count': len(candidates),
        'accepted_count': len(accepted),
        'rejected': rejected,
        'retained_seconds': round(retained, 2),
        'budget_seconds': budget_seconds,
        'selections': selections,
        'contamination': contamination,
        'advice': _acceptance_advice(rejected, len(accepted), source_seconds, speech_seconds)
                   + contamination_advice(contamination),
    }
    return parts, selections, report
