"""Build a clean, mono WAV voice-clone dataset from one recording.

This tool is intentionally offline and independent from the realtime audio
engine.  It decodes an input file through ffmpeg, finds speech-like regions,
splits them into bounded clips, applies gentle loudness matching, extracts
lightweight quality/F0 statistics, and writes a reproducible JSONL manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


DEFAULT_SR = 48_000
FRAME_MS = 20
HOP_MS = 10
MIN_CLIP_SEC = 2.0
MAX_CLIP_SEC = 10.0
TARGET_RMS = 0.10


@dataclass
class Clip:
    index: int
    start_sample: int
    end_sample: int
    reason: str = ""

    @property
    def duration_sec(self) -> float:
        return (self.end_sample - self.start_sample) / DEFAULT_SR


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_audio(path: Path, sample_rate: int = DEFAULT_SR) -> np.ndarray:
    """Decode with ffmpeg to mono float32 without touching the source file."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        local = Path(__file__).resolve().parents[1] / "tools" / "ffmpeg.exe"
        if local.exists():
            ffmpeg = str(local)
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to decode the input audio")
    cmd = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg decode failed: {error}")
    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise RuntimeError("decoded audio is empty")
    # Do not hard-clip the decoded float stream here. Lossy MP3 decoding can
    # produce inter-sample peaks above 1.0; preserving them lets the caller
    # measure the condition and apply one gentle, documented scale before WAV
    # export instead of introducing distortion at decode time.
    return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)


def frame_rms(audio: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(audio) < frame:
        padded = np.pad(audio, (0, frame - len(audio)))
    else:
        padded = audio
    count = 1 + max(0, (len(padded) - frame) // hop)
    values = np.empty(count, dtype=np.float64)
    window = np.hanning(frame).astype(np.float32)
    for i in range(count):
        chunk = padded[i * hop : i * hop + frame]
        if len(chunk) < frame:
            chunk = np.pad(chunk, (0, frame - len(chunk)))
        values[i] = math.sqrt(float(np.mean((chunk * window) ** 2)) + 1e-12)
    return values


def dbfs(values: np.ndarray | float) -> np.ndarray | float:
    return 20.0 * np.log10(np.maximum(np.asarray(values), 1e-8))


def _hysteresis_runs(active: np.ndarray, hop: int, hang_frames: int) -> list[tuple[int, int]]:
    """Build runs with a small hangover so consonants are not chopped."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    quiet = 0
    for i, on in enumerate(active):
        if on:
            if start is None:
                start = i
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet >= hang_frames:
                end = i - quiet + 1
                runs.append((start * hop, max(end * hop, start * hop + hop)))
                start = None
                quiet = 0
    if start is not None:
        runs.append((start * hop, len(active) * hop + hop))
    return runs


def detect_speech_regions(audio: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
    frame = int(sample_rate * FRAME_MS / 1000)
    hop = int(sample_rate * HOP_MS / 1000)
    rms = frame_rms(audio, frame, hop)
    levels = dbfs(rms)
    finite = levels[np.isfinite(levels)]
    noise_floor = float(np.percentile(finite, 20)) if len(finite) else -70.0
    # A conservative adaptive threshold. The floor keeps quiet room tone from
    # becoming a training clip while the +10 dB margin retains soft speech.
    on_db = max(noise_floor + 10.0, -52.0)
    off_db = on_db - 6.0
    active = levels >= on_db
    # Add a very small hysteresis pass: a frame between off/on keeps the prior
    # state. This is more reliable than a hard threshold for breathy speech.
    state = False
    for i, level in enumerate(levels):
        if state:
            if level < off_db:
                state = False
        elif level >= on_db:
            state = True
        active[i] = state
    runs = _hysteresis_runs(active, hop, hang_frames=max(2, int(180 / HOP_MS)))
    merge_gap = int(sample_rate * 0.28)
    pad = int(sample_rate * 0.06)
    merged: list[tuple[int, int]] = []
    for start, end in runs:
        start = max(0, start - pad)
        end = min(len(audio), end + pad)
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def split_region_at_quiet(audio: np.ndarray, start: int, end: int, sample_rate: int) -> list[tuple[int, int]]:
    max_samples = int(MAX_CLIP_SEC * sample_rate)
    min_samples = int(MIN_CLIP_SEC * sample_rate)
    if end - start <= max_samples:
        return [(start, end)]
    parts: list[tuple[int, int]] = []
    cursor = start
    frame = int(sample_rate * 0.04)
    hop = int(sample_rate * 0.02)
    target = int(sample_rate * 7.0)
    while end - cursor > max_samples:
        preferred = min(cursor + target, end - min_samples)
        search_start = max(cursor + min_samples, preferred - int(sample_rate * 1.0))
        search_end = min(end - min_samples, preferred + int(sample_rate * 1.0))
        if search_end <= search_start:
            cut = cursor + max_samples
        else:
            local = audio[search_start:search_end]
            energies = frame_rms(local, frame, hop)
            cut_i = int(np.argmin(energies))
            cut = search_start + cut_i * hop + frame // 2
            cut = min(max(cut, cursor + min_samples), cursor + max_samples)
        parts.append((cursor, cut))
        cursor = cut
    if end - cursor > 0:
        parts.append((cursor, end))
    return parts


def make_clips(audio: np.ndarray, regions: Sequence[tuple[int, int]], sample_rate: int) -> list[Clip]:
    raw: list[tuple[int, int]] = []
    for start, end in regions:
        raw.extend(split_region_at_quiet(audio, start, end, sample_rate))
    clips: list[Clip] = []
    for i, (start, end) in enumerate(raw, 1):
        clips.append(Clip(i, int(start), int(end)))
    return clips


def estimate_f0(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
    """Return per-frame autocorrelation F0 values and voiced ratio.

    This is deliberately a conservative offline quality statistic, not the
    realtime estimator used by the app. It is robust enough to detect whether
    a recording contains ordinary voiced speech without changing its audio.
    """
    if len(audio) < int(sample_rate * 0.03):
        return np.empty(0), 0.0
    # 16 kHz is sufficient for an 80--500 Hz speaking voice range.
    ds = audio[:: max(1, sample_rate // 16_000)]
    fs = sample_rate // max(1, sample_rate // 16_000)
    frame = int(fs * 0.032)
    hop = int(fs * 0.010)
    nfft = 1
    while nfft < frame * 2:
        nfft *= 2
    values: list[float] = []
    voiced = 0
    window = np.hanning(frame)
    min_lag = max(1, int(fs / 500.0))
    max_lag = min(frame - 2, int(fs / 80.0))
    for pos in range(0, max(1, len(ds) - frame + 1), hop):
        x = ds[pos : pos + frame].astype(np.float64, copy=False)
        if len(x) < frame:
            x = np.pad(x, (0, frame - len(x)))
        rms = math.sqrt(float(np.mean(x * x)) + 1e-12)
        if rms < 0.008:
            values.append(0.0)
            continue
        x = (x - np.mean(x)) * window
        spectrum = np.fft.rfft(x, nfft)
        ac = np.fft.irfft(np.abs(spectrum) ** 2, nfft)[:frame]
        if ac[0] <= 1e-10:
            values.append(0.0)
            continue
        segment = ac[min_lag : max_lag + 1]
        lag_i = int(np.argmax(segment)) + min_lag
        peak = float(ac[lag_i] / ac[0])
        if peak < 0.30:
            values.append(0.0)
            continue
        # Parabolic interpolation provides a less quantized median.
        shift = 0.0
        if 1 <= lag_i < len(ac) - 1:
            a, b, c = ac[lag_i - 1], ac[lag_i], ac[lag_i + 1]
            den = a - 2 * b + c
            if abs(den) > 1e-12:
                shift = 0.5 * (a - c) / den
        f0 = fs / max(1e-6, lag_i + shift)
        if 70.0 <= f0 <= 550.0:
            values.append(float(f0))
            voiced += 1
        else:
            values.append(0.0)
    result = np.asarray(values, dtype=np.float64)
    return result, float(voiced / max(1, len(result)))


def clip_stats(audio: np.ndarray, sample_rate: int, threshold_db: float) -> dict:
    finite = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    rms = float(np.sqrt(np.mean(finite * finite) + 1e-12))
    peak = float(np.max(np.abs(finite))) if len(finite) else 0.0
    frame = int(sample_rate * FRAME_MS / 1000)
    hop = int(sample_rate * HOP_MS / 1000)
    frames = frame_rms(finite, frame, hop)
    levels = dbfs(frames)
    silence_ratio = float(np.mean(levels < threshold_db)) if len(levels) else 1.0
    f0, voiced_ratio = estimate_f0(finite, sample_rate)
    voiced_f0 = f0[f0 > 0]
    return {
        "duration_sec": float(len(finite) / sample_rate),
        "rms_dbfs": float(dbfs(rms)),
        "peak_dbfs": float(dbfs(peak)),
        "peak_linear": peak,
        "clipping_samples": int(np.count_nonzero(np.abs(finite) >= 0.999)),
        "silence_ratio": silence_ratio,
        "voiced_ratio": voiced_ratio,
        "f0_median_hz": float(np.median(voiced_f0)) if len(voiced_f0) else None,
        "f0_mean_hz": float(np.mean(voiced_f0)) if len(voiced_f0) else None,
        "f0_min_hz": float(np.min(voiced_f0)) if len(voiced_f0) else None,
        "f0_max_hz": float(np.max(voiced_f0)) if len(voiced_f0) else None,
        "f0_values_hz": voiced_f0.tolist(),
    }


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.rint(audio * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int, int]:
    with wave.open(str(path), "rb") as wf:
        channels, width, rate, frames = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
        raw = wf.readframes(frames)
    if width != 2:
        raise ValueError(f"unsupported WAV width: {width}")
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate, channels


def validate_dataset(output: Path, metadata: list[dict], sample_rate: int) -> dict:
    accepted = [m for m in metadata if m["status"] == "accepted"]
    rejected = [m for m in metadata if m["status"] == "rejected"]
    checks = {
        "accepted_files_exist": True,
        "accepted_wav_readable": True,
        "mono_16bit_48khz": True,
        "duration_bounds": True,
        "finite_samples": True,
        "no_clipping": True,
        "no_silent_clip": True,
    }
    failures: list[dict] = []
    for item in accepted:
        path = output / item["wav_path"]
        try:
            if not path.exists():
                raise ValueError("missing file")
            audio, rate, channels = read_wav(path)
            if rate != sample_rate or channels != 1:
                checks["mono_16bit_48khz"] = False
                raise ValueError(f"format {rate} Hz/{channels} ch")
            if len(audio) / rate < MIN_CLIP_SEC or len(audio) / rate > MAX_CLIP_SEC + 0.01:
                checks["duration_bounds"] = False
                raise ValueError("duration outside bounds")
            if not np.isfinite(audio).all():
                checks["finite_samples"] = False
                raise ValueError("non-finite samples")
            if int(np.count_nonzero(np.abs(audio) >= 0.999)):
                checks["no_clipping"] = False
                raise ValueError("clipping")
            if float(np.sqrt(np.mean(audio * audio) + 1e-12)) < 0.005:
                checks["no_silent_clip"] = False
                raise ValueError("near silent")
        except Exception as exc:  # validation report should explain every failure
            checks["accepted_wav_readable"] = False
            failures.append({"id": item["id"], "error": str(exc)})
    passed = all(checks.values()) and bool(accepted)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": bool(passed),
        "checks": checks,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "failure_count": len(failures),
        "failures": failures,
    }


def percentile(values: Iterable[float], p: float) -> float | None:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, p)) if len(arr) else None


def decision(accepted: list[dict], total_speech_sec: float, validation_passed: bool) -> tuple[str, list[str]]:
    reasons: list[str] = []
    effective_min = total_speech_sec / 60.0
    if not validation_passed:
        reasons.append("Validatorに失敗したクリップがあります")
    if effective_min < 5.0:
        reasons.append(f"有効発話が{effective_min:.1f}分で、POC目安の5分未満です")
    if len(accepted) < 30:
        reasons.append(f"採用クリップが{len(accepted)}本で、発話多様性の目安30本未満です")
    if any((m.get("silence_ratio") or 1.0) > 0.40 for m in accepted):
        reasons.append("無音率が高いクリップがあります")
    if any((m.get("voiced_ratio") or 0.0) < 0.15 for m in accepted):
        reasons.append("有声音区間が短いクリップがあります")
    return ("TRAINING GO" if not reasons else "MORE RECORDING RECOMMENDED"), reasons


def build_report(
    output: Path,
    source: Path,
    source_duration: float,
    metadata: list[dict],
    validation: dict,
    decision_text: str,
    decision_reasons: list[str],
    source_hash: str,
    sample_rate: int,
    source_peak: float,
    source_overs: int,
    source_scale: float,
) -> None:
    accepted = [m for m in metadata if m["status"] == "accepted"]
    rejected = [m for m in metadata if m["status"] == "rejected"]
    durations = [m["duration_sec"] for m in accepted]
    all_f0 = [f for m in accepted for f in m.get("f0_values_hz", [])]
    rms = [m["rms_dbfs"] for m in accepted]
    peaks = [m["peak_dbfs"] for m in accepted]
    silence = [m["silence_ratio"] for m in accepted]
    clips = sum(m["clipping_samples"] for m in metadata)
    lines = [
        "# Anime Voice Clone Dataset v1 report",
        "",
        f"- Generated: {datetime.now().astimezone().isoformat()}",
        f"- Source: `{source}`",
        f"- Source SHA-256: `{source_hash}`",
        f"- Decode format: mono, {sample_rate} Hz, 16-bit PCM WAV",
        f"- Decoded source peak: {source_peak:.4f} ({dbfs(source_peak):.2f} dBFS); oversamples >= 1.0: {source_overs}",
        f"- Applied safe source scale before segmentation: {source_scale:.6f}",
        "",
        "## 判定",
        "",
        f"**{decision_text}**",
        "",
        *(f"- {reason}" for reason in (decision_reasons or ["有効クリップ、品質検査、時間条件を満たしています"])),
        "",
        "## 収録量",
        "",
        f"- 元音声時間: {source_duration:.2f} 秒 ({source_duration/60:.2f} 分)",
        f"- 有効発声音声時間: {sum(durations):.2f} 秒 ({sum(durations)/60:.2f} 分)",
        f"- 採用クリップ数: {len(accepted)}",
        f"- 除外クリップ数: {len(rejected)}",
        f"- 平均クリップ長: {np.mean(durations):.2f} 秒" if durations else "- 平均クリップ長: n/a",
        f"- 最短 / 最長: {min(durations):.2f} / {max(durations):.2f} 秒" if durations else "- 最短 / 最長: n/a",
        "",
        "## 出力音量・無音・クリッピング",
        "",
        f"- 正規化後の平均RMS: {np.mean(rms):.2f} dBFS" if rms else "- 平均RMS: n/a",
        f"- RMS p5 / p50 / p95: {percentile(rms,5)} / {percentile(rms,50)} / {percentile(rms,95)} dBFS" if rms else "- RMS分布: n/a",
        f"- Peak p50 / 最大: {percentile(peaks,50)} / {max(peaks):.2f} dBFS" if peaks else "- Peak: n/a",
        f"- 平均無音率: {np.mean(silence)*100:.1f}%" if silence else "- 平均無音率: n/a",
        f"- 出力クリッピングサンプル総数: {clips} (Validatorでは0を確認)",
        "",
        "## F0概要",
        "",
        f"- 有効F0フレーム数: {len(all_f0)}",
        f"- F0中央値: {percentile(all_f0,50):.2f} Hz" if all_f0 else "- F0中央値: n/a",
        f"- F0 p5 / p25 / p50 / p75 / p95: {percentile(all_f0,5)} / {percentile(all_f0,25)} / {percentile(all_f0,50)} / {percentile(all_f0,75)} / {percentile(all_f0,95)} Hz" if all_f0 else "- F0分布: n/a",
        "",
        "## Validator",
        "",
        f"- 結果: {'PASS' if validation['passed'] else 'FAIL'}",
        f"- チェック: `{json.dumps(validation['checks'], ensure_ascii=False)}`",
        "",
        "## 除外理由",
        "",
    ]
    reasons: dict[str, int] = {}
    for item in rejected:
        reasons[item.get("reject_reason", "unknown")] = reasons.get(item.get("reject_reason", "unknown"), 0) + 1
    lines.extend(f"- {key}: {value}" for key, value in sorted(reasons.items()))
    lines += [
        "",
        "## 制約と次の判断",
        "",
        "このPOCは単一録音からの切り出しです。文字起こし・話者別の発声網羅性・感情ラベルは付与していません。学習を開始できる量でも、実運用向けには別セッションの追加録音を混ぜると過学習を抑えられます。",
        "",
        "元のMP3は変更・削除していません。Realtimeアプリ、Prosody、Text-Aware処理には変更を加えていません。",
    ]
    (output / "dataset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SR)
    args = parser.parse_args(argv)
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"input not found: {source}")
    output.mkdir(parents=True, exist_ok=True)
    for folder in ("wav", "rejected"):
        (output / folder).mkdir(exist_ok=True)
    audio = decode_audio(source, args.sample_rate)
    source_peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    source_overs = int(np.count_nonzero(np.abs(audio) >= 1.0))
    # MP3 inter-sample overs are common and are not a reason to discard an
    # otherwise clean recording. Scale once, before VAD/F0, to keep every WAV
    # sample below full scale without per-frame hard clipping.
    source_scale = min(1.0, 0.95 / source_peak) if source_peak > 0.95 else 1.0
    audio = audio * source_scale
    source_duration = len(audio) / args.sample_rate
    regions = detect_speech_regions(audio, args.sample_rate)
    clips = make_clips(audio, regions, args.sample_rate)
    # Use the detected off threshold again for per-clip silence statistics.
    all_rms = frame_rms(audio, int(args.sample_rate * FRAME_MS / 1000), int(args.sample_rate * HOP_MS / 1000))
    floor_db = float(np.percentile(dbfs(all_rms), 20)) if len(all_rms) else -70.0
    off_db = max(floor_db + 4.0, -58.0)
    raw_stats: list[dict] = []
    for clip in clips:
        segment = audio[clip.start_sample : clip.end_sample]
        stats = clip_stats(segment, args.sample_rate, off_db)
        stats["clip"] = clip
        raw_stats.append(stats)
    # A clip must contain enough actual speech and cannot be a pathological
    # decoder artefact. Keep the rejection rules explicit in metadata.
    accepted_raw: list[dict] = []
    rejected_raw: list[dict] = []
    for stats in raw_stats:
        duration = stats["duration_sec"]
        reason = ""
        if duration < MIN_CLIP_SEC:
            reason = "too_short"
        elif duration > MAX_CLIP_SEC + 0.01:
            reason = "too_long"
        elif stats["silence_ratio"] > 0.60:
            reason = "too_much_silence"
        elif stats["rms_dbfs"] < -55.0:
            reason = "near_silent"
        elif stats["voiced_ratio"] < 0.10:
            reason = "insufficient_voiced_f0"
        elif not np.isfinite(stats["peak_linear"]):
            reason = "non_finite"
        if reason:
            stats["status"] = "rejected"
            stats["reject_reason"] = reason
            rejected_raw.append(stats)
        else:
            stats["status"] = "accepted"
            accepted_raw.append(stats)
    target_rms = float(np.median([10 ** (s["rms_dbfs"] / 20.0) for s in accepted_raw])) if accepted_raw else TARGET_RMS
    metadata: list[dict] = []
    source_hash = sha256_file(source)
    for stats in raw_stats:
        clip: Clip = stats.pop("clip")
        status = stats["status"]
        segment = audio[clip.start_sample : clip.end_sample]
        gain_db = 0.0
        if status == "accepted":
            raw_rms = 10 ** (stats["rms_dbfs"] / 20.0)
            gain_db = float(np.clip(20 * math.log10(max(1e-8, target_rms / max(raw_rms, 1e-8))), -3.0, 3.0))
            segment = segment * (10 ** (gain_db / 20.0))
            output_peak = float(np.max(np.abs(segment))) if len(segment) else 0.0
            if output_peak > 0.98:
                safe_gain = 0.98 / output_peak
                segment = segment * safe_gain
                gain_db += float(20 * math.log10(safe_gain))
            wav_rel = f"wav/clip_{clip.index:04d}.wav"
            write_wav(output / wav_rel, segment, args.sample_rate)
        else:
            wav_rel = f"rejected/clip_{clip.index:04d}_{stats['reject_reason']}.wav"
            write_wav(output / wav_rel, segment, args.sample_rate)
        output_rms = float(np.sqrt(np.mean(segment * segment) + 1e-12)) if len(segment) else 0.0
        output_peak = float(np.max(np.abs(segment))) if len(segment) else 0.0
        item = {
            "id": f"clip_{clip.index:04d}",
            "status": status,
            "reject_reason": stats.get("reject_reason"),
            "source": str(source),
            "source_sha256": source_hash,
            "source_start_sec": round(clip.start_sample / args.sample_rate, 6),
            "source_end_sec": round(clip.end_sample / args.sample_rate, 6),
            "wav_path": wav_rel,
            "sample_rate": args.sample_rate,
            "channels": 1,
            "duration_sec": stats["duration_sec"],
            "rms_dbfs": float(dbfs(output_rms)),
            "peak_dbfs": float(dbfs(output_peak)),
            "raw_rms_dbfs": stats["rms_dbfs"],
            "raw_peak_dbfs": stats["peak_dbfs"],
            "clipping_samples": stats["clipping_samples"],
            "silence_ratio": stats["silence_ratio"],
            "voiced_ratio": stats["voiced_ratio"],
            "f0_median_hz": stats["f0_median_hz"],
            "f0_mean_hz": stats["f0_mean_hz"],
            "f0_min_hz": stats["f0_min_hz"],
            "f0_max_hz": stats["f0_max_hz"],
            "f0_values_hz": stats["f0_values_hz"],
            "normalization_gain_db": gain_db,
        }
        metadata.append(item)
    with (output / "metadata.jsonl").open("w", encoding="utf-8") as fh:
        for item in metadata:
            fh.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    validation = validate_dataset(output, metadata, args.sample_rate)
    accepted = [m for m in metadata if m["status"] == "accepted"]
    decision_text, reasons = decision(accepted, sum(m["duration_sec"] for m in accepted), validation["passed"])
    accepted_durations = [m["duration_sec"] for m in accepted]
    f0_values = [f for m in accepted for f in m.get("f0_values_hz", [])]
    validation.update(
        {
            "source": str(source),
            "source_sha256": source_hash,
            "source_duration_sec": source_duration,
            "effective_speech_duration_sec": sum(accepted_durations),
            "clip_duration_sec": {
                "mean": float(np.mean(accepted_durations)) if accepted_durations else None,
                "min": float(min(accepted_durations)) if accepted_durations else None,
                "max": float(max(accepted_durations)) if accepted_durations else None,
            },
            "silence_ratio_mean": float(np.mean([m["silence_ratio"] for m in accepted])) if accepted else None,
            "rms_dbfs_mean": float(np.mean([m["rms_dbfs"] for m in accepted])) if accepted else None,
            "peak_dbfs_max": float(max(m["peak_dbfs"] for m in accepted)) if accepted else None,
            "f0_hz": {
                "median": percentile(f0_values, 50),
                "p5": percentile(f0_values, 5),
                "p95": percentile(f0_values, 95),
                "frame_count": len(f0_values),
            },
            "decision": decision_text,
            "decision_reasons": reasons,
            "source_decoded_peak_linear": source_peak,
            "source_decoded_oversamples": source_overs,
            "source_scale": source_scale,
        }
    )
    (output / "validation_report.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    build_report(
        output,
        source,
        source_duration,
        metadata,
        validation,
        decision_text,
        reasons,
        source_hash,
        args.sample_rate,
        source_peak,
        source_overs,
        source_scale,
    )
    summary = {
        "source_duration_sec": source_duration,
        "regions": len(regions),
        "accepted": len(accepted),
        "rejected": len(metadata) - len(accepted),
        "effective_duration_sec": sum(m["duration_sec"] for m in accepted),
        "validation": validation["passed"],
        "decision": decision_text,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
