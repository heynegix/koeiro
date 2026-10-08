"""Build and validate the v0.6 Prosody Teacher Dataset v2.

This is an offline, deterministic batch tool.  It consumes the v0.5 FCPE
feature cache and never modifies the source WAVs or the old dataset.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.alignment import align
from src.dataset.features import relative_f0
from src.dataset.teachers import (
    clean_f0, corrected_anime_target, hybrid_target, relative_clean_f0,
    rule_target, target_sample, teacher_scores,
)


SEED = 600
RULE_CONFIG = dict(range_expansion=1.35, rise_boost=1.20, fall_boost=1.05,
                   onset_lift=0.30, energy_scale=1.15, pitch_clamp=[-1.0, 1.4],
                   energy_clamp=1.5, frame_ms=10, rule_version="v0.6.1")
RULE_KW = {key: RULE_CONFIG[key] for key in ("range_expansion", "rise_boost", "fall_boost", "onset_lift", "energy_scale", "pitch_clamp", "energy_clamp")}


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_feature(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def category(text: str) -> str:
    if any(x in text for x in ("？", "?")):
        return "question"
    if any(x in text for x in ("！", "!")):
        return "surprise"
    if any(x in text for x in ("ねえ", "おーい", "もしもし", "みんな", "聞いて")):
        return "call"
    if any(x in text for x in ("どうしよう", "わから", "困った", "えーっと", "なんで")):
        return "confusion"
    if len(text) <= 9:
        return "short"
    if len(text) >= 24:
        return "long"
    return "normal"


def choose_eval(rows: list[dict[str, str]], count: int = 50) -> list[dict[str, str]]:
    buckets: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        buckets.setdefault(category(row.get("text", "")), []).append(row)
    for values in buckets.values():
        values.sort(key=lambda x: x["id"])
    labels = ["short", "normal", "question", "surprise", "confusion", "call", "long"]
    chosen: list[dict[str, str]] = []
    cursor = {label: 0 for label in labels}
    while len(chosen) < count:
        progressed = False
        for label in labels:
            values = buckets.get(label, [])
            if cursor[label] < len(values) and values[cursor[label]] not in chosen:
                chosen.append(values[cursor[label]])
                cursor[label] += 1
                progressed = True
                if len(chosen) >= count:
                    break
        if not progressed:
            break
    if len(chosen) < count:
        for row in sorted(rows, key=lambda x: x["id"]):
            if row not in chosen:
                chosen.append(row)
            if len(chosen) >= count:
                break
    return chosen[:count]


def mio_target(neutral: dict[str, np.ndarray], anime: dict[str, np.ndarray], mapping: np.ndarray) -> dict[str, np.ndarray]:
    nr, _ = relative_clean_f0(neutral["f0"], neutral["voiced"])
    ar, _ = relative_clean_f0(anime["f0"], anime["voiced"])
    mapping = np.clip(mapping, 0, len(ar) - 1)
    anime_energy = anime["energy"].astype(np.float32)
    av = anime["voiced"] > 0
    anime_energy = anime_energy - (np.median(anime_energy[av]) if np.any(av) else np.median(anime_energy))
    nv = neutral["voiced"].astype(np.float32)
    target_rel = ar[mapping].astype(np.float32)
    delta = (target_rel - nr).astype(np.float32)
    energy = neutral["energy"].astype(np.float32)
    return dict(neutral_f0=neutral["f0"], neutral_voiced=nv,
                neutral_relative_f0_st=nr, baseline_f0=np.float32(0),
                target_relative_f0_st=target_rel, target_delta_f0_st=delta,
                neutral_energy=energy, neutral_relative_energy=(energy - np.median(energy[nv > 0]) if np.any(nv > 0) else energy),
                target_energy=(energy + np.clip(anime_energy[mapping], -1.5, 1.5)).astype(np.float32),
                target_delta_energy=np.clip(anime_energy[mapping], -1.5, 1.5).astype(np.float32),
                target_voiced=(anime["voiced"][mapping] > 0).astype(np.float32),
                time=neutral["time"], duration=neutral["duration"], hop_seconds=neutral["hop_seconds"],
                teacher_type=np.array("mio_anime"))


def save_npz(path: Path, values: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def serial_scores(scores: dict[str, object]) -> dict[str, float]:
    return {k: float(v) for k, v in scores.items() if isinstance(v, (float, int, np.floating, np.integer))}


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return dict(count=0)
    keys = ["f0_std_ratio", "f0_range_ratio", "f0_iqr_ratio", "energy_std_ratio", "energy_range_ratio",
            "duration_ratio", "alignment_score", "pitch_score", "energy_score", "timing_score", "overall_score"]
    out: dict[str, object] = {"count": len(rows)}
    for key in keys:
        values = np.asarray([r[key] for r in rows], dtype=np.float64)
        out[key] = dict(mean=float(np.mean(values)), median=float(np.median(values)), p95=float(np.percentile(values, 95)),
                        minimum=float(np.min(values)), maximum=float(np.max(values)))
    out["std_greater_count"] = int(sum(r["f0_std_ratio"] > 1.0 for r in rows))
    out["range_greater_count"] = int(sum(r["f0_range_ratio"] > 1.0 for r in rows))
    out["std_goal_count"] = int(sum(r["f0_std_ratio"] > 1.15 for r in rows))
    out["range_goal_count"] = int(sum(r["f0_range_ratio"] > 1.10 for r in rows))
    return out


def svg_curve(path: Path, series: list[tuple[str, np.ndarray]], title: str) -> None:
    width, height = 800, 300
    all_values = np.concatenate([np.asarray(v, dtype=np.float64) for _, v in series])
    finite = all_values[np.isfinite(all_values)]
    low, high = (float(np.min(finite)), float(np.max(finite))) if len(finite) else (0., 1.)
    if high - low < 1e-6:
        high = low + 1.
    colors = ["#2878b5", "#d95f02", "#31a354", "#756bb1"]
    lines = []
    for index, (label, values) in enumerate(series):
        values = np.asarray(values, dtype=np.float64)
        points = []
        for i, value in enumerate(values):
            x = 40 + 710 * i / max(1, len(values) - 1)
            y = 260 - 220 * (float(value) - low) / (high - low) if np.isfinite(value) else 260
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(f'<polyline fill="none" stroke="{colors[index % len(colors)]}" stroke-width="1.5" points="{" ".join(points)}"/>')
        lines.append(f'<text x="{60 + index * 140}" y="20" fill="{colors[index % len(colors)]}">{label}</text>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="40" y="15" font-size="12">{title}</text><line x1="40" y1="260" x2="750" y2="260" stroke="#888"/><line x1="40" y1="40" x2="40" y2="260" stroke="#888"/>{"".join(lines)}</svg>'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def svg_histogram(path: Path, values: np.ndarray, title: str, bins: int = 10) -> None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    counts, edges = np.histogram(values, bins=bins) if len(values) else (np.zeros(bins), np.arange(bins + 1))
    width, height = 800, 300
    max_count = max(1, int(np.max(counts)))
    bars = []
    for i, count in enumerate(counts):
        x = 45 + i * 700 / bins
        w = 680 / bins
        h = 210 * int(count) / max_count
        bars.append(f'<rect x="{x:.1f}" y="250-h" width="{w - 2:.1f}" height="{h:.1f}" fill="#2878b5"/>'.replace("250-h", f"{250-h:.1f}"))
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="40" y="20" font-size="13">{title}</text><line x1="40" y1="250" x2="755" y2="250" stroke="#888"/>{"".join(bars)}</svg>'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("voice_sample/prosody_dataset"))
    parser.add_argument("--features", type=Path, default=Path("voice_sample/features"))
    parser.add_argument("--output", type=Path, default=Path("voice_sample/prosody_dataset_v2"))
    parser.add_argument("--eval-count", type=int, default=50)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.root / "sentences.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda x: x["id"])
    by_id = {row["id"]: row for row in rows}
    eval_rows = choose_eval(rows, args.eval_count)
    cache: dict[str, dict[str, np.ndarray]] = {}

    def pair(identifier: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, dict[str, float]]:
        if identifier not in cache:
            cache[identifier] = {style: load_feature(args.features / style / f"{identifier}.npz") for style in ("neutral", "anime")}
        neutral, anime = cache[identifier]["neutral"], cache[identifier]["anime"]
        mapping, quality = align(neutral, anime)
        return neutral, anime, mapping, quality

    evaluation: dict[str, list[dict[str, object]]] = {name: [] for name in ("A_MioTTS", "B_Rule", "C_Hybrid", "E_MioRule")}
    eval_records = []
    for row in eval_rows:
        identifier = row["id"]
        neutral, anime, mapping, aq = pair(identifier)
        targets = {
            "A_MioTTS": mio_target(neutral, anime, mapping),
            "B_Rule": rule_target(neutral, **RULE_KW),
            "C_Hybrid": hybrid_target(neutral, anime, mapping),
            "E_MioRule": corrected_anime_target(neutral, anime, mapping),
        }
        record = {"id": identifier, "text": row.get("text", ""), "category": category(row.get("text", "")), "alignment": aq}
        for name, target in targets.items():
            scores = serial_scores(teacher_scores(neutral, target, aq))
            evaluation[name].append(scores)
            record[name] = scores
        eval_records.append(record)
    summary_eval = {name: summarize(values) for name, values in evaluation.items()}
    # Prefer hybrid when it meets the pitch goals; otherwise choose the best
    # numerical teacher rather than silently assuming it.
    ranked = sorted(summary_eval, key=lambda name: (summary_eval[name]["std_goal_count"] / args.eval_count + summary_eval[name]["range_goal_count"] / args.eval_count,
                                                    summary_eval[name]["overall_score"]["mean"]), reverse=True)
    primary_name = "C_Hybrid" if "C_Hybrid" in ranked and (summary_eval["C_Hybrid"]["std_goal_count"] >= args.eval_count * .75 and summary_eval["C_Hybrid"]["range_goal_count"] >= args.eval_count * .75) else ranked[0]
    target_factory = {"A_MioTTS": mio_target, "B_Rule": lambda n, a, m: rule_target(n, **RULE_KW),
                      "C_Hybrid": hybrid_target, "E_MioRule": corrected_anime_target}[primary_name]

    # Generate the chosen teacher for all 500 pairs only after the 50-sentence comparison.
    target_dir = args.output / "targets" / primary_name
    sample_dir = args.output / "samples"
    manifest = []
    accepted = []
    rejected = []
    all_scores = []
    shape_errors = 0
    nonfinite_errors = 0
    clamp_errors = 0
    extreme_velocity_count = 0
    for index, row in enumerate(rows):
        identifier = row["id"]
        try:
            neutral, anime, mapping, aq = pair(identifier)
            target = target_factory(neutral, anime, mapping)
            scores = serial_scores(teacher_scores(neutral, target, aq))
            all_scores.append(scores)
            save_npz(target_dir / f"{identifier}.npz", target)
            # Keep a compact copy of the validated input features in the v2 tree.
            for style in ("neutral", "anime"):
                save_npz(args.output / "features" / style / f"{identifier}.npz", cache[identifier][style])
            alignment = dict(mapping=mapping, mean_cost=np.float32(aq.get("mean_cost", 0)),
                             path_frames=np.int32(aq.get("path_frames", 0)), longest_flat_seconds=np.float32(aq.get("longest_flat_seconds", 0)))
            save_npz(args.output / "aligned" / f"{identifier}.npz", alignment)
            reasons = []
            pitch_delta = np.asarray(target["target_delta_f0_st"], dtype=np.float64)
            energy_delta = np.asarray(target["target_delta_energy"], dtype=np.float64)
            if len(pitch_delta) != len(neutral["f0"]) or len(energy_delta) != len(neutral["f0"]):
                shape_errors += 1
                reasons.append("shape mismatch")
            if not np.isfinite(pitch_delta).all() or not np.isfinite(energy_delta).all():
                nonfinite_errors += 1
                reasons.append("non-finite target")
            if np.max(np.abs(pitch_delta)) > 1.5001 or np.max(np.abs(energy_delta)) > 1.5001:
                clamp_errors += 1
                reasons.append("pitch clamp exceeded")
            voiced_mask = np.asarray(target.get("target_voiced", neutral["voiced"])) > 0
            velocity = np.diff(pitch_delta)
            velocity_mask = voiced_mask[1:] & voiced_mask[:-1]
            if np.any(velocity_mask) and np.max(np.abs(velocity[velocity_mask])) > 0.85:
                extreme_velocity_count += 1
                reasons.append("extreme velocity")
            if aq.get("mean_cost", 99) > 2.5 or aq.get("normalized_endpoint_error", 99) > .02:
                reasons.append("alignment quality")
            if scores["validity_score"] < 1:
                reasons.append("validity")
            if reasons:
                rejected.append(dict(id=identifier, reasons=reasons, scores=scores))
            else:
                sample = target_sample(neutral, target, mapping, scores, identifier)
                save_npz(sample_dir / f"{identifier}.npz", sample)
                accepted.append(identifier)
                manifest.append(dict(id=identifier, split="", source=str(args.root), teacher_type=primary_name,
                                     duration=float(neutral["duration"]), frames=int(len(neutral["f0"])), **scores))
        except Exception as exc:
            rejected.append(dict(id=identifier, reasons=[f"exception: {type(exc).__name__}: {exc}"]))
        if (index + 1) % 100 == 0:
            print(f"processed {index + 1}/{len(rows)}", flush=True)

    # Fixed, ID-level split with no neutral/anime leakage.
    ordered = list(accepted)
    np.random.default_rng(SEED).shuffle(ordered)
    n = len(ordered); n_train = int(n * .8); n_val = int(n * .1)
    splits = dict(seed=SEED, train=ordered[:n_train], validation=ordered[n_train:n_train + n_val], test=ordered[n_train + n_val:])
    split_by_id = {identifier: split for split in ("train", "validation", "test") for identifier in splits[split]}
    for row in manifest:
        row["split"] = split_by_id[row["id"]]
    (args.output / "manifest.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False, allow_nan=False) for row in manifest) + "\n", encoding="utf-8")
    write_json(args.output / "splits" / "splits.json", splits)
    write_json(args.output / "teacher_config.json", dict(teacher_type=primary_name, source_miotts="existing cached anime features",
        feature_extractor="torchfcpe-0.0.4 cached v0.5", seed=SEED, **RULE_CONFIG))

    # Validation and compact report artifacts.
    critical_errors = nonfinite_errors + shape_errors + clamp_errors
    validation = dict(total_pairs=len(rows), accepted=len(accepted), rejected=len(rejected), critical_errors=critical_errors,
                      all_finite=nonfinite_errors == 0, target_clamp_ok=clamp_errors == 0,
                      extreme_velocity_count=extreme_velocity_count, shape_errors=shape_errors,
                      nonfinite_errors=nonfinite_errors, clamp_errors=clamp_errors,
                      rejected_ids=[r["id"] for r in rejected])
    write_json(args.output / "validation_report.json", validation)
    write_json(args.output / "teacher_comparison.json", dict(eval_set=[r["id"] for r in eval_rows], records=eval_records, summary=summary_eval, primary_teacher=primary_name))
    for identifier in [row["id"] for row in eval_rows[:20]]:
        neutral, anime, mapping, _ = pair(identifier)
        target = target_factory(neutral, anime, mapping)
        svg_curve(args.output / "previews" / f"{identifier}_pitch.svg",
                  [("neutral", target["neutral_relative_f0_st"]), ("target", target["target_relative_f0_st"])], f"{identifier} pitch (semitones)")
        svg_curve(args.output / "previews" / f"{identifier}_energy.svg",
                  [("neutral", target["neutral_energy"]), ("target", target["target_energy"])], f"{identifier} energy (dB)")
    svg_curve(args.output / "previews" / "quality_scores.svg", [("overall", np.asarray([x["overall_score"] for x in all_scores]))], "overall quality score")
    histogram_fields = {"pitch": "pitch_score", "energy": "energy_score", "timing": "timing_score",
                        "alignment": "alignment_score", "overall": "overall_score"}
    for name, field in histogram_fields.items():
        svg_histogram(args.output / "previews" / f"{name}_score_histogram.svg",
                      np.asarray([x[field] for x in all_scores]), f"{name} score histogram")

    summary_all = summarize(all_scores)
    go = (summary_all["std_goal_count"] >= len(all_scores) * .75 and summary_all["range_goal_count"] >= len(all_scores) * .75 and validation["critical_errors"] == 0)
    model_config = dict(input_dim=5, history_ms=500, frame_ms=10, hidden_size=96, target_dims=["delta_f0_st", "delta_energy"],
                        loss_weights={"f0": 1.0, "energy": .35}, train_split="train", validation_split="validation", test_split="test")
    if go:
        write_json(args.output / "recommended_model_config.json", model_config)
    else:
        (args.output / "why_not_train.md").write_text("# Why ProsodyNet training is NO-GO\n\nThe v0.6 target acceptance thresholds were not met. Inspect teacher_comparison.json and previews before changing the rule.\n", encoding="utf-8")
    report = ["# Prosody Dataset v2 (v0.6)", "", f"Primary teacher: **{primary_name}**", f"Evaluation set: {len(eval_rows)} sentences", "", "## Teacher comparison (50 sentences)", ""]
    for name in evaluation:
        s = summary_eval[name]
        report.append(f"- **{name}**: F0 std ratio mean {s['f0_std_ratio']['mean']:.3f}; range ratio mean {s['f0_range_ratio']['mean']:.3f}; energy std ratio {s['energy_std_ratio']['mean']:.3f}; overall {s['overall_score']['mean']:.3f}; pitch-goal {s['std_goal_count']}/{args.eval_count} std, {s['range_goal_count']}/{args.eval_count} range")
    report += ["", "## Full dataset", f"- Total: {len(rows)}; accepted: {len(accepted)}; rejected: {len(rejected)}", f"- Target F0 std > neutral: {summary_all['std_greater_count']}/{len(all_scores)} ({100*summary_all['std_greater_count']/max(1,len(all_scores)):.1f}%)", f"- Target F0 range > neutral: {summary_all['range_greater_count']}/{len(all_scores)} ({100*summary_all['range_greater_count']/max(1,len(all_scores)):.1f}%)", f"- Minimum goals (>1.15 std, >1.10 range): {summary_all['std_goal_count']}/{len(all_scores)}, {summary_all['range_goal_count']}/{len(all_scores)}", f"- Splits: train {len(splits['train'])}, validation {len(splits['validation'])}, test {len(splits['test'])}", f"- Validation critical errors: {validation['critical_errors']}", f"- ProsodyNet training decision: **{'GO' if go else 'NO-GO'}**", "", "Pitch uses the cleaned neutral relative F0 with range expansion, rise boost, onset lift, smoothing and a -1.0/+1.4 semitone correction clamp. Hybrid uses existing MioTTS anime energy and timing through DTW; the old WAV/features remain untouched.", "", "Preview curves are SVG files under `previews/`; numerical validation is automatic and perceptual review remains a human task."]
    (args.output / "dataset_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    comparison_report = ["# v0.6 Teacher Comparison", "", "Fixed evaluation set (deterministic seed 600): " + ", ".join(r["id"] for r in eval_rows), ""]
    for name in evaluation:
        s = summary_eval[name]
        comparison_report.append(f"## {name}\nF0 std ratio mean/median: {s['f0_std_ratio']['mean']:.3f}/{s['f0_std_ratio']['median']:.3f}\nF0 range ratio mean/median: {s['f0_range_ratio']['mean']:.3f}/{s['f0_range_ratio']['median']:.3f}\nEnergy std ratio: {s['energy_std_ratio']['mean']:.3f}\nDuration ratio: {s['duration_ratio']['mean']:.3f}\nAlignment score: {s['alignment_score']['mean']:.3f}\nOverall score: {s['overall_score']['mean']:.3f}\nPitch goals: std {s['std_goal_count']}/{args.eval_count}, range {s['range_goal_count']}/{args.eval_count}\n")
    comparison_report.append(f"Primary teacher selected: **{primary_name}**. Teacher D was not run because no additional reference was supplied; Teacher E is the MioTTS+rule correction comparison.")
    Path("voice_sample/v06_teacher_comparison.md").write_text("\n".join(comparison_report), encoding="utf-8")
    print(json.dumps(dict(primary_teacher=primary_name, accepted=len(accepted), rejected=len(rejected), summary=summary_all, go=go), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
