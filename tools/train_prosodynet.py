"""Train and benchmark the offline v0.7 ProsodyNet-S POC.

No realtime audio code is imported.  The fixed v0.6 Dataset v2 split is used
as-is, and all normalization statistics are computed from the train IDs only.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.prosodynet.data import (WindowDataset, causal_features, load_sample, load_splits,
                                 normalize, split_leakage, targets, training_statistics)
from src.prosodynet.losses import prosody_loss
from src.prosodynet.model import MODEL_SPECS, ProsodyNetS


SEED = 707
DEFAULT_CONFIG = dict(seed=SEED, sequence_length=50, history_ms=500, frame_ms=10,
                      batch_size=64, learning_rate=0.001, epochs=10, patience=4,
                      num_threads=4, loss_weights=dict(pitch=1.0, energy=.30, velocity=.20, smoothness=.05))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(max(1, int(DEFAULT_CONFIG.get("num_threads", 4))))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def stats(values: list[float] | np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return dict(avg=0., p50=0., p95=0., p99=0., max=0.)
    return dict(avg=float(np.mean(values)), p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)), p99=float(np.percentile(values, 99)), max=float(np.max(values)))


def batch_loss(model: ProsodyNetS, batch: dict[str, torch.Tensor], device: torch.device, weights: dict[str, float]) -> tuple[torch.Tensor, dict[str, float]]:
    prediction, _ = model(batch["x"].to(device))
    return prosody_loss(prediction, batch["y"].to(device), batch["mask"].to(device), batch["voiced"].to(device), weights)


def run_epoch(model: ProsodyNetS, loader: DataLoader, device: torch.device, weights: dict[str, float], optimizer: torch.optim.Optimizer | None) -> dict[str, float]:
    model.train(optimizer is not None)
    records: list[dict[str, float]] = []
    for batch in loader:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        loss, values = batch_loss(model, batch, device, weights)
        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        records.append(values)
    keys = ("pitch", "energy", "velocity", "smoothness", "total")
    return {key: float(np.mean([row[key] for row in records])) for key in keys}


def train_one(name: str, root: Path, output: Path, splits: dict[str, list[str]], config: dict[str, Any], stats_info: dict[str, Any], device: torch.device, resume: bool = False) -> dict[str, Any]:
    spec = MODEL_SPECS[name]
    model = ProsodyNetS(input_dim=len(stats_info["features"]), **spec).to(device)
    train_data = WindowDataset(root, splits["train"], stats_info, config["sequence_length"])
    val_data = WindowDataset(root, splits["validation"], stats_info, config["sequence_length"])
    generator = torch.Generator().manual_seed(int(config["seed"]))
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=3, min_lr=1e-5)
    checkpoint_dir = output / "checkpoints" / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    start_epoch = 1
    best_val = float("inf")
    best_epoch = 0
    wait = 0
    history: list[dict[str, Any]] = []
    prior_train_seconds = 0.0
    if resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_val = float(state.get("best_val", best_val))
        best_epoch = int(state.get("best_epoch", 0))
        wait = int(state.get("wait", 0))
        history = list(state.get("history", []))
        prior_train_seconds = float(state.get("train_seconds", 0.0))
        if prior_train_seconds <= 0:
            metadata_path = output / "models" / name / "metadata.json"
            if metadata_path.exists():
                try:
                    prior_train_seconds = float(json.loads(metadata_path.read_text(encoding="utf-8")).get("train_seconds", 0.0))
                except (OSError, ValueError, TypeError):
                    prior_train_seconds = 0.0
    started = time.perf_counter()
    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        train_values = run_epoch(model, train_loader, device, config["loss_weights"], optimizer)
        with torch.no_grad():
            val_values = run_epoch(model, val_loader, device, config["loss_weights"], None)
        scheduler.step(val_values["total"])
        record = dict(epoch=epoch, train=train_values, validation=val_values, learning_rate=optimizer.param_groups[0]["lr"])
        history.append(record)
        state = dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     best_val=best_val, best_epoch=best_epoch, wait=wait, history=history,
                     train_seconds=prior_train_seconds + (time.perf_counter() - started), model_name=name, config=config)
        torch.save(state, last_path)
        if val_values["total"] < best_val - 1e-6:
            best_val = val_values["total"]
            best_epoch = epoch
            wait = 0
            state.update(best_val=best_val, best_epoch=best_epoch, wait=wait)
            torch.save(state, best_path)
        else:
            wait += 1
        if wait >= int(config["patience"]):
            break
    elapsed = time.perf_counter() - started
    if best_path.exists():
        best_state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_state["model"])
        best_epoch = int(best_state.get("best_epoch", best_epoch))
        best_val = float(best_state.get("best_val", best_val))
    total_train_seconds = prior_train_seconds + elapsed
    best_record = next((row for row in history if int(row["epoch"]) == best_epoch), None)
    metadata = dict(model_name=name, parameter_count=model.parameter_count, architecture=dict(type="causal_conv1d_gru", **spec),
                    frame_ms=config["frame_ms"], history_ms=config["history_ms"], input_features=stats_info["features"],
                    outputs=["delta_f0_st", "delta_energy"], training_dataset=str(root), teacher_version="v0.6 C Hybrid",
                    best_epoch=best_epoch, best_train_loss=(best_record["train"]["total"] if best_record else None),
                    best_validation_loss=best_val, train_seconds=total_train_seconds, seed=config["seed"])
    write_json(output / "logs" / f"{name}_history.json", history)
    model_dir = output / "models" / name
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, model_dir / "best.pt")
    shutil.copy2(last_path, model_dir / "last.pt")
    write_json(model_dir / "metadata.json", metadata)
    return dict(model=model, metadata=metadata, history=history, best_path=str(best_path), last_path=str(last_path),
                train_windows=len(train_data), validation_windows=len(val_data))


@torch.no_grad()
def evaluate_model(model: ProsodyNetS, root: Path, identifiers: list[str], stats_info: dict[str, Any], device: torch.device) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {key: [] for key in ("pred_pitch", "pred_pitch_quantized", "target_pitch", "pred_energy", "target_energy", "voiced", "mask")}
    per_sample: list[dict[str, Any]] = []
    raw_clamp_violations = 0
    for identifier in identifiers:
        sample = load_sample(root, identifier)
        x = torch.from_numpy(normalize(causal_features(sample), stats_info)).unsqueeze(0).to(device)
        prediction, _ = model(x, return_raw=False)
        pred = prediction[0].cpu().numpy()
        target = targets(sample)
        voiced = sample["neutral_voiced"] > 0
        mask = np.ones(len(target), dtype=bool)
        pitch_mask = mask & voiced
        pitch_error = pred[:, 0] - target[:, 0]
        energy_error = pred[:, 1] - target[:, 1]
        quantized = np.clip(np.round(pred[:, 0] / .125) * .125, -1.0, 1.4)
        velocity_mask = pitch_mask[1:] & pitch_mask[:-1]
        velocity_error = (np.diff(pred[:, 0]) - np.diff(target[:, 0]))[velocity_mask]
        second_pred = np.diff(pred[:, 0], n=2)
        second_target = np.diff(target[:, 0], n=2)
        per_sample.append(dict(id=identifier, pitch_mae=float(np.mean(np.abs(pitch_error[pitch_mask])) if np.any(pitch_mask) else 0.),
                               energy_mae=float(np.mean(np.abs(energy_error))), voiced_ratio=float(np.mean(voiced))))
        collected["pred_pitch"].append(pred[:, 0]); collected["pred_pitch_quantized"].append(quantized); collected["target_pitch"].append(target[:, 0])
        collected["pred_energy"].append(pred[:, 1]); collected["target_energy"].append(target[:, 1]); collected["voiced"].append(voiced.astype(np.float32)); collected["mask"].append(mask.astype(np.float32))
        raw_clamp_violations += int(np.sum((pred[:, 0] < -1.00001) | (pred[:, 0] > 1.40001) | (pred[:, 1] < -1.50001) | (pred[:, 1] > 1.50001)))
    pitch = np.concatenate(collected["pred_pitch"]); target_pitch = np.concatenate(collected["target_pitch"]); quantized = np.concatenate(collected["pred_pitch_quantized"])
    energy = np.concatenate(collected["pred_energy"]); target_energy = np.concatenate(collected["target_energy"]); voiced = np.concatenate(collected["voiced"]) > 0
    pitch_mask = voiced
    metrics = dict(pitch_mae=float(np.mean(np.abs(pitch[pitch_mask] - target_pitch[pitch_mask]))),
                   pitch_rmse=float(np.sqrt(np.mean((pitch[pitch_mask] - target_pitch[pitch_mask]) ** 2))),
                   energy_mae=float(np.mean(np.abs(energy - target_energy))),
                   energy_rmse=float(np.sqrt(np.mean((energy - target_energy) ** 2))),
                   quantized_pitch_mae=float(np.mean(np.abs(quantized[pitch_mask] - target_pitch[pitch_mask]))),
                   nan_inf=int((not np.isfinite(pitch).all()) or (not np.isfinite(energy).all())),
                   clamp_violations=int(raw_clamp_violations), per_sample=per_sample)
    velocity_errors = []
    smooth_errors = []
    for identifier in identifiers:
        sample = load_sample(root, identifier)
        x = torch.from_numpy(normalize(causal_features(sample), stats_info)).unsqueeze(0).to(device)
        pred = model(x)[0][0].cpu().numpy(); truth = targets(sample); v = sample["neutral_voiced"] > 0
        valid = v[1:] & v[:-1]
        if np.any(valid): velocity_errors.extend((np.diff(pred[:, 0]) - np.diff(truth[:, 0]))[valid])
        if len(pred) > 2: smooth_errors.extend(np.diff(pred[:, 0], n=2) - np.diff(truth[:, 0], n=2))
    metrics["velocity_mae"] = float(np.mean(np.abs(velocity_errors)) if velocity_errors else 0.)
    metrics["smoothness_error"] = float(np.mean(np.abs(smooth_errors)) if smooth_errors else 0.)
    metrics["second_derivative_pred_mean_abs"] = float(np.mean(np.abs(smooth_errors)) if smooth_errors else 0.)
    return metrics, {key: np.asarray(value, dtype=object) for key, value in collected.items()}


@torch.no_grad()
def streaming_benchmark(model: ProsodyNetS, root: Path, identifiers: list[str], stats_info: dict[str, Any], device: torch.device, max_frames: int = 3000) -> dict[str, Any]:
    model.eval()
    timings: list[float] = []
    total_audio = 0.0
    state = model.initial_state(1, device)
    previous_threads = torch.get_num_threads()
    # One-frame CPU inference is dominated by thread-pool startup otherwise.
    torch.set_num_threads(1)
    for identifier in identifiers:
        sample = load_sample(root, identifier)
        features = normalize(causal_features(sample), stats_info)
        total_audio += len(features) * .010
        state = model.reset_state(state)
        for frame in features:
            if len(timings) >= max_frames:
                break
            tensor = torch.from_numpy(frame).to(device)
            started = time.perf_counter_ns()
            model.step(tensor, state)
            timings.append((time.perf_counter_ns() - started) / 1e6)
        if len(timings) >= max_frames:
            break
    torch.set_num_threads(previous_threads)
    summary = stats(timings)
    summary["rtf"] = float(sum(timings) / 1000.0 / max(total_audio, 1e-9))
    summary["frames"] = len(timings)
    try:
        import psutil
        summary["ram_mb"] = float(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
        summary["cpu_percent"] = float(psutil.cpu_percent(interval=.1))
    except Exception:
        summary["ram_mb"] = None; summary["cpu_percent"] = None
    return summary


def svg_plot(path: Path, series: list[tuple[str, np.ndarray]], title: str, zero: bool = False) -> None:
    width, height = 820, 320
    arrays = [np.asarray(values, dtype=np.float64) for _, values in series]
    finite = np.concatenate([value[np.isfinite(value)] for value in arrays if np.isfinite(value).any()]) if arrays else np.array([0.])
    low, high = float(np.min(finite)), float(np.max(finite))
    if zero: low, high = min(low, 0.), max(high, 0.)
    if high - low < 1e-8: high = low + 1.
    colors = ["#2878b5", "#d95f02", "#31a354", "#756bb1"]
    content = [f'<text x="40" y="18" font-size="13">{title}</text>', '<line x1="40" y1="270" x2="780" y2="270" stroke="#888"/>']
    if low < 0 < high:
        y = 270 - 230 * (0 - low) / (high - low); content.append(f'<line x1="40" y1="{y:.1f}" x2="780" y2="{y:.1f}" stroke="#ddd"/>')
    for index, (label, values) in enumerate(series):
        points = []
        for i, value in enumerate(np.asarray(values, dtype=np.float64)):
            x = 40 + 740 * i / max(1, len(values) - 1)
            y = 270 - 230 * (float(value) - low) / (high - low) if np.isfinite(value) else 270
            points.append(f"{x:.1f},{y:.1f}")
        color = colors[index % len(colors)]
        content.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.4" points="{" ".join(points)}"/>')
        content.append(f'<text x="{50 + index * 180}" y="38" fill="{color}">{label}</text>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/>{"".join(content)}</svg>', encoding="utf-8")


def evaluate_and_report(name: str, result: dict[str, Any], root: Path, output: Path, splits: dict[str, list[str]], stats_info: dict[str, Any], device: torch.device) -> dict[str, Any]:
    metrics, _ = evaluate_model(result["model"], root, splits["test"], stats_info, device)
    benchmark = streaming_benchmark(result["model"], root, splits["test"], stats_info, device)
    model_path = Path(result["best_path"])
    metrics.update(dict(parameters=result["model"].parameter_count, model_size_bytes=model_path.stat().st_size,
                        best_epoch=result["metadata"]["best_epoch"], best_train_loss=result["metadata"].get("best_train_loss"), best_validation_loss=result["metadata"]["best_validation_loss"],
                        train_seconds=result["metadata"]["train_seconds"], streaming=benchmark,
                        train_windows=result["train_windows"], validation_windows=result["validation_windows"]))
    write_json(output / "benchmarks" / f"{name}.json", metrics)
    worst = sorted(metrics["per_sample"], key=lambda row: row["pitch_mae"], reverse=True)[:10]
    write_json(output / "reports" / f"{name}_worst_cases.json", worst)
    history = result["history"]
    train_loss = np.asarray([row["train"]["total"] for row in history]); val_loss = np.asarray([row["validation"]["total"] for row in history])
    svg_plot(output / "plots" / f"{name}_loss.svg", [("train", train_loss), ("validation", val_loss)], f"{name} loss")
    # A fixed subset of the Test Set is used for all three models.
    for identifier in splits["test"][:20]:
        sample = load_sample(root, identifier)
        x = torch.from_numpy(normalize(causal_features(sample), stats_info)).unsqueeze(0).to(device)
        prediction = result["model"](x)[0][0].detach().cpu().numpy()
        truth = targets(sample)
        svg_plot(output / "test_predictions" / name / f"{identifier}_pitch.svg",
                 [("neutral", sample["neutral_f0_st"]), ("teacher", truth[:, 0]), ("net", prediction[:, 0])], f"{name} {identifier} pitch", True)
        svg_plot(output / "test_predictions" / name / f"{identifier}_energy.svg",
                 [("teacher", truth[:, 1]), ("net", prediction[:, 1])], f"{name} {identifier} energy", True)
        svg_plot(output / "test_predictions" / name / f"{identifier}_error.svg",
                 [("pitch error", prediction[:, 0] - truth[:, 0]), ("energy error", prediction[:, 1] - truth[:, 1])], f"{name} {identifier} error", True)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("voice_sample/prosody_dataset_v2"))
    parser.add_argument("--output", type=Path, default=Path("voice_sample/prosodynet_v07"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="ignore existing checkpoints")
    args = parser.parse_args()
    config = dict(DEFAULT_CONFIG)
    config["loss_weights"] = dict(DEFAULT_CONFIG["loss_weights"])
    if args.config and args.config.exists():
        config.update(json.loads(args.config.read_text(encoding="utf-8")))
    config["seed"] = int(config.get("seed", SEED))
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "configs" / "training.json", config)
    seed_everything(config["seed"])
    root = args.dataset
    splits = load_splits(root)
    leakage = split_leakage(splits)
    if leakage:
        raise ValueError(f"Dataset leakage: {leakage}")
    stats_info = training_statistics(root, splits["train"])
    write_json(args.output / "normalization.json", stats_info)
    write_json(args.output / "reports" / "split_check.json", dict(sizes={key: len(value) for key, value in splits.items()}, leakage=leakage,
                                                                      train_ids=splits["train"], validation_ids=splits["validation"], test_ids=splits["test"]))
    device = torch.device("cpu")
    results: dict[str, dict[str, Any]] = {}
    metrics: dict[str, Any] = {}
    for name in args.models:
        print(f"training {name}", flush=True)
        result = train_one(name, root, args.output, splits, config, stats_info, device, args.resume and not args.fresh)
        results[name] = result
        print(f"evaluating {name}", flush=True)
        metrics[name] = evaluate_and_report(name, result, root, args.output, splits, stats_info, device)
    # Rule and identity are explicit offline baselines, not learned models.
    identity = dict(pitch_mae=float(np.mean([np.mean(np.abs(targets(load_sample(root, i))[:, 0])) for i in splits["test"]])),
                    energy_mae=float(np.mean([np.mean(np.abs(targets(load_sample(root, i))[:, 1])) for i in splits["test"]])))
    write_json(args.output / "benchmarks" / "baselines.json", dict(rule_teacher=dict(pitch_mae=0., energy_mae=0., runtime="offline target already available"), identity=identity))
    # Select a quality/latency balanced offline candidate.  Realtime integration
    # remains NO-GO until human A/B establishes no audible regression.
    ranked = sorted(args.models, key=lambda name: (metrics[name]["pitch_mae"], metrics[name]["energy_mae"], metrics[name]["streaming"]["p95"], metrics[name]["parameters"]))
    best = ranked[0]
    # A much larger Medium model is not preferred for a marginal improvement.
    if "small" in metrics and "medium" in metrics:
        pitch_gain = (metrics["small"]["pitch_mae"] - metrics["medium"]["pitch_mae"]) / max(metrics["small"]["pitch_mae"], 1e-9)
        energy_gain = (metrics["small"]["energy_mae"] - metrics["medium"]["energy_mae"]) / max(metrics["small"]["energy_mae"], 1e-9)
        if pitch_gain < .10 and energy_gain < .10:
            best = "small"
    report = ["# ProsodyNet-S v0.7 POC", "", f"Dataset: {root}", f"Split: train {len(splits['train'])} / validation {len(splits['validation'])} / test {len(splits['test'])}", f"Leakage: {'PASS' if not leakage else 'FAIL'}", "", "## Model comparison", "", "| Model | Parameters | Size | Pitch MAE | Energy MAE | Velocity MAE | Quantized Pitch MAE | Streaming p95 ms | RTF | Best epoch |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in args.models:
        m = metrics[name]; report.append(f"| {name} | {m['parameters']:,} | {m['model_size_bytes']:,} B | {m['pitch_mae']:.4f} st | {m['energy_mae']:.4f} dB | {m['velocity_mae']:.4f} | {m['quantized_pitch_mae']:.4f} st | {m['streaming']['p95']:.4f} | {m['streaming']['rtf']:.5f} | {m['best_epoch']} |")
        report.append(f"  - best train/validation loss {m.get('best_train_loss')} / {m['best_validation_loss']:.6f}; curves: `plots/{name}_loss.svg`")
    bm = metrics[best]
    quality_pass = bm["pitch_mae"] < .20 and bm["energy_mae"] < .5 and bm["nan_inf"] == 0 and bm["clamp_violations"] == 0 and bm["streaming"]["p95"] < 5 and bm["streaming"]["rtf"] < 1
    report += ["", f"## Best offline candidate: **{best}**", f"Best validation epoch: {bm['best_epoch']}", f"Train time: {bm['train_seconds']:.2f} s", f"Test Pitch RMSE: {bm['pitch_rmse']:.4f} st", f"Test Energy RMSE: {bm['energy_rmse']:.4f} dB", f"Smoothness error: {bm['smoothness_error']:.4f}", f"Streaming avg/p50/p95/p99/max: {bm['streaming']['avg']:.4f}/{bm['streaming']['p50']:.4f}/{bm['streaming']['p95']:.4f}/{bm['streaming']['p99']:.4f}/{bm['streaming']['max']:.4f} ms", f"N150 CPU/RAM: {bm['streaming'].get('cpu_percent')}% / {bm['streaming'].get('ram_mb')} MB", "", "## Baselines", "Rule Teacher reproduces its target with zero offline prediction error and no neural runtime. Identity (zero correction) metrics are in `benchmarks/baselines.json`.", "", "## Decision", f"Offline numeric quality: **{'PASS' if quality_pass else 'FAIL'}**", "Realtime Integration: **NO-GO**", "", "The learned model is a causal, stateful reproduction of a synthetic Rule/Hybrid teacher. No human Rule-vs-Net A/B listening has been completed in this POC, and the Rule baseline is cheaper and has zero teacher error. Keep Rule Prosody for v0.8 until a human comparison shows no audible regression or a clear smoothness/style advantage.", "", "Prediction/error SVGs are under `test_predictions/`; loss curves are SVG because matplotlib is not installed in the training environment. Audio WAV previews were not generated because this dataset stores prosody controls, not a vocoder input/output path."]
    (args.output / "model_comparison.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(args.output / "reports" / "metrics.json", metrics)
    write_json(args.output / "reports" / "decision.json", dict(best_model=best, offline_quality_pass=quality_pass, realtime_integration="NO-GO", reason="Rule baseline is cheaper and human A/B not completed"))
    (args.output / "why_not_integrate.md").write_text("# Why realtime integration is NO-GO\n\nProsodyNet-S was trained and evaluated causally, but it distills the synthetic Rule/Hybrid teacher. The Rule baseline is already zero-error and cheaper. Human Rule-vs-Net listening has not established equal or better naturalness, jitter, Anime impression, or energy pumping. Keep the Rule path in v0.8 and repeat A/B after adding a real style/context signal.\n", encoding="utf-8")
    print(json.dumps(dict(best_model=best, metrics={name: {key: value for key, value in item.items() if key not in ("per_sample",)} for name, item in metrics.items()}, realtime_integration="NO-GO"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
