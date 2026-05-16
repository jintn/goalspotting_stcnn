#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import (
    mc3_18,
    r2plus1d_18,
    r3d_18,
)

from stage2_goal_verifier import MEAN, STD, parse_annotations

ARCH_TO_FACTORY = {
    "r2plus1d_18": r2plus1d_18,
    "mc3_18": mc3_18,
    "r3d_18": r3d_18,
}


@dataclass
class ModelSpec:
    name: str
    checkpoint: str
    arch: str = "auto"
    goal_class_id: int = 3
    conf_thresh: float = 0.5
    nms_s: float = 15.0
    smooth_k: int = 1
    clip_sec: float = 4.0
    clip_frames: int = 16
    clip_size: int = 112
    stride_s: float = 0.5


@dataclass
class MatchCounts:
    tp: int
    fp: int
    fn: int
    matched_deltas_s: List[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark multiple goal-spotting checkpoints on a fixed number of SoccerNet games."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("valset"))
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark_reports"))
    parser.add_argument("--games-file", type=Path, default=None, help="Optional text file with one game path per line.")
    parser.add_argument("--num-games", type=int, default=10, help="How many games to benchmark.")
    parser.add_argument("--sample-mode", choices=("first", "random"), default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tolerance-s", type=float, default=10.0, help="Match tolerance for goal detections.")
    parser.add_argument(
        "--max-video-seconds",
        type=float,
        default=None,
        help="Optional cap per half for quick smoke tests or short-form throughput checks.",
    )
    parser.add_argument("--half", choices=("both", "1", "2"), default="both")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="JSON file containing a list of model specs.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Inline model spec. Example: "
            "name=r2p,checkpoint=checkpoints/best4classes.pt,arch=r2plus1d_18,conf_thresh=0.6,nms_s=15,smooth_k=1"
        ),
    )
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "mps":
        return torch.device("mps")
    if device_arg == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_model_spec_text(text: str) -> ModelSpec:
    parts: Dict[str, str] = {}
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Invalid model spec chunk: {chunk!r}")
        key, value = chunk.split("=", 1)
        parts[key.strip()] = value.strip()

    required = {"name", "checkpoint"}
    missing = required - set(parts)
    if missing:
        raise ValueError(f"Model spec missing required keys: {sorted(missing)}")

    return ModelSpec(
        name=parts["name"],
        checkpoint=parts["checkpoint"],
        arch=parts.get("arch", "auto"),
        goal_class_id=int(parts.get("goal_class_id", 3)),
        conf_thresh=float(parts.get("conf_thresh", 0.5)),
        nms_s=float(parts.get("nms_s", 15.0)),
        smooth_k=int(parts.get("smooth_k", 1)),
        clip_sec=float(parts.get("clip_sec", 4.0)),
        clip_frames=int(parts.get("clip_frames", 16)),
        clip_size=int(parts.get("clip_size", 112)),
        stride_s=float(parts.get("stride_s", 0.5)),
    )


def load_model_specs(args: argparse.Namespace) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    if args.model_config is not None:
        payload = json.loads(args.model_config.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("--model-config must contain a JSON list")
        for item in payload:
            specs.append(ModelSpec(**item))
    for text in args.model:
        specs.append(parse_model_spec_text(text))
    if not specs:
        raise ValueError("Provide at least one model via --model or --model-config.")
    return specs


def discover_games(data_dir: Path) -> List[str]:
    games: List[str] = []
    for labels_path in sorted(data_dir.rglob("Labels-v2.json")):
        game_dir = labels_path.parent
        has_video = (game_dir / "1_224p.mkv").exists() or (game_dir / "2_224p.mkv").exists()
        if has_video:
            games.append(str(game_dir.relative_to(data_dir)))
    return games


def select_games(data_dir: Path, games_file: Optional[Path], num_games: int, sample_mode: str, seed: int) -> List[str]:
    if games_file is not None:
        games = [line.strip() for line in games_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        games = discover_games(data_dir)

    if num_games <= 0 or num_games >= len(games):
        return games

    if sample_mode == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(games, num_games))

    return games[:num_games]


def checkpoint_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "model" in ckpt:
            return ckpt["model"]
        return ckpt
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)!r}")


def infer_out_features(state: Dict[str, torch.Tensor]) -> int:
    if "fc.weight" in state:
        return int(state["fc.weight"].shape[0])
    if "fc.1.weight" in state:
        return int(state["fc.1.weight"].shape[0])
    raise KeyError("Could not infer classifier output size from checkpoint.")


def build_backbone(arch: str, out_features: int, state: Dict[str, torch.Tensor]) -> nn.Module:
    factory = ARCH_TO_FACTORY[arch]
    model = factory(weights=None)
    in_features = model.fc.in_features
    if "fc.weight" in state:
        model.fc = nn.Linear(in_features, out_features)
    else:
        model.fc = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(in_features, out_features))
    return model


def infer_architecture(state: Dict[str, torch.Tensor], out_features: int) -> str:
    errors: List[str] = []
    for arch in ARCH_TO_FACTORY:
        model = build_backbone(arch, out_features=out_features, state=state)
        try:
            model.load_state_dict(state, strict=True)
            return arch
        except Exception as exc:
            errors.append(f"{arch}: {exc}")
    raise RuntimeError("Could not infer architecture from checkpoint.\n" + "\n".join(errors))


def load_checkpoint_model(spec: ModelSpec, device: torch.device) -> Tuple[nn.Module, str, int]:
    checkpoint_path = Path(spec.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = checkpoint_state_dict(ckpt)
    out_features = infer_out_features(state)
    arch = spec.arch
    if arch == "auto":
        arch = infer_architecture(state, out_features=out_features)
    if arch not in ARCH_TO_FACTORY:
        raise ValueError(f"Unsupported architecture: {arch}")

    model = build_backbone(arch, out_features=out_features, state=state)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, arch, out_features


def moving_average(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return arr
    pad = k // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(k, dtype=np.float32) / k, mode="valid")


def postprocess_goal_predictions(
    raw_curve: Sequence[Tuple[float, Sequence[float]]],
    goal_class_id: int,
    conf_thresh: float,
    nms_radius_s: float,
    smooth_k: int,
    out_features: int,
) -> List[Dict[str, float]]:
    if not raw_curve:
        return []

    timestamps = np.asarray([item[0] for item in raw_curve], dtype=np.float32)

    if out_features == 1:
        goal_probs = np.asarray([float(item[1][0]) for item in raw_curve], dtype=np.float32)
    else:
        if goal_class_id >= out_features:
            raise ValueError(
                f"goal_class_id={goal_class_id} is out of range for out_features={out_features}. "
                "Set goal_class_id correctly for this checkpoint."
            )
        goal_probs = np.asarray([float(item[1][goal_class_id]) for item in raw_curve], dtype=np.float32)

    goal_probs = moving_average(goal_probs, smooth_k)
    candidates = sorted(zip(timestamps.tolist(), goal_probs.tolist()), key=lambda x: -x[1])

    detections: List[Dict[str, float]] = []
    for ts, prob in candidates:
        if prob < conf_thresh:
            continue
        if all(abs(ts - det["timestamp_s"]) >= nms_radius_s for det in detections):
            detections.append({"timestamp_s": float(ts), "confidence": float(prob)})

    detections.sort(key=lambda item: item["timestamp_s"])
    return detections


def infer_raw_curve_on_video(
    model: nn.Module,
    video_path: Path,
    device: torch.device,
    clip_sec: float,
    clip_frames: int,
    clip_size: int,
    stride_s: float,
    batch_size: int,
    max_video_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"raw": [], "video_duration_s": 0.0, "wall_time_s": 0.0, "num_windows": 0, "peak_gpu_mb": 0.0}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        return {"raw": [], "video_duration_s": 0.0, "wall_time_s": 0.0, "num_windows": 0, "peak_gpu_mb": 0.0}

    if max_video_seconds is not None and max_video_seconds > 0:
        total_frames = min(total_frames, max(1, int(max_video_seconds * fps)))

    step = max(1, int(clip_sec * fps / clip_frames))
    window_frames = step * clip_frames
    stride_frames = max(1, int(stride_s * fps))
    first_emit = window_frames - 1

    frame_buf: List[np.ndarray] = []
    raw: List[Tuple[float, List[float]]] = []
    batch_clips: List[torch.Tensor] = []
    batch_centers: List[float] = []
    use_amp = device.type == "cuda"
    peak_gpu_bytes = 0

    def flush_batch() -> None:
        nonlocal peak_gpu_bytes
        if not batch_clips:
            return
        clips = torch.stack(batch_clips).float().to(device)
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(clips)
            else:
                logits = model(clips)

        if logits.ndim == 1:
            logits = logits.unsqueeze(1)

        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=1)

        probs_np = probs.detach().cpu().numpy()
        for center_s, class_probs in zip(batch_centers, probs_np):
            raw.append((float(center_s), class_probs.tolist()))
        batch_clips.clear()
        batch_centers.clear()
        if device.type == "cuda":
            peak_gpu_bytes = max(peak_gpu_bytes, torch.cuda.max_memory_allocated(device))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    for frame_idx in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (clip_size, clip_size))
        if len(frame_buf) == window_frames:
            frame_buf.pop(0)
        frame_buf.append(frame)

        if frame_idx >= first_emit and (frame_idx - first_emit) % stride_frames == 0:
            frames = [frame_buf[i * step] for i in range(clip_frames)]
            arr = np.stack(frames).astype(np.float32) / 255.0
            arr = (arr - MEAN) / STD
            clip = torch.from_numpy(arr).permute(3, 0, 1, 2)
            center_sec = (frame_idx - window_frames // 2) / fps
            batch_clips.append(clip)
            batch_centers.append(center_sec)
            if len(batch_clips) >= batch_size:
                flush_batch()

    cap.release()
    flush_batch()
    wall_time_s = time.perf_counter() - start
    return {
        "raw": sorted(raw, key=lambda item: item[0]),
        "video_duration_s": total_frames / fps,
        "wall_time_s": wall_time_s,
        "num_windows": len(raw),
        "peak_gpu_mb": peak_gpu_bytes / (1024 ** 2),
    }


def match_detections(pred_times: Sequence[float], gt_times: Sequence[float], tolerance_s: float) -> MatchCounts:
    gt_used = [False] * len(gt_times)
    tp = 0
    fp = 0
    matched_deltas: List[float] = []

    for pred_t in sorted(pred_times):
        best_idx = None
        best_abs_delta = None
        best_delta = 0.0
        for idx, gt_t in enumerate(gt_times):
            if gt_used[idx]:
                continue
            delta = pred_t - gt_t
            abs_delta = abs(delta)
            if abs_delta <= tolerance_s and (best_abs_delta is None or abs_delta < best_abs_delta):
                best_idx = idx
                best_abs_delta = abs_delta
                best_delta = delta
        if best_idx is None:
            fp += 1
            continue
        gt_used[best_idx] = True
        tp += 1
        matched_deltas.append(best_delta)

    fn = len(gt_times) - tp
    return MatchCounts(tp=tp, fp=fp, fn=fn, matched_deltas_s=matched_deltas)


def params_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def checkpoint_size_mb(checkpoint_path: Path) -> float:
    return checkpoint_path.stat().st_size / (1024 ** 2)


def maybe_compute_gmacs(model: nn.Module, spec: ModelSpec, device: torch.device) -> Optional[float]:
    try:
        from thop import profile  # type: ignore
    except Exception:
        return None

    model = model.to(device)
    dummy = torch.randn(1, 3, spec.clip_frames, spec.clip_size, spec.clip_size, device=device)
    macs, _params = profile(model, inputs=(dummy,), verbose=False)
    return float(macs) / 1e9


def precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if (tp + fp) else 0.0


def recall(tp: int, fn: int) -> float:
    return tp / (tp + fn) if (tp + fn) else 0.0


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


def mean_or_nan(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def half_values(half_arg: str) -> Iterable[int]:
    if half_arg == "1":
        return (1,)
    if half_arg == "2":
        return (2,)
    return (1, 2)


def benchmark_model(
    spec: ModelSpec,
    games: Sequence[str],
    data_dir: Path,
    device: torch.device,
    batch_size: int,
    tolerance_s: float,
    half_arg: str,
    max_video_seconds: Optional[float],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model, resolved_arch, out_features = load_checkpoint_model(spec, device=device)
    gmacs = maybe_compute_gmacs(model, spec, device=device)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_deltas: List[float] = []
    total_windows = 0
    total_wall_time_s = 0.0
    total_video_s = 0.0
    peak_gpu_mb = 0.0
    game_rows: List[Dict[str, Any]] = []

    for game_rel in games:
        labels_path = data_dir / game_rel / "Labels-v2.json"
        events_by_half = parse_annotations(labels_path, goal_class_id=spec.goal_class_id)
        game_tp = 0
        game_fp = 0
        game_fn = 0
        game_video_s = 0.0
        game_wall_s = 0.0
        game_windows = 0
        game_peak_gpu_mb = 0.0
        game_deltas: List[float] = []

        for half in half_values(half_arg):
            video_path = data_dir / game_rel / f"{half}_224p.mkv"
            if not video_path.exists():
                continue

            inf = infer_raw_curve_on_video(
                model=model,
                video_path=video_path,
                device=device,
                clip_sec=spec.clip_sec,
                clip_frames=spec.clip_frames,
                clip_size=spec.clip_size,
                stride_s=spec.stride_s,
                batch_size=batch_size,
                max_video_seconds=max_video_seconds,
            )
            detections = postprocess_goal_predictions(
                raw_curve=inf["raw"],
                goal_class_id=spec.goal_class_id,
                conf_thresh=spec.conf_thresh,
                nms_radius_s=spec.nms_s,
                smooth_k=spec.smooth_k,
                out_features=out_features,
            )
            gt_times = [t for t, _cls in events_by_half.get(half, [])]
            counts = match_detections(
                pred_times=[det["timestamp_s"] for det in detections],
                gt_times=gt_times,
                tolerance_s=tolerance_s,
            )

            game_tp += counts.tp
            game_fp += counts.fp
            game_fn += counts.fn
            game_deltas.extend(counts.matched_deltas_s)
            game_video_s += inf["video_duration_s"]
            game_wall_s += inf["wall_time_s"]
            game_windows += inf["num_windows"]
            game_peak_gpu_mb = max(game_peak_gpu_mb, inf["peak_gpu_mb"])

        total_tp += game_tp
        total_fp += game_fp
        total_fn += game_fn
        all_deltas.extend(game_deltas)
        total_video_s += game_video_s
        total_wall_time_s += game_wall_s
        total_windows += game_windows
        peak_gpu_mb = max(peak_gpu_mb, game_peak_gpu_mb)

        game_precision = precision(game_tp, game_fp)
        game_recall = recall(game_tp, game_fn)
        game_f1 = f1(game_precision, game_recall)
        game_rows.append(
            {
                "model_name": spec.name,
                "architecture": resolved_arch,
                "game_rel": game_rel,
                "tp": game_tp,
                "fp": game_fp,
                "fn": game_fn,
                "precision": round(game_precision, 6),
                "recall": round(game_recall, 6),
                "f1": round(game_f1, 6),
                "matched_mean_delta_s": round(mean_or_nan(game_deltas), 6),
                "video_duration_s": round(game_video_s, 3),
                "wall_time_s": round(game_wall_s, 3),
                "num_windows": game_windows,
                "windows_per_s": round(game_windows / game_wall_s, 4) if game_wall_s else 0.0,
                "realtime_factor": round(game_video_s / game_wall_s, 4) if game_wall_s else 0.0,
                "peak_gpu_mb": round(game_peak_gpu_mb, 2),
            }
        )

    overall_precision = precision(total_tp, total_fp)
    overall_recall = recall(total_tp, total_fn)
    overall_f1 = f1(overall_precision, overall_recall)

    summary_row = {
        "model_name": spec.name,
        "architecture": resolved_arch,
        "checkpoint": str(Path(spec.checkpoint).resolve()),
        "params_million": round(params_count(model) / 1e6, 3),
        "checkpoint_size_mb": round(checkpoint_size_mb(Path(spec.checkpoint)), 2),
        "gmacs_per_clip": None if gmacs is None else round(gmacs, 3),
        "games": len(games),
        "halves": sum(1 for row in game_rows if row["video_duration_s"] > 0),
        "total_video_duration_s": round(total_video_s, 3),
        "total_wall_time_s": round(total_wall_time_s, 3),
        "avg_game_wall_time_s": round(total_wall_time_s / len(games), 3) if games else 0.0,
        "avg_half_windows": round(total_windows / max(1, sum(1 for row in game_rows if row["video_duration_s"] > 0)), 2),
        "latency_ms_per_window": round((total_wall_time_s / total_windows) * 1000.0, 3) if total_windows else 0.0,
        "throughput_windows_per_s": round(total_windows / total_wall_time_s, 3) if total_wall_time_s else 0.0,
        "realtime_factor": round(total_video_s / total_wall_time_s, 3) if total_wall_time_s else 0.0,
        "peak_gpu_mb": round(peak_gpu_mb, 2),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": round(overall_precision, 6),
        "recall": round(overall_recall, 6),
        "f1": round(overall_f1, 6),
        "matched_mean_delta_s": round(mean_or_nan(all_deltas), 6),
        "matched_abs_mean_delta_s": round(mean_or_nan([abs(x) for x in all_deltas]), 6),
        "required_lookahead_s": round(spec.clip_sec / 2.0, 3),
        "estimated_avg_alert_latency_s": round(spec.clip_sec / 2.0 + spec.stride_s / 2.0, 3),
        "estimated_worst_alert_latency_s": round(spec.clip_sec / 2.0 + spec.stride_s, 3),
        "goal_class_id": spec.goal_class_id,
        "conf_thresh": spec.conf_thresh,
        "nms_s": spec.nms_s,
        "smooth_k": spec.smooth_k,
        "clip_sec": spec.clip_sec,
        "clip_frames": spec.clip_frames,
        "clip_size": spec.clip_size,
        "stride_s": spec.stride_s,
    }
    return summary_row, game_rows


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    specs = load_model_specs(args)
    games = select_games(
        data_dir=args.data_dir,
        games_file=args.games_file,
        num_games=args.num_games,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    per_game_rows: List[Dict[str, Any]] = []

    for spec in specs:
        print(f"\n=== Benchmarking {spec.name} ===")
        summary_row, game_rows = benchmark_model(
            spec=spec,
            games=games,
            data_dir=args.data_dir,
            device=device,
            batch_size=args.batch_size,
            tolerance_s=args.tolerance_s,
            half_arg=args.half,
            max_video_seconds=args.max_video_seconds,
        )
        summary_rows.append(summary_row)
        per_game_rows.extend(game_rows)
        print(json.dumps(summary_row, indent=2))

    summary_rows.sort(key=lambda row: (-row["recall"], -row["realtime_factor"], -row["f1"]))
    write_csv(args.out_dir / "model_summary.csv", summary_rows)
    write_csv(args.out_dir / "per_game_metrics.csv", per_game_rows)

    metadata = {
        "data_dir": str(args.data_dir.resolve()),
        "device": str(device),
        "games": games,
        "num_games": len(games),
        "batch_size": args.batch_size,
        "tolerance_s": args.tolerance_s,
        "half": args.half,
        "max_video_seconds": args.max_video_seconds,
        "models": [asdict(spec) for spec in specs],
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nWrote summary to {args.out_dir / 'model_summary.csv'}")
    print(f"Wrote per-game metrics to {args.out_dir / 'per_game_metrics.csv'}")


if __name__ == "__main__":
    main()
