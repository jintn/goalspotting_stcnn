from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18
from tqdm.auto import tqdm

try:
    import cv2
except ImportError:  # pragma: no cover - optional local dependency
    cv2 = None


MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

CLASS_NAMES_STAGE2 = ["non_goal", "goal"]
NUM_CLASSES_STAGE2 = 2

GOAL_LABELS = {"goal"}


@dataclass(frozen=True)
class Stage1Config:
    checkpoint_path: str
    model_name: str = "r2plus1d_18"
    goal_class_id: int = 3
    conf_thresh: float = 0.2
    nms_s: float = 30.0
    smooth_k: int = 5
    tol_s: float = 10.0
    clip_sec: float = 4.0
    clip_frames: int = 16
    clip_size: Tuple[int, int] = (112, 112)
    stride_s: float = 0.5
    num_classes: int = 4


@dataclass(frozen=True)
class Stage2Config:
    clip_sec: float = 16.0
    clip_frames: int = 32
    clip_size: Tuple[int, int] = (112, 112)
    batch_size: int = 16
    epochs: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    dropout_p: float = 0.5
    stage2_thresh: float = 0.5
    stage2_thresh_list: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_type(device: torch.device) -> str:
    return "cuda" if device.type == "cuda" else "cpu"


def parse_annotations(labels_path: str | Path, goal_class_id: int = 3) -> Dict[int, List[Tuple[float, int]]]:
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    events: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
    for ann in data.get("annotations", []):
        label = ann.get("label", "").strip().lower()
        if label not in GOAL_LABELS:
            continue
        half_str, _ = ann["gameTime"].split(" - ")
        half = int(half_str)
        pos_s = int(ann["position"]) / 1000.0
        events[half].append((pos_s, goal_class_id))
    return dict(events)


def get_local_split_games(data_dir: str | Path, split: str) -> List[str]:
    try:
        from SoccerNet.Downloader import getListGames
    except ImportError as exc:
        raise ImportError(
            "SoccerNet is required to resolve official train/valid splits."
        ) from exc

    data_dir = Path(data_dir)
    games = getListGames(split)
    return [g for g in games if (data_dir / g / "Labels-v2.json").exists()]


def make_game_key(game_rel: str, half: int) -> str:
    return f"{game_rel}__H{half}"


def split_game_key(game_key: str) -> Tuple[str, int]:
    game_rel, half_part = game_key.rsplit("__H", 1)
    return game_rel, int(half_part)


def _moving_average(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return arr
    pad = k // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(k, dtype=np.float32) / k, mode="valid")


def postprocess_stage1(
    raw_curve: Sequence[Tuple[float, Sequence[float]]],
    conf_thresh: float,
    nms_radius_s: float,
    smooth_k: int,
    num_classes: int,
) -> List[Dict[str, Any]]:
    if not raw_curve:
        return []

    cs_arr = [x[0] for x in raw_curve]
    all_dets: List[Dict[str, Any]] = []
    for class_id in range(1, num_classes):
        p_arr = np.array([x[1][class_id] for x in raw_curve], dtype=np.float32)
        if smooth_k > 1:
            p_arr = _moving_average(p_arr, smooth_k)
        candidates = sorted(zip(cs_arr, p_arr.tolist()), key=lambda x: -x[1])
        class_dets: List[Dict[str, Any]] = []
        for cs, prob in candidates:
            if prob < conf_thresh:
                continue
            if all(abs(cs - d["timestamp_s"]) >= nms_radius_s for d in class_dets):
                class_dets.append(
                    {"timestamp_s": float(cs), "confidence": float(prob), "class_id": class_id}
                )
        all_dets.extend(class_dets)
    all_dets.sort(key=lambda x: x["timestamp_s"])
    return all_dets


def load_stage1_model(checkpoint_path: str | Path, device: torch.device) -> nn.Module:
    backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt

    in_features = backbone.fc.in_features
    if "fc.weight" in state:
        out_features = state["fc.weight"].shape[0]
        backbone.fc = nn.Linear(in_features, out_features)
    else:
        out_features = state["fc.1.weight"].shape[0]
        dropout_p = 0.4
        backbone.fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_features, out_features))

    backbone.load_state_dict(state)
    backbone.to(device)
    backbone.eval()
    return backbone


def _softmax_probs(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=1).detach().cpu().numpy()


def read_clip_uniform(
    video_path: str | Path,
    center_sec: float,
    clip_sec: float,
    n_frames: int,
    size: Tuple[int, int],
) -> Optional[torch.Tensor]:
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for clip extraction.")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps_v = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        return None

    duration_s = total_frames / fps_v
    half = clip_sec / 2.0
    start_s = max(0.0, center_sec - half)
    end_s = min(duration_s, center_sec + half)

    if end_s <= start_s:
        end_s = min(duration_s, start_s + (1.0 / fps_v))

    frame_positions = np.linspace(start_s * fps_v, max(start_s * fps_v, end_s * fps_v - 1), n_frames)
    frames: List[np.ndarray] = []

    for frame_pos in frame_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(frame_pos)))
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, size)
        frames.append(frame)

    cap.release()

    while len(frames) < n_frames:
        frames.append(frames[-1] if frames else np.zeros((*size, 3), dtype=np.uint8))

    arr = np.stack(frames).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(arr).permute(3, 0, 1, 2)


def _clip_cache_path(
    cache_dir: Path,
    video_path: str,
    center_sec: float,
    clip_sec: float,
    n_frames: int,
    size: Tuple[int, int],
) -> Path:
    key = f"{video_path}|{center_sec:.3f}|{clip_sec:.1f}|{n_frames}|{size[0]}x{size[1]}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.npy"


class Stage2Dataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        clip_sec: float,
        clip_frames: int,
        clip_size: Tuple[int, int],
        cache_dir: Optional[str | Path] = None,
        transform: Optional[Any] = None,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.clip_sec = clip_sec
        self.clip_frames = clip_frames
        self.clip_size = clip_size
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        video_path = str(row["video_path"])
        center_sec = float(row["timestamp_s"])
        label = int(row["label"])

        clip: Optional[torch.Tensor] = None
        if self.cache_dir is not None:
            clip_path = _clip_cache_path(
                self.cache_dir, video_path, center_sec, self.clip_sec, self.clip_frames, self.clip_size
            )
            if clip_path.exists():
                clip = torch.from_numpy(np.load(clip_path))

        if clip is None:
            clip = read_clip_uniform(
                video_path=video_path,
                center_sec=center_sec,
                clip_sec=self.clip_sec,
                n_frames=self.clip_frames,
                size=self.clip_size,
            )
            if clip is None:
                clip = torch.zeros(3, self.clip_frames, *self.clip_size, dtype=torch.float32)
            elif self.cache_dir is not None:
                np.save(clip_path, clip.numpy())

        if self.transform is not None:
            clip = self.transform(clip)

        return clip, label


def infer_stage1_raw_on_video(
    model: nn.Module,
    video_path: str | Path,
    device: torch.device,
    clip_sec: float,
    clip_frames: int,
    clip_size: Tuple[int, int],
    stride_s: float,
    batch_size: int,
) -> List[Tuple[float, List[float]]]:
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for Stage-1 video inference.")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps_v = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        return []

    step = max(1, int(clip_sec * fps_v / clip_frames))
    window_frames = step * clip_frames
    stride_frames = max(1, int(stride_s * fps_v))

    frame_buf: deque[np.ndarray] = deque(maxlen=window_frames)
    raw: List[Tuple[float, List[float]]] = []
    batch_clips: List[torch.Tensor] = []
    batch_centers: List[float] = []
    use_amp = device.type == "cuda"

    def flush_batch() -> None:
        if not batch_clips:
            return
        clips = torch.stack(batch_clips).float().to(device)
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(clips)
            else:
                logits = model(clips)
        probs = _softmax_probs(logits)
        for center_sec, class_probs in zip(batch_centers, probs):
            raw.append((float(center_sec), class_probs.tolist()))
        batch_clips.clear()
        batch_centers.clear()

    first_emit = window_frames - 1

    for frame_idx in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, clip_size)
        frame_buf.append(frame)

        if frame_idx >= first_emit and (frame_idx - first_emit) % stride_frames == 0:
            buf = list(frame_buf)
            frames = [buf[i * step] for i in range(clip_frames)]
            arr = np.stack(frames).astype(np.float32) / 255.0
            arr = (arr - MEAN) / STD
            clip = torch.from_numpy(arr).permute(3, 0, 1, 2)
            center_sec = (frame_idx - window_frames // 2) / fps_v
            batch_clips.append(clip)
            batch_centers.append(center_sec)
            if len(batch_clips) >= batch_size:
                flush_batch()

    cap.release()
    flush_batch()
    return sorted(raw, key=lambda x: x[0])


def build_stage1_raw_curves(
    games: Sequence[str],
    data_dir: str | Path,
    stage1_cfg: Stage1Config,
    device: torch.device,
    batch_size: int = 16,
) -> Tuple[Dict[str, List[Tuple[float, List[float]]]], Dict[str, List[float]]]:
    model = load_stage1_model(stage1_cfg.checkpoint_path, device=device)
    raw_by_key: Dict[str, List[Tuple[float, List[float]]]] = {}
    gt_goal_times: Dict[str, List[float]] = {}
    data_dir = Path(data_dir)

    for game_rel in tqdm(games, desc="Stage1 inference"):
        labels_path = data_dir / game_rel / "Labels-v2.json"
        events = parse_annotations(labels_path, goal_class_id=stage1_cfg.goal_class_id)
        for half in (1, 2):
            video_path = data_dir / game_rel / f"{half}_224p.mkv"
            if not video_path.exists():
                continue
            game_key = make_game_key(game_rel, half)
            raw_curve = infer_stage1_raw_on_video(
                model=model,
                video_path=video_path,
                device=device,
                clip_sec=stage1_cfg.clip_sec,
                clip_frames=stage1_cfg.clip_frames,
                clip_size=stage1_cfg.clip_size,
                stride_s=stage1_cfg.stride_s,
                batch_size=batch_size,
            )
            raw_by_key[game_key] = raw_curve
            gt_goal_times[game_key] = [t for t, _ in events.get(half, [])]
    return raw_by_key, gt_goal_times


def save_raw_curves_pickle(
    path: str | Path,
    raw_by_key: Mapping[str, Any],
    gt_goal_times: Mapping[str, Any],
) -> None:
    import pickle

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"all_raw": dict(raw_by_key), "gt_goal_times": dict(gt_goal_times)}, f)


def _normalize_pickle_game_key(raw_key: Any) -> str:
    if isinstance(raw_key, str):
        if "__H" in raw_key:
            return raw_key
        return raw_key

    if isinstance(raw_key, tuple) and len(raw_key) == 2:
        game_part, half = raw_key
        game_rel = str(game_part)
        prefixes = ["/home/jinny/aspotting/dataset/", "/content/drive/MyDrive/aspotting/dataset/"]
        for prefix in prefixes:
            if game_rel.startswith(prefix):
                game_rel = game_rel[len(prefix) :]
                break
        return make_game_key(game_rel, int(half))

    raise ValueError(f"Unsupported pickle key format: {raw_key!r}")


def load_raw_curves_pickle(path: str | Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    import pickle

    with open(path, "rb") as f:
        payload = pickle.load(f)

    if "gt_goal_times" in payload:
        raw_by_key = {_normalize_pickle_game_key(k): v for k, v in payload["all_raw"].items()}
        gt_goal_times = {_normalize_pickle_game_key(k): v for k, v in payload["gt_goal_times"].items()}
        return raw_by_key, gt_goal_times

    if "all_raw" in payload and "all_gt" in payload:
        raw_by_key = {_normalize_pickle_game_key(k): v for k, v in payload["all_raw"].items()}
        goal_gt_src = payload["all_gt"].get(3, {})
        gt_goal_times = {_normalize_pickle_game_key(k): v for k, v in goal_gt_src.items()}
        return raw_by_key, gt_goal_times

    raise KeyError("Pickle must contain either {'all_raw','gt_goal_times'} or {'all_raw','all_gt'}.")


def filter_pickle_to_games(
    raw_by_key: Mapping[str, Any],
    gt_goal_times: Mapping[str, Any],
    games: Sequence[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    game_set = set(games)
    raw_out = {}
    gt_out = {}
    for game_key, raw_curve in raw_by_key.items():
        game_rel, _half = split_game_key(game_key)
        if game_rel in game_set:
            raw_out[game_key] = raw_curve
            gt_out[game_key] = list(gt_goal_times.get(game_key, []))
    return raw_out, gt_out


def build_stage2_manifest(
    raw_by_key: Mapping[str, Sequence[Tuple[float, Sequence[float]]]],
    gt_goal_times: Mapping[str, Sequence[float]],
    data_dir: str | Path,
    stage1_cfg: Stage1Config,
    split_name: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    data_dir = Path(data_dir)

    for game_key, raw_curve in tqdm(raw_by_key.items(), desc=f"Stage2 manifest {split_name}"):
        goal_candidates = [
            d
            for d in postprocess_stage1(
                raw_curve=raw_curve,
                conf_thresh=stage1_cfg.conf_thresh,
                nms_radius_s=stage1_cfg.nms_s,
                smooth_k=stage1_cfg.smooth_k,
                num_classes=stage1_cfg.num_classes,
            )
            if d["class_id"] == stage1_cfg.goal_class_id
        ]
        game_rel, half = split_game_key(game_key)
        video_path = data_dir / game_rel / f"{half}_224p.mkv"
        gt_times = list(gt_goal_times.get(game_key, []))
        for det in goal_candidates:
            t = float(det["timestamp_s"])
            conf = float(det["confidence"])
            label = int(any(abs(t - g) <= stage1_cfg.tol_s for g in gt_times))
            rows.append(
                {
                    "split": split_name,
                    "game_key": game_key,
                    "game_rel": game_rel,
                    "half": half,
                    "video_path": str(video_path),
                    "timestamp_s": t,
                    "stage1_confidence": conf,
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def save_manifest(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_stage2_model(dropout_p: float, num_classes: int = 2) -> nn.Module:
    model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    model.fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(model.fc.in_features, num_classes))
    return model


def make_class_weights(labels: Sequence[int], num_classes: int = 2) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    total = counts.sum()
    weights = total / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def make_stage2_loaders(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    stage2_cfg: Stage2Config,
    cache_dir: Optional[str | Path] = None,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = Stage2Dataset(
        train_df,
        clip_sec=stage2_cfg.clip_sec,
        clip_frames=stage2_cfg.clip_frames,
        clip_size=stage2_cfg.clip_size,
        cache_dir=cache_dir,
    )
    valid_ds = Stage2Dataset(
        valid_df,
        clip_sec=stage2_cfg.clip_sec,
        clip_frames=stage2_cfg.clip_frames,
        clip_size=stage2_cfg.clip_size,
        cache_dir=cache_dir,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=stage2_cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=stage2_cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, valid_loader


def _epoch_pass(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss = 0.0
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_goal_probs: List[float] = []

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for clips, labels in tqdm(loader, leave=False):
        clips = clips.float().to(device)
        labels = labels.long().to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(clips)
                    loss = criterion(logits, labels)
            else:
                logits = model(clips)
                loss = criterion(logits, labels)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        total_loss += loss.item() * len(labels)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_goal_probs.extend(probs[:, 1].detach().cpu().tolist())

    avg_loss = total_loss / max(1, len(all_labels))
    accuracy = accuracy_score(all_labels, all_preds)
    goal_precision = precision_score(all_labels, all_preds, labels=[1], average="binary", zero_division=0)
    goal_recall = recall_score(all_labels, all_preds, labels=[1], average="binary", zero_division=0)
    goal_f1 = f1_score(all_labels, all_preds, labels=[1], average="binary", zero_division=0)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "goal_precision": goal_precision,
        "goal_recall": goal_recall,
        "goal_f1": goal_f1,
        "labels": all_labels,
        "preds": all_preds,
        "goal_probs": all_goal_probs,
    }


def train_stage2(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    stage2_cfg: Stage2Config,
    output_dir: str | Path,
    cache_dir: Optional[str | Path] = None,
    device: Optional[torch.device] = None,
    num_workers: int = 2,
) -> Tuple[nn.Module, Dict[str, Any], Path]:
    device = device or get_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, valid_loader = make_stage2_loaders(
        train_df=train_df,
        valid_df=valid_df,
        stage2_cfg=stage2_cfg,
        cache_dir=cache_dir,
        num_workers=num_workers,
    )

    model = build_stage2_model(dropout_p=stage2_cfg.dropout_p).to(device)
    class_weights = make_class_weights(train_df["label"].tolist()).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=stage2_cfg.lr, weight_decay=stage2_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage2_cfg.epochs)

    best_goal_recall = -1.0
    best_goal_f1 = -1.0
    best_ckpt_path = output_dir / "stage2_best.pt"
    history: List[Dict[str, Any]] = []

    for epoch in range(1, stage2_cfg.epochs + 1):
        tr = _epoch_pass(model, train_loader, criterion, device, optimizer=optimizer)
        va = _epoch_pass(model, valid_loader, criterion, device, optimizer=None)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": tr["loss"],
                "train_accuracy": tr["accuracy"],
                "train_goal_precision": tr["goal_precision"],
                "train_goal_recall": tr["goal_recall"],
                "train_goal_f1": tr["goal_f1"],
                "val_loss": va["loss"],
                "val_accuracy": va["accuracy"],
                "val_goal_precision": va["goal_precision"],
                "val_goal_recall": va["goal_recall"],
                "val_goal_f1": va["goal_f1"],
            }
        )

        should_save = False
        if va["goal_recall"] > best_goal_recall:
            should_save = True
        elif math.isclose(va["goal_recall"], best_goal_recall) and va["goal_f1"] > best_goal_f1:
            should_save = True

        if should_save:
            best_goal_recall = va["goal_recall"]
            best_goal_f1 = va["goal_f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "stage2_cfg": stage2_cfg.__dict__,
                    "class_names": CLASS_NAMES_STAGE2,
                    "best_val_goal_recall": best_goal_recall,
                    "best_val_goal_f1": best_goal_f1,
                },
                best_ckpt_path,
            )

        print(
            f"Epoch {epoch:02d}/{stage2_cfg.epochs}  "
            f"train loss={tr['loss']:.4f} acc={tr['accuracy']:.4f} goalF1={tr['goal_f1']:.4f}  |  "
            f"val loss={va['loss']:.4f} acc={va['accuracy']:.4f} "
            f"goalP={va['goal_precision']:.4f} goalR={va['goal_recall']:.4f} goalF1={va['goal_f1']:.4f}"
        )

    model.load_state_dict(torch.load(best_ckpt_path, map_location=device)["model_state_dict"])
    history_df = pd.DataFrame(history)
    return model, {"history": history_df, "class_weights": class_weights.cpu().numpy()}, best_ckpt_path


def load_stage2_checkpoint(
    checkpoint_path: str | Path,
    device: Optional[torch.device] = None,
) -> nn.Module:
    device = device or get_device()
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("stage2_cfg", {})
    dropout_p = cfg.get("dropout_p", 0.5)
    model = build_stage2_model(dropout_p=dropout_p).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def add_stage2_scores(
    df: pd.DataFrame,
    model: nn.Module,
    stage2_cfg: Stage2Config,
    device: Optional[torch.device] = None,
    batch_size: Optional[int] = None,
    cache_dir: Optional[str | Path] = None,
    num_workers: int = 2,
) -> pd.DataFrame:
    device = device or get_device()
    batch_size = batch_size or stage2_cfg.batch_size

    ds = Stage2Dataset(
        df,
        clip_sec=stage2_cfg.clip_sec,
        clip_frames=stage2_cfg.clip_frames,
        clip_size=stage2_cfg.clip_size,
        cache_dir=cache_dir,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    all_goal_probs: List[float] = []
    model.eval()
    with torch.no_grad():
        for clips, _ in tqdm(loader, desc="Stage2 scoring", leave=False):
            clips = clips.float().to(device)
            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(clips)
            else:
                logits = model(clips)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_goal_probs.extend(probs.cpu().tolist())

    out = df.copy()
    out["stage2_goal_prob"] = all_goal_probs
    return out


def _group_preds(
    df: pd.DataFrame,
    score_col: str,
    threshold: float,
) -> Dict[str, List[Tuple[float, float]]]:
    grouped: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    if df.empty:
        return grouped
    keep = df[df[score_col] >= threshold]
    for row in keep.itertuples(index=False):
        grouped[row.game_key].append((float(row.timestamp_s), float(getattr(row, score_col))))
    for game_key in grouped:
        grouped[game_key].sort(key=lambda x: x[0])
    return grouped


def _match_predictions(
    preds: Sequence[Tuple[float, float]],
    gt_times: Sequence[float],
    tol_s: float,
) -> Tuple[int, int, int]:
    matched_gt: set[int] = set()
    tp = 0
    fp = 0
    for pred_t, _ in sorted(preds, key=lambda x: -x[1]):
        best_idx = None
        best_dist = tol_s + 1e-9
        for idx, gt_t in enumerate(gt_times):
            if idx in matched_gt:
                continue
            dist = abs(pred_t - gt_t)
            if dist <= tol_s and dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is None:
            fp += 1
        else:
            matched_gt.add(best_idx)
            tp += 1
    fn = max(0, len(gt_times) - tp)
    return tp, fp, fn


def compute_ap(
    pred_map: Mapping[str, Sequence[Tuple[float, float]]],
    gt_map: Mapping[str, Sequence[float]],
    tol_s: float,
) -> float:
    scored_preds: List[Tuple[str, float, float]] = []
    for game_key, preds in pred_map.items():
        for t, conf in preds:
            scored_preds.append((game_key, t, conf))
    scored_preds.sort(key=lambda x: -x[2])

    total_gt = sum(len(v) for v in gt_map.values())
    if total_gt == 0:
        return float("nan")

    matched: Dict[str, set[int]] = defaultdict(set)
    tp_flags: List[int] = []
    fp_flags: List[int] = []

    for game_key, pred_t, _conf in scored_preds:
        gt_times = list(gt_map.get(game_key, []))
        best_idx = None
        best_dist = tol_s + 1e-9
        for idx, gt_t in enumerate(gt_times):
            if idx in matched[game_key]:
                continue
            dist = abs(pred_t - gt_t)
            if dist <= tol_s and dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is None:
            tp_flags.append(0)
            fp_flags.append(1)
        else:
            matched[game_key].add(best_idx)
            tp_flags.append(1)
            fp_flags.append(0)

    cum_tp = np.cumsum(tp_flags)
    cum_fp = np.cumsum(fp_flags)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)
    recall = cum_tp / total_gt

    interp_prec = precision.copy()
    for i in range(len(interp_prec) - 2, -1, -1):
        interp_prec[i] = max(interp_prec[i], interp_prec[i + 1])

    recall_with_zero = np.concatenate(([0.0], recall))
    interp_with_zero = np.concatenate(([interp_prec[0] if len(interp_prec) else 0.0], interp_prec))
    change = np.where(recall_with_zero[1:] != recall_with_zero[:-1])[0]
    ap = float(np.sum((recall_with_zero[change + 1] - recall_with_zero[change]) * interp_with_zero[change + 1]))
    return ap


def evaluate_goal_predictions(
    pred_map: Mapping[str, Sequence[Tuple[float, float]]],
    gt_map: Mapping[str, Sequence[float]],
    tol_s: float = 10.0,
) -> Dict[str, float]:
    tp_total = 0
    fp_total = 0
    fn_total = 0

    all_keys = sorted(set(gt_map) | set(pred_map))
    for game_key in all_keys:
        tp, fp, fn = _match_predictions(pred_map.get(game_key, []), gt_map.get(game_key, []), tol_s)
        tp_total += tp
        fp_total += fp
        fn_total += fn

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    match_count = len({split_game_key(k)[0] for k in all_keys}) if all_keys else 0
    ap10 = compute_ap(pred_map, gt_map, tol_s=10.0)
    ap5 = compute_ap(pred_map, gt_map, tol_s=5.0)
    ap30 = compute_ap(pred_map, gt_map, tol_s=30.0)
    ap60 = compute_ap(pred_map, gt_map, tol_s=60.0)
    map_score = float(np.nanmean([ap5, ap10, ap30, ap60]))

    return {
        "goal_recall@10s": recall,
        "goal_precision@10s": precision,
        "goal_f1@10s": f1,
        "goal_FP": float(fp_total),
        "goal_FP_per_match": float(fp_total / match_count) if match_count > 0 else float("nan"),
        "goal_AP@10s": ap10,
        "mAP": map_score,
        "TP": float(tp_total),
        "FN": float(fn_total),
        "matches": float(match_count),
    }


def evaluate_stage1_vs_stage2(
    candidates_df: pd.DataFrame,
    gt_map: Mapping[str, Sequence[float]],
    stage2_thresh_list: Sequence[float],
    stage1_score_col: str = "stage1_confidence",
    stage2_score_col: str = "stage2_goal_prob",
    tol_s: float = 10.0,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    stage1_pred_map = _group_preds(candidates_df, score_col=stage1_score_col, threshold=-1.0)
    stage1_metrics = evaluate_goal_predictions(stage1_pred_map, gt_map, tol_s=tol_s)
    rows.append({"system": "stage1_only", "stage2_thresh": np.nan, **stage1_metrics})

    for thresh in stage2_thresh_list:
        pred_map = _group_preds(candidates_df, score_col=stage2_score_col, threshold=thresh)
        metrics = evaluate_goal_predictions(pred_map, gt_map, tol_s=tol_s)
        rows.append({"system": "stage1_plus_stage2", "stage2_thresh": thresh, **metrics})

    return pd.DataFrame(rows)


def select_best_stage2_threshold(
    eval_df: pd.DataFrame,
    recall_targets: Sequence[float] = (0.95, 0.90),
) -> pd.DataFrame:
    stage2_df = eval_df[eval_df["system"] == "stage1_plus_stage2"].copy()
    picked: List[pd.Series] = []
    for target in recall_targets:
        valid = stage2_df[stage2_df["goal_recall@10s"] >= target]
        if valid.empty:
            continue
        best = valid.sort_values(
            ["goal_FP_per_match", "goal_precision@10s", "goal_f1@10s"],
            ascending=[True, False, False],
        ).iloc[0]
        picked.append(best)
    if not picked:
        return pd.DataFrame(columns=eval_df.columns)
    return pd.DataFrame(picked).drop_duplicates(subset=["stage2_thresh"])
