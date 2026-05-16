"""
Build a jitter-augmented manifest and clip cache for 4-class training.

Jitter adds ±2 s offsets to goal clips (train split only), tripling
goal coverage. Shots and background are sampled the same way as the
non-jitter manifest.

Outputs (never overwrites existing paths):
  manifests/4class_jitter/train_clips.csv
  manifests/4class_jitter/valid_clips.csv
  clip_cache/4class_jitter/<md5>.npy  (one file per unique clip)
"""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path("/home/jinny/aspotting")
DATA_DIR     = PROJECT_DIR / "dataset"
MANIFEST_DIR = PROJECT_DIR / "manifests" / "4class_jitter_new"
CACHE_DIR    = PROJECT_DIR / "clip_cache" / "4class_jitter_new"

RANDOM_SEED = 42

FPS         = 25
CLIP_SEC    = 4.0
CLIP_FRAMES = 16
CLIP_SIZE   = (112, 112)

NUM_CLASSES = 4
CLASS_NAMES = ["background", "shots_off", "shots_on", "goal"]

SHOT_ON_LABELS  = {"shots on target", "penalty"}
SHOT_OFF_LABELS = {"shots off target"}

SHOTS_ON_PER_GOAL    = 2
SHOTS_OFF_PER_GOAL   = 2
NEG_RAND_PER_GOAL    = 4
NEG_PER_NO_GOAL_HALF = 12
MIN_NEG_FROM_GOAL_SEC = 12.0

JITTER_OFFSETS = (-2.0, 0.0, 2.0)   # applied to goal clips, train only

MEAN = np.array([0.43216, 0.394666, 0.37645],  dtype=np.float32)
STD  = np.array([0.22803, 0.22145,  0.216989], dtype=np.float32)

# ── Safety check ──────────────────────────────────────────────────────────────
for path, label in [(MANIFEST_DIR, "manifest dir"), (CACHE_DIR, "cache dir")]:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(
            f"{label} already exists and is not empty: {path}\n"
            "Delete it manually or choose a different output path."
        )
    path.mkdir(parents=True, exist_ok=True)

print(f"Manifest dir : {MANIFEST_DIR}")
print(f"Cache dir    : {CACHE_DIR}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_annotations(game_rel_path):
    json_path = DATA_DIR / game_rel_path / "Labels-v2.json"
    with open(json_path) as f:
        data = json.load(f)
    events = defaultdict(list)
    for ann in data["annotations"]:
        label = ann.get("label", "").strip().lower()
        half_str, _ = ann["gameTime"].split(" - ")
        half  = int(half_str)
        pos_s = int(ann["position"]) / 1000.0
        if label == "goal":
            events[half].append((pos_s, 3))
        elif label in SHOT_ON_LABELS:
            events[half].append((pos_s, 2))
        elif label in SHOT_OFF_LABELS:
            events[half].append((pos_s, 1))
    return dict(events)


def far_from_all(t, times, min_gap):
    return all(abs(t - x) >= min_gap for x in times)


def read_clip(video_path, center_sec):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps_v = cap.get(cv2.CAP_PROP_FPS) or FPS

    start_sec   = max(0.0, center_sec - CLIP_SEC / 2)
    start_frame = int(start_sec * fps_v)
    step        = max(1, int(CLIP_SEC * fps_v / CLIP_FRAMES))

    frames = []
    for i in range(CLIP_FRAMES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i * step)
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, CLIP_SIZE)
        frames.append(frame)
    cap.release()

    while len(frames) < CLIP_FRAMES:
        frames.append(frames[-1] if frames else np.zeros((*CLIP_SIZE, 3), np.uint8))

    arr = np.stack(frames).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(3, 0, 1, 2)   # (C, T, H, W)


def cache_path(video_path, center_sec):
    key = f"{video_path}_{center_sec:.4f}"
    h   = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{h}.npy"


def build_sample_list(game_list, is_train):
    samples = []
    jitters = JITTER_OFFSETS if is_train else (0.0,)

    for game in tqdm(game_list, desc=f"{'train' if is_train else 'valid'} manifest"):
        events = parse_annotations(game)
        for half in (1, 2):
            vid = DATA_DIR / game / f"{half}_224p.mkv"
            if not vid.exists():
                continue
            cap   = cv2.VideoCapture(str(vid))
            fps_v = cap.get(cv2.CAP_PROP_FPS) or FPS
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            dur = total / fps_v
            if dur < CLIP_SEC + 1:
                continue

            lo = CLIP_SEC / 2
            hi = dur - CLIP_SEC / 2

            half_events     = events.get(half, [])
            goals           = [t for t, c in half_events if c == 3]
            shots_on        = [t for t, c in half_events if c == 2]
            shots_off       = [t for t, c in half_events if c == 1]
            all_event_times = [t for t, _ in half_events]

            # Goals — jitter only on train
            for g in goals:
                for j in jitters:
                    c = float(np.clip(g + j, lo, hi))
                    samples.append((str(vid), c, 3, "goal"))

            # Shots on target
            cands = [t for t in shots_on if far_from_all(t, goals, MIN_NEG_FROM_GOAL_SEC)]
            n_on  = min(len(cands), SHOTS_ON_PER_GOAL * max(1, len(goals)))
            for t in (random.sample(cands, k=n_on) if n_on > 0 else []):
                samples.append((str(vid), float(np.clip(t, lo, hi)), 2, "shots_on"))

            # Shots off target
            cands  = [t for t in shots_off if far_from_all(t, goals, MIN_NEG_FROM_GOAL_SEC)]
            n_off  = min(len(cands), SHOTS_OFF_PER_GOAL * max(1, len(goals)))
            for t in (random.sample(cands, k=n_off) if n_off > 0 else []):
                samples.append((str(vid), float(np.clip(t, lo, hi)), 1, "shots_off"))

            # Background
            n_rand    = NEG_RAND_PER_GOAL * len(goals) if goals else NEG_PER_NO_GOAL_HALF
            forbidden = list(all_event_times)
            added = attempts = 0
            while added < n_rand and attempts < n_rand * 40:
                attempts += 1
                t = random.uniform(lo, hi)
                if far_from_all(t, forbidden, MIN_NEG_FROM_GOAL_SEC):
                    samples.append((str(vid), t, 0, "rand"))
                    forbidden.append(t)
                    added += 1

    return samples


# ── SoccerNet splits ──────────────────────────────────────────────────────────
try:
    from SoccerNet.Downloader import getListGames
    all_splits = {}
    for split in ("train", "valid"):
        games = getListGames(split)
        local = [g for g in games if (DATA_DIR / g / "Labels-v2.json").exists()]
        all_splits[split] = local
        print(f"{split:5s}  total={len(games):3d}  local={len(local):3d}")
except ImportError:
    raise ImportError("Install soccernet: pip install SoccerNet")

# ── Build manifests ───────────────────────────────────────────────────────────
random.seed(RANDOM_SEED)

print("\nBuilding train manifest (with jitter) …")
train_samples = build_sample_list(all_splits["train"], is_train=True)

print("Building valid manifest (no jitter) …")
valid_samples = build_sample_list(all_splits["valid"], is_train=False)

# Write CSVs
header = "video_path,center_sec,label,event_type\n"
for name, samples in [("train", train_samples), ("valid", valid_samples)]:
    csv_path = MANIFEST_DIR / f"{name}_clips.csv"
    with open(csv_path, "w") as f:
        f.write(header)
        for vid, sec, lbl, etype in samples:
            f.write(f"{vid},{sec:.6f},{lbl},{etype}\n")
    print(f"Saved {csv_path}  ({len(samples):,} rows)")

# Class distribution
for name, samples in [("train", train_samples), ("valid", valid_samples)]:
    print(f"\n{name} class distribution:")
    for cid, cname in enumerate(CLASS_NAMES):
        n = sum(1 for _, _, l, _ in samples if l == cid)
        print(f"  {cname:12s} (class {cid}): {n:,}")

# ── Pre-populate cache ────────────────────────────────────────────────────────
# Deduplicate — same (video, center_sec) may appear in both splits
all_clips = {(vid, sec) for vid, sec, _, _ in train_samples + valid_samples}
print(f"\nUnique clips to cache: {len(all_clips):,}")

skipped = written = errors = 0
for vid, sec in tqdm(all_clips, desc="Caching clips"):
    dst = cache_path(vid, sec)
    if dst.exists():
        skipped += 1
        continue
    arr = read_clip(vid, sec)
    if arr is None:
        errors += 1
        continue
    np.save(str(dst), arr)
    written += 1

print(f"\nDone — written={written:,}  skipped={skipped:,}  errors={errors:,}")
print(f"Cache : {CACHE_DIR}")
print(f"Manifest: {MANIFEST_DIR}")
