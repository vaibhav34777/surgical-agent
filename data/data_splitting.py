"""
Usage:
  python data_splitting.py --dataset_root /path/to/dataset --output_dir /path/to/output
"""

import argparse
import json
import random
import numpy as np
import multiprocessing
from pathlib import Path
from PIL import Image
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


NUM_WORKERS  = multiprocessing.cpu_count()
SEED         = 42

CLASS_NAMES = {
    0:  "Black Background",
    1:  "Abdominal Wall",
    2:  "Liver",
    3:  "Gastrointestinal Tract",
    4:  "Fat",
    5:  "Grasper",
    6:  "Connective Tissue",
    8:  "Cystic Duct",
    9:  "L-Hook Electrocautery",
    11: "Hepatic Vein",
}

EVAL_CLASSES = sorted(CLASS_NAMES.keys())

COLOR_TO_CLASS = {
    (127, 127, 127): 0,
    (255, 114, 114): 1,
    (210, 140, 140): 2,
    (255, 255, 255): 3,
    (186, 183, 75):  4,
    (170, 255, 0):   5,
    (255, 160, 165): 6,
    (169, 255, 184): 8,
    (231, 70,  156): 9,
    (0,   50,  128): 11,
    (255, 85,  0):   11,
}


def collect_pairs(root: Path):
    pairs = []
    for video_dir in sorted(root.iterdir()):
        if not video_dir.is_dir():
            continue
        video_name = video_dir.name
        for clip_dir in sorted(video_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            frames = sorted([
                f for f in clip_dir.iterdir()
                if f.suffix == ".png"
                and "_mask" not in f.name
                and "_watershed" not in f.name
            ])
            for frame_path in frames:
                color_mask_path = clip_dir / (frame_path.stem + "_color_mask.png")
                if color_mask_path.exists():
                    pairs.append({
                        "frame": str(frame_path),
                        "mask":  str(color_mask_path),
                        "video": video_name,
                        "clip":  clip_dir.name,
                    })
    return pairs


def analyze_mask(item):
    mask_rgb = np.array(Image.open(item["mask"]).convert("RGB"))
    flat     = mask_rgb.reshape(-1, 3)
    px       = defaultdict(int)
    for (r, g, b), cls_id in COLOR_TO_CLASS.items():
        match = (flat[:, 0] == r) & (flat[:, 1] == g) & (flat[:, 2] == b)
        count = int(match.sum())
        if count > 0:
            px[cls_id] += count
    return {"video": item["video"], "px": dict(px)}


def get_video_stats(pairs):
    video_pixel_counts = defaultdict(lambda: defaultdict(int))
    video_frame_counts = defaultdict(lambda: defaultdict(int))
    video_total_frames = defaultdict(int)

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(analyze_mask, item): item for item in pairs}
        pbar    = tqdm(as_completed(futures), total=len(futures), desc="Profiling videos")
        for future in pbar:
            result = future.result()
            video  = result["video"]
            video_total_frames[video] += 1
            for cls_id, px in result["px"].items():
                video_pixel_counts[video][cls_id] += px
                video_frame_counts[video][cls_id] += 1

    return dict(video_pixel_counts), dict(video_frame_counts), dict(video_total_frames)


def score_split(test_videos, all_videos, video_pixel_counts, video_total_frames):
    train_videos = [v for v in all_videos if v not in test_videos]
    train_px = defaultdict(int)
    test_px  = defaultdict(int)
    train_fr = 0
    test_fr  = 0

    for v in train_videos:
        train_fr += video_total_frames.get(v, 0)
        for cls_id, px in video_pixel_counts.get(v, {}).items():
            train_px[cls_id] += px

    for v in test_videos:
        test_fr += video_total_frames.get(v, 0)
        for cls_id, px in video_pixel_counts.get(v, {}).items():
            test_px[cls_id] += px

    if train_fr == 0 or test_fr == 0:
        return float("inf"), {}

    absent_penalty = 0
    dist_score     = 0.0

    for cls_id in EVAL_CLASSES:
        train_pct = train_px.get(cls_id, 0) / train_fr
        test_pct  = test_px.get(cls_id, 0)  / test_fr
        if test_pct == 0 and train_pct > 0:
            absent_penalty += 1000
        if train_pct + test_pct > 0:
            dist_score += abs(train_pct - test_pct) / max(train_pct, test_pct)

    return absent_penalty + dist_score, {
        "train_px": dict(train_px),
        "test_px":  dict(test_px),
        "train_fr": train_fr,
        "test_fr":  test_fr,
    }


def find_best_split(all_videos, video_pixel_counts, video_total_frames,
                    n_test=2, n_trials=500):
    random.seed(SEED)
    best_score    = float("inf")
    best_test_set = None
    best_info     = None

    for _ in tqdm(range(n_trials), desc="Searching splits"):
        test_videos = set(random.sample(all_videos, n_test))
        score, info = score_split(test_videos, all_videos, video_pixel_counts, video_total_frames)
        if score < best_score:
            best_score    = score
            best_test_set = test_videos
            best_info     = info

    return best_test_set, best_score, best_info


def print_split_report(best_test_set, all_videos, best_info, video_pixel_counts):
    train_videos = sorted([v for v in all_videos if v not in best_test_set])
    test_videos  = sorted(best_test_set)
    train_fr     = best_info["train_fr"]
    test_fr      = best_info["test_fr"]

    print(f"\n{'='*80}")
    print(f"  BEST STRATIFIED SPLIT FOUND")
    print(f"{'='*80}")
    print(f"  Train videos ({len(train_videos)}): {train_videos}")
    print(f"  Test  videos ({len(test_videos)}):  {test_videos}")
    print(f"  Train frames: {train_fr:,}  |  Test frames: {test_fr:,}")
    print(f"{'='*80}")
    print(f"  {'Cls':<5} {'Class Name':<26} {'Train %px':>10} {'Test %px':>10} {'Diff':>8}  {'Note':>18}")
    print(f"  {'-'*78}")

    for cls_id in EVAL_CLASSES:
        name   = CLASS_NAMES[cls_id]
        tr_pct = 100.0 * best_info["train_px"].get(cls_id, 0) / train_fr if train_fr else 0
        te_pct = 100.0 * best_info["test_px"].get(cls_id, 0)  / test_fr  if test_fr  else 0
        diff   = abs(tr_pct - te_pct)
        note   = "<< ABSENT IN TEST" if te_pct == 0 and tr_pct > 0 else ""
        print(f"  [{cls_id:2d}]  {name:<26} {tr_pct:>9.3f}% {te_pct:>9.3f}% {diff:>7.3f}%  {note}")

    print(f"{'='*80}")
    print(f"\n  Per-video class presence in TEST videos:")
    for v in test_videos:
        present = sorted(video_pixel_counts.get(v, {}).keys())
        absent  = [CLASS_NAMES[c] for c in EVAL_CLASSES if c not in present]
        print(f"  {v:<15}  present: {present}")
        if absent:
            print(f"  {'':15}  ABSENT : {absent}")


def plot_split_comparison(best_info, output_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 6))
    x       = np.arange(len(EVAL_CLASSES))
    w       = 0.4
    train_fr = best_info["train_fr"]
    test_fr  = best_info["test_fr"]
    tr = [100.0 * best_info["train_px"].get(c, 0) / train_fr for c in EVAL_CLASSES]
    te = [100.0 * best_info["test_px"].get(c, 0)  / test_fr  for c in EVAL_CLASSES]
    ax.bar(x - w/2, tr, w, label="Train", color="#3498db", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, te, w, label="Test",  color="#e74c3c", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"[{c}]\n{CLASS_NAMES[c][:12]}" for c in EVAL_CLASSES],
                       fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("pixels per frame (normalized %)")
    ax.set_title("Train vs Test Class Distribution — Best Stratified Split", fontsize=12)
    ax.legend()
    plt.tight_layout()
    out = output_dir / "split_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")


def save_splits(pairs, best_test_set, output_dir: Path):
    train_pairs = [p for p in pairs if p["video"] not in best_test_set]
    test_pairs  = [p for p in pairs if p["video"] in best_test_set]
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train.json", "w") as f:
        json.dump(train_pairs, f, indent=2)
    with open(output_dir / "test.json", "w") as f:
        json.dump(test_pairs, f, indent=2)
    print(f"\nSaved {len(train_pairs)} train and {len(test_pairs)} test pairs to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train and test sets")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to the dataset root directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save train.json and test.json")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    print("Collecting frame-mask pairs...")
    pairs      = collect_pairs(dataset_root)
    all_videos = sorted({p["video"] for p in pairs})
    print(f"Found {len(pairs)} pairs across {len(all_videos)} videos: {all_videos}")

    print(f"\nProfiling all {len(all_videos)} videos with {NUM_WORKERS} workers...")
    video_pixel_counts, video_frame_counts, video_total_frames = get_video_stats(pairs)

    print(f"\nSearching for best stratified split across 500 random trials...")
    best_test_set, best_score, best_info = find_best_split(
        all_videos, video_pixel_counts, video_total_frames, n_test=2, n_trials=500
    )

    print_split_report(best_test_set, all_videos, best_info, video_pixel_counts)
    plot_split_comparison(best_info, output_dir)
    save_splits(pairs, best_test_set, output_dir)


if __name__ == "__main__":
    main()