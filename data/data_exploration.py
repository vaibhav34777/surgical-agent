"""
Usage:
  python data_exploration.py --train_json /path/to/train.json --test_json /path/to/test.json --output_dir /path/to/output
"""

import argparse
import json
import numpy as np
import multiprocessing
from pathlib import Path
from PIL import Image
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


NUM_WORKERS = multiprocessing.cpu_count()

CLASS_NAMES = {
    0:  "Black Background",
    1:  "Abdominal Wall",
    2:  "Liver",
    3:  "Gastrointestinal Tract",
    4:  "Fat",
    5:  "Grasper",
    6:  "Connective Tissue",
    7:  "Blood",
    8:  "Cystic Duct",
    9:  "L-Hook Electrocautery",
    10: "Gallbladder",
    11: "Hepatic Vein",
}

COLOR_TO_CLASS = {
    (127, 127, 127): 0,
    (255, 114, 114): 1,
    (210, 140, 140): 2,
    (255, 255, 255): 3,
    (186, 183, 75):  4,
    (170, 255, 0):   5,
    (255, 160, 165): 6,
    (111, 74,  0):   7,
    (169, 255, 184): 8,
    (231, 70,  156): 9,
    (0,   50,  128): 10,
    (255, 85,  0):   11,
}

NUM_CLASSES = len(CLASS_NAMES)


def analyze_mask(item):
    mask_rgb  = np.array(Image.open(item["mask"]).convert("RGB"))
    flat      = mask_rgb.reshape(-1, 3)
    total_px  = flat.shape[0]
    class_px  = defaultdict(int)
    for (r, g, b), cls_id in COLOR_TO_CLASS.items():
        match = (flat[:, 0] == r) & (flat[:, 1] == g) & (flat[:, 2] == b)
        count = int(match.sum())
        if count > 0:
            class_px[cls_id] += count
    return {
        "video":    item["video"],
        "class_px": dict(class_px),
        "total_px": total_px,
    }


def run_analysis(json_path, label):
    with open(json_path) as f:
        pairs = json.load(f)

    pixel_counts   = defaultdict(int)
    frame_counts   = defaultdict(int)
    total_frames   = len(pairs)
    total_pixels   = 0
    video_classes  = defaultdict(set)

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(analyze_mask, item): item for item in pairs}
        pbar    = tqdm(as_completed(futures), total=len(futures), desc=f"Analyzing {label}")
        for future in pbar:
            result = future.result()
            total_pixels += result["total_px"]
            for cls_id, px in result["class_px"].items():
                pixel_counts[cls_id] += px
                frame_counts[cls_id] += 1
                video_classes[result["video"]].add(cls_id)

    return {
        "pixel_counts":  dict(pixel_counts),
        "frame_counts":  dict(frame_counts),
        "total_frames":  total_frames,
        "total_pixels":  total_pixels,
        "video_classes": dict(video_classes),
        "pairs":         pairs,
    }


def print_report(stats, label):
    px     = stats["pixel_counts"]
    fc     = stats["frame_counts"]
    tf     = stats["total_frames"]
    tp     = stats["total_pixels"]
    videos = stats["video_classes"]

    print(f"\n{'='*80}")
    print(f"  {label} SET ANALYSIS")
    print(f"{'='*80}")
    print(f"  Total frames  : {tf:,}")
    print(f"  Total pixels  : {tp:,}")
    print(f"  Total videos  : {len(videos)}")
    print(f"  Videos        : {sorted(videos.keys())}")
    print(f"{'='*80}")
    print(f"  {'Cls':<5} {'Class Name':<28} {'Pixels':>14} {'% Pixels':>10} {'Frames':>8} {'% Frames':>10}")
    print(f"  {'-'*75}")

    for cls_id in range(NUM_CLASSES):
        name      = CLASS_NAMES[cls_id]
        cls_px    = px.get(cls_id, 0)
        cls_fr    = fc.get(cls_id, 0)
        pct_px    = 100.0 * cls_px / tp if tp > 0 else 0.0
        pct_fr    = 100.0 * cls_fr / tf if tf > 0 else 0.0
        absent    = "  << ABSENT" if cls_px == 0 else ""
        print(f"  [{cls_id:2d}]  {name:<28} {cls_px:>14,} {pct_px:>9.3f}% {cls_fr:>8,} {pct_fr:>9.1f}%{absent}")

    print(f"{'='*80}")

    print(f"\n  Per-video class presence ({label}):")
    print(f"  {'-'*60}")
    for video, classes in sorted(videos.items()):
        absent = [CLASS_NAMES[c] for c in range(NUM_CLASSES) if c not in classes]
        present_ids = sorted(classes)
        print(f"  {video:<15}  present: {present_ids}")
        if absent:
            print(f"  {'':15}  ABSENT : {absent}")
    print(f"  {'-'*60}")


def plot_pixel_distribution(train_stats, test_stats, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for ax, stats, label in [(axes[0], train_stats, "Train"), (axes[1], test_stats, "Test")]:
        px     = stats["pixel_counts"]
        tp     = stats["total_pixels"]
        names  = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
        values = [100.0 * px.get(i, 0) / tp for i in range(NUM_CLASSES)]
        colors = ["#e74c3c" if i in {5, 9} else "#f39c12" if i in {7, 10, 11} else "#3498db"
                  for i in range(NUM_CLASSES)]
        bars   = ax.barh(names[::-1], values[::-1], color=colors[::-1], edgecolor="black", linewidth=0.5)
        ax.set_xlabel("% of total pixels")
        ax.set_title(f"{label} Set — Pixel Distribution", fontsize=12)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
        for bar, val in zip(bars, values[::-1]):
            if val > 0:
                ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}%", va="center", fontsize=7)

    red_patch    = plt.Rectangle((0, 0), 1, 1, fc="#e74c3c")
    orange_patch = plt.Rectangle((0, 0), 1, 1, fc="#f39c12")
    blue_patch   = plt.Rectangle((0, 0), 1, 1, fc="#3498db")
    fig.legend([red_patch, orange_patch, blue_patch],
               ["Tool classes", "Rare classes", "Normal classes"],
               loc="lower center", ncol=3, fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = output_dir / "pixel_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")


def plot_frame_presence(train_stats, test_stats, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for ax, stats, label in [(axes[0], train_stats, "Train"), (axes[1], test_stats, "Test")]:
        fc     = stats["frame_counts"]
        tf     = stats["total_frames"]
        names  = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
        values = [100.0 * fc.get(i, 0) / tf for i in range(NUM_CLASSES)]
        colors = ["#e74c3c" if i in {5, 9} else "#f39c12" if i in {7, 10, 11} else "#3498db"
                  for i in range(NUM_CLASSES)]
        bars   = ax.barh(names[::-1], values[::-1], color=colors[::-1], edgecolor="black", linewidth=0.5)
        ax.set_xlabel("% of frames where class appears")
        ax.set_title(f"{label} Set — Frame Presence", fontsize=12)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        for bar, val in zip(bars, values[::-1]):
            if val > 0:
                ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va="center", fontsize=7)

    plt.tight_layout()
    out = output_dir / "frame_presence.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")


def plot_train_test_comparison(train_stats, test_stats, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    x = np.arange(NUM_CLASSES)
    w = 0.4

    for ax, key, ylabel, title in [
        (axes[0], "pixel_counts",  "% of total pixels",  "Pixel Distribution: Train vs Test"),
        (axes[1], "frame_counts",  "% of frames",        "Frame Presence: Train vs Test"),
    ]:
        t_total = train_stats["total_pixels"] if key == "pixel_counts" else train_stats["total_frames"]
        v_total = test_stats["total_pixels"]  if key == "pixel_counts" else test_stats["total_frames"]
        t_vals  = [100.0 * train_stats[key].get(i, 0) / t_total for i in range(NUM_CLASSES)]
        v_vals  = [100.0 * test_stats[key].get(i, 0)  / v_total for i in range(NUM_CLASSES)]
        ax.bar(x - w/2, t_vals, w, label="Train", color="#3498db", edgecolor="black", linewidth=0.5)
        ax.bar(x + w/2, v_vals, w, label="Test",  color="#e74c3c", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"[{i}]\n{CLASS_NAMES[i][:10]}" for i in range(NUM_CLASSES)],
                           fontsize=7, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.legend()

    plt.tight_layout()
    out = output_dir / "train_test_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Analyze train and test splits")
    parser.add_argument("--train_json", type=str, required=True, help="Path to train.json")
    parser.add_argument("--test_json", type=str, required=True, help="Path to test.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save analysis outputs")
    args = parser.parse_args()

    train_json = args.train_json
    test_json = args.test_json
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using {NUM_WORKERS} parallel workers")

    train_stats = run_analysis(train_json, "Train")
    test_stats  = run_analysis(test_json,  "Test")

    print_report(train_stats, "TRAIN")
    print_report(test_stats,  "TEST")

    print("\nGenerating plots...")
    plot_pixel_distribution(train_stats, test_stats, output_dir)
    plot_frame_presence(train_stats, test_stats, output_dir)
    plot_train_test_comparison(train_stats, test_stats, output_dir)

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()