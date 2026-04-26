"""
Usage:
  python train_seg.py --train_json /path/to/train.json --test_json /path/to/test.json --save_dir /path/to/save
"""

import os
from dotenv import load_dotenv
load_dotenv()

import wandb
if os.getenv("WANDB_API_KEY"):
    wandb.login(key=os.getenv("WANDB_API_KEY"))

import warnings
warnings.filterwarnings("ignore")
import json
import argparse
import math
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm.auto import tqdm
import wandb

from transformers import SegformerForSemanticSegmentation



IMG_W         = 768
IMG_H         = 480
BATCH_SIZE    = 8
NUM_WORKERS   = 4
PREFETCH      = 2
EPOCHS        = 40
LR_BACKBONE   = 1e-5
LR_HEAD       = 1e-4
WEIGHT_DECAY  = 1e-2
WARMUP_EPOCHS = 4
PATIENCE      = 15
SAVE_EVERY    = 3
SEED          = 42
NUM_CLASSES   = 10

CLASS_NAMES = [
    "Black Background",
    "Abdominal Wall",
    "Liver",
    "Gastrointestinal Tract",
    "Fat",
    "Grasper",
    "Connective Tissue",
    "Cystic Duct",
    "L-Hook Electrocautery",
    "Hepatic Vein",
]

CLASS_ID_REMAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    8: 7,
    9: 8,
    11: 9,
}

COLOR_TO_CLASS = {
    (127, 127, 127): 0,
    (255, 114, 114): 1,
    (210, 140, 140): 2,
    (255, 255, 255): 3,
    (186, 183, 75):  4,
    (170, 255, 0):   5,
    (255, 160, 165): 6,
    (169, 255, 184): 7,
    (231, 70,  156): 8,
    (0,   50,  128): 9,
    (255, 85,  0):   9,
}

TOOL_COLORS = {
    (170, 255, 0),
    (231, 70,  156),
}

RARE_COLORS = {
    (169, 255, 184),
    (0,   50,  128),
    (255, 85,  0),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def color_mask_to_class_mask(mask_rgb: np.ndarray) -> np.ndarray:
    h, w       = mask_rgb.shape[:2]
    flat       = mask_rgb.reshape(-1, 3)
    class_flat = np.zeros(flat.shape[0], dtype=np.int64)
    for (r, g, b), cls_id in COLOR_TO_CLASS.items():
        match = (flat[:, 0] == r) & (flat[:, 1] == g) & (flat[:, 2] == b)
        class_flat[match] = cls_id
    return class_flat.reshape(h, w)


def has_color_pixels(mask_rgb: np.ndarray, color_set: set) -> bool:
    flat = mask_rgb.reshape(-1, 3)
    for (r, g, b) in color_set:
        if np.any((flat[:, 0] == r) & (flat[:, 1] == g) & (flat[:, 2] == b)):
            return True
    return False


def build_sample_weights(json_path: str, rare_weight: float = 3.0, tool_weight: float = 2.0) -> list:
    with open(json_path) as f:
        pairs = json.load(f)
    weights    = []
    rare_count = 0
    tool_count = 0
    print("Building sample weights...")
    for item in tqdm(pairs):
        mask_rgb = np.array(Image.open(item["mask"]).convert("RGB"))
        if has_color_pixels(mask_rgb, RARE_COLORS):
            weights.append(rare_weight)
            rare_count += 1
        elif has_color_pixels(mask_rgb, TOOL_COLORS):
            weights.append(tool_weight)
            tool_count += 1
        else:
            weights.append(1.0)
    n = len(weights)
    print(f"Rare-class frames : {rare_count}/{n} ({100*rare_count/n:.1f}%)")
    print(f"Tool-only frames  : {tool_count}/{n} ({100*tool_count/n:.1f}%)")
    print(f"Normal frames     : {n-rare_count-tool_count}/{n} ({100*(n-rare_count-tool_count)/n:.1f}%)")
    return weights


def get_train_transforms():
    return A.Compose([
        A.Resize(height=IMG_H, width=IMG_W, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=10,
                           border_mode=0, p=0.3),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08, p=0.8),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(std_range=(0.04, 0.2), p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.4),
        A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(20, 50),
                        hole_width_range=(20, 50), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_test_transforms():
    return A.Compose([
        A.Resize(height=IMG_H, width=IMG_W, p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class CholecSegDataset(Dataset):
    def __init__(self, json_path, transforms):
        with open(json_path) as f:
            self.pairs = json.load(f)
        self.transforms = transforms

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item       = self.pairs[idx]
        image      = np.array(Image.open(item["frame"]).convert("RGB"))
        mask_rgb   = np.array(Image.open(item["mask"]).convert("RGB"))
        class_mask = color_mask_to_class_mask(mask_rgb).astype(np.int64)
        augmented  = self.transforms(image=image, mask=class_mask)
        image_t    = augmented["image"]
        mask_t     = augmented["mask"].long()
        return image_t, mask_t


def focal_loss_multiclass(logits, targets, gamma=3.0):
    ce    = F.cross_entropy(logits, targets, reduction="none")
    pt    = torch.exp(-ce)
    focal = (1 - pt) ** gamma * ce
    return focal.mean()


def dice_loss_multiclass(logits, targets, smooth=1.0):
    probs        = F.softmax(logits, dim=1)
    targets_oh   = F.one_hot(targets, num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()
    dims         = (0, 2, 3)
    intersection = (probs * targets_oh).sum(dim=dims)
    cardinality  = probs.sum(dim=dims) + targets_oh.sum(dim=dims)
    dice_per_cls = (2 * intersection + smooth) / (cardinality + smooth)
    return 1 - dice_per_cls.mean()


def combined_loss(logits, targets):
    return focal_loss_multiclass(logits, targets) + 2.0 * dice_loss_multiclass(logits, targets)


def compute_per_class_dice(logits, targets, smooth=1e-6):
    preds       = logits.argmax(dim=1)
    dice_scores = []
    for cls in range(NUM_CLASSES):
        pred_cls = (preds == cls).float()
        tgt_cls  = (targets == cls).float()
        inter    = (pred_cls * tgt_cls).sum()
        denom    = pred_cls.sum() + tgt_cls.sum()
        if denom < 1:
            dice_scores.append(float("nan"))
        else:
            dice_scores.append(((2 * inter + smooth) / (denom + smooth)).item())
    return dice_scores


def compute_miou(logits, targets, smooth=1e-6):
    preds = logits.argmax(dim=1)
    ious  = []
    for cls in range(NUM_CLASSES):
        pred_cls = (preds == cls)
        tgt_cls  = (targets == cls)
        inter    = (pred_cls & tgt_cls).sum().item()
        union    = (pred_cls | tgt_cls).sum().item()
        ious.append(0.0 if union < 1 else (inter + smooth) / (union + smooth))
    return float(np.mean(ious))


def build_model(device):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    model.decode_head.classifier = nn.Conv2d(
        model.decode_head.classifier.in_channels, NUM_CLASSES, kernel_size=1
    )
    return model.to(device)


def get_param_groups(model):
    backbone_params = []
    head_params     = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "segformer.encoder" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    return [
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params,     "lr": LR_HEAD},
    ]


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def forward_pass(model, images):
    outputs   = model(pixel_values=images)
    logits_lr = outputs.logits
    logits    = F.interpolate(logits_lr, size=(IMG_H, IMG_W), mode="bilinear", align_corners=False)
    return logits


def train_one_epoch(model, loader, optimizer, scaler, scheduler, device, epoch):
    model.train()
    total_loss = total_miou = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1:02d} [Train]", leave=True, dynamic_ncols=True)

    for images, masks_gt in pbar:
        images   = images.to(device)
        masks_gt = masks_gt.to(device)

        optimizer.zero_grad()
        with autocast():
            logits = forward_pass(model, images)
            loss   = combined_loss(logits, masks_gt)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        miou        = compute_miou(logits.detach(), masks_gt)
        total_loss += loss.item()
        total_miou += miou

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "miou": f"{miou:.4f}",
            "lr_b": f"{scheduler.get_last_lr()[0]:.2e}",
            "lr_h": f"{scheduler.get_last_lr()[1]:.2e}",
        })

    n = len(loader)
    return total_loss / n, total_miou / n


@torch.no_grad()
def evaluate(model, loader, device, epoch):
    model.eval()
    total_loss       = 0.0
    total_miou       = 0.0
    class_dice_accum = defaultdict(list)

    pbar = tqdm(loader, desc=f"Epoch {epoch+1:02d} [Eval] ", leave=True, dynamic_ncols=True)

    for images, masks_gt in pbar:
        images   = images.to(device)
        masks_gt = masks_gt.to(device)

        with autocast():
            logits = forward_pass(model, images)
            loss   = combined_loss(logits, masks_gt)

        miou        = compute_miou(logits, masks_gt)
        dice_scores = compute_per_class_dice(logits, masks_gt)
        total_loss += loss.item()
        total_miou += miou

        for cls_id, d in enumerate(dice_scores):
            if not math.isnan(d):
                class_dice_accum[cls_id].append(d)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "miou": f"{miou:.4f}"})

    n              = len(loader)
    mean_loss      = total_loss / n
    mean_miou      = total_miou / n
    per_class_dice = {
        cls_id: float(np.mean(class_dice_accum[cls_id])) if class_dice_accum[cls_id] else float("nan")
        for cls_id in range(NUM_CLASSES)
    }
    valid_dices = [v for v in per_class_dice.values() if not math.isnan(v)]
    mean_dice   = float(np.mean(valid_dices)) if valid_dices else 0.0

    return mean_loss, mean_miou, mean_dice, per_class_dice


def print_per_class_dice(per_class_dice, epoch):
    tool_classes = {5, 8}
    lines        = [f"\n  Per-class Dice — Epoch {epoch+1}:"]
    for cls_id, dice in per_class_dice.items():
        tag  = " <-- TOOL" if cls_id in tool_classes else ""
        val  = f"{dice:.4f}" if not math.isnan(dice) else "  N/A "
        lines.append(f"    [{cls_id:2d}] {CLASS_NAMES[cls_id]:<25s} {val}{tag}")
    lines.append("")
    tqdm.write("\n".join(lines))


def run_mask_sanity_check(train_json):
    with open(train_json) as f:
        pairs = json.load(f)
    print("\nMask sanity check (5 random samples)...")
    for item in random.sample(pairs, 5):
        mask_rgb  = np.array(Image.open(item["mask"]).convert("RGB"))
        cls_mask  = color_mask_to_class_mask(mask_rgb)
        unique    = np.unique(cls_mask).tolist()
        total_px  = cls_mask.size
        mapped_px = sum(
            int(((mask_rgb[:, :, 0] == r) & (mask_rgb[:, :, 1] == g) & (mask_rgb[:, :, 2] == b)).sum())
            for (r, g, b) in COLOR_TO_CLASS
        )
        unmapped  = total_px - mapped_px
        print(f"  classes: {unique}  |  unmapped px: {unmapped:,} / {total_px:,} ({100*unmapped/total_px:.2f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Train SegFormer")
    parser.add_argument("--train_json", type=str, required=True, help="Path to train.json")
    parser.add_argument("--test_json", type=str, required=True, help="Path to test.json")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the trained model")
    args = parser.parse_args()

    train_json = args.train_json
    test_json = args.test_json
    save_dir = Path(args.save_dir)

    set_seed(SEED)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    wandb.init(
        project="surgical-agent-segformer",
        name=f"segformer-b2-10class-{datetime.now().strftime('%Y%m%d-%H%M')}",
        config={
            "model":         "nvidia/mit-b2",
            "num_classes":   NUM_CLASSES,  # Blood(7) and Gallbladder(10) removed
            "img_size":      f"{IMG_W}x{IMG_H}",
            "batch_size":    BATCH_SIZE,
            "lr_backbone":   LR_BACKBONE,
            "lr_head":       LR_HEAD,
            "weight_decay":  WEIGHT_DECAY,
            "epochs":        EPOCHS,
            "warmup_epochs": WARMUP_EPOCHS,
            "patience":      PATIENCE,
            "loss":          "focal(gamma=3)+2*dice_multiclass",
            "optimizer":     "AdamW",
            "scheduler":     "cosine_with_warmup",
            "sampler":       "WeightedRandomSampler(tool_weight=3)",
            "gpu":           torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
    )

    run_mask_sanity_check(train_json)

    print("Step 1: Building sample weights...")
    sample_weights = build_sample_weights(train_json, rare_weight=3.0, tool_weight=2.0)

    print("\nStep 2: Building datasets...")
    train_dataset = CholecSegDataset(train_json, get_train_transforms())
    test_dataset  = CholecSegDataset(test_json,  get_test_transforms())
    print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False,
    )

    print("\nStep 3: Building SegFormer-B2 (10-class, Blood+Gallbladder removed)...")
    model       = build_model(device)
    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_p:,} | Trainable: {trainable_p:,}")

    param_groups = get_param_groups(model)
    optimizer    = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    total_steps  = EPOCHS * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = GradScaler()

    best_miou      = 0.0
    best_ckpt_path = save_dir / "best_model.pt"
    no_improve     = 0

    print("\nStep 4: Training...\n")
    for epoch in range(EPOCHS):
        train_loss, train_miou = train_one_epoch(
            model, train_loader, optimizer, scaler, scheduler, device, epoch
        )
        eval_loss, eval_miou, eval_dice, per_class_dice = evaluate(
            model, test_loader, device, epoch
        )

        tqdm.write(
            f"Epoch {epoch+1:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} mIoU: {train_miou:.4f} | "
            f"Val Loss: {eval_loss:.4f} mIoU: {eval_miou:.4f} mDice: {eval_dice:.4f}"
        )

        print_per_class_dice(per_class_dice, epoch)

        log_dict = {
            "epoch":       epoch + 1,
            "train/loss":  train_loss,
            "train/miou":  train_miou,
            "val/loss":    eval_loss,
            "val/miou":    eval_miou,
            "val/dice":    eval_dice,
            "lr/backbone": scheduler.get_last_lr()[0],
            "lr/head":     scheduler.get_last_lr()[1],
        }
        for cls_id, dice in per_class_dice.items():
            if not math.isnan(dice):
                log_dict[f"val/dice_cls{cls_id:02d}_{CLASS_NAMES[cls_id].replace(' ', '_')}"] = dice
        wandb.log(log_dict)

        if eval_miou > best_miou:
            best_miou  = eval_miou
            no_improve = 0
            torch.save({
                "epoch":          epoch + 1,
                "state_dict":     model.state_dict(),
                "optimizer":      optimizer.state_dict(),
                "scheduler":      scheduler.state_dict(),
                "best_miou":      best_miou,
                "per_class_dice": per_class_dice,
            }, best_ckpt_path)
            tqdm.write(f"  --> New best model saved (mIoU: {best_miou:.4f})")
            wandb.run.summary["best_val_miou"] = best_miou
            wandb.run.summary["best_val_dice"] = eval_dice
            wandb.run.summary["best_epoch"]    = epoch + 1
            for cls_id, dice in per_class_dice.items():
                if not math.isnan(dice):
                    wandb.run.summary[f"best_dice_cls{cls_id:02d}_{CLASS_NAMES[cls_id].replace(' ', '_')}"] = dice
        else:
            no_improve += 1
            tqdm.write(f"  --> No improvement for {no_improve}/{PATIENCE} epochs")

        if (epoch + 1) % SAVE_EVERY == 0:
            ckpt_path = save_dir / f"checkpoint_epoch{epoch+1:03d}.pt"
            torch.save({
                "epoch":      epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "val_miou":   eval_miou,
            }, ckpt_path)
            tqdm.write(f"  --> Checkpoint saved: {ckpt_path.name}")

        if no_improve >= PATIENCE:
            tqdm.write(f"\nEarly stopping triggered at epoch {epoch+1}.")
            break

    wandb.finish()
    print(f"\nTraining complete. Best val mIoU: {best_miou:.4f}")
    print(f"Best model: {best_ckpt_path}")


if __name__ == "__main__":
    main()