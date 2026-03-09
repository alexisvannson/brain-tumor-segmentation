import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.Unet import Unet
from models.Unet3d import Unet3d
from datasets.brats_dataset import BraTSDataset
from datasets.brats_dataset3d import BraTSDataset3D
from scripts.training import create_model, load_config, _crop_masks


CLASSES = ["whole_tumor", "tumor_core", "enhancing_tumor"]


def dice_score(pred, target, eps=1e-6):
    """Compute per-class Dice score. pred: (B, C, ...) logits, target: (B, C, ...) binary."""
    pred = (torch.sigmoid(pred) > 0.5).float()
    spatial_dims = tuple(range(2, pred.dim()))
    intersection = (pred * target).sum(dim=spatial_dims)
    union = pred.sum(dim=spatial_dims) + target.sum(dim=spatial_dims)
    return ((2 * intersection + eps) / (union + eps)).mean(dim=0)  # (C,)


def iou_score(pred, target, eps=1e-6):
    """Compute per-class IoU. pred: (B, C, ...) logits, target: (B, C, ...) binary."""
    pred = (torch.sigmoid(pred) > 0.5).float()
    spatial_dims = tuple(range(2, pred.dim()))
    intersection = (pred * target).sum(dim=spatial_dims)
    union = pred.sum(dim=spatial_dims) + target.sum(dim=spatial_dims) - intersection
    return ((intersection + eps) / (union + eps)).mean(dim=0)  # (C,)


def evaluate(model, loader, device):
    model.eval()
    dice_sum = torch.zeros(3)
    iou_sum = torch.zeros(3)
    n_batches = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            masks = _crop_masks(masks, preds)
            dice_sum += dice_score(preds, masks).cpu()
            iou_sum += iou_score(preds, masks).cpu()
            n_batches += 1

    mean_dice = dice_sum / n_batches
    mean_iou  = iou_sum  / n_batches
    return mean_dice, mean_iou


def print_results(mean_dice, mean_iou, output_file=None):
    lines = [
        "",
        "=" * 52,
        f"{'Class':<20} {'Dice':>10} {'IoU':>10}",
        "-" * 52,
    ]
    for i, cls in enumerate(CLASSES):
        lines.append(f"{cls:<20} {mean_dice[i].item():>10.4f} {mean_iou[i].item():>10.4f}")
    lines += [
        "-" * 52,
        f"{'Mean':<20} {mean_dice.mean().item():>10.4f} {mean_iou.mean().item():>10.4f}",
        "=" * 52,
        "",
    ]
    output = "\n".join(lines)
    print(output)
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            f.write(output)
        print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained BraTS segmentation model")
    parser.add_argument("model", type=str, choices=["unet", "unet3d"], help="Model architecture")
    parser.add_argument("weights", type=str, help="Path to model weights (.pth)")
    parser.add_argument("--config", type=str, default=None, help="Override default config file path")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory from config")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size from config")
    parser.add_argument("--output", type=str, default=None, help="Path to save results (optional)")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = load_config(args.model)

    training_cfg = config.get("training", {})
    data_dir   = args.data_dir   or training_cfg.get("data_dir")
    batch_size = args.batch_size or training_cfg.get("batch_size", 2)
    num_workers = training_cfg.get("num_workers", 0)

    if not data_dir:
        raise ValueError("data_dir must be set in config or via --data_dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = create_model(args.model, config).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    print(f"Loaded weights from {args.weights}")

    dataset_cls = BraTSDataset3D if args.model == "unet3d" else BraTSDataset
    dataset = dataset_cls(data_dir)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"Evaluating on {len(dataset)} samples")

    mean_dice, mean_iou = evaluate(model, loader, device)
    print_results(mean_dice, mean_iou, output_file=args.output)


if __name__ == "__main__":
    main()
