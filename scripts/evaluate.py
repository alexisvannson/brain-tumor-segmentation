"""Per-patient evaluation on a patient-level split (the held-out test set by default).

For every patient the full volume is predicted (slice by slice for the 2D
model) and Dice, IoU and HD95 are computed for the three BraTS regions. The
per-patient table is saved as CSV; the summary reports mean ± std and median
across patients.
"""
import argparse
import csv
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.transforms import REGIONS
from scripts.metrics import binarize, case_metrics, summarize
from scripts.training import _crop_masks, create_model, get_split, load_config, predict, volume_dataset

CLASSES = REGIONS
METRICS = ["dice", "iou", "hd95"]


def evaluate(model, loader, device, threshold=0.5, with_hd95=True, slice_batch=16, case_ids=None):
    """Return a list of per-case rows: {"case", "<metric>_<region>": value, ...}."""
    model.eval()
    rows = []
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(loader, desc="Evaluating")):
            images, masks = images.to(device), masks.to(device)
            preds = predict(model, images, slice_batch)
            masks = _crop_masks(masks, preds)
            m = case_metrics(binarize(preds, threshold), masks > 0.5, with_hd95=with_hd95)
            for b in range(images.shape[0]):
                idx = i * loader.batch_size + b
                row = {"case": case_ids[idx] if case_ids else idx}
                for name, values in m.items():
                    for c, region in enumerate(CLASSES):
                        row[f"{name}_{region}"] = values[b, c].item()
                rows.append(row)
    return rows


def summary_table(rows):
    lines = [
        "",
        "=" * 72,
        f"{'Metric':<8} {'Region':<17} {'Mean':>9} {'± Std':>9} {'Median':>9}   n={len(rows)}",
        "-" * 72,
    ]
    for metric in METRICS:
        if f"{metric}_{CLASSES[0]}" not in rows[0]:
            continue
        values = torch.tensor([[r[f"{metric}_{c}"] for c in CLASSES] for r in rows])
        s = summarize(values)
        for c, region in enumerate(CLASSES):
            lines.append(f"{metric:<8} {region:<17} {s['mean'][c]:>9.4f} {s['std'][c]:>9.4f} {s['median'][c]:>9.4f}")
        lines.append(f"{metric:<8} {'mean':<17} {s['mean'].mean():>9.4f}")
        lines.append("-" * 72)
    lines[-1] = "=" * 72
    return "\n".join(lines) + "\n"


def save_rows(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained BraTS segmentation model per patient")
    parser.add_argument("model", type=str, choices=["unet", "unet3d"], help="Model architecture")
    parser.add_argument("weights", type=str, help="Path to model weights (.pth)")
    parser.add_argument("--config", type=str, default=None, help="Override default config file path")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory from config")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"],
                        help="Patient split to evaluate (default: held-out test set)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for binarisation")
    parser.add_argument("--slice_batch", type=int, default=None, help="Slices per forward pass for the 2D model")
    parser.add_argument("--no_hd95", action="store_true", help="Skip HD95 (faster)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save summary to this file and per-patient metrics to <output>.csv")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = load_config(args.model)

    if args.data_dir:
        config.setdefault("data", {})["data_dir"] = args.data_dir
    if not config.get("data", {}).get("data_dir"):
        raise ValueError("data.data_dir must be set in config or via --data_dir")
    split_file = config["data"].get("split_file")
    if not split_file or not os.path.exists(split_file):
        raise FileNotFoundError(
            f"Split file {split_file!r} not found. It is created by training; evaluating without it "
            "would mix training patients into the evaluation."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = create_model(args.model, config).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    print(f"Loaded weights from {args.weights}")

    patient_ids = get_split(config)[args.split]
    dataset = volume_dataset(config, args.model, patient_ids)
    num_workers = config.get("training", {}).get("num_workers", 0)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    print(f"Evaluating on {len(dataset)} patients ({args.split} split), threshold={args.threshold}")

    slice_batch = args.slice_batch or config.get("training", {}).get("batch_size", 16)
    rows = evaluate(model, loader, device, threshold=args.threshold, with_hd95=not args.no_hd95,
                    slice_batch=slice_batch, case_ids=dataset.patient_ids)

    output = summary_table(rows)
    print(output)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output)
        csv_path = os.path.splitext(args.output)[0] + ".csv"
        save_rows(rows, csv_path)
        print(f"Summary saved to {args.output}, per-patient metrics to {csv_path}")


if __name__ == "__main__":
    main()
