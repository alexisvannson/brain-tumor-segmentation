"""Per-case segmentation metrics following BraTS conventions.

A *case* is one patient volume (or, for 2D-only tests, one slice). Metrics are
computed per case and per region, then aggregated across cases. That way every
patient carries the same weight regardless of batch size or tumour size, and
the per-case table can be used for confidence intervals and paired tests.

Empty-region conventions (as in the BraTS challenge):
  * prediction and ground truth both empty → Dice = IoU = 1, HD95 = 0
  * exactly one of them empty             → Dice = IoU = 0, HD95 = HD95_EMPTY_PENALTY
"""
import numpy as np
import torch
from scipy import ndimage

HD95_EMPTY_PENALTY = 373.13  # BraTS: diagonal of the 240x240x155 volume, in mm


def binarize(logits, threshold=0.5):
    return torch.sigmoid(logits) > threshold


def dice_iou_per_case(pred, target):
    """pred, target: boolean tensors (B, C, *spatial). Returns two (B, C) tensors."""
    pred, target = pred.bool(), target.bool()
    dims = tuple(range(2, pred.dim()))
    inter = (pred & target).sum(dim=dims).double()
    p = pred.sum(dim=dims).double()
    t = target.sum(dim=dims).double()
    union = p + t - inter
    both_empty = (p + t) == 0
    dice = torch.where(both_empty, torch.ones_like(inter), 2 * inter / (p + t).clamp(min=1))
    iou = torch.where(both_empty, torch.ones_like(inter), inter / union.clamp(min=1))
    return dice.float(), iou.float()


def _surface(mask):
    return mask & ~ndimage.binary_erosion(mask)


def hd95(pred, target, spacing=None):
    """95th percentile symmetric Hausdorff distance for one binary region.

    pred, target: boolean numpy arrays of the same shape (2D or 3D).
    spacing: voxel size per axis (defaults to 1, i.e. mm for BraTS).
    """
    pred, target = pred.astype(bool), target.astype(bool)
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return HD95_EMPTY_PENALTY
    sp, st = _surface(pred), _surface(target)
    dt_target = ndimage.distance_transform_edt(~st, sampling=spacing)
    dt_pred = ndimage.distance_transform_edt(~sp, sampling=spacing)
    dists = np.concatenate([dt_target[sp], dt_pred[st]])
    return float(np.percentile(dists, 95))


def case_metrics(pred, target, with_hd95=True, spacing=None):
    """All metrics for a batch of cases.

    pred, target: boolean tensors (B, C, *spatial).
    Returns dict of (B, C) float tensors: "dice", "iou" and optionally "hd95".
    """
    dice, iou = dice_iou_per_case(pred, target)
    out = {"dice": dice, "iou": iou}
    if with_hd95:
        p, t = pred.cpu().numpy(), target.cpu().numpy()
        out["hd95"] = torch.tensor(
            [[hd95(p[b, c], t[b, c], spacing) for c in range(p.shape[1])] for b in range(p.shape[0])],
            dtype=torch.float32,
        )
    return out


def summarize(values):
    """values: (N_cases, C) tensor -> dict of per-class mean / std / median."""
    v = values.double()
    return {
        "mean": v.mean(dim=0),
        "std": v.std(dim=0, unbiased=True) if v.shape[0] > 1 else torch.zeros(v.shape[1], dtype=v.dtype),
        "median": v.median(dim=0).values,
    }
