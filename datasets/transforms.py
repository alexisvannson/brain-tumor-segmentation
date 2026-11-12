"""Label conversion and intensity normalisation shared by the 2D and 3D datasets."""
import torch

# Output channel order after `labels_to_regions`
REGIONS = ["whole_tumor", "tumor_core", "enhancing_tumor"]


def labels_to_regions(mask):
    """Convert one-hot BraTS labels to the nested evaluation regions.

    Input channels (``mask_format: labels``):
        0 = necrotic / non-enhancing core (label 1)
        1 = peritumoral edema             (label 2)
        2 = enhancing tumor               (label 4)
    Output channels:
        0 = WT = NCR ∪ ED ∪ ET
        1 = TC = NCR ∪ ET
        2 = ET
    Works for any trailing spatial shape: (3, H, W) or (3, D, H, W).
    """
    ncr, ed, et = mask[0] > 0, mask[1] > 0, mask[2] > 0
    return torch.stack([ncr | ed | et, ncr | et, et]).to(mask.dtype)


def prepare_mask(mask, mask_format):
    if mask_format == "labels":
        return labels_to_regions(mask)
    if mask_format == "regions":
        return mask
    raise ValueError(f"Unknown mask_format: {mask_format!r} (expected 'labels' or 'regions')")


def zscore(image, spatial_dims):
    """Per-channel z-score over brain voxels (non-zero), background kept at 0.

    `spatial_dims` are the axes the statistics are pooled over, e.g. (1, 2)
    for one slice (C, H, W), (1, 2, 3) for a whole volume (C, D, H, W), or
    (2, 3) for each slice of a volume independently.
    """
    brain = image != 0
    n = brain.sum(dim=spatial_dims, keepdim=True).clamp(min=1)
    mean = (image * brain).sum(dim=spatial_dims, keepdim=True) / n
    var = (((image - mean) * brain) ** 2).sum(dim=spatial_dims, keepdim=True) / n
    out = (image - mean) / (var.sqrt() + 1e-8)
    return out * brain
