import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.splits import group_by_patient, list_h5
from datasets.transforms import prepare_mask, zscore


class BraTSDataset(Dataset):
    """2D axial slices, one H5 file per slice.

    Args:
        data_dir:    directory with ``*.h5`` files.
        patient_ids: keep only slices of these patients (see `datasets.splits`).
                     ``None`` keeps everything.
        normalize:   ``"slice"`` for per-slice, per-modality z-score on brain
                     voxels, or ``None`` for raw intensities.
        mask_format: ``"labels"`` (NCR/ED/ET one-hot, converted to WT/TC/ET)
                     or ``"regions"`` (already WT/TC/ET).
    """

    def __init__(self, data_dir: str, patient_ids=None, normalize="slice", mask_format="labels"):
        groups = group_by_patient(list_h5(data_dir))
        if patient_ids is not None:
            keep = {str(p) for p in patient_ids}
            groups = {p: fs for p, fs in groups.items() if p in keep}
        self.files = sorted(f for fs in groups.values() for f in fs)
        self.normalize = normalize
        self.mask_format = mask_format

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with h5py.File(self.files[idx], "r") as f:
            # (H, W, 4) -> (4, H, W)
            image = torch.from_numpy(np.ascontiguousarray(f["image"][:].transpose(2, 0, 1))).float()
            # (H, W, 3) -> (3, H, W)
            mask = torch.from_numpy(np.ascontiguousarray(f["mask"][:].transpose(2, 0, 1))).float()
        if self.normalize == "slice":
            image = zscore(image, spatial_dims=(1, 2))
        elif self.normalize is not None:
            raise ValueError(f"2D dataset supports normalize='slice' or None, got {self.normalize!r}")
        return image, prepare_mask(mask, self.mask_format)
