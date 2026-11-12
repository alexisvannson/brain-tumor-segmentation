import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.splits import group_by_patient, list_h5, parse_slice_name
from datasets.transforms import prepare_mask, zscore


class BraTSDataset3D(Dataset):
    """One item per patient volume, returned as (C, D, H, W).

    Two storage layouts are supported:
      * one file per slice (``volume_{id}_slice_{n}.h5``, the Kaggle export):
        slices of a patient are stacked in slice order along D;
      * one file per volume, with image (H, W, D, 4) or (4, H, W, D) and mask
        (H, W, D, 3) or (3, H, W, D).

    Args:
        patient_ids: keep only these patients. ``None`` keeps all.
        normalize:   ``"volume"`` (z-score per modality over the whole brain),
                     ``"slice"`` (each slice independently, matching
                     `BraTSDataset` so 2D models see the same inputs at
                     evaluation as in training), or ``None``.
        mask_format: ``"labels"`` or ``"regions"``, see `datasets.transforms`.
    """

    def __init__(self, data_dir: str, patient_ids=None, normalize="volume", mask_format="labels"):
        groups = group_by_patient(list_h5(data_dir))
        if patient_ids is not None:
            keep = {str(p) for p in patient_ids}
            groups = {p: fs for p, fs in groups.items() if p in keep}
        self.patient_ids = list(groups.keys())
        self.groups = groups
        self.normalize = normalize
        self.mask_format = mask_format

    @property
    def files(self):
        """Files of the dataset, sorted (one per patient for volume files)."""
        return sorted(f for fs in self.groups.values() for f in fs)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        files = self.groups[self.patient_ids[idx]]
        if parse_slice_name(files[0])[1] is not None:
            image, mask = self._load_slices(files)
        else:
            image, mask = self._load_volume(files[0])

        if self.normalize == "volume":
            image = zscore(image, spatial_dims=(1, 2, 3))
        elif self.normalize == "slice":
            image = zscore(image, spatial_dims=(2, 3))
        elif self.normalize is not None:
            raise ValueError(f"Unknown normalize mode: {self.normalize!r}")
        return image, prepare_mask(mask, self.mask_format)

    @staticmethod
    def _load_slices(files):
        imgs, masks = [], []
        for path in files:
            with h5py.File(path, "r") as f:
                imgs.append(f["image"][:])   # (H, W, 4)
                masks.append(f["mask"][:])   # (H, W, 3)
        # (D, H, W, C) -> (C, D, H, W)
        image = torch.from_numpy(np.stack(imgs).transpose(3, 0, 1, 2).copy()).float()
        mask = torch.from_numpy(np.stack(masks).transpose(3, 0, 1, 2).copy()).float()
        return image, mask

    def _load_volume(self, path):
        with h5py.File(path, "r") as f:
            image = torch.from_numpy(f["image"][:]).float()
            mask = torch.from_numpy(f["mask"][:]).float()
        return self._to_cdhw(image, n_channels=4), self._to_cdhw(mask, n_channels=3)

    @staticmethod
    def _to_cdhw(t: torch.Tensor, n_channels: int) -> torch.Tensor:
        """Reorder (H, W, D, C) or (C, H, W, D) to (C, D, H, W)."""
        if t.dim() != 4:
            return t
        if t.shape[-1] == n_channels:      # (H, W, D, C)
            return t.permute(3, 2, 0, 1)
        if t.shape[0] == n_channels:       # (C, H, W, D)
            return t.permute(0, 3, 1, 2)
        raise ValueError(f"Cannot find a channel axis of size {n_channels} in shape {tuple(t.shape)}")
