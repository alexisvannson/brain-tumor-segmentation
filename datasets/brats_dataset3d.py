import os
import h5py
import torch
from torch.utils.data import Dataset


class BraTSDataset3D(Dataset):
    """
    Loads volumetric BraTS H5 files (one file per patient).

    Expected H5 layout (one of):
      - image: (H, W, D, 4)  →  output (4, D, H, W)
      - image: (4, H, W, D)  →  output (4, D, H, W)
      - mask:  (H, W, D, 3)  →  output (3, D, H, W)
      - mask:  (3, H, W, D)  →  output (3, D, H, W)
    """

    def __init__(self, data_dir: str):
        self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".h5")
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with h5py.File(self.files[idx], "r") as f:
            image = torch.from_numpy(f["image"][:]).float()
            mask  = torch.from_numpy(f["mask"][:]).float()

        # Normalise to (C, D, H, W)
        image = self._to_cdhw(image, n_channels=4)
        mask  = self._to_cdhw(mask,  n_channels=3)
        return image, mask

    @staticmethod
    def _to_cdhw(t: torch.Tensor, n_channels: int) -> torch.Tensor:
        """Reorder tensor to (C, D, H, W) regardless of storage layout."""
        if t.dim() == 4:
            # Find which dim has n_channels
            for i, s in enumerate(t.shape):
                if s == n_channels:
                    # Move channels to front then put depth first: (C, H, W, D) → (C, D, H, W)
                    t = t.permute(*[i] + [j for j in range(4) if j != i])
                    # Now t is (C, ?, ?, ?) — last dim might be D; ensure (C, D, H, W)
                    # Convention: if channels were last → (H, W, D, C) → after permute (C, H, W, D)
                    #              move D (index 3) to index 1
                    if i == 3:
                        t = t.permute(0, 3, 1, 2)
                    return t
        return t
