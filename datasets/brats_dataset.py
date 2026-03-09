import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
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
            # (H, W, 4) -> (4, H, W)
            image = torch.from_numpy(f["image"][:].transpose(2, 0, 1)).float()
            # (H, W, 3) -> (3, H, W)
            mask  = torch.from_numpy(f["mask"][:].transpose(2, 0, 1)).float()
        return image, mask
