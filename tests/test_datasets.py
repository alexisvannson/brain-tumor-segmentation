"""Tests for BraTSDataset (2D slices) and BraTSDataset3D (volumetric)."""
import os
import tempfile

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from datasets.brats_dataset import BraTSDataset
from datasets.brats_dataset3d import BraTSDataset3D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_2d_h5(path, h=32, w=32, img_ch=4, mask_ch=3):
    """Write a synthetic 2D-slice H5 file: image (H,W,C), mask (H,W,C)."""
    with h5py.File(path, "w") as f:
        f.create_dataset("image", data=np.random.rand(h, w, img_ch).astype(np.float32))
        f.create_dataset("mask",  data=np.random.randint(0, 2, (h, w, mask_ch)).astype(np.float32))


def _write_3d_h5_channels_last(path, h=16, w=16, d=16, img_ch=4, mask_ch=3):
    """3D H5 with channels-last layout: (H,W,D,C)."""
    with h5py.File(path, "w") as f:
        f.create_dataset("image", data=np.random.rand(h, w, d, img_ch).astype(np.float32))
        f.create_dataset("mask",  data=np.random.randint(0, 2, (h, w, d, mask_ch)).astype(np.float32))


# ---------------------------------------------------------------------------
# BraTSDataset (2D)
# ---------------------------------------------------------------------------

class TestBraTSDataset:
    @pytest.fixture
    def data_dir(self, tmp_path):
        for i in range(5):
            _write_2d_h5(tmp_path / f"sample_{i:03d}.h5")
        return str(tmp_path)

    def test_len(self, data_dir):
        ds = BraTSDataset(data_dir)
        assert len(ds) == 5

    def test_image_shape(self, data_dir):
        ds = BraTSDataset(data_dir)
        img, mask = ds[0]
        assert img.shape == (4, 32, 32), f"Unexpected image shape: {img.shape}"

    def test_mask_shape(self, data_dir):
        ds = BraTSDataset(data_dir)
        img, mask = ds[0]
        assert mask.shape == (3, 32, 32), f"Unexpected mask shape: {mask.shape}"

    def test_dtype_float32(self, data_dir):
        ds = BraTSDataset(data_dir)
        img, mask = ds[0]
        assert img.dtype == torch.float32
        assert mask.dtype == torch.float32

    def test_all_indices_accessible(self, data_dir):
        ds = BraTSDataset(data_dir)
        for i in range(len(ds)):
            img, mask = ds[i]
            assert img is not None

    def test_sorted_file_order(self, data_dir):
        """Files must be sorted so the dataset is deterministic."""
        ds = BraTSDataset(data_dir)
        assert ds.files == sorted(ds.files)

    def test_empty_dir(self, tmp_path):
        ds = BraTSDataset(str(tmp_path))
        assert len(ds) == 0

    def test_ignores_non_h5_files(self, tmp_path):
        _write_2d_h5(tmp_path / "real.h5")
        (tmp_path / "ignored.txt").write_text("hello")
        (tmp_path / "ignored.npy").write_bytes(b"\x00")
        ds = BraTSDataset(str(tmp_path))
        assert len(ds) == 1

    def test_dataloader_batching(self, data_dir):
        ds = BraTSDataset(data_dir)
        dl = DataLoader(ds, batch_size=2, shuffle=False)
        imgs, masks = next(iter(dl))
        assert imgs.shape == (2, 4, 32, 32)
        assert masks.shape == (2, 3, 32, 32)

    def test_values_in_valid_range(self, data_dir):
        """Without normalisation, raw values come through unchanged (np.random.rand >= 0)."""
        ds = BraTSDataset(data_dir, normalize=None)
        img, mask = ds[0]
        assert img.min() >= 0.0
        assert set(mask.unique().tolist()) <= {0.0, 1.0}

    def test_slice_normalisation(self, data_dir):
        ds = BraTSDataset(data_dir, normalize="slice")
        img, _ = ds[0]
        flat = img.reshape(4, -1)
        assert torch.allclose(flat.mean(dim=1), torch.zeros(4), atol=1e-4)
        assert torch.allclose(flat.std(dim=1, unbiased=False), torch.ones(4), atol=1e-3)

    def test_patient_filter(self, tmp_path):
        for pid in (1, 2, 3):
            for s in range(3):
                _write_2d_h5(tmp_path / f"volume_{pid}_slice_{s}.h5")
        ds = BraTSDataset(str(tmp_path), patient_ids=["1", "3"])
        assert len(ds) == 6
        assert not any("volume_2_" in f for f in ds.files)

    def test_different_spatial_sizes(self, tmp_path):
        """Dataset should handle H5 files with different image sizes."""
        _write_2d_h5(tmp_path / "big.h5",   h=64, w=64)
        _write_2d_h5(tmp_path / "small.h5", h=16, w=16)
        ds = BraTSDataset(str(tmp_path))
        img_big,   _ = ds[0]
        img_small, _ = ds[1]
        assert img_big.shape[-1] != img_small.shape[-1]


# ---------------------------------------------------------------------------
# BraTSDataset3D
# ---------------------------------------------------------------------------

class TestBraTSDataset3D:
    @pytest.fixture
    def data_dir(self, tmp_path):
        for i in range(4):
            _write_3d_h5_channels_last(tmp_path / f"vol_{i:03d}.h5")
        return str(tmp_path)

    def test_len(self, data_dir):
        ds = BraTSDataset3D(data_dir)
        assert len(ds) == 4

    def test_image_shape_channels_last_input(self, data_dir):
        """Input (H,W,D,4) should be transposed to (4,D,H,W)."""
        ds = BraTSDataset3D(data_dir)
        img, mask = ds[0]
        assert img.shape[0] == 4,  f"Expected 4 channels, got {img.shape}"
        assert img.dim() == 4,      "Image should be 4-dimensional (C,D,H,W)"

    def test_mask_shape_channels_last_input(self, data_dir):
        """Input (H,W,D,3) should become (3,D,H,W)."""
        ds = BraTSDataset3D(data_dir)
        img, mask = ds[0]
        assert mask.shape[0] == 3,  f"Expected 3 channels, got {mask.shape}"
        assert mask.dim() == 4

    def test_dtype_float32(self, data_dir):
        ds = BraTSDataset3D(data_dir)
        img, mask = ds[0]
        assert img.dtype == torch.float32
        assert mask.dtype == torch.float32

    def test_all_indices_accessible(self, data_dir):
        ds = BraTSDataset3D(data_dir)
        for i in range(len(ds)):
            img, mask = ds[i]
            assert img is not None

    def test_sorted_file_order(self, data_dir):
        ds = BraTSDataset3D(data_dir)
        assert ds.files == sorted(ds.files)

    def test_empty_dir(self, tmp_path):
        ds = BraTSDataset3D(str(tmp_path))
        assert len(ds) == 0

    def test_ignores_non_h5_files(self, tmp_path):
        _write_3d_h5_channels_last(tmp_path / "real.h5")
        (tmp_path / "notes.txt").write_text("ignore me")
        ds = BraTSDataset3D(str(tmp_path))
        assert len(ds) == 1

    def test_dataloader_batching(self, data_dir):
        ds = BraTSDataset3D(data_dir)
        dl = DataLoader(ds, batch_size=2, shuffle=False)
        imgs, masks = next(iter(dl))
        assert imgs.shape[0] == 2
        assert imgs.shape[1] == 4   # channels
        assert imgs.dim() == 5      # (B, C, D, H, W)

    def test_stacks_slice_files_per_patient(self, tmp_path):
        """Kaggle layout: one file per slice -> one item per patient, slices in index order."""
        for pid in (7, 8):
            for s in (0, 1, 2, 10):  # 10 must come after 2 (numeric, not lexicographic)
                with h5py.File(tmp_path / f"volume_{pid}_slice_{s}.h5", "w") as f:
                    f.create_dataset("image", data=np.full((8, 8, 4), s + 1, dtype=np.float32))
                    f.create_dataset("mask", data=np.zeros((8, 8, 3), dtype=np.float32))
        ds = BraTSDataset3D(str(tmp_path), normalize=None)
        assert len(ds) == 2
        assert ds.patient_ids == ["7", "8"]
        img, mask = ds[0]
        assert img.shape == (4, 4, 8, 8) and mask.shape == (3, 4, 8, 8)
        assert img[0, :, 0, 0].tolist() == [1, 2, 3, 11]

    def test_spatial_dims_correct(self, data_dir):
        """Spatial dims should be (D, H, W) = (16, 16, 16) for our fixture."""
        ds = BraTSDataset3D(data_dir)
        img, _ = ds[0]
        # channels-last (H,W,D,C)=(16,16,16,4) → (C,D,H,W)=(4,16,16,16)
        assert img.shape == (4, 16, 16, 16), f"Unexpected shape: {img.shape}"
