"""Tests for dice_loss and _crop_masks from the training script."""
import pytest
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.training import dice_loss, _crop_masks


# ---------------------------------------------------------------------------
# dice_loss
# ---------------------------------------------------------------------------

class TestDiceLoss:
    # --- basic properties ---

    def test_returns_scalar(self):
        pred   = torch.randn(2, 3, 16, 16)
        target = torch.randint(0, 2, (2, 3, 16, 16)).float()
        loss = dice_loss(pred, target)
        assert loss.dim() == 0

    def test_range_zero_to_one(self):
        for _ in range(10):
            pred   = torch.randn(2, 2, 16, 16)
            target = torch.randint(0, 2, (2, 2, 16, 16)).float()
            loss = dice_loss(pred, target)
            assert 0.0 <= loss.item() <= 1.0 + 1e-5, f"Out of range: {loss.item()}"

    def test_perfect_prediction_near_zero(self):
        """When sigmoid(pred) perfectly matches binary target, dice ≈ 0."""
        target = torch.ones(1, 1, 8, 8)          # all foreground
        pred   = torch.full((1, 1, 8, 8), 10.0)  # very high logit → sigmoid ≈ 1
        loss = dice_loss(pred, target)
        assert loss.item() < 0.01, f"Expected near-zero loss, got {loss.item()}"

    def test_worst_prediction_near_one(self):
        """Completely wrong predictions → dice ≈ 1."""
        target = torch.ones(1, 1, 8, 8)
        pred   = torch.full((1, 1, 8, 8), -10.0)  # sigmoid ≈ 0 vs all-ones target
        loss = dice_loss(pred, target)
        assert loss.item() > 0.95, f"Expected near-one loss, got {loss.item()}"

    def test_all_zero_target(self):
        """All-background target: dice should be well-defined (eps prevents div-by-zero)."""
        target = torch.zeros(1, 1, 8, 8)
        pred   = torch.randn(1, 1, 8, 8)
        loss = dice_loss(pred, target)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_no_nan_on_random_input(self):
        for _ in range(20):
            pred   = torch.randn(2, 4, 16, 16)
            target = torch.randint(0, 2, (2, 4, 16, 16)).float()
            assert not torch.isnan(dice_loss(pred, target))

    def test_gradient_flows(self):
        pred = torch.randn(2, 2, 16, 16, requires_grad=True)
        target = torch.randint(0, 2, (2, 2, 16, 16)).float()
        dice_loss(pred, target).backward()
        assert pred.grad is not None
        assert not pred.grad.isnan().any()

    # --- 2D vs 3D ---

    def test_2d_input(self):
        pred   = torch.randn(2, 3, 16, 16)
        target = torch.randint(0, 2, (2, 3, 16, 16)).float()
        loss = dice_loss(pred, target)
        assert loss.dim() == 0

    def test_3d_input(self):
        pred   = torch.randn(1, 3, 8, 8, 8)
        target = torch.randint(0, 2, (1, 3, 8, 8, 8)).float()
        loss = dice_loss(pred, target)
        assert loss.dim() == 0
        assert 0.0 <= loss.item() <= 1.0 + 1e-5

    def test_3d_perfect_prediction(self):
        target = torch.ones(1, 1, 4, 4, 4)
        pred   = torch.full((1, 1, 4, 4, 4), 10.0)
        assert dice_loss(pred, target).item() < 0.01

    def test_3d_no_nan(self):
        for _ in range(10):
            pred   = torch.randn(1, 2, 8, 8, 8)
            target = torch.randint(0, 2, (1, 2, 8, 8, 8)).float()
            assert not torch.isnan(dice_loss(pred, target))

    def test_multi_channel_average(self):
        """dice_loss averages over batch AND channels."""
        pred   = torch.randn(4, 5, 8, 8)
        target = torch.randint(0, 2, (4, 5, 8, 8)).float()
        loss = dice_loss(pred, target)
        assert loss.dim() == 0  # single scalar

    def test_batch_size_one(self):
        pred   = torch.randn(1, 1, 16, 16)
        target = torch.randint(0, 2, (1, 1, 16, 16)).float()
        assert dice_loss(pred, target).dim() == 0

    def test_consistent_with_bce_combined(self):
        """BCE + dice should be finite and positive for random inputs."""
        pred   = torch.randn(2, 3, 16, 16)
        target = torch.randint(0, 2, (2, 3, 16, 16)).float()
        bce  = nn.BCEWithLogitsLoss()(pred, target)
        dice = dice_loss(pred, target)
        combined = bce + dice
        assert combined.item() > 0
        assert not torch.isnan(combined)


# ---------------------------------------------------------------------------
# _crop_masks
# ---------------------------------------------------------------------------

class TestCropMasks:
    # --- 2D ---

    def test_no_crop_when_same_size(self):
        masks = torch.ones(2, 3, 20, 20)
        preds = torch.ones(2, 3, 20, 20)
        result = _crop_masks(masks, preds)
        assert result.shape == masks.shape

    def test_2d_crop_correct_size(self):
        masks = torch.arange(32 * 32).reshape(1, 1, 32, 32).float()
        preds = torch.zeros(1, 1, 20, 20)
        result = _crop_masks(masks, preds)
        assert result.shape == (1, 1, 20, 20)

    def test_2d_crop_is_center(self):
        """Center crop should extract the central region."""
        masks = torch.zeros(1, 1, 10, 10)
        masks[0, 0, 3:7, 3:7] = 1.0   # mark center 4x4
        preds = torch.zeros(1, 1, 4, 4)
        result = _crop_masks(masks, preds)
        # For a 10→4 center crop: offset = (10-4)//2 = 3
        assert result.shape == (1, 1, 4, 4)
        assert result.sum().item() == 16.0  # the marked region is fully inside

    def test_2d_multi_batch_channels(self):
        masks = torch.randn(4, 3, 32, 32)
        preds = torch.randn(4, 3, 20, 20)
        result = _crop_masks(masks, preds)
        assert result.shape == (4, 3, 20, 20)

    def test_2d_odd_size_difference(self):
        """When size diff is odd, floor division should still produce correct output size."""
        masks = torch.ones(1, 1, 11, 11)
        preds = torch.zeros(1, 1, 6, 6)
        result = _crop_masks(masks, preds)
        assert result.shape == (1, 1, 6, 6)

    # --- 3D ---

    def test_3d_no_crop_when_same_size(self):
        masks = torch.ones(1, 2, 10, 10, 10)
        preds = torch.ones(1, 2, 10, 10, 10)
        result = _crop_masks(masks, preds)
        assert result.shape == masks.shape

    def test_3d_crop_correct_size(self):
        masks = torch.randn(2, 3, 32, 32, 32)
        preds = torch.randn(2, 3, 20, 20, 20)
        result = _crop_masks(masks, preds)
        assert result.shape == (2, 3, 20, 20, 20)

    def test_3d_center_crop_values(self):
        masks = torch.zeros(1, 1, 10, 10, 10)
        masks[0, 0, 3:7, 3:7, 3:7] = 1.0
        preds = torch.zeros(1, 1, 4, 4, 4)
        result = _crop_masks(masks, preds)
        assert result.shape == (1, 1, 4, 4, 4)
        assert result.sum().item() == 4 ** 3  # entire central cube

    def test_3d_multi_batch_channels(self):
        masks = torch.randn(3, 4, 32, 32, 32)
        preds = torch.randn(3, 4, 20, 20, 20)
        result = _crop_masks(masks, preds)
        assert result.shape == (3, 4, 20, 20, 20)

    def test_routing_2d_vs_3d(self):
        """4-dim tensors route to 2D branch, 5-dim to 3D branch."""
        m2 = torch.randn(1, 1, 32, 32)
        p2 = torch.randn(1, 1, 20, 20)
        assert _crop_masks(m2, p2).shape == (1, 1, 20, 20)

        m3 = torch.randn(1, 1, 32, 32, 32)
        p3 = torch.randn(1, 1, 20, 20, 20)
        assert _crop_masks(m3, p3).shape == (1, 1, 20, 20, 20)
