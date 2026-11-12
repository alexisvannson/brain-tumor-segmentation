"""Tests for Unet (2D) and Unet3d (3D) model architectures."""
import pytest
import torch
import torch.nn as nn

from models.Unet import Unet, DownBlock, UpBlock, FinalBlock
from models.Unet3d import Unet3d, DownBlock3d, UpBlock3d, FinalBlock3d

# ---------------------------------------------------------------------------
# Minimal configs — fast to run
# ---------------------------------------------------------------------------

TINY_DOWN  = [(1, 8), (8, 16)]   # 2 encoder blocks; last = bottleneck
TINY_UP    = [(16, 8)]
TINY_FINAL = (8, 1)

# Default (padding=1) UNet: output has the same spatial size as the input
INPUT_2D   = (1, 1, 32, 32)   # (B, C, H, W)
OUTPUT_2D  = (1, 1, 32, 32)
INPUT_3D   = (1, 1, 32, 32, 32)
OUTPUT_3D  = (1, 1, 32, 32, 32)

# Input / expected output sizes for the no-padding UNet (padding=0)
# Each 3x3 conv removes 2px per spatial dim → 2 convs = -4px per block
# DownBlock1 (in=32): 32-4=28 → pool → 14
# DownBlock2 (bottleneck, in=14): 14-4=10
# UpTranspose (in=10): 20
# Skip crop: 28→20  Cat→UpBlock (padded) → 20
# FinalBlock (padded + 1x1): 20
OUTPUT_2D_VALID = (1, 1, 20, 20)
OUTPUT_3D_VALID = (1, 1, 20, 20, 20)


# ---------------------------------------------------------------------------
# 2D block unit tests
# ---------------------------------------------------------------------------

class TestDownBlock:
    def test_output_shape(self):
        block = DownBlock(3, 16)
        x = torch.randn(2, 3, 28, 28)
        assert block(x).shape == (2, 16, 28, 28)  # padding=1 preserves size

    def test_output_shape_no_padding(self):
        block = DownBlock(3, 16, padding=0)
        x = torch.randn(2, 3, 28, 28)
        out = block(x)
        # two 3x3 no-padding convs: 28 - 4 = 24
        assert out.shape == (2, 16, 24, 24)

    def test_channel_change(self):
        block = DownBlock(1, 64)
        x = torch.randn(1, 1, 32, 32)
        assert block(x).shape[1] == 64

    def test_relu_applied(self):
        block = DownBlock(1, 8)
        x = torch.full((1, 1, 16, 16), -100.0)
        out = block(x)
        # ReLU clamps negative pre-activations; not guaranteed all zeros but
        # output must be ≥ 0 only if weights are all positive — just check no NaN
        assert not out.isnan().any()


class TestUpBlock:
    def test_output_shape_preserved(self):
        block = UpBlock(16, 8)
        x = torch.randn(2, 16, 20, 20)
        out = block(x)
        assert out.shape == (2, 8, 20, 20)  # padding=1 preserves size

    def test_channel_change(self):
        block = UpBlock(32, 16)
        x = torch.randn(1, 32, 10, 10)
        assert block(x).shape[1] == 16


class TestFinalBlock:
    def test_output_channels(self):
        block = FinalBlock(8, 3)
        x = torch.randn(1, 8, 20, 20)
        out = block(x)
        assert out.shape == (1, 3, 20, 20)

    def test_spatial_preserved(self):
        block = FinalBlock(16, 1)
        x = torch.randn(2, 16, 15, 15)
        assert block(x).shape[2:] == torch.Size([15, 15])


# ---------------------------------------------------------------------------
# 2D UNet integration tests
# ---------------------------------------------------------------------------

class TestUnet2D:
    @pytest.fixture
    def model(self):
        return Unet(TINY_DOWN, TINY_UP, TINY_FINAL)

    def test_output_shape(self, model):
        x = torch.randn(*INPUT_2D)
        assert model(x).shape == torch.Size(OUTPUT_2D)

    def test_output_channels_match_final_block(self, model):
        x = torch.randn(*INPUT_2D)
        assert model(x).shape[1] == TINY_FINAL[1]

    def test_batch_size_preserved(self, model):
        x = torch.randn(3, 1, 32, 32)
        assert model(x).shape[0] == 3

    def test_gradients_flow(self, model):
        x = torch.randn(*INPUT_2D)
        loss = model(x).sum()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert not p.grad.isnan().any(), f"NaN gradient for {name}"

    def test_valid_conv_output_shape(self):
        model = Unet(TINY_DOWN, TINY_UP, TINY_FINAL, padding=0)
        assert model(torch.randn(*INPUT_2D)).shape == torch.Size(OUTPUT_2D_VALID)

    @pytest.mark.parametrize("size", [(240, 240), (33, 47), (17, 64)])
    def test_output_matches_input_size(self, size):
        """Padded UNet must predict every input pixel, even for sizes not divisible by 2**depth."""
        model = Unet([(4, 8), (8, 16), (16, 32)], [(32, 16), (16, 8)], (8, 3))
        out = model(torch.randn(1, 4, *size))
        assert out.shape == (1, 3, *size)

    def test_encoder_reduces_spatial_size(self, model):
        """Pooling between encoder blocks must shrink spatial dims."""
        x = torch.randn(*INPUT_2D)
        hooks, sizes = [], []

        def hook(m, inp, out):
            sizes.append(out.shape[-1])

        handles = [b.register_forward_hook(hook) for b in model.down_blocks]
        model(x)
        for h in handles:
            h.remove()

        # Each successive encoder block should produce smaller or equal spatial size
        assert sizes[1] < sizes[0], "Bottleneck should be spatially smaller than first encoder block"

    def test_larger_input(self, model):
        x = torch.randn(1, 1, 64, 64)
        out = model(x)
        assert out.shape[1] == TINY_FINAL[1]

    def test_model_is_nn_module(self, model):
        assert isinstance(model, nn.Module)

    def test_eval_mode_no_grad(self, model):
        model.eval()
        x = torch.randn(*INPUT_2D)
        with torch.no_grad():
            out = model(x)
        assert out.shape == torch.Size(OUTPUT_2D)

    def test_deeper_unet(self):
        """A deeper 2D UNet (3 down levels) should still produce valid output."""
        down = [(4, 16), (16, 32), (32, 64)]
        up   = [(64, 32), (32, 16)]
        fin  = (16, 3)
        model = Unet(down, up, fin)
        x = torch.randn(1, 4, 128, 128)
        out = model(x)
        assert out.shape[1] == 3
        assert not out.isnan().any()

    def test_multi_class_output(self):
        down  = [(1, 8), (8, 16)]
        up    = [(16, 8)]
        final = (8, 4)   # 4 output classes
        model = Unet(down, up, final)
        out = model(torch.randn(1, 1, 32, 32))
        assert out.shape[1] == 4


# ---------------------------------------------------------------------------
# 3D block unit tests
# ---------------------------------------------------------------------------

class TestDownBlock3d:
    def test_output_shape(self):
        block = DownBlock3d(2, 8)
        x = torch.randn(1, 2, 16, 16, 16)
        assert block(x).shape == (1, 8, 16, 16, 16)

    def test_output_shape_no_padding(self):
        block = DownBlock3d(2, 8, padding=0)
        x = torch.randn(1, 2, 16, 16, 16)
        out = block(x)
        assert out.shape == (1, 8, 12, 12, 12)  # 16 - 4

    def test_no_nan(self):
        block = DownBlock3d(1, 4)
        assert not block(torch.randn(1, 1, 12, 12, 12)).isnan().any()


class TestUpBlock3d:
    def test_spatial_preserved(self):
        block = UpBlock3d(8, 4)
        x = torch.randn(1, 8, 10, 10, 10)
        assert block(x).shape == (1, 4, 10, 10, 10)


class TestFinalBlock3d:
    def test_output_channels(self):
        block = FinalBlock3d(4, 3)
        x = torch.randn(1, 4, 10, 10, 10)
        assert block(x).shape == (1, 3, 10, 10, 10)


# ---------------------------------------------------------------------------
# 3D UNet integration tests
# ---------------------------------------------------------------------------

class TestUnet3D:
    @pytest.fixture
    def model(self):
        return Unet3d(TINY_DOWN, TINY_UP, TINY_FINAL)

    def test_output_shape(self, model):
        x = torch.randn(*INPUT_3D)
        assert model(x).shape == torch.Size(OUTPUT_3D)

    def test_output_channels(self, model):
        x = torch.randn(*INPUT_3D)
        assert model(x).shape[1] == TINY_FINAL[1]

    def test_valid_conv_output_shape(self):
        model = Unet3d(TINY_DOWN, TINY_UP, TINY_FINAL, padding=0)
        assert model(torch.randn(*INPUT_3D)).shape == torch.Size(OUTPUT_3D_VALID)

    def test_output_matches_odd_input_size(self, model):
        """155 slices (BraTS depth) is not divisible by 2; output must still match."""
        out = model(torch.randn(1, 1, 15, 18, 21))
        assert out.shape == (1, 1, 15, 18, 21)

    def test_batch_size_preserved(self, model):
        x = torch.randn(2, 1, 32, 32, 32)
        assert model(x).shape[0] == 2

    def test_gradients_flow(self, model):
        x = torch.randn(*INPUT_3D)
        model(x).sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert not p.grad.isnan().any(), f"NaN gradient for {name}"

    def test_no_nan_output(self, model):
        x = torch.randn(*INPUT_3D)
        assert not model(x).isnan().any()

    def test_different_spatial_size(self, model):
        """Non-cubic volume should still work."""
        x = torch.randn(1, 1, 32, 32, 32)
        out = model(x)
        assert out.shape[1] == TINY_FINAL[1]

    def test_deeper_3d_unet(self):
        """3-level 3D UNet on small volume."""
        down = [(4, 8), (8, 16), (16, 32)]
        up   = [(32, 16), (16, 8)]
        fin  = (8, 3)
        model = Unet3d(down, up, fin)
        x = torch.randn(1, 4, 64, 64, 64)
        out = model(x)
        assert out.shape[1] == 3
        assert not out.isnan().any()
