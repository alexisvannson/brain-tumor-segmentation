import torch
import torch.nn as nn
import torch.nn.functional as F


class DownBlock3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, padding: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=padding)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class UpBlock3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class FinalBlock3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1)
        self.final = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.final(x)


class Unet3d(nn.Module):
    """3D U-Net. See `models.Unet.Unet` for the meaning of ``padding``."""
    spatial_dims = 3

    def __init__(self, down_layers, up_layers, final_block, padding: int = 1):
        super().__init__()
        self.padding = padding
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder — last block is the bottleneck (no pool after it)
        self.down_blocks = nn.ModuleList([
            DownBlock3d(*layer, padding=padding) for layer in down_layers
        ])

        # Decoder — upsample then merge skip connection
        self.up_transposes = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for in_ch, out_ch in up_layers:
            self.up_transposes.append(nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2))
            # After center-crop and concat with skip, input channels double
            self.up_blocks.append(UpBlock3d(2 * out_ch, out_ch))

        final_in, final_out = final_block
        self.final = FinalBlock3d(final_in, final_out)

    def _crop_skip(self, skip, target):
        """Center-crop skip connection to match target spatial dimensions (D, H, W)."""
        _, _, td, th, tw = target.shape
        _, _, sd, sh, sw = skip.shape
        d0 = (sd - td) // 2
        h0 = (sh - th) // 2
        w0 = (sw - tw) // 2
        return skip[:, :, d0:d0+td, h0:h0+th, w0:w0+tw]

    def forward(self, x):
        size = x.shape[2:]
        if self.padding:
            m = 2 ** (len(self.down_blocks) - 1)
            pd, ph, pw = ((-s) % m for s in size)
            x = F.pad(x, (0, pw, 0, ph, 0, pd))

        # Encoder: save skip connections before pooling
        skips = []
        for block in self.down_blocks[:-1]:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.down_blocks[-1](x)

        # Decoder
        for up_t, up_b, skip in zip(self.up_transposes, self.up_blocks, reversed(skips)):
            x = up_t(x)
            skip = self._crop_skip(skip, x)
            x = torch.cat((x, skip), dim=1)
            x = up_b(x)

        x = self.final(x)
        if self.padding:
            x = x[:, :, :size[0], :size[1], :size[2]]
        return x


if __name__ == "__main__":
    import yaml

    with open("configs/unet3d.yaml") as f:
        cfg = yaml.safe_load(f)["model"]

    down_layers = [tuple(l) for l in cfg["down_layers"]]
    up_layers   = [tuple(l) for l in cfg["up_layers"]]
    final_block = tuple(cfg["final_block"])

    model = Unet3d(down_layers, up_layers, final_block)
    tensor = torch.rand(1, 4, 64, 64, 64)
    pred = model(tensor)
    print(pred.shape)
