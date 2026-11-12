import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def _parse(layer):
    return layer if isinstance(layer, tuple) else (layer, layer)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, padding: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=padding)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class FinalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1)
        self.final = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.final(x)
        return x


class Unet(nn.Module):
    """2D U-Net.

    With ``padding=1`` (default) the encoder keeps spatial size, inputs are
    zero-padded to a multiple of 2**n_pools and the output is cropped back, so
    the prediction covers the full input. ``padding=0`` reproduces the original
    valid-convolution U-Net, whose output is a centre crop of the input.
    """
    spatial_dims = 2

    def __init__(self, down_layers, up_layers, final_block, padding: int = 1):
        super().__init__()
        self.padding = padding
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder — last block is the bottleneck (no pool after it)
        self.down_blocks = nn.ModuleList([
            DownBlock(*_parse(layer), padding=padding) for layer in down_layers
        ])

        # Decoder — upsample then merge skip connection
        self.up_transposes = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for layer in up_layers:
            in_ch, out_ch = _parse(layer)
            self.up_transposes.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            # After center-crop and concat with skip, input channels to UpBlock double
            self.up_blocks.append(UpBlock(2 * out_ch, out_ch))

        final_in, final_out = final_block
        self.final = FinalBlock(final_in, final_out)

    def _crop_skip(self, skip, target):
        """Center-crop skip connection to match target spatial dimensions."""
        _, _, h, w = target.shape
        _, _, sh, sw = skip.shape
        top  = (sh - h) // 2
        left = (sw - w) // 2
        return TF.crop(skip, top=top, left=left, height=h, width=w)

    def forward(self, x):
        size = x.shape[2:]
        if self.padding:
            m = 2 ** (len(self.down_blocks) - 1)
            ph, pw = (-size[0]) % m, (-size[1]) % m
            x = F.pad(x, (0, pw, 0, ph))

        # Encoder: save skip connections before pooling
        skips = []
        for down_block in self.down_blocks[:-1]:
            x = down_block(x)
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
            x = x[:, :, :size[0], :size[1]]
        return x


if __name__ == "__main__":
    import yaml

    with open("configs/unet.yaml") as f:
        cfg = yaml.safe_load(f)["model"]

    down_layers = [tuple(l) for l in cfg["down_layers"]]
    up_layers   = [tuple(l) for l in cfg["up_layers"]]
    final_block = tuple(cfg["final_block"])

    model = Unet(down_layers, up_layers, final_block)

    tensor = torch.rand(1, 4, 240, 240)
    pred = model(tensor)
    print(pred.shape)
