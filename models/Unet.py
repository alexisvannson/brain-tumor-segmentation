import torch
import torch.nn as nn
import torchvision.transforms.functional as torchvision

class DownBlock(nn.Module):
    def __init__(self, input_channels: int, out_value=None):
        super(DownBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = out_value if out_value else 2 * self.input_channels
        self.conv1 = nn.Conv2d(self.input_channels, self.output_channels, kernel_size=3) #padding=1
        self.conv2 = nn.Conv2d(self.output_channels, self.output_channels, kernel_size=3) # padding=1
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        shape = x.shape
        print(shape)
        x = self.relu(self.conv1(x))
        shape = x.shape
        print(shape)
        x = self.relu(self.conv2(x))
        shape = x.shape
        print(shape)
        return x

class UpBlock(nn.Module):
    def __init__(self, input_channels: int, out_value=None):
        super(UpBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = out_value if out_value else int(self.input_channels) // 2
        self.conv1 = nn.Conv2d(self.input_channels, self.output_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.output_channels, self.output_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class FinalBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FinalBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv1 = nn.Conv2d(self.input_channels, self.output_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.output_channels, self.output_channels, kernel_size=3, padding=1)
        self.final = nn.Conv2d(self.output_channels, self.output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.final(x)
        return x

def _parse(layer):
      return layer if isinstance(layer, tuple) else (layer, layer)

class Unet(nn.Module):
    def __init__(self, layers, bottleneck_block, final_block):
        super(Unet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        down_layers, up_layers = layers
        down_blocks = []
        for layer in down_layers:
            in_ch, out_ch = _parse(layer)
            down_blocks.append(DownBlock(in_ch, out_ch))
            if layer != down_layers[-1]:
                down_blocks.append(self.pool)
        self.downwards = nn.ModuleList(down_blocks)

        up_blocks = []
        for layer in up_layers[:-1]:
            in_ch, out_ch = _parse(layer)
            up_blocks.append(UpBlock(in_ch, out_ch))
            up_blocks.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
        last = up_layers[-1]
        up_blocks.append(UpBlock(last))
        up_blocks.append(nn.Conv2d(last, last, kernel_size=1))
        self.upwards = nn.ModuleList(up_blocks)
        final_in, final_out = final_block
        self.final = FinalBlock(final_in, final_out)
   
    def forward(self, x):
        output_values = []
        output_shapes = []
        for layer in self.downwards:
            x = layer(x)
            output_values += [x]
            output_shapes += [x.shape]

        for idx, layer in enumerate(self.upwards):
            
            down_image = output_values[-( idx + 1)]
            _, _ , down_width, down_height = down_image.shape
            _, _, current_width, current_height = x.shape
            delta1 , delta2 = (down_height - current_height) // 2 , (down_width - current_width) // 2

            residual = torchvision.crop(img=down_image, top=delta1, left=delta2, height=current_height, width=current_width)
            print("residual shape: ", residual.shape)
            print("current tensor shape: ", x.shape)
            x = torch.concat((x, residual), dim=1)
            x = layer(x)
            print(x.shape)
       
        return  x

if __name__ == "__main__":
    layers = ([(1, 64), 64, 128, 256], [1024, 512, 256, 128])
    dense_block = 512
    final_block = (64, 2)
    model = Unet(layers, dense_block, final_block)

    tensor = torch.rand(1, 1, 572, 572)

    pred = model(tensor)
    print(pred.shape)