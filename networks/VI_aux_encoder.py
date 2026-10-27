import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class AuxiliaryEncoder(nn.Module):
    def __init__(self, in_channels=5):
        super(AuxiliaryEncoder, self).__init__()
        # The output channels are chosen to match the feature dimensions of the main encoder's stages.
        self.initial_downsample = nn.Conv2d(in_channels, 144, kernel_size=4, stride=4)
        
        self.inc = DoubleConv(144, 144)
        self.down1 = Down(144, 288)
        self.down2 = Down(288, 576)
        self.down3 = Down(576, 1152)

    def forward(self, aux_input):
        # 1. Initial processing of auxiliary input
        x_downsampled = self.initial_downsample(aux_input)
        
        # 2. Extract features independently and sequentially
        ax1 = self.inc(x_downsampled)
        ax2 = self.down1(ax1)
        ax3 = self.down2(ax2)
        ax4 = self.down3(ax3)

        return ax1, ax2, ax3, ax4

if __name__ == '__main__':
    # Validate the four output scales using a five-channel auxiliary input.
    aux_encoder = AuxiliaryEncoder(in_channels=5).cuda()
    # Create a dummy 5-channel input tensor
    dummy_input = torch.randn(1, 5, 512, 512).cuda()
    
    with torch.no_grad():
        f1, f2, f3, f4 = aux_encoder(dummy_input)
    
    print(f"Shape of feature map 1: {f1.shape}")
    print(f"Shape of feature map 2: {f2.shape}")
    print(f"Shape of feature map 3: {f3.shape}")
    print(f"Shape of feature map 4: {f4.shape}")
