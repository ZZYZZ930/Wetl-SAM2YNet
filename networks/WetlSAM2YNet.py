import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from networks.VI_aux_encoder import AuxiliaryEncoder
from networks.ACGF import ACGF


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CrossAttentionFusion(nn.Module):
    """Context-aware cross-attention fusion module."""
    def __init__(self, in_channels):
        super(CrossAttentionFusion, self).__init__()
        # Convolutional layer to align and process concatenated features
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # Attention modules
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()
        
        # Zero initialization preserves the main feature stream at training onset.
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, main_feat, aux_feat):
        # 1. Concatenate main and auxiliary features to get context
        combined_feat = torch.cat([main_feat, aux_feat], dim=1)
        
        # 2. Generate attention gates based on joint context
        attention_context = self.conv(combined_feat)
        channel_gate = self.ca(attention_context)
        spatial_gate = self.sa(attention_context)
        
        # 3. Modulate auxiliary features with the joint attention gates
        refined_aux = aux_feat * channel_gate * spatial_gate
        
        # 4. Add the modulated auxiliary features through a scaled residual path
        fused_feat = main_feat + self.scale * refined_aux
        return fused_feat


class GatedFusion(nn.Module):
    """Gated fusion module for main and auxiliary features."""
    def __init__(self, in_channels):
        super(GatedFusion, self).__init__()
        # Convolutional layer to convert auxiliary features into attention gates with same channels as main features
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()  # Sigmoid restricts gate values between 0 and 1
        )
        # Zero initialization preserves the main feature stream at training onset.
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, main_feat, aux_feat):
        # 1. Generate attention gate from auxiliary features
        attention_gate = self.gate_conv(aux_feat)
        # 2. Gate main features and perform residual connection with original main features
        # The scaled gated residual is zero at initialization.
        fused_feat = main_feat + self.scale * (main_feat * attention_gate)
        return fused_feat


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
    
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # in_channels is the combined width of the upsampled and skip features.
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Align the upsampled tensor with the skip tensor before concatenation.
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net
    

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class WetlSAM2YNet(nn.Module):
    def __init__(self, checkpoint_path=None, num_classes=21):
        super(WetlSAM2YNet, self).__init__()    
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        # Instantiate auxiliary encoder
        self.aux_encoder = AuxiliaryEncoder(in_channels=5)

        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )
        
        # Multi-scale gated fusion between the SAM2 and auxiliary encoders.
        self.acgf1 = ACGF(144)
        self.acgf2 = ACGF(288)
        self.acgf3 = ACGF(576)
        self.acgf4 = ACGF(1152)

        # RFB modules
        self.rfb1 = RFB_modified(144, 64)
        self.rfb2 = RFB_modified(288, 64)
        self.rfb3 = RFB_modified(576, 64)
        self.rfb4 = RFB_modified(1152, 64)
        
        # Decoder input channels include the upsampled and skip features.
        # Up(decoder_in + skip_connection, decoder_out)
        # Each fused skip feature contains 64 channels after receptive-field processing.
        self.up1 = Up(64 + 64, 64) 
        self.up2 = Up(64 + 64, 64)
        self.up3 = Up(64 + 64, 64)
        
        # Multi-class segmentation head.
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x, aux_x):
        # Main encoder (R, G, NIR)
        x1, x2, x3, x4 = self.encoder(x)
        
        # Auxiliary encoder (B + 4 vegetation indices)
        ax1, ax2, ax3, ax4 = self.aux_encoder(aux_x)

        # Fuse corresponding feature scales from both encoders.
        f1 = self.acgf1(x1, ax1)
        f2 = self.acgf2(x2, ax2)
        f3 = self.acgf3(x3, ax3)
        f4 = self.acgf4(x4, ax4)
        
        # RFB modules
        f1_rfb, f2_rfb, f3_rfb, f4_rfb = self.rfb1(f1), self.rfb2(f2), self.rfb3(f3), self.rfb4(f4)
        
        # Decode the fused multi-scale features.
        d3 = self.up1(f4_rfb, f3_rfb)
        d2 = self.up2(d3, f2_rfb)
        d1 = self.up3(d2, f1_rfb)
        
        # Output
        out = self.head(d1)
        
        # Upsample to input size
        return F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)


if __name__ == "__main__":
    with torch.no_grad():
        # Validate output dimensions for 21 semantic classes.
        model = WetlSAM2YNet(num_classes=21).cuda()
        # Main input (R, G, NIR)
        x = torch.randn(1, 3, 512, 512).cuda()
        # Auxiliary input (B + 4 VIs)
        aux_x = torch.randn(1, 5, 512, 512).cuda()
        
        out = model(x, aux_x)
        print(out.shape)
