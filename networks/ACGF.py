import torch
import torch.nn as nn

class ACGF(nn.Module):
    def __init__(self, in_channels):
        """
        Args:
            in_channels (int): Number of channels in each input feature map.
        """
        super(ACGF, self).__init__()

        # Processing block for SAM features (F_SAM)
        # 1x1 Conv -> BatchNorm -> ReLU
        self.process_sam = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Processing block for auxiliary-encoder features (F_aux)
        # 1x1 Conv -> BatchNorm -> ReLU
        self.process_aux = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Sigmoid activation to create the attention map
        self.sigmoid = nn.Sigmoid()

        # Learnable scaling parameter, initialized to 0
        # This suppresses the attention-weighted auxiliary contribution at initialization.
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, f_sam, f_aux):
        """
        Args:
            f_sam (torch.Tensor): Feature map from the SAM branch.
            f_aux (torch.Tensor): Feature map from the auxiliary branch.
        
        Returns:
            torch.Tensor: The fused feature map.
        """
        # 1. Feature preprocessing
        f_sam_prime = self.process_sam(f_sam)
        f_aux_prime = self.process_aux(f_aux)

        # 2. Generate the attention map
        # Element-wise multiplication of the processed features
        attention_input = f_sam_prime * f_aux_prime
        # Pass through sigmoid to get the attention map 'A'
        attention_map = self.sigmoid(attention_input)

        # 3. Apply attention
        # Element-wise multiplication of the attention map with the processed auxiliary features
        f_att_aux = attention_map * f_aux_prime

        # 4. Fuse the feature streams
        # Element-wise addition with a learnable gate
        # The output is the SAM feature plus the scaled, attention-weighted auxiliary feature.
        # Initially alpha is 0, so F_out = F'_SAM
        fused_output = f_sam_prime + self.alpha * f_att_aux

        return fused_output

# Module-level shape validation.
if __name__ == '__main__':
    # Representative decoder-stage dimensions.
    in_channels = 64
    H, W = 128, 128
    batch_size = 4

    acgf = ACGF(in_channels=in_channels)
    print(acgf)

    # Synthetic feature maps for validating tensor dimensions.
    sam_features = torch.randn(batch_size, in_channels, H, W)
    aux_features = torch.randn(batch_size, in_channels, H, W)

    fused_features = acgf(sam_features, aux_features)

    print(f"\nInput shape (f_sam): {sam_features.shape}")
    print(f"Input shape (f_aux): {aux_features.shape}")
    print(f"Output shape (fused): {fused_features.shape}")

    # Fusion preserves the spatial and channel dimensions.
    assert fused_features.shape == sam_features.shape
    print("\nTest passed: Output shape is correct.")
