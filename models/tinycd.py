import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBlock(nn.Module):
    """Basic Convolutional Block with Conv2d -> BatchNorm2d -> GELU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_act=True):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        if use_act:
            layers.append(nn.GELU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution for minimal compute and fast inference."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style Channel Attention block."""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        reduced_channels = max(8, in_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.GELU(),
            nn.Linear(reduced_channels, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_out = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        att = self.sigmoid(avg_out + max_out)
        return x * att


class MultiScaleDifferenceModule(nn.Module):
    """
    Computes absolute differences and cross-features between T1 and T2 feature representations,
    then applies depthwise separable convolutions and channel attention.
    """
    def __init__(self, in_channels, out_channels):
        super(MultiScaleDifferenceModule, self).__init__()
        # Input has concatenated: [f1, f2, |f1 - f2|] => 3 * in_channels
        self.fusion = DepthwiseSeparableConv(in_channels * 3, out_channels)
        self.ca = ChannelAttention(out_channels)

    def forward(self, f1, f2):
        diff = torch.abs(f1 - f2)
        concat = torch.cat([f1, f2, diff], dim=1)
        out = self.fusion(concat)
        out = self.ca(out)
        return out


class TinyCD(nn.Module):
    """
    TinyCD: A Lightweight Siamese Network for Fast and Accurate Change Detection.
    
    Paper Concept: Andrea Codegoni et al., "TinyCD: A (Not So) Deep Learning Approach for Change Detection"
    
    Key Features:
    - Shared Siamese Backbone (EfficientNet-B0 or EfficientNet-B4) for multi-scale feature extraction.
    - Low computational footprint (<5M parameters with EfficientNet-B0 backbone).
    - Multi-scale difference interaction module.
    - Progressive top-down decoder with skip connections.
    """
    def __init__(self, backbone_name='efficientnet_b0', pretrained=True, num_classes=1):
        super(TinyCD, self).__init__()
        self.backbone_name = backbone_name
        
        # Load backbone
        if backbone_name == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            # Stage channels for EfficientNet-B0:
            # stage 1 (1/2): features[1] -> 16
            # stage 2 (1/4): features[2] -> 24
            # stage 3 (1/8): features[3] -> 40
            # stage 4 (1/16): features[5] -> 112
            # stage 5 (1/32): features[7] -> 320
            self.enc_channels = [24, 40, 112, 320] # 1/4, 1/8, 1/16, 1/32
            self.stage0 = backbone.features[0:2] # 1/2
            self.stage1 = backbone.features[2:3] # 1/4 (24 channels)
            self.stage2 = backbone.features[3:4] # 1/8 (40 channels)
            self.stage3 = backbone.features[4:6] # 1/16 (112 channels)
            self.stage4 = backbone.features[6:8] # 1/32 (320 channels)
        elif backbone_name == 'efficientnet_b4':
            weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b4(weights=weights)
            self.enc_channels = [32, 56, 160, 448]
            self.stage0 = backbone.features[0:2]
            self.stage1 = backbone.features[2:3]
            self.stage2 = backbone.features[3:4]
            self.stage3 = backbone.features[4:6]
            self.stage4 = backbone.features[6:8]
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Choose 'efficientnet_b0' or 'efficientnet_b4'.")

        # Multi-scale feature interaction & difference modules
        dec_dim = 64
        self.diff_modules = nn.ModuleList([
            MultiScaleDifferenceModule(self.enc_channels[0], dec_dim),       # 1/4
            MultiScaleDifferenceModule(self.enc_channels[1], dec_dim),       # 1/8
            MultiScaleDifferenceModule(self.enc_channels[2], dec_dim * 2),   # 1/16
            MultiScaleDifferenceModule(self.enc_channels[3], dec_dim * 2),   # 1/32
        ])

        # Progressive Decoder
        self.dec4_to_3 = ConvBlock(dec_dim * 2, dec_dim * 2)
        self.fuse_3 = ConvBlock(dec_dim * 2 + dec_dim * 2, dec_dim * 2)

        self.dec3_to_2 = ConvBlock(dec_dim * 2, dec_dim)
        self.fuse_2 = ConvBlock(dec_dim + dec_dim, dec_dim)

        self.dec2_to_1 = ConvBlock(dec_dim, dec_dim)
        self.fuse_1 = ConvBlock(dec_dim + dec_dim, dec_dim)

        # Final full-resolution upsampling head (1/4 -> 1/1)
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 1/4 -> 1/2
            ConvBlock(dec_dim, dec_dim // 2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 1/2 -> 1/1
            ConvBlock(dec_dim // 2, dec_dim // 4),
            nn.Conv2d(dec_dim // 4, num_classes, kernel_size=1)
        )

    def extract_features(self, x):
        """Extract multi-scale features from a single temporal image."""
        x0 = self.stage0(x)  # 1/2
        f1 = self.stage1(x0) # 1/4
        f2 = self.stage2(f1) # 1/8
        f3 = self.stage3(f2) # 1/16
        f4 = self.stage4(f3) # 1/32
        return [f1, f2, f3, f4]

    def forward(self, t1, t2):
        """
        Forward pass for paired temporal images.
        t1: (B, 3, H, W)
        t2: (B, 3, H, W)
        Returns:
        logits: (B, num_classes, H, W) (raw logits before sigmoid)
        """
        # 1. Siamese multi-scale feature extraction
        feats1 = self.extract_features(t1)
        feats2 = self.extract_features(t2)

        # 2. Multi-scale difference extraction
        d1 = self.diff_modules[0](feats1[0], feats2[0]) # 1/4, dim: 64
        d2 = self.diff_modules[1](feats1[1], feats2[1]) # 1/8, dim: 64
        d3 = self.diff_modules[2](feats1[2], feats2[2]) # 1/16, dim: 128
        d4 = self.diff_modules[3](feats1[3], feats2[3]) # 1/32, dim: 128

        # 3. Top-down Decoder path
        # 1/32 -> 1/16
        up4 = F.interpolate(d4, size=d3.shape[2:], mode='bilinear', align_corners=False)
        p3 = self.fuse_3(torch.cat([self.dec4_to_3(up4), d3], dim=1))

        # 1/16 -> 1/8
        up3 = F.interpolate(p3, size=d2.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.fuse_2(torch.cat([self.dec3_to_2(up3), d2], dim=1))

        # 1/8 -> 1/4
        up2 = F.interpolate(p2, size=d1.shape[2:], mode='bilinear', align_corners=False)
        p1 = self.fuse_1(torch.cat([self.dec2_to_1(up2), d1], dim=1))

        # 4. Final Prediction Map (1/4 -> 1/1)
        out = self.final_upsample(p1)
        
        # Ensure output matches input dimensions if odd padding
        if out.shape[2:] != t1.shape[2:]:
            out = F.interpolate(out, size=t1.shape[2:], mode='bilinear', align_corners=False)
            
        return out


def count_parameters(model):
    """Utility to count trainable parameters in Millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


if __name__ == '__main__':
    # Unit test for architecture verification
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TinyCD(backbone_name='efficientnet_b0', pretrained=False).to(device)
    t1 = torch.randn(2, 3, 256, 256).to(device)
    t2 = torch.randn(2, 3, 256, 256).to(device)
    out = model(t1, t2)
    print(f"TinyCD (EfficientNet-B0) created successfully!")
    print(f"Trainable Parameters: {count_parameters(model):.2f} M")
    print(f"Input shape: {t1.shape} -> Output shape: {out.shape}")
    assert out.shape == (2, 1, 256, 256), f"Expected shape (2, 1, 256, 256), got {out.shape}"
