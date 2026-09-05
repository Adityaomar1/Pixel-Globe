import torch
import torch.nn as nn


# ============================================================
# DOUBLE CONVOLUTION BLOCK
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# SIAMESE U-NET FOR CHANGE DETECTION
# ============================================================

class ChangeDetectionModel(nn.Module):

    def __init__(self):
        super().__init__()

        # ====================================================
        # SHARED ENCODER
        # ====================================================

        self.enc1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        # ====================================================
        # BOTTLENECK
        # ====================================================

        self.bottleneck = DoubleConv(256, 512)

        # ====================================================
        # DECODER
        # ====================================================

        self.up4 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(64, 32)

        # ====================================================
        # FINAL OUTPUT
        # ====================================================

        self.output = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    # ========================================================
    # ENCODER FOR ONE IMAGE
    # ========================================================

    def encode(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        e3 = self.enc3(
            self.pool2(e2)
        )

        e4 = self.enc4(
            self.pool3(e3)
        )

        b = self.bottleneck(
            self.pool4(e4)
        )

        return e1, e2, e3, e4, b

    # ========================================================
    # FORWARD PASS
    # ========================================================

    def forward(self, img1, img2):

        # ----------------------------------------------------
        # Encode both images using the SAME encoder
        # ----------------------------------------------------

        e1_1, e2_1, e3_1, e4_1, b1 = self.encode(img1)

        e1_2, e2_2, e3_2, e4_2, b2 = self.encode(img2)

        # ----------------------------------------------------
        # Feature differences
        # ----------------------------------------------------

        d1 = torch.abs(e1_1 - e1_2)
        d2 = torch.abs(e2_1 - e2_2)
        d3 = torch.abs(e3_1 - e3_2)
        d4 = torch.abs(e4_1 - e4_2)
        db = torch.abs(b1 - b2)

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        x = self.up4(db)

        x = torch.cat(
            [x, d4],
            dim=1
        )

        x = self.dec4(x)

        x = self.up3(x)

        x = torch.cat(
            [x, d3],
            dim=1
        )

        x = self.dec3(x)

        x = self.up2(x)

        x = torch.cat(
            [x, d2],
            dim=1
        )

        x = self.dec2(x)

        x = self.up1(x)

        x = torch.cat(
            [x, d1],
            dim=1
        )

        x = self.dec1(x)

        # ----------------------------------------------------
        # Output logits
        # ----------------------------------------------------

        x = self.output(x)

        return x