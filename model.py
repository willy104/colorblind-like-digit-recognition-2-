import torch.nn as nn
import config as cfg


class ConvBlock(nn.Module):
    """Convolution + BatchNorm + LeakyReLU (+ optional MaxPool)."""

    def __init__(self, in_ch, out_ch, ks=3, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=ks, padding=ks // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CNN(nn.Module):
    """CNN for digit recognition (classes 0–9).

    Architecture:
      5 ConvBlocks (3→16→32→64→128→256) with MaxPool after each of the
      first 4 blocks, followed by AdaptiveAvgPool2d(1,1) to collapse
      spatial dimensions, then two FC layers with Dropout for
      regularisation.
    """

    def __init__(self, num_classes=cfg.NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            # 256×256 → 128×128
            ConvBlock(3, 16, pool=True),
            # 128×128 → 64×64
            ConvBlock(16, 32, pool=True),
            # 64×64 → 32×32
            ConvBlock(32, 64, pool=True),
            # 32×32 → 16×16
            ConvBlock(64, 128, pool=True),
            # 16×16, no additional pool
            ConvBlock(128, 256, pool=False),
        )
        # Global average pooling → (batch, 256, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
