"""
Module containing the full UNet architecture.
"""

import torch.nn as nn

from src.unet.architecture_blocks import DoubleConv, Down, Up


class UNet(nn.Module):
    def __init__(self, n_output_channels: int = 1, bilinear: bool = False):
        """
        Input of model must have shape of: [1, 3, WIDTH, HEIGHT], where 3 is the channel size (e.g. RGB).

        :param n_output_channels: int -- Number of output channels of the model
        :param bilinear: bool -- Whether to use simple upsampling using bilinear transformation or ConvTranspose2d
        """
        super().__init__()
        self.bilinear = bilinear
        self.n_output_channels = n_output_channels

        # Encoder layers
        self.layer1 = DoubleConv(3, 64)
        self.layer2 = Down(64, 128)
        self.layer3 = Down(128, 256)
        self.layer4 = Down(256, 512)
        self.layer5 = Down(512, 1024)

        # Decoder layers
        self.layer6 = Up(1024, 512, bilinear=self.bilinear)
        self.layer7 = Up(512, 256, bilinear=self.bilinear)
        self.layer8 = Up(256, 128, bilinear=self.bilinear)
        self.layer9 = Up(128, 64, bilinear=self.bilinear)

        # Output layers with the models logits outputs and # of classes = # of channels
        self.layer10 = nn.Conv2d(64, self.n_output_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        # Decoder with contracting paths
        x6 = self.layer6(x5, x4)
        x6 = self.layer7(x6, x3)
        x6 = self.layer8(x6, x2)
        x6 = self.layer9(x6, x1)

        return self.layer10(x6)
