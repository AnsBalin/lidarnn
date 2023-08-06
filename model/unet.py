import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Contract(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Expand(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Final(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, features0=64):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        features1 = features0 * 2
        features2 = features1 * 2
        features3 = features2 * 2
        features4 = features3 * 2

        self.in0 = DoubleConv(in_channels, features0)
        self.contract1 = Contract(features0, features1)
        self.contract2 = Contract(features1, features2)
        self.contract3 = Contract(features2, features3)
        self.contract4 = Contract(features3, features4)

        self.expand3 = Expand(features4, features3)
        self.expand2 = Expand(features4, features3)
        self.expand1 = Expand(features4, features3)
        self.expand0 = Expand(features4, features3)

        self.out0 = Final(features4, out_channels)

    def forward(self, x):
        x0 = self.in0(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract2(x2)
        x = self.contract2(x3)

        x = self.expand3(x, x3)
        x = self.expand2(x, x2)
        x = self.expand1(x, x1)
        x = self.expand0(x, x0)

        logits = self.out0(x)
        return logits
