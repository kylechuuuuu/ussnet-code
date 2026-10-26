import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=4, kernel_size=1)
        nn.Sigmoid()

    def forward(self, x):
        return self.conv(x)


class Ubase(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, model_channels=64, bilinear=False):
        super(Ubase, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.model_channels = model_channels

        model_channels = self.model_channels
        self.inc = DoubleConv(n_channels, model_channels)
        self.down1 = Down(model_channels, model_channels * 2)
        self.down2 = Down(model_channels * 2, model_channels * 4)
        self.down3 = Down(model_channels * 4, model_channels * 8)
        self.down4 = Down(model_channels * 8, model_channels * 16)
        self.up1 = Up(model_channels * 16, model_channels * 8, bilinear)
        self.up2 = Up(model_channels * 8, model_channels * 4, bilinear)
        self.up3 = Up(model_channels * 4, model_channels * 2, bilinear)
        self.up4 = Up(model_channels * 2, model_channels * 1, bilinear)
        self.outc = OutConv(model_channels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits





