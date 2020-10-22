"""
Name: modules_liu.py
Desc: This script contains the base modules and networks writen by liu Li.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class basic_conv_double(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(basic_conv_double, self).__init__()

        if not mid_channels:
            mid_channels = out_channels
        self.conv_double = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_double(x)


class UNet_down_block_liu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_down_block_liu, self).__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            basic_conv_double(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UNet_up_block_liu(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UNet_up_block_liu, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = basic_conv_double(in_channels, out_channels, mid_channels=in_channels//2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)         # w,h//2
            self.conv = basic_conv_double(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class outConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(outConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Encoder_liu(nn.Module):
    def __init__(self, in_channels, bilinear=True):
        super(Encoder_liu, self).__init__()

        self.inc = basic_conv_double(in_channels, 64)
        self.down1 = UNet_down_block_liu(64, 128)
        self.down2 = UNet_up_block_liu(128, 256)
        self.down3 = UNet_down_block_liu(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = UNet_down_block_liu(512, 1024 // factor)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5


class Decoder_liu(nn.Module):
    def __init__(self, out_channels, bilinear=True):
        super(Decoder_liu, self).__init__()

        factor = 2 if bilinear else 1
        self.up1 = UNet_up_block_liu(1024, 512 // factor, bilinear)
        self.up2 = UNet_up_block_liu(512, 256 // factor, bilinear)
        self.up3 = UNet_up_block_liu(256, 128 // factor, bilinear)
        self.up4 = UNet_up_block_liu(128, 64, bilinear)
        self.outc = outConv(64, out_channels)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)     # 64 channel
        logits = self.outc(x)
        return x, logits
