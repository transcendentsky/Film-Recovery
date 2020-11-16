"""
Name: modules.py
Desc: This script defines some base module for building networks.
"""
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet_down_block(nn.Module):

    def __init__(self, input_channel, output_channel, down_size=True):
        super(UNet_down_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, output_channel)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.down_size = down_size

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        if self.down_size:
            x = self.max_pool(x)
        return x

class UNet_up_block(nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel, up_sample=True):
        super(UNet_up_block, self).__init__()
        self.up_sampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, output_channel)
        self.relu = torch.nn.ReLU()
        self.up_sample = up_sample

    def forward(self, prev_feature_map, x):
        if self.up_sample:
            x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x

class UNet(nn.Module):
    def __init__(self,  downsample=6, in_channels=3, out_channels=3):
        super(UNet, self).__init__()

        self.in_channels, self.out_channels, self.downsample = in_channels, out_channels, downsample
        self.down1 = UNet_down_block(in_channels, 16, False)
        self.down_blocks = nn.ModuleList(
            [UNet_down_block(2**(4+i), 2**(5+i), True) for i in range(0, downsample)]
        )

        bottleneck = 2**(4 + downsample)
        self.mid_conv1 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, bottleneck)
        self.mid_conv2 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, bottleneck)
        self.mid_conv3 = torch.nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, bottleneck)

        self.up_blocks = nn.ModuleList(
            [UNet_up_block(2**(4+i), 2**(5+i), 2**(4+i)) for i in range(0, downsample)]
        )

        self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = nn.GroupNorm(8, 16)
        self.last_conv2 = nn.Conv2d(16, out_channels, 1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.down1(x)
        xvals = [x]
        for i in range(0, self.downsample):
            x = self.down_blocks[i](x)
            xvals.append(x)

        x = self.relu(self.bn1(self.mid_conv1(x)))
        x = self.relu(self.bn2(self.mid_conv2(x)))
        x = self.relu(self.bn3(self.mid_conv3(x)))

        for i in range(0, self.downsample)[::-1]:
            x = self.up_blocks[i](xvals[i], x)

        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.relu(self.last_conv2(x))
        #x = self.last_conv2(x)
        return x

'''
class UNetDepth(nn.Module):
    def __init__(self):
        super(UNetDepth, self).__init__()

        self.down_block1 = UNet_down_block(3, 16, False)
        self.down_block2 = UNet_down_block(16, 32, True)
        self.down_block3 = UNet_down_block(32, 64, True)
        self.down_block4 = UNet_down_block(64, 128, True)
        self.down_block5 = UNet_down_block(128, 256, True)
        self.down_block6 = UNet_down_block(256, 512, True)
        self.down_block7 = UNet_down_block(512, 1024, False)

        self.mid_conv1 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, 1024)
        self.mid_conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, 1024)
        self.mid_conv3 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn3 = torch.nn.GroupNorm(8, 1024)

        self.up_block1 = UNet_up_block(512, 1024, 512, False)
        self.up_block2 = UNet_up_block(256, 512, 256, True)
        self.up_block3 = UNet_up_block(128, 256, 128, True)
        self.up_block4 = UNet_up_block(64, 128, 64, True)
        self.up_block5 = UNet_up_block(32, 64, 32, True)
        self.up_block6 = UNet_up_block(16, 32, 16, True)

        self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = nn.GroupNorm(8, 16)
        self.last_conv2 = nn.Conv2d(16, 1, 1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.x1 = self.down_block1(x)
        x = self.x2 = self.down_block2(self.x1)
        x = self.x3 = self.down_block3(self.x2)
        x = self.x4 = self.down_block4(self.x3)
        x = self.x5 = self.down_block5(self.x4)
        x = self.x6 = self.down_block6(self.x5)
        x = self.x7 = self.down_block7(self.x6)

        x = self.relu(self.bn1(self.mid_conv1(x)))
        x = self.relu(self.bn2(self.mid_conv2(x)))
        x = self.relu(self.bn3(self.mid_conv3(x)))

        x = self.up_block1(self.x6, x)
        x = self.up_block2(self.x5, x)
        x = self.up_block3(self.x4, x)
        x = self.up_block4(self.x3, x)
        x = self.up_block5(self.x2, x)
        x = self.up_block6(self.x1, x)
        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)
        return x
'''

class UNetDepth(nn.Module):
    def __init__(self):
        super(UNetDepth, self).__init__()

        self.down_block1 = UNet_down_block(3, 16, False)
        self.down_block2 = UNet_down_block(16, 32, True)
        self.down_block3 = UNet_down_block(32, 64, True)
        self.down_block4 = UNet_down_block(64, 128, True)
        self.down_block5 = UNet_down_block(128, 256, True)
        self.down_block6 = UNet_down_block(256, 512, False)

        self.mid_conv1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, 512)
        self.mid_conv2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, 512)
        self.mid_conv3 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.bn3 = torch.nn.GroupNorm(8, 512)

        self.up_block1 = UNet_up_block(256, 512, 256, False)
        self.up_block2 = UNet_up_block(128, 256, 128, True)
        self.up_block3 = UNet_up_block(64, 128, 64, True)
        self.up_block4 = UNet_up_block(32, 64, 32, True)
        self.up_block5 = UNet_up_block(16, 32, 16, True)

        self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = nn.GroupNorm(8, 16)
        self.last_conv2 = nn.Conv2d(16, 1, 1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.x1 = self.down_block1(x)
        x = self.x2 = self.down_block2(self.x1)
        x = self.x3 = self.down_block3(self.x2)
        x = self.x4 = self.down_block4(self.x3)
        x = self.x5 = self.down_block5(self.x4)
        x = self.x6 = self.down_block6(self.x5)

        x = self.relu(self.bn1(self.mid_conv1(x)))
        x = self.relu(self.bn2(self.mid_conv2(x)))
        x = self.relu(self.bn3(self.mid_conv3(x)))

        x = self.up_block1(self.x5, x)
        x = self.up_block2(self.x4, x)
        x = self.up_block3(self.x3, x)
        x = self.up_block4(self.x2, x)
        x = self.up_block5(self.x1, x)
        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)
        return x

class UNet_sim(nn.Module):
    def __init__(self, downsample=4, in_channels=3, out_channels=3):
        super(UNet_sim, self).__init__()
        self.downsample, self.in_channels, self.out_channels = downsample, in_channels, out_channels
        self.conv = ConvBlock(in_channels, 64)
        self.down_blocks = nn.ModuleList(
            [UNet_down_block(2 ** (6 + i), 2 ** (7 + i), True) for i in range(0, downsample)]
        )
        bottleneck = 2 ** (6 + downsample)
        self.mid_conv1 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, bottleneck)
        self.mid_conv2 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, bottleneck)
        self.mid_conv3 = torch.nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, bottleneck)
        self.up_blocks = nn.ModuleList(
            [UNet_up_block(2 ** (6 + i), 2 ** (7 + i), 2 ** (6 + i)) for i in range(0, downsample)]
        )
        self.last_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.last_bn = nn.GroupNorm(8, 64)
        self.last_conv2 = nn.Conv2d(64, out_channels, 1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        xvals = [x]
        for i in range(0, self.downsample):
            x = self.down_blocks[i](x)
            xvals.append(x)
        x = self.relu(self.bn1(self.mid_conv1(x)))
        x = self.relu(self.bn2(self.mid_conv2(x)))
        x = self.relu(self.bn3(self.mid_conv3(x)))
        for i in range(0, self.downsample)[::-1]:
            x = self.up_blocks[i](xvals[i], x)
        x = self.last_bn(self.last_conv1(x))
        x = self.last_conv2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, downsample=6, in_channels=3):
        """:downsample the number of down blocks
           :in_channels the channel of input tensor
        """
        super(Encoder, self).__init__()
        self.in_channels, self.downsample = in_channels, downsample
        self.down1 = UNet_down_block(in_channels, 16, False)
        self.down_blocks = nn.ModuleList(
            [UNet_down_block(2 ** (4 + i), 2 ** (5 + i), True) for i in range(0, downsample)]
        )
        bottleneck = 2 ** (4 + downsample)
        self.mid_conv1 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, bottleneck)
        self.mid_conv2 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, bottleneck)
        self.mid_conv3 = torch.nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, bottleneck)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.down1(x)
        xvals = [x]
        for i in range(0, self.downsample):
            x = self.down_blocks[i](x)
            xvals.append(x)
        x = self.relu(self.bn1(self.mid_conv1(x)))
        x = self.relu(self.bn2(self.mid_conv2(x)))
        x = self.relu(self.bn3(self.mid_conv3(x)))
        return xvals, x

class Decoder(nn.Module):
    def __init__(self, downsample, out_channels, combine_num=0):
        super(Decoder, self).__init__()
        self.out_channels, self.downsample = out_channels, downsample
        self.combine_num = combine_num
        self.up_blocks = nn.ModuleList(
            [UNet_up_block(2 ** (4 + i), 2 ** (5 + i), 2 ** (4 + i)) for i in range(0, self.downsample)])
        self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = nn.GroupNorm(8, 16)
        self.last_conv2 = nn.Conv2d(16, self.out_channels, 1, padding=0)
        self.up_sampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.relu = nn.ReLU()

    def forward(self, xvals, x):
        devals = []
        for i in range(0, self.downsample)[::-1]:
            x = self.up_blocks[i](xvals[i], x)
            if i < self.combine_num:
                devals.append(x)
        y = self.last_bn(self.last_conv1(x))
        y = self.last_conv2(x)
        if len(devals) > 0:
            for j, decode in enumerate(devals):
                for _ in range(len(devals) - 1 - j):
                    decode = self.up_sampling(decode)
                devals[j] = decode
            combine_x = torch.cat(devals[::-1], dim=1)
            return y, combine_x
        else:
            return y, x


class Encoder_sim(nn.Module):
    def __init__(self, downsample=4, in_channels=3):
        super(Encoder_sim, self).__init__()
        self.downsample = downsample
        self.conv = ConvBlock(in_channels, 64)
        self.down_blocks = nn.ModuleList(
            [UNet_down_block(2 ** (6 + i), 2 ** (7 + i), True) for i in range(0, downsample)]
        )
        bottleneck = 2 ** (6 + self.downsample)
        self.mid_conv1 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, bottleneck)
        self.mid_conv2 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, bottleneck)
        self.mid_conv3 = torch.nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, bottleneck)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        xvals = [x]
        for i in range(0, self.downsample):
            x = self.down_blocks[i](x)
            xvals.append(x)
        x = self.relu(self.bn1(self.mid_conv1(x)))
        x = self.relu(self.bn2(self.mid_conv2(x)))
        x = self.relu(self.bn3(self.mid_conv3(x)))
        return xvals, x

class Decoder_sim(nn.Module):
    def __init__(self, downsample, out_channels):
        super(Decoder_sim, self).__init__()
        self.downsample, self.out_channels = downsample, out_channels
        self.up_blocks = nn.ModuleList(
            [UNet_up_block(2 ** (6 + i), 2 ** (7 + i), 2 ** (6 + i)) for i in range(0, self.downsample)]
        )
        self.last_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.last_bn = nn.GroupNorm(8, 64)
        self.last_conv2 = nn.Conv2d(64, self.out_channels, 1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, xvals, x):
        for i in range(0, self.downsample)[::-1]:
            x = self.up_blocks[i](xvals[i], x)
        y = self.last_bn(self.last_conv1(x))
        y = self.last_conv2(y)
        return y, x


class ThreeD2NorDepth(nn.Module):
    def __init__(self, downsample=3, use_simple=True):
        super(ThreeD2NorDepth, self).__init__()
        if use_simple:
            self.threeD_encoder = Encoder_sim(downsample=downsample, in_channels=3)
            self.normal_decoder = Decoder_sim(downsample=downsample, out_channels=3)
            self.depth_decoder = Decoder_sim(downsample=downsample, out_channels=1)
        else:
            self.threeD_encoder = Encoder(downsample=downsample, in_channels=3)
            self.normal_decoder = Decoder(downsample=downsample, out_channels=3, combine_num=0)
            self.depth_decoder = Decoder(downsample=downsample, out_channels=1, combine_num=0)

    def forward(self, x):
        xvals, x = self.threeD_encoder(x)
        nor, _ = self.normal_decoder(xvals, x)
        dep, _ = self.depth_decoder(xvals, x)
        return nor, dep

class AlbedoDecoder_sim(nn.Module):
    def __init__(self, downsample=6, out_channels=1):
        super(AlbedoDecoder_sim, self).__init__()
        self.out_channels, self.downsample = out_channels, downsample
        self.up_blocks = nn.ModuleList(
            [UNet_up_block(2 ** (7 + i), 2 ** (8 + i), 2 ** (7 + i)) for i in range(0, self.downsample)])
        self.last_conv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.last_bn = nn.GroupNorm(8, 64)
        self.last_conv2 = nn.Conv2d(64, self.out_channels, 1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, xvals, x):
        for i in range(0, self.downsample)[::-1]:
            x = self.up_blocks[i](xvals[i], x)
        y = self.last_bn(self.last_conv1(x))
        y = self.last_conv2(y)
        return y, x

class AlbedoDecoder(nn.Module):
    def __init__(self, downsample=6, out_channels=1):
        super(AlbedoDecoder, self).__init__()
        self.out_channels, self.downsample = out_channels, downsample
        self.up_blocks = nn.ModuleList(
            [UNet_up_block(2 ** (5 + i), 2 ** (6 + i), 2 ** (5 + i)) for i in range(0, self.downsample)])
        self.last_conv1 = nn.Conv2d(32, 32, 3, padding=1)
        self.last_bn = nn.GroupNorm(8, 32)
        self.last_conv2 = nn.Conv2d(32, self.out_channels, 1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, xvals, x):
        for i in range(0, self.downsample)[::-1]:
            x = self.up_blocks[i](xvals[i], x)
        y = self.last_bn(self.last_conv1(x))
        y = self.last_conv2(y)
        return y, x

class ConvBlock(nn.Module):
    def __init__(self, f1, f2, kernel_size=3, padding=1, use_groupnorm=False, groups=8, dilation=1, transpose=False):
        super(ConvBlock, self).__init__()
        self.transpose = transpose
        self.conv = nn.Conv2d(f1, f2, (kernel_size, kernel_size), dilation=dilation, padding=padding*dilation)
        if self.transpose:
            self.convt = nn.ConvTranspose2d(
                f1, f1, (3, 3), dilation=dilation, stride=2, padding=dilation, output_padding=1
            )
        if use_groupnorm:
            self.bn = nn.GroupNorm(groups, f1)
        else:
            self.bn = nn.BatchNorm2d(f1)

    def forward(self, x):
        # x = F.dropout(x, 0.04, self.training)
        x = self.bn(x)
        if self.transpose:
            # x = F.upsample(x, scale_factor=2, mode='bilinear')
            x = F.relu(self.convt(x))
            # x = x[:, :, :-1, :-1]
        x = F.relu(self.conv(x))
        return x