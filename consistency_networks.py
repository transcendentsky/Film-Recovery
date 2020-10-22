import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import MultiStepLR
from modules import UNet_up_block, UNet_down_block, UNet, UNetDepth, ConvBlock


class GeoMetryEncoder(nn.Module):
    def __init__(self, downsample=6, in_channels=3):
        super(GeoMetryEncoder, self).__init__()
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


class GeoMetryDecoder(nn.Module):
    def __init__(self, downsample=6, out_channels=3):
        super(GeoMetryDecoder,self).__init__()
        self.downsample = downsample
        self.up_blocks = nn.ModuleList(
            [UNet_up_block(2 ** (4 + i), 2 ** (5 + i), 2 ** (4 + i)) for i in range(0, downsample)])
        self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = nn.GroupNorm(8, 16)
        self.last_conv2 = nn.Conv2d(16, out_channels, 1, padding=0)
        self.up_sampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.relu = nn.ReLU()

    def forward(self, xvals, x):
        envals = []
        for i in range(0, self.downsample)[::-1]:
            x = self.up_blocks[i](xvals[i], x)
            if i < 3:
                envals.append(x)
        for i, val in enumerate(envals):
            for _ in range(len(envals) - 1 - i):
                val = self.up_sampling(val)
            envals[i] = val
        combine_z = torch.cat(envals[::-1], dim=1)
        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)
        return combine_z, x


class UVEncoder(nn.Module):
    def __init__(self, downsample=6, in_channels=3, bilinear=True):
        super(UVEncoder, self).__init__()
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


class UVDecoder(nn.Module):
    def __init__(self, downsample=6, out_channels=3):
        super(UVDecoder, self).__init__()
        self.downsample = downsample
        self.up_blocks = nn.ModuleList(
            [UNet_up_block(2 ** (4 + i), 2 ** (5 + i), 2 ** (4 + i)) for i in range(0, downsample)])
        self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = nn.GroupNorm(8, 16)
        self.last_conv2 = nn.Conv2d(16, out_channels, 1, padding=0)
        self.up_sampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.relu = nn.ReLU()

    def forward(self, xvals, x):
        for i in range(0, self.downsample)[::-1]:
            x = self.up_blocks[i](xvals[i], x)
        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)
        return x


class AlbeDecoder(nn.Module):
    def __init__(self, downsample=6, out_channels=3):
        super(AlbeDecoder, self).__init__()
        self.downsample = downsample
        self.up_blocks = nn.ModuleList(
            [UNet_up_block(2 ** (5 + i), 2 ** (6 + i), 2 ** (5 + i)) for i in range(0, downsample)])
        self.last_conv1 = nn.Conv2d(32, 32, 3, padding=1)
        self.last_bn = nn.GroupNorm(8, 32)
        self.last_conv2 = nn.Conv2d(32, out_channels, 1, padding=0)
        self.up_sampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.relu = nn.ReLU()

    def forward(self, xvals, x):
        for i in range(0, self.downsample)[::-1]:
            x = self.up_blocks[i](xvals[i], x)
        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)
        return x


class UnwarpNet(nn.Module):
    def __init__(self):
        super(UnwarpNet, self).__init__()
        self.geometry_encoder = GeoMetryEncoder(downsample=6, in_channels=3)
        self.threed_decoder = GeoMetryDecoder(downsample=6, out_channels=3)
        self.depth_decoder = GeoMetryDecoder(downsample=6, out_channels=1)
        self.normal_decoder = GeoMetryDecoder(downsample=6, out_channels=3)
        self.background_decoder = GeoMetryDecoder(downsample=6, out_channels=1)
        self.uv_encoder = UVEncoder(downsample=6, in_channels=336)
        self.uv_decoder = UVDecoder(downsample=6, out_channels=2)
        self.albedo_decoder = AlbeDecoder(downsample=6, out_channels=1)

    def forward(self, x):
        gxvals, gx_encode = self.geometry_encoder(x)
        threed_feature, threed_map = self.threed_decoder(gxvals, gx_encode)
        depth_feature, depth_map = self.depth_decoder(gxvals, gx_encode)
        normal_feature, normal_map = self.normal_decoder(gxvals, gx_encode)
        background_feature, background_map = self.background_decoder(gxvals, gx_encode)
        geo_feature = torch.cat([threed_feature, normal_feature, depth_feature], dim=1)
        b, c, h, w = geo_feature.size()
        geo_feature_mask = geo_feature.mul(background_map.expand(b, c, h, w))
        uxvals, ux_encode = self.uv_encoder(geo_feature_mask)
        uv_map = self.uv_decoder(uxvals, ux_encode)
        combinevals = []
        for gx, ux in zip(gxvals, uxvals):
            combinevals.append(torch.cat([gx, ux], dim=1))
        combinex_encode = torch.cat([gx_encode, ux_encode], dim=1)
        albedo_map = self.albedo_decoder(combinevals, combinex_encode)
        return uv_map, threed_map, normal_map, albedo_map, depth_map, background_map


if __name__ == '__main__':
    x = torch.ones((32,3,256,256))
    unwarp = UnwarpNet()
    uv_map, threed_map, normal_map, albedo_map, depth_map, background_map = unwarp(x)
    print(uv_map.shape)
    print(threed_map.shape)
    print(normal_map.shape)
    print(albedo_map.shape)
    print(depth_map.shape)
    print(background_map.shape)


        

