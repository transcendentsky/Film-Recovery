import torch
import torch.nn as nn
from torch.nn import init
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from tutils import *

class basic_conv_double(nn.Module):
    """keep w and h"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(basic_conv_double, self).__init__()

        if not mid_channels:
            mid_channels = out_channels
        self.conv_double = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),   #Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_double(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            basic_conv_double(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)      #这个时候channel不变
            self.conv = basic_conv_double(in_channels, out_channels, mid_channels=in_channels //2)      #但是这边的out_channel为四分之一，与down传过来的channel合并，正好是一半的channel
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)         # w,h//2
            self.conv = basic_conv_double(in_channels, out_channels)


    def forward(self, x1, x2):
        """x1为从上一层up_conv得到的，由于卷积所以有尺寸的减小；x2为encoder得到的=[batch, channel, height, width]"""
        x1 = self.up(x1)
        diffY = x2.size()[2] -x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)        # along channel
        return self.conv(x)


class outConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(outConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, bilinear=True):
        super(Encoder, self).__init__()

        self.inc = basic_conv_double(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x1, x2, x3, x4, x5

class Decoder(nn.Module):
    def __init__(self, out_channels, bilinear=True):
        super(Decoder, self).__init__()

        factor = 2 if bilinear else 1
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = outConv(64, out_channels)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)     # 64 channel
        logits = self.outc(x)
        return x, logits





class Net(nn.Module):
    def __init__(self, bilinear=True):
        super(Net, self).__init__()
        self.bilinear = bilinear

        self.encoder1 = Encoder(in_channels=3, bilinear=self.bilinear)
        self.decoder_3d = Decoder(out_channels=3, bilinear=self.bilinear)
        self.decoder_depth = Decoder(out_channels=1, bilinear=self.bilinear)
        self.decoder_normal = Decoder(out_channels=3, bilinear=self.bilinear)
        self.decoder_back = Decoder(out_channels=1, bilinear=self.bilinear)

        self.encoder2 = Encoder(in_channels=192, bilinear=self.bilinear)
        self.decoder_uv = Decoder(out_channels=2, bilinear=self.bilinear)
        # self.decoder_bw = Decoder(out_channels=2, bilinear=self.bilinear)
        self.decoder_albedo = Decoder(out_channels=1, bilinear=self.bilinear)

    def forward(self, x):
        x1_1, x2_1, x3_1, x4_1, x5_1 = self.encoder1(x)
        coor_feature, coor_map = self.decoder_3d(x1_1, x2_1, x3_1, x4_1, x5_1)
        depth_feature, depth_map = self.decoder_depth(x1_1, x2_1, x3_1, x4_1, x5_1)
        normal_feature, normal_map = self.decoder_normal(x1_1, x2_1, x3_1, x4_1, x5_1)
        back_feature, back_map = self.decoder_back(x1_1, x2_1, x3_1, x4_1, x5_1)

        geo_feature = torch.cat([coor_feature, normal_feature, depth_feature], 1)
        b, c, h, w = geo_feature.size()             # 乘background
        geo_feature_mask = geo_feature.mul(back_map.expand(b, c, h, w))

        x1_2, x2_2, x3_2, x4_2, x5_2 = self.encoder2(geo_feature_mask)
        uv_feature, uv_map = self.decoder_uv(x1_2, x2_2, x3_2, x4_2, x5_2)
        # bw_feature, bw_map = self.decoder_bw(x1_2, x2_2, x3_2, x4_2, x5_2)
        albedo_feature, albedo_map = self.decoder_albedo(x1_2, x2_2, x3_2, x4_2, x5_2)

        return uv_map, coor_map, normal_map, albedo_map, depth_map, back_map


class Decoder_2(nn.Module):
    def __init__(self, out_channels, bilinear=True):
        super(Decoder_2, self).__init__()

        factor = 2 if bilinear else 1
        self.up1 = Up(2048, 1024 // factor, bilinear)
        self.up2 = Up(1024, 512 // factor, bilinear)         # 256+256 input ---512+
        self.up3 = Up(512, 256 // factor, bilinear)
        self.up4 = Up(256, 64, bilinear)
        self.outc = outConv(64, out_channels)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)     # 64 channel
        logits = self.outc(x)
        return x, logits


class Net_a(nn.Module):
    def __init__(self, bilinear=True):
        super(Net_a, self).__init__()
        self.bilinear = bilinear

        self.encoder1 = Encoder(in_channels=3, bilinear=self.bilinear)
        self.decoder_3d = Decoder(out_channels=3, bilinear=self.bilinear)
        self.decoder_depth = Decoder(out_channels=1, bilinear=self.bilinear)
        self.decoder_normal = Decoder(out_channels=3, bilinear=self.bilinear)
        self.decoder_back = Decoder(out_channels=1, bilinear=self.bilinear)

        self.encoder2 = Encoder(in_channels=192, bilinear=self.bilinear)
        self.decoder_uv = Decoder(out_channels=2, bilinear=self.bilinear)
        # self.decoder_bw = Decoder(out_channels=2, bilinear=self.bilinear)
        # self.decoder_albedo = Decoder(out_channels=1, bilinear=self.bilinear)
        self.decoder_albedo_2 = Decoder_2(out_channels=1, bilinear=self.bilinear)

    def forward(self, x):
        x1_1, x2_1, x3_1, x4_1, x5_1 = self.encoder1(x)
        coor_feature, coor_map = self.decoder_3d(x1_1, x2_1, x3_1, x4_1, x5_1)
        depth_feature, depth_map = self.decoder_depth(x1_1, x2_1, x3_1, x4_1, x5_1)
        normal_feature, normal_map = self.decoder_normal(x1_1, x2_1, x3_1, x4_1, x5_1)
        back_feature, back_map = self.decoder_back(x1_1, x2_1, x3_1, x4_1, x5_1)

        geo_feature = torch.cat([coor_feature, normal_feature, depth_feature], 1)
        b, c, h, w = geo_feature.size()
        geo_feature_mask = geo_feature.mul(back_map.expand(b, c, h, w))

        x1_2, x2_2, x3_2, x4_2, x5_2 = self.encoder2(geo_feature_mask)          #  [batch, c, h, w]
        uv_feature, uv_map = self.decoder_uv(x1_2, x2_2, x3_2, x4_2, x5_2)
        # bw_feature, bw_map = self.decoder_bw(x1_2, x2_2, x3_2, x4_2, x5_2)
        x1 = torch.cat([x1_1, x1_2], 1)         # [5, 128, 256, 256]            # 加入网络图中虚线的部分
        x2 = torch.cat([x2_1, x2_2], 1)         # [5, 256, 127, 127]
        x3 = torch.cat([x3_1, x3_2], 1)         # [5, 512, 63, 63]
        x4 = torch.cat([x4_1, x4_2], 1)         # [5, 1024, 31, 31]
        x5 = torch.cat([x5_1, x5_2], 1)         # [5, 1024, 15, 15]
        # print(x1.shape, x1_1.shape, x1_2.shape)
        # print(x2.shape, x2_1.shape, x2_2.shape)
        # print(x3.shape, x3_1.shape, x3_2.shape)
        # print(x4.shape, x4_1.shape, x4_2.shape)
        # print(x5.shape, x5_1.shape, x5_2.shape)

        # albedo_feature, albedo_map = self.decoder_albedo(x1_2, x2_2, x3_2, x4_2, x5_2)
        albedo_feature, albedo_map = self.decoder_albedo_2(x1, x2, x3, x4, x5)
        # print(albedo_feature_2.shape, albedo_map_2.shape)
        # print(albedo_feature.shape, albedo_map.shape)

        return uv_map, coor_map, normal_map, albedo_map, depth_map, back_map


class Net_slim(nn.Module):
    def __init__(self, bilinear=True):
        super(Net_slim, self).__init__()
        self.bilinear = bilinear

        self.encoder1 = Encoder(in_channels=3, bilinear=self.bilinear)
        # self.decoder_uv = Decoder(out_channels=2, bilinear=self.bilinear)
        self.decoder_3d = Decoder(out_channels=3, bilinear=self.bilinear)
        # self.decoder_depth = Decoder(out_channels=1, bilinear=self.bilinear)
        # self.decoder_normal = Decoder(out_channels=3, bilinear=self.bilinear)

        self.encoder2 = Encoder(in_channels=64, bilinear=self.bilinear)
        self.decoder_bw = Decoder(out_channels=2, bilinear=self.bilinear)
        # self.decoder_albedo = Decoder(out_channels=3, bilinear=self.bilinear)

    def forward(self, x):
        x1_1, x2_1, x3_1, x4_1, x5_1 = self.encoder1(x)
        # uv_feature, uv_map = self.decoder_uv(x1_1, x2_1, x3_1, x4_1, x5_1)
        coor_feature, coor_map = self.decoder_3d(x1_1, x2_1, x3_1, x4_1, x5_1)
        # depth_feature, depth_map = self.decoder_depth(x1_1, x2_1, x3_1, x4_1, x5_1)
        # normal_feature, normal_map = self.decoder_normal(x1_1, x2_1, x3_1, x4_1, x5_1)

        # geo_feature = torch.cat([uv_feature, coor_feature, normal_feature], 1)
        x1_2, x2_2, x3_2, x4_2, x5_2 = self.encoder2(coor_feature)
        bw_feature, bw_map = self.decoder_bw(x1_2, x2_2, x3_2, x4_2, x5_2)
        # albedo_feature, albedo_map = self.decoder_albedo(x1_2, x2_2, x3_2, x4_2, x5_2)

        return coor_map, bw_map


