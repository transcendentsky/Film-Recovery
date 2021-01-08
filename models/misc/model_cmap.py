import torch
import torch.nn as nn
import torch.nn.functional as F
from . import modules


class UnwarpNet_cmap(nn.Module):
    def __init__(self, use_simple=False, combine_num=3, train_mod="all"):
        super(UnwarpNet_cmap, self).__init__()
        self.combine_num = combine_num
        self.use_simple = use_simple
        self.train_mod = train_mod
        print("Training Mode: ", self.train_mod)
        # if use_simple:
        #     self.geo_encoder = modules.Encoder_sim(downsample=4, in_channels=3)
        #     self.threeD_decoder = modules.Decoder_sim(downsample=4, out_channels=3)
        #     self.second_encoder = modules.Encoder_sim(downsample=4, in_channels= 2 ** 6+3)
        #     self.uv_decoder = modules.Decoder_sim(downsample=4, out_channels=2)
        #     self.albedo_decoder = modules.Decoder_sim(downsample=4, out_channels=1)
        # else:
        self.geo_encoder = modules.Encoder(downsample=6, in_channels=3)
        self.threeD_decoder = modules.Decoder(downsample=6, out_channels=3, combine_num=self.combine_num)
        self.mask_decoder = modules.Decoder(downsample=6, out_channels=1, combine_num=0)
        bottle_neck = sum([2 ** (i + 4) for i in range(self.combine_num)])
        self.second_encoder = modules.Encoder(downsample=6, in_channels=bottle_neck + 3)
        self.uv_decoder = modules.Decoder(downsample=6, out_channels=2, combine_num=0)
        # self.albedo_decoder = modules.AlbedoDecoder(downsample=6, out_channels=1)
        self.albedo_decoder = modules.Decoder(downsample=6, out_channels=1, combine_num=0)

    def forward(self, x):
        gxvals, gx_encode = self.geo_encoder(x)
        threeD_map, threeD_feature = self.threeD_decoder(gxvals, gx_encode)
        threeD_map = nn.functional.tanh(threeD_map)
        mask_map, mask_feature = self.mask_decoder(gxvals, gx_encode)
        mask_map = torch.nn.functional.sigmoid(mask_map)
        mask_map_2 = torch.where(mask_map > 0.5, torch.ones_like(mask_map), torch.zeros_like(mask_map)).float()
        if self.train_mod == "cmap_only":
            return threeD_map
        else:
            geo_feature = torch.cat([threeD_feature, x], dim=1)
            b, c, h, w = geo_feature.size()
            geo_feature_mask = geo_feature.mul(mask_map.expand(b, c, h, w))
            
            secvals, sec_encode = self.second_encoder(geo_feature_mask)
            uv_map, _ = self.uv_decoder(secvals, sec_encode)
            uv_map = nn.functional.tanh(uv_map)
            alb_map, _ = self.albedo_decoder(secvals, sec_encode)
            alb_map = nn.functional.tanh(alb_map)
            return uv_map, threeD_map, alb_map, mask_map


class UnwarpNet(nn.Module):
    def __init__(self, use_simple=False, combine_num=3, use_constrain=True, constrain_configure=None, wo_bg=False):
        super(UnwarpNet, self).__init__()
        self.combine_num = combine_num
        self.use_simple = use_simple
        self.use_constrain = use_constrain
        self.constrain_configure = constrain_configure
        
        self.geo_encoder = modules.Encoder(downsample=6, in_channels=3)
        self.threeD_decoder = modules.Decoder(downsample=6, out_channels=3, combine_num=self.combine_num)
        self.normal_decoder = modules.Decoder(downsample=6, out_channels=3, combine_num=self.combine_num)
        self.depth_decoder = modules.Decoder(downsample=6, out_channels=1, combine_num=self.combine_num)
        self.mask_decoder = modules.Decoder(downsample=6, out_channels=1, combine_num=0)
        bottle_neck = sum([2 ** (i + 4) for i in range(self.combine_num)])
        self.second_encoder = modules.Encoder(downsample=6, in_channels=bottle_neck * 3+3)
        self.uv_decoder = modules.Decoder(downsample=6, out_channels=2, combine_num=0)
        #self.albedo_decoder = modules.AlbedoDecoder(downsample=6, out_channels=1)
        self.albedo_decoder = modules.Decoder(downsample=6, out_channels=1,combine_num=0)
        
        self.wo_bg = wo_bg
    
    def forward(self, x):
        gxvals, gx_encode = self.geo_encoder(x)
        threeD_map, threeD_feature = self.threeD_decoder(gxvals, gx_encode)
        threeD_map = nn.functional.tanh(threeD_map)
        dep_map, dep_feature = self.depth_decoder(gxvals, gx_encode)
        dep_map = nn.functional.tanh(dep_map)
        nor_map, nor_feature = self.normal_decoder(gxvals, gx_encode)
        nor_map = nn.functional.tanh(nor_map)
        
        if self.wo_bg:
            mask_map, mask_feature = self.mask_decoder(gxvals, gx_encode)
            mask_map = torch.nn.functional.sigmoid(mask_map)
            mask_map_2 = torch.where(mask_map > 0.5, torch.ones_like(mask_map), torch.zeros_like(mask_map)).float()
        
        geo_feature = torch.cat([threeD_feature, nor_feature, dep_feature, x], dim=1)
        
        if self.wo_bg:
            b, c, h, w = geo_feature.size()
            geo_feature = geo_feature.mul(mask_map.expand(b, c, h, w))
        secvals, sec_encode = self.second_encoder(geo_feature)
        
        uv_map, _ = self.uv_decoder(secvals, sec_encode)
        uv_map = nn.functional.tanh(uv_map)
        
        alb_map, _ = self.albedo_decoder(secvals, sec_encode)
        alb_map = nn.functional.tanh(alb_map)
        
        return uv_map, threeD_map, nor_map, alb_map, dep_map, mask_map, mask_map_2


class UnwarpNet_wo_bg(nn.Module):
    def __init__(self, use_simple=False, combine_num=3, use_constrain=True, constrain_configure=None):
        super(UnwarpNet, self).__init__()
        self.combine_num = combine_num
        self.use_simple = use_simple
        self.use_constrain = use_constrain
        self.constrain_configure = constrain_configure
        
        self.geo_encoder = modules.Encoder(downsample=6, in_channels=3)
        self.threeD_decoder = modules.Decoder(downsample=6, out_channels=3, combine_num=self.combine_num)
        self.normal_decoder = modules.Decoder(downsample=6, out_channels=3, combine_num=self.combine_num)
        self.depth_decoder = modules.Decoder(downsample=6, out_channels=1, combine_num=self.combine_num)
        # self.mask_decoder = modules.Decoder(downsample=6, out_channels=1, combine_num=0)
        bottle_neck = sum([2 ** (i + 4) for i in range(self.combine_num)])
        self.second_encoder = modules.Encoder(downsample=6, in_channels=bottle_neck * 3+3)
        self.uv_decoder = modules.Decoder(downsample=6, out_channels=2, combine_num=0)
        #self.albedo_decoder = modules.AlbedoDecoder(downsample=6, out_channels=1)
        self.albedo_decoder = modules.Decoder(downsample=6, out_channels=1,combine_num=0)
    
    def forward(self, x):
        gxvals, gx_encode = self.geo_encoder(x)
        threeD_map, threeD_feature = self.threeD_decoder(gxvals, gx_encode)
        threeD_map = nn.functional.tanh(threeD_map)
        dep_map, dep_feature = self.depth_decoder(gxvals, gx_encode)
        dep_map = nn.functional.tanh(dep_map)
        nor_map, nor_feature = self.normal_decoder(gxvals, gx_encode)
        nor_map = nn.functional.tanh(nor_map)
        
        geo_feature = torch.cat([threeD_feature, nor_feature, dep_feature, x], dim=1)
        secvals, sec_encode = self.second_encoder(geo_feature)
        uv_map, _ = self.uv_decoder(secvals, sec_encode)
        uv_map = nn.functional.tanh(uv_map)
        
        alb_map, _ = self.albedo_decoder(secvals, sec_encode)
        alb_map = nn.functional.tanh(alb_map)
        
        return uv_map, threeD_map, nor_map, alb_map, dep_map, None, None

if __name__ == '__main__':
    data_dir ='/home1/qiyuanwang/film_generate/npy'


    ###################
    x = torch.ones((32, 3, 256, 256))
    model = UnwarpNet(use_simple=False, combine_num=3, use_constrain=True, constrain_configure=None)
    uv_map, threeD_map, nor_map, alb_map, dep_map, mask_map, nor_from_threeD, dep_from_threeD, nor_from_dep, dep_from_nor = model(
        x)
    print(uv_map.shape)
    print(threeD_map.shape)
    print(nor_map.shape)
    print(alb_map.shape)
    print(dep_map.shape)
    print(mask_map.shape)
    print(nor_from_threeD.shape)
    print(dep_from_threeD.shape)
    print(nor_from_dep.shape)
    print(dep_from_nor.shape)
