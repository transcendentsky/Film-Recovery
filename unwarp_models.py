import torch
import torch.nn as nn
import torch.nn.functional as F
from models.misc import modules

constrain_path = {
    ('threeD', 'normal'): (True, True, ''),
    ('threeD', 'depth'): (True, True, ''),
    ('normal', 'depth'): (True, True, ''),
    ('depth', 'normal'): (True, True, ''),

}


class UnwarpNet(nn.Module):
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
        self.mask_decoder = modules.Decoder(downsample=6, out_channels=1, combine_num=0)
        bottle_neck = sum([2 ** (i + 4) for i in range(self.combine_num)])
        self.second_encoder = modules.Encoder(downsample=6, in_channels=bottle_neck * 3 + 3)
        self.uv_decoder = modules.Decoder(downsample=6, out_channels=2, combine_num=0)
        # self.albedo_decoder = modules.AlbedoDecoder(downsample=6, out_channels=1)
        self.albedo_decoder = modules.Decoder(downsample=6, out_channels=1, combine_num=0)
        self.deform_decoder = modules.Decoder(downsample=6, out_channels=2, combine_num=0)
        self.dep2nor = None
        self.threeD_to_nor2dep = None
        self.nor2dep = None
        
    def forward(self, x):
        gxvals, gx_encode = self.geo_encoder(x)
        threeD_map, threeD_feature = self.threeD_decoder(gxvals, gx_encode)
        threeD_map = nn.functional.tanh(threeD_map)
        dep_map, dep_feature = self.depth_decoder(gxvals, gx_encode)
        dep_map = nn.functional.tanh(dep_map)
        nor_map, nor_feature = self.normal_decoder(gxvals, gx_encode)
        nor_map = nn.functional.tanh(nor_map)
        mask_map, mask_feature = self.mask_decoder(gxvals, gx_encode)
        mask_map = torch.nn.functional.sigmoid(mask_map)
        # geo_feature = torch.cat([threeD_feature, nor_feature, dep_feature], dim=1)
        geo_feature = torch.cat([threeD_feature, nor_feature, dep_feature, x], dim=1)
        b, c, h, w = geo_feature.size()
        geo_feature_mask = geo_feature.mul(mask_map.expand(b, c, h, w))
        secvals, sec_encode = self.second_encoder(geo_feature_mask)
        uv_map, _ = self.uv_decoder(secvals, sec_encode)
        uv_map = nn.functional.tanh(uv_map)
        alb_map, _ = self.albedo_decoder(secvals, sec_encode)
        alb_map = nn.functional.tanh(alb_map)
        deform_map, _ = self.deform_decoder(secvals, sec_encode)
        deform_map = nn.functional.tanh(deform_map)
        return uv_map, threeD_map, nor_map, alb_map, dep_map, mask_map, \
               None, None, None, None, None, deform_map
