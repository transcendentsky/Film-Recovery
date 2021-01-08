import torch
import torch.nn as nn
import torch.nn.functional as F
from . import modules
import numpy as np
from tutils import *


class DeformNet(nn.Module):
    def __init__(self, num=0):
        super(DeformNet, self).__init__()
        self.geo_encoder = modules.Encoder(downsample=6, in_channels=3)
        self.deform_decoder = modules.Decoder(downsample=6, out_channels=2, combine_num=1)
        
    def forward(self, x):
        gxvals, gx_encode = self.geo_encoder(x)
        df_map, df_feature = self.deform_decoder(gxvals, gx_encode)
        df_map = nn.functional.tanh(df_map)
        return df_map, df_feature
        
 
    
@tfuncname
def construct_plain_cmap(bs, img_size=256, pad_size=0):
    
    r, g = np.meshgrid(np.linspace(0.1,1,img_size), np.linspace(1,0.1,img_size))
    b = np.ones((img_size,img_size)) * 0.1
    rgb = np.stack([b,g,r], axis=-1)
    pad_img = None
    if pad_size > 0:
        BLACK = [0,0,0] # WHITE = [1,1,1]
        pad_img= cv2.copyMakeBorder(rgb.copy(),pad_size,pad_size,pad_size,pad_size,
                                    cv2.BORDER_CONSTANT,value=BLACK)
    # repeat for batches
    rgb = rgb[np.newaxis, :,:,:]
    rgb = np.repeat(rgb, bs, axis=0)
    rgb = torch.from_numpy(rgb.transpose((0,3,1,2))).cuda().float()
    return rgb, pad_img
    # import ipdb; ipdb.set_trace()
    
@tfuncname
def construct_plain_bg(bs, img_size=256, pad_size=0):
    bg = np.ones((img_size,img_size,1)) 
    pad_img = None
    if pad_size > 0:
        BLACK = [0,0,0] # WHITE = [1,1,1]
        pad_img= cv2.copyMakeBorder(rgb.copy(),pad_size,pad_size,pad_size,pad_size,
                                    cv2.BORDER_CONSTANT,value=BLACK)
    # import ipdb; ipdb.set_trace()
    bg = bg[np.newaxis, :,:,:]
    bg = np.repeat(bg, bs, axis=0)
    bg = torch.from_numpy(bg.transpose((0,3,1,2))).cuda().float()
    return bg, pad_img

if __name__ == "__main__":
    import cv2
    from dataloader.print_img import print_img_auto
    rgb,_ = construct_plain_cmap(3)
    print_img_auto(rgb[0,:,:,:], "cmap", fname="./plain_cmap_print4.jpg")