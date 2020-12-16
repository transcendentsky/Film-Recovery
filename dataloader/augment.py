# coding: utf-8
import numpy as np
import cv2
import os
import time
import pickle
import torch
from .print_img import print_img_auto
import random
import numpy as np

def augment(_tuple, method, imsize=256):
    
    ori,ab,dep,nor,cmap,uv,bg = _tuple
    ori = cv2.resize(ori, (448, 448))
    
    if "crop" in method:
        crop_rate = 0.9
        w,h,c = ori.shape
        # if True:
        #     print(ori.shape)
        #     print(ab.shape)
        #     print(dep.shape)
        #     print(nor.shape)
        #     print(cmap.shape)
        #     print(uv.shape)
        #     print(bg.shape)
        
        randw = crop_rate+(0.99-crop_rate)*random.random()
        randh = crop_rate+(0.99-crop_rate)*random.random()
        # print("crop: ", randw, randh)
        x  = int( (1-randw)/2.*w )
        y  = int( (1-randw)/2.*h )
        xx = int( w - (1-randw)/2.*w )
        yy = int( h - (1-randw)/2.*h )
        
        ori = ori[y:yy, x:xx, :]
        ab  = ab[y:yy, x:xx, :]
        dep = dep[y:yy, x:xx, :]
        nor = nor[y:yy, x:xx, :]
        cmap= cmap[y:yy, x:xx, :]
        uv  = uv[y:yy, x:xx, :]
        bg  = bg[y:yy, x:xx]
        
        ori = cv2.resize(ori, (imsize, imsize))        
        ab  = cv2.resize(ab , (imsize, imsize))[:,:,np.newaxis]
        dep = cv2.resize(dep, (imsize, imsize))
        nor = cv2.resize(nor, (imsize, imsize))
        cmap= cv2.resize(cmap, (imsize, imsize))
        uv  = cv2.resize(uv, (imsize, imsize))
        bg  = cv2.resize(bg, (imsize, imsize))[:,:,np.newaxis]
        # print("resize over?")
        #print("aug ?22 ab", ab.shape, bg.shape)
        
    return ori,ab,dep,nor,cmap,uv,bg
    
    
    
def test_augment():
    
    from .load_data_2 import filmDataset_3
    from .data_process import reprocess_auto
    
    from torch.utils.data import Dataset, DataLoader
    data_path = "/home1/quanquan/datasets/generate/mesh_film_small/"
    dataset = filmDataset_3(data_path, load_mod="nobw")

    #loader = DataLoader(dataset, batch_size=1, shuffle=False)
    #for batch_idx, data in enumerate(loader):
    
    ori,ab,dep,nor,cmap,uv,bg = dataset.__getitem__(0)
    print("dasdasdasdsa")
    #print_img_auto(ori, "ori", fname="test_img/ori1.jpg")
    #print_img_auto(ab , "ab" , fname="test_img/ab1.jpg")
    #print_img_auto(dep, "depth", fname="test_img/dep1.jpg")
    #print_img_auto(nor, "normal", fname="test_img/nor1.jpg")
    #print_img_auto(cmap, "cmap", fname="test_img/cmap1.jpg")
    #print_img_auto(uv , "uv" , fname="test_img/uv1.jpg")
    #print_img_auto(bg , "bg" , fname="test_img/bg1.jpg")
    #ori,ab,dep,nor,cmap,uv,bg = augment((ori,ab,dep,nor,cmap,uv,bg), method="crop", imsize=256)
    
    ori = reprocess_auto(ori, "ori")
    ab  = reprocess_auto(ab,  "ab")
    dep = reprocess_auto(dep, "depth")
    nor = reprocess_auto(nor, "normal")
    cmap= reprocess_auto(cmap, "cmap")
    uv  = reprocess_auto(uv , "uv")
    bg  = reprocess_auto(bg, "bg")
    #ori = reprocess_auto(ori, "ori")
    #ori = reprocess_auto(ori, "ori")
    
    
    print_img_auto(ori, "ori", fname="test_img/ori.jpg")
    print_img_auto(ab , "ab" , fname="test_img/ab.jpg")
    print_img_auto(dep, "depth", fname="test_img/dep.jpg")
    print_img_auto(nor, "normal", fname="test_img/nor.jpg")
    print_img_auto(cmap, "cmap", fname="test_img/cmap.jpg")
    print_img_auto(uv , "uv" , fname="test_img/uv.jpg")
    print_img_auto(bg , "bg" , fname="test_img/bg.jpg")
    

if __name__ == "__main__":
    test_augment()
    
    