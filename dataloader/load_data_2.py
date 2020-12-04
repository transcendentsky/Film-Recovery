# -*- coding: utf-8 -*-
import torchvision.transforms as transforms
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
import cv2
from scipy.interpolate import griddata
from tutils import *
from torch import nn
from dataloader import data_process, print_img, uv2bw, bw2deform
from .augment import augment
from .data_process import process_auto
from .print_img import print_img_auto,print_img_with_reprocess

ori_image_dir = 'data_2000/Mesh_Film/npy/'

EPOCH = 1
test_BATCH_SIZE = 100

class filmDataset_2(Dataset):
    # dir = "/home1/quanquan/datasets/generate/mesh_film_small/"

    def __init__(self, npy_dir, load_mod="all"):
        self.npy_dir = npy_dir
        self.npy_list = np.array([x.path for x in os.scandir(npy_dir) if x.name.endswith(".npy")])
        print("Dataset Length: ", len(self.npy_list))
        
    def __getitem__(self, index):
        npy_path = self.npy_list[index]


class filmDataset_3(Dataset):
    def __init__(self, image_dir, load_mod="all"):
        self.image_dir = image_dir
        self.image_name = np.array([x.name for x in os.scandir(self.image_dir +'albedo/') if self.check_paths(x.name)])    #x.name.endswith(".png") and    #x.path 则为路径
        # self.image_name = np.array([x for x in self.image_name_pre if check_paths(x)])
        self.image_name.sort()
        self.input_size =(256, 256)
        self.load_mod = load_mod
        # print(self.record_files)

    def check_paths(self, x):
        image_name = x
        ori_name = self.image_dir + 'img/' + image_name
        ab_name  = self.image_dir + 'img/' + image_name
        dep_name = self.image_dir + 'depth/' + image_name[:-3] + 'exr'
        nor_name = self.image_dir + 'depth/' + image_name[:-3] + 'exr'
        cmap_name= self.image_dir + 'depth/' + image_name[:-3] + 'exr'
        uv_name  = self.image_dir + 'depth/' + image_name[:-3] + 'exr'

        for name in [ori_name, ab_name, dep_name, nor_name, cmap_name, uv_name]:
            if not os.path.exists(name):
                return False
        return True


    def check_paths2(self, paths):
        for name in paths:
            if not os.path.exists(name):
                return False
        return True

    def __getitem__(self, index):
        image_name = self.image_name[index]
        if self.load_mod in ["all", "original", "nobw"]:
            """loading"""
            ori = cv2.imread(self.image_dir + 'img/' + image_name)
            ab = cv2.imread(self.image_dir + 'albedo/' + image_name)
            ab = cv2.cvtColor(ab, cv2.COLOR_BGR2GRAY)
            ab = ab[:,:,np.newaxis]
            #depth = cv2.imread(self.image_dir + 'depth/' + image_name)
            depth = cv2.imread(self.image_dir + 'depth/' + image_name[:-3] + 'exr', cv2.IMREAD_UNCHANGED)
            normal = cv2.imread(self.image_dir + 'shader_normal/' + image_name[:-3] + 'exr', cv2.IMREAD_UNCHANGED)    #[-1,1]
            cmap = cv2.imread(self.image_dir + '3dmap/' + image_name[:-3] + 'exr', cv2.IMREAD_UNCHANGED)      #[0,1]
            uv = cv2.imread(self.image_dir + 'uv/' + image_name[:-3] + 'exr', cv2.IMREAD_UNCHANGED)        #[0,1]
            uv = uv[:,:,1:]
            bg = cv2.threshold(depth[:,:,0], 0, 1, cv2.THRESH_BINARY)[1]   
            #print("shape uv", uv.shape)
            
        if self.load_mod == "original":
            print("output original data")
            return ori,ab,depth,normal,cmap,uv,bg
            
        if self.load_mod == "nobw":
            """processing"""
            ori,ab,dep,nor,cmap,uv,bg= augment((ori,ab,depth,normal,cmap,uv,bg), method=["crop"], imsize=256)
            #print("ab121", ab.shape)
            ori = process_auto(ori, "ori")
            ab  = process_auto(ab , "ab" )
            dep = process_auto(dep,  "depth")
            nor = process_auto(nor, "normal")
            cmap= process_auto(cmap, "cmap")
            uv  = process_auto(uv,  "uv")
            bg  = process_auto(bg,  "bg")
    
            # ori,ab,dep,nor,cmap,uv,bg
            # Bug exists because of no flaot()
            #print("ab", ab.shape)
            #print("ori", ori.shape)
            #print("dep", dep.shape, bg.shape)
            return torch.from_numpy(ori.transpose((2,0,1))).float(), \
                   torch.from_numpy(ab.transpose((2,0,1))).float(), \
                   torch.from_numpy(dep.transpose((2,0,1))).float(), \
                   torch.from_numpy(nor.transpose((2,0,1))).float(), \
                   torch.from_numpy(cmap.transpose((2,0,1))).float(), \
                   torch.from_numpy(uv.transpose((2,0,1))).float(), \
                   torch.from_numpy(bg.transpose((2,0,1))).float(), \

        if self.load_mod == "all":
            """processing"""
            ori,ab,dep,nor,cmap,uv,bg= augment((ori,ab,depth,normal,cmap,uv,bg), method=["crop"], imsize=256)
            bw  = uv2bw.uv2backward_trans_3(uv, bg)
            
            ori = process_auto(ori, "ori")
            ab  = process_auto(ab , "ab" )
            dep = process_auto(dep,  "depth")
            nor = process_auto(nor, "normal")
            cmap= process_auto(cmap, "cmap")
            uv  = process_auto(uv,  "uv")
            bg  = process_auto(bg,  "bg")
            bw  = process_auto(bw,  "bw")
    
            # ori,ab,dep,nor,cmap,uv,bg,bw
            return torch.from_numpy(ori.transpose((2,0,1))).float(), \
                   torch.from_numpy(ab.transpose((2,0,1))).float(), \
                   torch.from_numpy(dep.transpose((2,0,1))).float(), \
                   torch.from_numpy(nor.transpose((2,0,1))).float(), \
                   torch.from_numpy(cmap.transpose((2,0,1))).float(), \
                   torch.from_numpy(uv.transpose((2,0,1))).float(), \
                   torch.from_numpy(bg.transpose((2,0,1))).float(), \
                   torch.from_numpy(bw.transpose((2,0,1))).float(), \
                           
    def __len__(self):
        return len(self.image_name)

class filmDataset_old(Dataset):
    """
    Using with Dataset generated by Qiyuan / Liuli
    """
    def __init__(self, npy_dir, load_mod="all", npy_dir_2=None):
        self.npy_dir = npy_dir
        self.npy_list = np.array([x.path for x in os.scandir(npy_dir) if x.name.endswith(".npy")])
        print("Dataset Length: ", len(self.npy_list))
        if npy_dir_2!=None:
            self.npy_list_2 = np.array([x.path for x in os.scandir(npy_dir_2) if x.name.endswith(".npy")])
            self.npy_list = np.append(self.npy_list, self.npy_list_2)
        self.npy_list.sort()
        self.load_mod = load_mod

    def __getitem__(self, index):
        npy_path = self.npy_list[index]
        """loading"""
        # data = np.load(self.npy_dir + '/' + npy_name, allow_pickle=True)[()]
        data = np.load(npy_path, allow_pickle=True)[()]
        ori = data['ori']
        ab = data['ab']
        # bmap = data['bmap']
        depth = data['depth']
        normal = data['normal']
        uv = data['uv']
        cmap = data['cmap']
        background = data['background']
        
        if self.load_mod == "all":
            return torch.from_numpy(ori), \
               torch.from_numpy(ab), \
               torch.from_numpy(depth), \
               torch.from_numpy(normal), \
               torch.from_numpy(cmap), \
               torch.from_numpy(uv), \
               torch.from_numpy(background)
        # ori_1080 = data['ori_1080']
        elif self.load_mod == "uvbw":
            ### ----------- Generate BW map ---------------
            uv_2 = data_process.reprocess_auto(uv.transpose((1,2,0)), "uv")
            mask = data_process.reprocess_auto(background.transpose((1,2,0)), "background")
            bw = uv2bw.uv2backward_trans_3(uv_2, mask)
            bw = data_process.process_auto(bw, "bw")
            bw = bw.transpose((2,0,1))
            ### --------------------------------------------
            return  torch.from_numpy(cmap), \
                torch.from_numpy(uv), \
                torch.from_numpy(bw), \
               torch.from_numpy(background)

        elif self.load_mod == "test_uvbw_mapping":
            ### ----------- Generate BW map ---------------
            uv_2 = data_process.reprocess_auto(uv.transpose((1,2,0)), "uv")
            mask = data_process.reprocess_auto(background.transpose((1,2,0)), "background")
            bw = uv2bw.uv2backward_trans_3(uv_2, mask)
            bw = data_process.process_auto(bw, "bw")
            bw = bw.transpose((2,0,1))
            ### --------------------------------------------
            return  torch.from_numpy(cmap), \
                torch.from_numpy(uv), \
                torch.from_numpy(bw), \
               torch.from_numpy(background), \
               torch.from_numpy(ori)
        elif self.load_mod == "deform_cmap":
            ###  ----------- Generate BW map ---------------
            uv_2 = data_process.reprocess_auto(uv.transpose((1,2,0)), "uv")
            mask = data_process.reprocess_auto(background.transpose((1,2,0)), "background")
            bw = uv2bw.uv2backward_trans_3(uv_2, mask)
            bw2 = data_process.process_auto(bw, "bw")
            bw2 = bw2.transpose((2,0,1))
            deform = bw2deform.bw2deform(bw)
            deform = data_process.process_auto(deform, "deform")
            deform = deform.transpose((2,0,1))

            ### --------------------------------------------
            return  torch.from_numpy(cmap), \
                torch.from_numpy(uv), \
                torch.from_numpy(deform), \
                torch.from_numpy(bw2), \
                torch.from_numpy(ori), \
               torch.from_numpy(background)
        
        elif self.load_mod == "all_deform":
            ###  ----------- Generate BW map ---------------
            uv_2 = data_process.reprocess_auto(uv.transpose((1,2,0)), "uv")
            mask = data_process.reprocess_auto(background.transpose((1,2,0)), "background")
            bw = uv2bw.uv2backward_trans_3(uv_2, mask)
            bw2 = data_process.process_auto(bw, "bw")
            bw2 = bw2.transpose((2,0,1))
            deform = bw2deform.bw2deform(bw)
            deform = data_process.process_auto(deform, "deform")
            deform = deform.transpose((2,0,1))
            # -----------------------------------------
            return torch.from_numpy(ori), \
               torch.from_numpy(ab), \
               torch.from_numpy(depth), \
               torch.from_numpy(normal), \
               torch.from_numpy(cmap), \
               torch.from_numpy(uv), \
               torch.from_numpy(deform), \
               torch.from_numpy(background)

    def __len__(self):
        return len(self.npy_list)
        
    
def test_dataset():
    
    from .load_data_2 import filmDataset_3
    from .data_process import reprocess_auto
    
    from torch.utils.data import Dataset, DataLoader
    data_path = "/home1/quanquan/datasets/generate/mesh_film_small/"
    dataset = filmDataset_3(data_path, load_mod="nobw")

    ori,ab,dep,nor,cmap,uv,bg = dataset.__getitem__(0)
    
    print_img_with_reprocess(ori, "ori")
    print_img_with_reprocess(ab,  "ab")
    print_img_with_reprocess(dep, "depth")
    print_img_with_reprocess(nor, "normal")
    print_img_with_reprocess(cmap, "cmap")
    print_img_with_reprocess(uv , "uv")
    print_img_with_reprocess(bg,  "bg")
    print("dasdasdasdsa")

if __name__ == "__main__":
    test_dataset()