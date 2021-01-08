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
from .extra_bg import get_composed_imgs

ori_image_dir = 'data_2000/Mesh_Film/npy/'

EPOCH = 1
test_BATCH_SIZE = 100

class RealDataset(Dataset):
    def __init__(self, image_dir, load_mod="all", reg_start=None, reg_end=".jpg"):
        self.image_dir = image_dir
        self.reg_start = reg_start
        self.reg_end = reg_end
        self.image_names = np.array([x.name for x in os.scandir(self.image_dir) if self.check_path(x.name)])
        self.image_names.sort()
        
    def check_path(self, x):
        if self.reg_start is None or x.startswith(self.reg_start):
            if self.reg_end is None or x.endswith(self.reg_end):
                return True
        return False
        
    def __getitem__(self, index):
        image_name = self.image_names[index]
        ori = cv2.imread(os.path.join(self.image_dir , image_name))
        ori2 = cv2.resize(ori, (256,256))
        ori2 = process_auto(ori2, "ori")
        
        return torch.from_numpy(ori2.transpose((2,0,1))).float() ,\
            torch.from_numpy(ori.transpose((2,0,1))).float()
    
    def __len__(self):
        return len(self.image_names)

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
        self.image_name = np.array([x.name for x in os.scandir(self.image_dir +'img/') if self.check_paths(x.name)])    #x.name.endswith(".png") and    #x.path 则为路径
        # self.image_name = np.array([x for x in self.image_name_pre if check_paths(x)])
        self.image_name.sort()
        self.input_size =(256, 256)
        self.load_mod = load_mod
        # self.load_mod = load_mod if type(load_mod) == list else list(load_mod)
        # print(self.record_files)

    def check_paths(self, x):
        image_name = x
        ori_name = self.image_dir + 'img/' + image_name
        ab_name  = self.image_dir + 'albedo/' + image_name
        dep_name = self.image_dir + 'depth/' + image_name[:-3] + 'exr'
        nor_name = self.image_dir + '3dmap/' + image_name[:-3] + 'exr'
        cmap_name= self.image_dir + 'shader_normal/' + image_name[:-3] + 'exr'
        uv_name  = self.image_dir + 'uv/' + image_name[:-3] + 'exr'

        for name in [ori_name, ab_name, dep_name, nor_name, cmap_name, uv_name]:
            if not os.path.exists(name):
                return False
        return True
            
    #def prenormal(self):
        #for name in self.image_name:
            
    def check_paths2(self, paths):
        for name in paths:
            if not os.path.exists(name):
                return False
        return True

    def __getitem__(self, index):
        image_name = self.image_name[index]
        image_dir = self.image_dir
        if True:
            """loading"""
            ori = cv2.imread(image_dir + 'img/' + image_name)
            ab = cv2.imread(image_dir + 'albedo/' + image_name)
            ab = cv2.cvtColor(ab, cv2.COLOR_BGR2GRAY)
            ab = ab[:,:,np.newaxis]
            #depth = cv2.imread(self.image_dir + 'depth/' + image_name)
            dep = cv2.imread(image_dir + 'depth/' + image_name[:-3] + 'exr', cv2.IMREAD_UNCHANGED)
            nor = cv2.imread(image_dir + 'shader_normal/' + image_name[:-3] + 'exr', cv2.IMREAD_UNCHANGED)    #[-1,1]
            cmap = cv2.imread(image_dir + '3dmap/' + image_name[:-3] + 'exr', cv2.IMREAD_UNCHANGED)      #[0,1]
            uv = cv2.imread(image_dir + 'uv/' + image_name[:-3] + 'exr', cv2.IMREAD_UNCHANGED)        #[0,1]
            uv = uv[:,:,1:]
            bg = cv2.threshold(dep[:,:,0], 0, 1, cv2.THRESH_BINARY)[1]   
            
            #print("shape uv", uv.shape)
            
        if self.load_mod == "original":
            print("output original data")
            return ori,ab,dep,nor,cmap,uv,bg

        if self.load_mod == "nobw":
            """processing"""
            ori,ab,dep,nor,cmap,uv,bg= augment((ori,ab,dep,nor,cmap,uv,bg), method=["crop"], imsize=256)

            ori = process_auto(ori, "ori")
            ab  = process_auto(ab , "ab" )
            dep = process_auto(dep, "exr") #
            nor = process_auto(nor, "exr") #
            cmap= process_auto(cmap, "exr") #
            uv  = process_auto(uv,  "uv")
            bg  = process_auto(bg,  "bg")
    
            return torch.from_numpy(ori.transpose((2,0,1))).float(), \
                   torch.from_numpy(ab.transpose((2,0,1))).float(), \
                   torch.from_numpy(dep.transpose((2,0,1))).float(), \
                   torch.from_numpy(nor.transpose((2,0,1))).float(), \
                   torch.from_numpy(cmap.transpose((2,0,1))).float(), \
                   torch.from_numpy(uv.transpose((2,0,1))).float(), \
                   torch.from_numpy(bg.transpose((2,0,1))).float(), \

        if self.load_mod == "all":
            """processing"""
            ori,ab,dep,nor,cmap,uv,bg= augment((ori,ab,dep,nor,cmap,uv,bg), method=["crop"], imsize=256)
            bw  = uv2bw.uv2backward_trans_3(uv, bg)
            
            ori = process_auto(ori, "ori")
            ab  = process_auto(ab , "ab" )
            dep = process_auto(dep,  "exr")
            nor = process_auto(nor, "exr")
            cmap= process_auto(cmap, "exr")
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
        
        if self.load_mod == "new_ab":
            ori,ab,dep,nor,cmap,uv,bg= augment((ori,ab,dep,nor,cmap,uv,bg), method=[], imsize=256)
            #print("ab121", ab.shape)
            ori = process_auto(ori, "ori")
            ab  = process_auto(ab , "ab" )
            dep = process_auto(dep,  "exr")
            nor = process_auto(nor, "exr")
            cmap= process_auto(cmap, "exr")
            uv  = process_auto(uv,  "uv")
            bg  = process_auto(bg,  "bg")
            ori2 = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)[:,:,np.newaxis]
            ab_diff = np.where(bg > 0, (ori2 - ab), 0)
            ab_diff = process_auto(ab_diff , "ab" )
            
            return torch.from_numpy(ori.transpose((2,0,1))).float(), \
                   torch.from_numpy(ab_diff.transpose((2,0,1))).float(), \
                   torch.from_numpy(dep.transpose((2,0,1))).float(), \
                   torch.from_numpy(nor.transpose((2,0,1))).float(), \
                   torch.from_numpy(cmap.transpose((2,0,1))).float(), \
                   torch.from_numpy(uv.transpose((2,0,1))).float(), \
                   torch.from_numpy(bg.transpose((2,0,1))).float(), \
            
        if self.load_mod == "extra_bg":
            if random.random() > 0.66:  # 
                ori,ab,dep,nor,cmap,uv,bg = get_composed_imgs((ori,ab,dep,nor,cmap,uv,bg))    
                ori,ab,dep,nor,cmap,uv,bg = augment((ori,ab,dep,nor,cmap,uv,bg), method=["crop"], imsize=256, crop_rate=0.7)
            else:
                ori,ab,dep,nor,cmap,uv,bg = augment((ori,ab,dep,nor,cmap,uv,bg), method=["crop"], imsize=256, crop_rate=0.9)
            # import ipdb; ipdb.set_trace()
            # return ori,ab,dep,nor,cmap,uv,bg
            ori = process_auto(ori, "ori")
            ab  = process_auto(ab , "ab" )
            dep = process_auto(dep, "exr") #
            nor = process_auto(nor, "exr") #
            cmap= process_auto(cmap, "exr") #
            uv  = process_auto(uv,  "uv")
            bg  = process_auto(bg,  "bg")
            # ab-diff 
            ori2 = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)[:,:,np.newaxis]
            ab_diff = np.where(bg > 0, (ori2 - ab), 0)
            ab_diff = process_auto(ab_diff , "ab" )
            
            return torch.from_numpy(ori.transpose((2,0,1))).float(), \
                   torch.from_numpy(ab_diff.transpose((2,0,1))).float(), \
                   torch.from_numpy(dep.transpose((2,0,1))).float(), \
                   torch.from_numpy(nor.transpose((2,0,1))).float(), \
                   torch.from_numpy(cmap.transpose((2,0,1))).float(), \
                   torch.from_numpy(uv.transpose((2,0,1))).float(), \
                   torch.from_numpy(bg.transpose((2,0,1))).float(),          
        
        raise ValueError
        
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
        
def data_cleaning(data_dir="/home1/quanquan/datasets/generate/mesh_film_small/"):
    
    def check_paths(image_dir, x):
        image_name = x
        ori_name = image_dir + 'img/' + image_name
        ab_name  = image_dir + 'albedo/' + image_name
        dep_name = image_dir + 'depth/' + image_name[:-3] + 'exr'
        nor_name = image_dir + '3dmap/' + image_name[:-3] + 'exr'
        cmap_name= image_dir + 'shader_normal/' + image_name[:-3] + 'exr'
        uv_name  = image_dir + 'uv/' + image_name[:-3] + 'exr'

        for name in [ori_name, ab_name, dep_name, nor_name, cmap_name, uv_name]:
            if not os.path.exists(name):
                return False
        return True
        
    def del_files(image_dir, x):
        # remove files
        image_name = x
        ori_name = image_dir + 'img/' + image_name
        ab_name  = image_dir + 'albedo/' + image_name
        dep_name = image_dir + 'depth/' + image_name[:-3] + 'exr'
        nor_name = image_dir + '3dmap/' + image_name[:-3] + 'exr'
        cmap_name= image_dir + 'shader_normal/' + image_name[:-3] + 'exr'
        uv_name  = image_dir + 'uv/' + image_name[:-3] + 'exr'
        
        if os.path.exists(ori_name):
            os.remove(ori_name)
        if os.path.exists(ori_name):
            os.remove(ab_name)
        if os.path.exists(ori_name):
            os.remove(dep_name)
        if os.path.exists(ori_name):
            os.remove(nor_name)
        if os.path.exists(ori_name):
            os.remove(cmap_name)
        if os.path.exists(ori_name):
            os.remove(uv_name)
        
        
    for x in os.scandir(data_dir+"img/"):
        if not check_paths(data_dir, x.name):
            print("remove files: ", data_dir+ x.name)
            del_files(data_dir, x.name)
            continue

# --------------------------------------------------------
#                       Test
# --------------------------------------------------------
def pre_normal(image_dir):
    data_path = "/home1/quanquan/datasets/generate/mesh_film_hypo_alpha2/"
    dataset = filmDataset_3(data_path, load_mod="nobw")
    
def test_dataset():
    
    from .load_data_2 import filmDataset_3
    from .data_process import reprocess_auto
    from torch.utils.data import Dataset, DataLoader
    data_path = "/home1/quanquan/datasets/generate/mesh_film_small/"
    dataset = filmDataset_3(data_path, load_mod="original")

    #ori,ab,dep,nor,cmap,uv,bg = dataset.__getitem__(0)
    
    ori,ab,dep,nor,cmap,uv,bg = dataset.__getitem__(0)
    print("----- Test Data ------")
    print(np.min(nor), np.min(dep), np.min(cmap))
    print(np.max(nor),np.max(dep),np.max(cmap))
    
    print_img_with_reprocess(ori, "ori")
    print_img_with_reprocess(ab,  "ab")
    print_img_with_reprocess(dep, "depth")
    print_img_with_reprocess(nor, "normal")
    print_img_with_reprocess(cmap, "cmap")
    print_img_with_reprocess(uv , "uv")
    print_img_with_reprocess(bg,  "bg")
    print("dasdasdasdsa")

def test_extra_bg():
    data_path = "/home1/quanquan/datasets/generate/mesh_film_hypo_alpha2/"
    dataset = filmDataset_3(data_path, load_mod="extra_bg") 
    ori,ab,dep,nor,cmap,uv,bg = dataset.__getitem__(0)
    
    print("Print Img Auto: ")
    print_img_with_reprocess(ori, "ori")
    print_img_with_reprocess(ab,  "ab")
    print_img_with_reprocess(dep, "exr")
    print_img_with_reprocess(nor, "exr")
    print_img_with_reprocess(cmap, "exr")
    print_img_with_reprocess(uv , "uv")
    print_img_with_reprocess(bg,  "bg")
    # print_img_auto(ori, "ori")
    # print_img_auto(ab,  "ab")
    # print_img_auto(dep, "depth")
    # print_img_auto(nor, "normal")
    # print_img_auto(cmap, "cmap")
    # print_img_auto(uv , "uv")
    # print_img_auto(bg,  "bg")

def test_cmap_xyz():
    data_path = "/home1/quanquan/datasets/generate/mesh_film_hypo_alpha2/"
    dataset = filmDataset_3(data_path, load_mod="original")
    ori,ab,dep,nor,cmap,uv,bg = dataset.__getitem__(0)
    
    print_img_auto(cmap, "cmap")
    print_img_auto(cmap[:,:,0], "bg")
    print_img_auto(cmap[:,:,1], "bg")
    print_img_auto(cmap[:,:,2], "bg")
            

if __name__ == "__main__":
    # data_cleaning()
    # test_dataset()
    # test_extra_bg()
    test_cmap_xyz()