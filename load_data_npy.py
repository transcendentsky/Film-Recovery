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
from genDataNPY import repropocess
from dataloader import data_process, print_img, uv2bw


os.environ["CUDA_VISIBLE_DEVICES"] = "4"

ori_image_dir = 'data_2000/Mesh_Film/npy/'

EPOCH = 1
test_BATCH_SIZE = 100


class filmDataset(Dataset):
    def __init__(self, npy_dir, load_mod="all", npy_dir_2=None):
        self.npy_dir = npy_dir
        self.npy_list = np.array([x.path for x in os.scandir(npy_dir) if x.name.endswith(".npy")])[:32]
        if npy_dir_2!=None:
            self.npy_list_2 = np.array([x.path for x in os.scandir(npy_dir_2) if x.name.endswith(".npy")])
            self.npy_list = np.append(self.npy_list, self.npy_list_2)
        self.npy_list.sort()
        self.load_mod = load_mod
        # self.input_size =(256, 256)
        # print(self.record_files)

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
            uv_2 = data_process.repropocess_auto(uv, "uv")
            uv_2 = uv_2.transpose((1,2,0))
            mask = background.transpose((1,2,0))
            bw = uv2bw.uv2backward_trans_3(uv_2, mask)
            bw = data_process.process_auto(bw, "bw")
            bw = bw.transpose((2,0,1)) / 255.0

            # print_img.print_cmap(cmap)
            # print_img.print_bw((bw+1)/2*255)

            return  torch.from_numpy(cmap), \
                torch.from_numpy(uv), \
                torch.from_numpy(bw), \
               torch.from_numpy(background)

    def __len__(self):
        return len(self.npy_list)


def test_corners(bw):
    print("test loss FUNC")
    bw_tensor = torch.from_numpy(bw.astype(np.float32).transpose((2,0,1)))
    bw_tensor = torch.unsqueeze(bw_tensor, axis=0)
    print("bw_map: ", bw_tensor.shape)
    replication = nn.ReplicationPad2d(1)
    bw_tensor = replication(bw_tensor)
    sample = nn.MaxPool2d(kernel_size=1, stride=64)
    bw_tensor = sample(bw_tensor)
    print(bw_tensor)


def mainT():

    device = torch.device("cuda")
    dataset_test = filmDataset(npy_dir=ori_image_dir)
    dataset_test_loader = DataLoader(dataset_test,
                                     batch_size=test_BATCH_SIZE,
                                     num_workers=1,
                                     shuffle=False,)
                                     # collate_fn = collate_fn)
                                     # collate_fn=callot.PadCollate(dim=0))     #
    print('dataset_test_loader', dataset_test_loader)
    for epoch in range(EPOCH):
        start_time = time.time()
        for i, data in enumerate(dataset_test_loader):

            print('start')
            """
            data 的数据格式是[tuple, tuple]_batchsize个，每个tuple里面是三个Tensor
            """
            ori = data[0]
            ab = data[1]
            depth = data[2]
            normal = data[3]
            uv = data[4]
            cmap = data[5]
            # uv = data[6]
            ori, ab, depth, normal, uv, cmap = ori.to(device), ab.to(device), depth.to(device), normal.to(device), uv.to(device), cmap.to(device)

            print('ori', ori.size())
            print('It took {} seconds to load {} samples'.format(float(time.time()-start_time), test_BATCH_SIZE))
            start_time = time.time()


    # duration = float(time.time()-start_time)
    # print('It cost', duration, 'seconds')


if __name__ =='__main__':
    data_dir ='/home1/qiyuanwang/film_generate/npy'
    d = filmDataset(data_dir, load_mod="uvbw")
    data = d.__getitem__(2)
    print(data[0].shape) # cmap
    print(data[1].shape)
    print(data[2].shape)
    print(data[3].shape)

    print_img.print_img_tensor(data_process.repropocess_auto(data[0], "cmap"), "cmap")
    print_img.print_img_tensor(data_process.repropocess_auto(data[1], "uv"), "uv")
    print_img.print_img_tensor(data_process.repropocess_auto(data[2], "bw"), "bw")
    print_img.print_img_tensor(data_process.repropocess_auto(data[3], "background"), "background")

    ########
    # mainT()
