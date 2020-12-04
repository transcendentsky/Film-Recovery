# -*- coding: utf-8 -*-
import torchvision.transforms as transforms
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
import cv2

ori_image_dir = 'data_2000/Mesh_Film/npy/'

EPOCH = 1
test_BATCH_SIZE = 100

def bw2deform(bw):
    assert type(bw) == np.ndarray
    im_size = bw.shape[-1]
    bw = ((bw + 1.)/2.)*im_size
    x = np.arange(im_size)
    y = np.arange(im_size)
    xi, yi = np.meshgrid(x, y)
    bw[0, :, :] = bw[0, :, :] - yi
    bw[1, :, :] = bw[1, :, :] - xi
    return bw/im_size

class filmDataset(Dataset):
    def __init__(self, npy_dir, npy_dir_2=None):
        self.npy_dir = npy_dir
        self.npy_list = np.array([x.path for x in os.scandir(npy_dir) if x.name.endswith(".npy")])
        if npy_dir_2!=None:
            self.npy_list_2 = np.array([x.path for x in os.scandir(npy_dir_2) if x.name.endswith(".npy")])
            self.npy_list = np.append(self.npy_list, self.npy_list_2)
        self.npy_list.sort()
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
        bw = data['bw']
        # ori_1080 = data['ori_1080']


        return torch.from_numpy(ori), \
               torch.from_numpy(ab), \
               torch.from_numpy(depth), \
               torch.from_numpy(normal), \
               torch.from_numpy(cmap), \
               torch.from_numpy(uv), \
               torch.from_numpy(background), \
               torch.from_numpy(bw),\
               # torch.from_numpy(ori_1080), \
            # torch.from_numpy(bmap)

    # torch.from_numpy(bmap), \
    # torch.unsqueeze(torch.from_numpy(depth),0), \

    def __len__(self):
        return len(self.npy_list)


class filmDataset_with_name(Dataset):
    def __init__(self, npy_dir, npy_dir_2=None):
        self.npy_dir = npy_dir
        self.npy_list = np.array([x.path for x in os.scandir(npy_dir) if x.name.endswith(".npy")])
        if npy_dir_2!=None:
            self.npy_list_2 = np.array([x.path for x in os.scandir(npy_dir_2) if x.name.endswith(".npy")])
            self.npy_list = np.append(self.npy_list, self.npy_list_2)
        self.npy_list.sort()
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
        bw = data['bw']
        name = npy_path.split('/')[-1].split('.')[0]
        # ori_1080 = data['ori_1080']


        return torch.from_numpy(ori), \
               torch.from_numpy(ab), \
               torch.from_numpy(depth), \
               torch.from_numpy(normal), \
               torch.from_numpy(cmap), \
               torch.from_numpy(uv), \
               torch.from_numpy(background), \
               name,\
               torch.from_numpy(bw),\
               # torch.from_numpy(ori_1080), \
            # torch.from_numpy(bmap)

    # torch.from_numpy(bmap), \
    # torch.unsqueeze(torch.from_numpy(depth),0), \

    def __len__(self):
        return len(self.npy_list)

class DeFilmDataset(Dataset):
    def __init__(self, npy_dir):
        self.npy_dir = npy_dir
        self.npy_list = np.array([x.path for x in os.scandir(npy_dir) if x.name.endswith(".npy")])
        self.npy_list.sort()

    def __getitem__(self, index):
        npy_path = self.npy_list[index]
        """loading"""
        # data = np.load(self.npy_dir + '/' + npy_name, allow_pickle=True)[()]
        data = np.load(npy_path, allow_pickle=True)[()]
        ori = data['ori']
        ab = data['ab']
        depth = data['depth']
        normal = data['normal']
        uv = data['uv']
        cmap = data['cmap']
        background = data['background']
        bw = data['bw']
        deform = bw2deform(bw.copy())
        name = npy_path.split('/')[-1].split('.')[0]
        
        # ori_1080 = data['ori_1080']


        return torch.from_numpy(ori), \
               torch.from_numpy(ab), \
               torch.from_numpy(depth), \
               torch.from_numpy(normal), \
               torch.from_numpy(cmap), \
               torch.from_numpy(uv), \
               torch.from_numpy(background), \
               torch.from_numpy(bw),\
               torch.from_numpy(deform),\
               name,\
               # torch.from_numpy(ori_1080), \
            # torch.from_numpy(bmap)

    # torch.from_numpy(bmap), \
    # torch.unsqueeze(torch.from_numpy(depth),0), \

    def __len__(self):
        return len(self.npy_list)
        
class single_test(Dataset):
    def __init__(self, npy_dir):
        self.npy_dir = npy_dir
        #self.npy_list = np.array([x.path for x in os.scandir(npy_dir) if x.name.endswith(".npy")])
        self.npy_list = np.array([x.path for x in os.scandir(npy_dir)])
        self.npy_list.sort()
        self.npy_list = self.npy_list[:100]

    def __getitem__(self, index):
        npy_path = self.npy_list[index]
        if npy_path[-3:] == 'npy': 
            #ori = np.load(npy_path)
            ori = np.load(npy_path,allow_pickle=True)[()]['ori']
        else:
            ori = np.transpose((cv2.resize(cv2.imread(npy_path),(256,256))/255.*2. - 1.),(2,0,1)).astype(np.float32)
        name = npy_path.split('/')[-1].split('.')[0]


        return torch.from_numpy(ori), \
               name,\

    def __len__(self):
        return len(self.npy_list)

def mainT():

    device = torch.device("cuda")
    #dataset_test = filmDataset(npy_dir=ori_image_dir)
    ori_image_dir = '/home1/qiyuanwang/film_generate/npy_with_bw'
    dataset_test = filmDataset_with_name(npy_dir=ori_image_dir)
    dataset_test_loader = DataLoader(dataset_test,
                                     batch_size=50,
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
            uv = data[5]
            cmap = data[4]
            mask = data[6]
            name = data[7]
            bw = data[8]
            
            # uv = data[6]
            ori, ab, depth, normal, uv, cmap = ori.to(device), ab.to(device), depth.to(device), normal.to(device), uv.to(device), cmap.to(device)

            print('ori', ori.size())
            print('It took {} seconds to load {} samples'.format(float(time.time()-start_time), test_BATCH_SIZE))
            start_time = time.time()
            print(name)


    # duration = float(time.time()-start_time)
    # print('It cost', duration, 'seconds')

if __name__ =='__main__':
    mainT()
