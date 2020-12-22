import numpy as np
import cv2
import os
import sys
from tutils import *
import random
from tqdm import tqdm

def img_padding(img_path):
    parent, name = os.path.split(img_path)
    ori = cv2.imread(img_path)
    h, w, c = ori.shape
    print("img shape: ", h,w,c)
    imsize = max(h,w) + 40
    img_pad = np.zeros((imsize, imsize, c))
    pad_h = int((imsize - h)/2)
    pad_w = int((imsize - w)/2)
    
    img_pad[pad_h:pad_h+h, pad_w:pad_w+w,:] = ori
    cv2.imwrite(tfilename("imgshow_test/img_pad_{}.jpg".format(name)), img_pad)
    
    gkernel_size = int(imsize/12)
    print("Kernel" , gkernel_size)
    for i in tqdm(range(51)):
        img_pad = cv2.GaussianBlur(img_pad,(gkernel_size,gkernel_size),0)
        if i % 10 == 0:  
            img_pad2 = img_pad
            img_pad2[pad_h:pad_h+h, pad_w:pad_w+w,:] = ori
            cv2.imwrite(tfilename("imgshow_test2/pad_gaus_{}_{}.jpg".format(i, name)), img_pad2)

if __name__ == "__main__":
    img_dir = "/home1/quanquan/datasets/real_films/real_data_img"
    img_list = np.array([x.path for x in os.scandir(img_dir) ])
    img_list.sort()
    print(img_list)
    
    # img_padding(random.choice(img_list))
    for x in img_list:
        img_padding(x)