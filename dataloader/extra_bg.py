# coding: utf-8
# Add background from real films for synthesis data 
# 给合成数据添加真实图片的背景

import numpy as np
import os
import sys
import cv2
from tutils import *


def get_background(img_path):
    # (525,351) (1444,1567)  real_film 16
    # (537,369) (1437,1413)  real_film 12
    
    bg_dir = "imgshow_test2"
    img_list = ["imgshow_test2/pad_gaus_40_real_film_12.jpg.jpg" , "imgshow_test2/pad_gaus_40_real_film_16.jpg.jpg"]
    img_bbox_list = [[(525,351), (1444,1567)], [(537,369), (1437,1413)]] 

    print("bg path: ", img_path)
    parent, name = os.path.split(img_path)
    ori = cv2.imread(img_path)
    
    h, w, c = ori.shape
    print("img shape: ", h,w,c)
    # imsize = max(h,w) + 40
    # img_pad = np.zeros((imsize, imsize, c))
    # pad_h = int((imsize - h)/2)
    # pad_w = int((imsize - w)/2)
    
    # img_pad[pad_h:pad_h+h, pad_w:pad_w+w,:] = ori
    # hole_size = int(imsize/2)
    # pad_hole = int((imsize - hole_size)/2)
    
    # img_pad[pad_hole:pad_hole+hole_size, pad_hole:pad_hole+hole_size, :] = 0
    
    cv2.imwrite(tfilename("imgshow_test_bg/img_pad_{}.jpg".format(name)), img_pad)
    
def get_bg():
    index = random.choice([0,1])
    bg_dir = "imgshow_test2"
    img_list = ["imgshow_test2/pad_gaus_40_real_film_16.jpg.jpg" , "imgshow_test2/pad_gaus_40_real_film_12.jpg.jpg"]
    img_bbox_list = [[(525,351), (1444,1567)], [(537,369), (1437,1413)]] 
    img_path = img_list[index]
    print("bg path: ", img_path)
    parent, name = os.path.split(img_path)
    ori = cv2.imread(img_path)  
    
    img_pad = ori
    x1, y1 = img_bbox_list[index][0]
    x2, y2 = img_bbox_list[index][1]
    img_pad[y1:y2, x1:x2, :] = 0
    
    cv2.imwrite(tfilename("imgshow_test_bg/hallowed.jpg"), img_pad)
    return ori, (x1,x2,y1,y2), img_pad

def get_bbox_fake_data(bg):
    h,w = bg.shape
    print("BG shape: ", h,w)
    X, Y = np.meshgrid(range(h), range(w))
    avg = np.mean(X)
    X = np.where(bg>0, X, int(avg))
    Y = np.where(bg>0, Y, int(avg))
    x_min = int(np.min(X))
    x_max = int(np.max(X))
    y_min = int(np.min(Y))
    y_max = int(np.max(Y))
    
    print(x_min, x_max, y_min, y_max)
    return (x_min, x_max, y_min, y_max)

def test_get_content():
    from .load_data_2 import filmDataset_3
    
    dataset = filmDataset_3("/home1/quanquan/datasets/generate/mesh_film_small/", load_mod="original")
    ori,ab,dep,nor,cmap,uv,bg = dataset.__getitem__(5)
    print("img shapes: ", ori.shape, bg.shape)
    ori = cv2.resize(ori, (448,448))
    x_min, x_max, y_min, y_max = get_bbox_fake_data(bg)
    
    # result = np.zeros((y_max-y_min, x_max-x_min, 3))
    result = ori[y_min:y_max, x_min:x_max, :]
    cv2.imwrite(tfilename("imgshow_test_bg/test_content.jpg"), ori)
    cv2.imwrite(tfilename("imgshow_test_bg/test_content2.jpg"), result)

if __name__ == "__main__":
    # background_img_dir = "imgshow_test2"
    # bg_list = np.array([x.path for x in os.scandir(background_img_dir) if x.name.endswith("jpg")])
    # bg_list.sort()
    # get_background(bg_list[0])
    
    get_bg()
    
    test_get_content()