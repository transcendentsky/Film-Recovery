import torch
import cv2
import numpy as np

def blur_bw_np(bm, imsize=None):
    print("bw shape:", bm.shape)
    w,h = bm.shape[0], bm.shape[1]
    bm0=cv2.blur(bm[:,:,0],(3,3))
    bm1=cv2.blur(bm[:,:,1],(3,3))
    if imsize is not None:
        bm0=cv2.resize(bm0,(imsize[0],imsize[1]))# 
        bm1=cv2.resize(bm1,(imsize[0],imsize[1]))
    bm=np.stack([bm0,bm1],axis=-1)
    return bm


def resize_albedo_np(ori, img, ab):
    """
    ori: image with the real size (1080, 1080, 3)
    img: resized ori  (256, 256, 3)
    ab : predicted ab (256, 256, 3)
    """
    assert img.shape == (256,256,3) , "Error, but got {}".format(img.shape)
    assert ab.shape  == (256,256,3), "Error, but got {}".format(ab.shape)

    w,h = ori.shape[0], ori.shape[1]

    diff = ab - img

    diff0 = cv2.blur(diff[:,:,0], (5,5))  # Or kernel of (3,3)
    diff1 = cv2.blur(diff[:,:,1], (5,5))  # Or kernel of (3,3)
    diff2 = cv2.blur(diff[:,:,2], (5,5))  # Or kernel of (3,3)

    diff0=cv2.resize(diff0,(h,w)) 
    diff1=cv2.resize(diff1,(h,w))
    diff2=cv2.resize(diff2,(h,w)) 

    diff = np.stack([diff0, diff1, diff2], axis=-1)

    ab_large = ori + diff

    return ab_large