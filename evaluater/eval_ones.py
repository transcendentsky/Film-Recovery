# coding: utf-8
import numpy as np
import math
import numpy as np
from PIL import Image 
from scipy.signal import convolve2d
# from skimage.measure import compare_ssim as ssim
# from skimage.metrics import structural_similarity as ssim
from ssim.ssimlib import SSIM as ssim
import cv2
from .eval_ssim import cal_tensor_ssim, cal_tensor_msssim
import torch
from dataloader.bw_mapping import bw_mapping_batch_3
from dataloader.data_process import process_np2t_auto

"""
Methods:

cal_CC 
    cal_CC_batch  # input shape [B, C, H, W]
cal_PSNR
cal_SSIM
cal_MSE
cal_MAE 

"""
def cal_ALL(img1, img2):
    return cal_MSE(img1, img2), cal_CC(img1, img2), cal_PSNR(img1, img2), cal_SSIM(img1, img2)


def cal_MAE(img1, img2):
    assert type(img1) is np.ndarray, "Type of img1 is Error, expected np.ndarray but got {}".format(type(img1))
    assert type(img2) is np.ndarray, "Type of img2 is Error, expected np.ndarray but got {}".format(type(img2))
    
    mae = np.mean( abs(img1 - img2)  )
    return mae  

def cal_CC(img1, img2):
    assert type(img1) is np.ndarray, "Type of img1 is Error, expected np.ndarray but got {}".format(type(img1))
    assert type(img2) is np.ndarray, "Type of img2 is Error, expected np.ndarray but got {}".format(type(img2))
    cc = np.corrcoef(img1.reshape(-1), img2.reshape(-1))[0,1]
    return cc

def cal_CC_batch(img1, img2):
    """
    Average Correlation Coefficient
    Input type: Numpy (ndarray)  输入 numpy
    Output: Avg_precions
    """
    assert type(img1) is np.ndarray, "Type of img1 is Error, expected np.ndarray but got {}".format(type(img1))
    assert type(img2) is np.ndarray, "Type of img2 is Error, expected np.ndarray but got {}".format(type(img2))
    pccs=0
    b= img1.shape[0]
    for batch in range(b):
        pred_b = img1[batch, :, :, :].reshape(-1)
        GT_b = img2[batch, :, :, :].reshape(-1)
        cc = np.corrcoef(pred_b, GT_b)[0,1]
        pccs += cc
    return pccs/b


def cal_PSNR(img1, img2):
    """
    PSNR
    """
    assert type(img1) is np.ndarray, "Type of img1 is Error, expected np.ndarray but got {}".format(type(img1))
    assert type(img2) is np.ndarray, "Type of img2 is Error, expected np.ndarray but got {}".format(type(img2))
    assert np.max(img1) > 1, "The value should be [0, 255], max_value is {}".format(np.max(img1))
    assert np.max(img2) > 1, "The value should be [0, 255], max_value is {}".format(np.max(img2))
    
    mse = np.mean( (img1/255. - img2/255.) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def cal_MSE(img1, img2):
    """
    MSE
    """
    assert type(img1) is np.ndarray, "Type of img1 is Error, expected np.ndarray but got {}".format(type(img1))
    assert type(img2) is np.ndarray, "Type of img2 is Error, expected np.ndarray but got {}".format(type(img2))
    assert np.max(img1) > 1, "The value should be [0, 255], max_value is {}".format(np.max(img1))
    assert np.max(img2) > 1, "The value should be [0, 255], max_value is {}".format(np.max(img2))
    
    mse = np.mean( (img1/255. - img2/255.) ** 2 )
    return mse

def cal_SSIM(img1, img2, rgb=True):
    # print("rgb ", rgb)
    # return ssim(img1,img1) ## for rgb , func in skimage, work bad
    # _ssim = ssim(img2)
    # img1 = img1[np.newaxis, :,:,:].transpose(0,3,1,2)
    # img2 = img2[np.newaxis, :,:,:].transpose(0,3,1,2)

    return None

def cal_2SSIM_batch(img1, img2):
    img1 = img1.transpose(0,3,1,2)
    img2 = img2.transpose(0,3,1,2)
    img1tensor = torch.from_numpy(img1)
    img2tensor = torch.from_numpy(img2)
    
    ssim = cal_tensor_ssim(img1tensor, img2tensor)
    msssim = cal_tensor_msssim(img1tensor, img2tensor)

    return ssim, msssim
    
def cal_de_mean_ALL(img_gt, bw, bw_gt):
    # img2,  are  ground truth
    assert np.ndim(img_gt) == 3
    assert np.ndim(bw) == 3
    assert np.ndim(bw_gt) == 3
    
    dewarp1 = bw_mapping_batch_3(img_gt[np.newaxis, :,:,:], bw[np.newaxis,:,:,:])[0,:,:,:]
    dewarp_gt = bw_mapping_batch_3(img_gt[np.newaxis, :,:,:], bw_gt[np.newaxis,:,:,:])[0,:,:,:]
        
    dewarp_tensor = process_np2t_auto(dewarp1[np.newaxis, :,:,:])
    dewarp_gt_tensor = process_np2t_auto(dewarp_gt[np.newaxis, :,:,:])
    
    p1 = cal_PSNR(dewarp1, dewarp_gt)    
    c1 = cal_CC(dewarp1, dewarp_gt)
    s1 = cal_tensor_ssim(dewarp_tensor, dewarp_gt_tensor)
    ms1 = cal_tensor_msssim(dewarp_tensor, dewarp_gt_tensor)
    
    bw2, std, m1, m2 = process_de_mean(bw, bw_gt)
    
    dewarp2 = bw_mapping_batch_3(img_gt[np.newaxis, :,:,:], bw2[np.newaxis,:,:,:])[0,:,:,:]  
    dewarp2_tensor = process_np2t_auto(dewarp2[np.newaxis, :,:,:])
    
    p2 = cal_PSNR(dewarp2, dewarp_gt)    
    c2 = cal_CC(dewarp2, dewarp_gt)
    s2 = cal_tensor_ssim(dewarp2_tensor, dewarp_gt_tensor)
    ms2 = cal_tensor_msssim(dewarp2_tensor, dewarp_gt_tensor)
    
    return p1, c1, s1, ms1, p2, c2, s2, ms2, std, m1, m2
    
    
def process_de_mean(bw, bw_gt):
    assert np.ndim(bw) == 3
    assert np.ndim(bw_gt) == 3
    
    diff1 = bw - bw_gt
    max1 = np.max(diff1)
    min1 = np.min(diff1)
    diff_p1 = (diff1 - min1) / (max1 - min1) * 255
    mean1_1 = np.average(diff1[:,:,0])
    mean1_2 = np.average(diff1[:,:,1])

    diff1[:,:,0] = diff1[:,:,0] - mean1_1
    diff1[:,:,1] = diff1[:,:,1] - mean1_2

    bw_np_11 = bw[:,:,0] - mean1_1
    bw_np_12 = bw[:,:,1] - mean1_2
    bw_np_1 = np.stack([bw_np_11, bw_np_12], axis=-1)
    # --------------------
    std1 = np.std(diff1)
    m1_1 = np.abs(mean1_1)
    m1_2 = np.abs(mean1_2)
    
    # return bw_np_1
    return bw_np_1, std1, m1_1, m1_2

if __name__ == "__main__":
    a = np.random.rand(300,300,3)
    b = np.random.rand(300,300,3)
    aa = cv2.imread("medical/corgi1.jpg")
    bb = aa
    # print(aa.shape ,type(aa))
    # print(cal_CC(a, b))
    # cal_PSNR(a, b)
    # cal_MSE(a, b)
    # cal_MAE(a, b)
    print(cal_SSIM(a, b))