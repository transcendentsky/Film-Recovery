# coding: utf-8
import numpy as np
import math
import numpy as np
from PIL import Image 
from scipy.signal import convolve2d
# from skimage.measure import compare_ssim as ssim
from skimage.metrics import structural_similarity as ssim
import cv2
from .eval_ones import cal_CC, cal_PSNR, cal_SSIM, cal_MSE, cal_MAE 
from dataloader.uv2bw import uv2backward_trans_2, uv2backward_trans_3
import torch
from dataloader.uv2bw import uv2backward_batch_with_reprocess, uv2backward_batch
from dataloader.bw_mapping import bw_mapping_batch_2, bw_mapping_batch_3
from dataloader.data_process import reprocess_auto, reprocess_auto_batch
from tutils import *

def np2tensor(batch_a):
    a = batch_a.transpose((0,3,1,2))
    tensor_a = torch.from_numpy(a)
    return tensor_a

def tensor2np(batch_a):
    a = batch_a.detach().cpu().numpy().transpose((0, 2, 3, 1))
    return a

def cal_metrix_tensor_batch(img1_tensor, img2_tensor, metrix):
    img1 = tensor2np(img1_tensor)
    img2 = tensor2np(img2_tensor)
    return cal_metrix_np_batch(img1, img2, metrix)

def cal_metrix_np_batch(img1, img2, metrix):
    """
    Cal all metrix with two Numpy matrix
    """
    # assert np.max(img1) > 1, "The value should be [0, 255], max_value is {}".format(np.max(img1))
    # assert np.max(img2) > 1, "The value should be [0, 255], max_value is {}".format(np.max(img2))
    assert np.ndim(img1) == 4, "np.ndim Error ! Got {}".format(img1.shape)
    assert np.ndim(img2) == 4, "np.ndim Error ! Got {}".format(img2.shape)

    criterion_dict = {
        "cc": cal_CC, 
        "psnr": cal_PSNR,
        "ssim": cal_SSIM,
        "mse": cal_MSE,
        "mae": cal_MAE 
    }
    criterion = criterion_dict[metrix]

    bs = img1.shape[0]
    loss = 0.0
    for i in range(bs):
        loss += criterion(img1[i, :, :, :], img2[i, :, :, :])
        # print(loss)
    return loss/(bs*1.0), loss

# @tfuncname
def uvbw_loss_np_batch(uv, bw, bw_gt, mask, ori, metrix):
    ### For batches
    assert type(bw) is np.ndarray, "TypeError Got {}".format(type(bw))
    assert type(ori) is np.ndarray, "TypeError Got {}".format(type(ori))

    bw_from_uv = uv2backward_batch(uv, mask)
    ori_uv = bw_mapping_batch_3(ori, bw_from_uv)
    ori_bw = bw_mapping_batch_3(ori, bw)
    ori_bw_gt = bw_mapping_batch_3(ori, bw_gt)

    uv_bw_loss,  total_uv_bw_loss  = cal_metrix_np_batch(bw_from_uv, bw_gt, metrix)
    bw_loss,     total_bw_loss     = cal_metrix_np_batch(bw,    bw_gt,      metrix)
    ori_uv_loss, total_ori_uv_loss = cal_metrix_np_batch(ori_uv, ori_bw_gt, metrix)
    ori_bw_loss, total_ori_bw_loss = cal_metrix_np_batch(ori_bw, ori_bw_gt, metrix)
    
    return total_uv_bw_loss, total_bw_loss, total_ori_uv_loss, total_ori_bw_loss

def uvbw_loss_tensor_batch(uv_tensor, bw_tensor, mask_tensor, ori_tensor, metrix):
    ### uv2backward_trans_2
    assert type(uv_tensor) is torch.Tensor, "TypeError, Got {}".format(type(uv_tensor))
    uv = tensor2np(uv_tensor)
    bw = tensor2np(bw_tensor)
    mask = tensor2np(mask_tensor)
    ori = tensor2np(ori_tensor)

    uv = reprocess_auto_batch(uv, "uv")
    bw = reprocess_auto_batch(bw, "bw")
    mask = reprocess_auto_batch(mask, "background")
    ori = reprocess_auto_batch(ori, "ori")
    return uvbw_loss_np_batch(uv, bw, mask, ori, metrix)

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
    print(cal_SSIM(a, b, rgb=True))