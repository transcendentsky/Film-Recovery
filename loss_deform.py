# coding: utf-8
# filename: loss_deform.py

import torch
import torch.nn as nn
from torch.nn import SmoothL1Loss
from dataloader.bw2deform import deform2bw_tensor_batch
from dataloader.data_process import reprocess_t2t_auto
from dataloader.bw_mapping import bw_mapping_tensor_batch
from evaluater.eval_ssim import cal_tensor_ssim

avg = nn.AvgPool2d(256, stride=1)
smoothl1 = SmoothL1Loss()
criterion2 = torch.nn.MSELoss()

def loss_deform(deform_img, deform_gt, imsize=256):
    diff = deform_img - deform_gt
    matrix2 = torch.ones_like(diff) * -1
    diff_ind = torch.where(diff>0, torch.ones_like(diff), matrix2)
    _avg = avg(diff)
    diff_wo_avg = diff - _avg
    diff_w_ind = torch.where(diff_wo_avg>0, torch.ones_like(diff), matrix2)
    ind = (diff_ind * diff_w_ind).detach()
    loss_ = torch.where(ind > 0, (diff - _avg), torch.zeros_like(diff))

    loss1 = smoothl1(loss_, torch.zeros_like(loss_))  
    ## Loss1 可以改成 SSIM Loss 或其他， 或者同时使用
    _std = torch.std(diff_wo_avg)
    loss2 = criterion2(_avg, torch.zeros_like(_avg))
    loss3 = criterion2(_std, torch.zeros_like(_std))
    return loss1, loss2, loss3

def recon_loss(deform_img, ori, dewarp_gt, imsize=256):
    deform_img = reprocess_t2t_auto(deform_img, "deform")
    bw = deform2bw_tensor_batch(deform_img)
    dewarp = bw_mapping_tensor_batch(ori, bw)
    ssim = cal_tensor_ssim(dewarp, dewarp_gt)
    loss = torch.mean(ssim)
    return loss


if __name__ == "__main__":
    tensor_a = torch.ones((3,3,256,256))
    tensor_b = torch.ones((3,3,256,256))
    l1, l2, l3 = loss_deform(tensor_a, tensor_b)
    print(l1, l2, l3)