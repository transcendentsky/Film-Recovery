import torch
import torch.nn as nn
from torch.nn import SmoothL1Loss

def loss_deform(deform_img, deform_gt, imsize=256):
    avg = nn.AvgPool2d(256, stride=1)
    diff = deform_img - deform_gt
    _avg = avg(diff)
    diff_wo_avg = diff - _avg
    smoothl1 = SmoothL1Loss()
    loss1 = smoothl1(deform_img-_avg, deform_gt)
    criterion2 = torch.nn.MSELoss()
    loss2 = criterion2(_avg, torch.zeros_like(_avg))
    return loss1, loss2

if __name__ == "__main__":
    tensor_a = torch.ones((3,3,256,256))
    tensor_b = torch.ones((3,3,256,256))
    l1, l2 = loss_deform(tensor_a, tensor_b)
    print(l1, l2)