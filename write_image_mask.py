import cv2
import torch
import numpy as np
from genDataNPY import repropocess_mask

def write_image(image_float, dir):
    image_uint8 = ((image_float+1)/2 *255).type(torch.uint8).cpu().numpy()
    cv2.imwrite(dir, image_uint8)


def write_cmap_gauss(image_float, dir, mean=[0.100, 0.326, 0.289], std=[0.096, 0.332, 0.298],mask=None):
    image_float = repropocess_mask(image_float.detach().cpu().numpy(), mean, std, mask.detach().cpu().numpy())
    image_uint8 = (image_float *255).astype(np.uint8)
    cv2.imwrite(dir, image_uint8)

def write_image_01(image_float, dir):
    image_uint8 = (image_float *255).type(torch.uint8).cpu().numpy()
    cv2.imwrite(dir, image_uint8)

def write_image_np(image_float, dir):

    """input (only backward map from uv2bmap function): [h, w, c] in [-1, 1]"""
    #bb_numpy = (-1.) * np.ones((256, 256, 1))
    bb_numpy = (-1.) * np.ones((256, 256, 1))
    image_float_3 = np.concatenate((image_float, bb_numpy), 2)
    if dir[-3:] == 'exr':
        image = ((image_float_3+1.)/2.).astype(np.float32)
    else:
        image = ((image_float_3+1)/2 *255).astype(np.uint8)
    cv2.imwrite(dir, image)


def write_image_tensor(image, dir, dist, mean=None, std=None, device=None,mask=None):

    """input: [channel, h, w]"""
    image_hwc = image.transpose(0, 1).transpose(1, 2)
    channel = image_hwc.size()[2]
    if channel ==2:
        #bb = (-1) * torch.ones((256, 256, 1)).to(device)
        bb = (-1) * torch.ones((256, 256, 1)).to(device)
        image_hwc = torch.cat([image_hwc, bb], 2)
    elif channel ==1:
        image_hwc = image_hwc[:, :, 0]

    if dist == 'gauss':
        mask_hwc = mask.transpose(0, 1).transpose(1, 2)
        mask_hwc = mask_hwc[:,:,0]
        write_cmap_gauss(image_hwc, dir, mean, std, mask_hwc)
    elif dist == '01':
        write_image_01(image_hwc, dir)
    elif dist == 'std':
        write_image(image_hwc, dir)