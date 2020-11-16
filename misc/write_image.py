import cv2
import torch
import numpy as np
from genDataNPY import repropocess

def write_image(image_float, dir):
    image_uint8 = ((image_float+1)/2 *255).type(torch.uint8).cpu().numpy()
    cv2.imwrite(dir, image_uint8)


def write_cmap_gauss(image_float, dir, mean=[0.100, 0.326, 0.289], std=[0.096, 0.332, 0.298]):
    image_float = repropocess(image_float.detach().cpu().numpy(), mean, std)
    if dir[-3:]=='exr':
        cv2.imwrite(dir,image_float)
    else:
        image_uint8 = (np.clip(image_float,0.,1.)*255).astype(np.uint8)
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


def write_image_tensor(image, dir, dist, mean=None, std=None, device=None):

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
        write_cmap_gauss(image_hwc, dir, mean, std)
    elif dist == '01':
        write_image_01(image_hwc, dir)
    elif dist == 'std':
        write_image(image_hwc, dir)

def re_normalize(tensor, mean, std, inplace=True):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if tensor.ndim < 3:
        raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor

def write_tensor_to_image(tensor, save_path, mean, std, device=None):
    tensor = re_normalize(tensor.detach(), mean, std)
    if tensor.shape[0] == 3:
        torchvision.utils.save_image(torch.stack([tensor[2, :, :], tensor[1, :, :], tensor[0, :, :]], 0), fp=save_path, padding=0)
    elif tensor.shape[0] == 2:
        zeros = torch.zeros_like(tensor[0, :, :].unsqueeze(0), device=device)
        tensor = torch.cat([tensor, zeros], dim=0)
        torchvision.utils.save_image(torch.stack([tensor[2, :, :], tensor[1, :, :], tensor[0, :, :]], 0), fp=save_path,
                                     padding=0)
    else:
        torchvision.utils.save_image(tensor, fp=save_path,
                                     padding=0)

