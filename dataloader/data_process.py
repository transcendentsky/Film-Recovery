import numpy as np
import torch
from tutils import *

def reprocess_auto_batch(input_, img_type):
    if type(input_) is torch.Tensor:
        input_np = input_.detach().cpu().numpy().transpose((0,2,3,1))
    elif type(input_) is np.ndarray:
        input_np = input_
    else:
        raise TypeError("Wrong type: Got {}".format(type(img)))
    
    bs = input_np.shape[0]
    output = np.zeros_like(input_np)
    for i in range(bs):
        output[i,:,:,:] = reprocess_np_auto(input_np[i,:,:,:], img_type)
    return output

def reprocess_auto(input_, img_type):
    p("[*] Reprocess ", img_type, input_.shape)
    if type(input_) is torch.Tensor:
        img = input_.detach().cpu().numpy().transpose((1,2,0))
        assert img.shape[0] > 3, "Dims Error!! the expected Dims is (256?, 256?, n) but GOT {}".format(input_.shape)
        return reprocess_np_auto(img, img_type)
    elif type(input_) is np.ndarray:
        assert input_.shape[0] > 3, "Dims Error!! the expected Dims is (256?, 256?, n) but GOT {}".format(input_.shape)
        return reprocess_np_auto(input_, img_type)
    else:
        raise TypeError("Wrong type: Got {}".format(type(img)))

def reprocess_np_auto(input_, img_type, process_type="qiyuan"):
    """
    Reprocess for Maps
        depth, normal, cmap
        uv, bw, background
        ori, ab

    Process type: liuli / qiyuan
    """
    assert type(input_) is np.ndarray, "Trans TypeError!! the expected type is np.ndarray but GOT {}".format(type(input_))
    assert input_.shape[0] > 3, "Dims Error!! the expected Dims is (256?, 256?, n) but GOT {}".format(input_.shape)
    assert np.ndim(input_) <= 3, "np.ndim Error"

    if process_type == "liuli":
        if img_type == "normal":
            mean=[0.584, 0.294, 0.300]
            std=[0.483, 0.251, 0.256]
        elif img_type == "cmap":
            mean=[0.100, 0.326, 0.289]
            std=[0.096, 0.332, 0.298]
        elif img_type == "depth":
            mean=[0.316, 0.316, 0.316]
            std=[0.309, 0.309, 0.309]
    elif process_type == "qiyuan":
        if img_type == "normal":
            mean=[0.5619, 0.2881, 0.2917]
            std=[0.5619, 0.7108, 0.7083]
        elif img_type == "cmap":
            mean=[0.1108, 0.3160, 0.2859]
            std=[0.7065, 0.6840, 0.7141]
        elif img_type == "depth": 
            mean=[0.5, 0.5, 0.5]
            std=[0.5, 0.5, 0.5]

    if img_type in ["normal", "cmap"]:
        ###  Recover from STD and MEAN
        x = np.zeros(input_.shape)
        assert x.shape[2] == 3, "np.shape Error, Got {}".format(x.shape)
        x[:, :, 0] = input_[:, :, 0] * std[0] + mean[0]
        x[:, :, 1] = input_[:, :, 1] * std[1] + mean[1]
        x[:, :, 2] = input_[:, :, 2] * std[2] + mean[2]

    elif img_type in ["depth"]:
        if np.ndim(input_)==3:  # (256, 256, 1)
            x = input_[:, :, 0]
            x[:, :] = x[:, :] * std[0] + mean[0]
        elif np.ndim(input_)==2:  
            x = np.zeros(input_.shape)
            x[:, :] = input_[:, :] * std[0] + mean[0]
            

    elif img_type in ["bw", "uv", "background"]:
        ### Recover from [-1, 1] to  [0, 1]
        if img_type == "uv":
            x = (input_ + 1.0) / 2.0
        elif img_type == "bw":
            x = (input_ + 1.0) / 2.0 * 255.0
        elif img_type == "background":
            if np.ndim(input_)==3:  # (256, 256, 1)
                x = input_
            elif np.ndim(input_)==2:  
                x = input_[:, :, np.newaxis]
    
    elif img_type in ["ori", "ab"]:
        x = (input_ + 1.) / 2. * 255.

    else:
        raise ValueError("[Trans BUG] reprocess_auto Error")

    return x


def process_auto(input_, img_type, process_type="qiyuan"): 
    """
    Reprocess for Maps
        depth, normal, cmap
        uv, bw, background
        ori, ab

    Process type: liuli / qiyuan
    """
    assert type(input_) is np.ndarray, "Trans TypeError!! the expected type is np.ndarray but GOT {}".format(type(input_))
    assert input_.shape[0] > 3, "Dims Error!! the expected Dims is (256?, 256?, n) but GOT {}".format(input_.shape)
    assert np.ndim(input_) <= 3, "np.ndim Error"

    input_ = input_.astype(np.float32)

    if process_type == "liuli":
        if img_type == "normal":
            mean=[0.584, 0.294, 0.300]
            std=[0.483, 0.251, 0.256]
        elif img_type == "cmap":
            mean=[0.100, 0.326, 0.289]
            std=[0.096, 0.332, 0.298]
        elif img_type == "depth":
            mean=[0.316, 0.316, 0.316]
            std=[0.309, 0.309, 0.309]
    elif process_type == "qiyuan":
        if img_type == "normal":
            mean=[0.5619, 0.2881, 0.2917]
            std=[0.5619, 0.7108, 0.7083]
        elif img_type == "cmap":
            mean=[0.1108, 0.3160, 0.2859]
            std=[0.7065, 0.6840, 0.7141]
        elif img_type == "depth": 
            mean=[0.5, 0.5, 0.5]
            std=[0.5, 0.5, 0.5]


    if img_type in ["depth", "normal", "cmap"]:  # original value is [0, 1]

        ###  Recover from STD and MEAN
        x = np.zeros(input_.shape)
        assert x.shape[2] == 3, "np.shape Error, Got {}".format(x.shape)
        if np.ndim(input_)==3:           # cmap, normal
            x[:, :, 0] = (input_[:, :, 0] - mean[0]) / std[0]
            x[:, :, 1] = (input_[:, :, 1] - mean[1]) / std[1]
            x[:, :, 2] = (input_[:, :, 2] - mean[2]) / std[2]

    elif img_type in ["depth"]:
        if np.ndim(input_)==3:  # (256, 256, 1)
            x = np.zeros(input_.shape)
            x[:, :, 0] = (input_[:, :, 0] - mean[0]) / std[0]
        elif np.ndim(input_)==2:  
            x = np.zeros(input_.shape)
            x[:, :] = (input_[:, :] - mean[0]) / std[0]
            x = x[:, :, np.newaxis]

    elif img_type in ["bw", "uv", "background"]:
        ### Recover from [-1, 1] to  [0, 1]
        if img_type == "uv":
            x = input_ * 2.0 - 1.0
        elif img_type == "bw":
            x = input_ / 255.0 * 2.0 - 1.0
        elif img_type == "background":
            x = input_

    elif img_type in ["ori", "ab"]:
        x = input_  / 255. * 2. -1.

    else:
        raise ValueError("[Trans BUG] reprocess_auto Error")

    return x