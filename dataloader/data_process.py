import numpy as np
import torch
from tutils import *

def reprocess_auto_batch(_input, img_type):
    if type(_input) is torch.Tensor:
        input_np = _input.detach().cpu().numpy().transpose((0,2,3,1))
    elif type(_input) is np.ndarray:
        input_np = _input
    else:
        raise TypeError("Wrong type: Got {}".format(type(img)))
    
    bs = input_np.shape[0]
    output = np.zeros_like(input_np)
    for i in range(bs):
        output[i,:,:,:] = reprocess_np_auto(input_np[i,:,:,:], img_type)
    return output

def reprocess_auto(_input, img_type):
    p("[*] Reprocess ", img_type, _input.shape)
    if type(_input) is torch.Tensor:
        img = _input.detach().cpu().numpy().transpose((1,2,0))
        assert img.shape[0] > 3, "Dims Error!! the expected Dims is (256?, 256?, n) but GOT {}".format(_input.shape)
        return reprocess_np_auto(img, img_type)
    elif type(_input) is np.ndarray:
        assert _input.shape[0] > 3, "Dims Error!! the expected Dims is (256?, 256?, n) but GOT {}".format(_input.shape)
        return reprocess_np_auto(_input, img_type)
    else:
        raise TypeError("Wrong type: Got {}".format(type(img)))

def reprocess_np_auto(_input, img_type, process_type="qiyuan"):
    """
    Reprocess for Maps
        depth, normal, cmap
        uv, bw, background
        ori, ab

    Process type: liuli / qiyuan
    """
    assert type(_input) is np.ndarray, "Trans TypeError!! the expected type is np.ndarray but GOT {}".format(type(_input))
    assert _input.shape[0] > 3, "Dims Error!! the expected Dims is (256?, 256?, n) but GOT {}".format(_input.shape)
    assert np.ndim(_input) <= 3, "np.ndim Error"

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
        x = np.zeros(_input.shape)
        assert x.shape[2] == 3, "np.shape Error, Got {}".format(x.shape)
        x[:, :, 0] = _input[:, :, 0] * std[0] + mean[0]
        x[:, :, 1] = _input[:, :, 1] * std[1] + mean[1]
        x[:, :, 2] = _input[:, :, 2] * std[2] + mean[2]

    elif img_type in ["depth"]:
        if np.ndim(_input)==3:  # (256, 256, 1)
            x = _input[:, :, 0]
            x[:, :] = x[:, :] * std[0] + mean[0]
        elif np.ndim(_input)==2:  
            x = np.zeros(_input.shape)
            x[:, :] = _input[:, :] * std[0] + mean[0]
            

    elif img_type in ["bw", "uv", "background"]:
        ### Recover from [-1, 1] to  [0, 1]
        if img_type == "uv":
            x = (_input + 1.0) / 2.0
        elif img_type == "bw":
            x = (_input + 1.0) / 2.0 * 255.0
        elif img_type == "background":
            if np.ndim(_input)==3:  # (256, 256, 1)
                x = _input
            elif np.ndim(_input)==2:  
                x = _input[:, :, np.newaxis]
    
    elif img_type in ["ori", "ab"]:
        x = (_input + 1.) / 2. * 255.

    elif img_type in ["deform"]:
        x = _input * 255.

    else:
        raise ValueError("[Trans BUG] reprocess_auto Error")

    return x

def process_to_tensor(_input):
    if np.ndim(_input) == 3 :
        return torch.from_numpy(_input.transpose(2,0,1))
    elif np.ndim(_input) == 4:
        return torch.from_numpy(_input.transpose(0,3,1,2))
    else:
        raise ValueError("Got ndim {}".format(np.ndim(_input)))

def process_tensor_batch(_input):
    assert np.ndim(_input) == 4

def process_auto(_input, img_type, process_type="qiyuan"): 
    """
    Reprocess for Maps
        depth, normal, cmap
        uv, bw, background
        ori, ab

    Process type: liuli / qiyuan
    """
    assert type(_input) is np.ndarray, "Trans TypeError!! the expected type is np.ndarray but GOT {}".format(type(_input))
    assert _input.shape[0] > 3, "Dims Error!! the expected Dims is (256?, 256?, n) but GOT {}".format(_input.shape)
    assert np.ndim(_input) <= 3, "np.ndim Error"

    _input = _input.astype(np.float32)

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
        x = np.zeros(_input.shape)
        assert x.shape[2] == 3, "np.shape Error, Got {}".format(x.shape)
        if np.ndim(_input)==3:           # cmap, normal
            x[:, :, 0] = (_input[:, :, 0] - mean[0]) / std[0]
            x[:, :, 1] = (_input[:, :, 1] - mean[1]) / std[1]
            x[:, :, 2] = (_input[:, :, 2] - mean[2]) / std[2]

    elif img_type in ["depth"]:
        if np.ndim(_input)==3:  # (256, 256, 1)
            x = np.zeros(_input.shape)
            x[:, :, 0] = (_input[:, :, 0] - mean[0]) / std[0]
        elif np.ndim(_input)==2:  
            x = np.zeros(_input.shape)
            x[:, :] = (_input[:, :] - mean[0]) / std[0]
            x = x[:, :, np.newaxis]

    elif img_type in ["bw", "uv", "background"]:
        ### Recover from [-1, 1] to  [0, 1]
        if img_type == "uv":
            x = _input * 2.0 - 1.0
        elif img_type == "bw":
            x = _input / 255.0 * 2.0 - 1.0
        elif img_type == "background":
            x = _input

    elif img_type in ["ori", "ab"]:
        x = _input  / 255. * 2. -1.

    elif img_type in ["deform"]:
        x = _input /255.

    
    else:
        raise ValueError("[Trans BUG] reprocess_auto Error, Got img_type=", img_type)

    return x

def reprocess_t2t_auto(_input, img_type, process_type="qiyuan"):
    
    if img_type in ["bw", "ori"]:
        x = _input / 255.0 * 2.0 - 1.0
    elif img_type in ["deform"]:
        x = _input / 255.0
    elif img_type in ["np"]:
        x = _input.detach().cpu().numpy().tranpose(0,2,3,1)

    else:
        raise NotImplementedError("Error! Got type: {}".format(img_type))

    return x