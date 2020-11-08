import numpy as np
import torch

def bw2deform(bw):
    assert type(bw) == np.ndarray
    imsize = bw.shape[0]
    x = np.arange(imsize)
    y = np.arange(imsize)
    xi, yi = np.meshgrid(x, y)
    bw[:,:,0] = bw[:,:,0] - yi
    bw[:,:,1] = bw[:,:,1] - xi
    # bw[:,:,0] = bw[:,:,0] - xi
    # bw[:,:,1] = bw[:,:,1] - yi
    return bw

def deform2bw_np(deform):
    pass
    raise NotImplementedError("WTF! this Func NOT implemented")

def deform2bw_tensor_batch(deform):
    imsize = deform.size(-1)
    x = torch.arange(imsize)
    y = torch.arange(imsize)
    xi, yi = torch.meshgrid(x, y)
    for i in deform.size[0]:
        deform[i, 0, :, :] = deform[i, 0, :, :] + yi
        deform[i, 1, :, :] = deform[i, 1, :, :] + xi
    return deform