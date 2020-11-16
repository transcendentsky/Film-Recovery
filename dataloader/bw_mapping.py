import numpy as np
from scipy.interpolate import griddata
import torch
import torch.nn.functional as F
from tutils import *



def bw_mapping_tensor_batch(image, bw_map, device="cuda", imsize=256):
    # Processed ori and Processed bw_map
    
    assert type(bw_map) is torch.tensor

    bw_map = bw_map.transpose(1,2).transpose(2,3).transpose(1,2)
    bw_map = bw_map / (imsize*1.0) * 2.0 - 1.0
    output_tensor = F.grid_sample(input=image, grid=bw_map, align_corners=True)
    return output_tensor
    

def bw_mapping_single_3(image, bw_map):
    return bw_mapping_batch_3(image[np.newaxis,:,:,:], bw_map[np.newaxis,:,:,:])[0,:,:,:]

def bw_mapping_batch_3(image, bw_map, device="cuda", imsize=255):
    assert type(image) is np.ndarray, "Error Got {}".format(type(image))
    assert type(bw_map) is np.ndarray, "Error Got {}".format(type(bw_map))
    assert bw_map.shape[3] == 2, "shape Error, Got {}".format(bw_map.shape)

    image = image.transpose((0, 3, 1, 2))   # (2, 1)  ????
    image_tensor = torch.from_numpy(image).type(torch.float32).to(device)
    bw_map = bw_map[:, :, :, ::-1]
    bw_map = bw_map.transpose((0, 2, 1, 3)) / (imsize*1.0) * 2.0 - 1.0  # (2, 1)  ????  
    bw_map_tensor = torch.from_numpy(bw_map).type(torch.float32).to(device)
    output_tensor = F.grid_sample(input=image_tensor, grid=bw_map_tensor, align_corners=True)
    ###  torch -> np
    output = output_tensor.detach().cpu().numpy().transpose((0,2,3,1))
    return output

def bw_mapping_one(bw_map, image, device="cuda"):
    image = torch.unsqueeze(image, 0)  # [1, 3, 256, 256]
    image_t = image.transpose(2, 3)  # b c h w
    # bw
    # from [h, w, 2]
    # to  4D tensor [-1, 1] [b, h, w, 2]
    bw_map = torch.from_numpy(bw_map).type(torch.float32).to(device)  # numpy to tensor
    bw_map = torch.unsqueeze(bw_map, 0)
    # bw_map = bw_map.transpose(1, 2).transpose(2, 3)
    output = F.grid_sample(input=image, grid=bw_map, align_corners=True)
    output_t = F.grid_sample(input=image_t, grid=bw_map, align_corners=True)  # tensor
    output = output.transpose(1, 2).transpose(2, 3)
    output = output.squeeze()
    output_t = output_t.transpose(1, 2).transpose(2, 3)
    output_t = output_t.squeeze()
    return output_t  # transpose(1,2).transpose(0,1)
    # ensure output [c, h, w]

