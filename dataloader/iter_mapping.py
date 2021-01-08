import numpy as np
from models.misc.deform_model import construct_plain_cmap
from .print_img import print_img_auto
import torch
import torch.nn.functional as F
from tutils import *

def iter_mapping(ori:torch.Tensor, iter_map:torch.Tensor, iter_num:int=7):
    assert type(ori) == torch.Tensor, "Error, but got {}".format(type(ori))
    assert ori.size(-1) == 256, "Error, but Got {}".format(ori.size())
    bs = ori.size(0)
    X, Y = np.meshgrid(np.linspace(-1,1,256), np.linspace(-1,1,256))
    base_grid = np.stack([X,Y], axis=-1)
    base_grid = torch.from_numpy(base_grid[np.newaxis, :,:,:]).cuda().float()
    base_grid = base_grid.expand((bs,256,256,2))
    iter_map = iter_map.permute(0,2,3,1)
    # import ipdb; ipdb.set_trace()
    grid = iter_map + base_grid
    grid = grid.expand(iter_map.size())
    # import ipdb; ipdb.set_trace()
        
    output_tensor = ori
    for _ in range(iter_num):
        output_tensor = F.grid_sample(input=ori, grid=grid, align_corners=False)
    
    return output_tensor
    ###  torch -> np
    # output = output_tensor.detach().cpu().numpy().transpose((0,2,3,1))


def test_iter_mapping():
    rgb, pad_rgb2 = construct_plain_cmap()
    pad_rgb = pad_rgb2.transpose(2,0,1)
    rgb_tensor = torch.from_numpy(pad_rgb[np.newaxis,:,:,:])
    X, Y = np.meshgrid(np.linspace(-1,1,256), np.linspace(-1,1,256))
    base_grid = np.stack([X,Y], axis=-1)
    base_grid = torch.from_numpy(base_grid[np.newaxis, :,:,:])
    base_grid = base_grid.expand((6,256,256,2))
    base_grid = torch.unsqueeze(base_grid[0,:,:,:], 0)
    output_tensor = F.grid_sample(input=rgb_tensor, grid=base_grid, align_corners=False)
    for _ in range(6):
        output_tensor = F.grid_sample(input=output_tensor, grid=base_grid, align_corners=False)
    # import ipdb; ipdb.set_trace()
    print_img_auto(pad_rgb2, "cmap", fname="./test_iter2.jpg")
    print_img_auto(output_tensor[0,:,:,:], "cmap", fname="./test_iter.jpg")
    # import ipdb; ipdb.set_trace()
    base_grid = base_grid[0,:,:,:].cpu().numpy()
    base_grid = (base_grid+1.0)/2.
    print_img_auto(base_grid, "uv", fname="test_iter3.jpg")
    
if __name__ == "__main__":
    test_iter_mapping()
    
    