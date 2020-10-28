# coding: utf-8
from scipy.interpolate import griddata
import numpy as np
from .data_process import reprocess_auto, reprocess_auto_batch
from tutils import *

def uv2bmap_in_tensor(uv_tensor, mask_tensor, imsize=256):
    # 要记得使用reprocess
    uv = reprocess_auto(uv_tensor, "uv")
    mask = reprocess_auto(mask_tensor, "background")
    assert uv.shape[2] == 2, "Error, Got {}".format(uv.shape)
    assert mask.shape[2] == 1, "Error, Got {}".format(mask.shape)
    bw = uv2backward_trans_3(uv, mask, imsize)
    return bw

def uv2backward_batch_with_reprocess(b_uv, b_mask, imsize=256):
    # 要记得使用reprocess
    assert type(b_uv) is np.ndarray, "Error, Got {}".format(type(b_uv))
    output = np.zeros_like(b_uv, dtype=np.float32)
    bs = b_uv.shape[0]
    for c in range(bs):
        uv = reprocess_auto(b_uv[c, :, :, :], "uv")
        mask = reprocess_auto(b_mask[c, :, :, :], "background")
        output[c,:,:,:] = uv2backward_trans_3(uv, mask, imsize)
    return output

def uv2backward_batch(b_uv, b_mask, imsize=256):
    # 要记得使用reprocess
    assert type(b_uv) is np.ndarray, "Error, Got {}".format(type(b_uv))
    output = np.zeros_like(b_uv, dtype=np.float32)
    bs = b_uv.shape[0]
    for c in range(bs):
        uv = b_uv[c, :, :, :]
        mask = b_mask[c, :, :, :]
        output[c,:,:,:] = uv2backward_trans_3(uv, mask, imsize)
    return output



def uv2backward_trans_2(uv, mask, imsize=256):
    w("There is some mistakes in uv2backward_trans_2, please use uv2backward_trans_3!")
    """
    Not use this one
    """
    ### Transform with "Linear and Nearest"
    ### Use with bw_mapping_2
    assert type(uv) is np.ndarray, "Error, Got {}".format(type(uv))
    assert type(mask) is np.ndarray, "Error, Got{}".format(type(mask))
    assert np.ndim(uv) == 3 , "ndim Error , GOt {}".format(np.ndim(uv))
    assert np.ndim(mask) == 3 , "ndim Error , GOt {}".format(np.ndim(mask))

    # -----------  Transpose the  (x, y) ------------
    img_rgb = uv # [:, :, ::-1]
    img_rgb[:,:,1] = 1-img_rgb[:,:,1]
    # img_rgb[:,:,0] = 1-img_rgb[:,:,0]
    
    s_x = (img_rgb[:,:,0]*imsize) # u
    s_y = (img_rgb[:,:,1]*imsize) # v
    mask = mask[:,:,0] > 0.6               # 这是什么 0.6 
    s_x = s_x[mask]
    s_y = s_y[mask]
    index = np.argwhere(mask)
    t_y = index[:, 0]  # t_y and t_x is a map
    t_x = index[:, 1]
    x = np.arange(imsize)
    y = np.arange(imsize)
    xi, yi = np.meshgrid(x, y)
    zx = griddata((s_x,s_y),t_x,(xi,yi),method='linear')   # griddata(points, values, point_grid, method="nearest/linear")
    zy = griddata((s_x,s_y),t_y,(xi,yi),method='linear') 

    zx_nearest = griddata((s_x,s_y),t_x,(xi,yi),method='nearest')   # griddata(points, values, point_grid, method="nearest/linear")
    zy_nearest = griddata((s_x,s_y),t_y,(xi,yi),method='nearest') 
    
    zx = np.where(np.isnan(zx), zx_nearest, zx)
    zy = np.where(np.isnan(zy), zy_nearest, zy)

    backward_img = np.stack([zy,zx],axis=2)
    
    # test_corners(backward_img)
    return backward_img

def uv2backward_trans_3(uv, mask, imsize=256):
    ### Transform with "Linear and Nearest"
    ### Use with bw_mapping_3
    assert type(uv) is np.ndarray, "Error, Got {}".format(type(uv))
    assert type(mask) is np.ndarray, "Error, Got{}".format(type(mask))
    assert np.ndim(uv) == 3 , "ndim Error , GOt {}".format(np.ndim(uv))
    assert np.ndim(mask) == 3 , "ndim Error , GOt {}".format(np.ndim(mask))

    # -----------  Transpose the  (x, y) ------------
    img_rgb = uv # [:, :, ::-1]
    # img_rgb[:,:,1] = 1-img_rgb[:,:,1]
    img_rgb[:,:,0] = 1-img_rgb[:,:,0]
    
    s_x = (img_rgb[:,:,0]*imsize) # u
    s_y = (img_rgb[:,:,1]*imsize) # v
    mask = mask[:,:,0] > 0.6               # 这是什么 0.6 
    # print("s_x.shape, ", s_x.shape)
    s_x = s_x[mask]
    s_y = s_y[mask]
    index = np.argwhere(mask)
    t_y = index[:, 0]  # t_y and t_x is a map
    t_x = index[:, 1]
    x = np.arange(imsize)
    y = np.arange(imsize)
    xi, yi = np.meshgrid(x, y)
    zx = griddata((s_x,s_y),t_x,(xi,yi),method='linear')   # griddata(points, values, point_grid, method="nearest/linear")
    zy = griddata((s_x,s_y),t_y,(xi,yi),method='linear') 

    zx_nearest = griddata((s_x,s_y),t_x,(xi,yi),method='nearest')   # griddata(points, values, point_grid, method="nearest/linear")
    zy_nearest = griddata((s_x,s_y),t_y,(xi,yi),method='nearest') 
    
    zx = np.where(np.isnan(zx), zx_nearest, zx)
    zy = np.where(np.isnan(zy), zy_nearest, zy)

    backward_img = np.stack([zy,zx],axis=2)
    
    # test_corners(backward_img)
    return backward_img

def uv2backward_trans(uv, mask, imsize=256):
    # print("uv.shape: ",uv.shape)
    # print("mask.shape: ", mask.shape)
    # print_uv(uv)
    img_rgb = uv
    img_rgb[:,:,1] = 1-img_rgb[:,:,1]
    s_x = (img_rgb[:,:,0]*imsize) # u
    s_y = (img_rgb[:,:,1]*imsize) # v
    mask = mask[:,:,0] > 0.6 
    s_x = s_x[mask]
    s_y = s_y[mask]
    index = np.argwhere(mask)
    t_y = index[:, 0]  # t_y and t_x is a map
    t_x = index[:, 1]
    x = np.arange(imsize)
    y = np.arange(imsize)
    xi, yi = np.meshgrid(x, y)
    zx = griddata((s_x,s_y),t_x,(xi,yi),method='linear')   # griddata(points, values, point_grid, method="nearest/linear")
    zy = griddata((s_x,s_y),t_y,(xi,yi),method='linear')
    backward_img = np.stack([zy,zx],axis=2)
    # print_bw(bw)
    return backward_img

if __name__ == "__main__":
    x = np.arange(5)
    y = np.arange(5)
    xi, yi = np.meshgrid(x, y)
    print(xi)