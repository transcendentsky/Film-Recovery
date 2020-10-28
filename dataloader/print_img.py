import numpy as np
from .data_process import reprocess_auto
from tutils import *
import cv2
import torch

def print_img_with_reprocess(img, img_type, fname=None):
    print("Printing Imgs with Reprocess")
    if type(img) is torch.Tensor:
        img = img.transpose(0,1).transpose(1,2)
        img = img.detach().cpu().numpy()
    img = reprocess_auto(img, img_type=img_type)
    print_img_np(img, img_type, fname=fname)


def print_img_auto(img, img_type, is_gt=True, fname=None):
    if type(img) is torch.Tensor:
        print_img_tensor(img, img_type, is_gt, fname)
    elif type(img) is np.ndarray:
        print_img_np(img, img_type, is_gt, fname)
    else:
        raise TypeError("Wrong type: Got {}".format(type(img)))

def print_img_tensor(img_tensor, img_type, is_gt=True, fname=None):
    print("[Printing] ", img_type)
    img_tensor = img_tensor.transpose(0,1).transpose(1,2)
    img_np = img_tensor.detach().cpu().numpy()
    print_img_np(img_np, img_type)

def print_img_np(img, img_type, is_gt=True, fname=None):
    """
    Imgs:
        depth, normal, cmap
        uv, bw, background
        ori, ab
    """
    assert np.ndim(img) == 3, "np.ndim Error, Got shape{}".format(img.shape)

    if img_type == "uv":
        print_uv(img, is_gt=is_gt, fname=fname)
    elif img_type == "bw":
        print_bw(img, is_gt, fname=fname)
    elif img_type == "background":
        print_back(img, is_gt, fname=fname)
    elif img_type == "cmap":
        print_cmap(img, is_gt, fname=fname)
    elif img_type == "normal":
        print_normal(img, is_gt, fname=fname)
    elif img_type == "depth":
        print_depth(img, is_gt, fname=fname)
    elif img_type == "ori":
        print_ori(img, is_gt, fname=fname)
    elif img_type == "ab":
        print_ab(img, is_gt, fname=fname)
    ### -------  Add Extra print func
    elif img_type == "bw_uv":
        print_bw_uv(img, is_gt, fname=fname)
    elif img_type == "deform":
        print_deform(img, is_gt, fname=fname)
    else:
        raise TypeError("print_img Error!!!")



# ---------------------  Print IMGS  ------------------------
# -----------------------------------------------------------------------------
def print_cmap(cmap, is_gt, epoch=0, fname=None):  # [0,1]
    subtitle = "gt" if is_gt else "pred"
    epoch_text = "epoch{}".format(epoch)
    cmap = cmap * 255
    cmap = cmap.astype(np.uint8)
    fname = tfilename(fname) if fname is not None else tfilename("imgshow",epoch_text,"cmap_"+subtitle+"/cmap_"+generate_name()+".jpg")
    cv2.imwrite(fname, cmap)

def print_normal(normal, is_gt, epoch=0, fname=None): # [0,1]
    subtitle = "gt" if is_gt else "pred"
    epoch_text = "epoch{}".format(epoch)
    normal = normal * 255
    normal = normal.astype(np.uint8)
    fname = tfilename(fname) if fname is not None else tfilename("imgshow",epoch_text,"normal_"+subtitle+"/nor_"+generate_name()+".jpg")
    cv2.imwrite(fname, normal)

def print_depth(depth, is_gt, epoch=0, fname=None): # [0,1]
    subtitle = "gt" if is_gt else "pred"
    epoch_text = "epoch{}".format(epoch)
    depth = depth * 255
    depth = depth.astype(np.uint8)
    fname = tfilename(fname) if fname is not None else tfilename("imgshow",epoch_text,"depth_"+subtitle+"/depth_"+generate_name()+".jpg")
    cv2.imwrite(fname, depth)

# -----------------------------------------------------------------------------  
# -----------------------------------------------------------------------------
def print_bw(bw, is_gt, epoch=0, fname=None): 
    subtitle = "gt" if is_gt else "pred"
    epoch_text = "epoch{}".format(epoch)
    assert np.max(bw) > 1, "Value Error??? all vallues <= 1"
    assert np.ndim(bw) == 3, "np.ndim Error"
    # print("print_bw: bw.shape", bw.shape)
    bw = bw.astype(np.uint8)
    bb = np.zeros((256,256,1))
    bw = np.concatenate([bw, bb], axis=2)
    fname = tfilename(fname) if fname is not None else tfilename("imgshow",epoch_text,"bw_"+subtitle+"/bw_"+generate_name()+".jpg")
    cv2.imwrite(fname, bw)
    ###  For test
    # cv2.imwrite(tfilename("bw_test/c1.jpg"), bw[:,:,0])
    # cv2.imwrite(tfilename("bw_test/c2.jpg"), bw[:,:,1])

def print_bw_uv(bw, is_gt, epoch, fname=None):
    subtitle = "gt" if is_gt else "pred"
    epoch_text = "epoch{}".format(epoch)
    fname = tfilename(fname) if fname is not None else tfilename("imgshow",epoch_text,"bw_uv_"+subtitle+"/bw_uv_"+generate_name()+".jpg")
    print_bw(bw, is_gt, epoch, fname=fname)


def print_uv(uv, is_gt, epoch=0, fname=None): # [0,1]
    subtitle = "gt" if is_gt else "pred"
    epoch_text = "epoch{}".format(epoch)
    uv = uv*255
    uv = uv.astype(np.uint8)
    bb = np.zeros((256,256,1))
    uv = np.concatenate([uv, bb], axis=2)
    fname = tfilename(fname) if fname is not None else tfilename("imgshow",epoch_text,"uv_"+subtitle+"/uv_"+generate_name()+".jpg")
    cv2.imwrite(fname, uv)

def print_back(background, is_gt, epoch=0, fname=None): # [0,1]
    subtitle = "gt" if is_gt else "pred"
    epoch_text = "epoch{}".format(epoch)
    background = background * 255
    background = background.astype(np.uint8)
    fname = tfilename(fname) if fname is not None else tfilename("imgshow",epoch_text,"bg_"+subtitle+"/bg_"+generate_name()+".jpg")
    cv2.imwrite(fname, background)
# -----------------------------------------------------------------------------

def print_ori(ori, is_gt, epoch=0, fname=None):
    subtitle = "gt" if is_gt else "pred"
    epoch_text = "epoch{}".format(epoch)
    ori = ori.astype(np.uint8)
    fname = tfilename(fname) if fname is not None else tfilename("imgshow",epoch_text,"ori_"+subtitle+"/ori_"+generate_name()+".jpg")
    cv2.imwrite(fname, ori)

def print_ori_dewarp(ori, is_gt, epoch=0, fname=None):
    subtitle = "gt" if is_gt else "pred"
    epoch_text = "epoch{}".format(epoch)
    fname = tfilename(fname) if fname is not None else tfilename("imgshow",epoch_text,"ori_dewarp"+subtitle+"/bw_uv_"+generate_name()+".jpg")
    print_ori(ori, is_gt, epoch=epoch, fname=fname)

def print_ab(ab, is_gt, epoch=0, fname=None):
    subtitle = "gt" if is_gt else "pred"
    epoch_text = "epoch{}".format(epoch)
    ab = ab.astype(np.uint8)
    fname = tfilename(fname) if fname is not None else tfilename("imgshow",epoch_text,"ab_"+subtitle+"/ab_"+generate_name()+".jpg")
    cv2.imwrite(fname, ab)

def print_deform(bw, is_gt, epoch=0, fname=None):
    imsize = bw.shape[1]
    x = np.arange(imsize)
    y = np.arange(imsize)
    xi, yi = np.meshgrid(x, y)
    bw[:,:,0] = bw[:,:,0] - xi
    bw[:,:,1] = bw[:,:,1] - yi
    subtitle = "gt" if is_gt else "pred"
    epoch_text = "epoch{}".format(epoch)
    assert np.max(bw) > 1, "Value Error??? all vallues <= 1"
    assert np.ndim(bw) == 3, "np.ndim Error"
    # print("print_bw: bw.shape", bw.shape)
    bw = bw.astype(np.uint8)
    bb = np.zeros((256,256,1))
    bw = np.concatenate([bw, bb], axis=2)
    fname = tfilename(fname) if fname is not None else tfilename("imgshow",epoch_text,"bw_"+subtitle+"/bw_"+generate_name()+".jpg")
    cv2.imwrite(fname, bw)
    ###  For test
    # cv2.imwrite(tfilename("bw_test/c1.jpg"), bw[:,:,0])
    # cv2.imwrite(tfilename("bw_test/c2.jpg"), bw[:,:,1])