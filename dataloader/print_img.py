import numpy as np
from .data_process import reprocess_auto
from tutils import *
import cv2
import torch

def print_img_with_reprocess(img, img_type, fname=None):
    #print("Printing Imgs with Reprocess")
    if type(img) is torch.Tensor:
        img = img.transpose(0,1).transpose(1,2)
        img = img.detach().cpu().numpy()
    assert np.ndim(img) <=3 
    img = reprocess_auto(img, img_type=img_type)
    print_img_np(img, img_type, fname=fname)


def print_img_auto(img, img_type, is_gt=True, fname=None):
    # print("[Warning] Pause to use print img, "); return
    if type(img) is torch.Tensor:
        print_img_tensor(img, img_type, is_gt, fname)
    elif type(img) is np.ndarray:
        print_img_np(img, img_type, is_gt, fname)
    else:
        raise TypeError("Wrong type: Got {}".format(type(img)))

def print_img_tensor(img_tensor, img_type, is_gt=True, fname=None):
    #print("[Printing] ", img_type)
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
    #assert np.ndim(img) == 3, "np.ndim Error, Got shape{}".format(img.shape)

    if img_type == "uv":
        print_uv(img, is_gt=is_gt, fname=fname)
    elif img_type == "bw":
        print_bw(img, is_gt, fname=fname)
    elif img_type in ["background", "bg"]:
        print_back(img, is_gt, fname=fname)
    elif img_type == "exr":
        print_cmap(img, is_gt, fname=fname)
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
    elif img_type == "deform":
        print_deform(img, is_gt, fname=fname)
    ### -------  Add Extra print func
    elif img_type == "bw_uv":
        print_bw_uv(img, is_gt, fname=fname)
    elif img_type == "deform_bw":
        print_deform_from_bw(img, is_gt, fname=fname)
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
    w,h,c = bw.shape
    bw = bw.astype(np.uint8)
    bb = np.zeros((w,h,1))
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
    w,h,c = uv.shape
    uv = uv*255
    uv = uv.astype(np.uint8)
    bb = np.zeros((w,h,1))
    uv = np.concatenate([uv, bb], axis=2)
    fname = tfilename(fname) if fname is not None else tfilename("imgshow",epoch_text,"uv_"+subtitle+"/uv_"+generate_name()+".jpg")
    #d("print_uv func: ")
    #print(np.sum(uv[:,:,2]))
    cv2.imwrite(fname, uv)

def print_back(background, is_gt, epoch=0, fname=None): # [0,1]
    subtitle = "gt" if is_gt else "pred"
    epoch_text = "epoch{}".format(epoch)
    background = background * 255
    background = background.astype(np.uint8)
    fname = tfilename(fname) if fname is not None else tfilename("imgshow",epoch_text,"bg_"+subtitle+"/bg_"+generate_name()+".jpg")
    cv2.imwrite(fname, background)

def print_deform(df, is_gt, epoch=0, fname=None):
    subtitle = "gt" if is_gt else "pred"
    epoch_text = "epoch{}".format(epoch)
    df = df / 2.0 + 255. / 2.
    df = df.astype(np.uint8)
    w,h,c = df.shape    
    bb = np.zeros((w,h,1))
    df = np.concatenate([df, bb], axis=2)
    fname = tfilename(fname) if fname is not None else tfilename("imgshow",epoch_text,"df_"+subtitle+"/df_"+generate_name()+".jpg")
    cv2.imwrite(fname, df)
    ###  For test
    cv2.imwrite(tfilename("df_test/c1.jpg"), df[:,:,0])
    cv2.imwrite(tfilename("df_test/c2.jpg"), df[:,:,1])

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

def print_deform_from_bw(bw, is_gt, epoch=0, fname=None):
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
    w,h,c = bw.shape
    bb = np.zeros((w,h,1))
    bw = np.concatenate([bw, bb], axis=2)
    fname = tfilename(fname) if fname is not None else tfilename("imgshow",epoch_text,"bw_"+subtitle+"/bw_"+generate_name()+".jpg")
    cv2.imwrite(fname, bw)
    ###  For test
    # cv2.imwrite(tfilename("bw_test/c1.jpg"), bw[:,:,0])
    # cv2.imwrite(tfilename("bw_test/c2.jpg"), bw[:,:,1])

def print_large_interpolation(uv, mask, fname):
    uv_size = uv.shape[0]
    expand_size = uv_size * 10

    img_rgb = uv # [:, :, ::-1]
    # img_rgb[:,:,1] = 1-img_rgb[:,:,1]
    img_rgb[:,:,0] = 1-img_rgb[:,:,0]
    
    s_x = (img_rgb[:,:,0]*expand_size) # u
    s_y = (img_rgb[:,:,1]*expand_size) # v
    mask = mask[:,:,0] > 0.6   

    img_rgb = np.round(img_rgb)
    s_x = s_x[mask]
    s_y = s_y[mask]
    index = np.argwhere(mask)
    t_y = index[:, 0]  # t_y and t_x is a map
    t_x = index[:, 1]
    # x = np.arange(expand_size)
    # y = np.arange(expand_size)
    # xi, yi = np.meshgrid(x, y)
    mesh = np.zeros((expand_size, expand_size))


def test_new_img(img_path, output_name):
    # cv2.imread()
    uv = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)        #[0,1]
    uv = uv[:,:,1:]
    
    print_img_auto(uv, "uv", fname=output_name)
    

if __name__ == "__main__":
    import os
    from tutils import tfilename
    dirname = "/home1/quanquan/datasets/generate/mesh_film_small/uv"
    
    for x in os.scandir(dirname):
        if x.name.endswith("exr"):
            test_new_img(x.path, tfilename("test_img_old", x.name[:-4]+".jpg"))
            print(x.name)
        