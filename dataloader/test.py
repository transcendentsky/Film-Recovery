from .load_data_2 import filmDataset_old
from .data_process import reprocess_auto, reprocess_auto_batch
from .print_img import print_img_np, print_img_auto, print_img_with_reprocess
from evaluater.eval_batches import uvbw_loss_np_batch, uvbw_loss_tensor_batch, tensor2np, np2tensor
import numpy as np
from .bw_mapping import bw_mapping_batch_2, bw_mapping_batch_3
from tutils import *
from .uv2bw import uv2backward_trans_3, uv2backward_batch_with_reprocess, uv2backward_batch


@tfuncname
def test1():  # ok
    data_path = '/home1/qiyuanwang/film_generate/npy/'
    dataset = filmDataset_old(data_path, load_mod="all")
    data = dataset.__getitem__(0)
    ori = data[0].detach().cpu().numpy().transpose((1,2,0))
    ab = data[1].cpu().numpy().transpose((1,2,0))
    depth = data[2].cpu().numpy().transpose((1,2,0))
    normal = data[3].cpu().numpy().transpose((1,2,0))
    cmap = data[4].cpu().numpy().transpose((1,2,0))
    uv = data[5].cpu().numpy().transpose((1,2,0))
    background = data[6].cpu().numpy().transpose((1,2,0))

    print("=====  Reprocess  =====")
    ori = reprocess_auto(ori, img_type="ori")
    ab = reprocess_auto(ab, img_type="ab")
    depth = reprocess_auto(depth, img_type="depth")
    normal = reprocess_auto(normal, img_type="normal")
    cmap = reprocess_auto(cmap, img_type="cmap")
    uv = reprocess_auto(uv, img_type="uv")
    background = reprocess_auto(background, img_type="background")  # Ok

    print("[Printing] print imgs: ")
    print_img_auto(ori, img_type="ori")
    print_img_auto(ab, img_type="ab")
    print_img_auto(depth, img_type="depth")
    print_img_auto(normal, img_type="normal")
    print_img_auto(cmap, img_type="cmap")
    print_img_auto(uv, img_type="uv")
    print_img_auto(background, img_type="background")

@tfuncname
def test2():  # ok
    data_path = '/home1/qiyuanwang/film_generate/npy/'
    dataset = filmDataset_old(data_path, load_mod="uvbw")
    data = dataset.__getitem__(0)
    cmap = data[0]
    uv = data[1]
    bw = data[2]
    background = data[3]
    print("test2 ", bw.shape)
    print_img_with_reprocess(cmap, img_type="cmap")
    print_img_with_reprocess(uv, img_type="uv")
    print_img_with_reprocess(bw, img_type="bw")
    print_img_with_reprocess(background, img_type="background")

@tfuncname
def test3():
    from torch.utils.data import Dataset, DataLoader
    data_path = '/home1/qiyuanwang/film_generate/npy/'
    dataset = filmDataset_old(data_path, load_mod="test_uvbw_mapping")

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for batch_idx, data in enumerate(loader):
        threeD_map_gt = data[0]
        uv = data[1]
        bw = data[2]
        mask = data[3]
        ori = data[4]

        uv = reprocess_auto_batch(uv, "uv")
        mask = reprocess_auto_batch(mask, "background")
        ori = reprocess_auto_batch(ori, "ori")
        bw = reprocess_auto_batch(bw, "bw")
        # bw22 = reprocess_auto(bw[0,:,:,:], "bw")
        # bw11 = bw[0,:,:,:]
        # print("bw diff", np.sum(bw11-bw22))

        # print("Start calc loss")
        bw_loss, total_bw_loss, ori_loss, total_ori_loss = uvbw_loss_np_batch(uv, bw, mask, ori, metrix="mse")
        print("Loss: {} {} {} {}".format(bw_loss, total_bw_loss, ori_loss, total_ori_loss))

        print("Second Stage -----------------------------")
        bw_uv = uv2backward_batch(uv, mask)
        
        dewarp_ori_bw = bw_mapping_batch_3(ori, bw, device="cuda")
        dewarp_ori_uv = bw_mapping_batch_3(ori, bw_uv, device="cuda")

        # loss, total_loss = cal_metrix_np_batch(dewarp_ori_bw, dewarp_ori_uv, "psnr")
        print("ori loss: ", loss, total_loss)

        # print(dewarp_ori_bw)
        print(dewarp_ori_bw.shape)
        
        print_img_auto(dewarp_ori_bw[0,:,:,:], img_type="ori", fname="test/5/test_ori/ori_bw.jpg")
        print_img_auto(dewarp_ori_uv[0,:,:,:], img_type="ori", fname="test/5/test_ori/ori_uv.jpg")

        break

@tfuncname
def test4():
    data_path = '/home1/qiyuanwang/film_generate/npy/'
    dataset = filmDataset_old(data_path, load_mod="test_uvbw_mapping")
    data = dataset.__getitem__(0)
    # cmap = data[0]
    # uv = data[1]
    bw = data[2]
    # background = data[3]
    ori = data[4]
    print("Start calc")

    bw = reprocess_auto(bw, "bw")
    ori = reprocess_auto(ori, "ori")
    # print(ori)
    print_img_auto(ori, img_type="ori")
    print_img_auto(bw, img_type="bw")

    bw = bw[np.newaxis, :, :, :]
    ori = ori[np.newaxis, :, :, :]
    # print(bw)
    dewarp_ori = bw_mapping_batch_3(ori, bw, device="cuda")
    print("dewarp.shape", dewarp_ori.shape)
    print_img_auto(dewarp_ori[0, :, :, :], "ori")

@tfuncname
def test5():
    data_path = '/home1/qiyuanwang/film_generate/npy/'
    dataset = filmDataset_old(data_path, load_mod="test_uvbw_mapping")
    data = dataset.__getitem__(0)
    # cmap = data[0]
    # uv = data[1]
    bw = data[2]
    # mask = data[3]
    ori = data[4]

    # uv_true = reprocess_auto(uv, "uv")
    # mask_true = reprocess_auto(mask, "background")
    ori = reprocess_auto(ori, "ori")

    # bw_3 = uv2backward_trans_3(uv_true, mask_true)
    bw_2 = reprocess_auto(bw, "bw")

    # ------- bw 2 -> 3

    # print(np.sum(bw_3 - bw_2))
    # print_img_auto(bw_2, "bw", fname="bw_test/old.jpg")
    # print_img_auto(bw_3, "bw", fname="bw_test/new.jpg")
    
    dewarp_ori = bw_mapping_batch_3(ori[np.newaxis, :, :, :], bw_2[np.newaxis, :, :, :], device="cuda")
    print("dewarp.shape", dewarp_ori.shape)
    print_img_auto(dewarp_ori[0, :, :, :], "ori", fname="bw_test/ori_3.jpg")
    
    # dewarp_ori = bw_mapping_batch_3(ori[np.newaxis, :, :, :], bw_3[np.newaxis, :, :, :], device="cuda")
    # print("dewarp.shape", dewarp_ori.shape)
    # print_img_auto(dewarp_ori[0, :, :, :], "ori", fname="bw_test/ori_3.jpg")

if __name__ == "__main__":
    # test2()    
    # test1()
    # test5()
    # test4()
    test3()