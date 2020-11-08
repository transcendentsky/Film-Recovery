# coding: utf-8
"""
Name: train_with_constrain.py
Desc: This script contains the training and testing code of Unwrap Film.
Notations:
    ||rgb
        Mean: The original render image in color space.
        Shape: [batch_size, 3, 256, 256]
        Range: 0 ~ 255 -----> -1 ~ 1
    ||threeD_map :
        Mean: Three coordinate map as ground truth.
        Shape: [batch_size, 3, 256, 256]
        Range: 0.1 ~ 1 -----> -1 ~ 1
    ||nor_map:
        Mean: Normal map as ground truth.
        Shape: [batch_size, 3, 256, 256]
        Range: 0 ~ 1 -----> -1 ~ 1
    ||dep_map:
        Mean: Depth map as ground truth.
        Shape: [batch_size, 1, 256, 256]
        Range: 0.1 ~ 1 -----> -1 ~ 1
    ||uv_map:
        Mean: UV map as ground truth.
        Shape: [batch_size, 2, 256, 256]
        Range: 0 ~ 1 ------> -1 ~ 1
    ||mask_map:
        Mean: Background mask as ground truth.
        Shape: [batch_size, 1, 256, 256]
        Range: 0 or 1
    ||alb_map:
        Mean: Albedo map as ground truth.
        Shape: [batch_size, 1, 256, 256]
        Range: 0 ~ 255 -----> -1 ~ 1
    ||bcward_map:
        Mean: Backward map as ground truth.
        Shape: [batch_size, 2, 256, 256]
        Range: 0 ~ 1
    For any notations, we use original signature X as direct predict and X_gt as corresponding ground truth.
    X_from_source eg. dep_from3d   dep_from_nor means transfer
Constrain Path:
    1st path: 3D ---> Normal
              3D ---> Depth
              Normal ---> Depth
              Depth ---> Normal
    If need
    2nd path: 3D ---> Normal ---> Depth
              3D ---> Depth ----> Normal
"""
import torch
import torch.nn as nn
import models
import train_configs
from torch.utils.data import Dataset, DataLoader
from load_data_npy import filmDataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from cal_times import CallingCounter
import time
import os
# import metrics
import numpy as np
from write_image import *
from tensorboardX import SummaryWriter
import models2
import models3
from tutils import *
import load_data_npy
from dataloader.load_data_2 import filmDataset_old
from dataloader.uv2bw import uv2bmap_in_tensor
from dataloader.print_img import print_img_with_reprocess
from config import *
from dataloader.data_process import reprocess_auto_batch
from dataloader.uv2bw import uv2backward_batch
from evaluater.eval_ones import cal_PSNR, cal_MSE
from evaluater.eval_batches import uvbw_loss_np_batch
from dataloader.print_img import print_img_auto
from dataloader.bw_mapping import bw_mapping_batch_2, bw_mapping_batch_3



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "5, 0, 1, 2"
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return float(param_group['lr'])

# def train(args, model, device, train_loader, optimizer, criterion, epoch, writer, output_dir, isWriteImage, isVal=False,
#           test_loader=None):
#     return lr

def test2(*args):
    ### Cancel the test Func
    pass

@tfuncname
def test(args, model, test_loader, optimizer, criterion, \
    epoch, writer, output_dir, isWriteImage, isVal=False):
    model.eval()
    acc = 0
    device = "cuda"
    count = 0
    mse1, mse2, mse3, mse4 = 0, 0, 0, 0
    mae1, mae2, mae3, mae4 = 0, 0, 0, 0
    cc1, cc2, cc3, cc4 = 0, 0, 0, 0
    psnr1, psnr2, psnr3, psnr4 = 0, 0, 0, 0
    ssim1, ssim2, ssim3, ssim4 = 0, 0, 0, 0
    m1, m2, s1, s2 = 0,0,0,0

    for batch_idx, data in enumerate(test_loader):
        if batch_idx >= 100:
            break
        threeD_map_gt = data[0]
        uv_map_gt = data[1]
        bw_map_gt = data[2]
        mask_map_gt = data[3]
        ori_map_gt = data[4]

        uv_map_gt, threeD_map_gt, bw_map_gt, mask_map_gt =  uv_map_gt.to(device), threeD_map_gt.to(device), bw_map_gt.to(device), mask_map_gt.to(device)
        uv_pred_t, bw_pred_t= model(threeD_map_gt)
        uv_pred_t = torch.where(mask_map_gt>0, uv_pred_t, mask_map_gt)
        # loss_uv = criterion(uv_pred_t, uv_map_gt).float()
        # loss_bw = criterion(bw_pred_t, bw_map_gt).float()

        uv_np = reprocess_auto_batch(uv_pred_t, "uv")
        uv_gt_np = reprocess_auto_batch(uv_map_gt, "uv")
        bw_np = reprocess_auto_batch(bw_pred_t, "bw")
        bw_gt_np = reprocess_auto_batch(bw_map_gt, "bw")
        mask_np = reprocess_auto_batch(mask_map_gt, "background")
        bw_uv = uv2backward_batch(uv_np, mask_np)
        ori = reprocess_auto_batch(ori_map_gt, "ori")

        count += uv_np.shape[0]*1.0
        # ----------  MSE  -------------------
        # total_uv_bw_loss, total_bw_loss, total_ori_uv_loss, total_ori_bw_loss
        l1, l2, l3, l4 = uvbw_loss_np_batch(uv_np, bw_np, bw_gt_np, mask_np, ori, metrix="mse")
        mse1 += l1
        mse2 += l2
        mse3 += l3
        mse4 += l4
        writer.add_scalar('mse_one/uv_bw_loss' , l1,  global_step=batch_idx)
        writer.add_scalar('mse_one/bw_loss',     l2, global_step=batch_idx)
        writer.add_scalar('mse_one/ori_uv_loss', l3,  global_step=batch_idx)
        writer.add_scalar('mse_one/ori_bw_loss', l4, global_step=batch_idx)
        # --------------------------
        l1, l2, l3, l4 = uvbw_loss_np_batch(uv_np, bw_np, bw_gt_np, mask_np, ori, metrix="mae")
        mae1 += l1
        mae2 += l2
        mae3 += l3
        mae4 += l4
        writer.add_scalar('mae_one/uv_bw_loss' , l1,  global_step=batch_idx)
        writer.add_scalar('mae_one/bw_loss',     l2, global_step=batch_idx)
        writer.add_scalar('mae_one/ori_uv_loss', l3,  global_step=batch_idx)
        writer.add_scalar('mae_one/ori_bw_loss', l4, global_step=batch_idx)
        # ----------  CC  -------------------
        l1, l2, l3, l4 = uvbw_loss_np_batch(uv_np, bw_np, bw_gt_np, mask_np, ori, metrix="cc")
        cc1 += l1
        cc2 += l2
        cc3 += l3
        cc4 += l4
        writer.add_scalar('cc_one/uv_bw_loss' , l1,  global_step=batch_idx)
        writer.add_scalar('cc_one/bw_loss',     l2, global_step=batch_idx)
        writer.add_scalar('cc_one/ori_uv_loss', l3,  global_step=batch_idx)
        writer.add_scalar('cc_one/ori_bw_loss', l4, global_step=batch_idx)
        # ----------  PSNR  -------------------
        l1, l2, l3, l4 = uvbw_loss_np_batch(uv_np, bw_np, bw_gt_np, mask_np, ori, metrix="psnr")
        psnr1 += l1
        psnr2 += l2
        psnr3 += l3
        psnr4 += l4
        writer.add_scalar('psnr_one/uv_bw_loss' , l1,  global_step=batch_idx)
        writer.add_scalar('psnr_one/bw_loss',     l2, global_step=batch_idx)
        writer.add_scalar('psnr_one/ori_uv_loss', l3,  global_step=batch_idx)
        writer.add_scalar('psnr_one/ori_bw_loss', l4, global_step=batch_idx)
        # ----------  SSIM  -------------------
        # l1, l2, l3, l4 = uvbw_loss_np_batch(uv_np, bw_np, bw_gt_np, mask_np, ori, metrix="ssim")
        # ssim1 += l1
        # ssim2 += l2
        # ssim3 += l3
        # ssim4 += l4
        writer.add_scalar('ssim_one/uv_bw_loss' , l1,  global_step=batch_idx)
        writer.add_scalar('ssim_one/bw_loss',     l2, global_step=batch_idx)
        writer.add_scalar('ssim_one/ori_uv_loss', l3,  global_step=batch_idx)
        writer.add_scalar('ssim_one/ori_bw_loss', l4, global_step=batch_idx)
        print("Batch-idx {},  total_bw_loss {}".format(batch_idx, cc1))
        print("outputdir", output_dir)
        writer.add_scalar('mse/uv_bw_loss' , mse1/count,  global_step=batch_idx)
        writer.add_scalar('mse/bw_loss',     mse2/count, global_step=batch_idx)
        writer.add_scalar('mse/ori_uv_loss', mse3/count,  global_step=batch_idx)
        writer.add_scalar('mse/ori_bw_loss', mse4/count, global_step=batch_idx)
        writer.add_scalar('mae/uv_bw_loss' , mae1/count,  global_step=batch_idx)
        writer.add_scalar('mae/bw_loss',     mae2/count, global_step=batch_idx)
        writer.add_scalar('mae/ori_uv_loss', mae3/count,  global_step=batch_idx)
        writer.add_scalar('mae/ori_bw_loss', mae4/count, global_step=batch_idx)
        writer.add_scalar('cc/uv_bw_loss' , cc1/count,  global_step=batch_idx)
        writer.add_scalar('cc/bw_loss',     cc2/count, global_step=batch_idx)
        writer.add_scalar('cc/ori_uv_loss', cc3/count,  global_step=batch_idx)
        writer.add_scalar('cc/ori_bw_loss', cc4/count, global_step=batch_idx)
        writer.add_scalar('psnr/uv_bw_loss' , psnr1/count,  global_step=batch_idx)
        writer.add_scalar('psnr/bw_loss',     psnr2/count, global_step=batch_idx)
        writer.add_scalar('psnr/ori_uv_loss', psnr3/count,  global_step=batch_idx)
        writer.add_scalar('psnr/ori_bw_loss', psnr4/count, global_step=batch_idx)
        writer.add_scalar('ssim/uv_bw_loss' , ssim1/count,  global_step=batch_idx)
        writer.add_scalar('ssim/bw_loss',     ssim2/count, global_step=batch_idx)
        writer.add_scalar('ssim/ori_uv_loss', ssim3/count,  global_step=batch_idx)
        writer.add_scalar('ssim/ori_bw_loss', ssim4/count, global_step=batch_idx)
        
        # ------------  Write Imgs --------------------
        dewarp_ori_bw = bw_mapping_batch_3(ori, bw_np, device="cuda")
        dewarp_ori_uv = bw_mapping_batch_3(ori, bw_uv, device="cuda")
        dewarp_ori_gt = bw_mapping_batch_3(ori, bw_gt_np, device="cuda")
        fname_bw = tfilename(output_dir, "test_uvbw/batch_{}/ori_bw.jpg".format(batch_idx))
        fname_uv = tfilename(output_dir, "test_uvbw/batch_{}/ori_uv.jpg".format(batch_idx))
        fname_origt = tfilename(output_dir, "test_uvbw/batch_{}/ori_dewarp_gt.jpg".format(batch_idx))
        print_img_auto(dewarp_ori_bw[0,:,:,:], img_type="ori", is_gt=False, fname=fname_bw)
        print_img_auto(dewarp_ori_uv[0,:,:,:], img_type="ori", is_gt=False, fname=fname_uv)
        print_img_auto(dewarp_ori_gt[0,:,:,:], img_type="ori", is_gt=False, fname=fname_origt)
        print_img_auto(uv_np[0,:,:,:],    "uv", is_gt=False, fname=tfilename(output_dir, "test_uvbw/batch_{}/uv_pred.jpg".format(batch_idx)))
        print_img_auto(uv_gt_np[0,:,:,:], "uv", is_gt=False, fname=tfilename(output_dir, "test_uvbw/batch_{}/uv_gt.jpg".format(batch_idx)))
        print_img_auto(bw_np[0,:,:,:],    "bw", is_gt=False, fname=tfilename(output_dir, "test_uvbw/batch_{}/bw_pred.jpg".format(batch_idx)))
        print_img_auto(bw_gt_np[0,:,:,:], "bw", is_gt=False, fname=tfilename(output_dir, "test_uvbw/batch_{}/bw_gt.jpg".format(batch_idx)))
        print_img_auto(bw_uv[0,:,:,:],    "bw", is_gt=False, fname=tfilename(output_dir, "test_uvbw/batch_{}/bw_uv_pred.jpg".format(batch_idx)))
        print_img_auto(mask_np[0,:,:,:],  "background", is_gt=False, fname=tfilename(output_dir, "test_uvbw/batch_{}/bg_gt.jpg".format(batch_idx)))
        print_img_auto(ori[0,:,:,:],      "ori", is_gt=False, fname=tfilename(output_dir, "test_uvbw/batch_{}/ori_gt.jpg".format(batch_idx)))

        # Write bw diffs
        diff1 = bw_gt_np[0,:,:,:] - bw_np[0,:,:,:]
        diff2 = bw_gt_np[0,:,:,:] - bw_uv[0,:,:,:]
        max1 = np.max(diff1)
        max2 = np.max(diff2)
        min1 = np.min(diff1)
        min2 = np.min(diff2)
        max_both = np.max([max1, max2])
        min_both = np.min([min1, min2])
        diff_p1 = (diff1 - min_both) / (max_both - min_both) * 255
        diff_p2 = (diff2 - min_both) / (max_both - min_both) * 255
        mean1_1 = np.average(diff1[0,:,:,0])
        mean1_2 = np.average(diff1[0,:,:,1])
        mean2_1 = np.average(diff2[0,:,:,0])
        mean2_2 = np.average(diff2[0,:,:,1])
        # mean2 = np.average(diff2)
        std1 = np.std(diff1)
        std2 = np.std(diff2)
        m1_1 += np.abs(mean1_1)
        m1_2 += np.abs(mean1_2)
        m2_1 += np.abs(mean2_1)
        m2_2 += np.abs(mean2_2)
        s1 += std1
        s2 += std2

        diff1[0,:,:,0] = diff1[0,:,:,0] - mean1_1
        diff1[0,:,:,1] = diff1[0,:,:,1] - mean1_2
        diff2[0,:,:,0] = diff2[0,:,:,0] - mean2_1
        diff2[0,:,:,1] = diff2[0,:,:,1] - mean2_2
        writer.add_scalar("bw_all_single/mean1_1", mean1_1, global_step=batch_idx)
        writer.add_scalar("bw_all_single/mean1_2", mean1_2, global_step=batch_idx)
        writer.add_scalar("bw_all_single/mean2_1", mean2_1, global_step=batch_idx)
        writer.add_scalar("bw_all_single/mean2_2", mean2_2, global_step=batch_idx)
        writer.add_scalar("bw_all_single/std_1", std1, global_step=batch_idx)
        writer.add_scalar("bw_all_single/std_2", std2, global_step=batch_idx)
        writer.add_scalar("bw_all_total/m1_1", m1_1, global_step=batch_idx)
        writer.add_scalar("bw_all_total/m1_2", m1_2, global_step=batch_idx)
        writer.add_scalar("bw_all_total/m2_1", m2_1, global_step=batch_idx)
        writer.add_scalar("bw_all_total/m2_2", m2_2, global_step=batch_idx)
        writer.add_scalar("bw_all_total/mean2", m2, global_step=batch_idx)
        writer.add_scalar("bw_all_total/std_1", s1, global_step=batch_idx)
        writer.add_scalar("bw_all_total/std_2", s2, global_step=batch_idx)
        
        print_img_auto(diff_p1, "bw", fname=tfilename(output_dir, "test_uvbw/batch_{}/diff_bw.jpg".format(batch_idx)))
        print_img_auto(diff_p2, "bw", fname=tfilename(output_dir, "test_uvbw/batch_{}/diff_bwuv.jpg".format(batch_idx)))
        print_img_auto(diff_m1, "bw", fname=tfilename(output_dir, "test_uvbw/batch_{}/diff_m_bw.jpg".format(batch_idx)))
        print_img_auto(diff_m2, "bw", fname=tfilename(output_dir, "test_uvbw/batch_{}/diff_m_bwuv.jpg".format(batch_idx)))
        # Write diffs Ori
        # diff1 = np.abs(dewarp_ori_bw[0,:,:,:] - dewarp_ori_gt[0,:,:,:])
        # diff2 = np.abs(dewarp_ori_uv[0,:,:,:] - dewarp_ori_gt[0,:,:,:])
        # max1 = np.max(diff1)
        # max2 = np.max(diff2)
        # max_both = np.max([max1, max2])
        # diff1 = diff1 / max_both * 255
        # diff2 = diff2 / max_both * 255
        # mean1 = np.average(diff1)
        # mean2 = np.average(diff2)
        # std1 = np.std(diff2)
        # std2 = np.std(diff2)

        # diff_m1 = diff1 - mean1
        # diff_m2 = diff2 - mean2

        # print_img_auto(diff1, "bw", fname=tfilename(output_dir, "test_uvbw/batch_{}/diff_bw.jpg".format(batch_idx)))
        # print_img_auto(diff2, "bw", fname=tfilename(output_dir, "test_uvbw/batch_{}/diff_bwuv.jpg".format(batch_idx)))
        # print_img_auto(diff_m1, "bw", fname=tfilename(output_dir, "test_uvbw/batch_{}/diff_m_bw.jpg".format(batch_idx)))
        # print_img_auto(diff_m2, "bw", fname=tfilename(output_dir, "test_uvbw/batch_{}/diff_m_bwuv.jpg".format(batch_idx)))
                
        # print_img_auto(diffo1, "ori", fname=)
        # print_img_auto(bw_gt_np[0,:,:,:], "deform", fname=tfilename(output_dir, "test_uvbw/batch_{}/deform_gt.jpg".format(batch_idx)))
        # print_img_auto(bw_np[0,:,:,:], "deform", fname=tfilename(output_dir, "test_uvbw/batch_{}/deform_bw.jpg".format(batch_idx)))
        # print_img_auto(bw_uv[0,:,:,:], "deform", fname=tfilename(output_dir, "test_uvbw/batch_{}/deform_bwuv.jpg".format(batch_idx)))
        
    # print("Loss:")
    # print("mse: {} {}".format(mse_bw/count, mse_ori/count))
    # print("cc: {} {}".format(cc_bw/count, cc_ori/count))
    # print("psnr: {} {}".format(psnr_bw/count, psnr_ori/count))
    # print("ssim: {} {}".format(ssim_bw/count, ssim_ori/count))

    return 


def main():
    # Model Build
    model = models3.UnwarpNet()
    model2 = models3.Conf_Discriminator()
    
    args = train_configs.args
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        print(" [*] Set cuda: True")
        model = torch.nn.DataParallel(model.cuda())
        model2 = torch.nn.DataParallel(model2.cuda())
    else:
        print(" [*] Set cuda: False")
    # Load Dataset
    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
    dataset_test = filmDataset_old(npy_dir=args.test_path, load_mod='test_uvbw_mapping')
    dataset_test_loader = DataLoader(dataset_test,
                                     batch_size=1,
                                     shuffle=True,
                                     **kwargs)
    if UVBW_TRAIN:
        dataset_train = filmDataset_old(npy_dir=args.train_path, load_mod='uvbw')
        dataset_train_loader = DataLoader(dataset_train, batch_size=args.batch_size,
                                        shuffle=True, **kwargs)
    start_epoch = 1
    learning_rate = args.lr
    # Load Parameters
    #if args.pretrained:
    if DEFORM_TEST:
        # pre_model = "/home1/quanquan/film_code/test_output2/20201018-070501z0alvFmodels/uvbw/tv_constrain_35.pkl"
        pre_model = "/home1/quanquan/film_code/test_output2/20201021-094607K3qzNUmodels/uvbw/tv_constrain_69.pkl"
        pretrained_dict = torch.load(pre_model, map_location=None)
        model.load_state_dict(pretrained_dict['model_state'])
        start_lr = pretrained_dict['lr']
        start_epoch = pretrained_dict['epoch']
        print("Start_lr: {} ,  Start_epoch {}".format(start_lr, start_epoch))
    # Add Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Output dir setting
    output_dir = tdir(args.output_dir, generate_name())
    print("Saving Dir: ", output_dir)

    if args.use_mse:
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.L1Loss()
    if args.write_summary:
        writer_dir = tdir(output_dir, 'summary/' + args.model_name + '_start_epoch{}'.format(start_epoch))
        print("Using TensorboardX !")
        writer = SummaryWriter(logdir=writer_dir)
        # print(args.model_name)
    else:
        writer = 0
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    #start_lr = args.lr
    for epoch in range(start_epoch, args.epochs + 1):
        if UVBW_TRAIN:
            
            model.train()
            correct = 0
            for batch_idx, data in enumerate(dataset_train_loader):
                ori_map_gt = data[0].to(device)
                ab_map_gt = data[1].to(device)
                depth_map_gt = data[2].to(device)
                normal_map_gt = data[3].to(device)
                cmap_gt = data[4].to(device)
                uv_map_gt = data[5].to(device)
                df_map_gt = data[6].to(device)
                bg_map_gt = data[7].to(device)

                optimizer.zero_grad()
                uv_pred, cmap_pred, nor_pred, alb_pred, dep_pred, mask_map, \
                    nor_from_threeD, dep_from_threeD, nor_from_dep, dep_from_nor, df_map = model(ori_map_gt)

                # TODO: 这里需不需要改成这样
                uv = torch.where(mask_map>0.5, uv_pred, 0)
                loss_uv = criterion(uv_pred, uv_map_gt).float()
                loss_uv.backward()
                optimizer.step()
                lr = get_lr(optimizer)

                # Start using Confident 
                if epoch > 30:
                    loss_conf, loss_total = model2(dewarp_ori_pred, ori_map_gt)

                if batch_idx % args.log_intervals == 0:
                    print('\r Epoch:{}  batch index:{}/{}||lr:{:.8f}||loss_uv:{:.6f}||loss_bw:{:.6f}'.format(epoch, batch_idx + 1,
                                                                                len(dataset_train_loader.dataset) // args.batch_size,
                                                                                lr, loss_uv.item(), loss_bw.item()), end=" ")
                    if args.write_summary:
                        writer.add_scalar('summary/train_uv_loss', loss_uv.item(),
                                        global_step=epoch * len(dataset_train_loader) + batch_idx + 1)
                        # writer.add_scalar('summary/backward_loss', loss_bw.item(),
                        #                   global_step=epoch * len(train_loader) + batch_idx + 1)
                        writer.add_scalar('summary/lrate', lr, global_step=epoch * len(dataset_train_loader) + batch_idx + 1)
                if True: # Draw Image
                    if batch_idx == (len(dataset_train_loader.dataset) // args.batch_size)-1:
                        print('writing image')
                        for k in range(2):
                            uv_pred = uv_map[k, :, :, :]
                            uv_gt = uv_map_gt[k, :, :, :]
                            mask_gt = mask_map_gt[k, :, :, :]
                            bw_gt = df_map_gt[k, :, :, :]
                            bw_pred = bw_map[k, :, :, :]
                            cmap_gt = threeD_map_gt[k, :, :, :]

                            output_dir1 = tdir(output_dir + '/uvbw_train/', 'epoch_{}_batch_{}/'.format(epoch, batch_idx))
                            """pred"""
                            print_img_with_reprocess(uv_pred, img_type="uv", fname=tfilename(output_dir1 + 'train/epoch_{}/pred_uv_ind_{}'.format(epoch, k) + '.jpg'))
                            # print_img_with_reprocess(bw_from_uv, img_type="bw", fname=tfilename(output_dir1 + 'train/epoch_{}/bw_f_uv_ind_{}'.format(epoch, k) + '.jpg'))
                            print_img_with_reprocess(bw_pred, img_type="bw", fname=tfilename(output_dir1 + 'train/epoch_{}/pred_bw_ind_{}'.format(epoch, k) + '.jpg'))
                            """gt"""
                            print_img_with_reprocess(cmap_gt, img_type="cmap", fname=tfilename(output_dir1 + 'train/epoch_{}/gt_3D_ind_{}'.format(epoch, k) + '.jpg')) # Problem
                            print_img_with_reprocess(uv_gt, img_type="uv", fname=tfilename(output_dir1 + 'train/epoch_{}/gt_uv_ind_{}'.format(epoch, k) + '.jpg'))
                            print_img_with_reprocess(bw_gt, img_type="bw", fname=tfilename(output_dir1 + 'train/epoch_{}/gt_bw_ind_{}'.format(epoch, k) + '.jpg'))   # Problem
                            print_img_with_reprocess(mask_gt, img_type="background", fname=tfilename(output_dir1 + 'train/epoch_{}/gt_back_ind_{}'.format(epoch, k) + '.jpg'))
            

        else:
            test(args, model, dataset_test_loader, optimizer, criterion, epoch, \
                writer, output_dir, args.write_image_test)
            print("#"*22)
            break

        scheduler.step()
        if UVBW_TRAIN and args.save_model:
            state = {'epoch': epoch + 1,
                     'lr': lr,
                     'model_state': model.state_dict(),
                     'optimizer_state': optimizer.state_dict()
                     }
            torch.save(state, tfilename(output_dir, "models/uvbw/{}_{}.pkl".format(args.model_name, epoch)))

def exist_or_make(path):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == '__main__':
    main()
