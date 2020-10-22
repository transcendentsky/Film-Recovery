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
import metrics
import numpy as np
from write_image import *
from tensorboardX import SummaryWriter
from tv_losses import TVLoss
import math
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5,2,3"


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return float(param_group['lr'])


def train(args, model, device, train_loader, optimizer, criterion, epoch, writer, output_dir, isWriteImage, isVal=False,
          test_loader=None):
    model.train()
    correct = 0
    tv_critic = TVLoss().to(device)
    for batch_idx, data in enumerate(train_loader):
        rgb = data[0]
        alb_map_gt = data[1]
        dep_map_gt = data[2]
        nor_map_gt = data[3]
        threeD_map_gt = data[4]
        uv_map_gt = data[5]
        mask_map_gt = data[6]
        rgb, alb_map_gt, dep_map_gt, nor_map_gt, uv_map_gt, threeD_map_gt, mask_map_gt = rgb.to(device), alb_map_gt.to(
            device), dep_map_gt.to(device), \
                                                                                         nor_map_gt.to(
                                                                                             device), uv_map_gt.to(
            device), threeD_map_gt.to(device), mask_map_gt.to(device)
        optimizer.zero_grad()
        uv_map, threeD_map, nor_map, alb_map, dep_map, mask_map, nor_from_threeD, dep_from_threeD, nor_from_dep, dep_from_nor = model(
            rgb)
        bc_critic = nn.BCELoss()
        loss_mask = bc_critic(mask_map, mask_map_gt).float()
        loss_tv = tv_critic(mask_map).float()
        loss_threeD = criterion(threeD_map, threeD_map_gt).float()
        loss_uv = criterion(uv_map, uv_map_gt).float()
        loss_dep = criterion(dep_map, dep_map_gt).float()
        loss_nor = criterion(nor_map, nor_map_gt).float()
        loss_alb = criterion(alb_map, torch.unsqueeze(alb_map_gt[:, 0, :, :], 1)).float()
        if nor_from_threeD is not None:
            cons_threeD2nor = criterion(nor_from_threeD, nor_map_gt).float()
        else:
            cons_threeD2nor = torch.Tensor([0.]).to(device)
        if dep_from_threeD is not None:
            cons_threeD2dep = criterion(dep_from_threeD, dep_map_gt).float()
        else:
            cons_threeD2dep = torch.Tensor([0.]).to(device)
        if nor_from_dep is not None:
            cons_dep2nor = criterion(nor_from_dep, nor_map_gt).float()
        else:
            cons_dep2nor = torch.Tensor([0.]).to(device)
        if dep_from_nor is not None:
            cons_nor2dep = criterion(dep_from_nor, dep_map_gt).float()
        else:
            cons_nor2dep = torch.Tensor([0.]).to(device)
        loss = 4 * loss_uv + 4 * loss_alb + 4 * loss_nor + loss_dep + 2 * loss_mask + loss_threeD + \
               cons_threeD2nor + cons_threeD2dep + cons_dep2nor + cons_nor2dep + loss_tv
        loss.backward()
        optimizer.step()
        lr = get_lr(optimizer)
        if batch_idx % args.log_intervals == 0:
            print(
                'Epoch:{} \n batch index:{}/{}||lr:{:.8f}||loss:{:.6f}||alb:{:.4f}||threeD:{:.4f}||uv:{:.6f}||nor:{:.4f}||dep:{:.4f}||mask:{:.6f}||cons_t2n:{:6f}'
                'cons_t2d:{:.6f}||cons_n2d:{:.6f}||cons_d2n:{:.6f}||loss_tv:{:.6f}'.format(epoch, batch_idx + 1,
                                                                                           len(
                                                                                               train_loader.dataset) // args.batch_size,
                                                                                           lr, loss.item(),
                                                                                           loss_alb.item(),
                                                                                           loss_threeD.item(),
                                                                                           loss_uv.item(),
                                                                                           loss_nor.item(),
                                                                                           loss_dep.item(),
                                                                                           loss_mask.item(),
                                                                                           cons_threeD2nor.item(),
                                                                                           cons_threeD2dep.item(),
                                                                                           cons_nor2dep.item(),
                                                                                           cons_dep2nor.item(),
                                                                                           loss_tv.item()))
            if args.write_summary:
                writer.add_scalar('summary/train_loss', loss.item(),
                                  global_step=epoch * len(train_loader) + batch_idx + 1)
                writer.add_scalar('summary/train_cmap_loss', loss_threeD.item(),
                                  global_step=epoch * len(train_loader) + batch_idx + 1)
                writer.add_scalar('summary/train_uv_loss', loss_uv.item(),
                                  global_step=epoch * len(train_loader) + batch_idx + 1)
                writer.add_scalar('summary/train_normal_loss', loss_nor.item(),
                                  global_step=epoch * len(train_loader) + batch_idx + 1)
                writer.add_scalar('summary/train_depth_loss', loss_dep.item(),
                                  global_step=epoch * len(train_loader) + batch_idx + 1)
                writer.add_scalar('summary/train_ab_loss', loss_alb.item(),
                                  global_step=epoch * len(train_loader) + batch_idx + 1)
                writer.add_scalar('summary/train_back_loss', loss_mask.item(),
                                  global_step=epoch * len(train_loader) + batch_idx + 1)
                writer.add_scalar('summary/train_constrain_3d2normal', cons_threeD2nor.item(),
                                  global_step=epoch * len(train_loader) + batch_idx + 1)
                writer.add_scalar('summary/train_constrain_3d2depth', cons_threeD2dep.item(),
                                  global_step=epoch * len(train_loader) + batch_idx + 1)
                writer.add_scalar('summary/train_constrain_normal2depth', cons_nor2dep.item(),
                                  global_step=epoch * len(train_loader) + batch_idx + 1)
                writer.add_scalar('summary/train_constrain_depth2normal', cons_dep2nor.item(),
                                  global_step=epoch * len(train_loader) + batch_idx + 1)
                writer.add_scalar('summary/train_loss_tv', loss_tv.item(),
                                  global_step=epoch * len(train_loader) + batch_idx + 1)
                writer.add_scalar('summary/lrate', lr, global_step=epoch * len(train_loader) + batch_idx + 1)
        if isWriteImage:
            if batch_idx == (len(train_loader.dataset) // args.batch_size) - 1:
                print('writing image')
                if not os.path.exists(output_dir + 'train/epoch_{}_batch_{}'.format(epoch, batch_idx)):
                    os.makedirs(output_dir + 'train/epoch_{}_batch_{}'.format(epoch, batch_idx))
                for k in range(5):
                    albedo_pred = alb_map[k, :, :, :]
                    uv_pred = uv_map[k, :, :, :]
                    back_pred = mask_map[k, :, :, :]
                    back_pred = torch.round(back_pred)
                    cmap_pred = threeD_map[k, :, :, :]
                    depth_pred = dep_map[k, :, :, :]
                    normal_pred = nor_map[k, :, :, :]
                    ori_gt = rgb[k, :, :, :]
                    ab_gt = alb_map_gt[k, :, :, :]
                    uv_gt = uv_map_gt[k, :, :, :]
                    mask_gt = mask_map_gt[k, :, :, :]
                    cmap_gt = threeD_map_gt[k, :, :, :]
                    depth_gt = dep_map_gt[k, :, :, :]
                    normal_gt = nor_map_gt[k, :, :, :]
                    bw_gt = metrics.uv2bmap(uv_gt, mask_gt)
                    bw_pred = metrics.uv2bmap(uv_pred, back_pred)
                    dewarp_ori = metrics.bw_mapping(bw_pred, ori_gt, device)
                    dewarp_ab = metrics.bw_mapping(bw_pred, ab_gt, device)
                    dewarp_ori_gt = metrics.bw_mapping(bw_gt, ori_gt, device)
                    output_dir1 = output_dir + 'train/epoch_{}_batch_{}/'.format(epoch, batch_idx)
                    output_uv_pred = output_dir1 + 'pred_uv_ind_{}'.format(k) + '.jpg'
                    output_back_pred = output_dir1 + 'pred_back_ind_{}'.format(k) + '.jpg'
                    output_ab_pred = output_dir1 + 'pred_ab_ind_{}'.format(k) + '.jpg'
                    output_3d_pred = output_dir1 + 'pred_3D_ind_{}'.format(k) + '.jpg'
                    output_bw_pred = output_dir1 + 'pred_bw_ind_{}'.format(k) + '.jpg'
                    output_depth_pred = output_dir1 + 'pred_depth_ind_{}'.format(k) + '.jpg'
                    output_normal_pred = output_dir1 + 'pred_normal_ind_{}'.format(k) + '.jpg'
                    output_ori = output_dir1 + 'gt_ori_ind_{}'.format(k) + '.jpg'
                    output_uv_gt = output_dir1 + 'gt_uv_ind_{}'.format(k) + '.jpg'
                    output_ab_gt = output_dir1 + 'gt_ab_ind_{}'.format(k) + '.jpg'
                    output_cmap_gt = output_dir1 + 'gt_cmap_ind_{}'.format(k) + '.jpg'
                    output_back_gt = output_dir1 + 'gt_back_ind_{}'.format(k) + '.jpg'
                    output_bw_gt = output_dir1 + 'gt_bw_ind_{}'.format(k) + '.jpg'
                    output_dewarp_ori_gt = output_dir1 + 'gt_dewarpOri_ind_{}'.format(k) + '.jpg'
                    output_depth_gt = output_dir1 + 'gt_depth_ind_{}'.format(k) + '.jpg'
                    output_normal_gt = output_dir1 + 'gt_normal_ind_{}'.format(k) + '.jpg'
                    output_dewarp_ori = output_dir1 + 'dewarp_ori_ind_{}'.format(k) + '.jpg'
                    output_dewarp_ab = output_dir1 + 'dewarp_ab_ind_{}'.format(k) + '.jpg'
                    """pred"""
                    write_image_tensor(uv_pred, output_uv_pred, 'std', device=device)
                    write_image_tensor(back_pred, output_back_pred, '01')
                    write_image_tensor(albedo_pred, output_ab_pred, 'std')
                    write_image_tensor(cmap_pred, output_3d_pred, 'gauss', mean=[0.1108, 0.3160, 0.2859],
                                       std=[0.7065, 0.6840, 0.7141])
                    write_image_tensor(depth_pred, output_depth_pred, 'gauss', mean=[0.5], std=[0.5])
                    write_image_tensor(normal_pred, output_normal_pred, 'gauss', mean=[0.5619, 0.2881, 0.2917],
                                       std=[0.5619, 0.7108, 0.7083])
                    write_image_np(bw_pred, output_bw_pred)
                    """gt"""
                    write_image_tensor(ori_gt, output_ori, 'std')
                    write_image_tensor(uv_gt, output_uv_gt, 'std', device=device)
                    write_image_tensor(mask_gt, output_back_gt, '01')
                    write_image_tensor(ab_gt, output_ab_gt, 'std')
                    write_image_tensor(cmap_gt, output_cmap_gt, 'gauss', mean=[0.1108, 0.3160, 0.2859],
                                       std=[0.7065, 0.6840, 0.7141])
                    write_image_tensor(depth_gt, output_depth_gt, 'gauss', mean=[0.5], std=[0.5])
                    write_image_tensor(normal_gt, output_normal_gt, 'gauss', mean=[0.5619, 0.2881, 0.2917],
                                       std=[0.5619, 0.7108, 0.7083])
                    write_image_np(bw_gt, output_bw_gt)
                    write_image(dewarp_ori_gt, output_dewarp_ori_gt)
                    """dewarp"""
                    write_image(dewarp_ori, output_dewarp_ori)
                    write_image(dewarp_ab, output_dewarp_ab)
        if isVal and (batch_idx + 1) % 500 == 0:
            sstep = test.count + 1
            test(args, model, device, test_loader, criterion, epoch, writer, output_dir, args.write_image_val, sstep)

    return lr


@CallingCounter
def test(args, model, device, test_loader, criterion, epoch, writer, output_dir, isWriteImage, sstep):
    print('Testing')
    model.eval()
    test_loss = 0
    correct = 0
    metrics_uv = {'l1_norm': 0, 'mse': 0, 'pearsonr_metric': 0, 'cc': 0, 'psnr': 0, 'ssim': 0, 'mssim': 0}
    metrics_cmap = {'l1_norm': 0, 'mse': 0, 'pearsonr_metric': 0, 'cc': 0, 'psnr': 0, 'ssim': 0, 'mssim': 0}
    metrics_alb = {'l1_norm': 0, 'mse': 0, 'pearsonr_metric': 0, 'cc': 0, 'psnr': 0, 'ssim': 0, 'mssim': 0}
    metrics_dep = {'l1_norm': 0, 'mse': 0, 'pearsonr_metric': 0, 'cc': 0, 'psnr': 0, 'ssim': 0, 'mssim': 0}
    metrics_nor = {'l1_norm': 0, 'mse': 0, 'pearsonr_metric': 0, 'cc': 0, 'psnr': 0, 'ssim': 0, 'mssim': 0}
    metrics_bw = {'l1_norm': 0, 'mse': 0, 'pearsonr_metric': 0, 'cc': 0, 'psnr': 0, 'ssim': 0, 'mssim': 0}
    metrics_deori = {'l1_norm': 0, 'mse': 0, 'pearsonr_metric': 0, 'cc': 0, 'psnr': 0, 'ssim': 0, 'mssim': 0}
    metrics_dealb = {'l1_norm': 0, 'mse': 0, 'pearsonr_metric': 0, 'cc': 0, 'psnr': 0, 'ssim': 0, 'mssim': 0}
    with torch.no_grad():
        loss_sum = 0
        loss_sum_alb = 0
        loss_sum_threeD = 0
        loss_sum_uv = 0
        loss_sum_nor = 0
        loss_sum_dep = 0
        loss_sum_mask = 0
        cons_sum_t2n = 0
        cons_sum_t2d = 0
        cons_sum_n2d = 0
        cons_sum_d2n = 0
        loss_sum_tv = 0
        tv_critic = TVLoss().to(device)
        start_time = time.time()
        for batch_idx, data in enumerate(test_loader):
            rgb = data[0]
            alb_map_gt = data[1]
            dep_map_gt = data[2]
            nor_map_gt = data[3]
            threeD_map_gt = data[4]
            uv_map_gt = data[5]
            mask_map_gt = data[6]
            rgb, alb_map_gt, dep_map_gt, nor_map_gt, uv_map_gt, threeD_map_gt, mask_map_gt = rgb.to(
                device), alb_map_gt.to(device), dep_map_gt.to(device), nor_map_gt.to(device), uv_map_gt.to(
                device), threeD_map_gt.to(device), mask_map_gt.to(device)
            uv_map, threeD_map, nor_map, alb_map, dep_map, mask_map, nor_from_threeD, dep_from_threeD, nor_from_dep, dep_from_nor = model(
                rgb)
            bc_critic = nn.BCELoss()
            loss_mask = bc_critic(mask_map, mask_map_gt).float()
            loss_tv = tv_critic(mask_map).float()
            loss_threeD = criterion(threeD_map, threeD_map_gt).float()
            loss_uv = criterion(uv_map, uv_map_gt).float()
            loss_dep = criterion(dep_map, dep_map_gt).float()
            loss_nor = criterion(nor_map, nor_map_gt).float()
            loss_alb = criterion(alb_map, torch.unsqueeze(alb_map_gt[:, 0, :, :], 1)).float()
            if nor_from_threeD is not None:
                cons_threeD2nor = criterion(nor_from_threeD, nor_map_gt).float()
            else:
                cons_threeD2nor = torch.Tensor([0.]).to(device)
            if dep_from_threeD is not None:
                cons_threeD2dep = criterion(dep_from_threeD, dep_map_gt).float()
            else:
                cons_threeD2dep = torch.Tensor([0.]).to(device)
            if nor_from_dep is not None:
                cons_dep2nor = criterion(nor_from_dep, nor_map_gt).float()
            else:
                cons_dep2nor = torch.Tensor([0.]).to(device)
            if dep_from_nor is not None:
                cons_nor2dep = criterion(dep_from_nor, dep_map_gt).float()
            else:
                cons_nor2dep = torch.Tensor([0.]).to(device)
            test_loss = 4 * loss_uv + 4 * loss_alb + 4 * loss_nor + loss_dep + 2 * loss_mask + loss_threeD + \
                        cons_threeD2nor + cons_threeD2dep + cons_dep2nor + cons_nor2dep + loss_tv
            loss_sum = loss_sum + test_loss
            loss_sum_alb += loss_alb
            loss_sum_threeD += loss_threeD
            loss_sum_uv += loss_uv
            loss_sum_nor += loss_nor
            loss_sum_dep += loss_dep
            loss_sum_mask += loss_mask
            cons_sum_t2n += cons_threeD2nor
            cons_sum_t2d += cons_threeD2dep
            cons_sum_n2d += cons_nor2dep
            cons_sum_d2n += cons_dep2nor
            loss_sum_tv += loss_tv
            if args.calculate_CC:
                alb_map_gt = torch.unsqueeze(alb_map_gt[:, 0, :, :], 1)
                alb_map_gt_recover = re_normalize(alb_map_gt, mean=0.5, std=0.5, inplace=False)
                alb_map_recover = re_normalize(alb_map, mean=0.5, std=0.5, inplace=False)
                metric_op = metrics.film_metrics().to(device)
                metric_alb = metric_op(alb_map_recover, alb_map_gt_recover)
                uv_map_recover = re_normalize(uv_map, mean=[0.5, 0.5], std=[0.5, 0.5], inplace=False)
                uv_map_gt_recover = re_normalize(uv_map_gt, mean=[0.5, 0.5], std=[0.5, 0.5], inplace=False)
                metric_uv = metric_op(uv_map_recover, uv_map_gt_recover)
                threeD_map_recover = re_normalize(threeD_map, mean=[0.1108, 0.3160, 0.2859],
                                                  std=[0.7065, 0.6840, 0.7141], inplace=False)
                threeD_map_gt_recover = re_normalize(threeD_map_gt, mean=[0.1108, 0.3160, 0.2859],
                                                     std=[0.7065, 0.6840, 0.7141], inplace=False)
                metric_cmap = metric_op(threeD_map_recover, threeD_map_gt_recover)
                nor_map_recover = re_normalize(nor_map, mean=[0.5619, 0.2881, 0.2917],
                                               std=[0.5619, 0.7108, 0.7083], inplace=False)
                nor_map_gt_recover = re_normalize(nor_map_gt, mean=[0.5619, 0.2881, 0.2917],
                                                  std=[0.5619, 0.7108, 0.7083], inplace=False)
                metric_nor = metric_op(nor_map_recover, nor_map_gt_recover)
                dep_map_recover = re_normalize(dep_map, mean=0.5, std=0.5, inplace=False)
                dep_map_gt_recover = re_normalize(dep_map_gt, mean=0.5, std=0.5, inplace=False)
                metric_dep = metric_op(dep_map_recover, dep_map_gt_recover)
                bw_pred = metrics.uv2bmap4d(uv_map, mask_map)
                bw_gt = metrics.uv2bmap4d(uv_map_gt, mask_map_gt)
                bw_pred = torch.from_numpy(bw_pred).to(device)
                bw_gt = torch.from_numpy(bw_gt).to(device)
                metric_bw = metric_op(bw_pred, bw_gt)
                dewarp_ori = metrics.bw_mapping4d(bw_pred, rgb, device)
                dewarp_ori_gt = metrics.bw_mapping4d(bw_gt, rgb, device)
                metric_deori = metric_op(dewarp_ori, dewarp_ori_gt)
                dewarp_ab = metrics.bw_mapping4d(bw_pred, alb_map, device)
                dewarp_ab_gt = metrics.bw_mapping4d(bw_gt, alb_map_gt, device)
                metric_dealb = metric_op(torch.unsqueeze(dewarp_ab, 0), torch.unsqueeze(dewarp_ab_gt, 0))

                metrics_alb = {key: metrics_alb[key] + metric_alb[key] for key in metrics_alb.keys()}
                metrics_uv = {key: metrics_uv[key] + metric_uv[key] for key in metrics_uv.keys()}
                metrics_cmap = {key: metrics_cmap[key] + metric_cmap[key] for key in metrics_cmap.keys()}
                metrics_dep = {key: metrics_dep[key] + metric_dep[key] for key in metrics_dep.keys()}
                metrics_nor = {key: metrics_nor[key] + metric_nor[key] for key in metrics_nor.keys()}
                metrics_bw = {key: metrics_bw[key] + metric_bw[key] for key in metrics_bw.keys()}
                metrics_deori = {key: metrics_deori[key] + metric_deori[key] for key in metrics_deori.keys()}
                metrics_dealb = {key: metrics_dealb[key] + metric_dealb[key] for key in metrics_dealb.keys()}
            if isWriteImage:
                if batch_idx == (len(test_loader.dataset) // args.test_batch_size) - 1:
                    if not os.path.exists(output_dir + 'test/epoch_{}_batch_{}'.format(epoch, batch_idx)):
                        os.makedirs(output_dir + 'test/epoch_{}_batch_{}'.format(epoch, batch_idx))
                    print('writing test image')
                    # for k in range(args.test_batch_size):
                    for k in range(5):
                        albedo_pred = alb_map[k, :, :, :]
                        uv_pred = uv_map[k, :, :, :]
                        back_pred = mask_map[k, :, :, :]
                        back_pred = torch.round(back_pred)
                        cmap_pred = threeD_map[k, :, :, :]
                        depth_pred = dep_map[k, :, :, :]
                        normal_pred = nor_map[k, :, :, :]

                        ori_gt = rgb[k, :, :, :]
                        ab_gt = alb_map_gt[k, :, :, :]
                        uv_gt = uv_map_gt[k, :, :, :]
                        mask_gt = mask_map_gt[k, :, :, :]
                        cmap_gt = threeD_map_gt[k, :, :, :]
                        depth_gt = dep_map_gt[k, :, :, :]
                        normal_gt = nor_map_gt[k, :, :, :]

                        bw_gt = metrics.uv2bmap(uv_gt, mask_gt)
                        bw_pred = metrics.uv2bmap(uv_pred, back_pred)  # [-1,1], [256, 256, 3]
                        dewarp_ori = metrics.bw_mapping(bw_pred, ori_gt, device)
                        dewarp_ab = metrics.bw_mapping(bw_pred, ab_gt, device)
                        dewarp_ori_gt = metrics.bw_mapping(bw_gt, ori_gt, device)

                        output_dir1 = output_dir + 'test/epoch_{}_batch_{}/'.format(epoch, batch_idx)
                        output_uv_pred = output_dir1 + 'pred_uv_ind_{}'.format(k) + '.jpg'
                        output_back_pred = output_dir1 + 'pred_back_ind_{}'.format(k) + '.jpg'
                        output_ab_pred = output_dir1 + 'pred_ab_ind_{}'.format(k) + '.jpg'
                        output_3d_pred = output_dir1 + 'pred_3D_ind_{}'.format(k) + '.jpg'
                        output_bw_pred = output_dir1 + 'pred_bw_ind_{}'.format(k) + '.jpg'
                        output_depth_pred = output_dir1 + 'pred_depth_ind_{}'.format(k) + '.jpg'
                        output_normal_pred = output_dir1 + 'pred_normal_ind_{}'.format(k) + '.jpg'

                        output_ori = output_dir1 + 'gt_ori_ind_{}'.format(k) + '.jpg'
                        output_uv_gt = output_dir1 + 'gt_uv_ind_{}'.format(k) + '.jpg'
                        output_ab_gt = output_dir1 + 'gt_ab_ind_{}'.format(k) + '.jpg'
                        output_cmap_gt = output_dir1 + 'gt_cmap_ind_{}'.format(k) + '.jpg'
                        output_back_gt = output_dir1 + 'gt_back_ind_{}'.format(k) + '.jpg'
                        output_bw_gt = output_dir1 + 'gt_bw_ind_{}'.format(k) + '.jpg'
                        output_dewarp_ori_gt = output_dir1 + 'gt_dewarpOri_ind_{}'.format(k) + '.jpg'
                        output_depth_gt = output_dir1 + 'gt_depth_ind_{}'.format(k) + '.jpg'
                        output_normal_gt = output_dir1 + 'gt_normal_ind_{}'.format(k) + '.jpg'

                        output_dewarp_ori = output_dir1 + 'dewarp_ori_ind_{}'.format(k) + '.jpg'
                        output_dewarp_ab = output_dir1 + 'dewarp_ab_ind_{}'.format(k) + '.jpg'

                        """pred"""
                        write_image_tensor(uv_pred, output_uv_pred, 'std', device=device)
                        write_image_tensor(back_pred, output_back_pred, '01')
                        write_image_tensor(albedo_pred, output_ab_pred, 'std')
                        write_image_tensor(cmap_pred, output_3d_pred, 'gauss', mean=[0.1108, 0.3160, 0.2859],
                                           std=[0.7065, 0.6840, 0.7141])
                        write_image_tensor(depth_pred, output_depth_pred, 'gauss', mean=[0.5], std=[0.5])
                        write_image_tensor(normal_pred, output_normal_pred, 'gauss', mean=[0.5619, 0.2881, 0.2917],
                                           std=[0.5619, 0.7108, 0.7083])
                        write_image_np(bw_pred, output_bw_pred)
                        """gt"""
                        write_image_tensor(ori_gt, output_ori, 'std')
                        write_image_tensor(uv_gt, output_uv_gt, 'std', device=device)
                        write_image_tensor(mask_gt, output_back_gt, '01')
                        write_image_tensor(ab_gt, output_ab_gt, 'std')
                        write_image_tensor(cmap_gt, output_cmap_gt, 'gauss', mean=[0.1108, 0.3160, 0.2859],
                                           std=[0.7065, 0.6840, 0.7141])
                        write_image_tensor(depth_gt, output_depth_gt, 'gauss', mean=[0.5], std=[0.5])
                        write_image_tensor(normal_gt, output_normal_gt, 'gauss', mean=[0.5619, 0.2881, 0.2917],
                                           std=[0.5619, 0.7108, 0.7083])
                        write_image_np(bw_gt, output_bw_gt)

                        write_image(dewarp_ori_gt, output_dewarp_ori_gt)

                        """dewarp"""
                        write_image(dewarp_ori, output_dewarp_ori)
                        write_image(dewarp_ab, output_dewarp_ab)
            if (batch_idx + 1) % 20 == 0:
                print('It cost {} seconds to test {} images'.format(time.time() - start_time,
                                                                    (batch_idx + 1) * args.test_batch_size))
                start_time = time.time()
    test_loss = loss_sum / (len(test_loader.dataset) / args.test_batch_size)
    test_loss_alb = loss_sum_alb / (len(test_loader.dataset) / args.test_batch_size)
    test_loss_threeD = loss_sum_threeD / (len(test_loader.dataset) / args.test_batch_size)
    test_loss_uv = loss_sum_uv / (len(test_loader.dataset) / args.test_batch_size)
    test_loss_nor = loss_sum_nor / (len(test_loader.dataset) / args.test_batch_size)
    test_loss_dep = loss_sum_dep / (len(test_loader.dataset) / args.test_batch_size)
    test_loss_mask = loss_sum_mask / (len(test_loader.dataset) / args.test_batch_size)
    test_cons_t2n = cons_sum_t2n / (len(test_loader.dataset) / args.test_batch_size)
    test_cons_t2d = cons_sum_t2d / (len(test_loader.dataset) / args.test_batch_size)
    test_cons_n2d = cons_sum_n2d / (len(test_loader.dataset) / args.test_batch_size)
    test_cons_d2n = cons_sum_d2n / (len(test_loader.dataset) / args.test_batch_size)
    test_loss_tv = loss_tv / (len(test_loader.dataset) / args.test_batch_size)
    print(
        'Epoch:{} \n batch index:{}/{}||loss:{:.6f}||alb:{:.4f}||threeD:{:.4f}||uv:{:.6f}||nor:{:.4f}||dep:{:.4f}||mask:{:.6f}||cons_t2n:{:6f}'
        'cons_t2d:{:.6f}||cons_n2d:{:.6f}||cons_d2n:{:.6f}||loss_tv:{:.6f}'.format(epoch, batch_idx + 1,
                                                                                   len(
                                                                                       test_loader.dataset) // args.batch_size,
                                                                                   test_loss.item(),
                                                                                   test_loss_alb.item(),
                                                                                   test_loss_threeD.item(),
                                                                                   test_loss_uv.item(),
                                                                                   test_loss_nor.item(),
                                                                                   test_loss_dep.item(),
                                                                                   test_loss_mask.item(),
                                                                                   test_cons_t2n.item(),
                                                                                   test_cons_t2d.item(),
                                                                                   test_cons_n2d.item(),
                                                                                   test_cons_d2n.item(),
                                                                                   test_loss_tv.item()))
    if args.calculate_CC:
        num_iters = math.ceil(len(test_loader.dataset) / args.test_batch_size)
        metrics_alb = {key: metrics_alb[key] / num_iters for key in metrics_alb.keys()}
        metrics_uv = {key: metrics_uv[key] / num_iters for key in metrics_uv.keys()}
        metrics_cmap = {key: metrics_cmap[key] / num_iters for key in metrics_cmap.keys()}
        metrics_dep = {key: metrics_dep[key] / num_iters for key in metrics_dep.keys()}
        metrics_nor = {key: metrics_nor[key] / num_iters for key in metrics_nor.keys()}
        metrics_bw = {key: metrics_bw[key] / num_iters for key in metrics_bw.keys()}
        metrics_deori = {key: metrics_deori[key] / num_iters for key in metrics_deori.keys()}
        metrics_dealb = {key: metrics_dealb[key] / num_iters for key in metrics_dealb.keys()}
    if args.calculate_CC:
        for key in metrics_alb.keys():
            print(str(key) + '_uv:{:.6f}\t' + str(key) + '_dep:{:.6f}\t' + str(key) + '_nor:{:.6f}\t' + str(key) +
                  '_cmap:{:.6f}\t' + str(key) + '_alb:{:.6f}\t' + str(key) + '_bw:{:.6f}\t' + str(
                key) + '_deori:{:.6f}\t' +
                  str(key) + '_dealb:{:.6f}'.format(metrics_uv[key], metrics_dep[key], metrics_nor[key],
                                                    metrics_cmap[key], metrics_alb[key],
                                                    metrics_bw[key], metrics_deori[key], metrics_dealb[key]))
    if args.write_txt:
        txt_dir = 'output_txt/' + args.model_name + '.txt'
        f = open(txt_dir, 'a')
        f.write(
            'Epoch: {} \t Test Loss: {:.6f}, \t ab: {:.4f}, \t cmap: {:.4f}, \t uv: {:.6f}, \t normal: {:.4f}, \t depth: {:.4f}, \t back: {:.6f} , \t constrain 3d to normal: {:.4f}, \t constrain 3d to depth: {:.4f}, \t constrain normal to depth: {:.4f}, \t constrain depth to normal: {:.4f}, \t loss tv: {:.6f}\n'.format(
                epoch, test_loss.item(),
                test_loss_alb.item(), test_loss_threeD.item(), test_loss_uv.item(), test_loss_nor.item(),
                test_loss_dep.item(), test_loss_mask.item(), test_cons_t2n.item(), test_cons_t2d.item(),
                test_cons_n2d.item(), test_cons_d2n.item(), test_loss_tv.item()))
        for key in metrics_alb.keys():
            f.write(str(key) + '_uv:{:.6f}\t' + str(key) + '_dep:{:.6f}\t' + str(key) + '_nor:{:.6f}\t' + str(key) +
                    '_cmap:{:.6f}\t' + str(key) + '_alb:{:.6f}\t' + str(key) + '_bw:{:.6f}\t' + str(
                key) + '_deori:{:.6f}\t' +
                    str(key) + '_dealb:{:.6f}\n'.format(metrics_uv[key], metrics_dep[key], metrics_nor[key],
                                                        metrics_cmap[key], metrics_alb[key],
                                                        metrics_bw[key], metrics_deori[key], metrics_dealb[key]))
        f.close()
    if args.write_summary:
        print('sstep', sstep)
        # writer.add_scalar('test_acc', 100. * correct / len(test_loader.dataset), global_step=epoch+1)
        writer.add_scalar('summary/test_loss', test_loss.item(), global_step=sstep)
        writer.add_scalar('summary/test_loss_ab', test_loss_alb.item(), global_step=sstep)
        writer.add_scalar('summary/test_loss_cmap', test_loss_threeD.item(), global_step=sstep)
        writer.add_scalar('summary/test_loss_uv', test_loss_uv.item(), global_step=sstep)
        writer.add_scalar('summary/test_loss_normal', test_loss_nor.item(), global_step=sstep)
        writer.add_scalar('summary/test_loss_depth', test_loss_dep.item(), global_step=sstep)
        writer.add_scalar('summary/test_loss_back', test_loss_mask.item(), global_step=sstep)
        writer.add_scalar('summary/test_con_3d2nor', test_cons_t2n.item(), global_step=sstep)
        writer.add_scalar('summary/test_con_3d2dep', test_cons_t2d.item(), global_step=sstep)
        writer.add_scalar('summary/test_con_nor2dep', test_cons_n2d.item(), global_step=sstep)
        writer.add_scalar('summary/test_loss_dep2nor', test_cons_d2n.item(), global_step=sstep)
        writer.add_scalar('summary/test_loss_tv', test_loss_tv.item(), global_step=sstep)


def main():
    # Model Build
    model = models.UnwarpNet(use_simple=False, use_constrain=True, combine_num=1,
                             constrain_configure=models.constrain_path)
    args = train_configs.args
    isTrain = True
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        print(" [*] Set cuda: True")
        model = model.cuda()
    else:
        print(" [*] Set cuda: False")
    # model = model.to(device)
    # Load Dataset
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    dataset_test = filmDataset(npy_dir=args.test_path)
    dataset_test_loader = DataLoader(dataset_test,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     **kwargs)
    dataset_train = filmDataset(npy_dir=args.train_path)
    dataset_train_loader = DataLoader(dataset_train, batch_size=args.batch_size,
                                      shuffle=True, **kwargs)
    start_epoch = 1
    learning_rate = args.lr
    # Load Parameters
    # if True:
    if args.pretrained:
        # pretrained_dict = torch.load(pretrained_model_dir, map_location=None)
        # model_dict = model.state_dict()
        # # 1. filter out unnecessary keys
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # # 2. overwrite entries in the existing state dict
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)
        pretrained_dict = torch.load(args.pretrained_model_dir, map_location=None)
        model.load_state_dict(pretrained_dict['model_state'])
        start_lr = pretrained_dict['lr']
        start_epoch = pretrained_dict['epoch']
    # Add Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    # model, optimizer = amp.initialize(model, optimizer,opt_level='O1',loss_scale="dynamic",verbosity=0)
    model = torch.nn.DataParallel(model.cuda())
    if args.use_mse:
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.L1Loss()
    if args.visualize_para:
        for name, parameters in model.named_parameters():
            print(name, ':', parameters.size())
    if args.write_summary:
        if not os.path.exists('summary/' + args.model_name + '_start_epoch{}'.format(start_epoch)):
            os.makedirs('summary/' + args.model_name + '_start_epoch{}'.format(start_epoch))

        writer = SummaryWriter(logdir='summary/' + args.model_name + '_start_epoch{}'.format(start_epoch))
        print(args.model_name)
    else:
        writer = 0
    start_lr = args.lr
    print('start_lr', start_lr)
    print('start_epoch', start_epoch)
    for epoch in range(start_epoch, args.epochs + 1):
        if isTrain:
            lr = train(args, model, device, dataset_train_loader, optimizer, criterion, epoch, writer, args.output_dir,
                       args.write_image_train, isVal=False,
                       test_loader=dataset_test_loader)
            sstep = test.count + 1
            args.calculate_CC = False
            if epoch % 7 == 0:
                args.calculate_CC = True
            test(args, model, device, dataset_test_loader, criterion, epoch, writer, args.output_dir,
                 args.write_image_test, sstep)
        else:
            sstep = test.count + 1
            test(args, model, device, dataset_test_loader, criterion, epoch, writer, args.output_dir,
                 args.write_image_test, sstep)
            break
        scheduler.step()
        if isTrain and args.save_model:
            state = {'epoch': epoch + 1,
                     'lr': lr,
                     'model_state': model.state_dict(),
                     'optimizer_state': optimizer.state_dict()
                     }
            torch.save(state, args.save_model_dir + "{}_{}.pkl".format(args.model_name, epoch))


def exist_or_make(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == '__main__':
    main()
