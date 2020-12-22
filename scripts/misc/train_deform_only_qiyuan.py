import torch.nn as nn
import unwarp_models
from . import train_configs
from torch.utils.data import Dataset, DataLoader
from dataloader.load_data_2 import filmDataset_2, filmDataset_3
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
# from cal_times import CallingCounter
import entire_unwarp_models
import time
import os
import metrics
import numpy as np
from loss_deformes import loss_deform, deform2bw_tensor_batch, recon_loss
from write_image import *
from tensorboardX import SummaryWriter
import math
from tutils import *


def diceCoeff(pred, gt, smooth=1, activation=None):
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d activation function operation")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = 2 * (intersection + smooth) / (unionset + smooth)

    return loss.sum() / N


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 0"


################################################################################################
#                                             TOOLS
################################################################################################
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return float(param_group['lr'])


def exist_or_make(path):
    if not os.path.exists(path):
        os.mkdir(path)


################################################################################################
#                                            TRAIN
################################################################################################
def train(args, model, device, train_loader, optimizer, criterion, epoch, writer, output_dir_train, isWriteImage,
          isVal=False,
          test_loader=None):
    model.train()
    bc_critic = nn.BCELoss()
    for batch_idx, data in enumerate(train_loader):
        ########################################################################################
        #                                LOAD DATA AND PREDICT
        ########################################################################################
        rgb = data[0]
        alb_map_gt = data[1]
        dep_map_gt = data[2]
        nor_map_gt = data[3]
        cmap_map_gt = data[4]
        uv_map_gt = data[5]
        mask_map_gt = data[6]
        bw_map_gt = data[7]
        deform_map_gt = data[8]
        names = data[9]
        rgb, alb_map_gt, dep_map_gt, nor_map_gt, uv_map_gt, cmap_map_gt, mask_map_gt, bw_map_gt, deform_map_gt = \
            rgb.to(device), alb_map_gt.to(device), dep_map_gt.to(device), nor_map_gt.to(device), uv_map_gt.to(device), \
            cmap_map_gt.to(device), mask_map_gt.to(device), bw_map_gt.to(device), deform_map_gt.to(device)
        optimizer.zero_grad()
        cmap, nor_map, alb_map, dep_map, mask_map, deform_map = model(rgb)
        #bw_gt = metrics.uv2bmap4d(uv_map_gt, mask_map_gt)
        #dewarp_ori_gt = metrics.bw_mapping4d(bw_gt, rgb, device)
        dewarp_ori_gt = metrics.bw_mapping4d(bw_map_gt, rgb, device)
        ########################################################################################
        #                                    LOSS SUMMARY
        ########################################################################################
        loss_mask = bc_critic(mask_map, mask_map_gt).float()
        loss_cmap = criterion(cmap, cmap_map_gt).float()
        loss_dep = criterion(dep_map, dep_map_gt).float()
        loss_nor = criterion(nor_map, nor_map_gt).float()
        loss_alb = criterion(alb_map, torch.unsqueeze(alb_map_gt[:, 0, :, :], 1)).float()
        loss_smooth = criterion(deform_map, deform_map_gt)
        loss = 4 * loss_alb + 4 * loss_nor + loss_dep + 2 * loss_mask + loss_cmap + \
               loss_smooth
        loss.backward()
        optimizer.step()
        lr = get_lr(optimizer)
        alb_map_gt = torch.unsqueeze(alb_map_gt[:, 0, :, :], 1)
        if batch_idx % args.log_intervals == 0:
            print(
                'Epoch:{} \n batch index:{}/{}||lr:{:.8f}||loss:{:.6f}||alb:{:.6f}||threeD:{:.6f}||nor:{:.6f}||dep:{:.6f}||'
                'mask:{:.6f}||smooth:{:.6f}'.format(
                    epoch, batch_idx + 1,
                           len(train_loader.dataset) // args.batch_size,
                    lr, loss.item(),
                    loss_alb.item(),
                    loss_cmap.item(),
                    loss_nor.item(),
                    loss_dep.item(),
                    loss_mask.item(),
                    loss_smooth.item(),
                ))
            if args.write_summary:
                writer.add_scalar('summary/train_loss', loss.item(),
                                  global_step=epoch * len(train_loader) + batch_idx + 1)
                writer.add_scalar('summary/train_cmap_loss', loss_cmap.item(),
                                  global_step=epoch * len(train_loader) + batch_idx + 1)
                writer.add_scalar('summary/train_normal_loss', loss_nor.item(),
                                  global_step=epoch * len(train_loader) + batch_idx + 1)
                writer.add_scalar('summary/train_depth_loss', loss_dep.item(),
                                  global_step=epoch * len(train_loader) + batch_idx + 1)
                writer.add_scalar('summary/train_ab_loss', loss_alb.item(),
                                  global_step=epoch * len(train_loader) + batch_idx + 1)
                writer.add_scalar('summary/train_back_loss', loss_mask.item(),
                                  global_step=epoch * len(train_loader) + batch_idx + 1)
                writer.add_scalar('summary/train_smooth_loss', loss_smooth.item(),
                                  global_step=epoch * len(train_loader) + batch_idx + 1)
                writer.add_scalar('summary/lrate', lr, global_step=epoch * len(train_loader) + batch_idx + 1)
        if isWriteImage:
            if batch_idx == (len(train_loader.dataset) // args.batch_size) - 1:
                print('writing image')
                if not os.path.exists(output_dir_train + 'train/epoch_{}_batch_{}'.format(epoch, batch_idx)):
                    os.makedirs(output_dir_train + 'train/epoch_{}_batch_{}'.format(epoch, batch_idx))
                for k in range(5):
                    deform_bw_map = deform2bw_tensor_batch(deform_map.clone().to(device), device)
                    alb_pred = alb_map[k, :, :, :]
                    deform_pred = deform_map[k, :, :, :]
                    mask_pred = mask_map[k, :, :, :]
                    mask_pred = torch.round(mask_pred)
                    name = names[k]
                    cmap_pred = cmap[k, :, :, :]
                    dep_pred = dep_map[k, :, :, :]
                    nor_pred = nor_map[k, :, :, :]
                    ori_gt = rgb[k, :, :, :]
                    alb_gt = alb_map_gt[k, :, :, :]
                    uv_gt = uv_map_gt[k, :, :, :]
                    mask_gt = mask_map_gt[k, :, :, :]
                    cmap_gt = cmap_map_gt[k, :, :, :]
                    dep_gt = dep_map_gt[k, :, :, :]
                    deform_gt = deform_map_gt[k, :, :, :]
                    bw_real_gt = bw_map_gt[k, :, :, :]
                    nor_gt = nor_map_gt[k, :, :, :]
                    deform_bw_pred = deform_bw_map[k, :, :, :]
                    bw_gt = metrics.uv2bmap(uv_gt, mask_gt)
                    dewarp_deform_ori = metrics.bw_mapping(deform_bw_pred, ori_gt, device)
                    dewarp_deform_alb = metrics.bw_mapping(deform_bw_pred, alb_pred, device)
                    dewarp_ori_gt = metrics.bw_mapping(bw_gt, ori_gt, device)
                    dewarp_ori_real_gt = metrics.bw_mapping(bw_real_gt, ori_gt, device)
                    dewarp_alb_gt = metrics.bw_mapping(bw_gt, alb_gt, device)
                    output_dir = os.path.join(output_dir_train, 'train/epoch_{}_batch_{}/'.format(epoch, batch_idx))
                    output_mask_pred = os.path.join(output_dir, 'pred_mask_ind_{}'.format(name) + '.jpg')
                    output_alb_pred = os.path.join(output_dir, 'pred_alb_ind_{}'.format(name) + '.jpg')
                    output_cmap_pred = os.path.join(output_dir, 'pred_3D_ind_{}'.format(name) + '.jpg')
                    output_deform_bw_pred = os.path.join(output_dir, 'pred_deform_bw_ind_{}'.format(name) + '.exr')
                    output_dep_pred = os.path.join(output_dir, 'pred_dep_ind_{}'.format(name) + '.jpg')
                    output_nor_pred = os.path.join(output_dir, 'pred_normal_ind_{}'.format(name) + '.jpg')
                    output_ori = os.path.join(output_dir, 'gt_ori_ind_{}'.format(name) + '.jpg')
                    output_uv_gt = os.path.join(output_dir, 'gt_uv_ind_{}'.format(name) + '.exr')
                    output_alb_gt = os.path.join(output_dir, 'gt_alb_ind_{}'.format(name) + '.jpg')
                    output_cmap_gt = os.path.join(output_dir, 'gt_cmap_ind_{}'.format(name) + '.jpg')
                    output_mask_gt = os.path.join(output_dir, 'gt_mask_ind_{}'.format(name) + '.jpg')
                    output_bw_gt = os.path.join(output_dir, 'gt_bw_ind_{}'.format(name) + '.exr')
                    output_dewarp_ori_gt = os.path.join(output_dir, 'gt_dewarp_ori_ind_{}'.format(name) + '.jpg')
                    output_dewarp_ori_real_gt = os.path.join(output_dir, 'gt_dewarp_ori_real_ind_{}'.format(name) + '.jpg')
                    output_dewarp_alb_gt = os.path.join(output_dir, 'gt_dewarp_alb_ind_{}'.format(name) + '.jpg')
                    output_dep_gt = os.path.join(output_dir, 'gt_dep_ind_{}'.format(name) + '.jpg')
                    output_nor_gt = os.path.join(output_dir, 'gt_nor_ind_{}'.format(name) + '.jpg')
                    output_deform_dewarp_ori = os.path.join(output_dir,
                                                            'deform_dewarp_ori_ind_{}'.format(name) + '.jpg')
                    output_deform_dewarp_alb = os.path.join(output_dir,
                                                            'deform_dewarp_alb_ind_{}'.format(name) + '.jpg')
                    """pred"""
                    write_image_tensor(mask_pred, output_mask_pred, '01')
                    write_image_tensor(alb_pred, output_alb_pred, 'std')
                    write_image_tensor(cmap_pred, output_cmap_pred, 'gauss', mean=[0.1108, 0.3160, 0.2859],
                                       std=[0.7065, 0.6840, 0.7141])
                    write_image_tensor(dep_pred, output_dep_pred, 'gauss', mean=[0.5], std=[0.5])
                    write_image_tensor(nor_pred, output_nor_pred, 'gauss', mean=[0.5619, 0.2881, 0.2917],
                                       std=[0.5619, 0.7108, 0.7083])
                    write_image_tensor(deform_bw_pred, output_deform_bw_pred, 'std', device=device)
                    """gt"""
                    write_image_tensor(ori_gt, output_ori, 'std')
                    write_image_tensor(uv_gt, output_uv_gt, 'std', device=device)
                    write_image_tensor(mask_gt, output_mask_gt, '01')
                    write_image_tensor(alb_gt, output_alb_gt, 'std')
                    write_image_tensor(cmap_gt, output_cmap_gt, 'gauss', mean=[0.1108, 0.3160, 0.2859],
                                       std=[0.7065, 0.6840, 0.7141])
                    write_image_tensor(dep_gt, output_dep_gt, 'gauss', mean=[0.5], std=[0.5])
                    write_image_tensor(nor_gt, output_nor_gt, 'gauss', mean=[0.5619, 0.2881, 0.2917],
                                       std=[0.5619, 0.7108, 0.7083])
                    write_image_np(bw_gt, output_bw_gt)
                    write_image(dewarp_ori_gt, output_dewarp_ori_gt)
                    write_image(dewarp_ori_real_gt, output_dewarp_ori_real_gt)
                    write_image(dewarp_alb_gt, output_dewarp_alb_gt)
                    """dewarp"""
                    write_image(dewarp_deform_ori, output_deform_dewarp_ori)
                    write_image(dewarp_deform_alb, output_deform_dewarp_alb)
        if isVal and (batch_idx + 1) % 500 == 0:
            sstep = test.count + 1
            test(args, model, device, test_loader, criterion, epoch, writer, output_dir, args.write_image_val, sstep)

    return lr


@CallingCounter
def test(args, model, device, test_loader, criterion, epoch, writer, output_dir_test, isWriteImage, sstep):
    print('Testing')
    model.eval()
    metrics_cmap = {'l1_norm': 0, 'mse': 0, 'cc': 0, 'psnr': 0, 'ssim': 0, 'mssim': 0, 'ncc': 0}
    metrics_alb = {'l1_norm': 0, 'mse': 0, 'cc': 0, 'psnr': 0, 'ssim': 0, 'mssim': 0, 'ncc': 0}
    metrics_dep = {'l1_norm': 0, 'mse': 0, 'cc': 0, 'psnr': 0, 'ssim': 0, 'mssim': 0, 'ncc': 0}
    metrics_nor = {'l1_norm': 0, 'mse': 0, 'cc': 0, 'psnr': 0, 'ssim': 0, 'mssim': 0, 'ncc': 0}
    metrics_deform_bw = {'l1_norm': 0, 'mse': 0, 'cc': 0, 'psnr': 0, 'ssim': 0, 'mssim': 0, 'ncc': 0}
    metrics_deform_ori = {'l1_norm': 0, 'mse': 0, 'cc': 0, 'psnr': 0, 'ssim': 0, 'mssim': 0, 'ncc': 0}
    metrics_deform_alb = {'l1_norm': 0, 'mse': 0, 'cc': 0, 'psnr': 0, 'ssim': 0, 'mssim': 0, 'ncc': 0}
    with torch.no_grad():
        loss_sum = 0
        loss_sum_alb = 0
        loss_sum_cmap = 0
        loss_sum_nor = 0
        loss_sum_dep = 0
        loss_sum_mask = 0
        loss_sum_smooth = 0
        dice_sum_metric = 0
        bc_critic = nn.BCELoss()
        start_time = time.time()
        for batch_idx, data in enumerate(test_loader):
            rgb = data[0]
            alb_map_gt = data[1]
            dep_map_gt = data[2]
            nor_map_gt = data[3]
            cmap_map_gt = data[4]
            uv_map_gt = data[5]
            mask_map_gt = data[6]
            bw_map_gt = data[7]
            deform_map_gt = data[8]
            names = data[9]
            rgb, alb_map_gt, dep_map_gt, nor_map_gt, uv_map_gt, cmap_map_gt, mask_map_gt, bw_map_gt, deform_map_gt = \
                rgb.to(device), alb_map_gt.to(device), dep_map_gt.to(device), nor_map_gt.to(device), uv_map_gt.to(
                    device), \
                cmap_map_gt.to(device), mask_map_gt.to(device), bw_map_gt.to(device), deform_map_gt.to(device)
            cmap, nor_map, alb_map, dep_map, mask_map, deform_map = model(rgb)
            #bw_gt = metrics.uv2bmap4d(uv_map_gt, mask_map_gt)
            #dewarp_ori_gt = metrics.bw_mapping4d(bw_gt, rgb, device)
            dewarp_ori_gt = metrics.bw_mapping4d(bw_map_gt, rgb, device)
            ######################################################################################################################
            #                                                  TEST LOSS SUMMARY
            ######################################################################################################################
            loss_mask = bc_critic(mask_map, mask_map_gt).float()
            dice_metric = diceCoeff(mask_map, mask_map_gt).float()
            loss_cmap = criterion(cmap, cmap_map_gt).float()
            loss_dep = criterion(dep_map, dep_map_gt).float()
            loss_nor = criterion(nor_map, nor_map_gt).float()
            loss_alb = criterion(alb_map, torch.unsqueeze(alb_map_gt[:, 0, :, :], 1)).float()
            loss_smooth = criterion(deform_map, deform_map_gt)
            test_loss = 4 * loss_alb + 4 * loss_nor + loss_dep + 2 * loss_mask + loss_cmap + \
                        loss_smooth
            loss_sum = loss_sum + test_loss
            loss_sum_alb += loss_alb
            loss_sum_cmap += loss_cmap
            loss_sum_nor += loss_nor
            loss_sum_dep += loss_dep
            loss_sum_mask += loss_mask
            loss_sum_smooth += loss_smooth
            dice_sum_metric += dice_metric
            alb_map_gt = torch.unsqueeze(alb_map_gt[:, 0, :, :], 1)
            if args.calculate_CC:
                metric_op = metrics.film_metrics_ncc().to(device)
                # ALB
                alb_map_gt_recover = torch.clamp(re_normalize(alb_map_gt, mean=0.5, std=0.5, inplace=False), 0., 1.)
                alb_map_recover = torch.clamp(re_normalize(alb_map, mean=0.5, std=0.5, inplace=False), 0., 1.)
                metric_alb = metric_op(alb_map_recover, alb_map_gt_recover)
                # UV
                uv_map_gt_recover = torch.clamp(re_normalize(uv_map_gt, mean=[0.5, 0.5], std=[0.5, 0.5], inplace=False),
                                                0., 1.)
                # CMAP
                cmap_recover = torch.clamp(
                    re_normalize(cmap, mean=[0.1108, 0.3160, 0.2859], std=[0.7065, 0.6840, 0.7141], inplace=False), 0.,
                    1.)
                cmap_gt_recover = torch.clamp(
                    re_normalize(cmap_map_gt, mean=[0.1108, 0.3160, 0.2859], std=[0.7065, 0.6840, 0.7141],
                                 inplace=False),
                    0., 1.)
                metric_cmap = metric_op(cmap_recover, cmap_gt_recover)
                # Normal
                nor_map_recover = torch.clamp(
                    re_normalize(nor_map, mean=[0.5619, 0.2881, 0.2917], std=[0.5619, 0.7108, 0.7083], inplace=False),
                    0., 1.)
                nor_map_gt_recover = torch.clamp(
                    re_normalize(nor_map_gt, mean=[0.5619, 0.2881, 0.2917], std=[0.5619, 0.7108, 0.7083],
                                 inplace=False), 0., 1.)
                metric_nor = metric_op(nor_map_recover, nor_map_gt_recover)
                # Depth
                dep_map_recover = torch.clamp(re_normalize(dep_map, mean=0.5, std=0.5, inplace=False), 0., 1.)
                dep_map_gt_recover = torch.clamp(re_normalize(dep_map_gt, mean=0.5, std=0.5, inplace=False), 0., 1.)
                metric_dep = metric_op(dep_map_recover, dep_map_gt_recover)
                # UV2BW
                bw_gt = metrics.uv2bmap4d(uv_map_gt, mask_map_gt)
                bw_gt_recover = torch.clamp(
                    re_normalize(torch.from_numpy(np.transpose(bw_gt, (0, 3, 1, 2))).to(device), mean=[0.5, 0.5],
                                 std=[0.5, 0.5]), 0., 1.)
                # DEFORM
                deform_bw_pred = deform2bw_tensor_batch(deform_map.clone(), device)
                deform_bw_pred_recover = torch.clamp(
                    re_normalize(deform_bw_pred, mean=[0.5, 0.5], std=[0.5, 0.5], inplace=False), 0., 1.)
                metric_deform_bw = metric_op(deform_bw_pred_recover.float().contiguous(),
                                             bw_gt_recover.float().contiguous())
                # UV2BW Dewarp ORI
                dewarp_ori_gt = metrics.bw_mapping4d(bw_gt, rgb, device)
                dewarp_ori_gt_recover = torch.clamp(
                    re_normalize(dewarp_ori_gt.transpose(2, 3).transpose(1, 2), mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5], inplace=False), 0., 1.)
                # UV2BW Dewarp ALB
                dewarp_alb_gt = metrics.bw_mapping4d(bw_gt, alb_map_gt, device)
                dewarp_alb_gt_recover = torch.clamp(re_normalize(dewarp_alb_gt, mean=[0.5], std=[0.5], inplace=False),
                                                    0., 1.)
                # DEFORM DEWARP ORI
                deform_dewarp_ori = metrics.bw_mapping4d(deform_bw_pred, rgb, device)
                deform_dewarp_ori_recover = torch.clamp(
                    re_normalize(deform_dewarp_ori.transpose(2, 3).transpose(1, 2), mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5],
                                 inplace=False), 0., 1.)
                metric_deform_ori = metric_op(deform_dewarp_ori_recover.contiguous(),
                                              dewarp_ori_gt_recover.contiguous())
                # DEFORM DEWARP ALB
                deform_dewarp_alb = metrics.bw_mapping4d(deform_bw_pred, alb_map, device)
                deform_dewarp_alb_recover = torch.clamp(
                    re_normalize(deform_dewarp_alb, mean=[0.5], std=[0.5], inplace=False), 0., 1.)
                metric_deform_alb = metric_op(torch.unsqueeze(deform_dewarp_alb_recover, 1).contiguous(),
                                              torch.unsqueeze(dewarp_alb_gt_recover, 1).contiguous())
                # SUMMARY
                metrics_alb = {key: metrics_alb[key] + metric_alb[key] for key in metrics_alb.keys()}
                metrics_cmap = {key: metrics_cmap[key] + metric_cmap[key] for key in metrics_cmap.keys()}
                metrics_dep = {key: metrics_dep[key] + metric_dep[key] for key in metrics_dep.keys()}
                metrics_nor = {key: metrics_nor[key] + metric_nor[key] for key in metrics_nor.keys()}
                metrics_deform_bw = {key: metrics_deform_bw[key] + metric_deform_bw[key] for key in
                                     metrics_deform_bw.keys()}
                metrics_deform_ori = {key: metrics_deform_ori[key] + metric_deform_ori[key] for key in
                                      metrics_deform_ori.keys()}
                metrics_deform_alb = {key: metrics_deform_alb[key] + metric_deform_alb[key] for key in
                                      metrics_deform_alb.keys()}
            if isWriteImage:
                if batch_idx == (len(test_loader.dataset) // args.test_batch_size) - 2:
                    if not os.path.exists(output_dir_test + 'test/epoch_{}_batch_{}'.format(epoch, batch_idx)):
                        os.makedirs(output_dir_test + 'test/epoch_{}_batch_{}'.format(epoch, batch_idx))
                    print('writing test image')
                    for k in range(args.test_batch_size):
                        alb_pred = alb_map[k, :, :, :]
                        mask_pred = mask_map[k, :, :, :]
                        mask_pred = torch.round(mask_pred)
                        cmap_pred = cmap[k, :, :, :]
                        dep_pred = dep_map[k, :, :, :]
                        nor_pred = nor_map[k, :, :, :]
                        deform_bw_map = deform2bw_tensor_batch(deform_map.clone().to(device), device)
                        deform_bw_pred = deform_bw_map[k, :, :, :]
                        name = names[k]
                        ori_gt = rgb[k, :, :, :]
                        alb_gt = alb_map_gt[k, :, :, :]
                        uv_gt = uv_map_gt[k, :, :, :]
                        mask_gt = mask_map_gt[k, :, :, :]
                        cmap_gt = cmap_map_gt[k, :, :, :]
                        dep_gt = dep_map_gt[k, :, :, :]
                        nor_gt = nor_map_gt[k, :, :, :]

                        bw_gt = metrics.uv2bmap(uv_gt, mask_gt)
                        dewarp_ori_gt = metrics.bw_mapping(bw_gt, ori_gt, device)
                        dewarp_alb_gt = metrics.bw_mapping(bw_gt, alb_gt, device)
                        deform_dewarp_ori = metrics.bw_mapping(deform_bw_pred, ori_gt, device)
                        deform_dewarp_alb = metrics.bw_mapping(deform_bw_pred, alb_pred, device)
                        output_dir = os.path.join(output_dir_test, 'test/epoch_{}_batch_{}/'.format(epoch, batch_idx))
                        output_mask_pred = os.path.join(output_dir, 'pred_mask_ind_{}'.format(name) + '.jpg')
                        output_alb_pred = os.path.join(output_dir, 'pred_alb_ind_{}'.format(name) + '.jpg')
                        output_cmap_pred = os.path.join(output_dir, 'pred_3D_ind_{}'.format(name) + '.jpg')
                        output_deform_bw_pred = os.path.join(output_dir, 'pred_deform_bw_ind_{}'.format(name) + '.exr')
                        output_dep_pred = os.path.join(output_dir, 'pred_dep_ind_{}'.format(name) + '.jpg')
                        output_nor_pred = os.path.join(output_dir, 'pred_normal_ind_{}'.format(name) + '.jpg')
                        output_ori = os.path.join(output_dir, 'gt_ori_ind_{}'.format(name) + '.jpg')
                        output_uv_gt = os.path.join(output_dir, 'gt_uv_ind_{}'.format(name) + '.exr')
                        output_alb_gt = os.path.join(output_dir, 'gt_alb_ind_{}'.format(name) + '.jpg')
                        output_cmap_gt = os.path.join(output_dir, 'gt_cmap_ind_{}'.format(name) + '.jpg')
                        output_mask_gt = os.path.join(output_dir, 'gt_mask_ind_{}'.format(name) + '.jpg')
                        output_bw_gt = os.path.join(output_dir, 'gt_bw_ind_{}'.format(name) + '.exr')
                        output_dewarp_ori_gt = os.path.join(output_dir, 'gt_dewarpOri_ind_{}'.format(name) + '.jpg')
                        output_dewarp_alb_gt = os.path.join(output_dir, 'gt_dewarpAlb_ind_{}'.format(name) + '.jpg')
                        output_dep_gt = os.path.join(output_dir, 'gt_dep_ind_{}'.format(name) + '.jpg')
                        output_nor_gt = os.path.join(output_dir, 'gt_nor_ind_{}'.format(name) + '.jpg')
                        output_deform_dewarp_ori = os.path.join(output_dir,
                                                                'deform_dewarp_ori_ind_{}'.format(name) + '.jpg')
                        output_deform_dewarp_alb = os.path.join(output_dir,
                                                                'deform_dewarp_alb_ind_{}'.format(name) + '.jpg')

                        """pred"""
                        write_image_tensor(mask_pred, output_mask_pred, '01')
                        write_image_tensor(alb_pred, output_alb_pred, 'std')
                        write_image_tensor(cmap_pred, output_cmap_pred, 'gauss', mean=[0.1108, 0.3160, 0.2859],
                                           std=[0.7065, 0.6840, 0.7141])
                        write_image_tensor(dep_pred, output_dep_pred, 'gauss', mean=[0.5], std=[0.5])
                        write_image_tensor(nor_pred, output_nor_pred, 'gauss', mean=[0.5619, 0.2881, 0.2917],
                                           std=[0.5619, 0.7108, 0.7083])
                        write_image_tensor(deform_bw_pred, output_deform_bw_pred, 'std', device=device)
                        """gt"""
                        write_image_tensor(ori_gt, output_ori, 'std')
                        write_image_tensor(uv_gt, output_uv_gt, 'std', device=device)
                        write_image_tensor(mask_gt, output_mask_gt, '01')
                        write_image_tensor(alb_gt, output_alb_gt, 'std')
                        write_image_tensor(cmap_gt, output_cmap_gt, 'gauss', mean=[0.1108, 0.3160, 0.2859],
                                           std=[0.7065, 0.6840, 0.7141])
                        write_image_tensor(dep_gt, output_dep_gt, 'gauss', mean=[0.5], std=[0.5])
                        write_image_tensor(nor_gt, output_nor_gt, 'gauss', mean=[0.5619, 0.2881, 0.2917],
                                           std=[0.5619, 0.7108, 0.7083])
                        write_image_np(bw_gt, output_bw_gt)

                        write_image(dewarp_ori_gt, output_dewarp_ori_gt)
                        write_image(dewarp_alb_gt, output_dewarp_alb_gt)
                        """dewarp"""
                        write_image(deform_dewarp_ori, output_deform_dewarp_ori)
                        write_image(deform_dewarp_alb, output_deform_dewarp_alb)
            if (batch_idx + 1) % 20 == 0:
                print('It cost {} seconds to test {} images'.format(time.time() - start_time,
                                                                    (batch_idx + 1) * args.test_batch_size))
                start_time = time.time()
    test_loss = loss_sum / (len(test_loader.dataset) / args.test_batch_size)
    test_loss_alb = loss_sum_alb / (len(test_loader.dataset) / args.test_batch_size)
    test_loss_threeD = loss_sum_cmap / (len(test_loader.dataset) / args.test_batch_size)
    test_loss_nor = loss_sum_nor / (len(test_loader.dataset) / args.test_batch_size)
    test_loss_dep = loss_sum_dep / (len(test_loader.dataset) / args.test_batch_size)
    test_loss_mask = loss_sum_mask / (len(test_loader.dataset) / args.test_batch_size)
    test_loss_smooth = loss_sum_smooth / (len(test_loader.dataset) / args.test_batch_size)
    test_dice = dice_sum_metric / (len(test_loader.dataset) / args.test_batch_size)
    print(
        'Epoch:{} \n batch index:{}/{}||loss:{:.6f}||alb:{:.6f}||threeD:{:.6f}||nor:{:.6f}||dep:{:.6f}||mask:{:.6f}||'
        'smooth:{:.6f}||dice:{:.6f}'.format(
            epoch,
            batch_idx + 1,
            len(
                test_loader.dataset) // args.batch_size,
            test_loss.item(),
            test_loss_alb.item(),
            test_loss_threeD.item(),
            test_loss_nor.item(),
            test_loss_dep.item(),
            test_loss_mask.item(),
            test_loss_smooth.item(),
            test_dice.item(),
            ))
    if args.calculate_CC:
        num_iters = math.ceil(len(test_loader.dataset) / args.test_batch_size)
        metrics_alb = {key: metrics_alb[key] / num_iters for key in metrics_alb.keys()}
        metrics_cmap = {key: metrics_cmap[key] / num_iters for key in metrics_cmap.keys()}
        metrics_dep = {key: metrics_dep[key] / num_iters for key in metrics_dep.keys()}
        metrics_nor = {key: metrics_nor[key] / num_iters for key in metrics_nor.keys()}
        metrics_deform_bw = {key: metrics_deform_bw[key] / num_iters for key in metrics_deform_bw.keys()}
        metrics_deform_ori = {key: metrics_deform_ori[key] / num_iters for key in metrics_deform_ori.keys()}
        metrics_deform_alb = {key: metrics_deform_alb[key] / num_iters for key in metrics_deform_alb.keys()}
    if args.calculate_CC:
        for key in metrics_alb.keys():
            print(
                'Metric:{}\t dep:{:.6f}\t nor:{:.6f}\t cmap:{:.6f}\t alb:{:.6f}\t deform_bw:{:.6f}\t'
                ' deform_ori:{:.6f}\t deform_alb:{:.6f}\n'.format(
                    key, metrics_dep[key].item(), metrics_nor[key].item(),
                    metrics_cmap[key].item(), metrics_alb[key].item(),
                    metrics_deform_bw[key].item(), metrics_deform_ori[key].item(), metrics_deform_alb[key].item()))
    if args.write_txt:
        txt_dir = 'output_txt/' + args.model_name + '.txt'
        f = open(txt_dir, 'a')
        f.write(
            'Epoch: {} \t Test Loss: {:.6f}, \t ab: {:.4f}, \t cmap: {:.4f}, \t normal: {:.4f}, \t depth: {:.4f}, \t mask: {:.6f} , '
            ' \t smooth:{:.6f},\t dice:{:.6f}\n'.format(epoch,
                                                                                                        test_loss.item(),
                                                                                                        test_loss_alb.item(),
                                                                                                        test_loss_threeD.item(),
                                                                                                        test_loss_nor.item(),
                                                                                                        test_loss_dep.item(),
                                                                                                        test_loss_mask.item(),
                                                                                                        test_loss_smooth.item(),
                                                                                                        test_dice.item()))
        if args.calculate_CC:
            for key in metrics_alb.keys():
                f.write(
                    'Metric:{}\t dep:{:.6f}\t nor:{:.6f}\t cmap:{:.6f}\t alb:{:.6f}\t deform_bw:{:.6f}\t'
                    ' deform_ori:{:.6f}\t deform_alb:{:.6f}\n'.format(key,
                                                                      metrics_dep[key].item(), metrics_nor[key].item(),
                                                                      metrics_cmap[key].item(), metrics_alb[key].item(),
                                                                      metrics_deform_bw[key].item(),
                                                                      metrics_deform_ori[key].item(),
                                                                      metrics_deform_alb[key].item()))
        f.close()
    if args.write_summary:
        print('sstep', sstep)
        # writer.add_scalar('test_acc', 100. * correct / len(test_loader.dataset), global_step=epoch+1)
        writer.add_scalar('summary/test_loss', test_loss.item(), global_step=sstep)
        writer.add_scalar('summary/test_loss_ab', test_loss_alb.item(), global_step=sstep)
        writer.add_scalar('summary/test_loss_cmap', test_loss_threeD.item(), global_step=sstep)
        writer.add_scalar('summary/test_loss_normal', test_loss_nor.item(), global_step=sstep)
        writer.add_scalar('summary/test_loss_depth', test_loss_dep.item(), global_step=sstep)
        writer.add_scalar('summary/test_loss_back', test_loss_mask.item(), global_step=sstep)
        writer.add_scalar('summary/test_loss_smooth', test_loss_smooth.item(), global_step=sstep)


def main():
    # Model Build
    model = entire_unwarp_models.UnwarpNet_deform_only(use_simple=False, use_constrain=True, combine_num=1)
    args = train_configs.args
    args.model_name = 'deform_only'
    args.output_dir = '/home1/qiyuanwang/film_code/deform_only/'
    args.save_model_dir = '/home1/qiyuanwang/film_code/deform_only_model/'
    args.pretrained_model_dir = '/home1/qiyuanwang/film_code/deform_only_model/deform_only_10.pkl'
    exist_or_make(args.save_model_dir)
    isTrain = True
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        print(" [*] Set cuda: True")
        model = torch.nn.DataParallel(model.cuda())
    else:
        print(" [*] Set cuda: False")
    # model = model.to(device)
    # Load Dataset
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    dataset_test = DeFilmDataset(npy_dir=args.test_path)
    dataset_test_loader = DataLoader(dataset_test,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     **kwargs)
    dataset_train = DeFilmDataset(npy_dir=args.train_path)
    dataset_train_loader = DataLoader(dataset_train, batch_size=args.batch_size,
                                      shuffle=True, **kwargs)
    start_epoch = 1
    learning_rate = args.lr
    # Load Parameters
    #if True:
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
    #learning_rate = start_lr
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    # model, optimizer = amp.initialize(model, optimizer,opt_level='O1',loss_scale="dynamic",verbosity=0)
    # model = torch.nn.DataParallel(model.cuda())
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
    #print('start_lr', start_lr)
    #print('start_epoch', start_epoch)
    for epoch in range(start_epoch, args.epochs + 1):
        if isTrain:
            lr = train(args, model, device, dataset_train_loader, optimizer, criterion, epoch, writer, args.output_dir,
                       args.write_image_train, isVal=False,
                       test_loader=dataset_test_loader)
            sstep = test.count + 1
            args.calculate_CC = False
            if epoch % 10 == 0:
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


if __name__ == '__main__':
    main()
