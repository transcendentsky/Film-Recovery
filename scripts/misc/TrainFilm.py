from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
import os
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
#import load_data
from load_data_npy import filmDataset
from tensorboardX import SummaryWriter
import numpy as np
from network import Net             
# import torchsnooper
import cv2
from genDataNPY import repropocess
from scipy.interpolate import griddata
from write_image import write_image_tensor, write_image_np, write_image, write_image_01, write_image_np, write_cmap_gauss
import time
from cal_times import CallingCounter

# training or test
isTrain = False                          #"""""""""""""""""""""""""""

# setup dataloader
dr_dataset_train_1 = 'npy/'     #'Data_final/Part001/Mesh_Film/npy/'    # 2000
dr_dataset_train_2 = None       #'Data_final/Part003/Mesh_Film/npy/'    # 5429
dr_dataset_test = 'npy_test/'      #'Data_final/Part002/Mesh_Film/npy/'       #1389

# setup model
model_name = 'Try_0915'
preTrain = True                                  #""""""""""""""""""""""""""""

# optimizer
LearningRate = 0.001
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,5,6"

# output
write_summary = False
write_txt = True
write_image_train = True
write_image_val = False
write_image_test = True                    #""""""""""""""""""""""""""""
calculate_CC = False
summary_name = model_name
save_dir = 'model/'
output_dir ='/home1/share/film_output/' + model_name + '/'       #'output_image/'+ model_name + '/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(output_dir+'train/'):
    os.mkdir(output_dir+'train/')
if not os.path.exists(output_dir+'test/'):
    os.mkdir(output_dir+'test/')


pretrained_model_dir = '/home1/liuli/film_code/model/Model1_0908_0.0002_model.pkl'
#  /home1/liuli/film_code/model/Model3_0912_9.pkl                   
#  /home1/share/liuli/film_code/model/Model1_0908_0.0002_model.pkl  
#  /home1/share/liuli/film_code/model/Model5_0913_50.pkl            





# @torchsnooper.snoop()
def train(args, model, device, train_loader, optimizer, criterion, epoch, writer, output_dir, isWriteImage, isVal=False, test_loader=None):
    model.train()
    correct=0
    # print('begin')
    for batch_idx, data in enumerate(train_loader):

    #------------Setup data-----------#
        ori = data[0]       
        ab = data[1]        
        depth = data[2]     
        normal = data[3]    
        cmap = data[4]      
        uv = data[5]        
        background = data[6]    
        # ori_1080 = data[7]            
        # bmap = data[6]

        ori, ab, depth, normal, uv, cmap, back = ori.to(device), ab.to(device), depth.to(device), \
                                                 normal.to(device), uv.to(device), cmap.to(device), background.to(device) #bmap.to(device)
        optimizer.zero_grad()

        uv_map, coor_map, normal_map, albedo_map, depth_map, back_map = model(ori)

        # define loss
        loss_back = criterion(back_map, back).float()
        loss_cmap = criterion(coor_map, cmap).float()       # 3d map = coor_map = cmap
        loss_uv = criterion(uv_map, uv).float()
        loss_depth = criterion(depth_map, depth).float()
        loss_normal = criterion(normal_map, normal).float()
        # loss_bmap = criterion(bw_map, bmap).float()
        loss_ab = criterion(albedo_map, torch.unsqueeze(ab[:,0,:,:], 1).float())        

        loss = 4 * loss_uv + 4 * loss_ab + loss_normal + loss_depth + 2 * loss_back + loss_cmap

        loss.backward()
        optimizer.step()
        lrate = get_lr(optimizer)
        # print('0.2', loss)

        if batch_idx % args.log_interval == 0:
            # acc = 100 * correct/(data.size(1)* args.log_interval)
            print('Epoch: {} \nBatch index: {}/{}, \t Lr: {:.8f}, \t '
                  'Training Loss: {:.6f}, \t ab: {:.4f}, \t cmap: {:.4f}, \t uv: {:.6f}, \t normal: {:.4f}, \t depth: {:.4f}, \t back: {:.6f}'.format(
                epoch, batch_idx+1, len(train_loader.dataset)//args.batch_size, lrate, loss.item(),
                loss_ab.item(), loss_cmap.item(), loss_uv.item(), loss_normal.item(),  loss_depth.item(),  loss_back.item()
            ))

            if write_summary:
                # writer.add_scalar('summary/train_acc', acc, global_step=epoch*len(train_loader)+batch_idx+1)
                writer.add_scalar('summary/train_loss', loss.item(), global_step=epoch*len(train_loader)+batch_idx+1)
                writer.add_scalar('summary/train_cmap_loss', loss_cmap.item(), global_step=epoch * len(train_loader) + batch_idx + 1)
                writer.add_scalar('summary/train_uv_loss', loss_uv.item(),
                                  global_step=epoch * len(train_loader) + batch_idx + 1)
                writer.add_scalar('summary/train_normal_loss', loss_normal.item(),
                                  global_step=epoch * len(train_loader) + batch_idx + 1)
                writer.add_scalar('summary/train_depth_loss', loss_depth.item(),
                                  global_step=epoch * len(train_loader) + batch_idx + 1)
                writer.add_scalar('summary/train_ab_loss', loss_ab.item(),
                                  global_step=epoch * len(train_loader) + batch_idx + 1)
                writer.add_scalar('summary/train_back_loss', loss_back.item(),
                                  global_step=epoch * len(train_loader) + batch_idx + 1)
                writer.add_scalar('summary/lrate', lrate, global_step=epoch * len(train_loader) + batch_idx + 1)
            # acc = 0
            # correct = 0

        if isWriteImage:
            if batch_idx == len(train_loader.dataset)//args.batch_size:
                print('writting image')
                if not os.path.exists(output_dir + 'train/epoch_{}'.format(epoch)):
                    os.mkdir(output_dir + 'train/epoch_{}'.format(epoch))
                for k in range(5):     

                    albedo_pred = albedo_map[k, :, :, :]
                    uv_pred = uv_map[k, :, :, :]
                    back_pred = back_map[k, :, :, :]

                    ori_gt = ori[k, :, :, :]
                    ab_gt = ab[k, :, :, :]
                    uv_gt = uv[k, :, :, :]
                    back_gt = back[k, :, :, :]
                    bw_gt = uv2bmap(uv_gt, back_gt)
                    bw_pred = uv2bmap(uv_pred, back_pred)

                    # bw_gt = bmap[k, :, :, :]


                    dewarp_ori = bw_mapping(bw_pred, ori_gt, device)
                    dewarp_ab = bw_mapping(bw_pred, ab_gt, device)
                    dewarp_ori_gt = bw_mapping(bw_gt, ori_gt, device)

                    cmap_gt = cmap[k, :, :, :]
                    cmap_pred = coor_map[k, :, :, :]

                    # bw_gt = bw_gt.transpose(0, 1).transpose(1, 2)
                    # bw_pred = bw_pred.transpose(0, 1).transpose(1, 2)
                    bb = (-1) * torch.ones((256, 256, 1)).to(device)
                    bb_numpy = (-1) * np.ones((256, 256, 1))
                    """pred"""

                    write_image_np(np.concatenate((bw_pred, bb_numpy), 2),
                                output_dir + 'train/epoch_{}/pred_bw_ind_{}'.format(epoch, k) + '.jpg')

                    write_image(torch.cat([uv_pred.transpose(0, 1).transpose(1, 2), bb], 2),
                                output_dir + 'train/epoch_{}/pred_uv_ind_{}'.format(epoch, k) + '.jpg')
                    write_image_01(back_pred.transpose(0, 1).transpose(1, 2)[:, :, 0],
                                output_dir + 'train/epoch_{}/pred_back_ind_{}'.format(epoch, k) + '.jpg')
                    write_image(albedo_pred.transpose(0, 1).transpose(1, 2)[:,:,0],
                                output_dir + 'train/epoch_{}/pred_ab_ind_{}'.format(epoch, k) + '.jpg')
                    write_cmap_gauss(cmap_pred.transpose(0, 1).transpose(1, 2),
                                     output_dir + 'train/epoch_{}/pred_3D_ind_{}'.format(epoch, k) + '.jpg')

                    """gt"""
                    write_image(ori_gt.transpose(0, 1).transpose(1, 2),
                                output_dir + 'train/epoch_{}/gt_ori_ind_{}'.format(epoch, k) + '.jpg')
                    write_image(ab_gt.transpose(0, 1).transpose(1, 2)[:,:,0],
                                output_dir + 'train/epoch_{}/gt_ab_ind_{}'.format(epoch, k) + '.jpg')
                    write_cmap_gauss(cmap_gt.transpose(0, 1).transpose(1, 2),
                                     output_dir + 'train/epoch_{}/gt_3D_ind_{}'.format(epoch, k) + '.jpg')
                    write_image_np(np.concatenate((bw_gt, bb_numpy), 2),
                                output_dir + 'train/epoch_{}/gt_bw_ind_{}'.format(epoch, k) + '.jpg')
                    write_image(torch.cat([uv_gt.transpose(0, 1).transpose(1, 2), bb], 2),
                                output_dir + 'train/epoch_{}/gt_uv_ind_{}'.format(epoch, k) + '.jpg')
                    write_image_01(back_gt.transpose(0, 1).transpose(1, 2)[:,:,0],
                                output_dir + 'train/epoch_{}/gt_back_ind_{}'.format(epoch, k) + '.jpg')
                    write_image(dewarp_ori_gt,
                                output_dir + 'train/epoch_{}/gt_dewarpOri_ind_{}'.format(epoch, k) + '.jpg')

                    """dewarp"""
                    write_image(dewarp_ori, output_dir + 'train/epoch_{}/dewarp_ori_ind_{}'.format(epoch, k) + '.jpg')
                    write_image(dewarp_ab, output_dir + 'train/epoch_{}/dewarp_ab_ind_{}'.format(epoch, k) + '.jpg')
        if isVal and (batch_idx+1) % 100 == 0:
            sstep = test.count +1
            test(args, model, device, test_loader, criterion, epoch, writer, output_dir, write_image_val, sstep)

    return lrate

@CallingCounter
def test(args, model, device, test_loader, criterion, epoch, writer, output_dir, isWriteImage, sstep):
    print('Testing')
    # print('len(test_loader.dataset)', len(test_loader.dataset))
    model.eval()        # without batchNorm and dropout
    test_loss = 0
    correct = 0
    cc_uv=0
    cc_cmap=0
    cc_ab=0
    cc_bw = 0
    cc_dewarp_ori =0
    cc_dewarp_ab = 0

    with torch.no_grad():
        # for data in test_loader:
        loss_sum =0
        loss_sum_ab = 0
        loss_sum_cmap = 0
        loss_sum_uv = 0
        loss_sum_normal = 0
        loss_sum_depth = 0
        loss_sum_back = 0

        print(len(test_loader))
        start_time=time.time()
        for batch_idx, data in enumerate(test_loader):
            time0 = time.time()
            # print(test_loader)
            ori = data[0]
            ab = data[1]
            depth = data[2]
            normal = data[3]
            cmap = data[4]
            uv = data[5]
            background = data[6]
            # ori_1080 = data[7]

            ori, ab, depth, normal, uv, cmap, back = ori.to(device), ab.to(device), depth.to(device), \
                                                     normal.to(device), uv.to(device), cmap.to(device), background.to(
                device)  # bmap.to(device)
            uv_map, coor_map, normal_map, albedo_map, depth_map, back_map = model(ori)

            loss_back = criterion(back_map, back).float()
            loss_cmap = criterion(coor_map, cmap).float()
            loss_uv = criterion(uv_map, uv).float()
            loss_depth = criterion(depth_map, depth).float()
            loss_normal = criterion(normal_map, normal).float()
            # loss_bmap = criterion(bw_map, bmap).float()
            loss_ab = criterion(albedo_map, torch.unsqueeze(ab[:, 0, :, :], 1).float())

            test_loss = 4 * loss_uv + 4 *  loss_ab + loss_normal + loss_depth + 2* loss_back + loss_cmap      # + loss_bmap
            loss_sum = loss_sum + test_loss
            loss_sum_ab += loss_ab
            loss_sum_cmap += loss_cmap
            loss_sum_uv += loss_uv
            loss_sum_normal += loss_normal
            loss_sum_depth += loss_depth
            loss_sum_back += loss_back

            if calculate_CC:
                c_ab = cal_CC(albedo_map, torch.unsqueeze(ab[:, 0, :, :], 1))
                c_uv = cal_CC(uv_map, uv)
                c_cmap = cal_CC(coor_map, cmap)

                bw_pred = uv2bmap4d(uv_map, back_map)
                bw_gt = uv2bmap4d(uv, back)             # [b, h, w, 2]
                c_bw = cal_CC_np(bw_pred, bw_gt)

                """计算dewarp"""
                dewarp_ori = bw_mapping4d(bw_pred, ori, device)
                dewarp_ori_gt = bw_mapping4d(bw_gt, ori, device)
                c_dewarp_ori = cal_CC(dewarp_ori, dewarp_ori_gt)

                # print('c_dewarp_ori', c_dewarp_ori)

                dewarp_ab = bw_mapping4d(bw_pred, albedo_map, device)
                dewarp_ab_gt = bw_mapping4d(bw_gt, torch.unsqueeze(ab[:, 0, :, :], 1), device)

                c_dewarp_ab = cal_CC_ab(dewarp_ab, dewarp_ab_gt)

                cc_ab += c_ab
                cc_uv += c_uv
                cc_cmap += c_cmap
                cc_bw += c_bw
                cc_dewarp_ori += c_dewarp_ori
                cc_dewarp_ab += c_dewarp_ab

            # print('Epoch: {} \n'
            #       'Test Loss: {:.6f}, \t ab: {:.4f}, \t cmap: {:.4f}, \t uv: {:.4f}, \t normal: {:.4f}, \t depth: {:.4f}, \t back: {:.4f}'.format(
            #     epoch, test_loss.item(),
            #     loss_ab.item(), loss_cmap.item(), loss_uv.item(), loss_normal.item(),
            #     loss_depth.item(), loss_back.item()
            # ))
            # #print('CC_uv: {}\t CC_cmap: {}\t CC_ab: {}\t CC_bw: {}'.format(c_uv, c_cmap, c_ab, c_bw))
            # print('CC_uv: {}\t CC_cmap: {}\t CC_ab: {}'.format(c_uv, c_cmap, c_ab))
            # print(time.time() - time0)

            if isWriteImage:
                if True:        # batch_idx == 0:       write all the test images
                    if not os.path.exists(output_dir + 'test/epoch_{}_batch_{}'.format(epoch, batch_idx)):
                        os.mkdir(output_dir + 'test/epoch_{}_batch_{}'.format(epoch, batch_idx))
                    print('writting image')
                    for k in range(args.test_batch_size):
                        # print('k', k)
                        albedo_pred = albedo_map[k, :, :, :]
                        uv_pred = uv_map[k, :, :, :]
                        back_pred = back_map[k, :, :, :]
                        cmap_pred = coor_map[k, :, :, :]
                        depth_pred = depth_map[k, :, :, :]
                        normal_pred = normal_map[k, :, :, :]

                        ori_gt = ori[k, :, :, :]
                        ab_gt = ab[k, :, :, :]
                        uv_gt = uv[k, :, :, :]
                        back_gt = back[k, :, :, :]
                        cmap_gt = cmap[k, :, :, :]
                        depth_gt = depth[k, :, :, :]
                        normal_gt = normal[k, :, :, :]

                        bw_gt = uv2bmap(uv_gt, back_gt)
                        bw_pred = uv2bmap(uv_pred, back_pred)  # [-1,1], [256, 256, 3]

                        # bw_gt = bmap[k, :, :, :]

                        dewarp_ori = bw_mapping(bw_pred, ori_gt, device)
                        dewarp_ab = bw_mapping(bw_pred, ab_gt, device)
                        dewarp_ori_gt = bw_mapping(bw_gt, ori_gt, device)


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
                        write_image_tensor(cmap_pred, output_3d_pred, 'gauss', mean=[0.100, 0.326, 0.289], std=[0.096, 0.332, 0.298])
                        write_image_tensor(depth_pred, output_depth_pred, 'gauss', mean=[0.316], std=[0.309])
                        write_image_tensor(normal_pred, output_normal_pred, 'gauss', mean=[0.584, 0.294, 0.300], std=[0.483, 0.251, 0.256])
                        write_image_np(bw_pred, output_bw_pred)
                        """gt"""
                        write_image_tensor(ori_gt, output_ori, 'std')
                        write_image_tensor(uv_gt, output_uv_gt, 'std', device=device)
                        write_image_tensor(back_gt, output_back_gt, '01')
                        write_image_tensor(ab_gt, output_ab_gt, 'std')
                        write_image_tensor(cmap_gt, output_cmap_gt, 'gauss', mean=[0.100, 0.326, 0.289], std=[0.096, 0.332, 0.298])
                        write_image_tensor(depth_gt, output_depth_gt, 'gauss', mean=[0.316], std=[0.309])
                        write_image_tensor(normal_gt, output_normal_gt, 'gauss', mean=[0.584, 0.294, 0.300], std=[0.483, 0.251, 0.256])

                        write_image_np(bw_gt, output_bw_gt)

                        write_image(dewarp_ori_gt, output_dewarp_ori_gt)

                        """dewarp"""
                        write_image(dewarp_ori, output_dewarp_ori)
                        write_image(dewarp_ab, output_dewarp_ab)

            if (batch_idx+1) % 20 ==0:
                print('It cost {} seconds to test {} images'.format(time.time()-start_time, (batch_idx+1)*args.test_batch_size))
                start_time = time.time()

    test_loss = loss_sum /(len(test_loader.dataset)/args.test_batch_size)
    test_loss_ab = loss_sum_ab / (len(test_loader.dataset) / args.test_batch_size)
    test_loss_cmap = loss_sum_cmap / (len(test_loader.dataset) / args.test_batch_size)
    test_loss_uv = loss_sum_uv / (len(test_loader.dataset) / args.test_batch_size)
    test_loss_normal = loss_sum_normal / (len(test_loader.dataset) / args.test_batch_size)
    test_loss_depth = loss_sum_depth / (len(test_loader.dataset) / args.test_batch_size)
    test_loss_back = loss_sum_back / (len(test_loader.dataset) / args.test_batch_size)
    if calculate_CC:
        cc_uv = cc_uv / (len(test_loader.dataset)/args.test_batch_size)
        cc_cmap = cc_cmap / (len(test_loader.dataset) / args.test_batch_size)
        cc_ab = cc_ab / (len(test_loader.dataset) / args.test_batch_size)
        cc_bw = cc_bw / (len(test_loader.dataset) / args.test_batch_size)
        cc_dewarp_ori = cc_dewarp_ori / (len(test_loader.dataset) / args.test_batch_size)
        cc_dewarp_ab = cc_dewarp_ab / (len(test_loader.dataset) / args.test_batch_size)


    print('Epoch: {} \n'
          'Test Loss: {:.6f}, \t ab: {:.4f}, \t cmap: {:.4f}, \t uv: {:.6f}, \t normal: {:.4f}, \t depth: {:.4f}, \t back: {:.6f}'.format(
        epoch, test_loss,
        test_loss_ab.item(), test_loss_cmap.item(), test_loss_uv.item(), test_loss_normal.item(), test_loss_depth.item(), test_loss_back.item()
    ))
    if calculate_CC:
        print('CC_uv: {}\t CC_cmap: {}\t CC_ab: {}\t CC_bw: {}\t CC_dewarp_ori: {}\t CC_dewarp_ab: {}'.format(cc_uv, cc_cmap, cc_ab, cc_bw, cc_dewarp_ori, cc_dewarp_ab))
    if write_txt:
        txt_dir = 'output_txt/' + model_name + '.txt'
        f = open(txt_dir,'a')
        f.write('Epoch: {} \t Test Loss: {:.6f}, \t ab: {:.4f}, \t cmap: {:.4f}, \t uv: {:.6f}, \t normal: {:.4f}, \t depth: {:.4f}, \t back: {:.6f} CC_uv: {}\t CC_cmap: {}\t CC_ab: {}\t CC_bw: {}\t CC_dewarp_ori: {}\t CC_dewarp_ab: {}\n'.format(
        epoch, test_loss,
        test_loss_ab.item(), test_loss_cmap.item(), test_loss_uv.item(), test_loss_normal.item(), test_loss_depth.item(), test_loss_back.item(), cc_uv, cc_cmap, cc_ab, cc_bw, cc_dewarp_ori, cc_dewarp_ab))
        f.close()

    if write_summary:
        print('sstep', sstep)
        # writer.add_scalar('test_acc', 100. * correct / len(test_loader.dataset), global_step=epoch+1)
        writer.add_scalar('summary/test_loss', test_loss.item(), global_step=sstep)
        writer.add_scalar('summary/test_loss_ab', test_loss_ab.item(), global_step=sstep)
        writer.add_scalar('summary/test_loss_cmap', test_loss_cmap.item(), global_step=sstep)
        writer.add_scalar('summary/test_loss_uv', test_loss_uv.item(), global_step=sstep)
        writer.add_scalar('summary/test_loss_normal', test_loss_normal.item(), global_step=sstep)
        writer.add_scalar('summary/test_loss_depth', test_loss_depth.item(), global_step=sstep)
        writer.add_scalar('summary/test_loss_back', test_loss_back.item(), global_step=sstep)



def cal_CC(pred, GT):           
    """
    calculate CC
    """
    # input tensor [B, C, H, W]
    pccs=0
    pred = pred.detach().cpu().numpy()
    GT = GT.detach().cpu().numpy()
    b= pred.shape[0]
    for batch in range(b):
        pred_b = pred[batch, :, :, :].reshape(-1)
        GT_b = GT[batch, :, :, :].reshape(-1)
        # print('pred_b', pred_b)
        # print('GT_b', GT_b)
        # print(pred_b.max(), pred_b.min())
        # print(GT_b.max(), GT_b.min())
        cc = np.corrcoef(pred_b, GT_b)[0,1]
        # print('cc',cc)
        pccs += cc
    # print('pccs',pccs)
    # print('b',b)
    return pccs/b


def cal_CC_ab(pred, GT):        
    # input tensor [B, C, H, W]
    pccs=0
    pred = pred.detach().cpu().numpy()
    GT = GT.detach().cpu().numpy()
    b= pred.shape[0]
    for batch in range(b):
        pred_b = pred[batch, :, :].reshape(-1)
        GT_b = GT[batch, :, :].reshape(-1)
        # print('pred_b', pred_b)
        # print('GT_b', GT_b)
        # print(pred_b.max(), pred_b.min())
        # print(GT_b.max(), GT_b.min())
        cc = np.corrcoef(pred_b, GT_b)[0,1]         
        # print('cc',cc)
        pccs += cc
    # print('pccs',pccs)
    # print('b',b)
    return pccs/b

def cal_CC_np(pred, GT):
    # input numpy [B, H, W, C]
    pccs=0
    b, h, w, c = pred.shape
    for batch in range(b):
        pred_b = pred[batch, :, :, :].reshape(-1)
        GT_b = GT[batch, :, :, :].reshape(-1)
        pccs += np.corrcoef(pred_b, GT_b)[0,1]
    return pccs/b



def uv2bmap(uv, background):            
    uv = uv.detach().cpu().numpy()
    background = background.detach().cpu().numpy()
    img_bgr = (uv + 1) / 2  # [c h w]
    img_rgb = img_bgr[::-1, :, :]
    img_rgb[1, :, :] = 1 - img_rgb[1, :, :]
    s_x = (img_rgb[0, :, :] * 256)
    s_y = (img_rgb[1, :, :] * 256)
    mask = background[0, :, :] > 0  #0.6

    s_x = s_x[mask]
    s_y = s_y[mask]
    index = np.argwhere(mask)
    t_y = index[:, 0]
    t_x = index[:, 1]
    x = np.arange(256)
    y = np.arange(256)
    xi, yi = np.meshgrid(x, y)
    # zz = np.zeros((256, 256))
    zx = griddata((s_x, s_y), t_x, (xi, yi), method='linear')
    zy = griddata((s_x, s_y), t_y, (xi, yi), method='linear')
    # backward_img = np.stack([zy, zx, zz], axis=2)
    backward_img = np.stack([zy, zx], axis=2)
    backward_img[np.isnan(backward_img)] = 0
    backward_img = (backward_img/ 256)*2 -1
    #    np.save('C:/tmp/'+uv_path.split('/')[-1].split('.')[0]+'_backward',backward_img)
    #    cv2.imwrite('C:/tmp/'+uv_path.split('/')[-1].split('.')[0]+'_backward.png',backward_img*255)
    return backward_img


def uv2bmap4d(uv, background):

    """input: [batch, channel, h ,w]"""
    """output: numpy"""
    batch = uv.size()[0]
    uv = uv.detach().cpu().numpy()
    background = background.detach().cpu().numpy()
    output = np.zeros(shape=(0, 256, 256, 2))
    for c in range(batch):
        img_bgr = (uv[c, :, :, :] + 1) / 2  # [c h w]
        img_rgb = img_bgr[::-1, :, :]
        img_rgb[1, :, :] = 1 - img_rgb[1, :, :]
        s_x = (img_rgb[0, :, :] * 256)
        s_y = (img_rgb[1, :, :] * 256)
        mask = background[c, 0, :, :] > 0  #0.6

        s_x = s_x[mask]
        s_y = s_y[mask]
        index = np.argwhere(mask)
        t_y = index[:, 0]
        t_x = index[:, 1]
        x = np.arange(256)
        y = np.arange(256)
        xi, yi = np.meshgrid(x, y)
        zx = griddata((s_x, s_y), t_x, (xi, yi), method='linear')
        zy = griddata((s_x, s_y), t_y, (xi, yi), method='linear')
        backward_img = np.stack([zy, zx], axis=2)
        backward_img[np.isnan(backward_img)] = 0
        backward_img = (backward_img/ 256) *2-1            # [h, w, 2]
        backward_img = np.expand_dims(backward_img, axis=0)
        output = np.concatenate((output, backward_img), 0)

    return output


def bw_mapping(bw_map, image, device):


    image = torch.unsqueeze(image, 0)       #[1, 3, 256, 256]
    image_t = image.transpose(2,3)     
    # bw
    # from [h, w, 2]
    # to  4D tensor [-1, 1] [b, h, w, 2]
    bw_map = torch.from_numpy(bw_map).type(torch.float32).to(device)            
    bw_map = torch.unsqueeze(bw_map, 0)
    # bw_map = bw_map.transpose(1, 2).transpose(2, 3)
    output = F.grid_sample(input=image, grid=bw_map)
    output_t = F.grid_sample(input=image_t, grid=bw_map)            
    output = output.transpose(1, 2).transpose(2, 3)
    output = output.squeeze()
    output_t = output_t.transpose(1, 2).transpose(2, 3)
    output_t = output_t.squeeze()
    return output_t#.transpose(1,2).transpose(0,1)          


def bw_mapping4d(bw_map, image, device):

    """image"""       #[batch, 3, 256, 256]
    image_t = image.transpose(2,3)     
    # bw
    # from [h, w, 2]
    # to  4D tensor [-1, 1] [b, h, w, 2]
    bw_map = torch.from_numpy(bw_map).type(torch.float32).to(device)
    # bw_map = torch.unsqueeze(bw_map, 0)
    # bw_map = bw_map.transpose(1, 2).transpose(2, 3)
    output = F.grid_sample(input=image, grid=bw_map)
    output_t = F.grid_sample(input=image_t, grid=bw_map)
    output = output.transpose(1, 2).transpose(2, 3)
    output = output.squeeze()
    output_t = output_t.transpose(1, 2).transpose(2, 3)
    output_t = output_t.squeeze()
    return output_t#.transpose(1,2).transpose(0,1)          

# def write_image(image_float, dir):
#     image_uint8 = ((image_float+1)/2 *255).type(torch.uint8).cpu().numpy()
#     cv2.imwrite(dir, image_uint8)
#
# def write_image_np(image_float, dir):
#     image_uint8 = ((image_float+1)/2 *255).astype(np.uint8)
#     cv2.imwrite(dir, image_uint8)
#
# def write_cmap_gauss(image_float, dir, mean=[0.100, 0.326, 0.289], std=[0.096, 0.332, 0.298]):
#     image_float = repropocess(image_float.detach().cpu().numpy(), mean, std)
#     image_uint8 = (image_float *255).astype(np.uint8)
#     cv2.imwrite(dir, image_uint8)
#
# def write_image_01(image_float, dir):
#     image_uint8 = (image_float *255).type(torch.uint8).cpu().numpy()
#     cv2.imwrite(dir, image_uint8)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return float(param_group['lr'])

def main():
    # Training settings
    # global sstep
    sstep = 0
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',       # 50 for 4 gpu
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',     # 100 for 4 gpu
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=LearningRate, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.85, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save_model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--visualize_para', action='store_true', default=True,
                        help='For visualizing the Model parameters')
    parser.add_argument('--pretrained',  action='store_true', default=preTrain,
                        help='Load model parameters from pretrained model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    dataset_test = filmDataset(npy_dir=dr_dataset_test)
    dataset_test_loader = DataLoader(dataset_test,
                                     batch_size=args.test_batch_size,
                                     # num_workers=1,
                                     shuffle=False,
                                     **kwargs)
    dataset_train = filmDataset(npy_dir=dr_dataset_train_1, npy_dir_2=dr_dataset_train_2)
    dataset_train_loader = DataLoader(dataset_train,
                                     batch_size=args.batch_size,
                                     # num_workers=1,
                                     shuffle=True,
                                     **kwargs)

    # model = Net().to(device)
    model = Net()
    model = torch.nn.DataParallel(model, device_ids=[0])
    model = model.to(device)

    start_epoch = 1
    start_lr = args.lr
    args.pretrained = False
    if args.pretrained:
        # pretrained_dict = torch.load(pretrained_model_dir, map_location=None)
        # model_dict = model.state_dict()
        # # 1. filter out unnecessary keys
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # # 2. overwrite entries in the existing state dict
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)

        pretrained_dict = torch.load(pretrained_model_dir, map_location=None)
        model.load_state_dict(pretrained_dict['model_state'])
        start_lr = pretrained_dict['lr']
        start_epoch = pretrained_dict['epoch']

    # start_lr = 0.00005


    optimizer = optim.Adam(model.parameters(), lr=start_lr)
        # Adadelta(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    if args.visualize_para:
        for name, parameters in model.named_parameters():
            print(name, ':', parameters.size())

    if write_summary:
        if not os.path.exists('summary/' + summary_name +'_start_epoch{}'.format(start_epoch)):
            os.mkdir('summary/' + summary_name+'_start_epoch{}'.format(start_epoch))

        writer = SummaryWriter(logdir='summary/' + summary_name+'_start_epoch{}'.format(start_epoch))
        print(summary_name)
    else:
        writer = 0

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5,eta_min=4e-08)

    print('start_lr', start_lr)
    print('start_epoch', start_epoch)
    isTrain = True
    write_image_train = False
    write_image_test = False
    """start training/ test"""
    for epoch in range(start_epoch, args.epochs + 1):

        if isTrain:
            lr = train(args, model, device, dataset_train_loader, optimizer, criterion, epoch, writer, output_dir,
                       write_image_train, isVal=True, test_loader=dataset_test_loader)
            sstep = test.count + 1
            test(args, model, device, dataset_test_loader, criterion, epoch, writer, output_dir, write_image_test,
                 sstep)
        else:
            sstep = test.count +1
            test(args, model, device, dataset_test_loader, criterion, epoch, writer, output_dir, write_image_test, sstep)
            break

        # if epoch % 2 ==0:
        scheduler.step()            # change lr with gamma decay


        if isTrain and args.save_model:
            state ={'epoch': epoch+1,   #  saving the next epoch
                    'lr': lr,           #  saving the lr of next epoch
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()
            }
            torch.save(state, save_dir+"{}_{}.pkl".format(model_name, epoch))


def exist_or_make(path):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == '__main__':
    main()

    # ckpt = torch.load('model/' + pretrain_name + '/model_' + str(pretrain_epoch) + '.pth')
    # model_dict = model.state_dict()
    # restore_dict = {}
    # for (k, v) in ckpt.items():
    #     if k in model_dict:
    #         restore_dict[k] = v
    # model_dict.update(restore_dict)
    # model.load_state_dict(model_dict)