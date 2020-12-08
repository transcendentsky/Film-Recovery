import torch.nn as nn
from . import train_configs
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import os
import numpy as np
from tensorboardX import SummaryWriter
import math
# from apex.parallel import DistributedDataParallel as DDP
# from apex.fp16_utils import *
# from apex import amp, optimizers
from tutils import *
from models.misc.model_cmap import UnwarpNet_cmap
from dataloader.load_data_2 import filmDataset_3
from dataloader.print_img import print_img_auto, print_img_with_reprocess

# HyperParams for Scripts / Names
modelname = "new_data"
output_dir = tdir("output/train", "newdata"+generate_name())
writer = SummaryWriter(logdir=tdir(output_dir, "summary"))
max_epoch = 500

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return float(param_group['lr'])

def main():
    # -----------------------------------  Model Build -------------------------
    model = UnwarpNet_cmap(combine_num=1)
    args = train_configs.args
    isTrain = True
    model = torch.nn.DataParallel(model.cuda()) 
    start_epoch = 1
    # Load Parameters
    # if args.pretrained:
    if True:
        print("Loading Pretrained model~")
        #""/home1/quanquan/code/film_code/output/train/aug20201129-210822-VktsHX/cmap_aug_19.pkl""
        pretrained_dict = torch.load("/home1/quanquan/code/Film-Recovery/cmap_only_45.pkl", map_location=None)
        model.load_state_dict(pretrained_dict['model_state'])
        start_lr = pretrained_dict['lr']
        start_epoch = pretrained_dict['epoch']
    # ------------------------------------  Load Dataset  -------------------------
    kwargs = {'num_workers': 8, 'pin_memory': True} 
    # dataset_test = filmDataset_3(npy_dir="/home1/quanquan/datasets/generate/mesh_film_small/")
    # dataset_test_loader = DataLoader(dataset_test,batch_size=args.test_batch_size, shuffle=False, **kwargs)
    dataset_train = filmDataset_3("/home1/quanquan/datasets/generate/mesh_film_small_alpha/", load_mod="nobw")
    dataset_train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, **kwargs)
    
    # ------------------------------------  Optimizer  -------------------------
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    # model, optimizer = amp.initialize(model, optimizer,opt_level='O1',loss_scale="dynamic",verbosity=0)
    #criterion = torch.nn.MSELoss()  
    criterion = torch.nn.L1Loss()
    bc_critic = nn.BCELoss() 
    
    if args.visualize_para:
        for name, parameters in model.named_parameters():
            print(name, ':', parameters.size())
    start_lr = args.lr
    
    # -----------------------------------  Training  ---------------------------
    for epoch in range(start_epoch, max_epoch + 1):
        loss_value, loss_cmap_value, loss_ab_value, loss_uv_value = 0,0,0,0
        model.train()
        datalen = len(dataset_train)
        print("Output dir:", output_dir)
        for batch_idx, data in enumerate(dataset_train_loader):
            
            ori_gt = data[0].cuda()
            ab_gt  = data[1].cuda()
            dep_gt = data[2].cuda()
            nor_gt = data[3].cuda()
            cmap_gt= data[4].cuda()
            uv_gt  = data[5].cuda()
            bg_gt  = data[6].cuda()
            
            optimizer.zero_grad()
            uv, cmap, ab = model(ori_gt)               
            # print("ab shapes: ", ab.shape, ab_gt.shape)
            
            loss_cmap = criterion(cmap, cmap_gt).float()
            loss_ab = criterion(ab, ab_gt).float()
            loss_uv   = criterion(uv, uv_gt).float()
            loss = loss_uv + loss_ab + loss_cmap 
            loss.backward()
            optimizer.step()
            scheduler.step()
            print("\r Epoch[{}/{}] \t batch:{}/{} \t \t loss: {}".format(epoch, max_epoch, batch_idx,datalen, loss.item()), end=" ") 
            
            lr = get_lr(optimizer)
            # w("check code")
            # break
        
        writer_tb((loss_value/(batch_idx+1), loss_ab_value/(batch_idx+1), loss_uv_value/(batch_idx+1), loss_cmap_value/(batch_idx+1), lr), epoch=epoch)
        write_imgs_2((cmap[0,:,:,:], uv[0,:,:,:], ab[0,:,:,:], ori_gt[0,:,:,:], cmap_gt[0,:,:,:], uv_gt[0,:,:,:], ab_gt[0,:,:,:]), epoch)

        if isTrain and args.save_model:
            state = {'epoch': epoch + 1,
                     'lr': lr,
                     'model_state': model.state_dict(),
                     'optimizer_state': optimizer.state_dict()
                     }
            torch.save(state, tfilename(output_dir, "{}_{}.pkl".format("cmap_aug", epoch)))

def writer_tb(loss_tuple, epoch):
    loss, loss_ab, loss_uv, loss_cmap, lr = loss_tuple
    writer.add_scalar('summary/train_loss',        loss,      global_step=epoch)
    writer.add_scalar('summary/train_cmap_loss',   loss_cmap, global_step=epoch)
    writer.add_scalar('summary/train_uv_loss',     loss_uv,   global_step=epoch)
    # writer.add_scalar('summary/train_normal_loss', loss_nor,  global_step=epoch)
    # writer.add_scalar('summary/train_depth_loss',  loss_dep,  global_step=epoch)
    writer.add_scalar('summary/train_ab_loss',     loss_ab,   global_step=epoch)
    # writer.add_scalar('summary/train_back_loss',   loss_bg,   global_step=epoch)
    writer.add_scalar('summary/lrate',             lr,               global_step=epoch)


def write_imgs_2(img_tuple, epoch, type_tuple=None, name_tuple=None):
    cmap, uv, ab, \
        ori_gt, cmap_gt, uv_gt, ab_gt = img_tuple

    #print_img_auto(ori,  "ori",  fname=tfilename(output_dir,"imgshow", "ori.jpg"))
    print_img_with_reprocess(cmap, "cmap",  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "cmap.jpg"))
    print_img_with_reprocess(uv ,  "uv"  ,  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "uv.jpg"))
    # print_img_auto(bg ,  "bg"  ,  fname=tfilename(output_dir,"imgshow", "bg.jpg"))
    print_img_with_reprocess(ab ,  "ab"  ,  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "ab.jpg"))
    
    print_img_with_reprocess(ori_gt,  "ori" ,  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "ori_gt.jpg"))
    print_img_with_reprocess(cmap_gt, "cmap",  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "cmap_gt.jpg"))
    print_img_with_reprocess(uv_gt ,  "uv"  ,  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "uv_gt.jpg"))
    # print_img_auto(bg_gt ,  "bg"  ,  fname=tfilename(output_dir,"imgshow", "bg.jpg"))
    print_img_with_reprocess(ab_gt ,  "ab"  ,  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "ab_gt.jpg"))

# def write_imgs(img_tuple, name_tuple):
#     img_list = list(img_tuple)
#     name_list = list(name_tuple)
#     assert len(img_list) == len(name_list)
#     for i in range(len(img_list)):
#         print_img_auto(img_list[i], name_list[i], fname=tfilename(output_dir, name_list[i]+".jpg")

if __name__ == "__main__":
    main()


