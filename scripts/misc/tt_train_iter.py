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
from tutils import *
from models.misc.model_cmap import UnwarpNet_cmap, UnwarpNet
from models.misc.deform_model import DeformNet, construct_plain_bg, construct_plain_cmap
from dataloader.load_data_2 import filmDataset_3, RealDataset
from dataloader.print_img import print_img_auto, print_img_with_reprocess
from dataloader.data_process import reprocess_np_auto, reprocess_auto
from dataloader.uv2bw import uv2backward_trans_3
from dataloader.bw_mapping import bw_mapping_single_3
from models.misc.loss import tv_loss, TVLoss
from tqdm import tqdm
from dataloader.iter_mapping import iter_mapping

# HyperParams for Scripts / Names
modelname = "iter-7-0.01"
if not train_configs.args.test: 
    random_name = generate_name()
    output_dir = tdir("output/train", modelname+random_name)
    writer = SummaryWriter(logdir=tdir(output_dir, "summary"))
    output_dir_eval = tdir("output/eval", modelname+random_name)
else: 
    output_dir_test = tdir("output/test", modelname+generate_name())
    writer = SummaryWriter(logdir=tdir(output_dir_test, "summary"))
    output_dir = output_dir_test
    
max_epoch = 500

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return float(param_group['lr'])

def main():
    # -----------------------------------  Model Build -------------------------
    # model  = UnwarpNet(combine_num=1)
    model2 = DeformNet()
    args = train_configs.args
    isTrain = True
    # model  = torch.nn.DataParallel(model.cuda()) 
    model2 = torch.nn.DataParallel(model2.cuda()) 
    start_epoch = 1
    # Load Parameters
    # if args.pretrained:
    if True:
        print("Loading Pretrained model~")
        # "/home1/quanquan/code/Film-Recovery/output/train/std120201220-014112-xA836H/cmap_aug_500.pkl"
        # "/home1/quanquan/code/Film-Recovery/output/train/extrabg20201223-025124-sJKxHA/model/extrabg_310.pkl"
        pretrained_dict = torch.load("/home1/quanquan/code/Film-Recovery/output/train/iter20201229-084255-1htiye/model/iter_5.pkl", map_location=None)
        start_lr = pretrained_dict['lr']
        start_epoch = pretrained_dict['epoch'] if pretrained_dict['epoch'] < 100 else 100
        # -----------------------  Load partial model  ---------------------
        model_dict=model2.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict['model_state'].items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # -------------------------------------------------------------------
        # model.load_state_dict(pretrained_dict['model_state'])
        model2.load_state_dict(model_dict)
    # ------------------------------------  Load Dataset  -------------------------
    kwargs = {'num_workers': 8, 'pin_memory': True} 
    # dataset_test = filmDataset_3(npy_dir="/home1/quanquan/datasets/generate/mesh_film_small/")
    # dataset_test_loader = DataLoader(dataset_test,batch_size=args.test_batch_size, shuffle=False, **kwargs)
    dataset_eval = RealDataset("imgshow_test2", load_mod="new_ab", reg_start="pad_gaus_40", reg_end="jpg") # reg_str="pad_gaus_40"
    dataset_eval_loader = DataLoader(dataset_eval, batch_size=1, shuffle=False, **kwargs)
    dataset_train = filmDataset_3("/home1/quanquan/datasets/generate/mesh_film_hypo_alpha2/", load_mod="extra_bg")
    dataset_train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, **kwargs)
    
    # ------------------------------------  Optimizer  -------------------------
    optimizer = optim.Adam(model2.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=2, gamma=args.gamma)
    #criterion = torch.nn.MSELoss()  
    criterion = torch.nn.L1Loss()
    bc_critic = nn.BCELoss() 
    # tv_loss = tv_loss
    
    if args.visualize_para:
        for name, parameters in model.named_parameters():
            print(name, ':', parameters.size())
    start_lr = args.lr
    global_step = 0
    
    # -----------------------------------  Training  ---------------------------
    for epoch in range(start_epoch, max_epoch + 1):
        model2.train()
        loss_value, loss_cmap_value, loss_ab_value, loss_uv_value, loss_bg_value = 0,0,0,0,0
        loss_nor_value, loss_dep_value = 0,0
        loss_bg_t_value, loss_cmap_t_value, loss_deform_value,loss_tv_value = 0,0,0,0
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
            
            deform, _ = model2(ori_gt)      
            loss_tv = tv_loss(deform, 0.01)
            bg_template, pad_bg = construct_plain_bg(ori_gt.size(0),img_size=256)
            cmap_template, pad_cmap = construct_plain_cmap(ori_gt.size(0), img_size=256)
            dewarp_bg_t   = iter_mapping(bg_template  , deform)
            dewarp_cmap_t = iter_mapping(cmap_template, deform)
            loss_bg_t   = criterion(dewarp_bg_t, bg_gt)
            loss_cmap_t = criterion(dewarp_cmap_t, cmap_gt)
            loss_deform = loss_bg_t + loss_cmap_t + loss_tv
            loss_deform.backward()
            loss_deform_value += loss_deform
            loss_bg_t_value   += loss_bg_t
            loss_cmap_t_value += loss_cmap_t_value
            loss_tv_value += loss_tv
            optimizer.step()
            # global_step += 1
            lr = get_lr(optimizer)
            writer.add_scalar('summary/lrate_batch', lr, global_step=global_step)
            print("Epoch[\t{}/{}] \t batch:\t{}/{} \t lr:{} \t loss: {}".format(epoch, max_epoch, batch_idx,datalen,lr, loss_deform_value/(batch_idx+1)), end=" ") 
            print(f"loss_t: {loss_tv_value/(batch_idx+1)}, ")
            
            # w("check code")
            # break
        
        # ------ Scheduler Step -------
        # scheduler.step()
        writer.add_scalar('summary/loss_bg_t'  , loss_bg_t_value/(batch_idx+1)  , global_step=epoch)
        writer.add_scalar('summary/loss_cmap_t', loss_cmap_t_value/(batch_idx+1), global_step=epoch)     
        writer.add_scalar('summary/loss_tv', loss_tv_value/(batch_idx+1), global_step=epoch)
        print_img_with_reprocess(dewarp_bg_t[0,:,:,:]  , "bg"  , fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "bg.jpg"))
        print_img_with_reprocess(dewarp_cmap_t[0,:,:,:], "cmap", fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "cmap.jpg"))
        print_img_with_reprocess(ori_gt[0,:,:,:]       , "ori" ,  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "ori_gt.jpg")) 
        print_img_with_reprocess(bg_gt[0,:,:,:]        ,  "bg" ,  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "bg_gt.jpg"))
        print_img_with_reprocess(cmap_gt[0,:,:,:]      ,  "exr",  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "cmap_gt.jpg")) #

        if isTrain and args.save_model and epoch %5 == 0:
            state = {'epoch': epoch + 1,
                     'lr': lr,
                     'model_state': model2.state_dict(),
                     'optimizer_state': optimizer.state_dict()
                     }
            torch.save(state, tfilename(output_dir, "model", "{}_{}.pkl".format(modelname, epoch)))
        
        # -----------  Evaluation  -------------
        if True: 
            model2.eval()
            p(output_dir_eval)
            for batch_idx, data in tqdm(enumerate(dataset_eval_loader)):
                ori_gt = data[0].cuda()
                ori_gt_large = data[1].cuda()
                
                deform, _ = model2(ori_gt)               
                bg_template, pad_bg = construct_plain_bg(ori_gt.size(0),img_size=256)
                cmap_template, pad_cmap = construct_plain_cmap(ori_gt.size(0), img_size=256)
                dewarp_bg_t   = iter_mapping(bg_template  , deform)
                dewarp_cmap_t = iter_mapping(cmap_template, deform)
                
                print_img_with_reprocess(dewarp_bg_t[0,:,:,:]  , "bg"  , fname=tfilename(output_dir_eval,"imgshow/epoch_{}".format(batch_idx), "bg.jpg"))
                print_img_with_reprocess(dewarp_cmap_t[0,:,:,:], "cmap", fname=tfilename(output_dir_eval,"imgshow/epoch_{}".format(batch_idx), "cmap.jpg"))
                print_img_with_reprocess(ori_gt[0,:,:,:]       , "ori" ,  fname=tfilename(output_dir_eval,"imgshow/epoch_{}".format(batch_idx), "ori_gt.jpg")) 
                
                if batch_idx >25:
                    break

def writer_tb(loss_tuple, epoch):
    loss, loss_ab, loss_uv, \
        loss_cmap, loss_nor, loss_dep, \
            loss_bg, lr = loss_tuple
    writer.add_scalar('summary/train_loss',        loss,      global_step=epoch)
    writer.add_scalar('summary/train_cmap_loss',   loss_cmap, global_step=epoch)
    writer.add_scalar('summary/train_uv_loss',     loss_uv,   global_step=epoch)
    writer.add_scalar('summary/train_normal_loss', loss_nor,  global_step=epoch)
    writer.add_scalar('summary/train_depth_loss',  loss_dep,  global_step=epoch)
    writer.add_scalar('summary/train_ab_loss',     loss_ab,   global_step=epoch)
    writer.add_scalar('summary/train_bg_loss',     loss_bg,   global_step=epoch)
    writer.add_scalar('summary/lrate',             lr,        global_step=epoch)


def gt_clip(img):
    img = torch.clamp(img, min=-1, max=1)
    return img

def write_imgs_2(img_tuple, epoch, type_tuple=None, name_tuple=None, training=True):
    print("Writing Images to ", output_dir)
    if training:
        cmap, uv, ab, bg, nor, dep, bg2, ori_gt,\
            cmap_gt, uv_gt, ab_gt, bg_gt, nor_gt, dep_gt = img_tuple
    else:
        cmap, uv, ab, bg, nor, dep, bg2, ori_gt = img_tuple
    
    print_img_with_reprocess(uv ,  "uv"  ,  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "uv.jpg"))
    print_img_with_reprocess(ab ,  "ab"  ,  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "ab.jpg"))
    print_img_with_reprocess(bg ,  "bg"  ,  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "bg.jpg"))
    print_img_with_reprocess(bg2,  "bg"  ,  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "bg2.jpg"))
    #reprocess_np_auto(cmap, "")
    print_img_with_reprocess(cmap,  "exr"  ,  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "cmap.jpg")) #
    print_img_with_reprocess(nor ,  "exr"  ,  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "nor.jpg")) #
    print_img_with_reprocess(dep ,  "exr"  ,  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "dep.jpg")) #
    print_img_with_reprocess(ori_gt,  "ori" ,  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "ori_gt.jpg")) 
    
    if training:
        print_img_with_reprocess(uv_gt ,  "uv"  ,  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "uv_gt.jpg"))
        print_img_with_reprocess(ab_gt ,  "ab"  ,  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "ab_gt.jpg"))
        print_img_with_reprocess(bg_gt ,  "bg"  ,  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "bg_gt.jpg"))
        
        print_img_with_reprocess(cmap_gt, "exr" ,  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "cmap_gt.jpg")) #
        print_img_with_reprocess(nor_gt , "exr" ,  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "nor_gt.jpg")) #
        print_img_with_reprocess(dep_gt , "exr" ,  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "dep_gt.jpg")) #
        print_img_with_reprocess(gt_clip(cmap_gt),  "exr"  ,  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "cmap_gt2.jpg")) #
        print_img_with_reprocess(gt_clip(nor_gt ),  "exr"  ,  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "nor_gt2.jpg")) #
        print_img_with_reprocess(gt_clip(dep_gt ),  "exr"  ,  fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "dep_gt2.jpg")) #
    
    uv = reprocess_auto(uv, "uv")
    bg2 = reprocess_auto(bg2, "bg")
    ori_gt = reprocess_auto(ori_gt, "ori")
    bw = uv2backward_trans_3(uv, bg2)
    dewarp = bw_mapping_single_3(ori_gt, bw)
    
    if training:
        uv_gt = reprocess_auto(uv_gt, "uv")
        bg_gt = reprocess_auto(bg_gt, "bg")
        bw_gt = uv2backward_trans_3(uv_gt, bg_gt)
        bw2 = uv2backward_trans_3(uv, bg_gt)
        dewarp_gt = bw_mapping_single_3(ori_gt, bw_gt)
        dewarp2 = bw_mapping_single_3(ori_gt, bw2)
    
    print_img_auto(bw,   "bw", fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "bw.jpg"))
    print_img_auto(dewarp,   "ori", fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "dewarp.jpg"))
    
    if training:
        print_img_auto(bw_gt,"bw", fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "bw_gt.jpg"))
        print_img_auto(bw2,  "bw", fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "bw2.jpg"))
        print_img_auto(dewarp_gt,"ori", fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "dewarp_gt.jpg"))
        print_img_auto(dewarp2,  "ori", fname=tfilename(output_dir,"imgshow/epoch_{}".format(epoch), "dewarp2.jpg"))
    

def test():
    # -----------------------------------  Model Build -------------------------
    from tqdm import tqdm
    model = DeformNet()
    args = train_configs.args
    model = torch.nn.DataParallel(model.cuda()) 
    start_epoch = 1
    if True:
        print("Loading Pretrained model~") 
        # "/home1/quanquan/code/Film-Recovery/output/train/std120201220-014112-xA836H/cmap_aug_500.pkl"
        # "/home1/quanquan/code/Film-Recovery/output/train/extrabg20201223-025124-sJKxHA/model/extrabg_190.pkl"
        # "/home1/quanquan/code/Film-Recovery/output/train/extrabg_ab20201224-230459-AvKPR7/model/extrabg_ab_455.pkl"
        # "/home1/quanquan/code/Film-Recovery/output/train/iter20201229-221658-CxBn85/model/iter_175.pkl"
        pretrained_dict = torch.load("/home1/quanquan/code/Film-Recovery/output/train/iter-7-0.0120201230-032707-wyKn9z/model/iter-7-0.01_390.pkl", map_location=None)
        model.load_state_dict(pretrained_dict['model_state'])
    # ------------------------------------  Load Dataset  -------------------------
    kwargs = {'num_workers': 8, 'pin_memory': True} 
    # "/home1/quanquan/datasets/real_films/pad_img"
    # "/home1/quanquan/datasets/generate/mesh_film_hypo_alpha2/img"
    # "imgshow_test2"
    dataset_test = RealDataset("imgshow_test2", load_mod="new_ab", reg_start="pad_gaus_40", reg_end="jpg") # reg_str="pad_gaus_40"
    dataset_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, **kwargs)
    model.eval()
    
    p(output_dir_test)
    for batch_idx, data in tqdm(enumerate(dataset_loader)):
        ori_gt = data[0].cuda()
        ori_gt_large = data[1].cuda()
        
        deform, _ = model(ori_gt)
        
        bg_template, pad_bg = construct_plain_bg(ori_gt.size(0),img_size=256)
        cmap_template, pad_cmap = construct_plain_cmap(ori_gt.size(0), img_size=256)
        dewarp_bg_t   = iter_mapping(bg_template  , deform)
        dewarp_cmap_t = iter_mapping(cmap_template, deform)
        
        print_img_with_reprocess(dewarp_bg_t[0,:,:,:]  , "bg"  , fname=tfilename(output_dir_test,"imgshow/epoch_{}".format(batch_idx), "bg.jpg"))
        print_img_with_reprocess(dewarp_cmap_t[0,:,:,:], "cmap", fname=tfilename(output_dir_test,"imgshow/epoch_{}".format(batch_idx), "cmap.jpg"))
        print_img_with_reprocess(ori_gt[0,:,:,:]       , "ori" ,  fname=tfilename(output_dir_test,"imgshow/epoch_{}".format(batch_idx), "ori_gt.jpg")) 
        
        if batch_idx >25:
            break
        

if __name__ == "__main__":
    
    if train_configs.args.test:
        print("Starting Testing Stage")
        test()
    else:
        print("Starting Training Stage")
        main()
        

