import torch.nn as nn
import unwarp_models
from torch.utils.data import Dataset, DataLoader
# from load_data_npy import filmDataset, single_test
from scripts.misc import train_configs
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import models.misc.models as models
import os
import numpy as np
from dataloader.bw2deform import deform2bw_tensor_batch
from tensorboardX import SummaryWriter
import math

from dataloader.data_process import reprocess_auto_batch, reprocess_t2t_auto, reprocess_auto, process_auto, process_to_tensor
from dataloader.bw_mapping import bw_mapping_batch_3, bw_mapping_tensor_batch, bw_mapping_single_3
from dataloader.uv2bw import uv2backward_trans_3
from dataloader.print_img import print_img_auto
from dataloader.myblur import resize_albedo_np, blur_bw_np, resize_albedo_np2

from utils.tutils import *

constrain_path = {
    ('threeD', 'normal'): (False, True, ''),
    ('threeD', 'depth'): (False, True, ''),
    ('normal', 'depth'): (False, True, ''),
    ('depth', 'normal'): (False, True, ''),

}

def test_single(model, imgpath, writer):
    test_name = generate_name()

    img_ori = cv2.imread(imgpath)
    parent, imgname = os.path.split(imgpath)
    img = cv2.resize(img_ori, (256,256))
    _input = process_auto(img, "ori")
    img_tensor = process_to_tensor(_input[np.newaxis,:,:,:])
    img_tensor = img_tensor.cuda()
    uv_map, cmap, nor_map, alb_map, dep_map, mask_map, \
               _, _, _, _, _, deform_map = model(img_tensor)

    alb_pred = alb_map[0, :, :, :]
    uv_pred = uv_map[0, :, :, :]
    mask_pred = mask_map[0, :, :, :]
    mask_pred = torch.round(mask_pred)
    cmap_pred = cmap[0, :, :, :]
    dep_pred = dep_map[0, :, :, :]
    nor_pred = nor_map[0, :, :, :]
    deform_bw_map = deform2bw_tensor_batch(deform_map.detach().cpu())
    deform_bw_pred = deform_bw_map[0, :, :, :]

    uv_np = reprocess_auto(uv_pred, "uv")
    mask_np = reprocess_auto(mask_pred, "background")
    alb_np = reprocess_auto(alb_pred, "ab")
    # cmap_np = reprocess_auto(cmap_pred, "cmap")

    bw_np = uv2backward_trans_3(uv_np, mask_np)
    
    dewarp_np = bw_mapping_single_3(img, bw_np)
    
    bw_large = blur_bw_np(bw_np, img_ori)
    alb_large, diff, ori_gray, img_gray = resize_albedo_np2(img_ori, img, alb_np)

    dewarp_ab_large = bw_mapping_single_3(alb_large, bw_large)
    dewarp_large    = bw_mapping_single_3(img_ori  , bw_large)
    print("shape: ", dewarp_ab_large.shape)

    print_img_auto(ori_gray , "ori" , fname=tfilename("output_test_single/", "real-img-"+test_name, imgname+"_ori_gray.jpg"))
    print_img_auto(img_gray , "ori" , fname=tfilename("output_test_single/", "real-img-"+test_name, imgname+"_img_gray.jpg"))
    print_img_auto(alb_np   , "ab" , fname=tfilename("output_test_single/", "real-img-"+test_name, imgname+"_ab_np.jpg"))
    print_img_auto(uv_np    , "uv" , fname=tfilename("output_test_single/", "real-img-"+test_name, imgname+"_uv_np.jpg"))
    print_img_auto(bw_np    , "bw" , fname=tfilename("output_test_single/", "real-img-"+test_name, imgname+"_bw_np.jpg"))
    print_img_auto(img_ori  , "ori", fname=tfilename("output_test_single/", "real-img-"+test_name, imgname+"_imgori.jpg"))
    print_img_auto(img      , "ori", fname=tfilename("output_test_single/", "real-img-"+test_name, imgname+"_ori.jpg"))
    print_img_auto(dewarp_np, "ori", fname=tfilename("output_test_single/", "real-img-"+test_name, imgname+"_dewarp_ori.jpg"))
    print_img_auto(dewarp_ab_large, "ab", fname=tfilename("output_test_single/", "real-img-"+test_name, imgname+"_large_ab.jpg"))
    print_img_auto(dewarp_large, "ori", fname=tfilename("output_test_single/", "real-img-"+test_name, imgname+"_large_ori.jpg"))
    print_img_auto(alb_large, "ab", fname=tfilename("output_test_single/", "real-img-"+test_name, imgname+"_al_large.jpg"))
    print_img_auto(diff, "ab", fname=tfilename("output_test_single/", "real-img-"+test_name, imgname+"_diff_large.jpg"))

    # img[:,:,0] = 0
    # print_img_auto(img, "ori", fname=tfilename("output_test_single/", "real-img-"+test_name, imgname+"_ori1.jpg"))
    # img[:,:,1] = 0
    # print_img_auto(img, "ori", fname=tfilename("output_test_single/", "real-img-"+test_name, imgname+"_ori2.jpg"))

    
def test_uv():
    parent = "/home1/qiyuanwang/film_code/eval/deform_unwarp/pred_uv_ind_"
    im1 = "CT500-CT412_12_6-4-2wXldE3caZ0005.exr"
    im2 = "CT500-CT420_1_4-4-svQ0rgJUEu0032.exr"
    im3 = "CT500-CT424_10_6-4-NDz7niawME0010.exr"
    im4 = "CT500-CT426_4_4-4-8pCH2dKNvn0031.exr"

    for im in [im1, im2, im3, im4]:
        print(im)
        uv = cv2.imread(parent+im, cv2.IMREAD_UNCHANGED)
        print(uv.shape)
        print_img_auto(uv, "uv", fname=tfilename("uvtest", im[:-4]+".jpg"))

def main():
    # Model Build
    model = unwarp_models.UnwarpNet(use_simple=False, use_constrain=True, combine_num=1,
                                   constrain_configure=constrain_path)
    # model = models.UnwarpNet_cmap(use_simple=False, combine_num=1)
    args = train_configs.args
    args.test_batch_size = 1
    args.batch_size = 1
    args.model_name = 'eval_deform'
    #args.output_dir = '/home1/qiyuanwang/film_code/eval/real_data'
    #args.output_dir = '/home1/qiyuanwang/film_code/eval/black_pad_data'
    #args.output_dir = '/home1/qiyuanwang/film_code/eval/cmap_display'
    args.pretrained_model_dir = '/home1/qiyuanwang/film_code/model_summary/deform_model/deform_unwarp_100.pkl'
    # args.pretrained_model_dir = 'model_summary/model_cmap_only/cmap_only_40.pkl'
    # args.save_model_dir = '/home1/qiyuanwang/film_code/deform_model/'
    # exist_or_make(args.output_dir)
    isTrain = True
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        print(" [*] Set cuda: True")
        model = torch.nn.DataParallel(model.cuda())
        # model = model.cuda()
    else:
        print(" [*] Set cuda: False")
    # model = model.to(device)
    # Load Dataset
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    #dataset_test = single_test(npy_dir='/home1/qiyuanwang/film_code/real_data_img/')
    #dataset_test = single_test(npy_dir='/home1/qiyuanwang/film_code/black_pad_img/')
    # dataset_test = single_test(npy_dir='/home1/qiyuanwang/film_generate/npy_test_with_bw/')
    # dataset_test_loader = DataLoader(dataset_test,
    #                                  batch_size=args.test_batch_size,
    #                                  shuffle=False,
    #                                  **kwargs)
    if True:
        pretrained_dict = torch.load(args.pretrained_model_dir, map_location=None)
        model.load_state_dict(pretrained_dict['model_state'])
    if args.use_mse:
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.L1Loss()
    # if args.visualize_para:
    #     for name, parameters in model.named_parameters():
    #         print(name, ':', parameters.size())
    args.write_summary = False
    if args.write_summary:
        if not os.path.exists('summary/' + args.model_name + '_start_epoch{}'.format(1)):
            os.makedirs('summary/' + args.model_name + '_start_epoch{}'.format(1))

        writer = SummaryWriter(logdir='summary/' + args.model_name + '_start_epoch{}'.format(1))
        print(args.model_name)
    else:
        writer = 0
    #test_deform(args, model, device, dataset_test_loader, criterion, writer, args.output_dir,
    #     args.write_image_test)
    #test_single(args, model, device, dataset_test_loader, criterion, writer, args.output_dir,
    #    args.write_image_test)
    # datadir = ""
    # for filename in os.scan(datadir):
    #     imgpath = os.path.join(datadir, filename)
        
    test_img = '/home1/qiyuanwang/film_code/real_data_img/real_film_1.jpg'
    test_single(model, test_img, writer)
        


if __name__ == '__main__':
    main()
