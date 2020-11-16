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
import metrics
import numpy as np
from write_image import *
from tensorboardX import SummaryWriter
import modules
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
model_name = 'n2d'
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return float(param_group['lr'])


def train(args, model, device, train_loader, optimizer, criterion, epoch, writer, output_dir, isWriteImage, isVal=False,
          test_loader=None):
    model.train()
    correct = 0
    for batch_idx, data in enumerate(train_loader):
        dep_map_gt = data[2]
        nor_map_gt = data[3]
        dep_map_gt, nor_map_gt = dep_map_gt.to(device), nor_map_gt.to(device)
        optimizer.zero_grad()
        dep_map = model(nor_map_gt)
        loss = criterion(dep_map, dep_map_gt).float()
        loss.backward()
        optimizer.step()
        lr = get_lr(optimizer)
        if batch_idx % args.log_intervals == 0:
            print(
                'Epoch:{} \n batch index:{}/{}||lr:{:.8f}||loss:{:.6f}'.format(epoch, batch_idx + 1,
                                                                                           len(
                                                                                               train_loader.dataset) // args.batch_size,
                                                                                           lr, loss.item()))
            if args.write_summary:
                writer.add_scalar('summary/train_loss', loss.item(),
                                  global_step=epoch * len(train_loader) + batch_idx + 1)
                writer.add_scalar('summary/lrate', lr, global_step=epoch * len(train_loader) + batch_idx + 1)
        if isWriteImage:
            if batch_idx == (len(train_loader.dataset) // args.batch_size)-1:
                print('writing image')
                if not os.path.exists(output_dir + 'train/epoch_{}_batch_{}'.format(epoch, batch_idx)):
                    os.makedirs(output_dir + 'train/epoch_{}_batch_{}'.format(epoch, batch_idx))
                for k in range(5):
                    """gt"""
                    output_dir1 = output_dir + 'train/epoch_{}_batch_{}/'.format(epoch, batch_idx)
                    output_depth_pred = output_dir1 + 'pred_nor_ind_{}'.format(k) + '.jpg'
                    output_depth_gt = output_dir1 + 'gt_nor_ind_{}'.format(k) + '.jpg'
                    write_image_tensor(dep_map[k, :, :, :], output_depth_pred, 'gauss', mean=[0.316], std=[0.309])
                    write_image_tensor(dep_map_gt[k, :, :, :], output_depth_gt, 'gauss', mean=[0.316], std=[0.309])
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
    cc_dep = 0
    with torch.no_grad():
        loss_sum = 0
        start_time = time.time()
        for batch_idx, data in enumerate(test_loader):
            dep_map_gt = data[2]
            nor_map_gt = data[3]
            dep_map_gt, nor_map_gt = dep_map_gt.to(device), nor_map_gt.to(device)
            dep_map = model(nor_map_gt)
            test_loss = criterion(dep_map, dep_map_gt).float()
            loss_sum = loss_sum + test_loss
            if args.calculate_CC:
                c_dep = metrics.calculate_CC_metrics(dep_map, dep_map_gt)
                cc_dep += c_dep
            if isWriteImage and batch_idx == (len(test_loader.dataset) // args.test_batch_size)-1:
                if True:
                    if not os.path.exists(output_dir + 'test/epoch_{}_batch_{}'.format(epoch, batch_idx)):
                        os.makedirs(output_dir + 'test/epoch_{}_batch_{}'.format(epoch, batch_idx))
                    print('writing test image')
                    for k in range(args.test_batch_size):
                        output_dir1 = output_dir + 'test/epoch_{}_batch_{}/'.format(epoch, batch_idx)
                        output_depth_pred = output_dir1 + 'pred_nor_ind_{}'.format(k) + '.jpg'
                        output_depth_gt = output_dir1 + 'gt_nor_ind_{}'.format(k) + '.jpg'
                        write_image_tensor(dep_map[k, :, :, :], output_depth_pred, 'gauss', mean=[0.316], std=[0.309])
                        write_image_tensor(dep_map_gt[k, :, :, :], output_depth_gt, 'gauss', mean=[0.316], std=[0.309])
            if (batch_idx + 1) % 20 == 0:
                print('It cost {} seconds to test {} images'.format(time.time() - start_time,
                                                                    (batch_idx + 1) * args.test_batch_size))
                start_time = time.time()
    test_loss = loss_sum / (len(test_loader.dataset) / args.test_batch_size)
    print(
        'Epoch:{} \n batch index:{}/{}||loss:{:.6f}'.format(epoch, batch_idx + 1,
                                                                   len(test_loader.dataset) // args.batch_size,
                                                                   test_loss.item()))
    if args.calculate_CC:
        cc_dep = cc_dep / (len(test_loader.dataset) / args.test_batch_size)
    if args.calculate_CC:
        print('CC_nor: {}\t '.format(cc_dep))

    if args.write_txt:
        txt_dir = 'output_txt/' + model_name + '.txt'
        f = open(txt_dir, 'a')
        f.write(
            'Epoch: {} \t Test Loss: {:.6f}\t Test CC: {:.6f}\n'.format(
                epoch, test_loss.item(), cc_dep))
        f.close()
    if args.write_summary:
        print('sstep', sstep)
        # writer.add_scalar('test_acc', 100. * correct / len(test_loader.dataset), global_step=epoch+1)
        writer.add_scalar('summary/test_loss', test_loss.item(), global_step=sstep)

def main():
    # Model Build
    model = modules.UNetDepth()
    args = train_configs.args
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
    #if args.pretrained:
    if True:
        # pretrained_dict = torch.load(pretrained_model_dir, map_location=None)
        # model_dict = model.state_dict()
        # # 1. filter out unnecessary keys
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # # 2. overwrite entries in the existing state dict
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)
        pretrained_dict = torch.load('model/n2d_model/n2d_5.pkl', map_location=None)
        model.load_state_dict(pretrained_dict['model_state'])
        start_lr = pretrained_dict['lr']
        start_epoch = pretrained_dict['epoch']
    # Add Optimizer
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    if args.use_mse:
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.L1Loss()
    if args.visualize_para:
        for name, parameters in model.named_parameters():
            print(name, ':', parameters.size())
    if args.write_summary:
        if not os.path.exists('summary/' + model_name + '_start_epoch{}'.format(start_epoch)):
            os.makedirs('summary/' + model_name + '_start_epoch{}'.format(start_epoch))

        writer = SummaryWriter(logdir='summary/' + model_name + '_start_epoch{}'.format(start_epoch))
    else:
        writer = 0
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    #start_lr = args.lr
    print('start_lr', start_lr)
    output_dir = 'n2d_result/'
    for epoch in range(start_epoch, args.epochs + 1):
        if isTrain:
            lr = train(args, model, device, dataset_train_loader, optimizer, criterion, epoch, writer, output_dir,
                       args.write_image_train, isVal=False,
                       test_loader=dataset_test_loader)
            sstep = test.count + 1
            test(args, model, device, dataset_test_loader, criterion, epoch, writer, output_dir,
                 args.write_image_test, sstep)
        else:
            sstep = test.count + 1
            test(args, model, device, dataset_test_loader, criterion, epoch, writer, output_dir,
                 args.write_image_test, sstep)
            break
        scheduler.step()
        t2d_save_model = 'model/n2d_model/'
        exist_or_make(t2d_save_model)
        if isTrain and t2d_save_model:
            state = {'epoch': epoch + 1,
                     'lr': lr,
                     'model_state': model.state_dict(),
                     'optimizer_state': optimizer.state_dict()
                     }
            torch.save(state, t2d_save_model + "{}_{}.pkl".format(model_name, epoch))


def exist_or_make(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    main()
