import numpy as np
import torch as torch
import numpy as np
import torch as torch
from scipy.interpolate import griddata
import torch.nn.functional as F
import torch.nn as nn
import audtorch
import piq
def calculate_CC_metrics(pred, gt):
    """
    Calculate CC Metrics
    :param pred:
    :param gt:
    :return:
    """
    if isinstance(pred, np.ndarray) and isinstance(gt, np.ndarray):
        pccs = 0
        for i in range(pred.shape[0]):
            if len(pred.shape) == 3:
                pred_i = pred[i, :, :].reshape(-1)
                gt_i = gt[i, :, :].reshape(-1)
            else:
                pred_i = pred[i, :, :, :].reshape(-1)
                gt_i = gt[i, :, :, :].reshape(-1)
            cc = np.corrcoef(pred_i, gt_i)[0, 1]
            pccs += cc
        result = pccs / pred.shape[0]
    else:
        pred = pred.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()
        pccs = 0
        for i in range(pred.shape[0]):
            if len(pred.shape) == 3:
                pred_i = pred[i, :, :].reshape(-1)
                gt_i = gt[i, :, :].reshape(-1)
            else:
                pred_i = pred[i, :, :, :].reshape(-1)
                gt_i = gt[i, :, :, :].reshape(-1)
            cc = np.corrcoef(pred_i, gt_i)[0, 1]
            pccs += cc
        result = pccs / pred.shape[0]
    return result


def uv2bmap(uv, background):
    uv = uv.detach().cpu().numpy()
    background = background.detach().cpu().numpy()
    img_bgr = (uv + 1) / 2  # [c h w]
    img_rgb = img_bgr[::-1, :, :]
    img_rgb[1, :, :] = 1 - img_rgb[1, :, :]
    s_x = (img_rgb[0, :, :] * 256)
    s_y = (img_rgb[1, :, :] * 256)
    mask = background[0, :, :] > 0
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
    backward_img = (backward_img / 256) * 2 - 1
    #    np.save('C:/tmp/'+uv_path.split('/')[-1].split('.')[0]+'_backward',backward_img)
    #    cv2.imwrite('C:/tmp/'+uv_path.split('/')[-1].split('.')[0]+'_backward.png',backward_img*255)
    return backward_img


def uv2bmap4d(uv, background):
    """input: [batch, channel, h ,w]"""
    """output: numpy"""
    batch = uv.size()[0]
    uv = uv.detach().cpu().numpy()
    background = background.detach().cpu().numpy()
    output = np.zeros(shape=(0, 256, 256, 2),dtype=np.float32)
    for c in range(batch):
        img_bgr = (uv[c, :, :, :] + 1.) / 2.  # [c h w]
        img_rgb = img_bgr[::-1, :, :]
        img_rgb[1, :, :] = 1 - img_rgb[1, :, :]
        s_x = (img_rgb[0, :, :] * 256)
        s_y = (img_rgb[1, :, :] * 256)
        mask = background[c, 0, :, :] > 0.1  # 0.6

        s_x = s_x[mask]
        s_y = s_y[mask]
        index = np.argwhere(mask)
        t_y = index[:, 0]
        t_x = index[:, 1]
        x = np.arange(256)
        y = np.arange(256)
        xi, yi = np.meshgrid(x, y)
        zx = griddata((s_x, s_y), t_x, (xi, yi), method='linear',fill_value=0)
        zy = griddata((s_x, s_y), t_y, (xi, yi), method='linear',fill_value=0)
        backward_img = np.stack([zy, zx], axis=2)
        backward_img = (backward_img / 256.) * 2. - 1.  # [h, w, 2]
        backward_img = np.expand_dims(backward_img, axis=0)
        output = np.concatenate((output, backward_img), 0)
    return output


def bw_mapping(bw_map, image, device):
    image = torch.unsqueeze(image, 0)  # [1, 3, 256, 256]
    image_t = image.transpose(2, 3)  # b c h w
    # bw
    # from [h, w, 2]
    # to  4D tensor [-1, 1] [b, h, w, 2]
    bw_map = torch.from_numpy(bw_map).type(torch.float32).to(device)  # numpy to tensor
    bw_map = torch.unsqueeze(bw_map, 0)
    # bw_map = bw_map.transpose(1, 2).transpose(2, 3)
    output = F.grid_sample(input=image, grid=bw_map, align_corners=True)
    output_t = F.grid_sample(input=image_t, grid=bw_map, align_corners=True)  # tensor
    output = output.transpose(1, 2).transpose(2, 3)
    output = output.squeeze()
    output_t = output_t.transpose(1, 2).transpose(2, 3)
    output_t = output_t.squeeze()
    return output_t  # transpose(1,2).transpose(0,1)
    # ensure output [c, h, w]


def bw_mapping4d(bw_map, image, device):
    """image"""  # [batch, 3, 256, 256]
    image_t = image.transpose(2, 3)  # b c h w
    # bw
    # from [h, w, 2]
    # to  4D tensor [-1, 1] [b, h, w, 2]
    bw_map = torch.from_numpy(bw_map).type(torch.float32).to(device)
    # bw_map = torch.unsqueeze(bw_map, 0)
    # bw_map = bw_map.transpose(1, 2).transpose(2, 3)
    output = F.grid_sample(input=image, grid=bw_map, align_corners=True)
    output_t = F.grid_sample(input=image_t, grid=bw_map, align_corners=True)
    output = output.transpose(1, 2).transpose(2, 3)
    output = output.squeeze()
    output_t = output_t.transpose(1, 2).transpose(2, 3)
    output_t = output_t.squeeze()
    return output_t
    # transpose(1,2).transpose(0,1)
    # ensure output [c, h, w]
    
class film_metrics(nn.Module):
    def __init__(self, l1_norm=True, mse=True, pearsonr=True, cc=True, psnr=True, ssim=True, mssim=True):
        super(film_metrics, self).__init__()
        self.l1_norm = l1_norm
        self.mse = mse
        self.pearsonr = pearsonr
        self.cc = cc
        self.psnr = psnr
        self.ssim  =ssim
        self.mssim = mssim

    def forward(self, predict, target):
        if self.l1_norm:
            l1_norm_metric = nn.functional.l1_loss(predict, target)
        if self.mse:
            mse_norm_metric = nn.functional.mse_loss(predict, target)
        if self.pearsonr:
            pearsonr_metric = audtorch.metrics.functional.pearsonr(predict, target).mean()
        if self.cc:
            cc_metric = audtorch.metrics.functional.concordance_cc(predict, target).mean()
        if self.psnr:
            psnr_metric = piq.psnr(predict, target, data_range=1., reduction='none').mean()
        if self.ssim:
            ssim_metric = piq.ssim(predict, target, data_range=1.)
        if self.mssim:
            mssim_metric = piq.multi_scale_ssim(predict, target, data_range=1.)
        metric_summary = {'l1_norm': l1_norm_metric,
                          'mse': mse_norm_metric,
                          'pearsonr_metric': pearsonr_metric,
                          'cc': cc_metric,
                          'psnr': psnr_metric,
                          'ssim': ssim_metric,
                          'mssim': mssim_metric
                          }
        return metric_summary
