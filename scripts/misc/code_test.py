"""
3d map to normal map
"""
import cv2
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import math
import torch.nn.functional as F
import pickle
image_ori = cv2.imread('image/ori_CT500-CT5_2-Pxa0025.png')
# image_bw = cv2.imread('image/bw_CT500-CT5_2-Pxa0025.png')   # the third channel == 0
image_depth = cv2.imread('image/depth_CT500-CT5_2-Pxa0025.png', cv2.IMREAD_GRAYSCALE)
# 应该是写的时候就用的cv2 所以现在读进来的时候还是前两个通道。


# image_uv = cv2.imread('image/uv_CT500-CT5_2-Pxa0025.exr', cv2.IMREAD_UNCHANGED)  #BGR
# image_3d = cv2.imread('Mesh_Film/3dmap_max_min/CT500-CT218_7-3zs.pkl', cv2.IMREAD_UNCHANGED)
image_3d = open('Mesh_Film/3dmap_max_min/CT500-CT218_7-3zs.pkl','rb+')
image_3d = pickle.load(image_3d)

image_normal = cv2.imread('Mesh_Film/compositor_normal/CT500-CT218_7-3zs0013.exr', cv2.IMREAD_UNCHANGED)
# image_nos = cv2.imread('image/nos_CT500-CT5_2-Pxa0025.exr', cv2.IMREAD_UNCHANGED)
# plt.subplot(231);plt.imshow(image_ori)
# plt.subplot(232);plt.imshow(image_bw)
# plt.subplot(233);plt.imshow(image_depth)

# plt.subplot(234);plt.imshow(np.uint8(image_uv*255))
# plt.subplot(235);plt.imshow(np.uint8(image_3d*255))
# plt.subplot(236);plt.imshow(np.uint8(image_normal*255))
# plt.show()

# """测试uvmap的通道情况
# 第0个channel非0即1
# """
# uv_shape_list =[]
# name_list = np.array([x.name for x in os.scandir('image/uv') if x.name.endswith(".exr")])
# for i, name in enumerate(name_list):
#     uv = cv2.imread('image/uv/' + name, cv2.IMREAD_UNCHANGED)
#     uv_3 = (uv[:,:,0]*255).astype(np.uint8)
#     cv2.imwrite('uv_2/'+name[:-4]+'.jpg', uv_3)
#     # uv_shape_list.append(uv_mean)
#
# print(uv_shape_list)
image_3d = cv2.resize(image_3d, (224, 224))
image_normal = cv2.resize(image_normal, (224, 224))
# cv2.imwrite('image_normal.jpg', image_normal)
#
# for x in range(224):
#     for y in range(224):
#         image_normal[x, y, :] = image_normal[x, y, :] / np.linalg.norm(image_normal[x, y, :], ord=2)
# cv2.imwrite('image_normal_norm.jpg', image_normal)
#

# plt.imshow(image_normal)
# plt.imshow(image_3d)



def double2uint8(input):
    mmax = np.max(input)
    mmin = np.min(input)
    input_01 = (input - mmin)/(mmax-mmin)
    output = (input_01*255).astype(np.uint8)
    return np.array(output)


def double012uint8(input):          # input为double，数据[-1,1]，转为uint8
    input_01 = (input +1)/2
    output = (input_01*255).astype(np.uint8)
    return np.array(output)


def out_product(input1, input2):
    a1, b1, c1 = input1[0], input1[1], input1[2]
    a2, b2, c2 = input2[0], input2[1], input2[2]
    output = [b1*c2-b2*c1, c1*a2-c2*a1, a1*b2-a2*b1]
    return np.array(output)


def cal_normal(ind1, ind2, tangent):
    """
    tengent: [9,3]
    """
    tangent1 = tangent[ind1, :]
    tangent2 = tangent[ind2, :]
    return out_product(tangent1, tangent2)

def normalize_vector(v):
    l2 = np.linalg.norm(v, ord=2)
    if l2 !=0:
        return v/l2
    else:
        return v



"""3d map 转normal"""
w,h,c  = image_3d.shape

output = np.zeros((w,h,c))
# output_bi = np.zeros((w,h,1))
tangent_1 = np.zeros((w,h,c))
tangent_2 = np.zeros((w,h,c))

# sum = np.zeros((9,3))
sum1 =[[0,0,0]]
text_map = np.ones((112,112,1))*(-1)
start = time.clock()
flag =1
for x in range(1, w-1): #[111]:#
    print(x)
    for y in range(1, h-1): #[111]:#
        # 对于除去边缘一圈的点，计算法向量
        sum = np.array([0, 0, 0])
        neighbor = image_3d[x-1:x+2, y-1:y+2, :]    # 取中心点周围一圈相邻点的空间坐标，
        center = image_3d[x, y, :]      # 中心点空间坐标
        # dist = np.zeros(9)
        tangent = np.zeros([9, 3])      # 存周围一圈点指向中心点的向量，在两个点无限接近的情况下，可以认为是中心点的切向量
        for i in range(3):
            for j in range(3):
                ind = i*3+j,
                # 去掉边缘效应，如果原pixel=[0,0,0]，则设为和center一样的值，这样算出来的切向量=0，则法向量=0，不计入最后取平均中
                if neighbor[i,j,0] ==0 and neighbor[i,j,1] ==1 and neighbor[i,j,2] ==2:
                    tangent[ind, :] = [0, 0, 0]
                else:
                    tangent[ind, :] = neighbor[i, j, :] - center
                # dist[ind] = np.linalg.norm(tangent[ind, :], ord=2)
        # ind = np.argpartition(dist[[0, 1, 2, 3, 5, 6, 7, 8]], 2)[:2]
        # if ind[0] >= 4:
        #     ind[0] = ind[0]+1
        # if ind[1] >= 4:
        #     ind[1] = ind[1] + 1

        # x1, y1 = ind[0]//3, ind[0]%3
        # x2, y2 = ind[1]//3, ind[1]%3
        #
        # a1, b1, c1 = neighbor[x1, y1, :] - center
        # a2, b2, c2 = neighbor[x2, y2, :] - center
        # tangent1 = tangent[ind[0], :]
        # tangent2 = tangent[ind[1], :]
        # theta = math.sin()
        # a1, b1, c1 = tangent[ind[0], :]
        # a2, b2, c2 = tangent[ind[1], :]
        # output[x,y,:] = [b1*c2-b2*c1, c1*a2-c2*a1, a1*b2-a2*b1]
        # tangent_1[x, y, :] = tangent1
        # tangent_2[x, y, :] = tangent2
        # temp = out_product(tangent1, tangent2)

        """
        0  1  2
        3  4  5
        6  7  8
        
        """
        for k,p in [[0,2],[1,5],[2,8],[5,7],[8,6],[7,3],[6,0]]:     # 或者也可取[[0,1],[1,2],[2,5],[5,8],[8,7],[7,6],[6,3],[3,0]]:
            # k, p 的选择没有特殊含义，只是
            temp = cal_normal(k, p, tangent)        #分别计算两两切向量的叉乘（法向）方向，再化为标准向量后取平均
            sum = sum + normalize_vector(temp)
            # sum1 = np.append(list(sum1), [list(normalize_vector(temp))], axis=0)
        output[x,y,:] = normalize_vector(sum)

        # if temp_length != 0:
        #     if np.sum(temp * output[x, y-1, :]) < 0:
        #         output[x, y, :] = (-1)*temp/temp_length
        #     else:
        #         output[x, y, :] = temp / temp_length
        #     output_bi[x, y, :] = 1
        ##统计是否两个向量成180度
            # sum = sum+1
            # if ind[0]+ind[1] ==8:
            #     cal=cal+1
            #     if ind[0]==0 or ind[1]==0:
            #         text_map[x-56, y-56]=0
            #     elif ind[0]==1 or ind[1]==1:
            #         text_map[x - 56, y - 56] = 1
            #     elif ind[0]==2 or ind[1]==2:
            #         text_map[x - 56, y - 56] = 2
            #     elif ind[0]==3 or ind[1]==3:
            #         text_map[x - 56, y - 56] = 3

        # #计算切向量图
        # tangent_1_length = np.linalg.norm(tangent_1[x,y,:], ord=2)
        # tangent_2_length = np.linalg.norm(tangent_2[x, y, :], ord=2)
        # if tangent_1_length != 0:
        #     tangent_1[x, y, :] = tangent_1[x,y,:]/tangent_1_length
        # if tangent_2_length != 0:
        #     tangent_2[x, y, :] = tangent_2[x,y,:]/tangent_2_length


print(time.clock()-start)
output1 = double2uint8(output)
# output_bi1 =double2uint8(output_bi)
# tangent_1 = double2uint8(tangent_1)
# tangent_2 = double2uint8(tangent_2)
# img_normal = double2uint8(image_normal)
# deta = np.abs(output1.astype(np.float32) - img_normal.astype(np.float32)).astype(np.uint8)
image_normal1= double012uint8(image_normal)
cv2.imwrite('gt_normal.jpg', image_normal1)
cv2.imwrite('gt_normal_0.jpg', image_normal1[:,:,0])
cv2.imwrite('gt_normal_1.jpg', image_normal1[:,:,1])
cv2.imwrite('gt_normal_2.jpg', image_normal1[:,:,2])

cv2.imwrite('gen_normal.jpg', output1)
cv2.imwrite('gen_normal_0.jpg', output1[:,:,0])
cv2.imwrite('gen_normal_1.jpg', output1[:,:,1])
cv2.imwrite('gen_normal_2.jpg', output1[:,:,2])



# # cv2.imwrite('gen_normal_bi.jpg', output_bi1)
# cv2.imwrite('tangent_1.jpg', tangent_1)
# cv2.imwrite('tangent_2.jpg', tangent_2)
#
# # cv2.imwrite('deta.jpg', deta)
#
# image_3d_1 = double2uint8(image_3d)
# cv2.imwrite('3d_channel_0.jpg', image_3d_1[:,:,0])
# cv2.imwrite('tangent_1_channel_0.jpg', tangent_1[:,:,0])
#
# cv2.imwrite('3d_channel_1.jpg', image_3d_1[:,:,1])
# cv2.imwrite('tangent_1_channel_1.jpg', tangent_1[:,:,1])
#
# cv2.imwrite('3d_channel_2.jpg', image_3d_1[:,:,2])
# cv2.imwrite('tangent_1_channel_2.jpg', tangent_1[:,:,2])
# 为啥output向量max 和min会为







# """bwmap矫正"""
# # image_3d = cv2.imread('image/3d_CT500-CT5_2-Pxa0025.exr', cv2.IMREAD_UNCHANGED)
# image_bw = cv2.imread('image/bw_CT500-CT5_2-Pxa0025.png')   # the third channel == 0
# # image_bw = cv2.resize(image_bw, (224, 224))
# image_bw_index = image_bw.astype(np.float16)/255
# image_ori = cv2.imread('image/ori_CT500-CT5_2-Pxa0025.png')
#
# # b,h,w,c
#
#
# # w, h = image_bw_index.shape[0:2]
# # w_ori, h_ori = image_ori.shape[0:2]
# # image_rect = np.zeros((w, h, 3), dtype=np.uint8)
# # start = time.clock()
# # for x in range(w):
# #     for y in range(h):
# #         x_c, y_c = image_bw_index[x, y, 0:2]
# #
# #         x_cc = (x_c * w_ori).astype(np.int16); y_cc = (y_c * h_ori).astype(np.int16)
# #         image_rect[x, y, :] = image_ori[x_cc, y_cc, :]
# # print('time', (time.clock()-start))
# # cv2.imwrite('rectification.jpg', image_rect)
#
# # input 4D tensor  [b, h, w, c]
# image_ori_tensor = torch.from_numpy(image_ori)
# image_ori_tensor = image_ori_tensor.float()
# image_ori_tensor = torch.unsqueeze(image_ori_tensor, 0)
#
# #bw 4D tensor [-1, 1] [b, h, w, 2]
# image_bw_index_tensor0 = torch.from_numpy(image_bw_index*2-1)
# image_bw_index_tensor1 = torch.unsqueeze(image_bw_index_tensor0, 0)
# image_bw_index_tensor2 = image_bw_index_tensor1[:,:,:,0:2]
# image_bw_index_tensor = image_bw_index_tensor2.float()
# start1 = time.clock()
#
#
# image_ori_tensor_bchw = image_ori_tensor.transpose(2, 3).transpose(1,2)
# image_un_wrap_bchw = F.grid_sample(input=image_ori_tensor_bchw, grid=image_bw_index_tensor)    # input as tensor
#
#
# # output nchw
# image_un_wrap_nhwc = image_un_wrap_bchw.transpose(1,2).transpose(2,3)
# image_un_wrap_hwc = image_un_wrap_nhwc.squeeze()
# image_un_wrap_hwc = image_un_wrap_hwc.type(torch.uint8).numpy()
# print('time', (time.clock()-start1))
# cv2.imwrite('rectification_1.jpg', image_un_wrap_hwc)


print('done')


