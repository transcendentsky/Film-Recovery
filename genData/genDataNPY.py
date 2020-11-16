import numpy as np
import cv2
import os
import time
import pickle
# image

root_dir = '/home1/quanquan/generate/mesh_film_small/'  #'database_test/'  #
write_dir = '/home1/liuli/film_code/'
output_name = 'npy_1'

def preprocessPNG(input, shape=None, resize=False, image_name=None):
    if resize:
        input = cv2.resize(input, shape)
    input_01 = input.astype(np.float32) / 255 * 2 - 1
    input_CHW = input_01.transpose((2,0,1))
    return input_CHW

def preprocessEXR(input, shape=None, resize=False):
    """input:[-1,1]"""
    if resize:
        input = cv2.resize(input, shape)
    input_01 = input.astype(np.float32)
    input_CHW = input_01.transpose((2,0,1))
    return input_CHW


def sta_within_channel(image_dir, type='depth'):
    input_size = 100

    image_name_list = np.array([x.name for x in os.scandir(image_dir + 'img/') if x.name.endswith(".png")])
    data_no = len(image_name_list)
    data_all_0 = np.zeros((input_size, input_size, data_no))
    data_all_1 = np.zeros((input_size, input_size, data_no))
    data_all_2 = np.zeros((input_size, input_size, data_no))
    print(type)
    for i, image_name in enumerate(image_name_list):
        # print(i)
        data = cv2.imread(image_dir + type + '/' + image_name[:-3] + 'exr', cv2.IMREAD_UNCHANGED)        # [0,1]--有效部分为[0.1-1]
        data = cv2.resize(data, (input_size, input_size))
        data_all_0[:, :, i] = data[:, :, 0]
        data_all_1[:, :, i] = data[:, :, 1]
        data_all_2[:, :, i] = data[:, :, 2]

    print(type, '_ave_0', data_all_0.mean())
    print(type, '_ave_1', data_all_1.mean())
    print(type, '_ave_2', data_all_2.mean())
    print(type, '_std_0', data_all_0.std())
    print(type, '_std_1', data_all_1.std())
    print(type, '_std_2', data_all_2.std())



def write_data(image_dir, write_dir, input_size):
    image_name_list = np.array([x.name for x in os.scandir(image_dir + 'img/') if x.name.endswith(".png")])
    if not os.path.exists(write_dir + 'npy/'):
        os.mkdir(write_dir + 'npy/')
    miss_data=0
    start_time = time.time()
    for i, image_name in enumerate(image_name_list):

        """loading"""
        ori = cv2.imread(image_dir + 'img/' + image_name)
        ab = cv2.imread(image_dir + 'albedo/' + image_name)
        # bmap = cv2.imread(image_dir + 'bimg/' + image_name)

        # bmap = cv2.imread(image_dir + 'backward_img/' + image_name[:-3] + 'exr', cv2.IMREAD_UNCHANGED)
        # print(bmap.max())
        depth = cv2.imread(image_dir + 'depth/' + image_name[:-3] + 'exr', cv2.IMREAD_UNCHANGED)
        # depth_max_min = pickle.load(open(image_dir + 'depth_max_min/' + image_name[:-3] + 'pkl', 'rb'))

        # normal = cv2.imread(image_dir + 'compositor_normal/' + image_name[:-3] + 'exr', cv2.IMREAD_UNCHANGED)  # [-1,1]
        nos = cv2.imread(image_dir + 'shader_normal/' + image_name[:-3] + 'exr', cv2.IMREAD_UNCHANGED)  # [0,1]

        uv = cv2.imread(image_dir + 'uv/' + image_name[:-3] + 'exr', cv2.IMREAD_UNCHANGED)              # [0,1]

        cmap = cv2.imread(image_dir + '3dmap/' + image_name[:-3] + 'exr', cv2.IMREAD_UNCHANGED)  # [0,1]
        # cmap_max_min = pickle.load(open(image_dir + '3dmap_max_min/' + image_name[:-3] + 'pkl', 'rb'))  # [0,1]

        # bmap[np.isnan(bmap)]=0

        if (ab is None) or (depth is None) or (nos is None) or (uv is None) or (cmap is None):
            miss_data += 1
            continue
        else:

            """processing"""
            ori_1080 = preprocessPNG(ori, shape=input_size, resize=False)
            ori = preprocessPNG(ori, shape=input_size, resize=True)
            ab = preprocessPNG(ab, shape=input_size, resize=True, image_name=image_name)


            background = cv2.threshold(depth, 0, 1, cv2.THRESH_BINARY)
            background = preprocessEXR(background[1], shape=input_size, resize=True)
            background = np.expand_dims(background[0, :, :], axis=0)

            # bmap = preprocessEXR(bmap[:, :, 0:2]*5-1, shape=input_size, resize=True)
            # depth0 = np.expand_dims(depth[:,:,0], axis=2)
            depth = preprocessEXR(preprocess(depth, mean=[0.316, 0.316, 0.316], std=[0.309, 0.309, 0.309]), shape=input_size, resize=True)
            depth = np.expand_dims(depth[0,:,:], axis=0) 
            normal = preprocessEXR(preprocess(nos, mean=[0.584, 0.294, 0.300], std=[0.483, 0.251, 0.256]), shape=input_size, resize=True)
            cmap = preprocessEXR(preprocess(cmap, mean=[0.100, 0.326, 0.289], std=[0.096, 0.332, 0.298]), shape=input_size, resize=True)
            uv = preprocessEXR(uv[:, :, 1:] * 2 - 1, shape=input_size, resize=True)

            output ={
                'ori_1080': ori_1080,
                'ori': ori,
                'ab': ab,
                # 'bmap': bmap,
                'depth': depth,
                'normal': normal,
                'uv': uv,
                'cmap': cmap,
                'background': background
            }
            np.save(write_dir + 'npy/' + image_name[:-3] + 'npy', output)

        if i % 100 == 0:
            print('It took {:.02f} seconds to write {} samples.'.format(float(time.time()-start_time), i+1))
            start_time = time.time()
    print('Wrote {} samples, and {} samples missed'.format(i+1, miss_data))


def printInfo(x, name):
    print(name, x.shape, x.max(), x.min())

def preprocess(x, mean, std):
    y = np.zeros(x.shape)
    y[:, :, 0] = (x[:, :, 0] - mean[0]) / std[0]
    y[:, :, 1] = (x[:, :, 1] - mean[1]) / std[1]
    y[:, :, 2] = (x[:, :, 2] - mean[2]) / std[2]
    return y

def repropocess(y, mean, std):
    x = np.zeros(y.shape)
    if np.ndim(y)==3:           # cmap, normal
        x[:, :, 0] = y[:, :, 0] * std[0] + mean[0]
        x[:, :, 1] = y[:, :, 1] * std[1] + mean[1]
        x[:, :, 2] = y[:, :, 2] * std[2] + mean[2]
    elif np.ndim(y)==2:         # depth
        x[:, :] = y[:, :] * std[0] + mean[0]
    return x

def repropocess_mask(y, mean, std, mask):
    x = np.zeros(y.shape)
    if np.ndim(y)==3:           # cmap, normal
        x[:, :, 0] = (y[:, :, 0] * std[0] + mean[0])*mask
        x[:, :, 1] = (y[:, :, 1] * std[1] + mean[1])*mask
        x[:, :, 2] = (y[:, :, 2] * std[2] + mean[2])*mask
    elif np.ndim(y)==2:         # depth
        x[:, :] = (y[:, :] * std[0] + mean[0])*mask
    return x

def write_image(image_float, dir, norm1 = False):
    if norm1:
        image_uint8 = (image_float * 255).astype(np.uint8)
    else:
        image_uint8 = ((image_float+1)/2 * 255).astype(np.uint8)
    # image_uint8 = ((image_float+1)/2 *255).astype(np.uint8)
    cv2.imwrite(dir, image_uint8)

def write_cmap_gauss(image_float, dir, mean=[0.100, 0.326, 0.289], std=[0.096, 0.332, 0.298]):
    image_float = repropocess(image_float, mean, std)
    image_uint8 = (image_float * 255).astype(np.uint8)
    cv2.imwrite(dir, image_uint8)


def mainR():
    npy_dir = write_dir + 'npy'
    npy_list = np.array([x.name for x in os.scandir(npy_dir) if x.name.endswith(".npy")])
    for i, npy_name in enumerate(npy_list[:6]):
        data = np.load(npy_dir + '/' + npy_name, allow_pickle=True)[()]
        ori_1080 = data['ori_1080']
        ori = data['ori']
        ab = data['ab']
        depth = data['depth']
        normal = data['normal']
        uv = data['uv']
        cmap = data['cmap']
        background = data['background']
        # bmap = data['bmap']

        printInfo(ori, 'ori')
        printInfo(ab, 'ab')
        printInfo(depth, 'depth')
        printInfo(normal, 'normal')
        printInfo(uv, 'uv')
        printInfo(cmap, 'cmap')
        # printInfo(bmap, 'bmap')
        print('done')

        # write_image(bmap.transpose(1,2,0)[:,:,0],
        #             'loaddata_test/bmap_0.jpg')
        # write_image(bmap.transpose(1,2,0)[:,:,1],
        #             'loaddata_test/bmap_1.jpg')
        # write_image(ori.transpose(1,2,0),
        #             'loaddata_test/ori.jpg')
        # write_image(ori_1080.transpose(1,2,0),
        #             'loaddata_test/ori_1080.jpg')
        # write_image(uv.transpose(1,2,0)[:,:,0],
        #             'loaddata_test/uv_0.jpg')
        # write_image(uv.transpose(1,2,0)[:,:,1],
        #             'loaddata_test/uv_1.jpg')
        # write_image(background.transpose(1,2,0)[:,:,0],
        #             'loaddata_test/background.jpg', norm1=True)
        # write_cmap_gauss(cmap.transpose(1,2,0),
        #                  'loaddata_test/cmap.jpg')
        write_image(ab.transpose(1,2,0)[:,:,0],
                    'loaddata_test/ab_0.jpg')
        write_image(ab.transpose(1,2,0)[:,:,1],
                    'loaddata_test/ab_1.jpg')
        write_image(ab.transpose(1,2,0)[:,:,2],
                    'loaddata_test/ab_2.jpg')
        write_image(ab.transpose(1,2,0),
                    'loaddata_test/ab.jpg')

        print('done')

def mainW():
    image_dir = root_dir  #'data_2000/Mesh_Film/'
    input_size = (256, 256)
    write_data(image_dir, write_dir, input_size)


if __name__ =='__main__':
    mainR()
    # sta_within_channel(root_dir, 'depth')
    # sta_within_channel(root_dir, 'shader_normal')
    # sta_within_channel(root_dir, 'uv')
    # sta_within_channel(root_dir, '3dmap')


