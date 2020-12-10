import cv2
import numpy as np
import os

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def transparent_img(img_path):
    # "/home1/quanquan/datasets/generate/head-texture/head-alpha/CT500-CT365_8.png"
    parent, name = os.path.split(img_path)
    output_dir = "/home1/quanquan/datasets/generate/head-texture/head-alpha-2/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # print(alpha_channel, alpha_channel.shape)
    alpha_channel = sigmoid(alpha_channel/255.0*10-8) *255.0
    
    aa = np.ones_like(alpha_channel)*255
    # print(alpha_channel, alpha_channel.shape)
    alpha_channel = np.clip(np.round(aa - alpha_channel), 1, 255)
    count_matrix = np.where(alpha_channel==0, 0, 1)
    w,h = count_matrix.shape
    print("count: ", np.sum(count_matrix))
    print("({})".format(w*h))
    alpha_channel = alpha_channel.astype(np.uint8)
    
    # print(alpha_channel, alpha_channel.shape)
    

    # print("shape: ", img.shape)

    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    print(output_dir+name)
    cv2.imwrite(output_dir + name, img_BGRA)
    
    # cv2.imwrite()

if __name__ == "__main__":
    npy_dir = "/home1/quanquan/datasets/generate/head-texture/head-texture/"
    npy_list = np.array([x.path for x in os.scandir(npy_dir) if x.name.endswith(".png")])
    # img_path = "dataloader/CT500-CT0_3.png"
    print("len: {}".format(len(npy_list)))
    i = 1
    for img_path in npy_list:
        transparent_img(img_path)
        print("writing alpha img : {}  ".format(i), end="")
        i += 1
        break
        print("test")
