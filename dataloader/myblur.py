import cv2
import numpy as np



def blur_bw_np(bm, ori):
    print("bw shape:", bm.shape)
    w,h = ori.shape[0], ori.shape[1]
    bm0=cv2.blur(bm[:,:,0],(3,3))
    bm1=cv2.blur(bm[:,:,1],(3,3))
    bm0=cv2.resize(bm0,(h,w))# 
    bm1=cv2.resize(bm1,(h,w))
    bm=np.stack([bm0,bm1],axis=-1)
    return bm

def resize_albedo_np(ori, img, ab):
    """
    ori: image with the real size (1080, 1080, 3)
    img: resized ori  (256, 256, 3)
    ab : predicted ab (256, 256, 3)
    """
    assert img.shape == (256,256,3) , "Error, but got {}".format(img.shape)
    assert ab.shape  == (256,256,1), "Error, but got {}".format(ab.shape)

    w,h = ori.shape[0], ori.shape[1]
    
    # BGR for cv2 imread
    # so RGB is 2,1,0
    # ori_gray = 0.3*ori[:,:,2]  + 0.59*ori[:,:,1] + 0.11*ori[:,:,0]
    ori_gray = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
    # img_gray = 0.3*img[:,:,2]  + 0.59*img[:,:,1] + 0.11*img[:,:,0]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diff = ab - img_gray[:,:,np.newaxis] 

    diff = cv2.blur(diff[:,:,0], (20,20))  # Or kernel of (3,3)

    diff=cv2.resize(diff,(h,w), )[:,:,np.newaxis] 

    # diff = np.stack([diff0, diff1, diff2], axis=-1)

    ab_large = ori_gray[:,:,np.newaxis] + diff

    return ab_large, diff, ori_gray[:,:,np.newaxis] , img_gray[:,:,np.newaxis] 
