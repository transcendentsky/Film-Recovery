from collections import OrderedDict
import os
import numpy as np
import torch
import random
import torchvision
import string

import random
import time
import cv2

from pathlib import Path

TUTILS_DEBUG = True
TUTILS_INFO = False
TUTILS_WARNING = True

def p(*s,end="\n", **kargs ):
    if TUTILS_INFO:
        print("[Trans Info] ", s, kargs, end="")
        print("", end=end)

def w(*s,end="\n", **kargs ):
    if TUTILS_WARNING or TUTILS_DEBUG:
        print("[Trans Warning] ", s, kargs, end="")
        print("", end=end)

def d(*s,end="\n", **kargs ):
    if TUTILS_DEBUG:
        print("[Trans Debug] ", s, kargs, end="")
        print("", end=end)

def tfuncname(func):
    def run(*argv, **kargs):
        print("--------------------------------------------")
        print("[Trans Utils] Print function name: ", end=" ")
        print(func.__name__)
        ret = func(*argv, **kargs)
        # if argv:
        #     ret = func(*argv)
        # else:
        #     ret = func()
        return ret
    return run

# @tfuncname
def tt():
    # print("[Trans Utils] ", end="")
    pass

def time_now():
    tt()
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())

def generate_random_str(n):
    tt()
    ran_str = ''.join(random.sample(string.ascii_letters + string.digits, n))
    return ran_str

def generate_name():
    tt()
    return time_now() + generate_random_str(6)
    
# def write_image_np(image, filename):
#     tt()
#     cv2.imwrite("wc_" + generate_random_str(5)+'-'+ time_now() +".jpg", image.astype(np.uint8))
#     pass

def tdir(*dir_paths):
    def checkslash(name):
        if name.startswith("/"):
            name = name[1:]
            return checkslash(name)
        else:
            return name
    names = [dir_paths[0]]
    for name in dir_paths[1:]:
        names.append(checkslash(name))
    dir_path = os.path.join(*names)
    d(dir_path)
    if not os.path.exists(dir_path):
        d("Create Dir Path: ", dir_path)
        os.makedirs(dir_path)

    return dir_path

def tfilename(*filenames):
    def checkslash(name):
        if name.startswith("/"):
            name = name[1:]
            return checkslash(name)
        else:
            return name
    names = [filenames[0]]
    for name in filenames[1:]:
        names.append(checkslash(name))
    filename = os.path.join(*names)
    d(filename)
    parent, name = os.path.split(filename)
    if not os.path.exists(parent):
        os.makedirs(parent)
    return filename

def ttsave(state, path, configs=None):
    path = tdir("trans_torch_models", path, generate_name())
    if configs is not None:
        assert type(configs) is dict
        config_path = tfilename(path, "configs.json")
        with open(config_path, "wb+") as f:
            config_js = json.dumps(configs)
            f.write(config_js)
    torch.save(state, tfilename(path, "model.pkl"))
    

if __name__ == "__main__":
    # tt()
    # tfilename("dasd", "/dasdsa", "/dsad")
    tdir("dasd", "/dsadads", "/dsdas")

