import numpy as np
import os
import shutil

dr_dataset_train = '/home1/qiyuanwang/film_generate/npy_256/'
dr_dataset_test = '/home1/qiyuanwang/film_generate/npy_test_256/'
test_no = 2334

image_name_list = np.array([x.name for x in os.scandir(dr_dataset_train) if x.name.endswith(".npy")])
image_no = len(image_name_list)
permutation = np.random.permutation(image_no)
np.random.shuffle(permutation)
np.random.shuffle(permutation)
np.random.shuffle(permutation)
np.random.shuffle(permutation)
np.random.shuffle(permutation)
for i in range(test_no):
    idx = permutation[i]
    name = image_name_list[idx]
    f = dr_dataset_train + name
    t = dr_dataset_test + name
    shutil.move(f, t)
    print(f)
