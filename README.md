**CT Film Recovery Network**
=======================
<!-- Description -->
dataset link: coming soon

# Usage
## Environment
```
python == 3.6 
pytorch == 1.6.0 
torchvison == 0.6
```
## Data preparation
We train/test our model on CTFilm20K Datasets 

We expect the directory structure to be the following:
```
path/to/CTFilm20K
    img/
        xxx.png
        xxx.png
        ...
    3dmap/
    albedo/
    depth/
    shader_normal/
    uv/
```
## Training
To train our model, run this script
```
python -m scripts.misc.train_unwarp.py
```
To evalutate our model, run this script
```
python -m scripts.misc.train_unwarp --test
```

# License
This code is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.


