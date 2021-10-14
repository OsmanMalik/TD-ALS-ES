# This script loads in the coil 100 data, downsamples each image to 32 x 32, and
# then saves all compressed images as a Matlab mat file.

import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.io import savemat

target_shape = (32, 32, 3, 7200)
path = "coil-100" # Add correct path to coil-100 dataset here
img_array = np.zeros(shape=target_shape, dtype=float)
class_array = np.zeros(shape=(7200), dtype=int)
file_list = [file for file in os.listdir(path=path) if file[-4:] == '.png']

for idx, file in enumerate(file_list):
    img = io.imread(join(path, file))
    img = downscale_local_mean(img, (4, 4, 1))
    img_array[:, :, :, idx] = np.array(img, dtype=float)
    class_array[idx] = int(file[3:file.find('_')])

mat_name = 'compressed_coil_100.mat'
savemat(mat_name, {'img_array': img_array, 'class_array': class_array})
