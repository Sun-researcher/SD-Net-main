from glob import glob
import sys
import json
import numpy as np
from PIL import Image
import os
import pandas as pd

def init_ff(phase, level='frame', n_frames=8):
    dataset_path_fake = 'data/FaceForensics++/manipulated_sequences/*/c23/frames/'
    dataset_path_real = 'data/FaceForensics++/original_sequences/*/c23/frames/'
    image_list = []
    label_list = []

    folder_list_real = sorted(glob(dataset_path_real + '*'))
    folder_list_fake = sorted(glob(dataset_path_fake + '*'))
    list_dict = json.load(open(f'data/FaceForensics++/{phase}.json', 'r'))

    filelist = []
    for i in list_dict:
        filelist += i
    print('The proportion of the training dataset ',len(filelist))

    folder_list_real = [i for i in folder_list_real if os.path.basename(i)[:3] in filelist]
    folder_list_fake = [i for i in folder_list_fake if os.path.basename(i)[:3] in filelist]

    if level == 'video':
        label_list_real = [0] * len(folder_list_real)
        label_list_fake = [1] * len(folder_list_fake)
        label_list=label_list_real+label_list_fake
        folder_list = folder_list_real + folder_list_fake
        return folder_list, label_list

    for i in range(len(folder_list_real)):
        images_temp_real = sorted(glob(folder_list_real[i] + '/*.png'))

        if n_frames < len(images_temp_real):
            images_temp_real = [images_temp_real[round(i)] for i in np.linspace(0, len(images_temp_real) - 1, n_frames)]
        image_list += images_temp_real
        label_list += [0] * len(images_temp_real)

    for i in range(len(folder_list_fake)):
        images_temp_fake = sorted(glob(folder_list_fake[i] + '/*.png'))

        if n_frames < len(images_temp_fake):
            images_temp_fake = [images_temp_fake[round(i)] for i in np.linspace(0, len(images_temp_fake) - 1, n_frames)]
        image_list += images_temp_fake
        label_list += [1] * len(images_temp_fake)

    print('totals:',len(image_list))

    for i in range(len(image_list)):
        image_list[i] = image_list[i].replace('\\', '/')

    return image_list, label_list

if __name__ == '__main__':
    init_ff('train')
