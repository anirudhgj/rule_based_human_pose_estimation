import os.path as osp
import os
import cv2
import numpy as np
import random
import copy
import scipy.io as sio
import glob as glob
import json
from natsort import natsorted
import math
import imutils
from skimage.color import rgb2hsv, hsv2rgb
from hyperparams import Hyperparameters
import pickle


H = Hyperparameters()


#######################################
#Augmentation functions
#######################################

def flip_image(image):
    return np.fliplr(image)

def do_aug(rgb_img):

    b_delta_val = np.random.random() / 8.0        # between (0. to 30.) of 255.
    c_delta_val = np.random.random() / 5.0 + 0.6         # between (0.6 to 0.8)
    h_delta_val = np.random.random() * 0.2 + 0.3   # between (0.3 to 0.5)

    hsv_img = rgb2hsv(rgb_img)
    hue_img = hsv_img[:, :, 0] + (h_delta_val if np.random.randint(2) > 0 else 0.)
    sat_img = hsv_img[:, :, 1] + (c_delta_val if np.random.randint(2) > 0 else 0.)
    value_img = hsv_img[:, :, 2] + (b_delta_val if np.random.randint(2) > 0 else 0.)
    aug_image = hsv2rgb(np.stack([hue_img, sat_img, value_img], 2))

    return aug_image

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def handle_batch_size(dset):
    """
    A helper function to round up the length of the dataset "dset" to make it compatible
    with the batch size to avoid tensorflow shape incompatiblity.

    dset: A set of data point ids.

    """
    dset = list(dset)
    while len(dset)%H.batch_size!=0:
        dset.append(dset[-1])
    return dset


data_dict = sio.loadmat("sample_data.mat")
frames = data_dict['frames']
slow_frames = data_dict['slow_frames']
aug_images = True
#mean =  np.array([103.939, 116.779, 123.68])
mean =  np.array([1, 1, 1]) * 0.4 * 255

def data_loader_cr(index):
    """
    Data Loader for contrastive learning.
    """
    img_seq = frames[index].copy()
    aug_seq = img_seq.copy()

    for i in range(H.seq_length):
        d = random.randint(0, 1)
        if d%2==0:
            aug_seq[i] = do_aug(aug_seq[i])
        else:
            aug_seq[i] = augment_brightness_camera_images(aug_seq[i].astype(np.uint8))
        if aug_images and np.random.randint(0,3) == 2:
            img_seq[i] = do_aug(img_seq[i])
        if aug_images and np.random.randint(0,3) == 2:
            aug_seq[i] = do_aug(aug_seq[i])
 
        img_seq[i] = (img_seq[i] - mean)/255.
        aug_seq[i] = (aug_seq[i] - mean)/255.
    return img_seq, aug_seq

def data_loader_pose_flip(index):
    """
    Data Loader for pose flip relation
    """
    img_seq = frames[index].copy()
    aug_seq = img_seq.copy()

    for i in range(H.seq_length):
        aug_seq[i] = flip_image(aug_seq[i])
        if aug_images and np.random.randint(0,3) == 2:
            img_seq[i] = do_aug(img_seq[i])
        if aug_images and np.random.randint(0,3) == 2:
            aug_seq[i] = do_aug(aug_seq[i])

        img_seq[i] = (img_seq[i] - mean)/255.
        aug_seq[i] = (aug_seq[i] - mean)/255.

    return img_seq, aug_seq

def data_loader_flip_backward(index):
    """
    Data Loader forflip backward relation
    """
    img_seq = frames[index].copy()
    aug_seq = img_seq.copy()

    for i in range(H.seq_length):
        aug_seq[i] = flip_image(aug_seq[i])
        if aug_images and np.random.randint(0,3) == 2:
            img_seq[i] = do_aug(img_seq[i])
        if aug_images and np.random.randint(0,3) == 2:
            aug_seq[i] = do_aug(aug_seq[i])

        img_seq[i] = (img_seq[i] - mean)/255.
        aug_seq[i] = (aug_seq[i] - mean)/255.

    #temporally reverse the image
    aug_seq = aug_seq[::-1, :, :, :]
    return img_seq, aug_seq

def data_loader_slow_backward(index):
    """
    Data Loader for slow backward relation
    """
    img_seq = slow_frames[index].copy()
    aug_seq = img_seq.copy()
    img_seq = img_seq[::2, :, :, :] 
    aug_seq = aug_seq[15 : 45, :, :, :]

    for i in range(H.seq_length):
        if aug_images and np.random.randint(0,3) == 2:
            img_seq[i] = do_aug(img_seq[i])
        if aug_images and np.random.randint(0,3) == 2:
            aug_seq[i] = do_aug(aug_seq[i])

        img_seq[i] = (img_seq[i] - mean)/255.
        aug_seq[i] = (aug_seq[i] - mean)/255.

    #temporally reverse the image
    aug_seq = aug_seq[::-1, :, :, :]
    return img_seq, aug_seq

#indices of data points to be used by data loaders.
datasets = [range(16) for i in range(4)] + [range(8)] 
datasets = [list(x) for x in datasets]
data_loaders = [data_loader_cr, data_loader_cr, data_loader_pose_flip,\
        data_loader_flip_backward, data_loader_slow_backward]
