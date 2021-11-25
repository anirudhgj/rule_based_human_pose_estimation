import os.path as osp
import os
import cv2
import numpy as np
import random
import copy
import scipy.io as sio
import tensorflow as tf
import glob as glob
import json
from natsort import natsorted
import math
import imutils
from skimage.color import rgb2hsv, hsv2rgb
from h36_load_annots import annot_files, frames, cameras
from transform_util import root_relative_to_view_norm_skeleton
from hyperparams import Hyperparameters
import pickle


H = Hyperparameters()

def read_dummy_list():

    dummy = []
    f = open("dummy_list.txt", "r")
    for x in f:
        dummy.append(x)

    print ("(dummy) ", dummy[:50])
    print ("type(dummy) ", type(dummy))
    return dummy, copy.copy(dummy)

def unit_norm(mat, dim=1):
    norm = (np.sqrt(np.sum(mat ** 2, dim)) + 1e-9)
    norm = np.expand_dims(norm, dim)
    mat = mat / norm
    return mat

def normalize_3d_pose(pred_3d):
    pelvis = pred_3d[0,:]
    rhip = unit_norm(pred_3d[9,:], dim=0) * 1.05 + pelvis
    lhip = unit_norm(pred_3d[13,:], dim=0) * 1.05 + pelvis
    neck = unit_norm(pred_3d[1,:], dim=0) * 4.75 + pelvis
    rs = unit_norm(pred_3d[2,:]-pred_3d[1,:], dim=0) * 1.37 + neck
    re = unit_norm(pred_3d[3,:]-pred_3d[2,:], dim=0) * 2.8 + rs
    rh = unit_norm(pred_3d[4,:]-pred_3d[3,:], dim=0) * 2.4 + re
    ls = unit_norm(pred_3d[5,:]-pred_3d[1,:], dim=0) * 1.37 + neck
    le = unit_norm(pred_3d[6,:]-pred_3d[5,:], dim=0) * 2.8 + ls
    lh = unit_norm(pred_3d[7,:]-pred_3d[6,:], dim=0) * 2.4 + le
    head = unit_norm(pred_3d[8,:]-pred_3d[1,:], dim=0) * 2.0 + neck
    rk = unit_norm(pred_3d[10,:]-pred_3d[9,:], dim=0) * 4.2 + rhip
    ra = unit_norm(pred_3d[11,:]-pred_3d[10,:], dim=0) * 3.6 + rk
    rf = unit_norm(pred_3d[12,:]-pred_3d[11,:], dim=0) *2.0 + ra
    lk = unit_norm(pred_3d[14,:]-pred_3d[13,:], dim=0) * 4.2 + lhip
    la = unit_norm(pred_3d[15,:]-pred_3d[14,:], dim=0) * 3.6 + lk
    lf = unit_norm(pred_3d[16,:]-pred_3d[15,:], dim=0) *2.0 + la

    skeleton = np.concatenate([np.expand_dims(pelvis, axis=0), np.expand_dims(neck, axis=0), 
                           np.expand_dims(rs, axis=0), np.expand_dims(re, axis=0),
                           np.expand_dims(rh, axis=0), np.expand_dims(ls, axis=0),
                           np.expand_dims(le, axis=0), np.expand_dims(lh, axis=0),
                           np.expand_dims(head, axis=0), np.expand_dims(rhip, axis=0),
                           np.expand_dims(rk, axis=0), np.expand_dims(ra, axis=0),
                           np.expand_dims(rf, axis=0), np.expand_dims(lhip, axis=0), 
                           np.expand_dims(lk, axis=0), np.expand_dims(la, axis=0), 
                           np.expand_dims(lf, axis=0)], axis=0)
    return skeleton


#######################################
#Augmentation functions
#######################################
def perform_flip(points):
    points = np.stack([points[:,0] + 2*(112 - points[:,0]), points[:,1]], axis=1)
    points_flipped = np.stack([points[0], points[1], points[5], points[6], points[7], \
                                     points[2], points[3], points[4], points[8], points[13], \
                                     points[14], points[15], points[16], points[9], points[10], points[11], points[12]], axis=0)
    return points_flipped


def perform_tilt(points, tilt_angle):
    gauss_mu_x_tilt = ( (points[:,1] - 112.0) * np.cos(tilt_angle) - (points[:,0] - 112.0) * np.sin(tilt_angle) ) + 112.0
    gauss_mu_y_tilt = ( (points[:,1] - 112.0) * np.sin(tilt_angle) + (points[:,0] - 112.0) * np.cos(tilt_angle) ) + 112.0
    return np.stack([gauss_mu_y_tilt, gauss_mu_x_tilt], axis=1)

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def flip_image(image):
    return np.fliplr(image)

def flip_3d(pose_3d):
    #pose_3d_flip = np.dot(pose_3d , np.asarray([[-1,0,0],[0,-1,0],[0,0,1]], dtype = np.float32))
    skeleton_flipped_1  = pose_3d * np.asarray([-1, 1, 1], dtype = np.float32)
    skeleton_flipped = np.stack([skeleton_flipped_1[0,:], skeleton_flipped_1[1,:],
                                    skeleton_flipped_1[5,:], skeleton_flipped_1[6,:], skeleton_flipped_1[7,:],
                                    skeleton_flipped_1[2,:], skeleton_flipped_1[3,:], skeleton_flipped_1[4,:],
                                    skeleton_flipped_1[8,:], skeleton_flipped_1[13,:],skeleton_flipped_1[14,:],
                                    skeleton_flipped_1[15,:],skeleton_flipped_1[16,:],skeleton_flipped_1[9,:],
                                    skeleton_flipped_1[10,:],skeleton_flipped_1[11,:],skeleton_flipped_1[12,:]], axis=0)
    
    return skeleton_flipped

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


#######################################
#H3.6M dataset related functions
#######################################

# h36_datapath = '../../../dataset/dataset_standardization/H36_cropped'
h36_datapath = '/data/vcl/anirudh_rule_based/remotes/h36_test/jogendra/project_smpl/adit/dataset_standardization/H36_cropped/'

subjects = ['S1','S5','S6','S7','S8','S9','S11']
activities = os.listdir(os.path.join(h36_datapath, 'S1'))


def get_h36_test():
    test_set = []
    for subject in ['S9', 'S11']:
        for activity in activities:
            assert len(annot_files[subject][activity]['poses2d']) == len(annot_files[subject][activity]['poses3d']) == len(frames[subject][activity])
            for frame_id in range(len(frames[subject][activity])):
                #"58860488":  #
                if str(cameras[subject][activity][frame_id]) == "60457274":
                    test_set.append("%s %s %d"%(subject, activity, frame_id))
    print("Length of H3.6M test set:", len(test_set))
    return handle_batch_size(test_set)

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

def data_loader_h36(id, userandom = True):

    while True:
        #try:
            try:
                id = id.decode('ascii')
            except:
                pass

            if isinstance(id, str) and len(list(id.split()))==3:
                pass
            else:
                print("invalid dataID")
                exit()

            subject, activity, frame_id_sub = id.split()
            frame_id_sub = int(frame_id_sub)
            cameras_curr = cameras[subject][activity]
            frame_indices_curr = frames[subject][activity]

            frame_indices_curr = frames[subject][activity]
            cameras_curr = cameras[subject][activity]

            camera = cameras_curr[frame_id_sub]
            frame_id_act = frame_indices_curr[frame_id_sub]

            pose_3d_32 = annot_files[subject][activity]['poses3d'][frame_id_sub]

            pose_3d = np.stack([pose_3d_32[0,:],pose_3d_32[16,:],pose_3d_32[25,:],pose_3d_32[26,:],pose_3d_32[27,:],pose_3d_32[17,:],pose_3d_32[18,:],pose_3d_32[19,:],pose_3d_32[15,:],pose_3d_32[1,:],pose_3d_32[2,:],pose_3d_32[3,:],pose_3d_32[4,:],pose_3d_32[6,:],pose_3d_32[7,:],pose_3d_32[8,:],pose_3d_32[10,:]],0)
            pose_3d[0,:] = (pose_3d[9,:] + pose_3d[13,:])/2.
            pose_3d = pose_3d - pose_3d[0]

            #pose_3d_final = normalize_3d_pose(pose_3d)
            pose_3d_final = pose_3d
            
            view_norm_pose_3d_final = pose_3d_final
            #view_norm_pose_3d_final = root_relative_to_view_norm_skeleton(pose_3d_final)[1].astype(np.float32)

            im_file_name = 'img_' + (6 - len(str(frame_id_act)))*'0' + str(frame_id_act) + '.jpg'

            im_file_path = os.path.join(h36_datapath, os.path.join(subject + '/' + activity + '/imageSequence/' + str(camera), im_file_name))
            person_image = cv2.imread(im_file_path)
            
            person_image = (person_image - np.array([1, 1, 1])*0.4*255).astype(np.float32)

            person_image = person_image[:,:,::-1].astype(np.float32) /255.0

            break

    pose_3d = view_norm_pose_3d_final.astype(np.float32)

    return person_image, pose_3d.astype(np.float32)


if __name__ == "__main__":
    d = get_h36_test()
    x = data_loader_h36(d[1000])
