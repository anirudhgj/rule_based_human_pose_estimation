import warnings
warnings.simplefilter(action = 'ignore' , category = FutureWarning)
import tensorflow as tf
import os
import glob
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as ming
import matplotlib.pyplot as plt
import random
from scipy import ndimage
from skimage.io import imread_collection
import cv2
from skimage.color import rgb2hsv, hsv2rgb
from hyperparams import Hyperparameters



H = Hyperparameters ()


def augment_pose_seq(pose_seq,z_limit=(0,360),y_limit=(-90,90)):
    pose_seq = np.expand_dims(pose_seq, axis=1)
    thetas = np.random.uniform(z_limit[0],z_limit[1], pose_seq.shape[0])
    thetas = np.stack([thetas]*pose_seq.shape[1], 1)
    k=[]
    for ct, xx in enumerate(thetas):
        k.append(pose_rotate(pose_seq[ct], np.expand_dims(thetas[ct], 1), pose_seq[ct].shape[0]))
    k = np.stack(k, 0)

    thetas = np.random.uniform(y_limit[0],y_limit[1], k.shape[0])
    thetas = np.stack([thetas]*k.shape[1], 1)
    p=[]
    for ct, xx in enumerate(thetas):
        p.append(rotate_y_axis(k[ct], np.expand_dims(thetas[ct], 1), k[ct].shape[0]))
    p = np.stack(p, 0)
    return k

def pose_rotate(points, theta, batch_size):
    
    theta = theta * np.pi / 180.0
    cos_vals = np.cos(theta)
    sin_vals = np.sin(theta)
    row_1 = np.concatenate([cos_vals, -sin_vals], axis=1)# 90 x 2
    row_2 = np.concatenate([sin_vals, cos_vals], axis=1)# 90 x 2
    row_12 = np.stack((row_1, row_2), axis=1)#90 x 2 x 2
    zero_size_row1x2 = np.zeros([batch_size, 1, 2])#90 x 1 x 2
    r1x2xZero = np.concatenate([row_12, zero_size_row1x2], axis=1)
    stacker = np.array([0.0, 0.0, 1.0])
    third_cols = np.reshape(np.tile(stacker, batch_size), [batch_size, 3])
    third_cols = np.expand_dims(third_cols, 2)
    rotation_matrix = np.concatenate([r1x2xZero, third_cols], axis=2)
    return np.matmul(points.reshape([points.shape[0], 17, 3]), rotation_matrix)


def rotate_y_axis(points,theta,batch_size):
    theta = theta * np.pi / 180
    cos_vals = np.cos(theta)#90 x 1
    sin_vals = np.sin(theta)
    zero_vals = np.zeros((batch_size,1))
    ones_vals = np.ones((batch_size,1))
    row_1 = np.concatenate([cos_vals, zero_vals],axis =1)#90 x2
    row_2 = np.concatenate([zero_vals ,ones_vals],axis=1)# 90 x 2
    row_12 = np.stack((row_1, row_2), axis=1)#90 x 2 x 2
    temp_3 = np.stack((-sin_vals,zero_vals),axis =2)#90 x 1 x 2
    temp_32 = np.concatenate([row_12,temp_3],axis = 1)#90 x 3 x 2
    third_cols = np.concatenate([sin_vals,zero_vals,cos_vals],axis=1)#90 x 3
    third_cols = np.expand_dims(third_cols, 2)
    rotation_matrix = np.concatenate([temp_32, third_cols], axis=2)
    return np.matmul(points.reshape([points.shape[0], 17, 3]), rotation_matrix)


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

def tilt_3d(pose_3d, angle):
    rotmat = np.asarray([[np.cos(angle), 0 , -np.sin(angle)],[0,1,0],[np.sin(angle), 0, np.cos(angle)]], dtype = np.float32)
    return np.dot(pose_3d, rotmat)

def perform_flip(pose_2d):
    pose_2d_flip = pose_2d * np.asarray([-1,1], dtype = np.float32)
    pose_2d_flip[:,0] = pose_2d_flip[:,0] + 224
    return pose_2d_flip

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
    pose_3d_flip = pose_3d * np.asarray([-1, 1, 1], dtype = np.float32)
    return pose_3d_flip


def get_rotmat_camera(alpha, beta, gamma):
    R_x = [ [1, 0, 0] ,
           [0, np.cos(alpha), -np.sin(alpha)] , 
           [0, np.sin(alpha), np.cos(alpha)] ]
    
    
    R_y = [ [np.cos(beta), 0, np.sin(beta)] ,
           [0, 1, 0] , 
           [-np.sin(beta), 0, np.cos(beta)] ]
    
    
    R_z = [ [np.cos(gamma), -np.sin(gamma), 0] ,
           [np.sin(gamma), np.cos(gamma), 0] , 
           [0, 0, 1] ]
    
       
    R = np.matmul(R_x, np.matmul(R_y, R_z))
    return R.astype(np.float32)

def _read_py_function(filename):

    poses_path = filename[0]
    image_path = filename[0].replace('.mat','.png')
    pose_image = cv2.imread(image_path)
    pose_image  = cv2.resize(pose_image,(224,224))
    img = pose_image / 255.0
    
    pose_2d = sio.loadmat(poses_path)['pose_2d']
    pose_3d = sio.loadmat(poses_path)['pose_3d']
    
    elevation = sio.loadmat(poses_path)['elevation'][0][0]    
    tilt = sio.loadmat(poses_path)['tilt'][0][0]
    azimuth = sio.loadmat(poses_path)['azimuthal'][0][0]
    
    angles_array = np.array( [elevation , tilt , azimuth ] )

    back_paths = filename[1]
    back_images = cv2.imread(back_paths)
    back_images = cv2.resize(back_images, (224,224))      # resizing and blurring
    bg = back_images / 255.0

        
    #appending background to foreground
    mask = np.max(img, -1) 
    mask = (mask>0.1)*1.0
    mask = np.expand_dims(mask, -1)          
            
    img = (1. - mask) * bg.astype(np.float32) + mask * img.astype(np.float32)
    img = img.astype(np.float32)
                         
    pose_2d = (pose_2d + 1)*112.0  
    person_image = img[:,:,::-1]
    
    R = get_rotmat_camera(angles_array[0],angles_array[1],angles_array[2])
    X = np.matmul(R, np.transpose(pose_3d, [1,0]))
    pose_3d_view = np.transpose(X, [1,0])
    
    if np.random.rand() < H.jitter_prob:
        person_image = do_aug(person_image)
    
    if np.random.rand() <  H.flip_prob:
        person_image = flip_image(person_image)
        pose_3d_view = flip_3d(pose_3d_view)
        pose_2d = perform_flip(pose_2d)
        
    if np.random.rand() <  H.tilt_prob:
        angle = np.random.randint(-H.tilt_limit, H.tilt_limit)
        person_image = rotate_image(person_image, angle)
        pose_3d_view = tilt_3d(pose_3d_view, np.radians(angle))
        pose_2d = perform_tilt(pose_2d, np.radians(angle))

    person_image = person_image - ( np.array([103.939, 116.779, 123.68]) / 255.0).astype(np.float32)
        
    return person_image.astype(np.float32),pose_3d_view.astype(np.float32)


def data_loader(data_path , back_path ,batch_size,NUM_THREADS):        

    poses = list(np.load(data_path))

    back = list(np.random.choice(glob.glob(back_path),len(poses)))
    data = [[poses[i],back[i]] for i in range(len(poses))]
    tr_dataset = tf.data.Dataset.from_tensor_slices((data))

    tr_dataset_poses = tr_dataset.map(lambda x:x[0],num_parallel_calls=NUM_THREADS)
    tr_dataset_back = tr_dataset.map(lambda x:x[1],num_parallel_calls=NUM_THREADS)
    tr_dataset_poses = tr_dataset_poses.shuffle(buffer_size=50) 
    tr_dataset = tf.data.Dataset.zip((tr_dataset_poses, tr_dataset_back))

    tr_dataset = tr_dataset.shuffle(buffer_size=20)
    tr_dataset = tr_dataset.map(
          lambda pose_path,back_path: tuple(tf.py_func(_read_py_function,
                                      [(pose_path,back_path)],
                                      [tf.float32,tf.float32])),
                                      num_parallel_calls=NUM_THREADS)

    tr_dataset = tr_dataset.batch(batch_size)
    tr_dataset = tr_dataset.prefetch(batch_size)

    iterator = tr_dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    return iterator , next_element


