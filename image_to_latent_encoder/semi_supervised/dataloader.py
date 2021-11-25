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
from sklearn.utils import shuffle
import pickle 

H = Hyperparameters ()

def get_data(data_path):
    
    if '.pkl' in data_path :
        data = pickle.load(open(data_path,'rb'),encoding='latin1')
        images_path = data['train']['image_paths']['source']
        images_path = [str(i) for i in images_path]
        poses_3d = data['train']['joints_3d_17']['source']
    if '.mat' in data_path :
        data = sio.loadmat(data_path)
        images_path = [str(i).strip() for i in data['images_path']]
        poses_3d = data['poses_3d']
    
    return images_path,poses_3d

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

def x_flip(sk_17,neg):
    left_indices = [2, 3, 4, 9, 10, 11, 12]
    right_indices = [5, 6, 7, 13, 14, 15, 16]
    sk_17_flip = sk_17.copy()
    l_idx, r_idx = left_indices, right_indices
    sk_17_flip[:, l_idx] = sk_17[:, r_idx]
    sk_17_flip[:, r_idx] = sk_17[:, l_idx]
    sk_17_flip[:, :, 0] *= neg
    sk_17 = sk_17_flip
    return sk_17



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


def _parse_function(images, poses):
    """Obtain the image from the images (for both training and validation).
    The following operations are applied:
        - Decode the image from jpeg format
    """
    image_string = tf.read_file(images)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)    
    image_decoded.set_shape([224,224,3])
    image_decoded = image_decoded[:,:, ::-1]
    return image_decoded, poses

def _normalize_data(images, poses):
    """Normalize images and poses within range 0-1."""
    images = tf.cast(images, tf.float32)
    images = tf.divide(images, tf.constant(255.0, tf.float32))
    images = tf.subtract(images, tf.constant(0.4, tf.float32))
    return images, poses

def _read_py_function(filename,pose):
    image = cv2.imread(filename.decode("utf-8") )
    image  = cv2.resize(image,(224,224))    
    image = image / 255.0

    pose_3d = pose

    
    if np.random.rand() < H.jitter_prob:
        image = do_aug(image)
    
    if np.random.rand() <  H.flip_prob:
        image = flip_image(image)
        pose_3d = x_flip(np.expand_dims(pose_3d,0),-1)
        pose_3d = np.squeeze(augment_pose_seq(pose_3d , z_limit=(180,180),y_limit=(0,0)))
        
    if np.random.rand() < H.tilt_prob :
        angle = np.random.randint(-H.tilt_limit, H.tilt_limit)
        image = rotate_image(image, angle)
        
        R = get_rotmat_camera(np.radians(angle),0,0)
        X = np.matmul(R, np.transpose(pose_3d, [1,0]))
        pose_3d = np.transpose(X, [1,0])       
        
    img = image - ( np.array([0.4])).astype(np.float32)
    img = img[:,:,::-1]
        
    return img.astype(np.float32),pose_3d.astype(np.float32)

def data_loader(data_paths , batch_size, NUM_THREADS): 
    
    images_path = []
    poses_3d = []
    poses_2d = []
    
    for path in data_paths :
        im,pose_3d = get_data(path)
        images_path.append(im)
        poses_3d.append(pose_3d)
    
    images_path = np.concatenate(images_path)
    '''
    ### 128 machine
    images_path = ['/sdb/jogendra/anirudh/mounts/503/'+i for i in images_path]
    '''
    
    poses_3d = np.concatenate(poses_3d)
    assert len(images_path) == len(poses_3d) , "Image list and pose list should have same length"
    
    num_samples = len(images_path)
    print ("number of train samples : ", num_samples )
    
    tr_dataset = tf.data.Dataset.from_tensor_slices((images_path, poses_3d))
    tr_dataset = tr_dataset.shuffle(num_samples)
    tr_dataset = tr_dataset.map(
          lambda filename,pose: tuple(tf.py_func(_read_py_function,
                                                      [filename,pose],
                                                      [tf.float32,tf.float32])),
                                                      num_parallel_calls=NUM_THREADS)

    tr_dataset = tr_dataset.batch(batch_size)
    tr_dataset = tr_dataset.prefetch(batch_size)
    
    iterator = tr_dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    return iterator , next_element


def get_list_mp(data_paths):
    images_path = []
    poses_3d = []
    poses_2d = []
    
    for path in data_paths :
        im,pose_3d = get_data(path)
        images_path.append(im)
        poses_3d.append(pose_3d)
    
    images_path = np.concatenate(images_path)
    poses_3d = np.concatenate(poses_3d)
    assert len(images_path) == len(poses_3d) , "Image list and pose list should have same length"      
    return images_path, poses_3d


def process_mp(arr):
    filename = arr[0]
    image = cv2.imread(filename)
    image  = cv2.resize(image,(224,224))    
    image = image / 255.0

    pose_3d = arr[1]

    if np.random.rand() < H.jitter_prob:
        image = do_aug(image)
    
    if np.random.rand() <  H.flip_prob:
        image = flip_image(image)
        pose_3d = x_flip(np.expand_dims(pose_3d,0),-1)
        pose_3d = np.squeeze(augment_pose_seq(pose_3d , z_limit=(180,180),y_limit=(0,0)))
        
    if np.random.rand() < H.tilt_prob :
        angle = np.random.randint(-H.tilt_limit, H.tilt_limit)
        image = rotate_image(image, angle)
        
        R = get_rotmat_camera(np.radians(angle),0,0)
        X = np.matmul(R, np.transpose(pose_3d, [1,0]))
        pose_3d = np.transpose(X, [1,0])       
        
    img = image - ( np.array([0.4])).astype(np.float32)
    img = img[:,:,::-1]
        
    return img.astype(np.float32),pose_3d.astype(np.float32)
# def 

#     images_path, poses_3d = shuffle(images_path, poses_3d, random_state=0)


#     num_chunks = len(images_path) // 
    
#     lst = range(50)
# np.array_split(lst, 5)
    
    
    
    
    
    
    
    
    
    
