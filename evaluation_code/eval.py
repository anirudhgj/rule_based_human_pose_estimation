import tensorflow as tf
import os
import glob
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as ming
import matplotlib.pyplot as plt
import random
from scipy import ndimage
import cv2
from termcolor import colored
from tensorflow.contrib import slim

import model_componets as comps
import cnn_model
import utils
from data_loader import get_h36_test, data_loader_h36 
from losses import compute_mpjpe_summary
from hyperparams import Hyperparameters
from tensorboard_logging import Logger
H = Hyperparameters ()



#######################################################
#           SESSION  DEFINITION
#######################################################

os.environ['CUDA_VISIBLE_DEVICES'] = H.cuda_device_id
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)


#######################################################
#              DATALOADER  
#######################################################


h36_test_data = get_h36_test()
h36_test_dataset = tf.data.Dataset.from_tensor_slices((h36_test_data))
dataset_human_test = h36_test_dataset.map(lambda z: tf.py_func(data_loader_h36, [z], [tf.float32, \
        tf.float32]), 32)
dataset_human_test = dataset_human_test.batch(H.batch_size)
dataset_human_test = dataset_human_test.prefetch(2) 
test_iterator = dataset_human_test.make_initializable_iterator()
test_string_handle = sess.run(test_iterator.string_handle())
handle_pl = tf.placeholder(tf.string, shape=[])
base_iterator = tf.data.Iterator.from_string_handle(handle_pl, dataset_human_test.output_types, dataset_human_test.output_shapes)


if not os.path.exists('./logs'):
    os.makedirs('./logs')
    


#######################################################
#              GRAPH  
#######################################################
image_ph, pose_ph = base_iterator.get_next()
image_ph = tf.reshape(image_ph,[H.batch_size, 224, 224, 3])
pose_ph = tf.reshape(pose_ph,[H.batch_size, 17, 3])

resnet_out = cnn_model.create_network(image_ph,'')
resnet_params = cnn_model.get_network_params('')
fc_embedding = cnn_model.embedding_branch(resnet_out)
fc_embedding_params = cnn_model.get_network_params('embedding')
pose_decoded = comps.DecoderNet(fc_embedding)
pose_decoded_params = cnn_model.get_network_params('Decoder_net')
pose_local_recon = pose_decoded['full_body_x']
pose_recon = comps.local_to_root_relative(pose_local_recon)


pose_pred = pose_recon
#######################################################
#              UTILS 
#######################################################

'''
Saving weights
'''
resnet_branch_weights = tf.train.Saver(resnet_params,max_to_keep=5)
embedding_branch_weights = tf.train.Saver(fc_embedding_params,max_to_keep=5)
decoder_branch_weights = tf.train.Saver(pose_decoded_params,max_to_keep=5)
train_writer = tf.summary.FileWriter(H.logdir_path_train,sess.graph)
val_writer = tf.summary.FileWriter(H.logdir_path_val)
train_L = Logger('./logs/',train_writer)
val_L = Logger('./logs/',val_writer)

'''
Loading weights
'''
sess.run(tf.global_variables_initializer())
tf.train.Saver(resnet_params).restore(sess,tf.train.latest_checkpoint(H.store_resnet_weights_path))
print(colored("restored resnet branch weights","yellow"))

tf.train.Saver(fc_embedding_params).restore(sess,tf.train.latest_checkpoint(H.store_embedding_branch_weights_path))
print(colored("Restored Embedding Branch weights","green"))

tf.train.Saver(pose_decoded_params).restore(sess,'../pretrained_weights/pose_encoder_decoder/decoder_iter-1475001')
print(colored("Restored Decoder weights","blue"))

def post_process(z):
    one_col = z[:,[0]]
    two_col = z[:,[1]]
    three_col = z[:,[2]]
    z[:,[0]] = one_col
    z[:,[1]] = three_col
    z[:,[2]] = -two_col
    pose = z.copy()
    poses_3d = np.squeeze(utils.augment_pose_seq(np.expand_dims(pose,0) , z_limit=(180,180),y_limit=(0,0)))    
    return poses_3d

#######################################################
#              TRAINING 
#######################################################
mplist = []
sess.run(test_iterator.initializer)
while True:
    try:
        images, pose_3d, pred_3d = sess.run([image_ph, pose_ph, pose_pred], feed_dict = {handle_pl : test_string_handle})
        pose_3d = np.array([post_process(i) for i in pose_3d])
        mpjpe = compute_mpjpe_summary(pose_3d, pred_3d)
        mplist.append(mpjpe)   
    except tf.errors.OutOfRangeError:
        print("Test MPJPE: %3f"%(sum(mplist)/len(mplist)))
        break
