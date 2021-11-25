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
from tensorflow.python import debug as tf_debug
import re

import motion_components
import cnn_model
from data_loader import datasets, data_loaders
from hyperparams import Hyperparameters
H = Hyperparameters ()


def pose_flip_tr(pose_embed,name='FlipRule'): # pose_embed shape : [None,32]
    mapped = motion_components.apply_rule_net(pose_embed,name, state_size = 32)
    return mapped 
    
def flip_backward_tr(motion_embed,name='flip_backward'): # motion_embed shape : [batch_size,seq_length,128]
    mapped = motion_components.apply_motion_rule_net(motion_embed,name)
    rule_state = mapped['mapped_state'] 
    return rule_state

def slow_backward_tr(motion_embed,name='SlowBackward'): # motion_embed shape : [batch_size,seq_length,128]
    mapped = motion_components.apply_motion_rule_net(motion_embed,name)
    rule_state = mapped['mapped_state'] 
    return rule_state

def motion_encoder(encoder_input,name='motion_encoder'):
    encoder_lstm_out=motion_components.apply_encoder(encoder_input,name)
    z_state = encoder_lstm_out['z_state']
    return z_state


#######################################################
#           SESSION  DEFINITION
#######################################################

os.environ['CUDA_VISIBLE_DEVICES'] = H.cuda_device_id
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)

#######################################################
#              DATALOADER  
#######################################################

#create tf.data.Dataset from the lists
datasets = [tf.data.Dataset.from_tensor_slices((x)) for x in datasets] 
for idx, data_loader in enumerate(data_loaders):
    datasets[idx] = datasets[idx].map(lambda z: tf.py_func(data_loader, [z], [tf.float32, \
            tf.float32]), 32)

dset_names = ["l_lcr", "l_hcr", "l_1_z", "l_1_v", "l_2_v"]
datasets = [dataset.shuffle(buffer_size = 10) for dataset in datasets]
datasets = [dataset.batch(H.batch_size) for dataset in datasets]
datasets = [dataset.prefetch(10) for dataset in datasets]
dataset_iterators = [dataset.make_initializable_iterator() for dataset in datasets]
string_handles = [sess.run(i.string_handle()) for i in dataset_iterators]
handle_pl = tf.placeholder(tf.string, shape=[])
base_iterator = tf.data.Iterator.from_string_handle(handle_pl, datasets[0].output_types, datasets[0].output_shapes)


if not os.path.exists('./logs'):
    os.makedirs('./logs')
    


#######################################################
#              GRAPH  
#######################################################
image_seq, positive_seq = base_iterator.get_next()
image_seq = tf.reshape(image_seq,[H.batch_size * H.seq_length, 224, 224, 3])
positive_seq = tf.reshape(positive_seq,[H.batch_size * H.seq_length, 224, 224, 3])

image_features = cnn_model.create_network(image_seq,'')
resnet_params = cnn_model.get_network_params('')
train_params = [x for x in resnet_params if "res3" in x.op.name]
image_embedding, image_rep = cnn_model.embedding_branch(image_features)

positive_features = cnn_model.create_network(positive_seq, '')
positive_embedding, positive_rep = cnn_model.embedding_branch(positive_features)

image_cr_embedding = cnn_model.mlp_head(image_rep, "pose_mlp_head")
pos_cr_embedding = cnn_model.mlp_head(positive_rep, "pose_mlp_head")

train_params_pose = cnn_model.get_network_params("pose_mlp_head")

image_cr_seq = tf.reshape(image_cr_embedding, [H.batch_size, H.seq_length, 128])
pos_cr_seq = tf.reshape(pos_cr_embedding, [H.batch_size, H.seq_length, 128])

image_seq_embedding = tf.reshape(image_embedding, [H.batch_size, H.seq_length, 32])
positive_seq_embedding = tf.reshape(positive_embedding, [H.batch_size, H.seq_length, 32])
positive_seq_embedding = tf.stop_gradient(positive_seq_embedding)


motion_embedding = motion_encoder(image_seq_embedding)
pos_motion_embedding = motion_encoder(positive_seq_embedding)

motion_cr_embedding = cnn_model.mlp_head(motion_embedding, "motion_mlp_head")
positive_cr_embedding = cnn_model.mlp_head(pos_motion_embedding, "motion_mlp_head")

train_params_motion = cnn_model.get_network_params("motion_mlp_head")

l1_loss = lambda gt, pred : tf.reduce_mean(tf.abs(gt-pred))

#############
# Pose flip
#############

flipped_embedding = pose_flip_tr(image_embedding)
loss1 = l1_loss(flipped_embedding, positive_embedding)
opt1 = tf.train.AdamOptimizer(H.learning_rate).minimize(loss = loss1,var_list = train_params)
#############
# Flip B/W
#############

flip_bw_embedding = flip_backward_tr(motion_embedding)
loss2 = l1_loss(flip_bw_embedding, pos_motion_embedding)
opt2 = tf.train.AdamOptimizer(H.learning_rate).minimize(loss = loss2,var_list = train_params)
#############
# Slow B/W
#############
slow_bw_embedding = slow_backward_tr(motion_embedding)
loss3 = l1_loss(slow_bw_embedding, pos_motion_embedding)
opt3 = tf.train.AdamOptimizer(H.learning_rate).minimize(loss = loss3,var_list = train_params)


############
# Higher order CR loss
############
T = 0.07
motion_embedding = tf.math.l2_normalize(motion_cr_embedding, axis = 1)
pos_motion_embedding = tf.math.l2_normalize(positive_cr_embedding, axis = 1)


pos = tf.exp(tf.reduce_sum(motion_embedding[0, :] * pos_motion_embedding[0, :])/T)
tot = tf.reduce_sum(tf.exp(tf.reduce_sum(motion_embedding[0,:] * motion_embedding[1:, :], 1)/T))
tot = tot + pos
loss4 = -tf.log(pos/tot)

opt4 = tf.train.AdamOptimizer(H.learning_rate*2).minimize(loss = loss4,var_list = train_params+train_params_motion)

#############
# Lower order CR loss
#############

#sample every 4th frame of the sequence, giving us 8 frames sampled from each seq.
image_seq_embedding = tf.math.l2_normalize(image_cr_seq[:, 0, :], axis = 1)
positive_seq_embedding = tf.math.l2_normalize(pos_cr_seq[:, 0, :], axis = 1)

image_seq_embedding = tf.reshape(image_seq_embedding, [H.batch_size, 128])
positive_seq_embedding = tf.reshape(positive_seq_embedding, [H.batch_size, 128])

pos = tf.exp(tf.reduce_sum(image_seq_embedding[0,:] * positive_seq_embedding[0,:])/T)
tot = tf.reduce_sum(tf.exp(tf.reduce_sum(image_seq_embedding[0,:] * image_seq_embedding[1:, :], 1)/T))
tot = tot + pos
loss5 = -tf.log(pos/tot)


opt5 = tf.train.AdamOptimizer(H.learning_rate*2).minimize(loss = loss5,var_list = train_params+train_params_pose)


#############
# add all train ops to a list
#############
opts = [opt5, opt4, opt1, opt2, opt3]
train_ops = opts #[]
losses = [loss5, loss4, loss1, loss2, loss3]



pose_flip_tr_params = cnn_model.get_network_params("FlipRule")
flip_backward_tr_params = cnn_model.get_network_params("flip_backward")
slow_backward_tr_params = cnn_model.get_network_params("SlowBackward")
motion_encoder_params = cnn_model.get_network_params("motion_encoder")
embedding_branch_params = cnn_model.get_network_params("embedding")


#######################################################
#              UTILS 
#######################################################

'''
Saving weights
'''
resnet_branch_weights = tf.train.Saver(resnet_params,max_to_keep=5)
train_writer = tf.summary.FileWriter(H.logdir_path_train,sess.graph)

'''
Loading weights
'''
sess.run(tf.global_variables_initializer())

tf.train.Saver(resnet_params).restore(sess,tf.train.latest_checkpoint(H.store_resnet_weights_path))
print(colored("restored resnet branch weights","yellow"))

tf.train.Saver(pose_flip_tr_params).restore(sess, tf.train.latest_checkpoint(H.flip_tr_path))
tf.train.Saver(flip_backward_tr_params).restore(sess, tf.train.latest_checkpoint(H.fb_tr_path))
tf.train.Saver(slow_backward_tr_params).restore(sess, tf.train.latest_checkpoint(H.sb_tr_path))
print(colored("restored transformer weights","yellow"))

tf.train.Saver(motion_encoder_params).restore(sess, H.motion_encoder_path)
print(colored("restored motion encoder weights","yellow"))

tf.train.Saver(embedding_branch_params).restore(sess, tf.train.latest_checkpoint(H.store_embedding_branch_weights_path))
print(colored("restored embedding branch weights","yellow"))


'''
Summary writer
'''
def get_summary_str(val, tag):

    summary_str = tf.Summary()
    summary_str.value.add(tag = tag, simple_value = val)

    return summary_str

#######################################################
#              TRAINING 
#######################################################
iteration = 0

for iterator in dataset_iterators:
    sess.run(iterator.initializer)

print("starting training loop!")
while True:
    dset = iteration%5
    try:
        lval, _ = sess.run([losses[dset], train_ops[dset]], feed_dict = \
                {handle_pl : string_handles[dset]})

        print("Iteration %d: loss %s value : %.5f"%(iteration, dset_names[dset], lval))
        summary_str = get_summary_str(lval, "losses/%s"%dset_names[dset])
        train_writer.add_summary(summary_str, iteration)
        train_writer.flush() # write to disk now
        iteration += 1
    except tf.errors.OutOfRangeError:
        sess.run(dataset_iterators[dset].initializer)

