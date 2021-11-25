import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import distributions as ds
from model_componets import EncoderNet, DecoderNet, DiscriminatorNet, get_parent_relative_joints, get_root_relative_joints, ZDiscriminatorNet
from model_componets import num_joints, num_zdim, num_params_per_joint, num_params_total, num_z_angles
import model_componets as comps
from termcolor import colored
num_interp = 10 

def interp_vector(a, b):
    arr = []
    for x in range(num_zdim):
        arr.append(tf.linspace(a[x], b[x], num_interp))
    arr = tf.stack(arr, 1)
    return arr

def interp_vector_set(a, b):
    arr_interp = []
    for x in range(num_interp):
        arr_interp.append(interp_vector(a[x], b[x]))
    arr_interp = tf.concat(arr_interp, 0)
    return arr_interp


###################################################################################
############################### Define Placeholders ###############################

# Network Inputs
input_x = tf.placeholder(tf.float32, shape=[256, num_joints, num_params_per_joint], name='input_x')
input_x_view_norm = tf.placeholder(tf.float32, shape=[256, num_joints, num_params_per_joint], name='input_x_view_norm')
input_z = tf.placeholder(tf.float32, shape=[256, num_zdim], name='input_decoder_z_joints')


# Learning Rates
lr_disc_ph = tf.placeholder(tf.float32, shape=[], name='lr_disc_ph')
lr_encoder_ph = tf.placeholder(tf.float32, shape=[], name='lr_encoder_ph')
lr_decoder_ph = tf.placeholder(tf.float32, shape=[], name='lr_decoder_ph')

# Weight Vector
weight_vec_ph = tf.placeholder(tf.float32, shape=[23, ], name='weight_vec_ph')


###################################################################################
########################### Define Model Architechture ############################

### First Cycle ####

x_real = input_x
x_view_norm_real, x_local_real = comps.root_relative_to_local(x_real)
encoder_real = EncoderNet(x_local_real)
z_real = encoder_real['z_joints']
decoder_real = DecoderNet(z_real)
x_local_recon = decoder_real['full_body_x']
x_recon = comps.local_to_root_relative(x_local_recon)
x_recon_resnet = comps.local_to_view_norm(x_local_recon)
x_real_dummy = comps.view_norm_to_root_relative(x_view_norm_real)
x_real_vs_dummy = tf.reduce_mean(tf.abs(x_real - x_real_dummy))

### Second Cycle ####
z_rand = input_z
decoder_fake = DecoderNet(z_rand)
x_local_fake = decoder_fake['full_body_x']
encoder_fake = EncoderNet(x_local_fake)
z_recon = encoder_fake['z_joints']
x_view_norm_fake = comps.local_to_view_norm(x_local_fake)#TODO


####################################################################################
############################### Disc for x #########################################
disc_real = DiscriminatorNet(x_view_norm_real)
disc_real_x_logits = disc_real['fcc_logits']

### Disc for x_hat ###
disc_fake = DiscriminatorNet(x_view_norm_fake)
disc_fake_x_logits = disc_fake['fcc_logits']

### Prediction for x_view_norm
z_pred_view_norm = comps.EncoderNet(comps.view_norm_to_local(input_x_view_norm))['z_joints']

###################################################################################
################################ Define losses ####################################

### Adversarial loss ###
tensor_loss_disc_real = tf.reduce_mean(tf.abs(disc_real_x_logits - tf.ones_like(disc_real_x_logits)), axis=0)
tensor_loss_disc_fake = tf.reduce_mean(tf.abs(disc_fake_x_logits + tf.ones_like(disc_fake_x_logits)), axis=0)
tensor_loss_gen_adv = tf.reduce_mean(tf.abs(disc_fake_x_logits - tf.ones_like(disc_fake_x_logits)), axis=0)

tensor_loss_disc_adv = (tensor_loss_disc_real + tensor_loss_disc_fake) / 2.

loss_disc_adv = tf.reduce_sum(weight_vec_ph * tensor_loss_disc_adv)

loss_gen_adv = tf.reduce_sum(weight_vec_ph * tensor_loss_gen_adv)

## Define accuracy ##
disc_acc_real = tf.reduce_mean(tf.cast(disc_real_x_logits >= 0, tf.float32)) * 100.0
disc_acc_fake = tf.reduce_mean(tf.cast(disc_fake_x_logits < 0, tf.float32)) * 100.0

disc_acc = (disc_acc_real + disc_acc_fake) / 2.
gen_acc = 100 - disc_acc_fake

### Cyclic loss ###
# [B, ]
tensor_x_loss = tf.reduce_mean((x_recon - x_real) ** 2, axis=[1, 2])
tensor_z_loss = tf.reduce_mean((z_rand - z_recon) ** 2, axis=1)
x_recon_loss_all = tf.reduce_mean(tf.abs(x_recon - x_real))
x_recon_loss_l1 = x_recon_loss_all #+  x_recon_loss_l1_only_left + x_recon_loss_l1_only_right
z_recon_loss_l1 = tf.reduce_mean(tf.abs(z_rand - z_recon))
tensor_c_loss = tensor_x_loss + tensor_z_loss
loss_x_recon = tf.reduce_mean(tensor_x_loss)
loss_z_recon = tf.reduce_mean(tensor_z_loss)



# loss_cyclic = loss_x_recon + loss_z_recon
loss_cyclic = x_recon_loss_l1 + z_recon_loss_l1

### Total loss ###
loss_disc = loss_disc_adv 
loss_encoder = 100 * loss_cyclic #+ 5 * loss_gen_z
loss_decoder = 100* loss_cyclic + 5 * loss_gen_adv


##############################bat#####################################################
########################## Define operations ######################################

def get_network_params(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)


param_batch_disc = get_network_params(scope='BatchDiscNet')
param_disc = get_network_params(scope='DiscNet')
param_encoder = get_network_params(scope='Encoder_net')
param_decoder = get_network_params(scope='Decoder_net')

train_only_x_op = tf.train.AdamOptimizer(learning_rate = lr_encoder_ph).minimize(x_recon_loss_l1 , var_list = param_encoder + param_decoder)
disc_train_op = tf.train.AdamOptimizer(learning_rate=lr_disc_ph).minimize(loss_disc, var_list=param_disc)
encoder_train_op = tf.train.AdamOptimizer(learning_rate=lr_encoder_ph).minimize(loss_encoder, var_list=param_encoder)
decoder_train_op = tf.train.AdamOptimizer(learning_rate=lr_decoder_ph).minimize(loss_decoder, var_list=param_decoder)

###################################################################################
############################# Get Parameters #######################################
param_disc = get_network_params(scope='DiscNet')
param_encoder = get_network_params(scope='Encoder_net')
param_decoder = get_network_params(scope='Decoder_net')

#################################################################################################
############################ Summary for the Generator images ###################################

scalars = [
    tf.summary.scalar('loss_disc', loss_disc_adv),
    tf.summary.scalar('disc_acc', disc_acc),
    tf.summary.scalar('disc_acc_fake', disc_acc_fake),
    tf.summary.scalar('disc_acc_real', disc_acc_real),
    tf.summary.scalar('gen_acc', gen_acc),
    tf.summary.scalar('loss_encoder', loss_encoder),
    tf.summary.scalar('loss_decoder', loss_decoder),
    tf.summary.scalar('loss_cyclic', loss_cyclic),
    tf.summary.scalar('loss_gen_adv', loss_gen_adv),
    tf.summary.scalar('loss_x_recon', loss_x_recon),
    tf.summary.scalar('loss_z_recon', loss_z_recon),
    tf.summary.scalar('loss_x_recon_l1', x_recon_loss_all),
    tf.summary.scalar('loss_z_recon_l1', z_recon_loss_l1),
    tf.summary.scalar('loss_real_dummy', x_real_vs_dummy),
]


z_j_histograms = [
    tf.summary.histogram('z_j_%02d' % i, z_real[:, i]) for i in range(num_zdim)
]


###################   content loss summaries   ####################################################
summary_merge_all = tf.summary.merge(scalars + z_j_histograms )
summary_merge_valid = tf.summary.merge(scalars + z_j_histograms)


def load_weights(iter_no, session, dir='pretrained_weights'):
    print ('trying to load iter weights...')
    print (colored(dir,"blue"))
    tf.train.Saver().restore(session,'./weights/AE_humans_mads_yt_mpi_l1/whole/whole-481000')
    print('Loaded iter weights')
    return iter_no


def load_best_weights(iter_no, session):
    try:
        print ('trying to load best weights...')
        tf.train.Saver(param_decoder).restore(session, 'pretrained_weights/decoder_best-%d' % iter_no)
        tf.train.Saver(param_encoder).restore(session, 'pretrained_weights/encoder_best-%d' % iter_no)
        tf.train.Saver(param_disc).restore(session, 'pretrained_weights/disc_best-%d' % iter_no)
    except Exception as ex:
        print('Could not load best weight... Trying to load Iter...')
    return iter_no


