#################################################################################
############################# Load Libraries ####################################
import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import matplotlib
matplotlib.use("Agg")
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
from termcolor import colored
import logging,argparse


from data_loader import Data_loader
from hyperparams import Hyperparameters
import pose_components
import motion_components
import utils
    


#################################################################################
############################# Start Session #####################################
print (colored("code started","red"))
H = Hyperparameters ()
D = Data_loader(H.data_path,H.seq_length,H.batch_size,H.num_joints)
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)

if not os.path.exists('./weights/'):
    os.makedirs('./weights')

#################################################################################
############################# graph ############################################

input_ph = tf.placeholder(tf.float32, shape= [None , H.num_joints ,3],name = 'skeleton_input')
global_step = tf.train.get_or_create_global_step()

encoder_out = pose_components.apply_pose_encoder(input_ph)#root relative
pose_encoder_params = pose_components.get_network_params("Encoder_net")#embeddings
encoder_input = tf.reshape(encoder_out,(-1,H.seq_length,32))#sequence of embeddings
encoder_lstm_out = motion_components.apply_encoder(encoder_input,name ='motion_encoder')
z_state = encoder_lstm_out['z_state']
z_outputs = encoder_lstm_out['z_outputs']
decoder_lstm_out = motion_components.apply_decoder(z_state,z_outputs,name = 'motion_decoder')
motion_recon = decoder_lstm_out['x_recon']
motion_recon_reshaped = tf.reshape(motion_recon,((-1,32)))
pose_recon = pose_components.apply_pose_decoder(motion_recon_reshaped)#view norm
pose_decoder_params = pose_components.get_network_params("Decoder_net")
loss_motion = tf.reduce_mean((pose_recon - input_ph)**2)
loss_motion_summary = tf.summary.scalar('loss',loss_motion)
summary_op = tf.summary.merge_all()

param_lstm_encoder = motion_components.get_network_params('motion_encoder')
param_lstm_decoder = motion_components.get_network_params('motion_decoder')

vars_to_minimize = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='motion_encoder') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='motion_decoder')

motion_loss_optimizer  = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(loss =loss_motion,global_step=global_step,var_list = vars_to_minimize)

sess.run(tf.global_variables_initializer())

#########################################################################################
############################# Loading Weights ############################################

# print (colored("loading weights","blue"))

# tf.train.Saver(pose_encoder_params).restore(sess,
#                                             '../pose_embedding_training_non_norm_range_ten_vneck/pretrained_weights/encoder_iter-1475001')
# print (colored("loaded pose_encoder weights","yellow"))

# tf.train.Saver(pose_decoder_params).restore(sess,
#                                             '../pose_embedding_training_non_norm_range_ten_vneck/pretrained_weights/decoder_iter-1475001')
# print colored("loaded pose_decoder weights","blue")


summary_writer_train = tf.summary.FileWriter(H.logdir_path_train, graph=tf.get_default_graph())
summary_writer_val = tf.summary.FileWriter(H.logdir_path_val, graph=tf.get_default_graph())

lstm_encoder_weights = tf.train.Saver(param_lstm_encoder)
lstm_decoder_weights = tf.train.Saver(param_lstm_decoder)
saver = tf.train.Saver()

#########################################################################################
############################# Start training ############################################
print (colored("started training","magenta"))

def train():

    for iteration_no in range(0,H.max_iterations):

        train_batch = np.asarray(D.get_sequence_batch_train())
        train_batch = train_batch[:,0:30]
        train_batch = np.reshape(utils.augment_pose_seq(train_batch),(-1,H.num_joints * 3))
        train_batch = train_batch.reshape((-1,H.num_joints , 3))
        feed_dict = {input_ph : train_batch}
        op_train_dict = sess.run({'loss':loss_motion,
                                  'optim':motion_loss_optimizer,
                                  'summary_op':summary_op,
                                  'g_step': global_step,
                                  'final_pose': pose_recon}, feed_dict=feed_dict)
    
        if iteration_no % 10 == 0:
        
            print ("global step", op_train_dict['g_step'],"train loss", op_train_dict['loss'])
            fig = plt.figure(figsize=(16, 8))
            fig_img = utils.gen_plot(fig,
                                     train_batch[:H.seq_length].reshape([H.seq_length, H.num_joints, 3]),
                                     op_train_dict['final_pose'][:H.seq_length].reshape([H.seq_length, H.num_joints, 3]),
                                     az=90)
            
            fig.savefig('test_train.png')
            fig_img = cv2.imread('test_train.png')[:,:,::-1]
            utils.log_images('output_pose', fig_img, op_train_dict['g_step'], summary_writer_train)
            plt.close()

        if iteration_no % 50 == 0:

            val(iteration_no)

        if iteration_no % 100 == 0:

            lstm_encoder_weights.save(sess,H.store_encoder_weights+H.exp_name+str(op_train_dict['g_step']))
            lstm_decoder_weights.save(sess,H.store_decoder_weights+H.exp_name+str(op_train_dict['g_step']))
            saver.save(sess,H.store_weights_path+H.exp_name+str(op_train_dict['g_step']))

        summary_writer_train.add_summary(op_train_dict['summary_op'], op_train_dict['g_step'])

        summary_writer_train.flush()

def val(iteration_no):

    val_batch = np.asarray(D.get_sequence_batch_valid())
    val_batch = val_batch[:,0:30]
    val_batch = val_batch.reshape((-1,H.num_joints,3))
    feed_dict = {input_ph : val_batch}
    op_val_dict = sess.run({'loss':loss_motion,
                            'summary_op':summary_op,
                            'g_step': global_step,
                            'final_pose': pose_recon}, feed_dict=feed_dict)

    fig = plt.figure(figsize=(16, 8))

    fig_img = utils.gen_plot(fig, val_batch[:H.seq_length].reshape([H.seq_length, H.num_joints, 3]), op_val_dict['final_pose'][:H.seq_length].reshape([H.seq_length, H.num_joints, 3]), az=90)

    fig.savefig('test_valid.png')
    fig_img = cv2.imread('test_valid.png')[:, :, ::-1]
    utils.log_images('output_pose', fig_img, op_val_dict['g_step'], summary_writer_val)
    print ("global step", op_val_dict['g_step'],"val loss", op_val_dict['loss'])
    summary_writer_val.add_summary(op_val_dict['summary_op'], op_val_dict['g_step'])
    summary_writer_val.flush()
    plt.close()


if __name__ == '__main__':
    train()
