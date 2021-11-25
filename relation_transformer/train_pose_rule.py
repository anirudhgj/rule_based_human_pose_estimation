#################################################################################
############################# Load Libraries ####################################
import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
import scipy.io as sio
import io
import cv2


import motion_components
import pose_components
from data_loader_pose import DataLoader
import vis_image as vis
import utils

if not os.path.exists('./weights/FlipTiltRule/') :
    os.makedirs('./weights/FlipTiltRule/')  
    
if not os.path.exists('./logs_flip_tilt_rule/train/') :
    os.makedirs('./logs_flip_tilt_rule/train/')   
                                               
if not os.path.exists('./logs_flip_tilt_rule/val/') :
    os.makedirs('./logs_flip_tilt_rule/val/')  

data_loader = DataLoader()


#############################################################################
######################### Variables Initializer #############################
num_epochs = 100000
num_joints=17
rule_tilt=50
epoch = 0
iteration_no =0
train_batch_size, test_batch_size = 256, 256
lr_disc, lr_encoder, lr_decoder = 0.000002, 0.000002, 0.000002
disp_step_save, disp_step_valid = 1000, 500

num_batches_train = data_loader.get_num_batches('train',  train_batch_size)
num_batches_test = data_loader.get_num_batches('test',  test_batch_size)

#################################################################################
############################# Start Session #####################################
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)

#################################################################################
############################# graph ############################################  

input_ph = tf.placeholder(tf.float32, shape= [None ,num_joints,3],name = 'skeleton_input')#without transformation 
gt_ph = tf.placeholder(tf.float32 , shape = [None,num_joints, 3], name = 'gt_input') #with transformation

global_step = tf.train.get_or_create_global_step()


z_real = pose_components.apply_pose_encoder(input_ph)
rule_state = motion_components.apply_pose_rule_net(z_real,'FlipTiltRule')
pose_recon = pose_components.apply_pose_decoder(rule_state)

loss = tf.reduce_mean((pose_recon - gt_ph )** 2)
loss_summary = tf.summary.scalar('loss',loss)
summary_op = tf.summary.merge_all()

pose_encoder_params = pose_components.get_network_params("Encoder_net")#embeddings
pose_decoder_params = pose_components.get_network_params("Decoder_net")#embeddings
rule_network_params = pose_components.get_network_params('FlipTiltRule')

loss_optimizer  = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss =loss,
                                                                          global_step=global_step,
                                                                          var_list = rule_network_params)
sess.run(tf.global_variables_initializer())

#################################################################################
############################# Load Weights ######################################
                                               
tf.train.Saver(pose_encoder_params).restore(sess,'../pretrained_weights/pose_encoder_decoder/encoder_iter-1475001')
print (colored("loaded pose_encoder weights","yellow"))

tf.train.Saver(pose_decoder_params).restore(sess,'../pretrained_weights/pose_encoder_decoder/decoder_iter-1475001')
print (colored("loaded pose_decoder weights","blue"))

if os.path.exists('./weights/{}/'.format('FlipTiltRule')) and len(os.listdir('./weights/{}/'.format('FlipTiltRule'))) !=0 :
    tf.train.Saver(rule_network_params).restore(sess,tf.train.latest_checkpoint('./weights/{}/'.format('FlipTiltRule')))
    print (colored("loaded Rule weights","green"))                                                                                                                                           
summary_writer_train = tf.summary.FileWriter('./logs_flip_tilt_rule/train/', graph=tf.get_default_graph())
summary_writer_val = tf.summary.FileWriter('./logs_flip_tilt_rule/val/', graph=tf.get_default_graph())

rule_network_weights = tf.train.Saver(rule_network_params,max_to_keep=5)
                                                                              
def val(iteration_no):
    data_loader.shuffle_data('test')
    batch_idx_test=np.random.choice(np.arange(num_batches_test))                                     
    val_inputs = data_loader.get_test_data_batch(test_batch_size, batch_idx_test)
    val_batch = np.expand_dims(val_inputs,
                                 axis = 1)
    val_batch_tilt = np.squeeze(utils.augment_pose_seq(val_batch,
                                                       z_limit=(0,0),
                                                       y_limit=(rule_tilt,rule_tilt)),
                                                       axis = 1)
    val_batch_tilt=val_batch_tilt.reshape((-1,num_joints,3))
    val_inputs_flip_tilt = utils.x_flip(val_batch_tilt)

    feed_dict = {input_ph : val_inputs , gt_ph : val_inputs_flip_tilt}
    op_val_dict = sess.run({'loss':loss,
                            'summary_op':summary_op,
                            'g_step': global_step,
                            'pose_decoder_preds': pose_recon}, feed_dict=feed_dict)
    
    fig = plt.figure(figsize=(20, 12))
    fig_img = vis.gen_plot_3(fig,
                             val_inputs[0],
                             val_inputs_flip_tilt[0],
                             op_val_dict['pose_decoder_preds'][0])

    fig.savefig('test.png')
    fig_img = cv2.imread('test.png')[:,:,::-1]
    utils.log_images('output_pose', fig_img, op_val_dict['g_step'], summary_writer_val)
    plt.close()
    
    summary_writer_val.add_summary(op_val_dict['summary_op'], op_val_dict['g_step'])
    summary_writer_val.flush()

    print ("global step", op_val_dict['g_step'],"val loss", op_val_dict['loss'])

#########################################################################
############################# Train #####################################

while epoch < num_epochs:
    epoch += 1
    data_loader.shuffle_data('train')
    for batch_idx in range(num_batches_train):
        iteration_no += 1
        x_inputs = data_loader.get_train_data_batch(train_batch_size, batch_idx)
        x_batch = np.expand_dims(x_inputs,
                                 axis = 1)
        x_batch_tilt = np.squeeze(utils.augment_pose_seq(x_batch,
                                                         z_limit=(0,0),
                                                         y_limit=(rule_tilt,rule_tilt)),                                                                                                    axis = 1)
        x_batch_tilt = x_batch_tilt.reshape((-1,num_joints,3))    
        x_inputs_flip_tilt = utils.x_flip(x_batch_tilt)

        feed_dict = {input_ph:x_inputs , gt_ph:x_inputs_flip_tilt}
        op_train_dict = sess.run({'loss':loss,
                                  'optim':loss_optimizer,
                                  'summary_op':summary_op,
                                  'g_step': global_step,
                                  'pose_decoder_preds': pose_recon}, feed_dict=feed_dict)   
        
        if iteration_no % 70 == 0:
            fig = plt.figure(figsize=(20, 12))
            fig_img = vis.gen_plot_3(fig,x_inputs[0],
                           x_inputs_flip_tilt[0],
                           op_train_dict['pose_decoder_preds'][0])
            
            fig_img.savefig('train.png')
            fig_img = cv2.imread('train.png')[:,:,::-1]
            utils.log_images('output_pose', fig_img, op_train_dict['g_step'], summary_writer_train)
            plt.close()
            
        if iteration_no % 100== 0:
            print ("global step", op_train_dict['g_step'],"train loss", op_train_dict['loss'])


        if iteration_no % 50 == 0:
                val(iteration_no)

        if iteration_no % 100 == 0:
                rule_network_weights.save(sess,'./weights/FlipTiltRule/relation_tr_'+str(op_train_dict['g_step']))

        summary_writer_train.add_summary(op_train_dict['summary_op'], op_train_dict['g_step'])
        summary_writer_train.flush()
                                       

                                               
                                               
                                               