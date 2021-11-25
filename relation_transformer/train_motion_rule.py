#################################################################################
############################# Load Libraries ####################################
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
import logging,argparse
import time 
from termcolor import colored



from data_loader_seq import Data_loader
from hyperparams import Hyperparameters
import utils
import rules_numpy
import pose_components 
import motion_components


#################################################################################
############################# Start Session #####################################

H = Hyperparameters ()
D = Data_loader(H.data_path,H.seq_length,H.batch_size,H.num_joints)


os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)

if not os.path.exists('./weights/'+H.rule_network_name+'/'):
    os.makedirs('./weights/'+H.rule_network_name+'/')

    
#################################################################################
############################# graph ############################################    
    
input_ph = tf.placeholder(tf.float32, shape= [None , H.num_joints ,3], name = 'skeleton_input')#without transformation 
gt_ph = tf.placeholder(tf.float32 , shape = [None, H.num_joints , 3], name = 'gt_input') #with transformation
global_step = tf.train.get_or_create_global_step()

def build_graph(model_input,rule_name,gt_flag = False ):
    encoder_out = pose_components.apply_pose_encoder(model_input)#root relative
    pose_encoder_params = pose_components.get_network_params("Encoder_net")#embeddings
    encoder_input = tf.reshape(encoder_out,(-1,H.seq_length,32))#sequence of embeddings
    encoder_lstm_out = motion_components.apply_encoder(encoder_input,name ='motion_encoder')
    z_state = encoder_lstm_out['z_state']
    z_outputs = encoder_lstm_out['z_outputs']
    if gt_flag == False:
        rule_net_out = motion_components.apply_motion_rule_net(z_state,name=rule_name)  
        rule_state = rule_net_out['mapped_state'] 
        rule_outputs = rule_net_out['mapped_outputs'] 
    else :
        rule_net_out = encoder_lstm_out
        rule_state = z_state
        rule_outputs = z_outputs
    decoder_lstm_out = motion_components.apply_decoder(rule_state,rule_outputs,name = 'motion_decoder')
    motion_recon = decoder_lstm_out['x_recon']
    motion_recon_reshaped = tf.reshape(motion_recon,((-1,32)))
    pose_recon = pose_components.apply_pose_decoder(motion_recon_reshaped)
    pose_decoder_params = pose_components.get_network_params("Decoder_net")
    rule_meta_outputs = {
        'model_input' : model_input,
        'encoder_lstm_out' : encoder_lstm_out,
        'rule_net_out' : rule_net_out,
        'decoder_lstm_out' : decoder_lstm_out,
        'pose_recon' : pose_recon,
        'pose_encoder_params' : pose_encoder_params,
        'pose_decoder_params' : pose_decoder_params
    }

    return rule_meta_outputs

preds = build_graph(input_ph,H.rule_network_name,gt_flag = False)
gt = build_graph(gt_ph,H.rule_network_name,gt_flag = True)


preds_rule_state = preds['rule_net_out']['mapped_state']
gt_rule_state = gt['encoder_lstm_out']['z_state']

pose_decoder_preds = preds['pose_recon']


loss = tf.reduce_mean((preds_rule_state - gt_rule_state )** 2)
loss_summary = tf.summary.scalar('loss',loss)
summary_op = tf.summary.merge_all()

pose_encoder_params = preds['pose_encoder_params']
pose_decoder_params = preds['pose_decoder_params']
param_lstm_encoder = motion_components.get_network_params('motion_encoder')
param_lstm_decoder = motion_components.get_network_params('motion_decoder')
rule_network_params = motion_components.get_network_params(H.rule_network_name)

vars_to_minimize = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=H.rule_network_name) 
loss_optimizer  = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss=loss,
                                                                          global_step=global_step,
                                                                          var_list = vars_to_minimize)

sess.run(tf.global_variables_initializer())


#########################################################################################
############################# Loading Weights ############################################

# print (colored("loading weights","blue"))
# tf.train.Saver(pose_encoder_params).restore(sess,
#                                             '../pose_embedding_training_non_norm_range_ten_vneck/pretrained_weights/encoder_iter-1475001')
# print (colored("loaded pose_encoder weights","yellow"))

# tf.train.Saver(pose_decoder_params).restore(sess,
#                                             '../pose_embedding_training_non_norm_range_ten_vneck/pretrained_weights/decoder_iter-1475001')
# print (colored("loaded pose_decoder weights","blue"))

# tf.train.Saver(param_lstm_encoder).restore(sess,'../motion_embedding_training_non_norm_range_ten_vneck/pretrained_weights/lstm_encoder/motion_net_expt_seq30_HuMaMpi91500')
# print (colored("loaded pose lstm_encoder weights","green"))

# tf.train.Saver(param_lstm_decoder).restore(sess,'../motion_embedding_training_non_norm_range_ten_vneck/pretrained_weights/lstm_decoder/motion_net_expt_seq30_HuMaMpi91500')
# print (colored("loaded pose lstm_decoder weights","red"))



# if os.path.exists('./weights/{}/'.format(H.rule_network_name)) and len(os.listdir('./weights/{}/'.format(H.rule_network_name))) !=0 :
#     tf.train.Saver(vars_to_minimize).restore(sess,tf.train.latest_checkpoint('./weights/{}/'.format(H.rule_network_name)))
#     print (colored("loaded Rule weights","green"))


summary_writer_train = tf.summary.FileWriter(H.logdir_path_train, graph=tf.get_default_graph())
summary_writer_val = tf.summary.FileWriter(H.logdir_path_val, graph=tf.get_default_graph())

rule_network_weights = tf.train.Saver(rule_network_params)

saver = tf.train.Saver()

def get_data_train(seed,rule):
    if rule=='slow_backward_rule' :
        train_batch_tmp = np.asarray(D.get_sequence_batch_train(slow_rule= True))
        train_batch = train_batch_tmp[0]
        train_batch = np.asarray(utils.augment_pose_seq(train_batch,seed=seed,set_seed=True))
        train_batch = np.reshape((train_batch[:,0:30]),(-1,H.num_joints , 3))

        gt_batch = train_batch_tmp[1]
        gt_batch = np.asarray(utils.augment_pose_seq(gt_batch,seed=seed,set_seed=True))
        gt_batch_cp = gt_batch.reshape((-1,H.seq_length,H.num_joints * 3)).copy()
        gt_batch_slow_reverse = np.array([list(reversed(i)) for i in gt_batch_cp]).reshape((-1,H.num_joints,3))#sequence reverse
        return train_batch, gt_batch_slow_reverse
    
    if rule=='flip_backward_rule' :
        
        train_batch = np.asarray(D.get_sequence_batch_train())
        train_batch = np.asarray(utils.augment_pose_seq(train_batch))
        train_batch = np.reshape((train_batch[:,0:30]),(-1,H.num_joints , 3))
        train_batch_cp = train_batch.reshape((-1,H.seq_length,H.num_joints * 3)).copy()
        train_batch_flip = rules_numpy.x_flip(train_batch).reshape((-1,H.seq_length,H.num_joints * 3))
        gt_batch_flip_reverse = np.array([list(reversed(i)) for i in train_batch_flip]).reshape((-1,H.num_joints,3))#sequence reverse
        return train_batch , gt_batch_flip_reverse
    
def get_data_val(seed,rule):
    if rule=='slow_backward_rule' :
        val_batch_tmp = np.asarray(D.get_sequence_batch_valid(slow_rule= True))
        val_batch = val_batch_tmp[0]
        val_batch = np.reshape((val_batch[:,0:30]),(-1,H.num_joints , 3))

        gt_batch = val_batch_tmp[1]
        gt_batch_cp = gt_batch.reshape((-1,H.seq_length,H.num_joints * 3)).copy()
        gt_batch_slow_reverse = np.array([list(reversed(i)) for i in gt_batch_cp]).reshape((-1,H.num_joints,3))#sequence reverse
        return val_batch, gt_batch_slow_reverse
    
    if rule=='flip_backward_rule' :
        
        val_batch = np.asarray(D.get_sequence_batch_valid())
        val_batch = np.reshape((val_batch[:,0:30]),(-1,H.num_joints , 3))
        val_batch_cp = val_batch.reshape((-1,H.seq_length,H.num_joints * 3)).copy()
        val_batch_flip = rules_numpy.x_flip(val_batch).reshape((-1,H.seq_length,H.num_joints * 3))
        gt_batch_flip_reverse = np.array([list(reversed(i)) for i in val_batch_flip]).reshape((-1,H.num_joints,3))#sequence reverse
        return val_batch , gt_batch_flip_reverse
    
    
    
#########################################################################################
############################# Start training ############################################
def train(rule):

    for iteration_no in range(0,H.max_iterations):

        seed = int(time.time())
                
        train_batch , gt = get_data_train(seed,rule)

        feed_dict = {input_ph : train_batch , gt_ph : gt}
        op_train_dict = sess.run({'loss':loss,
                                  'optim':loss_optimizer,
                                  'summary_op':summary_op,
                                  'g_step': global_step,
                                  'pose_decoder_preds': pose_decoder_preds}, feed_dict=feed_dict)

        if iteration_no % 10 == 0:
            print ("global step", op_train_dict['g_step'],"train loss", op_train_dict['loss'])
            fig = plt.figure(figsize=(20, 12))
            fig_img = utils.gen_plot3(fig,train_batch[:H.seq_length].reshape([H.seq_length, H.num_joints, 3]),
                                          op_train_dict['pose_decoder_preds'][:H.seq_length].reshape([H.seq_length, H.num_joints, 3]),
                                          gt[:H.seq_length].reshape([H.seq_length, H.num_joints, 3]), az=90)

            fig.savefig('test_train.png')
            fig_img = cv2.imread('test_train.png')[:,:,::-1]
            utils.log_images('output_pose', fig_img, op_train_dict['g_step'], summary_writer_train)
            plt.close()

        if iteration_no % 50 == 0:
                val(iteration_no,rule)

        if iteration_no % 100 == 0:
                rule_network_weights.save(sess,H.store_rule_network_weights+H.exp_name+str(op_train_dict['g_step']))
                saver.save(sess,H.store_weights_path+H.exp_name+str(op_train_dict['g_step']))

        summary_writer_train.add_summary(op_train_dict['summary_op'], op_train_dict['g_step'])
        summary_writer_train.flush()

def val(iteration_no,rule):
    
    seed = int(time.time())
    val_batch , gt = get_data_val(seed,rule)
   
    feed_dict = {input_ph : val_batch , gt_ph : gt}
    op_val_dict = sess.run({'loss':loss,
                            'summary_op':summary_op,
                            'g_step': global_step,
                            'pose_decoder_preds': pose_decoder_preds}, feed_dict=feed_dict)

    fig = plt.figure(figsize=(20, 12))
    fig_img = utils.gen_plot3(fig, val_batch[:H.seq_length].reshape([H.seq_length, H.num_joints, 3]), \
                                    op_val_dict['pose_decoder_preds'][:H.seq_length].reshape([H.seq_length, H.num_joints, 3]), \
                                    gt[:H.seq_length].reshape([H.seq_length, H.num_joints, 3]), az=90)

    fig.savefig('test_valid.png')
    fig_img = cv2.imread('test_valid.png')[:, :, ::-1]
    utils.log_images('output_pose', fig_img, op_val_dict['g_step'], summary_writer_val)
    print ("global step", op_val_dict['g_step'],"val loss", op_val_dict['loss'])
    summary_writer_val.add_summary(op_val_dict['summary_op'], op_val_dict['g_step'])
    summary_writer_val.flush()
    plt.close()

if __name__ == '__main__':                                        
    rule =  'flip_backward_rule'
    train(rule)