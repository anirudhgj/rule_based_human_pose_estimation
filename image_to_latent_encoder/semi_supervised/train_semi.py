#################################################################################
############################# Load Libraries ####################################
import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) 
import numpy as np
import scipy.io as sio
import random
import cv2
from termcolor import colored
import utils 
np.seterr(divide='ignore', invalid='ignore')

import image_components
import pose_components
import matplotlib.pyplot as plt
import dataloader
# from test_data_loader import get_h36_test, data_loader_h36
from hyperparams import Hyperparameters
import multiprocessing as mp 
from sklearn.utils import shuffle


H = Hyperparameters ()
if not os.path.exists('./weights'):
    os.makedirs('./weights')
    os.makedirs('./weights/resnet/')
    os.makedirs('./weights/embedding_branch/')



################################################################################
############################# Start Session ####################################

os.environ['CUDA_VISIBLE_DEVICES'] = H.cuda_device_id
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)


###############################################################################
############################# Data Loader #####################################



surreal_data_path = [
#                      '/data/vcl/anirudh_rule_based/codes_2020/data/surreal_data_vneck/surreal_train_1.mat',
#                      '/data/vcl/anirudh_rule_based/codes_2020/data/surreal_data_vneck/surreal_train_2.mat',
                     '/data/vcl/anirudh_rule_based/codes_2020/data/surreal_data_vneck/surreal_train_3.mat',
#                      '/data/vcl/anirudh_rule_based/codes_2020/data/surreal_data_vneck/surreal_train_4.mat',
#                      '/data/vcl/anirudh_rule_based/codes_2020/data/surreal_data_vneck/surreal_train_5.mat',
                     '/data/vcl/anirudh_rule_based/codes_2020/data/surreal_data_vneck/human36_s1_train.mat'
                        ]

'''
## 128 machine
surreal_data_path = [
#                      '/data/vcl/anirudh_rule_based/codes_2020/data/surreal_data_vneck/surreal_train_1.mat',
#                      '/data/vcl/anirudh_rule_based/codes_2020/data/surreal_data_vneck/surreal_train_2.mat',
#                      '/data/vcl/anirudh_rule_based/codes_2020/data/surreal_data_vneck/surreal_train_3.mat',
                     '/sdb/jogendra/anirudh/mounts/503/data/vcl/anirudh_rule_based/codes_2020/data/surreal_data_vneck/surreal_train_4.mat',
                     '/sdb/jogendra/anirudh/mounts/503/data/vcl/anirudh_rule_based/codes_2020/data/surreal_data_vneck/surreal_train_5.mat',
                     '/sdb/jogendra/anirudh/mounts/503/data/vcl/anirudh_rule_based/codes_2020/data/surreal_data_vneck/human36_s1_train.mat'
                        ]
'''

# images_path, poses_3d = dataloader.get_list_mp(surreal_data_path)
# num_batches = len(images_path) // H.batch_size 


# h36_test_data = get_h36_test()
# h36_test_dataset = tf.data.Dataset.from_tensor_slices((h36_test_data))
# dataset_human_test = h36_test_dataset.map(lambda z: tf.py_func(data_loader_h36, [z], [tf.float32, \
#         tf.float32]), 32)
# dataset_human_test = dataset_human_test.batch(H.batch_size)
# dataset_human_test = dataset_human_test.prefetch(2)
# test_iterator = dataset_human_test.make_initializable_iterator()
# test_img, test_pose = test_iterator.get_next()

train_paths = surreal_data_path
train_iterator , train_next_element = dataloader.data_loader(train_paths, H.batch_size , H.num_threads)

if not os.path.exists('./logs'):
    os.makedirs('./logs')
    


#################################################################################
############################# graph ############################################

image_ph = tf.placeholder(tf.float32 , shape = [None,H.image_size,H.image_size ,3] , name = 'image_ph')
pose_ph = tf.placeholder(tf.float32 , shape = [None,17,3] , name = 'pose_gt_ph')

def get_image_to_latent(input_image): # input_image shape:[None,224,224,3]
    resnet_out = image_components.create_network(input_image,'')
    resnet_params = image_components.get_network_params('')
    fc_embedding = image_components.embedding_branch(resnet_out)
    fc_embedding_params = image_components.get_network_params('embedding')
    outputs = {
        'resnet_params' :resnet_params,
        'im_embed' :fc_embedding,
        'fc_embedding_params':fc_embedding_params
    }
    return outputs
    
cnn_outputs = get_image_to_latent(image_ph)
embed=cnn_outputs['im_embed']
pose_pred = pose_components.apply_pose_decoder(embed)

loss = tf.reduce_mean(tf.abs(pose_pred - pose_ph))
loss_summary = tf.summary.scalar("pose_pred_loss",loss)
merge_summary = tf.summary.merge_all()

resnet_params = cnn_outputs['resnet_params']
fc_embedding_params = cnn_outputs['fc_embedding_params']
pose_decoded_params = pose_components.get_network_params('Decoder_net')

opt = tf.train.AdamOptimizer(H.learning_rate).minimize(loss = loss,
                                                       var_list = resnet_params + fc_embedding_params)


sess.run(tf.global_variables_initializer())


resnet_branch_weights = tf.train.Saver(resnet_params,max_to_keep=5)
embedding_branch_weights = tf.train.Saver(fc_embedding_params,max_to_keep=5)
decoder_branch_weights = tf.train.Saver(pose_decoded_params,max_to_keep=5)
train_writer = tf.summary.FileWriter(H.logdir_path_train,sess.graph)
# val_writer = tf.summary.FileWriter(H.logdir_path_val)

#########################################################################################
############################# Loading Weights ###########################################


tf.train.Saver(resnet_params).restore(sess,tf.train.latest_checkpoint('../weights/resnet/'))
print (colored("restored resnet branch weights","yellow"))

tf.train.Saver(fc_embedding_params).restore(sess,tf.train.latest_checkpoint('../weights/embedding_branch/'))
print (colored("Restored Embedding Branch weights","green"))

tf.train.Saver(pose_decoded_params).restore(sess,'../../pretrained_weights/pose_encoder_decoder/decoder_iter-1475001')
print (colored("Restored Decoder weights","blue"))


#########################################################################################
############################# Start training ############################################

# def val(sess,test_iterator):
    
#     loss_list = []
#     sess.run(test_iterator.initializer)
#     while True:
#         try:
#             images, pose_3d =  sess.run([test_img, test_pose])
#             val_loss,pred_3d = sess.run([loss,pose_pred], feed_dict = {image_ph : images , pose_ph : pose_3d})
#             loss_list.append(val_loss) 
#         except tf.errors.OutOfRangeError:
#             print("Val loss : ",sum(loss_list)/len(loss_list))
#             break
#     return sum(loss_list)/len(loss_list)


# for epoch in range(H.num_epochs):
#     images_path, poses_3d = shuffle(images_path, poses_3d, random_state=0)
#     combined_data = [[i,j] for i,j in zip(images_path,poses_3d)]
#     combined_data_batches = np.array_split(combined_data, num_batches)
    
#     for iteration in range(num_batches):
        
#         with mp.Pool(processes = 16) as p:
#             train_batch = p.map(dataloader.process_mp,combined_data_batches[iteration]) 
            
#         images = [i[0] for i in train_batch]
#         pose_3d = [i[1] for i in train_batch]
#         op_dict = sess.run({'loss' : loss , 'opt' : opt , 'loss_summary' : merge_summary ,
#                                 'pose_preds':pose_pred } , feed_dict = {image_ph : images,
#                                                                         pose_ph :  pose_3d})

#         if iteration % 10 == 0:
#             print ("loss is : " ,op_dict['loss'] , "Iteration No : ",iteration)
#             idx = np.random.choice(np.arange(H.batch_size))
#             fig = plt.figure(figsize=(16, 8))
#             fig_img = utils.gen_image_plot(fig,op_dict['pose_preds'][idx],pose_3d[idx],images[idx],az = 0)
#             fig.savefig('image_train.png')
#             fig_img = cv2.imread('image_train.png')[:,:,::-1]
#             utils.log_images('output_train', fig_img, iteration, train_writer)
#             plt.close()

#             train_writer.add_summary(op_dict['loss_summary'], iteration)
#             train_writer.flush()

#         if iteration % 300 == 0:
#             resnet_branch_weights.save(sess,H.store_resnet_weights_path+H.exp_name+str(iteration))
#             embedding_branch_weights.save(sess,H.store_embedding_branch_weights_path+H.exp_name+str(iteration))        
            
            
iteration = 0 
for epoch in range(H.num_epochs):
    sess.run(train_iterator.initializer)
    
    while True:
        iteration = iteration + 1
        try:
            if iteration%2 == 0 :
                images,pose_3d = sess.run(train_next_element)

                op_dict = sess.run({'loss' : loss , 'opt' : opt , 'loss_summary' : merge_summary ,
                                    'pose_preds':pose_pred } , feed_dict = {image_ph : images , pose_ph : pose_3d})
            
            if iteration % 10 == 0:
                print ("loss is : " ,op_dict['loss'] , "Iteration No : ",iteration)
                idx = np.random.choice(np.arange(H.batch_size))
                fig = plt.figure(figsize=(16, 8))
                fig_img = utils.gen_image_plot(fig,op_dict['pose_preds'][idx],pose_3d[idx],images[idx],az = 0)
                fig.savefig('image_train.png')
                fig_img = cv2.imread('image_train.png')[:,:,::-1]
                utils.log_images('output_train', fig_img, iteration, train_writer)
                plt.close()
                
                train_writer.add_summary(op_dict['loss_summary'], iteration)
                train_writer.flush()
                    
            if iteration % 300 == 0:
                resnet_branch_weights.save(sess,H.store_resnet_weights_path+H.exp_name+str(iteration))
                embedding_branch_weights.save(sess,H.store_embedding_branch_weights_path+H.exp_name+str(iteration))

        except tf.errors.OutOfRangeError:
            break
            
            
