import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
# import tf.nn as nn
from termcolor import colored
import math as m


def get_network_params(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

def create_network(images,name,reuse=tf.AUTO_REUSE):

    with tf.variable_scope(name, reuse= reuse):
        conv1 = tc.layers.conv2d(images, kernel_size=7, num_outputs=64, stride=2, scope='conv1')
        pool1 = tc.layers.max_pool2d(conv1, kernel_size=3, padding='same', scope='pool1')

        # Residual block 2a
        res2a_branch2a = tc.layers.conv2d(pool1, kernel_size=1, num_outputs=64, scope='res2a_branch2a')
        res2a_branch2b = tc.layers.conv2d(res2a_branch2a, kernel_size=3, num_outputs=64, scope='res2a_branch2b')
        res2a_branch2c = tc.layers.conv2d(res2a_branch2b, kernel_size=1, num_outputs=256, activation_fn=None, scope='res2a_branch2c')
        res2a_branch1 = tc.layers.conv2d(pool1, kernel_size=1, num_outputs=256, activation_fn=None, scope='res2a_branch1')
        res2a = tf.add(res2a_branch2c, res2a_branch1, name='res2a_add')
        res2a = tf.nn.relu(res2a, name='res2a')

        # Residual block 2b
        res2b_branch2a = tc.layers.conv2d(res2a, kernel_size=1, num_outputs=64, scope='res2b_branch2a')
        res2b_branch2b = tc.layers.conv2d(res2b_branch2a, kernel_size=3, num_outputs=64, scope='res2b_branch2b')
        res2b_branch2c = tc.layers.conv2d(res2b_branch2b, kernel_size=1, num_outputs=256, activation_fn=None, scope='res2b_branch2c')
        res2b = tf.add(res2b_branch2c, res2a, name='res2b_add')
        res2b = tf.nn.relu(res2b, name='res2b')

        # Residual block 2c
        res2c_branch2a = tc.layers.conv2d(res2b, kernel_size=1, num_outputs=64, scope='res2c_branch2a')
        res2c_branch2b = tc.layers.conv2d(res2b_branch2a, kernel_size=3, num_outputs=64, scope='res2c_branch2b')
        res2c_branch2c = tc.layers.conv2d(res2b_branch2b, kernel_size=1, num_outputs=256, activation_fn=None, scope='res2c_branch2c')
        res2c = tf.add(res2c_branch2c, res2b, name='res2c_add')
        res2c = tf.nn.relu(res2b, name='res2c')

        # Residual block 3a
        res3a_branch2a = tc.layers.conv2d(res2c, kernel_size=1, num_outputs=128, stride=2, scope='res3a_branch2a')
        res3a_branch2b = tc.layers.conv2d(res3a_branch2a, kernel_size=3, num_outputs=128, scope='res3a_branch2b')
        res3a_branch2c = tc.layers.conv2d(res3a_branch2b, kernel_size=1, num_outputs=512, activation_fn=None,scope='res3a_branch2c')
        res3a_branch1 = tc.layers.conv2d(res2c, kernel_size=1, num_outputs=512, activation_fn=None, stride=2, scope='res3a_branch1')
        res3a = tf.add(res3a_branch2c, res3a_branch1, name='res3a_add')
        res3a = tf.nn.relu(res3a, name='res3a')

        # Residual block 3b
        res3b_branch2a = tc.layers.conv2d(res3a, kernel_size=1, num_outputs=128, scope='res3b_branch2a')
        res3b_branch2b = tc.layers.conv2d(res3b_branch2a, kernel_size=3, num_outputs=128,scope='res3b_branch2b')
        res3b_branch2c = tc.layers.conv2d(res3b_branch2b, kernel_size=1, num_outputs=512, activation_fn=None,scope='res3b_branch2c')
        res3b = tf.add(res3b_branch2c, res3a, name='res3b_add')
        res3b = tf.nn.relu(res3b, name='res3b')

        # Residual block 3c
        res3c_branch2a = tc.layers.conv2d(res3b, kernel_size=1, num_outputs=128, scope='res3c_branch2a')
        res3c_branch2b = tc.layers.conv2d(res3c_branch2a, kernel_size=3, num_outputs=128,scope='res3c_branch2b')
        res3c_branch2c = tc.layers.conv2d(res3c_branch2b, kernel_size=1, num_outputs=512, activation_fn=None,scope='res3c_branch2c')
        res3c = tf.add(res3c_branch2c, res3b, name='res3c_add')
        res3c = tf.nn.relu(res3c, name='res3c')

        # Residual block 3d
        res3d_branch2a = tc.layers.conv2d(res3c, kernel_size=1, num_outputs=128, scope='res3d_branch2a')
        res3d_branch2b = tc.layers.conv2d(res3d_branch2a, kernel_size=3, num_outputs=128,scope='res3d_branch2b')
        res3d_branch2c = tc.layers.conv2d(res3d_branch2b, kernel_size=1, num_outputs=512, activation_fn=None,scope='res3d_branch2c')
        res3d = tf.add(res3d_branch2c, res3b, name='res3d_add')
        res3d = tf.nn.relu(res3d, name='res3d')

        # Residual block 4a
        res4a_branch2a = tc.layers.conv2d(res3d, kernel_size=1, num_outputs=256, stride=2, scope='res4a_branch2a')
        res4a_branch2b = tc.layers.conv2d(res4a_branch2a, kernel_size=3, num_outputs=256,scope='res4a_branch2b')
        res4a_branch2c = tc.layers.conv2d(res4a_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None,scope='res4a_branch2c')
        res4a_branch1 = tc.layers.conv2d(res3d, kernel_size=1, num_outputs=1024, activation_fn=None, stride=2, scope='res4a_branch1')
        res4a = tf.add(res4a_branch2c, res4a_branch1, name='res4a_add')
        res4a = tf.nn.relu(res4a, name='res4a')

        # Residual block 4b
        res4b_branch2a = tc.layers.conv2d(res4a, kernel_size=1, num_outputs=256, scope='res4b_branch2a')
        res4b_branch2b = tc.layers.conv2d(res4b_branch2a, kernel_size=3, num_outputs=256, scope='res4b_branch2b')
        res4b_branch2c = tc.layers.conv2d(res4b_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4b_branch2c')
        res4b = tf.add(res4b_branch2c, res4a, name='res4b_add')
        res4b = tf.nn.relu(res4b, name='res4b')

        # Residual block 4c
        res4c_branch2a = tc.layers.conv2d(res4b, kernel_size=1, num_outputs=256, scope='res4c_branch2a')
        res4c_branch2b = tc.layers.conv2d(res4c_branch2a, kernel_size=3, num_outputs=256, scope='res4c_branch2b')
        res4c_branch2c = tc.layers.conv2d(res4c_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4c_branch2c')
        res4c = tf.add(res4c_branch2c, res4b, name='res4c_add')
        res4c = tf.nn.relu(res4c, name='res4c')

        # Residual block 4d
        res4d_branch2a = tc.layers.conv2d(res4c, kernel_size=1, num_outputs=256, scope='res4d_branch2a')
        res4d_branch2b = tc.layers.conv2d(res4d_branch2a, kernel_size=3, num_outputs=256, scope='res4d_branch2b')
        res4d_branch2c = tc.layers.conv2d(res4d_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4d_branch2c')
        res4d = tf.add(res4d_branch2c, res4c, name='res4d_add')
        res4d = tf.nn.relu(res4d, name='res4d')

        # Residual block 4e
        res4e_branch2a = tc.layers.conv2d(res4d, kernel_size=1, num_outputs=256, scope='res4e_branch2a')
        res4e_branch2b = tc.layers.conv2d(res4e_branch2a, kernel_size=3, num_outputs=256, scope='res4e_branch2b')
        res4e_branch2c = tc.layers.conv2d(res4e_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4e_branch2c')
        res4e = tf.add(res4e_branch2c, res4d, name='res4e_add')
        res4e = tf.nn.relu(res4e, name='res4e')

        # Residual block 4f
        res4f_branch2a = tc.layers.conv2d(res4e, kernel_size=1, num_outputs=256, scope='res4f_branch2a')
        res4f_branch2b = tc.layers.conv2d(res4f_branch2a, kernel_size=3, num_outputs=256, scope='res4f_branch2b')
        res4f_branch2c = tc.layers.conv2d(res4f_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4f_branch2c')
        res4f = tf.add(res4f_branch2c, res4e, name='res4f_add')
        res4f = tf.nn.relu(res4f, name='res4f')
        
    return res4f


def embedding_branch(res4f, reuse=tf.AUTO_REUSE, drop1=1.0, drop2=1.0):
    with tf.variable_scope('embedding', reuse=reuse):
        conv512 = tc.layers.conv2d(res4f, kernel_size=3, num_outputs=512, stride=2, scope='conv512')
        conv256_1 = tc.layers.conv2d(conv512, kernel_size=3, num_outputs=256, stride=1, scope='conv256_1')
        conv128_1 = tc.layers.conv2d(conv256_1, kernel_size=3, num_outputs=128, stride=1, scope='conv128_1')
        conv64_1 = tc.layers.conv2d(conv128_1, kernel_size=3, num_outputs=64, stride=1, scope='conv64_1')
        flat_conv_embed = tc.layers.flatten(conv64_1)
        fc_embedding_1 = tc.layers.fully_connected(flat_conv_embed, num_outputs=1024,scope='fc')
        fc_embedding_2 = tc.layers.fully_connected(fc_embedding_1,num_outputs=1024,scope='fc_1')
        fc_embedding_3 = tc.layers.fully_connected(fc_embedding_2, num_outputs=512,scope='fc_2')
        fc_embedding_4 = tc.layers.fully_connected(fc_embedding_3, num_outputs=512,scope='fc_3')
        fc_embedding_5 = tc.layers.fully_connected(fc_embedding_4, num_outputs=256,scope='fc_4')
        fc_embedding_6 = tc.layers.fully_connected(fc_embedding_5, num_outputs=256,scope='fc_5')
        fc_embedding_7 = tc.layers.fully_connected(fc_embedding_6, num_outputs=128,scope='fc_6')
        fc_embedding_8 = tc.layers.fully_connected(fc_embedding_7, num_outputs=128,scope='fc_7')
        act = lambda x: 10 * tf.tanh(x/10)
        fc_embedding = tc.layers.fully_connected(fc_embedding_8, num_outputs=32,scope='fc_8', activation_fn=act)
        return fc_embedding   


def angles_branch(res4f, reuse=tf.AUTO_REUSE, drop1=1.0, drop2=1.0):
    with tf.variable_scope('angles', reuse=reuse):
        conv512 = tc.layers.conv2d(res4f, kernel_size=3, num_outputs=512, stride=2, scope='conv512_2')
        conv256_1 = tc.layers.conv2d(conv512, kernel_size=3, num_outputs=256, stride=1, scope='conv256_2')
        conv128_1 = tc.layers.conv2d(conv256_1, kernel_size=3, num_outputs=128, stride=1, scope='conv128_2')
        conv64_1 = tc.layers.conv2d(conv128_1, kernel_size=3, num_outputs=64, stride=1, scope='conv64_2')
        flat_conv_embed = tc.layers.flatten(conv64_1)
        fc_embedding_1 = tc.layers.fully_connected(flat_conv_embed, num_outputs=1024,scope='fc_6')
        fc_embedding_2 = tc.layers.fully_connected(fc_embedding_1, num_outputs=512,scope='fc_7')
        fc_embedding_3 = tc.layers.fully_connected(fc_embedding_2, num_outputs=256,scope='fc_8')
        fc_embedding_4 = tc.layers.fully_connected(fc_embedding_3, num_outputs=128,scope='fc_9')
        fc_embedding_5 = tc.layers.fully_connected(fc_embedding_4, num_outputs=32,scope='fc_10')
        fc_embedding_6 = tc.layers.fully_connected(fc_embedding_5,num_outputs = 8 ,scope = 'fc_11')
        fc_embedding   = tc.layers.fully_connected(fc_embedding_6,num_outputs = 6 ,scope = 'fc_12',activation_fn = tf.nn.tanh)#predicting the angles in radians
        fc_sin = tf.slice(fc_embedding,[0,0],[-1,3],name ="layer_sine")
        fc_cos = tf.slice(fc_embedding,[0,3],[-1,3],name ="layer_cosine")
        fc_embed_atan2 = tf.atan2(fc_sin,fc_cos, name = "layer_atan2")

        return fc_embed_atan2   



def regress_branch(fc_embedding,reuse = tf.AUTO_REUSE):
    with tf.variable_scope('regression',reuse = reuse):
        fc1 = tc.layers.fully_connected(fc_embedding,num_outputs=256,scope='r_fc')
        fc2 = tc.layers.fully_connected(fc1,num_outputs=126,scope='r_fc_2')
        fc3 = tc.layers.fully_connected(fc2,num_outputs=64,scope='r_fc_3')
        fc4= tc.layers.fully_connected(fc3,num_outputs=64,scope ='r_fc_4')
        fc_joints = tc.layers.fully_connected(fc4,num_outputs=45,scope= 'r_fc_j',activation_fn = None)
        return fc_joints

def get_relu_fn(use_relu=True, leak=0.2):
    def lrelu(x, leak=leak, name="lrelu"):
        return tf.maximum(x, leak * x)
    return tf.nn.relu if use_relu else lrelu

def apply_discriminator(out_5 , name ,use_relu = True, reuse = tf.AUTO_REUSE):
    with tf.variable_scope(name,reuse =reuse):
        disc_net_1 = tf.contrib.layers.fully_connected(out_5, 64 , activation_fn=get_relu_fn(use_relu))
        disc_net_2 = tf.contrib.layers.fully_connected(disc_net_1,32,activation_fn=get_relu_fn(use_relu))
        disc_net_3 = tf.contrib.layers.fully_connected(disc_net_2,32,activation_fn=get_relu_fn(use_relu))
        logits = tf.contrib.layers.fully_connected(disc_net_3, num_outputs=1, activation_fn=None)
        return logits


