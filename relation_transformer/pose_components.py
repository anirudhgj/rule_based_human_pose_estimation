
import numpy as np
import tensorflow as tf 

import model_componets as comps
import commons
import numpy as np, tensorflow as tf, model_componets as comps, commons

def apply_pose_encoder(x_input):

    x_real = x_input
    x_root_relative, x_local_real = comps.root_relative_to_local(x_real)
    encoder_real = comps.EncoderNet(x_local_real)
    z_real = encoder_real['z_joints']
    return z_real


def apply_pose_decoder(z_real):

    decoder_real = comps.DecoderNet(z_real)
    x_local_recon = decoder_real['full_body_x']
    x_recon = comps.local_to_root_relative(x_local_recon)
    return x_recon


def get_network_params(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
