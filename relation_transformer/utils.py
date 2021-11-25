import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from commons import transform_util as tr_utils, vis_util
from commons import skeleton_utils as sk_utils

import os
import glob
import tensorflow as tf
import random
import numpy as np
from transforms3d.euler import EulerFuncs
from PIL import Image
import numpy as np
from io import BytesIO
from hyperparams import Hyperparameters

H = Hyperparameters ()
euler_fx = EulerFuncs('rxyz')

eps = 0.00000005

def get_train_val_mat_files():
    all_videos = sorted(glob.glob('./data/mads/*.mat'))

    val_actions = ['Taichi_S6', 'HipHop_HipHop6', 'Jazz_Jazz6', 'Sports_Tennis_Left']

    val_videos = []

    for xx in all_videos:
        if '_'.join(xx.split('/')[-1].split('_')[:2]) in val_actions:
            val_videos.append(xx)
        elif '_'.join(xx.split('/')[-1].split('_')[:3]) in val_actions:
            val_videos.append(xx)

    train_videos = [x for x in all_videos if x not in val_videos]

    return np.array(train_videos), np.array(val_videos)


def get_a_cell(lstm_size):
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    return lstm


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import math as m

from matplotlib import gridspec

limb_parents = [0, 0, 1, 2, 3, 1, 5, 6, 1, 0, 9, 10, 11, 0, 13, 14, 15]
# limb_parents = [0, 0, 1, 2, 3, 1, 5, 6, 1, 0, 9, 10, 0, 12, 13]



def get_figure():
    fig = plt.figure(frameon=False, figsize=(8, 8))
    fig.clf()
    return fig


def fig2rgb_array(fig, expand=True):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
    return np.fromstring(buf, dtype=np.uint8).reshape(shape)


def figure_to_summary(fig, iteration_no, train_writer, vis_summary, vis_placeholder):
    image = fig2rgb_array(fig)
    train_writer.add_summary(vis_summary.eval(feed_dict={vis_placeholder: image}), global_step=iteration_no)
    plt.close(fig)


def draw_limbs_3d_plt(joints_3d, ax, limb_parents=limb_parents, z_tilt=True):
    for i in range(joints_3d.shape[0]):
        x_pair = [joints_3d[i, 0], joints_3d[limb_parents[i], 0]]
        y_pair = [joints_3d[i, 1], joints_3d[limb_parents[i], 1]]
        z_pair = [joints_3d[i, 2], joints_3d[limb_parents[i], 2]]
        if z_tilt:
            ax.plot(z_pair, x_pair, y_pair, linewidth=3, antialiased=True)
        else:
            ax.plot(x_pair, y_pair, z_pair, linewidth=3, antialiased=True)


def get_ax(joints_3d, fig, az=0, ele=10, subplot='111'):
    x, y, z = subplot
    ax = fig.add_subplot(subplot, projection='3d')

    lim = np.max(np.abs(joints_3d))
    ax.view_init(azim=az, elev=ele)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    return ax


def get_skeleton_plot(joints_3d, ax, limb_parents=limb_parents, title="", z_tilt=True):

    draw_limbs_3d_plt(joints_3d, ax, limb_parents, z_tilt)
    plt.title(title)


def plot_skeleton(joints_3d, ax, limb_parents=limb_parents, title="", z_tilt=True):
    get_skeleton_plot(joints_3d, ax, limb_parents, title, z_tilt=z_tilt)


def fig2data(fig):
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def plot_skeleton_and_scatter(ske, ax, mono=False):
    plot_skeleton(ske, ax, z_tilt=mono)


def gen_plot(fig, x_recon,inputs, seq_length = H.seq_length, az=0, el=10):

    gs = gridspec.GridSpec(2, 10, wspace=0.2, hspace=0.2)

    sample = np.linspace(0, seq_length-1, 10).astype(int)
    x_recon_sample = x_recon[sample]
    inputs_sample = inputs[sample]

    for xx in range(10):
        ax = fig.add_subplot(gs[xx], projection='3d')
        lim = np.max(np.abs(x_recon_sample[xx]))
        ax.view_init(azim=az, elev=el)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plot_skeleton_and_scatter(x_recon_sample[xx], ax)

    for xx in range(10, 20):
        ax = fig.add_subplot(gs[xx], projection='3d')
        lim = np.max(np.abs(inputs_sample[xx-10]))
        ax.view_init(azim=az, elev=el)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plot_skeleton_and_scatter(inputs_sample[xx-10], ax)

    return fig2data(fig)

def gen_plot3(fig, inputs,preds,gt, seq_length = H.seq_length, az=0, el=10):

    gs = gridspec.GridSpec(3, 15, wspace=0.2, hspace=0.2)

    # sample = np.linspace(0, seq_length-1, 10).astype(int)
    inputs_sample = inputs[::2]
    preds_sample = preds[::2]
    gt_sample = gt[::2]

    for xx in range(15):
        ax = fig.add_subplot(gs[xx], projection='3d')
        lim = np.max(np.abs(inputs_sample[xx]))
        ax.view_init(azim=az, elev=el)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plot_skeleton_and_scatter(inputs_sample[xx], ax)

    for xx in range(15, 30):
        ax = fig.add_subplot(gs[xx], projection='3d')
        lim = np.max(np.abs(preds_sample[xx-15]))
        ax.view_init(azim=az, elev=el)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plot_skeleton_and_scatter(preds_sample[xx-15], ax)

    for xx in range(30, 45):
        ax = fig.add_subplot(gs[xx], projection='3d')
        lim = np.max(np.abs(gt_sample[xx-30]))
        ax.view_init(azim=az, elev=el)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plot_skeleton_and_scatter(gt_sample[xx-30], ax)
    
    return fig2data(fig)

                                         
def x_flip(sk_17):
    left_indices = [2, 3, 4, 9, 10, 11, 12]
    right_indices = [5, 6, 7, 13, 14, 15, 16]
    sk_17_flip = sk_17.copy()
    l_idx, r_idx = left_indices, right_indices
    sk_17_flip[:, l_idx] = sk_17[:, r_idx]
    sk_17_flip[:, r_idx] = sk_17[:, l_idx]
    sk_17_flip[:, :, 0] *= -1
    sk_17 = sk_17_flip
    return sk_17




def augment_pose_seq(pose_seq,z_limit=(0,360),y_limit=(-20,20),seed=42,set_seed=False):
    if set_seed:
        np.random.seed(seed)
        
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

    if not random.getrandbits(1):
        p=p[:,::-1,:]

    return p 

def pose_rotate(points, theta, batch_size):

    theta = theta * m.pi / 180.0 
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
    theta = theta * m.pi / 180
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


def log_images(tag, image, step, writer):
    """Logs a list of images."""

    height, width, channel = image.shape
    image = Image.fromarray(image)
    output = BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()

    # Create an Image object
    img_sum = tf.Summary.Image(height=height,
                               width=width,
                               colorspace=channel,
                               encoded_image_string=image_string)
    # Create a Summary value
    im_summary = tf.Summary.Value(tag='%s' % (tag), image=img_sum)

    # Create and write Summary
    summary = tf.Summary(value=[im_summary])
    writer.add_summary(summary, step)


def get_exp_name():
    return os.path.basename(os.path.abspath('.'))

def get_a_cell(lstm_size):
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    return lstm

def unit_norm(v, axis=0):
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / (norm + 1e-9)


def copy_weights(iter_no, label='iter'):
    """
    Copied the Weights from weights/ to pretrained_weights/ given iteration number and label: 'iter' or 'best'
    """
    files = os.listdir('weights/')
    match_substr = '%s-%d' % (label, iter_no)
    files = [f for f in files if match_substr in f]
    for f in files:
        cmd = 'cp weights/%s pretrained_weights/' % f
        print (cmd)
        os.system(cmd)

def flatten(inputs):
    # type: (tf.Tensor) -> tf.Tensor
    """
    Reshapes a Tensor into a 2D tensor, retaining the batch (first) dimension.
    :param inputs: input Tensor
    :return: flattened Tensor
    """
    batch_size = tf.shape(inputs)[0]
    n_features = np.product(map(lambda x: x.value, inputs.shape[1:]))
    return tf.reshape(inputs, (batch_size, n_features))

def get_most_recent_iteration(label='best', dir='weights'):
    """
    Gets the most recent iteration number from weights/ dir of given label: ('best' or 'iter')
    """
    files = os.listdir(dir)
    files = [f for f in files if label in f]
    numbers = {int(f[f.index('-') + 1:f.index('.')]) for f in files}
    return max(numbers)


def copy_latest(type='best'):
    latest_iter = get_most_recent_iteration(type)
    copy_weights(latest_iter, type)
    return latest_iter


def local_2_global_batch_skeleton(x_batch):
    x_batch = unit_norm(x_batch, axis=2)
    x_batch = sk_utils.scale_local_skeleton(x_batch)
    x_batch_global = np.array([tr_utils.prior_local2global(x_batch[i]) for i in range(x_batch.shape[0])])
    return sk_utils.fit_skeleton_frames(x_batch_global)


def get_global_skeleton(x):
    x = unit_norm(x, axis=1)
    x = sk_utils.scale_local_skeleton(x)
    x = tr_utils.prior_local2global(x)
    x = sk_utils.fit_skeleton_frame(x)
    return x


def plot_local_skeleton(x, title=""):
    x = get_global_skeleton(x)
    return vis_util.plot_skeleton(x, title=title)


def plot_global_skeleton(x, title=""):
    return vis_util.plot_skeleton(x, title=title)


def compare_global_skeletons(x_original, x_reconstructed):
    fig = plt.figure(frameon=False, figsize=(14, 7))
    titles = ['Original Skeleton', 'Reconstructed Skeleton']
    skeletons = [x_original, x_reconstructed]
    for i in range(2):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        ax.set_axis_off()
        ax.clear()
        ax.view_init(azim=-90, elev=10)
        ax.set_xlim(-800, 800)
        ax.set_ylim(-800, 800)
        ax.set_zlim(-800, 800)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(titles[i])
        vis_util.draw_limbs_3d_plt(skeletons[i] * 100, ax)
    return fig


def compare_multiple_global_skeletons(skeletons):
    num_skeletons = len(skeletons)
    height = 6
    width = num_skeletons * 6
    fig = plt.figure(frameon=False, figsize=(width, height))
    titles = ['Skeleton: %d' % i for i in range(num_skeletons)]
    for i in range(num_skeletons):
        ax = fig.add_subplot(1, num_skeletons, i + 1, projection='3d')
        ax.set_axis_off()
        ax.clear()
        ax.view_init(azim=-90, elev=10)
        ax.set_xlim(-800, 800)
        ax.set_ylim(-800, 800)
        ax.set_zlim(-800, 800)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(titles[i])
        vis_util.draw_limbs_3d_plt(skeletons[i] * 100, ax)
    return fig


def compare_local_skeletons(x_original, x_recon):
    x_original = get_global_skeleton(x_original)
    x_recon = get_global_skeleton(x_recon)
    return compare_global_skeletons(x_original, x_recon)


def compare_multiple_local_skeletons(skeletons):
    global_skeletons = [get_global_skeleton(sk) for sk in skeletons]
    return compare_multiple_global_skeletons(global_skeletons)


def plot_2_skeletons(x_original, x_reconstructed):
    plot_local_skeleton(x_original, 'Original skeleton')
    plot_local_skeleton(x_reconstructed, 'Reconstructed Skeleton')

# copy_weights(859501)
