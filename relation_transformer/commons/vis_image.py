import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf


from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import io
limb_parents = [0, 0, 1, 2, 3, 1, 5, 6, 1, 0, 9, 10, 11, 0, 13, 14, 15]

def get_figure():           
    fig = plt.figure(frameon=False, figsize=(8, 8))
    fig.clf()
    return fig

def fig2rgb_array(fig, expand=True):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    # np.frombuffer(buf, dtype=np.uint8)
    ncols, nrows = fig.canvas.get_width_height()
    shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
    return np.fromstring(buf, dtype=np.uint8).reshape(shape)


def figure_to_summary(fig, iteration_no,  train_writer, vis_summary, vis_placeholder, mode=None):
    image = fig2rgb_array(fig,expand=True)

    # print(" gg", vis_placeholder)

    # if mode=='test':
    #     test_writer.add_summary(vis_summary.eval(feed_dict={vis_placeholder: image}), global_step= iteration_no )
    # else:    
    train_writer.add_summary(vis_summary.eval(feed_dict={vis_placeholder: image}), global_step= iteration_no )
    plt.close(fig)


def draw_limbs_3d_plt(joints_3d, ax, limb_parents=limb_parents, z_tilt = True):
    for i in range(joints_3d.shape[0]):
        x_pair = [joints_3d[i, 0], joints_3d[limb_parents[i], 0]]
        y_pair = [joints_3d[i, 1], joints_3d[limb_parents[i], 1]]
        z_pair = [joints_3d[i, 2], joints_3d[limb_parents[i], 2]]
        if z_tilt:
            ax.plot(z_pair, x_pair, y_pair, linewidth=3, antialiased=True)
        else:
            ax.plot(x_pair, y_pair,z_pair, linewidth=3, antialiased=True)
        

def get_ax(joints_3d, fig, arr = (1,1,1),az=0, ele=10):
    
    ax = fig.add_subplot(arr[0],arr[1],arr[2], projection='3d')

    lim = np.max(np.abs(joints_3d))
#   print("lim", lim)
    ax.view_init(azim=az, elev=ele)
    
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return ax

        
def get_skeleton_plot(joints_3d, ax, limb_parents=limb_parents, title="", z_tilt=True):
#     fig = plt.figure(frameon=False, figsize=(7, 7))
    draw_limbs_3d_plt(joints_3d, ax, limb_parents, z_tilt)
    plt.title(title)
    

def plot_skeleton(joints_3d, ax, limb_parents=limb_parents, title="", z_tilt=True):
    get_skeleton_plot(joints_3d, ax, limb_parents, title, z_tilt=z_tilt)
    

# def get_figure():
#   fig = plt.figure(num=0, figsize=(6, 4), dpi=300)
#   fig.clf()
#   return fig


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def get_root_relative_skeleton(ske, mono=False):
    if mono:
        ske[:, 0:2] = - ske[:, 0:2]     
    ske = ske - ske[0]
    return ske

def do_scatter(pose, ax, z_tilt=False):
  
    lim = np.max(np.abs(pose))

    if z_tilt:
        ax.scatter(pose[:,2], pose[:,0],pose[:,1])
    else:    
        ax.scatter(pose[:,0], pose[:,1],pose[:,2])
    for i in range(pose.shape[0]):
        x,y,z = pose[i]
        if not z_tilt:
            ax.text(x,y,z, str(i))
        else:
            ax.text(z,x,y, str(i))


def plot_skeleton_and_scatter(ske,ax, mono=False):
    plot_skeleton(ske,ax,z_tilt=mono)
    # do_scatter(ske,ax,z_tilt=mono)


def gen_plot(x_recon, inputs):
    """Create a pyplot plot and save to buffer."""
    fig = plt.figure(frameon=False, figsize=(8, 8))
    ax = get_ax(x_recon, fig, subplot='121', az=90)
    plot_skeleton_and_scatter(x_recon, ax)
    ax = get_ax(inputs, fig, subplot='122', az=90)
    plot_skeleton_and_scatter(inputs, ax)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return buf



def gen_plot_2(fig, x_recon, inputs):
    """Create a pyplot plot and save to buffer."""
    ax = get_ax(x_recon, fig, subplot='121', az=90)
    plot_skeleton_and_scatter(x_recon, ax)
    ax = get_ax(inputs, fig, subplot='122', az=90)
    plot_skeleton_and_scatter(inputs, ax)
    return fig

def gen_plot_4(fig, x_recon_l,x_recon_r,inputs_l,inputs_r ):
    """Create a pyplot plot and save to buffer."""
    ax1 = get_ax(x_recon_l, fig, subplot='141', az=0)
    plot_skeleton_and_scatter(x_recon_l, ax1)
    ax2 = get_ax(inputs_l, fig, subplot='142', az=0)
    plot_skeleton_and_scatter(inputs_l, ax2)
    ax3 = get_ax(x_recon_r, fig, subplot='143', az=0)
    plot_skeleton_and_scatter(x_recon_r, ax3)
    ax4 = get_ax(inputs_r, fig, subplot='144', az=0)
    plot_skeleton_and_scatter(inputs_r, ax4)
    return fig

