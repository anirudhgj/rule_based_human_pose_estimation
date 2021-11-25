from cv2 import cv2

import numpy as np

from scipy.io import loadmat

from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as Canvas
from mpl_toolkits.mplot3d import Axes3D

from commons import prior_sk_data as psk_data

# limb_parents = [1, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 14, 14, 1, 4, 7, 10, 13]
limb_parents = psk_data.limb_parents

plt_angles = [
    [-45, 10],
    [-135, 10],
    [-90, 10],
]

black = (0, 0, 0)
white = (1, 1, 1)


def draw_limbs_3d_plt(joints_3d, ax, limb_parents=limb_parents):
    for i in range(joints_3d.shape[0]):
        x_pair = [joints_3d[i, 0], joints_3d[limb_parents[i], 0]]
        y_pair = [joints_3d[i, 1], joints_3d[limb_parents[i], 1]]
        z_pair = [joints_3d[i, 2], joints_3d[limb_parents[i], 2]]
        ax.plot(x_pair, y_pair, zs=z_pair, linewidth=3, antialiased=True)


def get_sk_frame_figure(pred_3d, hmap_img, hmap_title):
    fig = plt.figure(frameon=False, figsize=(10, 10))
    for i, ang in enumerate(plt_angles):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        ax.set_axis_off()
        ax.clear()
        ax.view_init(azim=ang[0], elev=ang[1])
        ax.set_xlim(-800, 800)
        ax.set_ylim(-800, 800)
        ax.set_zlim(-800, 800)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        draw_limbs_3d_plt(pred_3d * 100, ax)
    ax = fig.add_subplot(2, 2, 4)
    ax.imshow(hmap_img)
    ax.set_title(hmap_title)
    return fig


def save_sk_frame_plt(video_project, pred_3d, img_id):
    hmap_img = cv2.imread(video_project.get_hmap_path(img_id))[:, :, [2, 1, 0]]
    hmap_title = 'Frame: %04d' % img_id
    fig = get_sk_frame_figure(pred_3d, hmap_img, hmap_title)
    fig.savefig(video_project.get_sk_frame_path(img_id))
    plt.close(fig)
    print('%s: Saved sk_frame for image %d' % (video_project.project_name, img_id))


def save_sk_frames_plt(video_project, start=0, end=None, translation=False):
    pred_3d = loadmat(video_project.get_pred_path())['pred_3d_fit']

    if translation:
        pelvis_position = video_project.get_preds()['pred_delta']
        pelvis_position[1] = pelvis_position[0]
        for i in xrange(1, pelvis_position.shape[0]):
            pelvis_position[i] += pelvis_position[i - 1]

    if end is None:
        end = len(pred_3d) - 1
    for i in xrange(start, end + 1):
        pred = pred_3d[i]
        if translation:
            pred[:, [0, 2]] += (pelvis_position[i] * .10)
        save_sk_frame_plt(video_project, pred, i)


def plot_skeleton(joints_3d, limb_parents=limb_parents, title=""):
    fig = plt.figure(frameon=False, figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(azim=90, elev=10)
    ax.set_xlim(-800, 800)
    ax.set_ylim(-800, 800)
    ax.set_zlim(-800, 800)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    draw_limbs_3d_plt(joints_3d * 100, ax, limb_parents)
    plt.title(title)
    img =  fig2data(fig)
    plt.close(fig)
    return img


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