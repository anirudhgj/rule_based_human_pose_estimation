from scipy.io import loadmat
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
# import tensorflow as tf
import commons.prior_sk_data as psk_data 
import matplotlib.gridspec as gridspec


from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import io

limb_parents1 = [0, 0, 1, 2, 3, 1, 5, 6, 1, 0, 9, 10, 0, 12,13]

def get_figure(gif=False):           
    fig = plt.figure(frameon=False, figsize=(20,8))
    if gif == "True":
        fig = plt.figure(frameon=False, figsize=(8,8))
    fig.clf()
    return fig


def get_ax(joints_3d, fig,subplots,az=0, ele=10):
    
    ax=fig.add_subplot(subplots[0],subplots[1],subplots[2],projection = '3d')
    lim = np.max(np.abs(joints_3d))
    ax.view_init(azim=az, elev=ele)
    
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return ax



def get_skeleton_plot(joints_3d, ax, limb_parents=limb_parents1, title="", z_tilt=True):
    draw_limbs_3d_plt(joints_3d, ax, limb_parents, z_tilt)
    plt.title(title)
    
            
def draw_limbs_3d_plt(joints_3d, ax, limb_parents=limb_parents1, z_tilt = True):
    for i in range(joints_3d.shape[0]):
        x_pair = [joints_3d[i, 0], joints_3d[limb_parents[i], 0]]
        y_pair = [joints_3d[i, 1], joints_3d[limb_parents[i], 1]]
        z_pair = [joints_3d[i, 2], joints_3d[limb_parents[i], 2]]
        if z_tilt:
            ax.plot(z_pair, x_pair, y_pair, linewidth=3, antialiased=True)
        else:
            ax.plot(x_pair, y_pair,z_pair, linewidth=3, antialiased=True)
            
                
def plot_skeleton(joints_3d, ax, limb_parents=limb_parents1, title="", z_tilt=True):
    get_skeleton_plot(joints_3d, ax, limb_parents1, title, z_tilt=z_tilt)
    
            
def plot_skeleton_and_scatter(ske,ax,title= "", mono=False):
    plot_skeleton(ske,ax,title,z_tilt=mono)


def get_root_relative_skeleton(ske, mono=False):
    if mono:
        ske[:, 0:2] = - ske[:, 0:2]     
    ske = ske - ske[0]
    return ske   
        

def gen_plot_sequence(fig,batch_output_joints,batch_input_joints,batch_output_gamma,batch_input_gamma):#dims = batch _size x sequence_length x (2 x (15 x 3))
    output_seq = batch_output_joints[50]#selecting 49th batch
    input_seq=batch_input_joints[50]
    output_gamma = batch_output_gamma[50]
    input_gamma= batch_input_gamma[50]
    j=1
    n=2
    for i in range(0,5):
        #inputs
        k=input_seq[i].reshape((30,3))
        male=k[0:15] 
        female=k[15:30]
        male=np.asarray(tf_util.view_norm_to_root_relative_skeleton(np.array([0,0,input_gamma[i][0]]),male))
        female=np.asarray(tf_util.view_norm_to_root_relative_skeleton(np.array([0,0,input_gamma[i][1]]),female))
        ax1 = get_ax1(male,fig,(2,10,j),az=90)
        ax2 = get_ax1(female,fig,(2,10,j+1),az=90)
        plot_skeleton_and_scatter(male,ax1)
        plot_skeleton_and_scatter(female,ax2)
        #outputs
        o=output_seq[i].reshape((32,3))
        male_out=o[0:15] 
        female_out=o[15:30] 
        male_out=np.asarray(tf_util.view_norm_to_root_relative_skeleton(np.array([0,0,output_gamma[i][0]]),male_out))
        female_out=np.asarray(tf_util.view_norm_to_root_relative_skeleton(np.array([0,0,output_gamma[i][1]]),female_out)) 
        ax3 = get_ax1(male_out,fig,(2,10,j+10),az=90)
        ax4 = get_ax1(female_out,fig,(2,10,n+10),az=90)
        plot_skeleton_and_scatter(male_out,ax3)
        plot_skeleton_and_scatter(female_out,ax4)
        j = j + 2
        n = n + 2
    return fig

def fig2rgb_array(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    shape = (nrows, ncols, 3) 
    return np.fromstring(buf, dtype=np.uint8).reshape(shape)

        
def gen_video(batch_output_joints,batch_input_joints,batch_output_gamma,batch_input_gamma,iteration_no):
    fig = get_figure(gif="True")
    output_seq = batch_output_joints[50]#selecting 49th batch
    input_seq=batch_input_joints[50]
    output_gamma = batch_output_gamma[50]
    input_gamma= batch_input_gamma[50]
    images=[]

    for i in range(50):
        #inputs
        k=input_seq[i].reshape((30,3))
        male=k[0:15] 
        female=k[15:30]
        male=np.asarray(tf_util.view_norm_to_root_relative_skeleton(np.array([0,0,input_gamma[i][0]]),male))
        female=np.asarray(tf_util.view_norm_to_root_relative_skeleton(np.array([0,0,input_gamma[i][1]]),female))
        ax1 = get_ax1(male,fig,(2,2,1),az=90)
        ax2 = get_ax1(female,fig,(2,2,2),az=90)
        plot_skeleton_and_scatter(male,ax1)
        plot_skeleton_and_scatter(female,ax2)

        #outputs
        o=output_seq[i].reshape((32,3))
        male_out=o[0:15] 
        female_out=o[15:30] 
        male_out=np.asarray(tf_util.view_norm_to_root_relative_skeleton(np.array([0,0,output_gamma[i][0]]),male_out))
        female_out=np.asarray(tf_util.view_norm_to_root_relative_skeleton(np.array([0,0,output_gamma[i][1]]),female_out)) 
        ax3 = get_ax1(male_out,fig,(2,2,3),az=90)
        ax4 = get_ax1(female_out,fig,(2,2,4),az=90)
        plot_skeleton_and_scatter(male_out,ax3)
        plot_skeleton_and_scatter(female_out,ax4)

        images.append(fig2rgb_array(fig))

    imageio.mimsave('./videos/'+str(iteration_no)+".gif", images, duration = 0.2)
    print "done"
    

def bigger_mat(ske):

    add_joint_raw={'right_ankle':0,
                   'left_ankle':1,
                   'right_knee':2,
                   'left_knee':3,
                   'right_hip':4,
                   'left_hip':5,
                   'right_wrist':6,
                   'left_wrist':7,
                   'right_elbow':8,
                   'left_elbow':9,
                   'right_shoulder':10,
                   'left_shoulder':11,
                   'head_top':12,
                   'neck':14,
                   'pelvis':13,
                   'right_foot':15,
                   'left_foot':16}

    modified_joint_names = {'pelvis':0,
                            'neck':1,
                            'right_shoulder':2,
                            'right_elbow':3,
                            'right_wrist':4,  
                            'left_shoulder':5,
                            'left_elbow':6,
                            'left_wrist':7,
                            'head_top':8,
                            'right_hip':9,
                            'right_knee':10,
                            'right_ankle':11,
                            'right_foot':12,
                            'left_hip':13,
                            'left_knee':14,
                            'left_ankle':15,
                            'left_foot':16}
    

    resized_frame = ske
    resized_frame[:,[2, 1]] = resized_frame[:,[1,2]]
    resized_frame =list(resized_frame)
    resized_frame.append(resized_frame[add_joint_raw['right_ankle']])
    resized_frame.append(resized_frame[add_joint_raw['left_ankle']])
    
    z=np.zeros((17,3))
    for key, value in modified_joint_names.items():
        z[value]=resized_frame[add_joint_raw[key]]

    a=z[modified_joint_names['pelvis']]
    z=z -a
    z=fit_skeleton_frame(z).tolist()
    del z[modified_joint_names['left_foot']]
    del z[modified_joint_names['right_foot']]
    z=np.asarray(z)
    return z

# k=loadmat("haha.mat")
# k =  k['batch'][0][0]
# print k.shape
# temp = k.reshape((30,3))
# male = temp[0:15]

# male = bigger_mat(male)

# fig = get_figure()
# ax1 = get_ax1(male,fig,(1,1,1),az=90)
# plot_skeleton_and_scatter(male,ax1)

