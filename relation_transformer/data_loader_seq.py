import numpy as np
from scipy.io import loadmat,savemat
from hyperparams import Hyperparameters
import os,glob
import random
import time


H = Hyperparameters()


class Data_loader(object):
    def __init__(self,data_path,sequence_length,batch_size,num_joints):
        self.data_path = data_path
        self.num_joints = num_joints 
        self.batch_size = batch_size
        self.sequence_length =sequence_length
        self.datasets = os.listdir(self.data_path)
        self.val_list = ['Taichi_S6', 'HipHop_HipHop6', 'Jazz_Jazz6', 'Sports_Tennis_Left']
        self.all_videos = [val for sublist in [[os.path.join(i[0], j) for j in i[2] if j.endswith('.mat') and 'train.mat' not in j] for i in os.walk(self.data_path+'/train/')] for val in sublist]


    def get_sequence_batch_train(self, slow_rule=False):
        my_video = np.random.choice(self.all_videos,self.batch_size)
        sequences = []
        for video in my_video:
            try :
                dat = loadmat(video)['pose_3d']
            except :
                print (loadmat(video).keys())
                print ("error detected")
                print (my_video)
                print (video)
                video = np.random.choice(self.all_videos)
                dat = loadmat(video)['pose_3d']

            sequences.append(random.randint(0,dat.shape[0]-self.sequence_length))
        batch = []
        for i in range(len(sequences)):
            k=loadmat(my_video[i])['pose_3d'][sequences[i]:sequences[i] + self.sequence_length].reshape((self.sequence_length,self.num_joints * 3))
            batch.append(k)
        if slow_rule :
            tmp_batch = np.array(batch).copy()
            input_batch, slow_batch = [] ,[]
            for i,seq in enumerate(tmp_batch) :
                raw_input = seq
                input_seq = np.array(raw_input[::2])
                slow_seq = np.array(raw_input[15:45])
                input_batch.append(input_seq)
                slow_batch.append(slow_seq)
            batch = [input_batch, slow_batch]

        return batch
        
    def get_sequence_batch_valid(self,slow_rule=False):
        
        mads_videos = os.listdir(os.path.join(self.data_path,'valid/'))
        videos = [video for video in mads_videos if video.rsplit('_',1)[0] in self.val_list]
        count = 1
        batch= []
        while True :
            if count > self.batch_size:
                break
            rv=random.choice(videos)
            seq_no = random.randint(0,loadmat(os.path.join(self.data_path,'valid',rv))['pose_3d'].shape[0]-self.sequence_length)
            k=loadmat(os.path.join(self.data_path,'valid',rv))['pose_3d'][seq_no:seq_no + self.sequence_length].reshape((self.sequence_length,H.num_joints*3))
            batch.append(k) 
            count = count + 1
            
        if slow_rule :
            tmp_batch = np.array(batch).copy()
            batch
            input_batch,slow_batch = [], []
            for i,seq in enumerate(tmp_batch) :
                raw_input = seq
                input_seq = np.array(raw_input[::2])
                slow_seq = np.array(raw_input[15:45])
                input_batch.append(input_seq)
                slow_batch.append(slow_seq)
            batch = [input_batch, slow_batch]
        return batch
        


