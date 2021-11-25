import os.path as osp
import os
import cv2
import numpy as np
import random
import copy
import scipy.io as sio
import tensorflow as tf
import glob as glob
import json
from natsort import natsorted
import math
import imutils
import pickle
import h5py
from multiprocessing import Pool


# h36_datapath = '../../../dataset/dataset_standardization/H36_cropped/'
h36_datapath = '/data/vcl/anirudh_rule_based/remotes/128/adit/dataset_standardization/H36_cropped'
subjects = ['S1','S5','S6','S7','S8','S9','S11']
cameras = ['54138969','55011271','58860488','60457274']

annot_files = {}
frames = {}
cameras = {}

for subject in subjects:
  subject_datapath = osp.join(h36_datapath, subject)
  activities = os.listdir(subject_datapath)
  annot_files[subject] = {}
  frames[subject] = {}
  cameras[subject] = {}

  print ("subject ", subject)

  for activity in activities:
    try:
      curr_annot_path = osp.join(subject_datapath, activity + '/annot.pickle')
      infile = open(curr_annot_path,'rb')
      #fix for python3 pickle: https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
      curr_annot_mat = pickle.load(infile, encoding = 'latin1')
      # curr_annot_mat = h5py.File(curr_annot_path,'r+')
      annot_files[subject][activity] = {}
      annot_files[subject][activity]['poses2d'] = curr_annot_mat['poses2d']
      annot_files[subject][activity]['poses3d'] = curr_annot_mat['poses3d']
      frames[subject][activity] = np.load(osp.join(subject_datapath, activity + '/frames.npy'))
      cameras[subject][activity] = np.load(osp.join(subject_datapath, activity + '/cameras.npy'))
      # print ("Subject " + subject + ", activity " + activity + " done")
    except Exception as e:
      print ("Subject " + subject + ", activity " + activity + " cannot be processed because of:")
      print (e)

