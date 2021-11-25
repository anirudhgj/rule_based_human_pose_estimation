import numpy as np

class Hyperparameters(object):

    def __init__(self):
        self.exp_name = 'cnn_training'
        self.logdir_path_train = './logs/' + self.exp_name + '/train'
        self.logdir_path_val = './logs/' + self.exp_name + '/val'
        self.store_weights_path = './weights/whole/'
        self.load_weights_path = './weights/whole/'
        self.store_resnet_weights_path = './weights/resnet/'
        self.store_decoder_weights_path = './weights/pose_decoder/'
        self.store_embedding_branch_weights_path = './weights/embedding_branch/'
        self.cuda_device_id = '0'
        self.learning_rate = 1e-05
        self.num_threads = 16
        self.num_epochs = 10
        self.num_joints = 17
        self.state_size = 128
        self.batch_size = 128
        self.max_iterations = 100000
        self.val_after = 100
        self.seq_length = 30
        self.num_stacked_lstm_layers = 2
        self.image_size = 224
        self.flip_prob = 0.4
        self.tilt_prob = 0.2
        self.tilt_limit = 10
        self.jitter_prob = 0.3
