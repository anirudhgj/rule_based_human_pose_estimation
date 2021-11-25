import numpy as np

class Hyperparameters(object):

    def __init__(self):
        self.exp_name = 'cnn_stickman_training'
        self.data_path = '../data/stickman/*/*.mat'
        self.back_path = '/data/vcl/sid/backgrounds/*/*/*.jpg'
        self.train_path = './data_split/train_paths.npy'
        self.val_path = './data_split/val_paths.npy'
        self.test_path = './data_split/test_paths.npy'
        self.logdir_path_train = './logs/' + self.exp_name + '/train'
        self.logdir_path_val = './logs/' + self.exp_name + '/val'
        self.store_weights_path = './weights/whole/'
        self.load_weights_path = './weights/whole/'
        self.store_resnet_weights_path = '../pretrained_weights/image_to_latent_encoder/resnet'
        self.store_decoder_weights_path = '../pretrained_weights/pose_encoder_decoder/decoder_iter-1475001'
        self.store_embedding_branch_weights_path = '../pretrained_weights/image_to_latent_encoder/embedding_branch'
        self.cuda_device_id = '0'
        self.learning_rate = 1e-05
        self.num_threads = 48
        self.num_epochs = 1000
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
