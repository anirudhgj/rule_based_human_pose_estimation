import tensorflow as tf
import numpy as np


def get_a_cell(lstm_size):
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    return lstm


def apply_encoder(input_x_seq,
                  name,
                  num_stacked_lstm_layers=2,
                  state_size=128,
                  reuse=tf.AUTO_REUSE):

    with tf.variable_scope(name, reuse=reuse):

        cell_fw_encoder = tf.nn.rnn_cell.MultiRNNCell(
            [get_a_cell(state_size) for _ in range(num_stacked_lstm_layers)]
        )

        cell_bw_encoder = tf.nn.rnn_cell.MultiRNNCell(
            [get_a_cell(state_size) for _ in range(num_stacked_lstm_layers)]
        )

        outputs_encoder, state_encoder = tf.nn.bidirectional_dynamic_rnn(cell_fw_encoder, cell_bw_encoder, input_x_seq, dtype=tf.float32)

        final_state_fw = state_encoder[0][1][1]

        final_state_bw = state_encoder[1][1][1]

        final_state = tf.concat([final_state_fw, final_state_bw], 1)

        act = lambda x: 10 * tf.tanh(x/10)

        final_state_fced = tf.contrib.layers.fully_connected(final_state, state_size, activation_fn=act)

        final_state_fced_stacked = tf.stack([final_state_fced]*input_x_seq.shape[1], 1)

        encoder_net={
            'input' : input_x_seq,
            'z_state':final_state_fced,
            'z_outputs' : final_state_fced_stacked

        }

    return encoder_net

def apply_decoder(z_state,
                  z_outputs,
                  name,
                  num_stacked_lstm_layers=2,
                  state_size=128,
                  reuse=tf.AUTO_REUSE):

    with tf.variable_scope(name, reuse=reuse):

        final_state_fced = z_state

        final_state_fced_stacked = z_outputs

        rnn_tuple_state_fw = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(final_state_fced, final_state_fced)
            for idx in range(num_stacked_lstm_layers)]
            )

        rnn_tuple_state_bw = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(final_state_fced, final_state_fced)
            for idx in range(num_stacked_lstm_layers)]
            )

        cell_fw_decoder = tf.nn.rnn_cell.MultiRNNCell(
            [get_a_cell(state_size) for _ in range(num_stacked_lstm_layers)]
        )

        cell_bw_decoder = tf.nn.rnn_cell.MultiRNNCell(
            [get_a_cell(state_size) for _ in range(num_stacked_lstm_layers)]
        )

        outputs_decoder, state_decoder = tf.nn.bidirectional_dynamic_rnn(cell_fw_decoder, cell_bw_decoder, 
                                                                    final_state_fced_stacked, 
                                                                    initial_state_fw=rnn_tuple_state_fw, initial_state_bw=rnn_tuple_state_bw)

        outputs_decoder_merged = tf.concat(outputs_decoder, 2)

        act = lambda x: 10 * tf.tanh(x/10)

        final_output_pred = tf.contrib.layers.fully_connected(outputs_decoder_merged, 32, activation_fn=act)


        decoder_net ={
            'z_state' :z_state,
            'z_outputs' : z_outputs,
            'x_recon' : final_output_pred 
        }

        return decoder_net

def apply_motion_rule_net(input,
                          name,
                          state_size=128,
                          seq_length=30,
                          reuse=tf.AUTO_REUSE):

    with tf.variable_scope(name, reuse=reuse):

        out_1 = tf.contrib.layers.fully_connected(input, 1024)
        out_2 = tf.contrib.layers.fully_connected(out_1, 512)
        out_3 = tf.contrib.layers.fully_connected(out_2, 256)

        act = lambda x: 10 * tf.tanh(x/10)

        out_4 = tf.contrib.layers.fully_connected(out_2, state_size,activation_fn=act)
        
        mapped_out_4 = tf.stack([out_4]* seq_length, 1)
        
        mapped={
            'mapped_state':out_4,
            'mapped_outputs':mapped_out_4
        }

    return mapped

def apply_rule_net(input,
                   name,
                   state_size=128,
                   reuse=tf.AUTO_REUSE):

    with tf.variable_scope(name, reuse=reuse):

        out_1 = tf.contrib.layers.fully_connected(input, 1024)
        out_2 = tf.contrib.layers.fully_connected(out_1, 512)
        out_3 = tf.contrib.layers.fully_connected(out_2, 256)

        act = lambda x: 10 * tf.tanh(x/10)

        out_4 = tf.contrib.layers.fully_connected(out_2, state_size,activation_fn=act)

    return out_4


def get_network_params(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
