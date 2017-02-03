from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



# Parameters
learning_rate = 0.001
training_iters = 50000
n_steps = 100 # timesteps per sequence of the input
batch_size = 128
display_step = 10

tau = 1.0
alpha = 1.0 - tau
dale_ratio = 0.8
rec_noise = .1


# Network Parameters
n_in = 2 #ins
n_hidden = 50 # hidden layer num of features
n_out = 2 # outs


# tf Graph input
x = tf.placeholder("float", [batch_size, n_steps, n_in])
y = tf.placeholder("float", [batch_size, n_steps, n_out])
init_state = tf.random_normal([batch_size, n_hidden], mean=0.0, stddev=rec_noise)

#dale's ratio matrix
dale_vec = np.ones(n_hidden)
dale_vec[int(dale_ratio*n_hidden):] = -1
dale = np.diag(dale_vec)

#trainable variables
with tf.variable_scope('rnn_cell'):
    U = tf.get_variable('U', [n_in, n_hidden])
    W = tf.get_variable('W', [n_hidden, n_hidden])
    Z = tf.get_variable('Z',[n_hidden, n_out])
    Dale = tf.Variable(dale, trainable = false, name='Dale')
    #b = tf.get_variable('b', [n_hidden], initializer=tf.constant_initializer(0.0))


#step function
#returns new_output, new_state
def rnn_cell(rnn_in, state):
    with tf.variable_scope('rnn_cell', reuse=True):
        U = tf.get_variable('U')
        W = tf.get_variable('W')
        Z = tf.get_variable('Z')
        Dale = tf.get_variable('Dale')
        new_state = state * tau + alpha * (tf.matmul(tf.nn.relu(state), tf.matmul(tf.abs(W), Dale)) + tf.matmul(tf.abs(rnn_in), U)) \
            + tf.random_normal(state.get_shape(), mean=0.0, stddev=rec_noise)
        new_output = tf.matmul(tf.nn.relu(new_state), tf.matmul(tf.abs(Z), Dale))
    return new_output, new_state


def RNN(x):
    rnn_inputs = tf.unpack(x, axis=1)

    state = init_state
    rnn_outputs = []
    rnn_states = []
    for rnn_input in rnn_inputs:
        output, state = rnn_cell(rnn_input, state)
        rnn_outputs.append(output)
        rnn_states.append(state)

    return rnn_outputs


pred = RNN(x)


def reg_loss(pred, y):
    return tf.reduce_mean(tf.square(pred - y))


# Define loss and optimizer
cost = reg_loss(pred, y)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()
