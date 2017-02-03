from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def set_params(nturns = 3, input_wait = 3, quiet_gap = 4, stim_dur = 3,
                    var_delay_length = 0, stim_noise = 0, rec_noise = .1,
                    sample_size = 128, epochs = 100, N_rec = 50, dale_ratio=0.8, tau=100):
    params = dict()
    params['nturns']          = nturns
    params['input_wait']       = input_wait
    params['quiet_gap']        = quiet_gap
    params['stim_dur']         = stim_dur
    params['var_delay_length'] = var_delay_length
    params['stim_noise']       = stim_noise
    params['rec_noise']        = rec_noise
    params['sample_size']      = sample_size
    params['epochs']           = epochs
    params['N_rec']            = N_rec
    params['dale_ratio']       = dale_ratio
    params['tau'] = tau

    return params

# This generates the training data for our network
# It will be a set of input_times and output_times for when we expect input
# and when the corresponding output is expected
def generate_trials(params):
    nturns = params['nturns']
    input_wait = params['input_wait']
    quiet_gap = params['quiet_gap']
    stim_dur = params['stim_dur']
    var_delay_length = params['var_delay_length']
    stim_noise = params['stim_noise']
    sample_size = int(params['sample_size'])

    if var_delay_length == 0:
        var_delay = np.zeros(sample_size, dtype=int)
    else:
        var_delay = np.random.randint(var_delay_length, size=sample_size) + 1

    input_times = np.zeros([sample_size, nturns], dtype=np.int)
    output_times = np.zeros([sample_size, nturns], dtype=np.int)

    turn_time = np.zeros(sample_size, dtype=np.int)

    for sample in np.arange(sample_size):
        turn_time[sample] = stim_dur + quiet_gap + var_delay[sample]
        for i in np.arange(nturns):
            input_times[sample, i] = input_wait + i * turn_time[sample]
            output_times[sample, i] = input_wait + i * turn_time[sample] + stim_dur

    seq_dur = int(max([output_times[sample, nturns - 1] + quiet_gap, sample in np.arange(sample_size)]))

    x_train = np.zeros([sample_size, seq_dur, 2])
    y_train = 0.5 * np.ones([sample_size, seq_dur, 1])
    for sample in np.arange(sample_size):
        for turn in np.arange(nturns):
            firing_neuron = np.random.randint(2)  # 0 or 1
            x_train[sample,
            input_times[sample, turn]:(input_times[sample, turn] + stim_dur),
            firing_neuron] = 1
            y_train[sample,
            output_times[sample, turn]:(input_times[sample, turn] + turn_time[sample]),
            0] = firing_neuron

    mask = np.zeros((sample_size, seq_dur))
    for sample in np.arange(sample_size):
        mask[sample, :] = [0 if x == .5 else 1 for x in y_train[sample, :, :]]

    x_train = x_train + stim_noise * np.random.randn(sample_size, seq_dur, 2)
    params['input_times'] = input_times
    params['output_times'] = output_times
    return x_train, y_train, mask




# Parameters
learning_rate = 0.001
training_iters = 200000
n_steps = 800 # timesteps per sequence of the input
batch_size = 128
display_step = 10

tau = 1.0
alpha = 1.0 - tau
dale_ratio = 0.8
rec_noise = .1


# Network Parameters
n_in = 2 #ins
n_hidden = 50 # hidden layer num of features
n_out = 1 # outs


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
    Z = tf.get_variable('Z', [n_hidden, n_out])
    Dale = tf.get_variable('Dale', [n_hidden, n_hidden], initializer=tf.constant_initializer(dale), trainable=False)
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
        #Is this next line right????
        new_output = tf.matmul(tf.nn.relu(new_state), tf.matmul(Dale, tf.abs(Z)))
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

    return tf.transpose(rnn_outputs, [1, 0, 2])


pred = RNN(x)


def reg_loss(pred, y):
    return tf.reduce_mean(tf.square(pred - y))


# Define loss and optimizer
cost = reg_loss(pred, y)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    params = set_params(sample_size= 128, input_wait=50, stim_dur=50, quiet_gap=100, nturns=5)
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y, mask = generate_trials(params)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) )
        step += 1
    print("Optimization Finished!")
    wrec = sess.run(W)


#todo: masking, visualization, saving trained networks, performance tests
