from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



class Model(object):

    def __init__(self, n_in, n_hidden, n_out, n_steps, tau, dale_ratio, rec_noise, batch_size):
        #network size
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.n_steps = n_steps
        self.batch_size = batch_size

        #neuro parameters
        self.tau = tau
        self.alpha = 1.0 - tau
        self.dale_ratio = dale_ratio
        self.rec_noise = rec_noise

        #dale matrix
        dale_vec = np.ones(n_hidden)
        dale_vec[int(dale_ratio * n_hidden):] = -1
        self.dale = np.diag(dale_vec)

        #tensorflow initializations
        self.x = tf.placeholder("float", [batch_size, n_steps, n_in])
        self.y = tf.placeholder("float", [batch_size, n_steps, n_out])
        self.init_state = tf.random_normal([batch_size, n_hidden], mean=0.0, stddev=rec_noise)

        # trainable variables
        with tf.variable_scope("model"):
            self.U = tf.get_variable('U', [n_in, n_hidden])
            self.W = tf.get_variable('W', [n_hidden, n_hidden])
            self.Z = tf.get_variable('Z', [n_hidden, n_out])
            self.Dale = tf.get_variable('Dale', [n_hidden, n_hidden], initializer=tf.constant_initializer(self.dale),
                                   trainable=False)
            self.brec = tf.get_variable('brec', [n_hidden], initializer=tf.constant_initializer(0.0))
            self.bout = tf.get_variable('bout', [n_hidden], initializer=tf.constant_initializer(0.0))

            self.predictions, self.states = self.compute_predictions()
            self.loss = self.reg_loss()

    #implement one step of the RNN
    def rnn_step(self, rnn_in, state):

        new_state = state * self.tau + self.alpha * (tf.matmul(tf.nn.relu(state), tf.matmul(tf.abs(self.W), self.Dale))
            + tf.matmul(tf.abs(rnn_in), self.U) + self.brec) + tf.random_normal(state.get_shape(), mean=0.0, stddev=self.rec_noise)
        # Is this next line right????
        new_output = tf.matmul(tf.nn.relu(new_state), tf.matmul(self.Dale, tf.abs(self.Z))) + self.bout
        return new_output, new_state

    #apply the step to a full input vector
    def compute_predictions(self):
        rnn_inputs = tf.unpack(self.x, axis=1)

        state = self.init_state
        rnn_outputs = []
        rnn_states = []
        for rnn_input in rnn_inputs:
            output, state = self.rnn_step(rnn_input, state)
            rnn_outputs.append(output)
            rnn_states.append(state)

        return tf.transpose(rnn_outputs, [1, 0, 2]), tf.transpose(rnn_states, [1, 0, 2])
        #return tf.scan(self.rnn_step, tf.transpose(self.x, [1,0,2]), initializer = self.init_state)

    #regularized loss function
    def reg_loss(self):
        return tf.reduce_mean(tf.square(self.predictions - self.y))

#train the model using Adam
def train(sess, model, generator, learning_rate, training_iters, batch_size, display_step):

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model.loss)
    sess.run(tf.initialize_all_variables())
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y, mask = generator.next()
        sess.run(optimizer, feed_dict={model.x: batch_x, model.y: batch_y})
        if step % display_step == 0:
            # Calculate batch loss
            loss = sess.run(model.loss, feed_dict={model.x: batch_x, model.y: batch_y})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss))
        step += 1
    print("Optimization Finished!")

#use a trained model to get test outputs
def test(self):
    return


