from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Model(object):
    def __init__(self, n_in, n_hidden, n_out, n_steps, tau, dale_ratio, rec_noise, batch_size):
        # network size
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.n_steps = n_steps
        self.batch_size = batch_size

        # neuro parameters
        self.dt = .1
        self.tau = tau
        self.alpha = self.dt / self.tau
        self.dale_ratio = dale_ratio
        self.rec_noise = rec_noise

        # dale matrix
        dale_vec = np.ones(n_hidden)
        dale_vec[int(dale_ratio * n_hidden):] = -1
        self.dale_rec = np.diag(dale_vec)
        dale_vec[int(dale_ratio * n_hidden):] = 0
        self.dale_out = np.diag(dale_vec)

        #connectivity
        self.connect_mat = np.ones((n_hidden, n_hidden)) - np.diag(np.ones(n_hidden))

        # tensorflow initializations
        self.x = tf.placeholder("float", [batch_size, n_steps, n_in])
        self.y = tf.placeholder("float", [batch_size, n_steps, n_out])
        self.output_mask = tf.placeholder("float", [batch_size, n_steps, n_out])

        self.init_state = tf.random_normal([batch_size, n_hidden], mean=0.0, stddev=rec_noise)

        # trainable variables
        with tf.variable_scope("model"):
            self.U = tf.get_variable('U', [n_hidden, n_in], initializer=tf.random_normal_initializer(stddev=.01))
            self.W = tf.get_variable('W', [n_hidden, n_hidden], initializer=tf.constant_initializer(self.initial_W()))
            self.Z = tf.get_variable('Z', [n_out, n_hidden], initializer=tf.random_normal_initializer(stddev=.01))
            self.Dale_rec = tf.get_variable('Dale_rec', [n_hidden, n_hidden],
                                            initializer=tf.constant_initializer(self.dale_rec),
                                            trainable=False)
            self.Connectivity = tf.get_variable('Connectivity', [n_hidden, n_hidden],
                                            initializer=tf.constant_initializer(self.connect_mat),
                                            trainable=False)
            self.Dale_out = tf.get_variable('Dale_out', [n_hidden, n_hidden],
                                            initializer=tf.constant_initializer(self.dale_out),
                                            trainable=False)
            self.brec = tf.get_variable('brec', [n_hidden], initializer=tf.constant_initializer(0.0))
            self.bout = tf.get_variable('bout', [n_out], initializer=tf.constant_initializer(0.0))

            self.predictions, self.states = self.compute_predictions()
            self.loss = tf.losses.mean_squared_error(self.y, self.predictions, weights=self.output_mask)

    # implement one step of the RNN

    def rnn_step(self, rnn_in, state):
        new_state = state * (1-self.alpha) + self.alpha * (tf.matmul(tf.nn.relu(state),
                                                               tf.matmul(tf.abs(self.W) * self.Connectivity, self.Dale_rec, name="in1"),
                                                               transpose_b=True, name="1")
                                                     + tf.matmul(rnn_in, tf.abs(self.U), transpose_b=True, name="2") + self.brec) + \
                                                        tf.random_normal(state.get_shape(), mean=0.0, stddev=self.rec_noise)
        new_output = tf.matmul(tf.nn.relu(new_state), tf.matmul(tf.abs(self.Z), self.Dale_out, name="in2"), transpose_b=True, name="3") + self.bout
        return new_output, new_state

    # apply the step to a full input vector
    def compute_predictions(self):
        rnn_inputs = tf.unstack(self.x, axis=1)

        state = self.init_state
        rnn_outputs = []
        rnn_states = []
        for rnn_input in rnn_inputs:
            output, state = self.rnn_step(rnn_input, state)
            rnn_outputs.append(output)
            rnn_states.append(state)

        return tf.transpose(rnn_outputs, [1, 0, 2]), tf.transpose(rnn_states, [1, 0, 2])
        # return tf.scan(self.rnn_step, tf.transpose(self.x, [1,0,2]), initializer = self.init_state)

    # regularized loss function
    def reg_loss(self):
        return tf.reduce_mean(tf.square(self.predictions - self.y))

    #fix spectral radius of recurrent matrix
    def initial_W(self):
        dale_scale = np.ones(50) * 1/.8;
        dale_scale[int(.8 * 50):] = 1/.2
        dale_scale = np.diag(dale_scale)

        W = np.matmul(abs(np.random.normal(scale=.01, size=(self.n_hidden, self.n_hidden))), self.dale_rec)
        W = np.matmul(W, dale_scale)
        rho = max(abs(np.linalg.eigvals(W)))
        return (1.1/rho) * W


# train the model using Adam
def train(sess, model, generator, learning_rate, training_iters, batch_size, display_step):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model.loss)
    sess.run(tf.global_variables_initializer())
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y, output_mask = generator.next()
        sess.run(optimizer, feed_dict={model.x: batch_x, model.y: batch_y, model.output_mask: output_mask})
        if step % display_step == 0:
            # Calculate batch loss
            loss = sess.run(model.loss, feed_dict={model.x: batch_x, model.y: batch_y, model.output_mask: output_mask})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss))
        step += 1
    print("Optimization Finished!")
    plt.imshow(np.matmul(abs(model.W.eval(session=sess)), model.dale_rec), interpolation="none")
    plt.colorbar()
    plt.show()
    #plt.imshow(np.matmul(abs(model.Z.eval(session=sess)), model.dale_out), interpolation="none")
    #plt.show()


# use a trained model to get test outputs
def test(sess, model, input):
    preds = sess.run(model.predictions, feed_dict={model.x: input})
    return preds

#visualize network output on a trial, compared to desired output
def visualize_trial(sess, model, input, desired_output):
    preds = test(sess, model, input)
    plt.plot()
    return
