from __future__ import print_function

import tensorflow as tf
import numpy as np

# Lets make sure to keep things object-oriented,
# so that all future networks we build will extend
# the Model class below

# This will mean (in the future) making Model less specific so
# that future networks will "fill in the specifics" instead
# i.e. we can make a denseRNN, a sparseRNN, a denseCNN etc


class Model(object):
    def __init__(self, params, autapse=True):

        # Network sizes (tensor dimensions)
        N_in    = self.N_in       = params['N_in']
        N_rec   = self.N_rec      = params['N_rec']
        N_out   = self.N_out      = params['N_out']
        N_steps = self.N_steps    = params['N_steps']
        N_batch = self.batch_size = params['N_batch']

        # Physical parameters
        self.dt = params['dt']
        self.tau = params['tau']
        self.alpha = params['alpha']
        self.dale_ratio = params['dale_ratio']
        self.rec_noise  = params['rec_noise']

        # Dale matrix
        dale_vec = np.ones(N_rec)
        if self.dale_ratio:
            dale_vec[int(self.dale_ratio * N_rec):] = -1
            self.dale_rec = np.diag(dale_vec)
            dale_vec[int(self.dale_ratio * N_rec):] = 0
            self.dale_out = np.diag(dale_vec)
        else:
            self.dale_rec = np.diag(dale_vec)
            self.dale_out = np.diag(dale_vec)

        # Connectivity
        self.connect_mat = np.ones((N_rec, N_rec))
        autapse = self.autapse = params.get('autapse', True)
        if not autapse:
            self.connect_mat -= np.diag(np.ones(N_rec))

        # Tensorflow initializations
        self.x = tf.placeholder("float", [N_batch, N_steps, N_in])
        self.y = tf.placeholder("float", [N_batch, N_steps, N_out])
        self.output_mask = tf.placeholder("float", [N_batch, N_steps, N_out])

        # trainable variables
        with tf.variable_scope("model"):
            
            self.init_state = tf.get_variable('init_state', [N_batch, N_rec],
                                              initializer=tf.random_normal_initializer(mean=0.1, stddev=0.01))

            # ------------------------------------------------
            # Trainable variables:
            # Weight matrices and bias weights
            # ------------------------------------------------

            # Input weight matrix:
            # (uniform initialization as in pycog)
            self.W_in = \
                tf.get_variable('W_in', [N_rec, N_in],
                                initializer=tf.constant_initializer(
                                    0.1 * np.random.uniform(-1, 1, size=(self.N_rec, self.N_in))))
            # Recurrent weight matrix:
            # (gamma (Dale) or normal (non-Dale) initialization)
            self.W_rec = \
                tf.get_variable(
                    'W_rec',
                    [N_rec, N_rec],
                    initializer=tf.constant_initializer(self.initial_W()))
            # Output weight matrix:
            # (uniform initialization as in pycog)
            self.W_out = tf.get_variable('W_out', [N_out, N_rec],
                                         initializer=tf.constant_initializer(
                                             0.1 * np.random.uniform(-1, 1, size=(self.N_out, self.N_rec))))
            # Recurrent bias:
            self.b_rec = tf.get_variable('b_rec', [N_rec], initializer=tf.constant_initializer(0.0))
            # Output bias:
            self.b_out = tf.get_variable('b_out', [N_out], initializer=tf.constant_initializer(0.0))

            # ------------------------------------------------
            # Non-trainable variables:
            # Overall connectivity and Dale's law matrices
            # ------------------------------------------------

            # Recurrent Dale's law weight matrix:
            self.Dale_rec = tf.get_variable('Dale_rec', [N_rec, N_rec],
                                            initializer=tf.constant_initializer(self.dale_rec),
                                            trainable=False)
            # Output Dale's law weight matrix:
            self.Dale_out = tf.get_variable('Dale_out', [N_rec, N_rec],
                                            initializer=tf.constant_initializer(self.dale_out),
                                            trainable=False)
            # Recurrent connectivity weight matrix:
            self.Connectivity = tf.get_variable('Connectivity', [N_rec, N_rec],
                                                initializer=tf.constant_initializer(self.connect_mat),
                                                trainable=False)

            # ------------------------------------------------
            # Network loss
            # ------------------------------------------------
            self.predictions, self.states = self.compute_predictions_scan()
            self.loss = self.reg_loss()

    # regularized loss function
    def reg_loss(self):
        return tf.reduce_mean(tf.square(self.output_mask * (self.predictions - self.y)))

    # implement one step of the RNN
    def rnn_step(self, rnn_in, state):
        
        if self.dale_ratio:
            new_state = (1-self.alpha) * state \
                        + self.alpha * (
                            tf.matmul(
                                tf.nn.relu(state),
                                tf.matmul(
                                    tf.abs(self.W_rec) * self.Connectivity,
                                    self.Dale_rec, name="in_1"),
                                transpose_b=True, name="1")
                            + tf.matmul(
                                rnn_in,
                                tf.abs(self.W_in),
                                transpose_b=True, name="2")
                            + self.b_rec)\
                        + tf.random_normal(state.get_shape(), mean=0.0, stddev=self.rec_noise)

            new_output = \
                tf.matmul(
                    tf.nn.relu(new_state),
                    tf.matmul(
                        tf.abs(self.W_out),
                        self.Dale_out,
                        name="in_2"),
                    transpose_b=True, name="3")\
                + self.b_out

        else:
            new_state = ((1-self.alpha) * state) \
                        + self.alpha * (
                            tf.matmul(
                                tf.nn.relu(state),
                                self.W_rec,
                                transpose_b=True, name="1")
                            + tf.matmul(
                                rnn_in,
                                tf.abs(self.W_in),
                                transpose_b=True, name="2")
                            + self.b_rec)\
                        + tf.random_normal(state.get_shape(), mean=0.0, stddev=self.rec_noise)
            new_output = \
                tf.matmul(
                    tf.nn.relu(new_state),
                    self.W_out,
                    transpose_b=True, name="3")\
                + self.b_out

            
        return new_output, new_state

    def rnn_step_scan(self, state, rnn_in):

        if self.dale_ratio:
            new_state = (1-self.alpha) * state \
                        + self.alpha * (
                            tf.matmul(
                                tf.nn.relu(state),
                                tf.matmul(
                                    tf.abs(self.W_rec) * self.Connectivity,
                                    self.Dale_rec, name="in_1"),
                                transpose_b=True, name="1")
                            + tf.matmul(
                                rnn_in,
                                tf.abs(self.W_in),
                                transpose_b=True, name="2")
                            + self.b_rec) \
                        + tf.random_normal(state.get_shape(), mean=0.0, stddev=self.rec_noise)
        else:
            new_state = ((1 - self.alpha) * state) \
                        + self.alpha * (
                            tf.matmul(
                                tf.nn.relu(state),
                                self.W_rec,
                                transpose_b=True, name="1")
                            + tf.matmul(
                                rnn_in,
                                tf.abs(self.W_in),
                                transpose_b=True, name="2")
                            + self.b_rec) \
                        + tf.random_normal(state.get_shape(), mean=0.0, stddev=self.rec_noise)

        return new_state
    def output_step_scan(self, dummy, new_state):

        if self.dale_ratio:
            new_output = tf.matmul(
                            tf.nn.relu(new_state),
                            tf.matmul(
                                tf.abs(self.W_out),
                                self.Dale_out,
                                name="in_2"),
                            transpose_b=True, name="3")\
                         + self.b_out

        else:
            new_output = tf.matmul(tf.nn.relu(new_state), self.W_out, transpose_b=True, name="3") + self.b_out

        return new_output

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


    def compute_predictions_scan(self):

        state = self.init_state
        rnn_states = \
            tf.scan(
                self.rnn_step_scan,
                tf.transpose(self.x, [1, 0, 2]),
                initializer=state,
                parallel_iterations=1)
        rnn_outputs = \
            tf.scan(
                self.output_step_scan,
                rnn_states,
                initializer=tf.zeros([self.batch_size, 1]),
                parallel_iterations= 1)
        return tf.transpose(rnn_outputs, [1, 0, 2]), tf.transpose(rnn_states, [1, 0, 2])


    #fix spectral radius of recurrent matrix
    def initial_W(self):
        
        #added gamma distributed initial weights as in pycog
        if self.dale_ratio:
            self.W_dist0 = 'gamma'
        else:
            self.W_dist0 = 'normal'
        
        if self.W_dist0 == 'normal':
            w0 = np.random.normal(scale=1, size=(self.N_rec, self.N_rec))
        elif self.W_dist0 == 'gamma':
            k = 2
            theta = 0.1/k
            w0 = np.random.gamma(k, theta, size=(self.N_rec, self.N_rec))
                
        
        if self.dale_ratio:
            W = np.matmul(abs(w0), self.dale_rec)
        else:
            W = w0
            
        rho = max(abs(np.linalg.eigvals(W)))#+np.diag(np.ones(self.N_rec)*(1-self.alpha))))) #add diagnal matrix 1-alpha to account for persistance tau
        return (1.1/rho) * W  # - .9*np.diag(np.ones(self.N_rec)*(1-self.alpha)) #correct for tau

    # train the model using Adam
    def train(self, sess, generator,
              learning_rate=.001, training_iters=50000,
              batch_size=64, display_step=10, weights_path= None):

        var_list = [self.W_rec, self.W_in, self.W_out,
                    self.b_rec, self.b_out,
                    self.init_state]

        optimizer = tf.train.\
            AdamOptimizer(learning_rate=learning_rate).\
            minimize(self.loss, var_list=var_list)

        sess.run(tf.global_variables_initializer())
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_x, batch_y, output_mask = generator.next()
            sess.run(optimizer, feed_dict={self.x: batch_x, self.y: batch_y, self.output_mask: output_mask})
            if step % display_step == 0:
                # Calculate batch loss
                loss = sess.run(self.loss,
                                feed_dict={self.x: batch_x, self.y: batch_y, self.output_mask: output_mask})
                print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss))
            step += 1
        print("Optimization Finished!")

        #save weights
        if weights_path:
            saver = tf.train.Saver()
            save_path = saver.save(sess, weights_path)
            print("Model saved in file: %s" % save_path)


    # use a trained model to get test outputs
    def test(self, sess, rnn_in, weights_path = None):
        if(weights_path):
            saver = tf.train.Saver()
            # Restore variables from disk.
            saver.restore(sess, weights_path)
            predictions, states = sess.run([self.predictions, self.states], feed_dict={self.x: rnn_in})
        else:
            predictions, states = sess.run([self.predictions, self.states], feed_dict={self.x: rnn_in})

        return predictions, states



