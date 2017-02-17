from __future__ import print_function

import tensorflow as tf
import numpy as np



class Model(object):

    def __init__(self, N_in=2, N_rec=50, N_out=1,
                 steps=100, tau=100, dale_ratio=0.8,
                 rec_noise=0.1, batch_size=256,
                 output_mask = None):
        #network size
        self.N_in = N_in
        self.N_rec = N_rec
        self.N_out = N_out
        self.steps = steps
        self.batch_size = batch_size

        #neuro parameters
        self.tau = tau
        self.alpha = 1.0 - tau
        self.dale_ratio = dale_ratio
        self.rec_noise = rec_noise

        #dale matrix
        dale_vec = np.ones(N_rec)
        dale_vec[int(dale_ratio * N_rec):] = -1
        self.dale_rec = np.diag(dale_vec)
        dale_vec[int(dale_ratio * N_rec):] = 0
        self.dale_out = np.diag(dale_vec)
        

        #tensorflow initializations 
        self.x = tf.placeholder("float", [batch_size, steps, N_in])
        self.y = tf.placeholder("float", [batch_size, steps, N_out])
        self.output_mask = output_mask #tf.placeholder("int", [batch_size, steps, N_out])
        self.init_state = tf.random_normal([batch_size, N_rec], mean=0.0, stddev=rec_noise)

        # trainable variables
        with tf.variable_scope("model"):
            #TODO: look into the initialization of the following 3 weight-matrices
            self.U = tf.get_variable('U', [N_rec, N_in])
            self.W = tf.get_variable('W', [N_rec, N_rec])
            self.Z = tf.get_variable('Z', [N_out, N_rec,])
            
            self.Dale_rec = tf.get_variable('Dale_rec', [N_rec, N_rec],
                                                initializer=tf.constant_initializer(self.dale_rec),
                                                trainable=False)
            self.Dale_out = tf.get_variable('Dale_out', [N_rec, N_rec],
                                                initializer=tf.constant_initializer(self.dale_out),
                                                trainable=False)
            if self.output_mask is not None:
                self.Mask = tf.get_variable('Mask', output_mask.shape,
                                                initializer=tf.constant_initializer(self.output_mask),
                                                trainable=False)
            else:
                self.Mask = 1.0
            
            # As they are now, these are trainable:
            self.brec = tf.get_variable('brec', [N_rec], initializer=tf.constant_initializer(0.0))
            self.bout = tf.get_variable('bout', [N_rec], initializer=tf.constant_initializer(0.0))

            self.predictions, self.states = self.compute_predictions()
            self.loss = tf.losses.mean_squared_error(self.predictions, self.y, weights=self.Mask)
            #self.loss = self.reg_loss()

    #implement one step of the RNN
    def rnn_step(self, rnn_in, state):

        new_state = state * self.tau \
                    + self.alpha \
                        * tf.transpose(
                            tf.tensordot(tf.matmul(tf.abs(self.W), self.Dale_rec),tf.nn.relu(state), [[1], [1]])
                            + tf.tensordot(self.U, tf.abs(rnn_in), [[1], [1]]))\
                    + tf.random_normal(state.get_shape(), mean=0.0, stddev=self.rec_noise)

        new_output = tf.transpose(
                        tf.tensordot(tf.matmul(tf.abs(self.Z), self.Dale_out),
                                     tf.nn.relu(new_state), [[1], [1]])) #+ self.bout)
        return new_output, new_state

    #apply the step to a full input vector
    def compute_predictions(self):
        rnn_inputs = tf.unstack(self.x, axis=1)

        state = self.init_state
        rnn_outputs = []
        rnn_states = []
        for rnn_input in rnn_inputs:                    # This is indeed slow
            output, state = self.rnn_step(rnn_input, state)
            rnn_outputs.append(output)
            rnn_states.append(state)

        return tf.transpose(rnn_outputs, [1, 0, 2]), tf.transpose(rnn_states, [1, 0, 2])
        #return tf.scan(self.rnn_step, tf.transpose(self.x, [1,0,2]), initializer = self.init_state)

    def reg_loss(self):
        return tf.reduce_mean(tf.multiply(tf.square(self.predictions - self.y), self.output_mask))

#train the model using Adam
def train(sess, model, generator, learning_rate, training_iters, batch_size, display_step):

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model.loss)
    #sess.run(tf.initialize_all_variables())
    sess.run(tf.global_variables_initializer())
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


