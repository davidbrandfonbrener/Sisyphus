import numpy as np

class Simulator(object):
    def __init__(self, model, sess):
        N_in    = self.N_in       = model.N_in
        N_rec   = self.N_rec      = model.N_rec
        N_out   = self.N_out      = model.N_out

        # Physical parameters
        self.dt = model.dt
        self.tau = model.tau
        self.alpha = model.alpha
        self.dale_ratio = model.dale_ratio
        self.rec_noise  = model.rec_noise

        # Dale matrix
        self.dale_rec = model.dale_rec
        self.dale_out = model.dale_out

        # Connectivity
        self.connect_mat = model.connect_mat


        # Trained matrices
        self.W_in  = model.W_in.eval(session=sess)
        self.W_rec = model.W_rec.eval(session=sess)
        self.W_out = model.W_out.eval(session=sess)

        self.b_rec = model.b_rec.eval(session=sess)
        self.b_out = model.b_out.eval(session=sess)

        # Initial state
        self.init_state = model.init_state

    def rnn_step(self, state, rnn_in):
        if self.dale_ratio:
            new_state = (1-self.alpha) * state \
                        + self.alpha * (
                            np.dot(
                                np.maximum(state, 0),
                                np.dot(
                                    self.dale_rec,
                                    np.absolute(self.W_rec) * self.Connectivity))
                            + np.dot(
                                np.absolute(self.W_in),
                                rnn_in)
                            + self.b_rec)\
                        + np.sqrt(2.0 * self.alpha * self.rec_noise * self.rec_noise)\
                          * np.random.normal(loc=0.0, scale=1.0, size=state.shape())

            new_output = \
                        np.dot(
                            np.dot(
                                self.Dale_out,
                                np.absolute(self.W_out)),
                            np.maximum(new_state, 0))\
                        + self.b_out

            return new_output, new_state

    # apply the step to a full input vector
    def run_trial(self, trial_input):

        rnn_inputs = np.split(trial_input.shape[0], axis=0)
        state = self.init_state
        rnn_outputs = []
        rnn_states = []
        for rnn_input in rnn_inputs:
            output, state = self.rnn_step(rnn_input, state)
            rnn_outputs.append(output)
            rnn_states.append(state)
        return rnn_outputs, rnn_states
