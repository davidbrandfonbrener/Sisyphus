import numpy as np
import itertools
from backend.simulation_tools import Simulator
from tasks import flip_flop
import matplotlib.pyplot as plt


# input: simulator object, params dict
# output: fixed points found by hahnloser strategy
#     assumes no biases as currently written
def hahnloser_fixed_point(sim, params, task):

    x, y, mask = task.build_train_batch(params)
    trial = x[0]
    target = y[0]

    outputs, states = sim.run_trial(trial)

    #TODO decide what input to use
    input_vec = np.matmul(np.absolute(sim.W_in), np.array([.1]*sim.N_in))
    input_mat = np.matmul(np.absolute(sim.W_in), np.transpose(trial))

    #set identity matrix of proper size
    identity = np.diag(np.ones(sim.N_rec))

    points = []
    for i, s in enumerate(states):
        # define active weight matrix
        Wp = np.matmul(np.absolute(sim.W_rec), sim.dale_rec)
        for index in range(sim.N_rec):
            if index < 0:
                Wp[:, index] = 0

        # check for fixed point
        I = np.matmul(np.linalg.inv(identity - Wp), input_mat[:, i])

        fixed = True
        for index in range(sim.N_rec):
            if s[0,index] < 0 and I[index] >= 0:
                fixed = False
            if s[0,index] >= 0 and I[index] < 0:
                fixed = False

        if fixed == True:
            print trial[i, :], s, I

    return points


if __name__ == '__main__':

    params = flip_flop.set_params(N_batch= 64, N_rec=30,
                           input_wait=5, stim_dur=5, quiet_gap=200, N_turns=4,
                           rec_noise=0, stim_noise=0.0,
                           dale_ratio=.8, tau=20, dt=10.)

    x,y,mask = flip_flop.build_train_batch(params)

    sim = Simulator(params, weights_path="../tasks/weights/flipflop.npz")

    for i in range(5):
        hahnloser_fixed_point(sim, params, flip_flop)