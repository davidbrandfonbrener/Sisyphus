from tasks import flip_flop
from backend import networks
import tensorflow as tf
import numpy as np



n_steps = [20, 50, 100, 200, 400]
n_rec = [10, 50, 200, 500]
N_batch = [32]

for ns in n_steps:
    for nr in n_rec:
        for nb in N_batch:
            q = (ns - 5) / 10
            s = (ns - 5) / 5 - q

            params = flip_flop.set_params(N_rec=nr, N_turns=5, input_wait=5, quiet_gap=q, stim_dur=s,
                                          var_delay_length=0, stim_noise=0.1, rec_noise=0.1,
                                          N_batch=nb, dale_ratio=0.8, dt=10, tau=100)

            generator = flip_flop.generate_train_trials(params)
            model = networks.Model(params)

            configuration = tf.ConfigProto(inter_op_parallelism_threads=10, intra_op_parallelism_threads=10)
            sess = tf.Session(config=configuration)

            t = model.train(sess, generator, training_iters=10000, learning_rate=.01, weights_path=None)

            print "n_steps: %d, n_rec: %d, n_batch: %d, time: %f" %(ns, nr, nb, t)

            tf.reset_default_graph()

