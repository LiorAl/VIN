import tensorflow as tf
from baselines.common import tf_util
from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.tf_util import adjust_shape
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.models import get_network_builder

import gym
import numpy as np
from utils_TF import conv2d_flipkernel


def VIN_network(**args):

    def VI_Block(X, S1, S2):
        k    = args.k    # Number of value iterations performed
        ch_i = args.ch_i # Channels in input layer
        ch_h = args.ch_h # Channels in initial hidden layer
        ch_q = args.ch_q # Channels in q layer (~actions)
        state_batch_size = args.statebatchsize # k+1 state inputs for each channel

        bias  = tf.Variable(np.random.randn(1, 1, 1, ch_h)    * 0.01, dtype=tf.float32)
        # weights from inputs to q layer (~reward in Bellman equation)
        w0    = tf.Variable(np.random.randn(3, 3, ch_i, ch_h) * 0.01, dtype=tf.float32)
        w1    = tf.Variable(np.random.randn(1, 1, ch_h, 1)    * 0.01, dtype=tf.float32)
        w     = tf.Variable(np.random.randn(3, 3, 1, ch_q)    * 0.01, dtype=tf.float32)
        # feedback weights from v layer into q layer (~transition probabilities in Bellman equation)
        w_fb  = tf.Variable(np.random.randn(3, 3, 1, ch_q)    * 0.01, dtype=tf.float32)
        w_o   = tf.Variable(np.random.randn(ch_q, 8)          * 0.01, dtype=tf.float32)

        # initial conv layer over image+reward prior
        h = conv2d_flipkernel(X, w0, name="h0") + bias

        r = conv2d_flipkernel(h, w1, name="r")
        q = conv2d_flipkernel(r, w, name="q")
        v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

        for i in range(0, k-1):
            rv = tf.concat([r, v], 3)
            wwfb = tf.concat([w, w_fb], 2)
            q = conv2d_flipkernel(rv, wwfb, name="q")
            v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

        # do one last convolution
        q = conv2d_flipkernel(tf.concat([r, v], 3),
                              tf.concat([w, w_fb], 2), name="q")

        # CHANGE TO THEANO ORDERING
        # Since we are selecting over channels, it becomes easier to work with
        # the tensor when it is in NCHW format vs NHWC
        q = tf.transpose(q, perm=[0, 3, 1, 2])

        # Select the conv-net channels at the state position (S1,S2).
        # This intuitively corresponds to each channel representing an action, and the convnet the Q function.
        # The tricky thing is we want to select the same (S1,S2) position *for each* channel and for each sample
        # TODO: performance can be improved here by substituting expensive
        #       transpose calls with better indexing for gather_nd
        bs = tf.shape(q)[0]
        rprn = tf.reshape(tf.tile(tf.reshape(tf.range(bs), [-1, 1]), [1, state_batch_size]), [-1])
        ins1 = tf.cast(tf.reshape(S1, [-1]), tf.int32)
        ins2 = tf.cast(tf.reshape(S2, [-1]), tf.int32)
        idx_in = tf.transpose(tf.stack([ins1, ins2, rprn]), [1, 0])
        q_out = tf.gather_nd(tf.transpose(q, [2, 3, 0, 1]), idx_in, name="q_out")

        # add logits
        logits = tf.matmul(q_out, w_o)
        # softmax output weights
        output = tf.nn.softmax(logits, name="output")
        # return logits, output
        return output

    return VI_Block



def build_VIN_policy(env, value_network=None,  normalize_observations=False, estimate_q=False, **policy_kwargs):

    policy_network = VIN_network()(**policy_kwargs)

    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None):
        ob_space = env.observation_space

        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)

        extra_tensors = {}

        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X

        encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            policy_latent = policy_network(encoded_x)
            if isinstance(policy_latent, tuple):
                policy_latent, recurrent_tensors = policy_latent

                if recurrent_tensors is not None:
                    # recurrent architecture, need a few more steps
                    nenv = nbatch // nsteps
                    assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                    policy_latent, recurrent_tensors = policy_network(encoded_x, nenv)
                    extra_tensors.update(recurrent_tensors)


        _v_net = value_network

        if _v_net is None or _v_net == 'shared':
            vf_latent = policy_latent
        else:
            if _v_net == 'copy':
                _v_net = policy_network
            else:
                assert callable(_v_net)

            with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
                # TODO recurrent architectures are not supported with value_network=copy yet
                vf_latent = _v_net(encoded_x)

        policy = PolicyWithValue(
            env=env,
            observations=X,
            latent=policy_latent,
            vf_latent=vf_latent,
            sess=sess,
            estimate_q=estimate_q,
            **extra_tensors
        )
        return policy

    return policy_fn


def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms

