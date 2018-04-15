import tensorflow as tf
import numpy as np
import math


# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder,
              output_size,
              scope,
              n_layers=2,
              size=500,
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out


class NNDynamicsModel():
    def __init__(self,
                 env,
                 n_layers,
                 size,
                 activation,
                 output_activation,
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        self.env = env
        self.batch_size = batch_size
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.sess = sess
        self.mean_s, self.std_s, self.mean_deltas, self.std_deltas, self.mean_a, self.std_a = normalization

        self.s_dim = env.observation_space.shape[0]
        self.a_dim = env.action_space.shape[0]

        self.input_placeholder = tf.placeholder(tf.float32, [None, self.s_dim + self.a_dim])
        self.label_placeholder = tf.placeholder(tf.float32, [None, self.s_dim])
        self.out = build_mlp(self.input_placeholder, self.s_dim, "dynamics", n_layers, size,
                             activation, output_activation)
        self.loss = tf.reduce_mean(tf.squared_difference(self.out, self.label_placeholder))
        self.train_fn = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """
        """YOUR CODE HERE """
        s = np.concatenate([d["state"] for d in data])
        sp = np.concatenate([d["next_state"] for d in data])
        a = np.concatenate([d["action"] for d in data])
        N = s.shape[0]
        train_indicies = np.arange(N)

        # normalize
        s_norm = (s - self.mean_s) / (self.std_s + 1e-7)
        a_norm = (a - self.mean_a) / (self.std_a + 1e-7)
        s_a = np.concatenate((s_norm, a_norm), axis=1)
        deltas_norm = ((sp - s) - self.mean_deltas) / (self.std_deltas + 1e-7)

        # train
        for j in range(self.iterations):
            np.random.shuffle(train_indicies)
            for i in range(int(math.ceil(N / self.batch_size))):
                start_idx = i * self.batch_size % N
                idx = train_indicies[start_idx:start_idx + self.batch_size]
                batch_x = s_a[idx, :]
                batch_y = deltas_norm[idx, :]
                self.sess.run([self.train_fn],
                              feed_dict={self.input_placeholder: batch_x, self.label_placeholder: batch_y})

        # input_queue = tf.train.slice_input_producer([images, next_states], shuffle=False)
        # image_batch, label_batch = tf.train.batch(input_queue, batch_size=self.batch_size, num_threads=1,capacity=self.batch_size)
        # for i in range(self.iterations):
        #     coord = tf.train.Coordinator()
        #     threads = tf.train.start_queue_runners(self.sess, coord)
        #     try:
        #         while not coord.should_stop():
        #             image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
        #
        #     except tf.errors.OutOfRangeError:
        #         pass
        #     finally:
        #         coord.request_stop()
        #     coord.join(threads)

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        s = (states - self.mean_s) / (self.std_s + 1e-7)
        a = (actions - self.mean_a) / (self.std_a + 1e-7)
        input = np.concatenate((s, a), axis=1)
        feed_dict = {self.input_placeholder: input}
        out = self.sess.run(self.out, feed_dict=feed_dict)
        out = out * self.std_deltas + self.mean_deltas + states
        return out
