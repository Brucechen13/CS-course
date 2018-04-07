# behavior clone

import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt


class behavior_clone:
    def __init__(self, input_dim=11 ,n_classs = 3, layers = 3, size = 10, learning_rate=0.001):
        self.n_classs = n_classs
        self.layers = layers
        self.size = size
        self.learning_rate = learning_rate

        self.X = tf.placeholder(tf.float32, shape=[None, input_dim], name = 'X')
        self.y = tf.placeholder(tf.float32, shape=[None, n_classs], name = 'y')

        layer = self.X
        for i in range(self.layers):
            layer = tf.layers.dense(layer, size, activation=tf.nn.relu)
        self.target = tf.layers.dense(layer, self.n_classs)

        self.loss = tf.reduce_mean(tf.squared_difference(self.target, self.y))

        self.update_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
    
    def train(self, session, train_data, train_label, iter_num=10):
        input_queue = tf.train.slice_input_producer([train_data, train_label], shuffle=False)
        train_batch, label_batch = tf.train.batch(input_queue, batch_size=10, num_threads=1, capacity=64)
        losses = []
        for i in range(iter_num):
            # train_batch_v, label_batch_v = session.run([train_batch, label_batch])
            #train_batch_v, label_batch_v = tf.train.shuffle_batch([train_data, train_label], batch_size=64, capacity=50000,min_after_dequeue=10000)
            for i in range(100):
                indic = np.random.choice(train_data.shape[0], 64)
                train_batch_v = train_data[indic, :]
                label_batch_v = train_label[indic, :]
                feed_dict = {"X:0":train_batch_v, "y:0":label_batch_v}
                loss, _ =  session.run([self.loss, self.update_op], feed_dict=feed_dict)
            print("loss: " + str(loss))
            losses.append(loss)
        # plt.subplot() # 第二整行
        # plt.plot(range(iter_num),losses) 
        # plt.show()
    
    def predict(self, session, test_data):
        feed_dict = {"X:0":test_data}
        test_pred = session.run([self.target], feed_dict=feed_dict)
        return test_pred


def main():
    with open('./homework/hw1/data/train.pkl', 'rb') as f:
        model = pickle.load(f)
        observations = model['observations']
        actions = model['actions']
        trian_data = observations
        train_label = actions.reshape(-1, actions.shape[-1])
        print(trian_data.shape)
        print(train_label.shape)
        bc = behavior_clone()
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            bc.train(session, trian_data, train_label)

            import load_policy
            print('loading and building expert policy')
            import os

            PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
            policy_path = os.path.join(PROJECT_ROOT, "experts/"+"Hopper-v1.pkl")
            policy_fn = load_policy.load_policy(policy_path)

            returns = []
            import gym
            env = gym.make("Hopper-v1")
            for i in range(20):
                print('iter', i)
                observations = []
                actions = []
                session.run(tf.global_variables_initializer())
                bc.train(session, trian_data, train_label)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    action = policy_fn(obs[None,:])
                    actions.append(np.array(action).reshape(-1,3))
                    action = bc.predict(session, np.array(obs).reshape(-1,11))
                    observations.append(obs)
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    if True:
                        env.render()
                    if steps % 100 == 0: print("%i/%i"%(steps, 1000))
                    if steps >= 1000:
                        break
                returns.append(totalr)
                trian_data = np.concatenate((trian_data,np.array(observations)), axis=0)
                new_label = np.array(actions).reshape(-1, 3)
                train_label = np.concatenate((train_label,new_label), axis=0)
            
            print('returns', returns)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))




if __name__ == '__main__':
    main()
        
