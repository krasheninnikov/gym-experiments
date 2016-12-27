import gym
import tensorflow as tf
import itertools
import collections
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model

if "../" not in sys.path:
    sys.path.append("../")

class ImgBuf:
    """
    A ring buffer for holding multiple consecutive preprocessed frames.
    append(image) adds the frame to the buffer instead of the oldest one present.
    get() returns all frames from the buffer in the order they were added.
    empty() empties the ring buffer
    """
    def __init__(self, shape):
        self.shape = shape
        self.frame_buf = np.zeros(self.shape)
        self.index = 0

    def append(self, image):
        self.frame_buf[self.index, :, :] = image
        self.index += 1
        self.index = self.index % self.shape[0]

    def get(self):
        idx = (self.index + np.arange(self.shape[0])) % self.shape[0]
        return self.frame_buf[idx, :, :].reshape(1, self.shape[0], self.shape[1], self.shape[2])

    def empty(self):
        self.frame_buf = np.zeros(self.shape)
        self.index = 0


class ReplayBuf:
    """
    Synchronised ring buffers that hold (s_t, a_t, reward, s_t_plus_1).
    :param s_t and s_t_plus_1 have shape of (replay_size, n_ch, img_side, img_side)
    :param a_t is an int vector (one-hot encoding)
    :param reward is int
    """
    def __init__(self, shape, n_actions):
        self.shape = shape
        self.index = 0

        self.s_t = np.zeros(shape)
        self.s_t_plus_1 = np.zeros(shape)
        self.n_actions = n_actions
        self.action = np.zeros((shape[0], n_actions))
        self.reward = np.zeros(shape[0])

    def append(self, s_t, a_t, reward, s_t_plus_1):
        self.s_t[self.index, :, :, :] = s_t
        self.s_t_plus_1[self.index, :, :, :] = s_t_plus_1

        act = np.zeros(self.n_actions)
        act[a_t] = 1
        self.action[self.index] = act
        self.reward[self.index] = reward

        self.index += 1
        self.index = self.index % self.shape[0]


def preprocess(img):
    """
    :param img: a frame from the Atari simulator
    :return: the frame cropped and downsampled to 80x80 grayscale
    """
    img = img[60:220]  # crop
    img = img[::2, ::2, 1]  # downsample by factor of 2
    return img.astype(np.float)

def clip_reward(reward):
    """
    Clips the rewards to either -1, 0 or 1 for more stable training of the network
    :param reward: int
    :return: clipped reward, int
    """

    if reward > 0:
        reward = 1
    elif reward < 0:
        reward = -1
    return reward

def make_init_replay(env, replay, eps):
    """
    Makes the initial experience replay dataset by taking
    random actions and recording (s_t, a_t, reward, s_t_plus_1)
    :param env: OpenAI gym env
    :param replay: ReplayBuf object
    :param eps: probability that the tuple (s_t, a_t, reward, s_t_plus_1) will be added to replay

    :return: replay: a full replay buffer
    """
    max_episode_len = 30000
    frame_buf = ImgBuf((3, 80, 80))
    i = 0

    while i < replay.shape[0]:

        s_t_plus_1 = None
        frame_buf.append(preprocess(env.reset()))
        s_t = frame_buf.get()
        a_t = env.action_space.sample()

        for t in range(max_episode_len):

            if s_t_plus_1 is not None:
                s_t = s_t_plus_1
                a_t = a_t_plus_1

            obs, reward, done, _ = env.step(a_t)
            frame_buf.append(preprocess(obs))
            s_t_plus_1 = frame_buf.get()

            if np.random.rand() < eps:
                replay.append(s_t, a_t, clip_reward(reward), s_t_plus_1)
                i += 1

            a_t_plus_1 = env.action_space.sample()

            if done or t == (max_episode_len-1):
                frame_buf.empty()
                replay.append(s_t_plus_1, a_t_plus_1, -1, s_t_plus_1)
                break


def make_model(n_actions=9):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(3, 80, 80), border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(256, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))

    model.add(Dense(n_actions, activation='linear'))
    # Compile model
    model.compile(loss='mse', optimizer=Adam())
    print(model.summary())

    return model


def train(model, replay, discount_factor=.999, epochs=10):
    batchsize = 64
    s_t, a_t, reward, s_t_plus_1 = (replay.s_t, replay.action, replay.reward, replay.s_t_plus_1)

    s_t_Q = model.predict(s_t)
    s_t_plus_1_Q = model.predict(s_t_plus_1)

    TD_error = a_t * (reward + discount_factor * np.amax(s_t_plus_1_Q, axis=1)).reshape(replay.shape[0], 1) - a_t * s_t_Q

    target = s_t_Q + TD_error
    model.fit(s_t, target, nb_epoch=epochs, batch_size=batchsize)

    return model


def make_epsilon_greedy_policy(estimator, epsilon, n_actions):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    :param estimator: An estimator that returns q values for a given frame_buf
    :param epsilon: The probability to select a random action . float between 0 and 1.
    :param n_actions: Number of actions in the environment.

    :return a function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length n_actions.

    """

    def policy_fn(observation):
        A = np.ones(n_actions, dtype=float) * epsilon / n_actions
        q_values = estimator.predict(observation, batch_size=1, verbose=0)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def main():


    resume = 0
    replay_size = 26000
    add_to_rep_prob = .1
    num_episodes = 100
    max_episode_len = 100000
    epsilon = .01
    epsilon_decay = .99
    frame_buf = ImgBuf((3, 80, 80))
    discount_factor = .999

    env = gym.envs.make("VideoPinball-v0")
    env.reset()

    replay = ReplayBuf(shape=(replay_size, 3, 80, 80), n_actions=env.action_space.n)
    print("making init replay ...")
    make_init_replay(env=env, replay=replay, eps=0.2)
    print("init replay done")
    print(replay.s_t.shape)

    stats = collections.defaultdict(lambda: np.zeros(num_episodes))

    tf.python.control_flow_ops = tf
    with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
        with tf.device("/gpu:0"):

            if resume == 0:
                Q1 = make_model(n_actions=env.action_space.n)
            else:
                print("loading model ...")
                Q1 = load_model('Q1.h5')

            init = tf.global_variables_initializer()
            sess.run(init)

            Q1 = train(Q1, replay, discount_factor=discount_factor, epochs=10)
            Q1.save('Q1.h5')

            for i_episode in range(num_episodes):
                print("starting episode ", i_episode, " ...")

                # The policy we're following
                policy = make_epsilon_greedy_policy(
                    Q1, epsilon * epsilon_decay ** i_episode, env.action_space.n)

                s_t_plus_1 = None

                frame_buf.append(preprocess(env.reset()))
                s_t = frame_buf.get()
                action_probs = policy(s_t)
                a_t = np.random.choice(np.arange(len(action_probs)), p=action_probs)

                for t in range(max_episode_len):

                    if s_t_plus_1 is not None:
                        s_t = s_t_plus_1
                        a_t = a_t_plus_1

                    obs, reward, done, _ = env.step(a_t)
                    frame_buf.append(preprocess(obs))
                    s_t_plus_1 = frame_buf.get()

                    action_probs = policy(s_t_plus_1)
                    a_t_plus_1 = np.random.choice(np.arange(len(action_probs)), p=action_probs)

                    if np.random.rand() < add_to_rep_prob:
                        replay.append(s_t, a_t, clip_reward(reward), s_t_plus_1)

                    stats['rewards'][i_episode] += reward
                    if done or t == (max_episode_len-1):
                        frame_buf.empty()
                        Q1 = train(Q1, replay, discount_factor=discount_factor, epochs=3)
                        Q1.save('Q1.h5')
                        # sys.stdout.flush()
                        print("episode ", i_episode, " done in ", t, " steps, reward is ", stats['rewards'][i_episode])
                        stats['episode_lengths'][i_episode] = t
                        break


if __name__ == "__main__":
    main()
