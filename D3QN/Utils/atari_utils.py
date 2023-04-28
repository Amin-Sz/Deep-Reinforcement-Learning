import numpy as np
import collections
import cv2
import gym

""" This code is taken from the following repository and slightly modified::
    https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/lib/wrappers.py
"""


class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env=None, repeat=4, no_ops=0):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.no_ops = no_ops
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros_like((2, self.shape))

    def step(self, action):
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            idx = i % 2
            self.frame_buffer[idx] = obs
            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, t_reward, done, info

    def reset(self):
        obs = self.env.reset()
        no_ops = np.random.choice(range(self.no_ops + 1))

        for _ in range(no_ops):
            obs, _, done, _ = self.env.step(0)  # Do nothing
            if done:
                self.env.reset()

        self.frame_buffer = np.zeros_like((2, self.shape))
        self.frame_buffer[0] = obs
        return obs


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0, high=1.0,
                                                shape=self.shape, dtype=np.float32)

    def observation(self, obs):
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:],
                                    interpolation=cv2.INTER_AREA)

        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0
        return new_obs


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                             env.observation_space.low.repeat(n_steps, axis=0),
                             env.observation_space.high.repeat(n_steps, axis=0),
                             dtype=np.float32)
        self.stack = collections.deque(maxlen=n_steps)

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, observation):
        self.stack.append(observation)
        obs = np.array(self.stack).reshape(self.observation_space.low.shape)

        return obs


def make_env(env_name, shape=(84, 84, 1), skip=4, no_ops=0, render_mode=None):
    if not render_mode:
        env = gym.make(env_name)
    else:
        env = gym.make(env_name, render_mode=render_mode)

    env = RepeatActionAndMaxFrame(env, skip, no_ops)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, skip)

    return env
