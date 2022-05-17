import numpy as np
import cv2
import collections


class PLEWrappers:
    def __init__(self, ple_env, shape=(84, 84), repeat=4, stack_length=4):
        if not ple_env.use_screen:
            raise Exception('PLEWrappers requires screen observations. '
                            'Set use_scree=True when creating the PLE environment.')

        self.ple_env = ple_env
        self.shape = shape
        self.repeat = repeat
        self.stack_length = stack_length
        self.observation_space_shape = (self.stack_length, *self.shape)
        self.action_space_n = self.ple_env.action_space_n
        self.stack = collections.deque(maxlen=self.stack_length)
        self.timestep = 0

    def reset(self):
        self.timestep = 0
        self.stack.clear()
        _ = self.ple_env.reset()
        observation, _, _, _ = self.ple_env.step(0)  # Start the game.
        processed_obs = self._process_frame(observation)
        for _ in range(self.stack_length):
            self.stack.append(processed_obs)

        return np.array(self.stack).squeeze(axis=1)

    def step(self, action: int):
        self.timestep += 1
        total_reward = 0.0
        for i in range(self.repeat):
            observation, reward, done, info = self.ple_env.step(action)
            total_reward += reward
            if done:
                break
        processed_obs = self._process_frame(observation)
        self.stack.append(processed_obs)

        return np.array(self.stack).squeeze(axis=1), total_reward, done, info

    def set_display_screen(self, display: bool):
        self.ple_env.set_display_screen(display)

    def _process_frame(self, observation):
        processed_obs = cv2.resize(observation[0], self.shape, interpolation=cv2.INTER_AREA)
        processed_obs = processed_obs.reshape((1, self.shape[0], self.shape[1]))
        return processed_obs  # Channels first


def test():
    from Utils.FlappyBird_wrapper import FlappyBirdEnvironment
    import matplotlib.pyplot as plt
    ple_env = FlappyBirdEnvironment(use_screen=True)
    env = PLEWrappers(ple_env, repeat=4)

    obs_0 = env.reset()
    done = False
    '''while not done:
        obs, r, done, info = env.step(0)'''
    for _ in range(10):
        obs, r, done, info = env.step(0)

    for p in range(env.stack_length):
        idx = '1' + str(env.stack_length) + str(p + 1)
        plt.subplot(int(idx))
        plt.imshow(obs[p])
        plt.title('timestep t' + str(p + 1 - env.stack_length) if p < env.repeat - 1 else 'timestep t')


if __name__ == '__main__':
    test()
