import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE


class FlappyBirdEnvironment:
    def __init__(self, height=512, width=288, pipe_gap=100, fps=30, display_screen=False):
        self.timestep = 0
        self.game = FlappyBird(width=width, height=height, pipe_gap=pipe_gap)
        self.p = PLE(self.game, fps=fps, display_screen=display_screen)
        self.p.init()

        self.observation_space_shape = self._get_observation().shape
        self.action_set = self.p.getActionSet()
        self.action_space_n = len(self.action_set)

    def reset(self):
        self.timestep = 0
        self.p.reset_game()
        observation = self._get_observation()
        return observation

    def step(self, action: int):
        reward = self.p.act(self.action_set[action])
        observation = self._get_observation()
        done = self.p.game_over()
        self.timestep += 1
        return observation, reward, done, None

    def _get_observation(self):
        return np.array(list(self.p.getGameState().values()))

    def set_display_screen(self, display: bool):
        self.p.display_screen = display
