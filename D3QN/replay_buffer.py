import numpy as np


class ReplayBuffer:
    def __init__(self, mem_size, state_dims, action_dims):
        self.size = mem_size
        self.counter = 0
        self.states = np.zeros((mem_size, *state_dims), dtype=np.float32)
        self.states_new = np.zeros((mem_size, *state_dims), dtype=np.float32)
        self.actions = np.zeros((mem_size, action_dims), dtype=np.int64)
        self.rewards = np.zeros((mem_size, 1), dtype=np.float32)
        self.dones = np.zeros((mem_size, 1), dtype=bool)

    def add_to_mem(self, state, action, reward, state_new, done):
        index = self.counter % self.size
        self.states[index, :] = state
        self.actions[index, :] = action
        self.rewards[index] = reward
        self.states_new[index, :] = state_new
        self.dones[index] = done
        self.counter += 1

    def sample(self, batch_size):
        indices = np.random.choice(np.min([self.size, self.counter]), size=batch_size, replace=False)
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        states_new = self.states_new[indices]
        dones = self.dones[indices]

        return states, actions, rewards, states_new, dones
