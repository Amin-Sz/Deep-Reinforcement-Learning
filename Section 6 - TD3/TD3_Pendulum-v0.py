import numpy as np
import matplotlib.pyplot as plt
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym


class ReplayBuffer:
    def __init__(self, mem_size, state_size, action_size):
        self.size = mem_size
        self.counter = 0
        self.states = np.zeros((mem_size, state_size))
        self.states_new = np.zeros((mem_size, state_size))
        self.actions = np.zeros((mem_size, action_size))
        self.rewards = np.zeros((mem_size, 1))
        self.dones = np.zeros((mem_size, 1), dtype=bool)

    def add_to_mem(self, state, action, reward, state_new, done):
        index = self.counter % self.size
        self.states[index, :] = state
        self.actions[index, :] = action
        self.rewards[index] = reward
        self.states_new[index, :] = state_new
        self.dones[index] = done
        self.counter = self.counter + 1

    def sample(self, batch_size):
        indices = np.random.choice(np.min([self.size, self.counter]), size=batch_size, replace=False)
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        states_new = self.states_new[indices]
        dones = self.dones[indices]
        return states, actions, rewards, states_new, dones


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dims=400, fc2_dims=300, learning_rate=1e-3):
        super(CriticNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.learning_rate = learning_rate

        self.fc1 = nn.Linear(self.state_dim + self.action_dim, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.device = T.device('cude:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state = T.tensor([state], dtype=T.float).to(self.device)
        action = T.tensor([action], dtype=T.float).to(self.device)
        s_a = T.cat((state, action), dim=1)

        Q = self.fc1(s_a)
        Q = F.relu(Q)
        Q = self.fc2(Q)
        Q = F.relu(Q)
        Q = self.fc3(Q)
        return Q


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_max, fc1_dims=400, fc2_dims=300, learning_rate=1e-3):
        super(ActorNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_max = action_max
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.learning_rate = learning_rate

        self.fc1 = nn.Linear(self.state_dim, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.device = T.device('cude:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = T.tensor([state], dtype=T.float).to(self.device)

        mu = self.fc1(state)
        mu = F.relu(mu)
        mu = self.fc2(mu)
        mu = F.relu(mu)
        mu = self.fc3(mu)
        mu = self.action_max * F.tanh(mu)
        return mu




















