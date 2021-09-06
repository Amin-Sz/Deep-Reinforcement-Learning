import numpy as np
import matplotlib.pyplot as plt
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gym
import pybullet_envs
import copy
import time


class Memory:
    def __init__(self, n_actions, batch_size):
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.next_values = []
        self.probs = []

    def add_to_mem(self, state, action, reward, done, value, next_value, prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.next_values.append(next_value)
        self.probs.append(prob)

    def sample_batches(self):
        memory_size = len(self.states)
        start_indices = np.arange(0, memory_size, self.batch_size)
        memory_indices = np.arange(memory_size, dtype=np.int64)
        np.random.shuffle(memory_indices)
        batches_indices = [memory_indices[idx:idx + self.batch_size] for idx in start_indices]

        return np.array(self.states), \
               np.array(self.actions).reshape((-1, self.n_actions)), \
               np.array(self.rewards).reshape((-1, 1)), \
               np.array(self.dones, dtype=bool).reshape((-1, 1)), \
               np.array(self.values).reshape((-1, 1)), \
               np.array(self.next_values).reshape((-1, 1)), \
               np.array(self.probs).reshape((-1, 1)), \
               batches_indices

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.next_values = []
        self.probs = []


class ValueNetwork(nn.Module):
    def __init__(self, state_dims, fc1_dims=512, fc2_dims=512, learning_rate=3e-4):
        super(ValueNetwork, self).__init__()
        self.state_dims = state_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr = learning_rate

        self.value_network = nn.Sequential(nn.Linear(self.state_dims, self.fc1_dims),
                                           nn.ReLU(),
                                           nn.Linear(self.fc1_dims, self.fc2_dims),
                                           nn.ReLU(),
                                           nn.Linear(self.fc2_dims, 1))

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, state):
        return self.value_network(state)


class ActorNetwork(nn.Module):
    def __init__(self, state_dims, n_actions, fc1_dims=512, fc2_dims=512, learning_rate=1e-4):
        super(ActorNetwork, self).__init__()
        self.state_dims = state_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr = learning_rate

        self.fc = nn.Sequential(nn.Linear(self.state_dims, self.fc1_dims),
                                nn.ReLU(),
                                nn.Linear(self.fc1_dims, self.fc2_dims),
                                nn.ReLU())
        self.mean = nn.Sequential(nn.Linear(self.fc2_dims, self.n_actions))
        self.sigma = nn.Sequential(nn.Linear(self.fc2_dims, self.n_actions))

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, state):
        output = self.fc(state)
        mean = self.mean(output)
        sigma = self.sigma(output)
        sigma = T.clamp(sigma, min=1e-6, max=1.0)
        distribution = Normal(mean, sigma)

        return distribution


class Agent:
    def __init__(self, state_dims, n_actions, action_max, batch_size, gamma, gae_lambda, n_epochs, horizon, eps):
        self.state_dims = state_dims
        self.n_actions = n_actions
        self.action_max = action_max
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.horizon = horizon
        self.eps = eps
        self.play_counter = 0
        self.update_counter = 0

        self.memory = Memory(n_actions=self.n_actions, batch_size=self.batch_size)

        self.value_network = ValueNetwork(state_dims=self.state_dims)
        self.actor_network = ActorNetwork(state_dims=self.state_dims, n_actions=self.n_actions)

    def get_action(self, state, training=True):
        state = T.tensor([state], dtype=T.float).to(self.actor_network.device)
        action_max = T.tensor(self.action_max, dtype=T.float).view(1, -1).to(self.actor_network.device)
        if training:
            with T.no_grad():
                dist = self.actor_network.forward(state)
                u = dist.sample()
                prob = dist.log_prob(u)
                prob = T.sum(prob, dim=1, keepdim=True).exp()

                action = action_max * T.tanh(u)
                action = action.cpu().detach().numpy()[0]
                u = u.cpu().detach().numpy()[0]
                prob = prob.cpu().detach().numpy()[0]

                value = self.value_network.forward(state)
                value = value.item()

            self.play_counter += 1
            return action, u, value, prob
        else:
            with T.no_grad():
                dist = self.actor_network.forward(state)
                u = dist.mean
                action = action_max * T.tanh(u)
                action = action.cpu().detach().numpy()[0]

            return action, None, None, None

    def update_network(self):
        for _ in range(self.n_epochs):
            self.update_counter += 1
            states, actions, rewards, dones, values, next_values, probs, batches_indices = self.memory.sample_batches()

            # Calculating advantages
            deltas = np.zeros(rewards.shape, dtype=np.float32)
            for idx in range(deltas.shape[0]):
                deltas[idx, 0] = rewards[idx, 0] + self.gamma * (1.0 - int(dones[idx, 0])) * next_values[idx, 0] - \
                                 values[idx, 0]

            advantages = copy.deepcopy(deltas)
            for idx in reversed(range(advantages.shape[0] - 1)):
                advantages[idx, 0] = advantages[idx, 0] + \
                                     (1.0 - int(dones[idx, 0])) * self.gamma * self.gae_lambda * advantages[idx + 1, 0]

            # Updating networks
            for batch in batches_indices:
                states_batch = T.tensor(states[batch], dtype=T.float).to(self.actor_network.device)
                adv_batch = T.tensor(advantages[batch], dtype=T.float).to(self.actor_network.device)
                values_batch = T.tensor(values[batch], dtype=T.float).to(self.actor_network.device)
                old_probs_batch = T.tensor(probs[batch], dtype=T.float).to(self.actor_network.device)
                actions_batch = T.tensor(actions[batch]).to(self.actor_network.device)

                # Updating the value network
                self.value_network.optimizer.zero_grad()
                target_value = adv_batch + values_batch
                value_loss = F.mse_loss(input=self.value_network.forward(states_batch),
                                        target=target_value)
                value_loss.backward()
                self.value_network.optimizer.step()

                # Updating the actor network
                distributions = self.actor_network.forward(states_batch)
                new_probs_batch = distributions.log_prob(actions_batch)
                new_probs_batch = T.sum(new_probs_batch, dim=1, keepdim=True).exp()
                r = new_probs_batch/old_probs_batch  # Probability ratio

                self.actor_network.optimizer.zero_grad()
                actor_loss = T.min(r * adv_batch, T.clamp(r, min=1 - self.eps, max=1 + self.eps) * adv_batch)
                actor_loss = -T.mean(actor_loss)
                actor_loss.backward()
                self.actor_network.optimizer.step()

        self.memory.clear_memory()

    def get_value(self, state):
        state = T.tensor([state], dtype=T.float).to(self.actor_network.device)
        value = self.value_network.forward(state)
        value = value.item()

        return value


def play_one_episode(agent, env):
    total_reward = 0
    counter = 0
    observation = env.reset()
    done = False
    while not done:
        a, u, value, prob = agent.get_action(observation)
        prev_observation = observation
        observation, reward, done, info = env.step(a)
        total_reward += reward
        counter += 1

        if counter >= env._max_episode_steps:
            agent.memory.add_to_mem(state=prev_observation, action=u, reward=reward, done=False,
                                    value=value, next_value=agent.get_value(observation), prob=prob)
        else:
            agent.memory.add_to_mem(state=prev_observation, action=u, reward=reward, done=bool(done),
                                    value=value, next_value=agent.get_value(observation), prob=prob)

        if agent.play_counter % agent.horizon == 0:
            agent.update_network()

    return agent, total_reward


def main(training=True):
    env = gym.make('InvertedPendulumSwingupBulletEnv-v0')
    dir_ = 'Section 5 - PPO/InvertedPendulumSwingupPyBulletEnv-v0'

    batch_size = 10
    gamma = 0.99
    gae_lambda = 0.95
    n_epochs = 4
    horizon = 50
    eps = 0.2
    agent = Agent(state_dims=env.observation_space.shape[0], n_actions=env.action_space.shape[0],
                  action_max=env.action_space.high, batch_size=batch_size,
                  gamma=gamma, gae_lambda=gae_lambda, n_epochs=n_epochs, horizon=horizon, eps=eps)

    if training:
        reward_set = []
        avg_score_set = []
        n_iteration = 1000
        for t in range(n_iteration):
            agent, total_reward = play_one_episode(agent, env)

            reward_set.append(total_reward)
            avg_score_set.append(np.mean(reward_set[-100:]))
            print('iteration #' + str(t + 1) + ' -----> ' +
                  'total reward:' + '%.2f' % total_reward +
                  ', average score:' + '%.2f' % np.mean(reward_set[-100:]) +
                  ', number of updates:' + str(agent.update_counter))

        # Plotting the learning curve
        axes = plt.axes()
        axes.set_ylim([np.min(reward_set) - 250, np.max(reward_set) + 100])
        plt.xlabel('Episode')
        plt.ylabel('Total reward')
        plt.plot(np.arange(1, len(reward_set) + 1), reward_set)
        plt.plot(np.arange(1, len(reward_set) + 1), avg_score_set)
        legend_2 = 'Running average of the last 100 total rewards (' + '%.2f' % np.mean(reward_set[-100:]) + ')'
        plt.legend(['Total reward', legend_2], loc=4)
        plt.show()
        plt.savefig(dir_ + '/Rewards_InvertedPendulumSwingupPyBulletEnv-v0')

        # Saving the networks
        T.save(agent.value_network.state_dict(), dir_ + '/value_network')
        T.save(agent.actor_network.state_dict(), dir_ + '/actor_network')

    else:
        # Loading the trained networks
        agent.value_network.load_state_dict(T.load(dir_ + '/value_network'))
        agent.actor_network.load_state_dict(T.load(dir_ + '/actor_network'))

        # Showing the video
        env.render(mode='human')
        for t in range(5):
            observation = env.reset()
            done = False
            total_reward = 0
            while not done:
                time.sleep(1.0 / 120)
                a, _, _, _ = agent.get_action(observation, training=False)
                observation, reward, done, info = env.step(a)
                total_reward = total_reward + reward
            print('video #' + str(t + 1) + ' -----> total reward:' + '%.2f' % total_reward)
        env.close()


if __name__ == '__main__':
    main(training=False)
