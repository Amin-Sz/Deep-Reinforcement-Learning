import numpy as np
import matplotlib.pyplot as plt
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import pybullet_envs
import time


class ReplayBuffer:
    def __init__(self, mem_size, state_dim, action_dim):
        self.size = mem_size
        self.counter = 0
        self.states = np.zeros((mem_size, state_dim))
        self.states_new = np.zeros((mem_size, state_dim))
        self.actions = np.zeros((mem_size, action_dim))
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
    def __init__(self, state_dim, action_dim, fc1_dims=512, fc2_dims=512, learning_rate=3e-3):
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
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        s_a = T.cat((state, action), dim=1)
        Q = self.fc1(s_a)
        Q = F.relu(Q)
        Q = self.fc2(Q)
        Q = F.relu(Q)
        Q = self.fc3(Q)
        return Q


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_max, fc1_dims=512, fc2_dims=512, learning_rate=1e-3):
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
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        mu = self.fc1(state)
        mu = F.relu(mu)
        mu = self.fc2(mu)
        mu = F.relu(mu)
        mu = self.fc3(mu)
        mu = self.action_max * T.tanh(mu)
        return mu


class Agent:
    def __init__(self, env, warm_up=1000, mem_size=1000000, batch_size=100, update_interval=2, gamma=0.99, tau=5e-3,
                 action_noise=0.1):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_max = env.action_space.high[0]
        self.action_min = env.action_space.low[0]
        self.warm_up = warm_up
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.gamma = gamma
        self.tau = tau
        self.action_noise = action_noise
        self.critic_update_counter = 0
        self.play_counter = 0

        self.replay_buffer = ReplayBuffer(mem_size=self.mem_size, state_dim=self.state_dim, action_dim=self.action_dim)
        self.critic_1 = CriticNetwork(state_dim=self.state_dim, action_dim=self.action_dim)
        self.critic_2 = CriticNetwork(state_dim=self.state_dim, action_dim=self.action_dim)
        self.actor = ActorNetwork(state_dim=self.state_dim, action_dim=self.action_dim, action_max=self.action_max)

        self.target_critic_1 = CriticNetwork(state_dim=self.state_dim, action_dim=self.action_dim)
        self.target_critic_2 = CriticNetwork(state_dim=self.state_dim, action_dim=self.action_dim)
        self.target_actor = ActorNetwork(state_dim=self.state_dim, action_dim=self.action_dim,
                                         action_max=self.action_max)
        self.update_target_networks(tau=1)

    def get_action(self, state, training=True):
        state = T.tensor([state], dtype=T.float).to(self.actor.device)
        if self.play_counter <= self.warm_up and training:
            action = self.env.action_space.sample()
        else:
            action = self.actor.forward(state)
            if training:
                action += self.action_noise * T.randn(action.size()).to(self.actor.device)
            action = T.clamp(action, min=self.action_min, max=self.action_max)
            action = action.cpu().detach().numpy()[0]

        self.play_counter += 1
        return action

    def update(self):
        states, actions, rewards, new_states, dones = self.replay_buffer.sample(batch_size=self.batch_size)
        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        new_states = T.tensor(new_states, dtype=T.float).to(self.actor.device)
        dones = T.tensor(dones).to(self.actor.device)

        target_noise = T.clamp(0.2*T.randn(actions.size()), min=-0.5, max=0.5)
        target_actions = self.target_actor.forward(new_states) + target_noise.to(self.actor.device)
        target_actions = T.clamp(target_actions, min=self.action_min, max=self.action_max)
        target_Q1 = self.target_critic_1.forward(new_states, target_actions)
        target_Q2 = self.target_critic_2.forward(new_states, target_actions)
        target_critic_value = T.min(target_Q1, target_Q2)
        target_critic_value[dones] = 0.0
        target = rewards + self.gamma*target_critic_value

        critic_1_value = self.critic_1.forward(states, actions)
        critic_2_value = self.critic_2.forward(states, actions)

        self.critic_1.optimizer.zero_grad()
        critic_1_loss = F.mse_loss(target, critic_1_value)

        self.critic_2.optimizer.zero_grad()
        critic_2_loss = F.mse_loss(target, critic_2_value)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.critic_update_counter += 1

        if self.critic_update_counter % self.update_interval == 0:
            self.actor.optimizer.zero_grad()
            actor_loss = self.critic_1.forward(states, self.actor.forward(states))
            actor_loss = -T.mean(actor_loss)
            actor_loss.backward()
            self.actor.optimizer.step()

            self.update_target_networks()

    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        critic_1_params = dict(self.critic_1.named_parameters())
        target_critic_1_params = dict(self.target_critic_1.named_parameters())
        for name in target_critic_1_params:
            target_critic_1_params[name] = tau*critic_1_params[name].clone() + \
                                           (1 - tau)*target_critic_1_params[name].clone()
        self.target_critic_1.load_state_dict(target_critic_1_params)

        critic_2_params = dict(self.critic_2.named_parameters())
        target_critic_2_params = dict(self.target_critic_2.named_parameters())
        for name in target_critic_2_params:
            target_critic_2_params[name] = tau*critic_2_params[name].clone() + \
                                           (1 - tau)*target_critic_2_params[name].clone()
        self.target_critic_2.load_state_dict(target_critic_2_params)

        actor_params = dict(self.actor.named_parameters())
        target_actor_params = dict(self.target_actor.named_parameters())
        for name in target_actor_params:
            target_actor_params[name] = tau*actor_params[name].clone() + \
                                        (1 - tau)*target_actor_params[name].clone()
        self.target_actor.load_state_dict(target_actor_params)


def play_one_game(agent, env):
    total_reward = 0
    observation = env.reset()
    done = False
    while not done:
        a = agent.get_action(observation)
        prev_observation = observation
        observation, reward, done, info = env.step(a)
        total_reward += reward

        agent.replay_buffer.add_to_mem(prev_observation, a, reward, observation, done)

        if agent.replay_buffer.counter > agent.batch_size:
            agent.update()

    return agent, total_reward


def main(training=True):
    env = gym.make('InvertedDoublePendulumBulletEnv-v0')
    gamma = 0.99
    tau = 5e-3
    warm_up = 1000
    action_noise = 0.1
    agent = Agent(env=env, warm_up=warm_up, gamma=gamma, tau=tau,
                  action_noise=action_noise)  # change values of hyperparameters if you want

    if training:
        reward_set = []
        avg_score_set = []
        n_iteration = 2000
        for t in range(n_iteration):
            agent, total_reward = play_one_game(agent, env)

            reward_set.append(total_reward)
            avg_score_set.append(np.mean(reward_set[-100:]))
            print('iteration #' + str(t + 1) + ' -----> ' +
                  'total reward:' + '%.2f' % total_reward +
                  ', average score:' + '%.2f' % np.mean(reward_set[-100:]))

        # Plotting the results
        axes = plt.axes()
        axes.set_ylim([np.min(reward_set) - 2500, np.max(reward_set) + 500])
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(np.arange(1, n_iteration + 1), reward_set)
        plt.plot(np.arange(1, n_iteration + 1), avg_score_set)
        legend_2 = 'Running average of the last 100 episodes (' + '%.2f' % np.mean(reward_set[-100:]) + ')'
        plt.legend(['Reward', legend_2], loc=4)
        plt.show()
        plt.savefig('Section 6 - TD3/InvertedDoublePendulumPyBulletEnv-v0/Rewards_InvertedDoublePendulumPyBulletEnv')

        # Saving the networks
        T.save(agent.critic_1.state_dict(), 'Section 6 - TD3/InvertedDoublePendulumPyBulletEnv-v0/critic_1')
        T.save(agent.critic_2.state_dict(), 'Section 6 - TD3/InvertedDoublePendulumPyBulletEnv-v0/critic_2')
        T.save(agent.actor.state_dict(), 'Section 6 - TD3/InvertedDoublePendulumPyBulletEnv-v0/actor')
        T.save(agent.target_critic_1.state_dict(), 'Section 6 - TD3/InvertedDoublePendulumPyBulletEnv-v0/target_critic_1')
        T.save(agent.target_critic_2.state_dict(), 'Section 6 - TD3/InvertedDoublePendulumPyBulletEnv-v0/target_critic_2')
        T.save(agent.target_actor.state_dict(), 'Section 6 - TD3/InvertedDoublePendulumPyBulletEnv-v0/target_actor')

    else:
        # Loading the trained networks
        agent.critic_1.load_state_dict(T.load('Section 6 - TD3/InvertedDoublePendulumPyBulletEnv-v0/critic_1'))
        agent.critic_2.load_state_dict(T.load('Section 6 - TD3/InvertedDoublePendulumPyBulletEnv-v0/critic_2'))
        agent.actor.load_state_dict(T.load('Section 6 - TD3/InvertedDoublePendulumPyBulletEnv-v0/actor'))
        agent.target_critic_1.load_state_dict(T.load('Section 6 - TD3/InvertedDoublePendulumPyBulletEnv-v0/'
                                                     'target_critic_1'))
        agent.target_critic_2.load_state_dict(T.load('Section 6 - TD3/InvertedDoublePendulumPyBulletEnv-v0/'
                                                     'target_critic_2'))
        agent.target_actor.load_state_dict(T.load('Section 6 - TD3/InvertedDoublePendulumPyBulletEnv-v0/target_actor'))

        # Showing the video
        env.render(mode='human')
        for t in range(5):
            observation = env.reset()
            done = False
            total_reward = 0
            while not done:
                time.sleep(1.0 / 120)
                a = agent.get_action(observation, training=False)
                observation, reward, done, info = env.step(a)
                total_reward = total_reward + reward
            print('video #' + str(t + 1) + ' -----> total reward:' + '%.2f' % total_reward)
        env.close()


if __name__ == '__main__':
    main(training=False)

