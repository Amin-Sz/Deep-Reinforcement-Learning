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
    def __init__(self, mem_size, state_dims, action_dims, history_length):
        self.size = mem_size
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.history_length = history_length
        self.counter = 0
        self.states = np.zeros((mem_size, state_dims))
        self.states_new = np.zeros((mem_size, state_dims))
        self.actions = np.zeros((mem_size, action_dims))
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
        indices = np.random.choice(range(self.history_length, np.min([self.size, self.counter])),
                                   size=batch_size, replace=False)

        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        states_new = self.states_new[indices]
        dones = self.dones[indices]

        if self.history_length == 0:
            hist_state = np.zeros((batch_size, 1, self.state_dims))
            hist_action = np.zeros((batch_size, 1, self.action_dims))
            hist_next_state = np.zeros((batch_size, 1, self.state_dims))
            hist_next_action = np.zeros((batch_size, 1, self.action_dims))
            hist_length = np.ones((batch_size, 1), dtype=np.int64)
        else:
            hist_state = np.zeros((batch_size, self.history_length, self.state_dims))
            hist_action = np.zeros((batch_size, self.history_length, self.action_dims))
            hist_next_state = np.zeros((batch_size, self.history_length, self.state_dims))
            hist_next_action = np.zeros((batch_size, self.history_length, self.action_dims))
            hist_length = self.history_length * np.ones((batch_size, 1), dtype=np.int64)

            for i, index in enumerate(indices):
                start_index = index - self.history_length
                if start_index < 0:
                    start_index = 0
                if True in self.dones[start_index:index]:
                    start_index = start_index + np.where(self.dones[start_index:index] == True)[0][-1] + 1

                length = index - start_index
                hist_state[i, 0:length] = self.states[start_index:index, :]
                hist_action[i, 0:length] = self.actions[start_index:index, :]
                hist_next_state[i, 0:length] = self.states_new[start_index:index, :]
                hist_next_action[i, 0:length] = self.actions[start_index + 1:index + 1]
                if length <= 0:
                    length = 1
                hist_length[i] = length

        return states, actions, rewards, states_new, dones, \
               hist_length, hist_state, hist_action, hist_next_state, hist_next_action


class CriticNetwork(nn.Module):
    def __init__(self, state_dims, action_dims, fc1_dims=128, lstm_hidden_size=128, fc2_dims=128,
                 fc3_dims=128, fc4_dims=128, learning_rate=1e-3):
        super(CriticNetwork, self).__init__()
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.fc1_dims = fc1_dims
        self.hidden_size = lstm_hidden_size
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.lr = learning_rate

        self.fc1 = nn.Linear(self.state_dims + self.action_dims, self.fc1_dims)
        self.lstm = nn.LSTM(self.fc1_dims, self.hidden_size, batch_first=True)
        self.fc2 = nn.Linear(self.state_dims + self.action_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.hidden_size + self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action, hist_observation, hist_action, hist_length):
        history = T.cat((hist_observation, hist_action), dim=-1)  # (batch, time steps, state_dims + action_dims)
        memory_out = F.relu(self.fc1(history))  # (batch, time steps, fc1_dims)
        memory_out, (hn, cn) = self.lstm(memory_out)  # (batch, time steps, hidden_size)

        last_time_steps = np.repeat(hist_length, repeats=self.hidden_size, axis=1).reshape((-1, 1, self.hidden_size))
        last_time_steps = T.tensor(last_time_steps, dtype=T.int64).to(self.device)
        memory_out = T.gather(memory_out, dim=1, index=last_time_steps - 1)  # take the last hidden state
        memory_out = memory_out.squeeze(dim=1)  # (batch, hidden_size)

        x = T.cat((state, action), dim=-1)
        current_feature_out = F.relu(self.fc2(x))

        out = T.cat((memory_out, current_feature_out), dim=-1)
        out = F.relu(self.fc3(out))
        q = self.fc4(out)

        return q


class ActorNetwork(nn.Module):
    def __init__(self, state_dims, action_dims, action_max, fc1_dims=128, lstm_hidden_size=128,
                 fc2_dims=128, fc3_dims=128, fc4_dims=128, learning_rate=1e-3):
        super(ActorNetwork, self).__init__()
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.fc1_dims = fc1_dims
        self.hidden_size = lstm_hidden_size
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.lr = learning_rate

        self.fc1 = nn.Linear(self.state_dims + self.action_dims, self.fc1_dims)
        self.lstm = nn.LSTM(self.fc1_dims, self.hidden_size, batch_first=True)
        self.fc2 = nn.Linear(self.state_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.hidden_size + self.fc2_dims, fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.action_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.action_max = T.tensor(action_max, dtype=T.float).to(self.device)

    def forward(self, state, hist_observation, hist_action, hist_length):
        history = T.cat((hist_observation, hist_action), dim=-1)  # (batch, time steps, state_dims + action_dims)
        memory_out = F.relu(self.fc1(history))  # (batch, time steps, fc1_dims)
        memory_out, (hn, cn) = self.lstm(memory_out)  # (batch, time steps, hidden_size)

        last_time_steps = np.repeat(hist_length, repeats=self.hidden_size, axis=1).reshape((-1, 1, self.hidden_size))
        last_time_steps = T.tensor(last_time_steps, dtype=T.int64).to(self.device)
        memory_out = T.gather(memory_out, dim=1, index=last_time_steps - 1)  # take the last hidden state
        memory_out = memory_out.squeeze(dim=1)  # (batch, hidden_size)

        current_feature_out = F.relu(self.fc2(state))

        out = T.cat((memory_out, current_feature_out), dim=-1)
        out = F.relu(self.fc3(out))
        mu = self.action_max * T.tanh(self.fc4(out))

        return mu


class Agent:
    def __init__(self, env, state_dims, action_dims, action_max, action_min, history_length, gamma=0.99,
                 batch_size=100, memory_size=1000000, warm_up=1000, tau=0.005, actor_noise=0.1, update_interval=2):
        self.env = env
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.action_max = action_max
        self.action_min = action_min
        self.history_length = history_length
        self.gamma = gamma
        self.batch_size = batch_size
        self.mem_size = memory_size
        self.warm_up = warm_up
        self.tau = tau
        self.actor_noise = actor_noise
        self.update_interval = update_interval
        self.critic_update_counter = 0
        self.play_counter = 0

        self.replay_buffer = ReplayBuffer(mem_size=self.mem_size, state_dims=self.state_dims,
                                          action_dims=self.action_dims, history_length=self.history_length)

        self.critic_1 = CriticNetwork(state_dims=self.state_dims, action_dims=self.action_dims)
        self.critic_2 = CriticNetwork(state_dims=self.state_dims, action_dims=self.action_dims)
        self.actor = ActorNetwork(state_dims=self.state_dims, action_dims=self.action_dims, action_max=self.action_max)
        self.target_critic_1 = CriticNetwork(state_dims=self.state_dims, action_dims=self.action_dims)
        self.target_critic_2 = CriticNetwork(state_dims=self.state_dims, action_dims=self.action_dims)
        self.target_actor = ActorNetwork(state_dims=self.state_dims, action_dims=self.action_dims,
                                         action_max=self.action_max)

        self.update_target_networks(tau=1)

    def get_action(self, state, hist_observation, hist_action, hist_length, training=True):
        state = T.tensor([state], dtype=T.float).to(self.actor.device)
        hist_observation = T.tensor([hist_observation], dtype=T.float).to(self.actor.device)
        hist_action = T.tensor([hist_action], dtype=T.float).to(self.actor.device)
        if hist_length == 0:
            hist_length = 1
        hist_length = np.atleast_2d(hist_length).reshape(-1, 1)

        if self.play_counter <= self.warm_up and training:
            action = self.env.action_space.sample()
        else:
            action = self.actor.forward(state=state, hist_observation=hist_observation, hist_action=hist_action,
                                        hist_length=hist_length)
            if training:
                action += self.actor_noise * T.randn(action.size()).to(self.actor.device)
            action = T.clamp(action, min=self.action_min[0], max=self.action_max[0])
            action = action.cpu().detach().numpy()[0]

        self.play_counter += 1
        return action

    def update(self):
        states, actions, rewards, new_states, dones, \
        hist_lengths, hist_state, hist_action, hist_next_state, \
        hist_next_action = self.replay_buffer.sample(batch_size=self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        new_states = T.tensor(new_states, dtype=T.float).to(self.actor.device)
        dones = T.tensor(dones).to(self.actor.device)
        hist_state = T.tensor(hist_state, dtype=T.float).to(self.actor.device)
        hist_action = T.tensor(hist_action, dtype=T.float).to(self.actor.device)
        hist_next_state = T.tensor(hist_next_state, dtype=T.float).to(self.actor.device)
        hist_next_action = T.tensor(hist_next_action, dtype=T.float).to(self.actor.device)

        target_noise = T.clamp(0.2*T.randn(actions.size()), min=-0.5, max=0.5)
        target_actions = self.target_actor.forward(state=new_states, hist_observation=hist_next_state,
                                                   hist_action=hist_next_action, hist_length=hist_lengths) + \
                         target_noise.to(self.actor.device)
        target_actions = T.clamp(target_actions, min=self.action_min[0], max=self.action_max[0])
        target_Q1 = self.target_critic_1.forward(state=new_states, action=target_actions,
                                                 hist_observation=hist_next_state, hist_action=hist_next_action,
                                                 hist_length=hist_lengths)
        target_Q2 = self.target_critic_2.forward(state=new_states, action=target_actions,
                                                 hist_observation=hist_next_state, hist_action=hist_next_action,
                                                 hist_length=hist_lengths)
        target_critic_value = T.min(target_Q1, target_Q2)
        target_critic_value[dones] = 0.0
        target = rewards + self.gamma*target_critic_value

        critic_1_value = self.critic_1.forward(state=states, action=actions, hist_observation=hist_state,
                                               hist_action=hist_action, hist_length=hist_lengths)
        critic_2_value = self.critic_2.forward(state=states, action=actions, hist_observation=hist_state,
                                               hist_action=hist_action, hist_length=hist_lengths)

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
            actor_actions = self.actor.forward(state=states, hist_observation=hist_state, hist_action=hist_action,
                                               hist_length=hist_lengths)
            actor_loss = self.critic_1.forward(state=states, action=actor_actions, hist_observation=hist_state,
                                               hist_action=hist_action, hist_length=hist_lengths)
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


def play_one_episode(agent, env):
    total_reward = 0.0
    observation = env.reset()
    done = False

    if agent.history_length == 0:
        hist_state = np.zeros((1, env.observation_space.shape[0]))
        hist_action = np.zeros((1, env.action_space.shape[0]))
        length = 1
    else:
        hist_state = np.zeros((agent.history_length, env.observation_space.shape[0]))
        hist_action = np.zeros((agent.history_length, env.action_space.shape[0]))
        length = 0

    while not done:
        a = agent.get_action(state=observation, hist_observation=hist_state, hist_action=hist_action,
                             hist_length=length)
        prev_observation = observation
        observation, reward, done, info = env.step(a)
        total_reward += reward

        agent.replay_buffer.add_to_mem(prev_observation, a, reward, observation, done)

        if agent.history_length > 0:
            if length < agent.history_length:
                hist_state[length, :] = prev_observation.reshape(1, -1)
                hist_action[length, :] = a.reshape(1, -1)
                length += 1
            else:
                hist_state[0:-1, :] = hist_state[1:, :]
                hist_state[-1, :] = prev_observation.reshape(1, -1)
                hist_action[0:-1, :] = hist_action[1:, :]
                hist_action[-1, :] = a.reshape(1, -1)

        if agent.replay_buffer.counter > agent.batch_size + agent.history_length:
            agent.update()

    return agent, total_reward


def main(training=True):
    env = gym.make('InvertedDoublePendulumBulletEnv-v0')
    gamma = 0.99
    batch_size = 100
    memory_size = 1000000
    tau = 0.005
    warm_up = 1000
    action_noise = 0.1
    history_length = 5
    agent = Agent(env=env, state_dims=env.observation_space.shape[0], action_dims=env.action_space.shape[0],
                  action_max=env.action_space.high, action_min=env.action_space.low, history_length=history_length,
                  gamma=gamma, batch_size=batch_size, memory_size=memory_size, warm_up=warm_up,
                  tau=tau, actor_noise=action_noise)

    if training:
        reward_set = []
        avg_score_set = []
        n_iteration = 2000
        for t in range(n_iteration):
            agent, total_reward = play_one_episode(agent, env)

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
        plt.savefig('Section 8 - LSTM-TD3/InvertedDoublePendulumPyBulletEnv-v0/'
                    'Rewards_InvertedDoublePendulumPyBulletEnv')

        # Saving the networks
        T.save(agent.critic_1.state_dict(), 'Section 8 - LSTM-TD3/InvertedDoublePendulumPyBulletEnv-v0/critic_1')
        T.save(agent.critic_2.state_dict(), 'Section 8 - LSTM-TD3/InvertedDoublePendulumPyBulletEnv-v0/critic_2')
        T.save(agent.actor.state_dict(), 'Section 8 - LSTM-TD3/InvertedDoublePendulumPyBulletEnv-v0/actor')
        T.save(agent.target_critic_1.state_dict(), 'Section 8 - LSTM-TD3/InvertedDoublePendulumPyBulletEnv-v0/'
                                                   'target_critic_1')
        T.save(agent.target_critic_2.state_dict(), 'Section 8 - LSTM-TD3/InvertedDoublePendulumPyBulletEnv-v0/'
                                                   'target_critic_2')
        T.save(agent.target_actor.state_dict(), 'Section 8 - LSTM-TD3/InvertedDoublePendulumPyBulletEnv-v0/'
                                                'target_actor')

    else:
        # Loading the trained networks
        agent.critic_1.load_state_dict(T.load('Section 8 - LSTM-TD3/InvertedDoublePendulumPyBulletEnv-v0/critic_1'))
        agent.critic_2.load_state_dict(T.load('Section 8 - LSTM-TD3/InvertedDoublePendulumPyBulletEnv-v0/critic_2'))
        agent.actor.load_state_dict(T.load('Section 8 - LSTM-TD3/InvertedDoublePendulumPyBulletEnv-v0/actor'))
        agent.target_critic_1.load_state_dict(T.load('Section 8 - LSTM-TD3/InvertedDoublePendulumPyBulletEnv-v0/'
                                                     'target_critic_1'))
        agent.target_critic_2.load_state_dict(T.load('Section 8 - LSTM-TD3/InvertedDoublePendulumPyBulletEnv-v0/'
                                                     'target_critic_2'))
        agent.target_actor.load_state_dict(T.load('Section 8 - LSTM-TD3/InvertedDoublePendulumPyBulletEnv-v0/'
                                                  'target_actor'))

        # Showing the video
        env.render(mode='human')
        for t in range(5):
            total_reward = 0.0
            observation = env.reset()
            done = False

            if agent.history_length == 0:
                hist_state = np.zeros((1, env.observation_space.shape[0]))
                hist_action = np.zeros((1, env.action_space.shape[0]))
                length = 1
            else:
                hist_state = np.zeros((agent.history_length, env.observation_space.shape[0]))
                hist_action = np.zeros((agent.history_length, env.action_space.shape[0]))
                length = 0

            while not done:
                time.sleep(1.0 / 120)
                a = agent.get_action(state=observation, hist_observation=hist_state, hist_action=hist_action,
                                     hist_length=length, training=False)
                prev_observation = observation
                observation, reward, done, info = env.step(a)
                total_reward += reward

                if agent.history_length > 0:
                    if length < agent.history_length:
                        hist_state[length, :] = prev_observation.reshape(1, -1)
                        hist_action[length, :] = a.reshape(1, -1)
                        length += 1
                    else:
                        hist_state[0:-1, :] = hist_state[1:, :]
                        hist_state[-1, :] = prev_observation.reshape(1, -1)
                        hist_action[0:-1, :] = hist_action[1:, :]
                        hist_action[-1, :] = a.reshape(1, -1)
            print('video #' + str(t + 1) + ' -----> total reward:' + '%.2f' % total_reward)
        env.close()


if __name__ == '__main__':
    main(training=False)

