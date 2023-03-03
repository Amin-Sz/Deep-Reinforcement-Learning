import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gym
from Utils.plot_learning_curve import plot_learning_curve


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
    def __init__(self, input_dims, n_actions, learning_rate=3e-4, fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()
        self.lr = learning_rate
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = nn.Linear(self.input_dims + self.n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        s_a = T.cat((state, action), dim=1)
        s_a = self.fc1(s_a)
        s_a = F.relu(s_a)
        s_a = self.fc2(s_a)
        s_a = F.relu(s_a)
        Q = self.fc3(s_a)
        return Q


class ValueNetwork(nn.Module):
    def __init__(self, input_dims, learning_rate=3e-4, fc1_dims=256, fc2_dims=256):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.lr = learning_rate
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = self.fc1(state)
        state = F.relu(state)
        state = self.fc2(state)
        state = F.relu(state)
        V = self.fc3(state)
        return V


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, action_max, learning_rate=3e-4, fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.action_max = action_max
        self.lr = learning_rate
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mean = nn.Linear(self.fc2_dims, n_actions)
        self.std = nn.Linear(self.fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = self.fc1(state)
        state = F.relu(state)
        state = self.fc2(state)
        state = F.relu(state)
        mean = self.mean(state)
        std = self.std(state)
        std = T.clamp(std, min=1e-6, max=1.0)  # Bounding standard deviation
        return mean, std

    def sample(self, state, reparameterize, training=True):
        action_max = T.tensor(self.action_max, dtype=T.float).view(1, -1).to(self.device)
        mean, std = self.forward(state)
        if training:
            distribution = Normal(mean, std)
            if reparameterize:
                u = distribution.rsample()
                a = action_max*T.tanh(u)
            else:
                u = distribution.sample()
                a = action_max*T.tanh(u)

            log_pi = T.sum(distribution.log_prob(u), dim=1, keepdim=True) - \
                     T.sum(T.log(action_max*(1.0 - T.multiply(T.tanh(u), T.tanh(u))) + 1e-6), dim=1, keepdim=True)
            return a, log_pi
        else:
            return action_max*T.tanh(mean), None


class Agent:
    def __init__(self, state_dims, action_dims, action_min, action_max, batch_size, reward_scale, alpha,
                 gamma=0.99, mem_size=1000000, tau=0.005):
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.action_min = action_min
        self.action_max = action_max
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

        self.replay_buffer = ReplayBuffer(mem_size=mem_size, state_dim=self.state_dims, action_dim=self.action_dims)

        self.actor = ActorNetwork(input_dims=self.state_dims, n_actions=self.action_dims, action_max=self.action_max)
        self.critic_1 = CriticNetwork(input_dims=self.state_dims, n_actions=self.action_dims)
        self.critic_2 = CriticNetwork(input_dims=self.state_dims, n_actions=self.action_dims)
        self.value = ValueNetwork(input_dims=self.state_dims)
        self.target_value = ValueNetwork(input_dims=self.state_dims)

        self.update_target_network(tau=1.0)

    def get_action(self, state, reparameterize=False, training=True):
        state = T.tensor([state], dtype=T.float).to(self.actor.device)
        action, _ = self.actor.sample(state, reparameterize=reparameterize, training=training)
        return action.cpu().detach().numpy()[0]

    def update_networks(self):
        states, actions, rewards, new_states, dones = self.replay_buffer.sample(batch_size=self.batch_size)
        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        new_states = T.tensor(new_states, dtype=T.float).to(self.actor.device)
        dones = T.tensor(dones).to(self.actor.device)

        V = self.value.forward(states)
        sampled_actions, log_probs = self.actor.sample(states, reparameterize=False)
        critic_value_1_current_policy = self.critic_1.forward(states, sampled_actions)
        critic_value_2_current_policy = self.critic_2.forward(states, sampled_actions)
        Q = T.min(critic_value_1_current_policy, critic_value_2_current_policy)
        self.value.optimizer.zero_grad()
        value_loss = (V - Q + log_probs*self.alpha)**2
        value_loss = 0.5*T.mean(value_loss)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        target_value = self.target_value.forward(new_states)
        target_value[dones] = 0.0
        target = rewards/self.alpha + self.gamma*target_value

        critic_value_1 = self.critic_1.forward(states, actions)
        critic_value_2 = self.critic_2.forward(states, actions)
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_1_loss = 0.5*F.mse_loss(critic_value_1, target)
        critic_2_loss = 0.5*F.mse_loss(critic_value_2, target)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward(retain_graph=True)
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        sampled_actions, log_probs = self.actor.sample(states, reparameterize=True)
        critic_value_1_current_policy = self.critic_1.forward(states, sampled_actions)
        critic_value_2_current_policy = self.critic_2.forward(states, sampled_actions)
        Q = T.min(critic_value_1_current_policy, critic_value_2_current_policy)
        self.actor.optimizer.zero_grad()
        actor_loss = log_probs*self.alpha - Q
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_target_network()

    def update_target_network(self, tau=None):
        if tau is None:
            tau = self.tau

        value_params = dict(self.value.named_parameters())
        target_value_params = dict(self.target_value.named_parameters())
        for name in target_value_params:
            target_value_params[name] = (1 - tau)*target_value_params[name].clone() \
                                        + tau*value_params[name].clone()

        self.target_value.load_state_dict(target_value_params)

    def save_networks(self, path):
        T.save(self.value.state_dict(), path + '/value_network')
        T.save(self.target_value.state_dict(),  path + '/target_value_network')
        T.save(self.critic_1.state_dict(),  path + '/critic_1_network')
        T.save(self.critic_2.state_dict(),  path + '/critic_2_network')
        T.save(self.actor.state_dict(),  path + '/actor_network')

    def load_networks(self, path):
        self.value.load_state_dict(T.load(path + '/value_network'))
        self.target_value.load_state_dict(T.load(path + '/target_value_network'))
        self.critic_1.load_state_dict(T.load(path + '/critic_1_network'))
        self.critic_2.load_state_dict(T.load(path + '/critic_2_network'))
        self.actor.load_state_dict(T.load(path + '/actor_network'))


def play_one_episode(agent, env, training, render=False):
    total_reward = 0
    counter = 0
    observation = env.reset()
    done = False
    while not done:
        if render:
            env.render()

        a = agent.get_action(observation, training=training)
        prev_observation = observation
        observation, reward, done, info = env.step(a)
        total_reward += reward
        counter += 1

        if training:
            if counter >= env._max_episode_steps:
                agent.replay_buffer.add_to_mem(prev_observation, a, reward, observation, done=False)
            else:
                agent.replay_buffer.add_to_mem(prev_observation, a, reward, observation, done)

            if agent.replay_buffer.counter > agent.batch_size:
                agent.update_networks()

    return agent, total_reward


def main(training):
    env_name = 'Hopper-v4'
    env = gym.make(env_name)
    path = 'Section 7 - SAC/' + env_name

    batch_size = 256
    reward_scale = 5.0
    alpha = 0.1
    gamma = 0.99
    tau = 0.005
    agent = Agent(state_dims=env.observation_space.shape[0], action_dims=env.action_space.shape[0],
                  action_min=env.action_space.low, action_max=env.action_space.high, batch_size=batch_size,
                  reward_scale=reward_scale, alpha=alpha, gamma=gamma, tau=tau)

    if training:
        reward_hist = []
        avg_score_hist = []
        test_score_hist = []
        best_test_score = -np.inf
        n_iteration = 1800
        test_frequency = 10
        for t in range(n_iteration):
            agent, total_reward_train = play_one_episode(agent, env, training=True)

            reward_hist.append(total_reward_train)
            avg_score_hist.append(np.mean(reward_hist[-100:]))
            print('iteration #{} -----> training score:{:.2f}, average score:{:.2f}, best test score:{:.2f}'
                  .format(t + 1, total_reward_train, avg_score_hist[-1], best_test_score))

            if total_reward_train > best_test_score:
                agent, total_reward_test = play_one_episode(agent, env, training=False)
                test_score_hist.append([t + 1, total_reward_test])
                if total_reward_test > best_test_score:
                    best_test_score = total_reward_test
                    agent.save_networks(path)

            if (t + 1) % test_frequency == 0 or t == 0:
                agent, total_reward_test = play_one_episode(agent, env, training=False)
                test_score_hist.append([t + 1, total_reward_test])
                print('=============================> test score:{:.2f} <=============================\n'
                      .format(total_reward_test))

                if total_reward_test > best_test_score:
                    best_test_score = total_reward_test
                    agent.save_networks(path)

        # Plot the learning curve
        plot_learning_curve(env_name=env_name, training_scores=reward_hist, avg_training_scores=avg_score_hist,
                            directory=path, test_scores=test_score_hist)

        # Saving the trained networks
        agent.save_networks(path=path)

    else:
        # Loading the trained networks
        agent.load_networks(path=path)

        # Showing the video
        for e in range(5):
            agent, total_reward = play_one_episode(agent, env, training=False, render=True)
            print('episode #{} -----> total reward:{:.2f}'.format(e + 1, total_reward))
        env.close()


if __name__ == '__main__':
    main(training=False)
