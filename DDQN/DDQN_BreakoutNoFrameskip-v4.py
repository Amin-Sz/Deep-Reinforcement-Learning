import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from Utils.atari_utils import make_env
from Utils.plot_learning_curve import plot_learning_curve


class ReplayBuffer:
    def __init__(self, mem_size, state_dim, action_dim):
        self.size = mem_size
        self.counter = 0
        self.states = np.zeros((mem_size, *state_dim), dtype=np.float32)
        self.states_new = np.zeros((mem_size, *state_dim), dtype=np.float32)
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
        self.counter += 1

    def sample(self, batch_size):
        indices = np.random.choice(np.min([self.size, self.counter]), size=batch_size, replace=False)
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        states_new = self.states_new[indices]
        dones = self.dones[indices]
        return states, actions, rewards, states_new, dones


class QNetwork(nn.Module):
    def __init__(self, in_channels, n_actions, fc1_dims=512, learning_rate=0.0002):
        super(QNetwork, self).__init__()
        self.in_channels = in_channels
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.lr = learning_rate

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
        self.fc1 = nn.Linear(3136, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.n_actions)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr)

    def forward(self, state):
        output = F.relu(self.conv1(state))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        output = nn.Flatten()(output)
        output = F.relu(self.fc1(output))
        q = self.fc2(output)

        return q


class Agent:
    def __init__(self, state_dims, n_actions, gamma, batch_size, final_exploration_state, update_frequency,
                 target_update_frequency, initial_eps, final_eps, memory_size=75000):
        self.state_dims = state_dims
        self.in_channels = self.state_dims[0]
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.mem_size = memory_size
        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.final_exploration_state = final_exploration_state
        self.update_frequency = update_frequency
        self.target_update_frequency = target_update_frequency
        self.play_counter = 0
        self.q_update_counter = 0

        self.replay_buffer = ReplayBuffer(mem_size=self.mem_size, state_dim=self.state_dims, action_dim=1)

        self.q_network = QNetwork(in_channels=self.in_channels, n_actions=self.n_actions)
        self.target_network = QNetwork(in_channels=self.in_channels, n_actions=self.n_actions)
        self.update_target_network()

    def get_action(self, state, training=True):
        state = T.tensor([state], dtype=T.float).to(self.q_network.device)
        if training:
            eps = max(self.final_eps,
                      (self.final_eps - self.initial_eps)/self.final_exploration_state * self.play_counter + self.initial_eps)
            self.play_counter += 1

            if np.random.rand() < eps:
                action = np.random.choice(self.n_actions)
                return action
            else:
                q = self.q_network.forward(state=state)
                action = T.argmax(q, dim=1)
                return action.cpu().detach().numpy()[0]
        else:
            q = self.q_network.forward(state=state)
            action = T.argmax(q, dim=1)
            return action.cpu().detach().numpy()[0]

    def update_networks(self):
        if self.play_counter % self.update_frequency == 0:
            states, actions, rewards, new_states, dones = self.replay_buffer.sample(batch_size=self.batch_size)
            states = T.tensor(states, dtype=T.float).to(self.q_network.device)
            actions = T.tensor(actions, dtype=T.int64).to(self.q_network.device)
            rewards = T.tensor(rewards, dtype=T.float).to(self.q_network.device)
            new_states = T.tensor(new_states, dtype=T.float).to(self.q_network.device)
            dones = T.tensor(dones).to(self.q_network.device)

            target_actions = self.q_network.forward(state=new_states)
            target_actions = T.argmax(target_actions, dim=1, keepdim=True)
            target_q = self.target_network.forward(state=new_states)
            target_q = T.gather(target_q, dim=1, index=target_actions)
            target_q[dones] = 0.0
            target = rewards + self.gamma*target_q

            q = self.q_network.forward(state=states)
            q = T.gather(q, dim=1, index=actions)

            self.q_network.optimizer.zero_grad()
            q_loss = F.smooth_l1_loss(input=q, target=target)
            q_loss.backward()
            self.q_network.optimizer.step()

        if self.play_counter % self.target_update_frequency == 0:
            self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_networks(self, directory):
        T.save(self.q_network.state_dict(), directory + '/q_network')
        T.save(self.target_network.state_dict(), directory + '/target_q_network')

    def load_networks(self, directory):
        self.q_network.load_state_dict(T.load(directory + '/q_network'))
        self.target_network.load_state_dict(T.load(directory + '/target_q_network'))


def play_one_episode(agent, env, training):
    observation = env.reset()
    observation, _, _, _ = env.step(1)  # Start the game
    done = False
    total_reward = 0.0
    lives = 5
    counter = 0

    while not done:
        a = agent.get_action(observation, training=training)
        prev_observation = observation
        observation, reward, done, info = env.step(a)
        total_reward += reward
        counter += 1

        if training:
            if info['lives'] < lives:
                lives = info['lives']
                agent.replay_buffer.add_to_mem(state=prev_observation, action=a, reward=reward,
                                               state_new=observation, done=True)
                observation, _, _, _ = env.step(1)  # Start the game
            else:
                agent.replay_buffer.add_to_mem(state=prev_observation, action=a, reward=reward,
                                               state_new=observation, done=done)
            if agent.replay_buffer.counter > agent.batch_size:
                agent.update_networks()
        else:
            if info['lives'] < lives:
                lives = info['lives']
                observation, _, _, _ = env.step(1)  # Start the game

    return agent, total_reward


def main(training):
    env_name = 'BreakoutNoFrameskip-v4'
    env = make_env(env_name)
    dir_ = env_name

    gamma = 0.99
    batch_size = 32
    final_exploration_state = 5000000
    initial_eps = 1.0
    final_eps = 0.01
    update_frequency = 4
    target_update_frequency = 30000
    agent = Agent(state_dims=env.observation_space.shape, n_actions=env.action_space.n, gamma=gamma,
                  batch_size=batch_size, final_exploration_state=final_exploration_state,
                  update_frequency=update_frequency, target_update_frequency=target_update_frequency,
                  initial_eps=initial_eps, final_eps=final_eps)

    if training:
        scores_train = []
        avg_scores_train = []
        scores_test = []
        n_iteration = 22000
        test_frequency = 100
        best_test_score = -np.inf

        for t in range(n_iteration):
            agent, total_reward_train = play_one_episode(agent, env, training=True)
            scores_train.append(total_reward_train)
            avg_scores_train.append(np.mean(scores_train[-100:]))
            print('iteration #' + str(t + 1) + ' -----> ' +
                  'training score:' + '%.2f' % total_reward_train +
                  ', average score:' + '%.2f' % np.mean(scores_train[-100:]))

            if t % test_frequency == 0 or t == n_iteration - 1:
                agent, total_reward_test = play_one_episode(agent, env, training=False)
                scores_test.append([t, total_reward_test])
                print('---- Test score: ' + '%.2f' % total_reward_test + ' ----')

                if total_reward_test > best_test_score:
                    best_test_score = total_reward_test
                    agent.save_networks(dir_)

        # Plotting the learning curve
        plot_learning_curve(env_name=env_name, directory=dir_, training_scores=scores_train,
                            avg_training_scores=avg_scores_train, test_scores=scores_test)

    else:
        env = make_env(env_name, render_mode='human')

        # Loading networks
        agent.load_networks(dir_)

        # Showing the video
        for t in range(1):
            observation = env.reset()
            observation, _, _, _ = env.step(1)  # Start the game
            done = False
            total_reward = 0.0
            lives = 5
            while not done:
                # env.render()
                a = agent.get_action(observation, training=False)
                observation, reward, done, info = env.step(a)

                if info['lives'] < lives:
                    lives = info['lives']
                    observation, _, _, _ = env.step(1)  # Start the game

                total_reward = total_reward + reward
            print('video #' + str(t + 1) + ' ----> total reward:' + '%.2f' % total_reward)
        env.close()


if __name__ == '__main__':
    main(training=False)

