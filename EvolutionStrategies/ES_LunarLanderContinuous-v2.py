import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import os
from plot_learning_curve import plot_learning_curve


"""
    cd into the "Section 3 - Evolution Strategies" folder first 
"""

os.environ['OMP_NUM_THREADS'] = '1'


class Policy(nn.Module):
    def __init__(self, state_dims, action_dims, action_max, fc1_dims, fc2_dims):
        super(Policy, self).__init__()
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.action_max = action_max
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = nn.Linear(self.state_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.action_dims)

        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        output = F.relu(self.fc1(state))
        output = F.relu(self.fc2(output))
        output = T.tanh(self.fc3(output)) * self.action_max

        return output

    def get_action(self, state):
        with T.no_grad():
            state = T.tensor([state], dtype=T.float).to(self.device)
            action = self.forward(state)

        return action.numpy()[0]


class EvolutionStrategies:
    def __init__(self, n_threads, n_workers, state_dims, action_dims, action_max, fc1_dims, fc2_dims,
                 initial_learning_rate, final_learning_rate, noise_std, directory, env_name: str, episode_max):
        self.n_threads = n_threads
        self.n_workers = n_workers
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.action_max = action_max
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.initial_lr = initial_learning_rate
        self.final_lr = final_learning_rate
        self.sigma = noise_std
        self.directory = directory
        self.env_name = env_name
        self.episode_max = episode_max

        self.global_policy = Policy(state_dims=self.state_dims, action_dims=self.action_dims,
                                    action_max=self.action_max, fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims)

    def train(self):
        env = gym.make(self.env_name)
        scores_history = []
        avg_scores_history = []
        best_score = -np.inf

        with T.no_grad():
            for episode in range(self.episode_max):
                all_scores = []
                all_epsilons = []
                pool = mp.Pool(processes=self.n_threads)
                outputs = pool.map(worker,
                                   [[self.global_policy, self.env_name, self.sigma] for _ in range(self.n_workers)])
                pool.close()
                pool.join()

                # outputs: (score, epsilons)
                for t in range(self.n_workers):
                    all_scores.append(outputs[t][0])
                    all_epsilons.append(outputs[t][1])

                lr = (self.final_lr - self.initial_lr)/self.episode_max * episode + self.initial_lr
                standardized_scores = standardize_scores(all_scores)
                global_policy_params = dict(self.global_policy.named_parameters())
                for name in global_policy_params:
                    params_update = T.zeros(global_policy_params[name].shape).to(self.global_policy.device)
                    for t in range(self.n_workers):
                        score = standardized_scores[t]
                        epsilons = all_epsilons[t]
                        params_update = params_update.clone() + \
                                        lr / (self.n_workers * self.sigma) * score * epsilons[name].clone()
                    global_policy_params[name] = global_policy_params[name].clone() + params_update.clone()
                self.global_policy.load_state_dict(global_policy_params)

                observation = env.reset()
                done = False
                total_reward = 0.0
                while not done:
                    a = self.global_policy.get_action(observation)
                    observation, reward, done, info = env.step(a)
                    total_reward += reward
                scores_history.append(total_reward)
                avg_scores_history.append(np.mean(scores_history[-100:]))

                if avg_scores_history[-1] >= best_score:
                    best_score = avg_scores_history[-1]
                    self.save_network()
                print('iteration #{} -----> score:{:.2f}, average score:{:.2f}, best score:{:.2f}'.
                       format(episode + 1, total_reward, avg_scores_history[-1], best_score))

        plot_learning_curve(env_name=self.env_name, directory=self.directory,
                            scores=scores_history, avg_scores=avg_scores_history)

    def save_network(self):
        T.save(self.global_policy.state_dict(), self.directory + '/policy_network')

    def load_network(self):
        self.global_policy.load_state_dict(T.load(self.directory + '/policy_network'))


def worker(inputs):
    global_policy = inputs[0]
    env_name = inputs[1]
    sigma = inputs[2]

    env = gym.make(env_name)
    local_policy = Policy(state_dims=global_policy.state_dims, action_dims=global_policy.action_dims,
                          action_max=global_policy.action_max, fc1_dims=global_policy.fc1_dims,
                          fc2_dims=global_policy.fc2_dims)

    local_policy_params = dict(local_policy.named_parameters())
    global_policy_params = dict(global_policy.named_parameters())
    epsilons = {}
    for name in local_policy_params:
        eps = T.randn(local_policy_params[name].shape).to(global_policy.device)
        epsilons[name] = eps
        local_policy_params[name] = global_policy_params[name].clone() + sigma * eps
    local_policy.load_state_dict(local_policy_params)

    observation = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        a = local_policy.get_action(state=observation)
        observation, reward, done, info = env.step(a)
        total_reward += reward
    env.close()

    return total_reward, epsilons


def standardize_scores(scores: list):
    scores = np.array(scores)
    mean = np.mean(scores)
    std = np.std(scores)
    standardized_scores = (scores - mean) / (std + 1e-5)

    return standardized_scores


def main(training: bool):
    env_name = 'LunarLanderContinuous-v2'
    env = gym.make(env_name)
    dir_ = env_name

    n_threads = 12
    n_workers = 24
    mp.set_start_method('spawn')
    fc1_dims = 64
    fc2_dims = 64
    initial_learning_rate = 0.03
    final_learning_rate = 0.01
    noise_std = 0.1
    episode_max = 300
    es = EvolutionStrategies(n_threads=n_threads, n_workers=n_workers, state_dims=env.observation_space.shape[0],
                             action_dims=env.action_space.shape[0], action_max=env.action_space.high[0],
                             fc1_dims=fc1_dims, fc2_dims=fc2_dims, initial_learning_rate=initial_learning_rate,
                             final_learning_rate=final_learning_rate, noise_std=noise_std, directory=dir_,
                             env_name=env_name, episode_max=episode_max)

    if training:
        es.train()

    else:
        es.load_network()
        for t in range(5):
            observation = env.reset()
            done = False
            total_reward = 0
            while not done:
                env.render()
                a = es.global_policy.get_action(observation)
                observation, reward, done, info = env.step(a)
                total_reward = total_reward + reward
            print('video #{} -----> score:{:.2f}'.format(t + 1, total_reward))
        env.close()


if __name__ == '__main__':
    main(training=False)
