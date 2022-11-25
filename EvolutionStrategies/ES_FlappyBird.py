import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from Utils.FlappyBird_utils import FlappyBirdEnvironment
import time
from Utils.plot_learning_curve import plot_learning_curve


# os.environ['OMP_NUM_THREADS'] = '1'


class Policy(nn.Module):
    def __init__(self, state_dims, n_actions, fc1_dims):
        super(Policy, self).__init__()
        self.state_dims = state_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims

        self.fc1 = nn.Linear(self.state_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.n_actions)

        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        output = F.relu(self.fc1(state))
        output = self.fc2(output)
        return output

    def get_action(self, state):
        with T.no_grad():
            state = T.tensor([state], dtype=T.float).to(self.device)
            action = T.argmax(self.forward(state), dim=1)
        return action.numpy()[0]


class EvolutionStrategies:
    def __init__(self, n_threads, n_workers, state_dims, n_actions, fc1_dims, initial_learning_rate,
                 final_learning_rate, noise_std, directory, env_name: str, episode_max):
        self.n_threads = n_threads
        self.n_workers = n_workers
        self.state_dims = state_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.initial_lr = initial_learning_rate
        self.final_lr = final_learning_rate
        self.sigma = noise_std
        self.directory = directory
        self.env_name = env_name
        self.episode_max = episode_max

        self.global_policy = Policy(state_dims=self.state_dims, n_actions=self.n_actions,
                                    fc1_dims=self.fc1_dims)

    def train(self):
        env = FlappyBirdEnvironment()
        scores_history = []
        avg_scores_history = []
        best_score = -np.inf

        with T.no_grad():
            for episode in range(self.episode_max):
                all_scores = []
                all_epsilons = []

                # Multiprocessing hinders the training
                '''pool = mp.Pool(processes=self.n_threads)
                outputs = pool.map(worker,
                                   [[self.global_policy, self.sigma] for _ in range(self.n_workers)])
                pool.close()
                pool.join()

                # outputs: (score, epsilons)
                for t in range(self.n_workers):
                    all_scores.append(outputs[t][0])
                    all_epsilons.append(outputs[t][1])'''

                for _ in range(self.n_workers):
                    score, epsilons = worker([self.global_policy, env, self.sigma])
                    all_scores.append(score)
                    all_epsilons.append(epsilons)

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
    env = inputs[1]
    sigma = inputs[2]

    # env = FlappyBirdEnvironment()
    local_policy = Policy(state_dims=global_policy.state_dims, n_actions=global_policy.n_actions,
                          fc1_dims=global_policy.fc1_dims)

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

    return total_reward, epsilons


def standardize_scores(scores: list):
    scores = np.array(scores)
    mean = np.mean(scores)
    std = np.std(scores)
    standardized_scores = (scores - mean) / (std + 1e-5)

    return standardized_scores


def main(training: bool):
    env_name = 'FlappyBird'
    env = FlappyBirdEnvironment()
    dir_ = env_name

    n_threads = 12
    n_workers = 50
    # mp.set_start_method('spawn')
    fc1_dims = 64
    initial_learning_rate = 0.03
    final_learning_rate = 0.005
    noise_std = 0.1
    episode_max = 3000
    es = EvolutionStrategies(n_threads=n_threads, n_workers=n_workers, state_dims=env.observation_space_shape[0],
                             n_actions=env.action_space_n, fc1_dims=fc1_dims,
                             initial_learning_rate=initial_learning_rate, final_learning_rate=final_learning_rate,
                             noise_std=noise_std, directory=dir_, env_name=env_name, episode_max=episode_max)

    if training:
        es.train()

    else:
        es.load_network()
        env.set_display_screen(display=True)
        for t in range(5):
            observation = env.reset()
            done = False
            total_reward = 0
            while not done:
                time.sleep(1 / 120)
                a = es.global_policy.get_action(observation)
                observation, reward, done, info = env.step(a)
                total_reward = total_reward + reward
            print('episode #{} -----> score:{:.2f}'.format(t + 1, total_reward))


if __name__ == '__main__':
    main(training=False)
