import numpy as np
import matplotlib.pyplot as plt


class EvolutionStrategies:
    def __init__(self, population_size, parameter_size, function, learning_rate, noise_std):
        self.population_size = population_size
        self.parameter_size = parameter_size
        self.function = function
        self.lr = learning_rate
        self.noise_std = noise_std
        self.parameters = np.random.randn(self.parameter_size)

    def train(self):
        epsilon_list = []
        F = []
        for _ in range(self.population_size):
            epsilon = np.random.randn(self.parameter_size)
            epsilon_list.append(epsilon)
            param_try = self.parameters + epsilon * self.noise_std
            F.append(self.function(param_try))
        mean = np.mean(F)
        std = np.std(F)
        standardized_F = (np.array(F) - mean)/std
        gradient = (standardized_F.dot(np.array(epsilon_list))) / (self.population_size*self.noise_std)
        self.parameters += self.lr*gradient

    def get_function_value(self):
        return self.function(self.parameters)


def cost_function(input_):
    x1 = input_[0]
    x2 = input_[1]
    x3 = input_[2]
    return -(x1 ** 2 + 0.1 * (x2 - 1) ** 2 + 0.5 * (x3 + 2) ** 2)


if __name__ == '__main__':
    agent = EvolutionStrategies(population_size=100, parameter_size=3, function=cost_function, learning_rate=0.001,
                                noise_std=0.1)

    # Training
    num_iteration = 500
    cost_function_set = []
    for _ in range(num_iteration):
        agent.train()
        cost_function_set.append(agent.get_function_value())
    best_parameters = agent.parameters

    # Plotting the results
    plt.xlabel('Episode')
    plt.ylabel('Cost function value')
    plt.plot(np.arange(1, num_iteration + 1), cost_function_set)
    plt.show()

