import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import to_categorical
import multiprocessing
from multiprocessing.dummy import Pool


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))


def relu(x):
    return x * (x > 0)


class ANN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.w1 = np.random.randn(self.input_size, self.hidden_size)/np.sqrt(self.input_size)
        self.b1 = np.zeros(self.hidden_size)
        self.w2 = np.random.randn(self.hidden_size, self.output_size)/np.sqrt(self.hidden_size)
        self.b2 = np.zeros(self.output_size)
        self.parameters = [self.w1, self.b1, self.w2, self.b2]

    def set_parameters(self, params):
        I, H, O = self.input_size, self.hidden_size, self.output_size
        w1 = params[0: I * H].reshape((I, H))
        b1 = params[I * H: I * H + H]
        w2 = params[I * H + H: I * H + H + H * O].reshape((H, O))
        b2 = params[I * H + H + H * O:]
        self.parameters = [w1, b1, w2, b2]

    def get_parameters(self):
        return np.concatenate([self.parameters[0].flatten(), self.parameters[1].flatten(), self.parameters[2].flatten(),
                               self.parameters[3].flatten()])

    def predict(self, x):  # (batch_size, n_features)
        x = x.dot(self.parameters[0]) + self.parameters[1]
        x = relu(x)
        x = x.dot(self.parameters[2]) + self.parameters[3]
        return softmax(x)

    def score(self, x, y):
        y_predict = self.predict(x)
        y_predict = np.argmax(y_predict, axis=1)
        y = np.argmax(y, axis=1)
        return np.mean(y_predict == y)


def calculate_reward(parameters):
    ann = ANN(I, H, O)
    ann.set_parameters(parameters)
    return ann.score(x_train, y_train)


def evolution_strategy(lr, sigma, population_size, model, f, n_iteration):
    n_parameters = len(model.get_parameters())
    rewards_train = np.zeros(n_iteration)
    rewards_test = np.zeros(n_iteration)

    for t in range(n_iteration):
        epsilon = np.random.randn(population_size, n_parameters)
        rewards = pool.map(f, [model.get_parameters() + sigma*epsilon[i, :] for i in range(population_size)])
        rewards = np.array(rewards)
        mean = np.mean(rewards)
        std = np.std(rewards)
        rewards_standardized = (rewards - mean)/std
        updated_parameters = model.get_parameters() + lr/(population_size*sigma) * rewards_standardized.dot(epsilon)
        model.set_parameters(updated_parameters)

        rewards_train[t] = mean
        test_reward = model.score(x_test, y_test)
        rewards_test[t] = test_reward
        if (t + 1) % 20 == 0 or t == 0:
            print('iteration #' + str(t + 1) + ' ---> train accuracy:' + '%.2f' % mean + ', test accuracy: ' +
                  '%.2f' % test_reward)

    return model, rewards_train, rewards_test


if __name__ == '__main__':
    pool = Pool(8)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])/255.
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])/255.
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    I = len(x_train[0])
    H = 200
    O = len(y_train[0])
    learning_rate = 0.2
    noise_std = 0.1
    population_size = 50
    n_iteration = 500
    network = ANN(I, H, O)

    network, reward_train, reward_test = evolution_strategy(lr=learning_rate, sigma=noise_std,
                                                            population_size=population_size, model=network,
                                                            f=calculate_reward, n_iteration=n_iteration)

    # Plotting the results
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.plot(np.arange(1, n_iteration + 1), reward_train)
    plt.plot(np.arange(1, n_iteration + 1), reward_test)
    plt.legend(['Train', 'Test'], loc=4)
    plt.show()
















