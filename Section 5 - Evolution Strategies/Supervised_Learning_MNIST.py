import numpy as np
import matplotlib.pyplot as plt


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
        self.parameters = params

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
        return np.mean(y_predict == y)





