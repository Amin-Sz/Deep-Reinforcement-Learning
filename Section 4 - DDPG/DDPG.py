import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense
from keras.optimizers import Adam
import gym


class NeuralNetwork(tf.keras.Model):
    def __init__(self, output_size, hidden_layers=[10]):
        super(NeuralNetwork, self).__init__()
        self.Layers = []
        dense = Dense(units=hidden_layers[0], activation='relu')
        self.Layers.append(dense)
        for n in hidden_layers[1:]:
            dense = Dense(units=n, activation='relu')
            self.Layers.append(dense)
        dense = Dense(units=output_size, activation='linear')
        self.Layers.append(dense)

    def call(self, x, training=None, mask=None):
        for layer in self.Layers:
            x = layer(x)
        return x


class Agent:
    def __init__(self, lr_Q, lr_pi, gamma, Q_layers, action_size, policy_layers):
        self.gamma = gamma
        self.Q_network = NeuralNetwork(output_size=1, hidden_layers=Q_layers)
        self.Q_network.compile(optimizer=Adam(learning_rate=lr_Q), loss='mean_squared_error')
        self.policy_network = NeuralNetwork(output_size=action_size, hidden_layers=policy_layers)










