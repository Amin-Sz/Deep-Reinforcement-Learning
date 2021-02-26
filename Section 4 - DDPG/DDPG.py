import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, Concatenate
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
    def __init__(self, lr_Q, lr_policy, gamma, Q_layers, action_size, policy_layers, tau):
        self.replay_buffer = []
        self.gamma = gamma
        self.tau = tau
        self.Q_network = NeuralNetwork(output_size=1, hidden_layers=Q_layers)
        self.Q_network.compile(optimizer=Adam(learning_rate=lr_Q), loss='mean_squared_error')
        self.policy_network = NeuralNetwork(output_size=action_size, hidden_layers=policy_layers)
        self.policy_optimizer = Adam(learning_rate=lr_policy)

        self.Q_network_target = self.Q_network
        self.policy_network_target = self.policy_network

    def update_Q(self, scaled_state, y):
        scaled_state = tf.convert_to_tensor(scaled_state, dtype=tf.float32)
        mu = self.policy_network.predict(scaled_state)
        x = Concatenate()([scaled_state, mu])
        self.Q_network.fit(x, y, verbose=0)

    def update_policy(self, scaled_state):
        scaled_state = tf.convert_to_tensor(scaled_state, dtype=tf.float32)
        with tf.GradientTape as tape:
            mu = self.policy_network.predict(scaled_state)
            x = Concatenate()([scaled_state, mu])
            loss = -tf.reduce_mean(self.Q_network(x))  # CHECK THIS
        gradients = tape.gradient(loss, self.policy_network.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))











