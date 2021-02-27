import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, Concatenate
from keras.optimizers import Adam
import gym
from sklearn.preprocessing import StandardScaler


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
    def __init__(self, lr_Q, lr_policy, Q_layers, action_size, policy_layers, tau):
        self.replay_buffer = []
        self.tau = tau
        self.Q_network = NeuralNetwork(output_size=1, hidden_layers=Q_layers)
        self.Q_network.compile(optimizer=Adam(learning_rate=lr_Q), loss=squared_error)  # CHECK THIS
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
            x = Concatenate()([scaled_state, mu])  # CHECK THIS
            loss = -tf.reduce_mean(self.Q_network(x))  # CHECK THIS
        gradients = tape.gradient(loss, self.policy_network.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))

    def update_Q_target(self):
        for i in range(len(self.Q_network_target.trainable_variables)):
            self.Q_network_target.trainable_variables[i].assign(
                self.Q_network_target.trainable_variables[i]*self.tau +
                self.Q_network.trainable_variables[i]*(1 - self.tau),
                read_value=False)

    def update_policy_target(self):
        for i in range(len(self.policy_network_target.trainable_variables)):
            self.Q_network_target.trainable_variables[i].assign(
                self.policy_network_target.trainable_variables[i]*self.tau +
                self.policy_network.trainable_variables[i]*(1 - self.tau),
                read_values=False)

    def get_action(self, scaled_state):
        scaled_state = tf.convert_to_tensor([scaled_state], dtype=tf.float32)
        action = self.policy_network.predict(scaled_state)
        action = tf.squeeze(action)
        action = action.numpy()
        for i in range(len(action)):
            if action[i] >= 1:
                action[i] = 1
            elif action[i] <= -1:
                action[i] = -1
            else:
                pass
        return action

    def get_Q_target(self, scaled_state):
        scaled_state = tf.convert_to_tensor([scaled_state], dtype=tf.float32)
        mu = self.policy_network.predict(scaled_state)
        x = Concatenate()([scaled_state, mu])
        return self.Q_network_target.predict(x)


def squared_error(y_true, y_predict):
    return tf.reduce_sum(tf.square(y_predict - y_true), axis=-1)


def get_scaler(env):
    scaler = StandardScaler()
    x = []
    for _ in range(10000):
        x.append(env.observation_space.sample())
    x = np.array(x)
    scaler.fit(x)
    return scaler


def play_one_game(env, agent, scaler):  # CHECK THIS
    observation = env.reset()
    done = False
    counter = 0
    total_reward = 0
    while not done:
        a = agent.get_action(scaler.transform(observation))  # CHECK THIS
        prev_observation = observation
        observation, reward, done, info = env.step([a])  # CHECK THIS
        total_reward = total_reward + reward
        counter = counter + 1
        if counter >= 2000:
            done = True

        agent.replay_buffer.append((prev_observation, a, reward, observation, done))
        if len(agent.replay_buffer) > max_buffer_size:
            agent.replay_buffer = agent.replay_buffer[1:]
    return total_reward, counter


# Creating the agent and environment
env = gym.make('MountainCarContinuous-v0')
scaler = get_scaler(env)
agent = Agent(lr_Q=0.005, lr_policy=0.005, Q_layers=[10, 20, 25], action_size=env.action_space.shape[0],
              policy_layers=[10, 20], tau=0.99)


# Training
num_iteration = 1000
max_buffer_size = 50000
min_buffer_size = 1000
batch_size = 32
gamma = 0.95  # Discount factor
reward_set = []
avg_reward_set = []

for t in range(num_iteration):
    counter, total_reward = play_one_game(env, agent, scaler)
    reward_set.append(total_reward)
    avg_reward_set.append(np.mean(reward_set[-100:]))
    if t % 100 == 0:
        print('iteration #', str(t), '--->', 'total reward:', str(total_reward), ', ',
              'averaged reward:', str(np.mean(reward_set[-100:])))

    if len(agent.replay_buffer) >= min_buffer_size:
        for j in range(counter):
            idx = np.random.choice(len(agent.replay_buffer), size=batch_size, replace=False)
            X = []
            Y = []
            for i in idx:
                (s_train, a_train, r_train, s2_train, done_train) = agent.replay_buffer[i][0]
                X.append(scaler.transform(s_train))
                target = r_train + gamma*(1 - done_train)*agent.get_Q_target(scaler.transform(s2_train))
                Y.append(target)
            agent.update_Q(X, Y)
            agent.update_policy(X)
            agent.update_Q_target()
            agent.update_policy_target()


# Plotting the results
plt.plot(reward_set)
plt.plot(avg_reward_set)




















