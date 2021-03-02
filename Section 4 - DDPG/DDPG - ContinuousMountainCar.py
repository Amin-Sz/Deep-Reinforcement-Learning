import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Concatenate
from keras.optimizers import Adam
import gym
from sklearn.preprocessing import StandardScaler


class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.mem_size = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def add_to_mem(self, state, action, reward, next_state, done):
        if self.mem_size >= self.max_size:
            self.mem_size = self.mem_size - 1
            self.states = self.states[1:]
            self.actions = self.actions[1:]
            self.rewards = self.rewards[1:]
            self.next_states = self.next_states[1:]
            self.dones = self.dones[1:]

        self.mem_size = self.mem_size + 1
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def sample(self, batch_size):
        idx = np.random.choice(self.mem_size, size=batch_size, replace=False)
        states = np.squeeze(np.array(self.states)[idx])
        actions = np.squeeze(np.array(self.actions)[idx])
        rewards = np.squeeze(np.array(self.rewards)[idx])
        next_states = np.squeeze(np.array(self.next_states)[idx])
        dones = np.squeeze(np.array(self.dones)[idx])
        return states, actions, rewards, next_states, dones


class Agent:
    def __init__(self, mem_max_size, lr_Q, lr_policy, state_size, action_size, Q_layers, policy_layers, tau):
        self.replay_buffer = ReplayBuffer(max_size=mem_max_size)
        self.tau = tau
        self.action_size = action_size
        self.Q_network = ann(input_size=int(state_size + action_size), layers=Q_layers)  # CHECK THE LOSS FUNCTION
        self.Q_network.compile(optimizer=Adam(learning_rate=lr_Q), loss='mean_squared_error')
        self.policy_network = ann(input_size=state_size, layers=policy_layers)
        self.policy_network.compile(optimizer=Adam(learning_rate=lr_policy))

        self.Q_network_target = ann(input_size=int(state_size + action_size), layers=Q_layers)  # CHECK THESE
        self.Q_network_target.compile(optimizer=Adam(learning_rate=lr_Q))
        self.policy_network_target = ann(input_size=state_size, layers=policy_layers)
        self.policy_network_target.compile(optimizer=Adam(learning_rate=lr_policy))

    '''def update_Q(self, scaled_state, y):
        scaled_state = tf.convert_to_tensor(scaled_state, dtype=tf.float32)
        scaled_state = tf.squeeze(scaled_state)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        mu = self.policy_network.predict(scaled_state)
        x = Concatenate()([scaled_state, mu])
        self.Q_network.fit(x, y, verbose=0)

    def update_policy(self, scaled_state):
        scaled_state = tf.convert_to_tensor(scaled_state, dtype=tf.float32)
        scaled_state = tf.squeeze(scaled_state)
        with tf.GradientTape() as tape:
            mu = self.policy_network.predict(scaled_state)
            x = Concatenate()([scaled_state, mu])  # CHECK THIS
            loss = -tf.reduce_mean(self.Q_network.predict(x))  # CHECK THIS
        gradients = tape.gradient(loss, self.policy_network.trainable_variables)
        self.policy_network.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))'''

    def update_networks(self, scaled_states, actions, rewards, scaled_next_states, dones, gamma):
        scaled_states = tf.convert_to_tensor(scaled_states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions.reshape(-1, 1), dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        scaled_next_states = tf.convert_to_tensor(scaled_next_states, dtype=tf.float32)
        dones = np.array(dones)
        dones = dones.astype(int)

        X_Q = Concatenate()([scaled_states, actions])
        Y_Q = rewards + gamma*(1 - dones)*self.get_Q_target(scaled_next_states)
        self.Q_network.fit(X_Q, Y_Q, verbose=0)

        with tf.GradientTape() as tape:
            mu = self.policy_network.call(scaled_states)  # This should be different than actions (due to noise)
            x = Concatenate()([scaled_states, mu])
            loss_value = -tf.reduce_mean(self.Q_network.call(x))
        gradients = tape.gradient(loss_value, self.policy_network.trainable_variables)
        self.policy_network.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))

    def update_Q_target(self):
        variables = []
        for i in range(len(self.Q_network_target.weights)):
            variables.append(self.Q_network_target.weights[i]*self.tau +
                             self.Q_network.weights[i]*(1 - self.tau))
        self.Q_network_target.set_weights(variables)

    def update_policy_target(self):
        variables = []
        for i in range(len(self.policy_network_target.weights)):
            variables.append(self.policy_network_target.weights[i]*self.tau +
                             self.policy_network.weights[i]*(1 - self.tau))
        self.policy_network_target.set_weights(variables)

    def get_action(self, scaled_state):
        scaled_state = tf.convert_to_tensor(scaled_state, dtype=tf.float32)
        action = self.policy_network.call(scaled_state) + tf.random.normal(shape=[self.action_size],
                                                                           mean=0.0, stddev=0.05)
        action = tf.clip_by_value(action, clip_value_min=-1, clip_value_max=1)
        action = tf.squeeze(action)
        return action

    def get_Q_target(self, scaled_state):
        mu = self.policy_network_target.call(scaled_state)
        x = Concatenate()([scaled_state, mu])
        return tf.squeeze(self.Q_network_target.call(x))


def ann(input_size, layers=[(10, 'relu')]):
    model = Sequential()
    for i, (neurons, act) in enumerate(layers):
        if i == 0:
            model.add(Dense(units=neurons, activation=act, input_dim=input_size))
        else:
            model.add(Dense(units=neurons, activation=act))
    return model


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


def play_one_game(env, scaler):  # CHECK THIS (that agent gets updated)
    observation = env.reset()
    done = False
    counter = 0
    total_reward = 0
    while not done:
        a = agent.get_action(scaler.transform([observation]))  # CHECK THIS
        prev_observation = observation
        observation, reward, done, info = env.step([a])  # CHECK THIS
        total_reward = total_reward + reward
        counter = counter + 1
        if counter >= 2000:
            done = True
        agent.replay_buffer.add_to_mem(prev_observation, a, reward, observation, done)
    return total_reward, counter


# Creating the scaler, agent and environment
env = gym.make('MountainCarContinuous-v0')
scaler = get_scaler(env)
max_buffer_size = int(1e6)
agent = Agent(mem_max_size=max_buffer_size, lr_Q=0.001, lr_policy=0.001, state_size=env.observation_space.shape[0],
              action_size=env.action_space.shape[0], Q_layers=[(300, 'relu'), (1, 'linear')],
              policy_layers=[(300, 'relu'), (env.action_space.shape[0], 'tanh')], tau=0.995)


# Training
num_iteration = 50
min_buffer_size = 10000
batch_size = 100
gamma = 0.99  # Discount factor
reward_set = []
avg_reward_set = []
counter_set = []

for t in range(num_iteration):
    total_reward, counter = play_one_game(env, scaler)

    if agent.replay_buffer.mem_size >= min_buffer_size:
        for j in range(counter):
            s, a, r, s2, done = agent.replay_buffer.sample(batch_size=batch_size)  # CHECK THIS
            agent.update_networks(scaler.transform(s), a, r, scaler.transform(s2), done, gamma=gamma)
            agent.update_Q_target()
            agent.update_policy_target()

    counter_set.append(counter)
    reward_set.append(total_reward)
    avg_reward_set.append(np.mean(reward_set[-100:]))
    if t % 1 == 0:
        print('iteration #' + str(t + 1), '--->', 'total reward:' + '%.3f' % total_reward + ', ',
              'averaged reward:' + '%.3f' % np.mean(reward_set[-100:]))


# Plotting the results
plt.plot(reward_set)
plt.plot(avg_reward_set)
plt.plot(counter_set)




















