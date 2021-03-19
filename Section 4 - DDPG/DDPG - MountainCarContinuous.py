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
    def __init__(self, mem_max_size, lr_Q, lr_policy, state_size, action_size, action_max, action_min, Q_layers,
                 policy_layers, tau):
        self.replay_buffer = ReplayBuffer(max_size=mem_max_size)
        self.tau = tau
        self.action_size = action_size
        self.action_min = action_min
        self.action_max = action_max
        self.critic_network = ann(input_size=int(state_size + action_size), layers=Q_layers)
        self.critic_network.compile(optimizer=Adam(learning_rate=lr_Q), loss='mean_squared_error')
        self.actor_network = ann(input_size=state_size, layers=policy_layers)
        self.actor_network.compile(optimizer=Adam(learning_rate=lr_policy))

        self.critic_network_target = ann(input_size=int(state_size + action_size), layers=Q_layers)
        self.critic_network_target.compile(optimizer=Adam(learning_rate=lr_Q))
        self.actor_network_target = ann(input_size=state_size, layers=policy_layers)
        self.actor_network_target.compile(optimizer=Adam(learning_rate=lr_policy))

    def update_networks(self, scaled_states, actions, rewards, scaled_next_states, dones, gamma):
        scaled_states = tf.convert_to_tensor(scaled_states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions.reshape(-1, 1), dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        scaled_next_states = tf.convert_to_tensor(scaled_next_states, dtype=tf.float32)
        dones = np.array(dones)
        dones = dones.astype(int)

        X_Q = Concatenate()([scaled_states, actions])
        Y_Q = rewards + gamma * (1 - dones) * self.get_critic_target(scaled_next_states)
        self.critic_network.fit(X_Q, Y_Q, verbose=0)

        with tf.GradientTape() as tape:
            mu = self.actor_network.call(scaled_states)  # This should be different than actions (due to noise)
            x = Concatenate()([scaled_states, mu])
            loss_value = -tf.reduce_mean(self.critic_network.call(x))
        gradients = tape.gradient(loss_value, self.actor_network.trainable_variables)
        self.actor_network.optimizer.apply_gradients(zip(gradients, self.actor_network.trainable_variables))

    def update_critic_target(self):
        variables = []
        for i in range(len(self.critic_network_target.weights)):
            variables.append(self.critic_network_target.weights[i] * self.tau +
                             self.critic_network.weights[i] * (1 - self.tau))
        self.critic_network_target.set_weights(variables)

    def update_actor_target(self):
        variables = []
        for i in range(len(self.actor_network_target.weights)):
            variables.append(self.actor_network_target.weights[i] * self.tau +
                             self.actor_network.weights[i] * (1 - self.tau))
        self.actor_network_target.set_weights(variables)

    def get_action(self, scaled_state, training=True):
        scaled_state = tf.convert_to_tensor(scaled_state, dtype=tf.float32)
        if training:
            action = self.actor_network.call(scaled_state)*self.action_max + tf.random.normal(shape=[self.action_size],
                                                                                              mean=0.0, stddev=0.01)
        else:
            action = self.actor_network.call(scaled_state)*self.action_max
        action = tf.clip_by_value(action, clip_value_min=self.action_min, clip_value_max=self.action_max)
        action = tf.squeeze(action)
        return action

    def get_critic_target(self, scaled_state):
        mu = self.actor_network_target.call(scaled_state)
        x = Concatenate()([scaled_state, mu])
        return tf.squeeze(self.critic_network_target.call(x))


def ann(input_size, layers=[(10, 'relu')]):
    model = Sequential()
    for i, (neurons, act) in enumerate(layers):
        if i == 0:
            model.add(Dense(units=neurons, activation=act, input_dim=input_size))
        else:
            model.add(Dense(units=neurons, activation=act))
    return model


def get_scaler(env):
    scaler = StandardScaler()
    x = []
    for _ in range(10000):
        x.append(env.observation_space.sample())
    x = np.array(x)
    scaler.fit(x)
    return scaler


def play_one_game(agent, env, scaler):
    observation = env.reset()
    done = False
    counter = 0
    total_reward = 0
    while not done:
        a = agent.get_action(scaler.transform([observation]))
        prev_observation = observation
        observation, reward, done, info = env.step([a])
        total_reward = total_reward + reward
        counter = counter + 1
        if counter >= 2000:
            done = True
        agent.replay_buffer.add_to_mem(prev_observation, a, reward, observation, done)
    return agent, total_reward, counter


def main(training=False):
    # Creating the scaler, agent and environment
    env = gym.make('MountainCarContinuous-v0')
    scaler = get_scaler(env)
    max_buffer_size = int(1e6)
    agent = Agent(mem_max_size=max_buffer_size, lr_Q=0.01, lr_policy=0.01, state_size=env.observation_space.shape[0],
                  action_size=env.action_space.shape[0], action_max=env.action_space.high[0],
                  action_min=env.action_space.low[0], Q_layers=[(64, 'relu'), (1, 'linear')],
                  policy_layers=[(64, 'relu'), (env.action_space.shape[0], 'tanh')], tau=0.99)

    if training:
        # Training
        num_iteration = 300
        min_buffer_size = 10000
        batch_size = 32
        gamma = 0.99  # Discount factor
        reward_set = []  # Stores rewards of each episode
        avg_reward_set = []  # Stores the average of the last 100 rewards

        for t in range(num_iteration):
            agent, total_reward, counter = play_one_game(agent, env, scaler)

            if agent.replay_buffer.mem_size >= min_buffer_size:
                for j in range(np.min([counter, 500])):
                    s, a, r, s2, done = agent.replay_buffer.sample(batch_size=batch_size)
                    agent.update_networks(scaler.transform(s), a, r, scaler.transform(s2), done, gamma=gamma)
                    agent.update_critic_target()
                    agent.update_actor_target()

            reward_set.append(total_reward)
            avg_reward_set.append(np.mean(reward_set[-100:]))
            if (t + 1) % 20 == 0 or t == 0:
                print('iteration #' + str(t + 1), '--->', 'total reward:' + '%.2f' % total_reward + ', ',
                      'average reward:' + '%.2f' % np.mean(reward_set[-100:]))

        # Plotting the train results
        axes = plt.axes()
        axes.set_ylim([np.min(reward_set) - 100, np.max(reward_set) + 10])
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(np.arange(1, num_iteration + 1), reward_set)
        plt.plot(np.arange(1, num_iteration + 1), avg_reward_set)
        legend_2 = 'Running average of the last 100 episodes (' + '%.2f' % np.mean(reward_set[-100:]) + ')'
        plt.legend(['Reward', legend_2], loc=4)
        plt.show()
        plt.savefig('Section 4 - DDPG/MountainCarContinuous/Rewards_MountainCarContinuous')

        # Saving the networks
        agent.actor_network.save('Section 4 - DDPG/MountainCarContinuous/actor_MountainCar.h5')
        agent.critic_network.save('Section 4 - DDPG/MountainCarContinuous/critic_MountainCar.h5')
        agent.actor_network_target.save('Section 4 - DDPG/MountainCarContinuous/actor_target_MountainCar.h5')
        agent.critic_network_target.save('Section 4 - DDPG/MountainCarContinuous/critic_target_MountainCar.h5')

    else:
        # Importing the trained networks
        agent.actor_network = tf.keras.models.load_model('Section 4 - DDPG/Pendulum-v0/actor_MountainCar.h5')
        agent.actor_network_target = tf.keras.models.load_model('Section 4 - DDPG/Pendulum-v0/actor_target_MountainCar.h5')
        agent.critic_network = tf.keras.models.load_model('Section 4 - DDPG/Pendulum-v0/critic_MountainCar.h5')
        agent.critic_network_target = tf.keras.models.load_model('Section 4 - DDPG/Pendulum-v0/critic_target_MountainCar.h5')

        # Showing the video
        observation = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()
            a = agent.get_action(scaler.transform([observation]), training=False)
            observation, reward, done, info = env.step([a])
            total_reward = total_reward + reward
        env.close()
        print('total reward:' + '%.2f' % total_reward)


if __name__ == '__main__':
    main(training=True)

