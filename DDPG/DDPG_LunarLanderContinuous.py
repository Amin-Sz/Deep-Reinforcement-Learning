import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.optimizers import Adam
import gym
from sklearn.preprocessing import StandardScaler


class ReplayBuffer:
    def __init__(self, mem_size, state_size, action_size):
        self.size = mem_size
        self.counter = 0
        self.states = np.zeros((mem_size, state_size))
        self.states_new = np.zeros((mem_size, state_size))
        self.actions = np.zeros((mem_size, action_size))
        self.rewards = np.zeros((mem_size, 1))
        self.dones = np.zeros((mem_size, 1), dtype=bool)

    def add_to_mem(self, state, action, reward, state_new, done):
        index = self.counter % self.size
        self.states[index, :] = state
        self.actions[index, :] = action
        self.rewards[index] = reward
        self.states_new[index, :] = state_new
        self.dones[index] = done
        self.counter = self.counter + 1

    def sample(self, batch_size):
        indices = np.random.choice(np.min([self.size, self.counter]), size=batch_size, replace=False)
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        states_new = self.states_new[indices]
        dones = self.dones[indices]
        return states, actions, rewards, states_new, dones


class Agent:
    def __init__(self, mem_size, lr_Q, lr_policy, state_size, action_size, action_max, action_min, Q_layers,
                 policy_layers, tau):
        self.replay_buffer = ReplayBuffer(mem_size=mem_size, state_size=state_size, action_size=action_size)
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
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        scaled_next_states = tf.convert_to_tensor(scaled_next_states, dtype=tf.float32)
        dones = dones.astype(int)

        X_Q = Concatenate()([scaled_states, actions])
        Y_Q = rewards + gamma * (1 - dones) * self.get_critic_target(scaled_next_states)
        # or Y_Q = rewards + gamma * (1 - dones) * self.get_critic_target(scaled_next_states).numpy().reshape(-1, 1)
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
        return self.critic_network_target.call(x)


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
        observation, reward, done, info = env.step(a)
        total_reward = total_reward + reward
        counter = counter + 1
        agent.replay_buffer.add_to_mem(prev_observation, a, reward, observation, done)
    return agent, total_reward, counter


def main(training=False):
    # Creating the scaler, agent and environment
    env = gym.make('LunarLanderContinuous-v2')
    scaler = get_scaler(env)
    buffer_size = int(1e6)
    agent = Agent(mem_size=buffer_size, lr_Q=0.0001, lr_policy=0.0001, state_size=env.observation_space.shape[0],
                  action_size=env.action_space.shape[0], action_max=env.action_space.high[0],
                  action_min=env.action_space.low[0], Q_layers=[(512, 'relu'), (1, 'linear')],
                  policy_layers=[(512, 'relu'), (env.action_space.shape[0], 'tanh')], tau=0.995)

    if training:
        # Training
        num_iteration = 5000
        batch_size = 128
        gamma = 0.99  # Discount factor
        reward_set = []  # Stores rewards of each episode
        avg_reward_set = []  # Stores the average of the last 100 rewards

        for t in range(num_iteration):
            agent, total_reward, counter = play_one_game(agent, env, scaler)

            if agent.replay_buffer.counter > batch_size:
                for j in range(counter):
                    s, a, r, s2, done = agent.replay_buffer.sample(batch_size=batch_size)
                    agent.update_networks(scaler.transform(s), a, r, scaler.transform(s2), done, gamma=gamma)
                    agent.update_critic_target()
                    agent.update_actor_target()

            reward_set.append(total_reward)
            avg_reward_set.append(np.mean(reward_set[-100:]))
            if (t + 1) % 100 == 0 or t == 0:
                print('iteration #' + str(t + 1), '--->', 'total reward:' + '%.2f' % total_reward + ', ',
                      'average score:' + '%.2f' % np.mean(reward_set[-100:]))

        # Plotting the train results
        axes = plt.axes()
        axes.set_ylim([np.min(reward_set) - 400, np.max(reward_set) + 50])
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(np.arange(1, num_iteration + 1), reward_set)
        plt.plot(np.arange(1, num_iteration + 1), avg_reward_set)
        plt.plot(np.ones(num_iteration)*200, 'r-')
        legend_2 = 'Running average of the last 100 episodes (' + '%.2f' % np.mean(reward_set[-100:]) + ')'
        plt.legend(['Reward', legend_2, 'Reward of 200'], loc=4)
        plt.show()
        plt.savefig('LunarLanderContinuous-v2/Rewards_LunarLander')

        # Saving the networks
        agent.actor_network.save('LunarLanderContinuous-v2/actor_LunarLander.h5')
        agent.critic_network.save('LunarLanderContinuous-v2/critic_LunarLander.h5')
        agent.actor_network_target.save('LunarLanderContinuous-v2/actor_target_LunarLander.h5')
        agent.critic_network_target.save('LunarLanderContinuous-v2/critic_target_LunarLander.h5')

    else:
        # Importing the trained networks
        agent.actor_network = tf.keras.models.load_model('LunarLanderContinuous-v2/actor_LunarLander.h5')
        agent.actor_network_target = tf.keras.models.load_model('LunarLanderContinuous-v2/actor_target_LunarLander.h5')
        agent.critic_network = tf.keras.models.load_model('LunarLanderContinuous-v2/critic_LunarLander.h5')
        agent.critic_network_target = tf.keras.models.load_model('LunarLanderContinuous-v2/critic_target_LunarLander.h5')

        # Showing the video
        for _ in range(10):
            observation = env.reset()
            done = False
            total_reward = 0
            while not done:
                env.render()
                a = agent.get_action(scaler.transform([observation]), training=False)
                observation, reward, done, info = env.step(a)
                total_reward = total_reward + reward
            print('total reward:' + '%.2f' % total_reward)
        env.close()


if __name__ == '__main__':
    main(training=False)

