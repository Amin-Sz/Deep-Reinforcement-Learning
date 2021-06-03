import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, BatchNormalization, Concatenate
from keras.activations import relu, tanh
from keras.optimizers import Adam
from keras.losses import MSE
import gym


class ReplayBuffer:
    def __init__(self, mem_size, state_dim, action_dim):
        self.size = mem_size
        self.counter = 0
        self.states = np.zeros((mem_size, state_dim))
        self.states_new = np.zeros((mem_size, state_dim))
        self.actions = np.zeros((mem_size, action_dim))
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


class CriticNetwork(tf.keras.Model):
    def __init__(self, input_dims, n_actions, fc1_dims=512, fc2_dims=512, learning_rate=2e-3, weight_decay=1e-2):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr = learning_rate
        self.decay = weight_decay

        self.fc1 = Dense(units=self.fc1_dims, input_dim=self.input_dims, activation='relu')
        self.fc2 = Dense(units=self.fc2_dims, input_dim=self.fc1_dims, activation='relu')
        self.fc_a = Dense(units=self.fc2_dims, input_dim=self.n_actions)
        self.fc3 = Dense(units=1, input_dim=self.fc2_dims, activation='linear')

        self.optimizer = Adam(learning_rate=self.lr)

    def call(self, inputs, training=None, mask=None):
        inputs = self.fc1(inputs)
        inputs = self.fc2(inputs)
        q = self.fc3(inputs)

        return q


class ActorNetwork(tf.keras.Model):
    def __init__(self, input_dims, n_actions, fc1_dims=512, fc2_dims=512, learning_rate=1e-3, weight_decay=1e-2):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr = learning_rate
        self.decay = weight_decay

        self.fc1 = Dense(units=self.fc1_dims, input_dim=self.input_dims, activation='relu')
        self.fc2 = Dense(units=self.fc2_dims, input_dim=self.fc1_dims, activation='relu')
        self.fc3 = Dense(units=n_actions, input_dim=self.fc2_dims, activation='tanh')

        self.optimizer = Adam(learning_rate=self.lr)

    def call(self, state, training=None, mask=None):
        state = self.fc1(state)
        state = self.fc2(state)
        action = self.fc3(state)

        return action


class Agent:
    def __init__(self, state_dims, action_dims, action_max, mem_size, gamma=0.99, tau=0.001, batch_size=64):
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.action_max = action_max
        self.mem_size = mem_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.replay_buffer = ReplayBuffer(mem_size=self.mem_size, state_dim=self.state_dims,
                                          action_dim=self.action_dims)

        self.actor = ActorNetwork(input_dims=self.state_dims, n_actions=self.action_dims)
        self.critic = CriticNetwork(input_dims=self.state_dims + self.action_dims, n_actions=self.action_dims)
        self.target_actor = ActorNetwork(input_dims=self.state_dims, n_actions=self.action_dims)
        self.target_critic = CriticNetwork(input_dims=self.state_dims + self.action_dims, n_actions=self.action_dims)

        self.update_target_networks(tau=1.0)

    def get_action(self, state, training=True):
        state = tf.constant([state], dtype=tf.float32)
        action = self.actor.call(state)
        if training:
            action += tf.random.normal(shape=action.shape, mean=0.0, stddev=0.01)
            action = tf.clip_by_value(action, clip_value_min=-1.0, clip_value_max=1.0)

        return action.numpy()[0]  # todo Check that actions are being stored properly in the buffer

    def update_networks(self):
        states, actions, rewards, new_states, dones = self.replay_buffer.sample(batch_size=self.batch_size)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.bool)
        dones = tf.cast(dones, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor.call(new_states)
            x_target = Concatenate(axis=1)([new_states, target_actions])
            target = rewards + self.gamma * (1.0 - dones) * self.target_critic.call(x_target)

            x = Concatenate(axis=1)([states, actions])
            critic_value = self.critic.call(x)
            critic_loss = tf.reduce_mean(MSE(y_true=target, y_pred=critic_value))
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            mu = self.actor.call(states)
            x_actor = Concatenate(axis=1)([states, mu])
            actor_loss = -tf.reduce_mean(self.critic.call(x_actor))
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        self.update_target_networks()

    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        variables = []
        for i in range(len(self.target_critic.weights)):
            variables.append(tau*self.critic.weights[i] +
                             (1.0 - tau)*self.target_critic.weights[i])
        self.target_critic.set_weights(variables)

        variables = []
        for i in range(len(self.target_actor.weights)):
            variables.append(tau*self.actor.weights[i] +
                             (1.0 - tau)*self.target_actor.weights[i])
        self.target_actor.set_weights(variables)


def play_one_game(env, agent):
    action_max = env.action_space.high
    observation = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        a = agent.get_action(observation)
        prev_observation = observation
        observation, reward, done, info = env.step(action_max * a)
        total_reward += reward

        agent.replay_buffer.add_to_mem(prev_observation, a, reward, observation, done)

        if agent.replay_buffer.counter > agent.batch_size:
            agent.update_networks()

    return agent, total_reward


def main(training):
    env = gym.make('Pendulum-v0')
    memory_size = 1000000
    tau = 0.005
    gamma = 0.99
    batch_size = 64
    agent = Agent(state_dims=env.observation_space.shape[0], action_dims=env.action_space.shape[0],
                  action_max=env.action_space.high, mem_size=memory_size,
                  tau=tau, gamma=gamma, batch_size=batch_size)  # Change values of hyperparameters if you want

    if training:
        # Training the agent
        n_iterations = 300
        reward_history = []
        avg_reward_history = []
        for t in range(n_iterations):
            agent, total_reward = play_one_game(env, agent)

            reward_history.append(total_reward)
            avg_reward_history.append(np.mean(reward_history[-100:]))
            if (t + 1) % 1 == 0:
                print('iteration #' + str(t + 1) + ' -----> total reward:' + '%.2f' % total_reward +
                      ', average score:' + '%.2f' % np.mean(reward_history[-100:]))

        # Plotting the train results
        axes = plt.axes()
        axes.set_ylim([np.min(reward_history) - 400, np.max(reward_history) + 50])
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(np.arange(1, n_iterations + 1), reward_history)
        plt.plot(np.arange(1, n_iterations + 1), avg_reward_history)
        legend_2 = 'Running average of the last 100 episodes (' + '%.2f' % np.mean(reward_history[-100:]) + ')'
        plt.legend(['Reward', legend_2], loc=4)
        plt.show()
        plt.savefig('Section 4 - DDPG/Pendulum-v0_TF/Rewards_Pendulum-v0')

        # Saving the networks
        agent.critic.save_weights('Section 4 - DDPG/Pendulum-v0_TF/critic_Pendulum-v0')
        agent.actor.save_weights('Section 4 - DDPG/Pendulum-v0_TF/actor_Pendulum-v0')
        agent.target_critic.save_weights('Section 4 - DDPG/Pendulum-v0_TF/target_critic_Pendulum-v0')
        agent.target_actor.save_weights('Section 4 - DDPG/Pendulum-v0_TF/target_actor_Pendulum-v0')

    else:
        # Loading networks
        agent.critic.load_weights('Section 4 - DDPG/Pendulum-v0_TF/critic_Pendulum-v0')
        agent.actor.load_weights('Section 4 - DDPG/Pendulum-v0_TF/actor_Pendulum-v0')
        agent.target_critic.load_weights('Section 4 - DDPG/Pendulum-v0_TF/target_critic_Pendulum-v0')
        agent.target_actor.load_weights('Section 4 - DDPG/Pendulum-v0_TF/target_actor_Pendulum-v0')

        # Showing video
        action_max = env.action_space.high
        for _ in range(10):
            observation = env.reset()
            done = False
            total_reward = 0
            while not done:
                env.render()
                a = agent.get_action(observation, training=False)
                observation, reward, done, info = env.step(action_max * a)
                total_reward = total_reward + reward
            print('total reward:' + '%.2f' % total_reward)
        env.close()


if __name__ == '__main__':
    main(training=False)

