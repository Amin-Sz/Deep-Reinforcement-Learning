import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, BatchNormalization
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
    def __init__(self, input_dims, n_actions, fc1_dims=400, fc2_dims=300, learning_rate=1e-3, weight_decay=1e-2):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr = learning_rate
        self.decay = weight_decay

        self.fc1 = Dense(units=self.fc1_dims, input_dim=self.input_dims)
        self.fc2 = Dense(units=self.fc2_dims, input_dim=self.fc1_dims)
        self.fc_a = Dense(units=self.fc2_dims, input_dim=self.n_actions)
        self.fc3 = Dense(units=1, input_dim=self.fc2_dims)

        self.optimizer = Adam(learning_rate=self.lr, decay=self.decay)

    def forward(self, x, a, training=True):
        x = self.fc1(x)
        x = BatchNormalization()(x, training)
        x = relu(x)

        x = self.fc2(x)
        x = BatchNormalization()(x, training)  # todo Change the training mode when you are training or evaluating
        a = self.fc_a(a)
        x_a = relu(tf.add(x, a))

        q = self.fc3(x_a)
        return q


class ActorNetwork(tf.keras.Model):
    def __init__(self, input_dims, n_actions, fc1_dims=400, fc2_dims=300, learning_rate=1e-4, weight_decay=1e-2):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr = learning_rate
        self.decay = weight_decay

        self.fc1 = Dense(units=self.fc1_dims, input_dim=self.input_dims)
        self.fc2 = Dense(units=self.fc2_dims, input_dim=self.fc1_dims)
        self.fc3 = Dense(units=n_actions, input_dim=self.fc2_dims)

        self.optimizer = Adam(learning_rate=self.lr, decay=self.decay)

    def forward(self, x, training=True):
        x = self.fc1(x)
        x = BatchNormalization()(x, training)  # todo Change the training mode when you are training or evaluating
        x = relu(x)

        x = self.fc2(x)
        x = BatchNormalization()(x, training)
        x = relu(x)

        x = self.fc3(x)
        a = tanh(x)  # todo Check that action_max must be multiplied here or in the play_one_game function
        return a


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
        self.critic = CriticNetwork(input_dims=self.state_dims, n_actions=self.action_dims)
        self.target_actor = ActorNetwork(input_dims=self.state_dims, n_actions=self.action_dims)
        self.target_critic = CriticNetwork(input_dims=self.state_dims, n_actions=self.action_dims)

        self.update_target_networks(tau=1.0)

    def get_action(self, state, training=True):
        state = tf.constant([state], dtype=tf.float32)
        action = self.actor.forward(state, training=False)
        if training:
            action += tf.random.normal(shape=(1, self.action_dims), mean=0.0, stddev=0.1)
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
            target_actions = self.target_actor.forward(new_states)
            target = rewards + self.gamma * (1.0 - dones) * self.target_critic.forward(new_states, target_actions)
            critic_value = self.critic.forward(states, actions)
            critic_loss = tf.reduce_mean(MSE(y_true=target, y_pred=critic_value))  # todo Remove reduce_mean to see its effects
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actor_loss = -tf.reduce_mean(self.critic.forward(states, self.actor.forward(states)))
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        self.update_target_networks()

    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        variables = []
        for i in range(len(self.target_critic.trainable_variables)):
            variables.append(tau*self.critic.trainable_variables[i] +
                             (1.0 - tau)*self.target_critic.trainable_variables[i])
        self.target_critic.set_weights(variables)

        variables = []
        for i in range(len(self.target_actor.trainable_variables)):
            variables.append(tau*self.actor.trainable_variables[i] +
                             (1.0 - tau)*self.target_actor.trainable_variables[i])
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
    env = gym.make('LunarLanderContinuous-v2')
    memory_size = 1000000
    tau = 0.001
    gamma = 0.99
    batch_size = 64
    agent = Agent(state_dims=env.observation_space.shape[0], action_dims=env.action_space.shape[0],
                  action_max=env.action_space.high, mem_size=memory_size,
                  tau=tau, gamma=gamma, batch_size=batch_size)  # Change values of hyperparameters if you want

    if training:
        # Training the agent
        n_iterations = 1500
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
        plt.plot(np.ones(n_iterations)*200, 'r-')
        legend_2 = 'Running average of the last 100 episodes (' + '%.2f' % np.mean(reward_history[-100:]) + ')'
        plt.legend(['Reward', legend_2, 'Reward of 200'], loc=4)
        plt.show()
        plt.savefig('Section 4 - DDPG/LunarLanderContinuous-v2_TF/Rewards_LunarLander')

        # Saving the networks
        agent.critic.save('Section 4 - DDPG/LunarLanderContinuous-v2_TF/critic_LunarLander')
        agent.actor.save('Section 4 - DDPG/LunarLanderContinuous-v2_TF/actor_LunarLander')
        agent.target_critic.save('Section 4 - DDPG/LunarLanderContinuous-v2_TF/target_critic_LunarLander')
        agent.target_actor.save('Section 4 - DDPG/LunarLanderContinuous-v2_TF/target_actor_LunarLander')

    else:
        # Loading networks
        agent.critic = tf.keras.models.load_model('Section 4 - DDPG/LunarLanderContinuous-v2_TF/critic_LunarLander')
        agent.actor = tf.keras.models.load_model('Section 4 - DDPG/LunarLanderContinuous-v2_TF/actor_LunarLander')
        agent.target_critic = tf.keras.models.load_model('Section 4 - DDPG/LunarLanderContinuous-v2_TF/'
                                                         'target_critic_LunarLander')
        agent.target_actor = tf.keras.models.load_model('Section 4 - DDPG/LunarLanderContinuous-v2_TF/'
                                                        'target_actor_LunarLander')

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

