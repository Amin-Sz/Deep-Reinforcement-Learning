import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gym


class ActorNetwork(tf.keras.Model):
    def __init__(self, state_dims, n_actions, fc1_dims=256, fc2_dims=256, learning_rate=5e-5):
        super(ActorNetwork, self).__init__()
        self.state_dims = state_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr = learning_rate

        self.fc1 = Dense(units=self.fc1_dims, input_dim=self.state_dims, activation='relu')
        self.fc2 = Dense(units=self.fc2_dims, input_dim=self.fc1_dims, activation='relu')
        self.pi = Dense(units=self.n_actions, input_dim=self.fc2_dims, activation='softmax')

        self.optimizer = Adam(learning_rate=self.lr)

    def call(self, state, training=None, mask=None):
        output = self.fc1(state)
        output = self.fc2(output)
        probabilities = self.pi(output)
        distribution = tfp.distributions.Categorical(probs=probabilities)

        return distribution


class CriticNetwork(tf.keras.Model):
    def __init__(self, state_dims, fc1_dims=256, fc2_dims=256, learning_rate=5e-4):
        super(CriticNetwork, self).__init__()
        self.state_dims = state_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr = learning_rate

        self.fc1 = Dense(units=self.fc1_dims, input_dim=self.state_dims, activation='relu')
        self.fc2 = Dense(units=self.fc2_dims, input_dim=self.fc1_dims, activation='relu')
        self.v = Dense(units=1, input_dim=self.fc2_dims, activation='linear')

        self.optimizer = Adam(learning_rate=self.lr)

    def call(self, state, training=None, mask=None):
        output = self.fc1(state)
        output = self.fc2(output)
        value = self.v(output)

        return value


class Agent:
    def __init__(self, state_dims, n_actions, gamma):
        self.state_dims = state_dims
        self.n_actions = n_actions
        self.gamma = gamma

        self.actor = ActorNetwork(state_dims=self.state_dims, n_actions=self.n_actions)
        self.critic = CriticNetwork(state_dims=self.state_dims)

    def get_action(self, state):
        state = tf.constant([state], dtype=tf.float32)
        dist = self.actor.call(state)
        action = dist.sample()
        return action.numpy()

    def update_networks(self, state, action, reward, new_state, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action = tf.convert_to_tensor([action], dtype=tf.int32)
        action = tf.reshape(action, [-1, 1])
        reward = tf.convert_to_tensor([reward], dtype=tf.float32)
        reward = tf.reshape(reward, [-1, 1])
        new_state = tf.convert_to_tensor([new_state], dtype=tf.float32)
        done = tf.convert_to_tensor([done], dtype=tf.float32)
        done = tf.reshape(done, [-1, 1])

        with tf.GradientTape() as tape:
            value = self.critic.call(state)
            next_value = tf.stop_gradient(self.critic.call(new_state))
            delta = reward + self.gamma*next_value*(1.0 - done) - value
            critic_loss = tf.pow(delta, 2)
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            distribution = self.actor.call(state)
            log_prob = distribution.log_prob(tf.reshape(action, -1))
            log_prob = tf.reshape(log_prob, [-1, 1])
            delta = tf.stop_gradient(delta)
            actor_loss = -tf.multiply(delta, log_prob)
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))


def play_one_episode(agent, env):
    observation = env.reset()
    done = False
    total_reward = 0.0
    counter = 0

    while not done:
        a = agent.get_action(observation)
        prev_observation = observation
        observation, reward, done, info = env.step(a[0])
        total_reward += reward
        counter += 1

        if counter >= env._max_episode_steps:
            agent.update_networks(state=prev_observation, action=a[0], reward=reward, new_state=observation, done=False)
        else:
            agent.update_networks(state=prev_observation, action=a[0], reward=reward, new_state=observation, done=done)

    return agent, total_reward


def main(training):
    env = gym.make('CartPole-v0')
    solved_score = 200
    dir_ = 'Section 1 - Actor-Critic/CartPole-v0'
    gamma = 0.99
    agent = Agent(state_dims=env.observation_space.shape[0], n_actions=env.action_space.n, gamma=gamma)

    if training:
        reward_set = []
        avg_score_set = []
        n_iteration = 5000
        for t in range(n_iteration):
            agent, total_reward = play_one_episode(agent, env)

            reward_set.append(total_reward)
            avg_score_set.append(np.mean(reward_set[-100:]))
            print('iteration #' + str(t + 1) + ' -----> ' +
                  'total reward:' + '%.2f' % total_reward +
                  ', average score:' + '%.2f' % np.mean(reward_set[-100:]))

            if np.mean(reward_set[-100:]) >= solved_score:
                print('-----  environment was solved after ' + str(t + 1) + ' episodes, ' +
                      'average score:' + '%.2f' % np.mean(reward_set[-100:]) + '  -----')
                break

        # Plotting the learning curve
        axes = plt.axes()
        axes.set_ylim([np.min(reward_set) - 50, np.max(reward_set) + 10])
        plt.xlabel('Episode')
        plt.ylabel('Total reward')
        plt.plot(np.arange(1, len(reward_set) + 1), reward_set)
        plt.plot(np.arange(1, len(reward_set) + 1), avg_score_set)
        legend_2 = 'Running average of the last 100 total rewards (' + '%.2f' % np.mean(reward_set[-100:]) + ')'
        plt.legend(['Total reward', legend_2], loc=4)
        plt.show()
        plt.savefig(dir_ + '/Rewards_CartPole-v0')

        # Saving the networks
        agent.actor.save_weights(dir_ + '/actor')
        agent.critic.save_weights(dir_ + '/critic')

    else:
        # Loading networks
        agent.actor.load_weights(dir_ + '/actor').expect_partial()
        agent.critic.load_weights(dir_ + '/critic').expect_partial()

        # Showing the video
        for t in range(5):
            observation = env.reset()
            done = False
            total_reward = 0
            while not done:
                env.render()
                a = agent.get_action(observation)
                observation, reward, done, info = env.step(a[0])
                total_reward = total_reward + reward
            print('video #' + str(t + 1) + ' ----> total reward:' + '%.2f' % total_reward)
        env.close()


if __name__ == '__main__':
    main(training=False)

