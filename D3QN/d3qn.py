import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

from replay_buffer import ReplayBuffer
from dueling_q_network import DuelingQNetwork, DuelingQNetworkANN


class Agent:
    def __init__(self, state_dims, n_actions, frame_as_input: bool,
                 memory_size, learning_rate, batch_size,
                 gamma, initial_eps, final_eps, final_eps_state,
                 update_frequency, target_update_frequency, scale_gradients):
        self.state_dims = state_dims
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.final_eps_state = final_eps_state
        self.update_frequency = update_frequency
        self.target_update_frequency = target_update_frequency
        self.scale = scale_gradients
        self.frame_counter = 0
        self.update_counter = 0

        self.replay_buffer = ReplayBuffer(mem_size=memory_size, state_dims=state_dims, action_dims=1)

        if frame_as_input:
            self.network = DuelingQNetwork(n_actions=n_actions)
            self.target_network = DuelingQNetwork(n_actions=n_actions)
        else:
            self.network = DuelingQNetworkANN(n_actions=n_actions)
            self.target_network = DuelingQNetworkANN(n_actions=n_actions)

        self._update_target_network()
        self.optimizer = Adam(learning_rate=learning_rate)

    def get_action(self, state, training=True):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        v, adv = self.network.call(state)
        q = self._get_q(v=v, adv=adv)
        action = tf.argmax(q, axis=1).numpy()[0]

        if training:
            eps = self._get_eps()
            self.frame_counter += 1
        else:
            eps = 0.001

        if eps > np.random.rand():
            return np.random.choice(self.n_actions)
        else:
            return action

    def update_networks(self):
        if self.frame_counter % self.update_frequency == 0:
            self.update_counter += 1

            states, actions, rewards, new_states, dones = self.replay_buffer.sample(batch_size=self.batch_size)
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)

            with tf.GradientTape() as tape:
                v_new_states, adv_new_states = self.network.call(new_states)
                q_new_states = self._get_q(v=v_new_states, adv=adv_new_states)
                target_actions = tf.argmax(q_new_states, axis=1)
                target_actions = tf.reshape(target_actions, (-1, 1))
                target_v, target_adv = self.target_network.call(new_states)
                target_q = self._get_q(v=target_v, adv=target_adv)
                target_q = tf.gather(target_q, indices=target_actions, axis=1, batch_dims=1)
                target = rewards + self.gamma * (1.0 - dones) * target_q
                target = tf.stop_gradient(target)

                v, adv = self.network.call(states)
                q = self._get_q(v=v, adv=adv)
                q = tf.gather(q, indices=actions, axis=1, batch_dims=1)
                loss = MeanSquaredError()(y_true=target, y_pred=q)

            q_gradients = tape.gradient(loss, self.network.trainable_variables)
            if self.scale:
                for idx in range(len(q_gradients)):
                    if idx == 4 or idx == 5:
                        q_gradients[idx] *= 1.0 / tf.sqrt(2.0)
                    q_gradients[idx] = tf.clip_by_value(q_gradients[idx], clip_value_min=-10.0, clip_value_max=10.0)

            self.optimizer.apply_gradients(zip(q_gradients, self.network.trainable_variables))

        if self.frame_counter % self.target_update_frequency == 0:
            self._update_target_network()

    def _get_q(self, v, adv):
        return v + (adv - tf.reduce_mean(adv, axis=1, keepdims=True))

    def _update_target_network(self):
        self.target_network.set_weights(self.network.get_weights())

    def _get_eps(self):
        eps = (self.final_eps - self.initial_eps) / self.final_eps_state * self.frame_counter + self.initial_eps
        return max(eps, self.final_eps)

    def save_networks(self, path):
        self.network.save_weights(path + '/network')
        self.target_network.save_weights(path + '/target_network')

    def load_networks(self, path):
        self.network.load_weights(path + '/network')
        self.target_network.load_weights(path + '/target_network')
