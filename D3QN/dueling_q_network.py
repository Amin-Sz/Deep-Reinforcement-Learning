import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense


class DuelingQNetwork(Model):
    def __init__(self, n_actions):
        super(DuelingQNetwork, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=8, strides=4, activation='relu', data_format='channels_first')
        self.conv2 = Conv2D(filters=64, kernel_size=4, strides=2, activation='relu', data_format='channels_first')
        self.conv3 = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', data_format='channels_first')

        self.v_hidden = Dense(units=512, activation='relu')
        self.adv_hidden = Dense(units=512, activation='relu')
        self.v = Dense(units=1, activation='linear')
        self.adv = Dense(units=n_actions, activation='linear')

    def call(self, inputs, training=None, mask=None):
        cnn_output = self.conv1(inputs)
        cnn_output = self.conv2(cnn_output)
        cnn_output = self.conv3(cnn_output)
        cnn_output = Flatten()(cnn_output)

        v = self.v_hidden(cnn_output)
        v = self.v(v)

        adv = self.adv_hidden(cnn_output)
        adv = self.adv(adv)

        return v, adv


class DuelingQNetworkANN(Model):
    def __init__(self, n_actions):
        super(DuelingQNetworkANN, self).__init__()
        self.fc1 = Dense(units=128, activation='relu')
        self.fc2 = Dense(units=128, activation='relu')

        self.v = Dense(units=1, activation='linear')
        self.adv = Dense(units=n_actions, activation='linear')

    def call(self, inputs, training=None, mask=None):
        output = self.fc1(inputs)
        output = self.fc2(output)

        v = self.v(output)
        adv = self.adv(output)

        return v, adv


def _test():
    network = DuelingQNetwork(n_actions=5)
    x = tf.random.normal((1, 4, 84, 84))
    v, adv = network.call(x)
    n_trainable_params = tf.reduce_sum([tf.reduce_prod(l.shape) for l in network.trainable_weights])
    n_non_trainable_params = tf.reduce_sum([tf.reduce_prod(l.shape) for l in network.non_trainable_weights])

    print('shape of V:', v.shape)
    print('shape of adv:', adv.shape)
    print('number of trainable parameters:', n_trainable_params)
    print('number of non-trainable parameters:', n_non_trainable_params)


if __name__ == '__main__':
    _test()
