import tensorflow as tf
from keras.layers import Dense


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





