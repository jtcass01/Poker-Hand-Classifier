import os

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

from data_utilities import load_data_and_normalize, load_data

class DNN(object):
    def __init__(self, feature_size, hidden_layer_dimensions, hidden_layer_activation_function='relu', optimizer='adam'):
        self.model = Sequential()

        for layer_index, layer_dimension in enumerate(hidden_layer_dimensions):
            if layer_index == 0:
                self.model.add(Dense(layer_dimension, input_shape=(feature_size,), activation=hidden_layer_activation_function))
            else:
                self.model.add(Dense(layer_dimension, activation=hidden_layer_activation_function))
        self.model.add(Dense(10, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.history = None

    def train(self, train_features, train_targets, test_features, test_targets, epochs, batch_size):
        self.history = self.model.fit(train_features, train_targets, epochs=epochs, batch_size=batch_size, validation_data=(test_features, test_targets))

    def plot_training_and_valdiation_loss(self):
        history_dict = self.history.history
        training_loss_values = history_dict['loss']
        validation_loss_values = history_dict['val_loss']

        epochs = range(1, len(training_loss_values) + 1)

        plt.plot(epochs, training_loss_values, 'bo', label='Training loss')
        plt.plot(epochs, validation_loss_values, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    cwd = os.getcwd()
    data_dir = cwd + os.path.sep + ".." + os.path.sep + ".." + os.path.sep + "data" + os.path.sep + "poker"
    test_data_file_location = data_dir + os.path.sep + "poker-hand-testing.data"
    train_data_file_location = data_dir + os.path.sep + "poker-hand-training-true.data"

    train_features, train_targets = load_data(test_data_file_location)
    test_features, test_targets = load_data(test_data_file_location)

    model = DNN(train_features.shape[1], hidden_layer_dimensions=(52, 13, 4))
    model.train(train_features, train_targets, test_features, test_targets, epochs=25, batch_size=512)
    model.plot_training_and_valdiation_loss()
