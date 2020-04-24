import os

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model

from sklearn.metrics import confusion_matrix

from data_utilities import load_data_and_normalize, load_data



plot_font = {
    'family' : 'normal',
    'weight' : 'bold',
    'size' : 22
}
matplotlib.rc('font', **plot_font)

class DNN(object):
    def __init__(self, feature_size, hidden_layer_dimensions, hidden_layer_activation_function='relu', optimizer='adam'):
        self.model = Sequential()

        for layer_index, layer_dimension in enumerate(hidden_layer_dimensions):
            if layer_index == 0:
                self.model.add(Dense(layer_dimension, input_shape=(feature_size,), activation=hidden_layer_activation_function))
            else:
                self.model.add(Dense(layer_dimension, activation=hidden_layer_activation_function))
        self.model.add(Dense(10, activation='softmax'))
        plot_model(self.model, to_file="model_plot.png", show_shapes=True, show_layer_names=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.history = None

    def train(self, train_features, train_targets, test_features, test_targets, epochs, batch_size):
        self.history = self.model.fit(train_features, train_targets, epochs=epochs, batch_size=batch_size, validation_data=(test_features, test_targets))

    def plot_training_and_valdiation_loss(self, hidden_layer_dimensions):
        history_dict = self.history.history
        training_loss_values = history_dict['loss']
        validation_loss_values = history_dict['val_loss']

        epochs = range(1, len(training_loss_values) + 1)

        plt.plot(epochs, training_loss_values, 'bo', label='Training loss')
        plt.plot(epochs, validation_loss_values, 'b', label='Validation loss')
        plt.title('Training and validation loss for hidden_layer_dimensions=' + str(hidden_layer_dimensions))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

def plot_confusion_matrix(y_test, y_pred, classes=['0','1', '2', '3', '4', '5', '6', '7', '8', '9'], normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    cnf_matrix = confusion_matrix(y_test, y_pred)

    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix without normalization")

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.
    for row_index, column_index in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(column_index, row_index, format(cnf_matrix[row_index, column_index], fmt),
                 horizontalalignment='center',
                 color='white' if cnf_matrix[row_index, column_index] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    cwd = os.getcwd()
    data_dir = cwd + os.path.sep + ".." + os.path.sep + ".." + os.path.sep + "data" + os.path.sep + "poker"
    test_data_file_location = data_dir + os.path.sep + "poker-hand-testing.data"
    train_data_file_location = data_dir + os.path.sep + "poker-hand-training-true.data"

    train_features, train_targets = load_data_and_normalize(test_data_file_location)
    test_features, test_targets = load_data_and_normalize(train_data_file_location)

    model = DNN(train_features.shape[1], hidden_layer_dimensions=(52, 13, 4))
    model.train(train_features, train_targets, test_features, test_targets, epochs=26, batch_size=512)
    model.plot_training_and_valdiation_loss(hidden_layer_dimensions=(52, 13, 4))
    test_pred = model.model.predict(X_test)
    plot_confusion_matrix(test_targets, test_pred)

    model = DNN(train_features.shape[1], hidden_layer_dimensions=(15, 10))
    model.train(train_features, train_targets, test_features, test_targets, epochs=26, batch_size=512)
    model.plot_training_and_valdiation_loss(hidden_layer_dimensions=(15, 10))
    test_pred = model.model.predict(X_test)
    plot_confusion_matrix(test_targets, test_pred)

    model = DNN(train_features.shape[1], hidden_layer_dimensions=(25, 12, 6))
    model.train(train_features, train_targets, test_features, test_targets, epochs=26, batch_size=512)
    model.plot_training_and_valdiation_loss(hidden_layer_dimensions=(25, 12, 6))
    test_pred = model.model.predict(X_test)
    plot_confusion_matrix(test_targets, test_pred)
