import numpy as np
import os

#from keras.models import Sequential
#from keras.layers import Dense
#from keras.utils import to_categorical

from data_utilities import load_data_and_normalize, load_data

if __name__ == "__main__":
    cwd = os.getcwd()
    data_dir = cwd + os.path.sep + ".." + os.path.sep + ".." + os.path.sep + "data" + os.path.sep + "poker"
    test_data_file_location = data_dir + os.path.sep + "poker-hand-testing.data"
    train_data_file_location = data_dir + os.path.sep + "poker-hand-training-true.data"

    train_features, train_targets = load_data(test_data_file_location)
    test_features, test_targets = load_data(test_data_file_location)

    print("train_features", train_features.shape, train_features)
    print("train_targets", train_targets.shape, train_targets)

#    encoded_targets = to_categorical(train_targets)
#    print("encoded_targets", encoded_targets.shape, encoded_targets)

    print("test_features", test_features.shape, test_features)

    print("test_targets", test_targets.shape, test_targets)

    """
    model = Sequential()

    # Input Layer
    model.add(Dense(, input_shape(train_features.shape[1],), activation='tanh'))

    # Hidden layer - Dense 52
    model.add(Dense(52, activation='tanh'))

    # Hidden layer - Dense 13
    model.add(Dense(13, activation='tanh'))

    # Output layer
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_features, train_)
    """
