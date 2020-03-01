import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

def load_data_and_normalize(file_location, column_headers=['card_1_suit', 'card_1_value', 'card_2_suit', 'card_2_value', 'card_3_suit', 'card_3_value', 'card_4_suit', 'card_4_value', 'card_5_suit', 'card_5_value', 'hand class']):
    # Load data
    data = pd.read_csv(file_location, sep=",", header=None, names=column_headers)
    features = data[data.columns[:-1]]
    targets = pd.DataFrame(to_one_hot(data[data.columns[-1:]], dimension=10), columns=["nothing", "one_pair", "two_pair", "three_of_a_kind", "straight", "flush", "full_house", "four_of_a_kind", "straight_flush", "royal_flush"])

    # Normalize features
    normalizer = preprocessing.MinMaxScaler()
    normalzied_features = normalizer.fit_transform(features)
    features = pd.DataFrame(normalzied_features, columns=column_headers[:-1])

    return features, targets

def load_data(file_location, column_headers=['card_1_suit', 'card_1_value', 'card_2_suit', 'card_2_value', 'card_3_suit', 'card_3_value', 'card_4_suit', 'card_4_value', 'card_5_suit', 'card_5_value', 'hand class']):
    # Load data
    data = pd.read_csv(file_location, sep=",", header=None, names=column_headers)
    features = data[data.columns[:-1]]
    targets = pd.DataFrame(to_one_hot(data[data.columns[-1:]], dimension=10), columns=["nothing", "one_pair", "two_pair", "three_of_a_kind", "straight", "flush", "full_house", "four_of_a_kind", "straight_flush", "royal_flush"])

    return features, targets

def to_one_hot(labels, dimension):
    results = np.zeros((len(labels), dimension))
    for label_index, label in enumerate(labels.to_numpy()):
        results[label_index, label] = 1.
    return results
