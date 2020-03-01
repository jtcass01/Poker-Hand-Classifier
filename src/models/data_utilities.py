import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

def load_data_and_normalize(file_location, column_headers=['card_1_suit', 'card_1_value', 'card_2_suit', 'card_2_value', 'card_3_suit', 'card_3_value', 'card_4_suit', 'card_4_value', 'card_5_suit', 'card_5_value', 'hand class']):
    # Load data
    data = pd.read_csv(file_location, sep=",", header=None, names=column_headers)
    features = data[data.columns[:-1]]
    targets = data[data.columns[-1:]]

    # Normalize features
    normalizer = preprocessing.MinMaxScaler()
    normalzied_features = normalizer.fit_transform(features)
    features = pd.DataFrame(normalzied_features, columns=column_headers[:-1])

    return features, targets

def load_data(file_location, column_headers=['card_1_suit', 'card_1_value', 'card_2_suit', 'card_2_value', 'card_3_suit', 'card_3_value', 'card_4_suit', 'card_4_value', 'card_5_suit', 'card_5_value', 'hand class']):
    # Load data
    data = pd.read_csv(file_location, sep=",", header=None, names=column_headers)
    features = data[data.columns[:-1]]
    targets = data[data.columns[-1:]]

    return features, targets
