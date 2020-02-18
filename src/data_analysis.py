import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

def load_data_and_normalize(file_location, title="", column_headers=['card_1_suit', 'card_1_value', 'card_2_suit', 'card_2_value', 'card_3_suit', 'card_3_value', 'card_4_suit', 'card_4_value', 'card_5_suit', 'card_5_value', 'hand class']):
    # Load data
    data = pd.read_csv(file_location, sep=",", header=None, names=column_headers)
    features = data[data.columns[:-1]]
    targets = data[data.columns[-1:]]

    # Normalize features
    normalizer = preprocessing.MinMaxScaler()
    normalzied_features = normalizer.fit_transform(features)
    features = pd.DataFrame(normalzied_features, columns=column_headers[:-1])

    ax = targets.plot.hist(bins=10, alpha=0.5, title=title)
    plt.show()

    return features, targets

if __name__ == "__main__":
    cwd = os.getcwd()
    data_dir = cwd + os.path.sep + ".." + os.path.sep + "data" + os.path.sep + "poker"
    test_data_file_location = data_dir + os.path.sep + "poker-hand-testing.data"
    train_data_file_location = data_dir + os.path.sep + "poker-hand-training-true.data"


    test_features, test_targets = load_data_and_normalize(test_data_file_location, title="Test set Target Distribution")
    train_features, train_targets = load_data_and_normalize(train_data_file_location, title="Train set Target Distribution")

    print("train_data")
    print("features", test_features)
    print("targets", test_targets)

    print("test_data")
    print("features", train_features)
    print("targets", train_targets)
