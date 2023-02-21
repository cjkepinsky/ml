import pandas as pd
from sklearn.model_selection import train_test_split


def load_train_test(x_path, y_path, train_size=0.7):
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)
    return train_test_split(X, y, train_size=train_size)
