import os

import numpy as np
import pandas as pd

def add_bias(X: np.array) -> np.array:
    m = len(X)
    return np.append(np.ones((m, 1)), X, axis=1)

def load_data_as_dataframe(name: str):
    testdata_abspath = '{}/data/{}'.format(
        os.path.dirname(os.path.abspath(__file__)), name)
    df = pd.read_csv(testdata_abspath, header=None)
    return df

def load_data(name: str, without_y: bool=False):
    df = load_data_as_dataframe(name)
    n = len(df.ix[0])
    X = df.ix[:, 0:n-2].values
    y = df.ix[:, n-1:n-1].values
    return add_bias(X), y
