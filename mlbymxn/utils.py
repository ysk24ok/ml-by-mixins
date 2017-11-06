import os

import numpy as np
import pandas as pd


def add_bias(X):
    if len(X.shape) != 2:
        raise ValueError('X should be 2-dimentional array')
    m = X.shape[0]
    if isinstance(X, np.ndarray):
        X_with_bias = np.append(np.ones((m, 1)), X, axis=1)
    else:
        raise ValueError('The type of X is not supported.')
    return X_with_bias

def load_data_as_dataframe(name: str, header=None, columns=None):
    if header is None and columns is None:
        raise ValueError('Both header and columns are None')
    testdata_abspath = '{}/data/{}'.format(
        os.path.dirname(os.path.abspath(__file__)), name)
    df = pd.read_csv(testdata_abspath, header=header, names=columns)
    return df

def load_data(name: str):
    if name == 'ex1data1.txt':
        df = load_data_as_dataframe(name, columns=('x', 'y'))
        m, n = df.shape
        X = df['x'].values.reshape((m, n-1))
        y = df['y'].values
    elif name in ('ex2data1.txt', 'ex2data2.txt'):
        df = load_data_as_dataframe(name, columns=('x1', 'x2', 'y'))
        m, n = df.shape
        X = df[['x1', 'x2']].values.reshape((m, n-1))
        y = df['y'].values
    elif name == 'ex7data1.csv':
        df = load_data_as_dataframe(name, columns=('x1', 'x2'))
        return df.values
    else:
        raise ValueError('{} is not supported.'.format(name))
    return add_bias(X), y
