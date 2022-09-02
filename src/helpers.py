# this scripts contains helper functions for run.py


import pandas as pd
import numpy as np
from pathlib import Path
import math

import sklearn.datasets

import math, copy

import time




##############################################################################################################
# define data cleaning fucns
##############################################################################################################

def load_data():
    housing = sklearn.datasets.fetch_california_housing()

    df = pd.DataFrame(data=np.c_[housing['data'], housing['target']],
                      columns=housing['feature_names'] + ['target'])
    df = df.drop(columns=['Latitude', 'Longitude'])

    return df


def train_test_split(df):
    n = len(df)

    # train test split (2/3 train, 1/3 test)
    n_train = round(2 / 3 * n)

    train_df = df[:n_train]
    test_df = df[n_train:]

    return train_df, test_df


def initial_rand(X):
    np.random.seed(1)

    m = X.shape[0]
    n = X.shape[1]

    w = np.random.randn(n).reshape(n, 1) * 0.01
    b = np.random.randint(0, 100) * 0.01

    return w, b


def initial_zeros(X):
    np.random.seed(1)

    # m = number of training examples
    m = X.shape[0]

    # n = number of features
    n = X.shape[1]

    w = np.zeros(n).reshape(n, 1).T
    b = 0

    return w, b


def set_train_vars(X_df):
    # m = number of training examples
    m = X_df.values.shape[0]

    # n = number of features
    n = len(X_df.drop(columns='target').columns)

    # X should be a matrix of with m (number training examples) rows and n (number features) columns
    X = X_df.drop(columns='target').values.reshape(m, n)

    # Y should be a matrix with 1 row and n columns
    Y = X_df['target'].values.reshape(1, m)

    return X, Y, m, n


##############################################################################################################
# define gradient descent functions
##############################################################################################################


def forward_prop(X, w, b):
    n = X.shape[0]
    # reshape step important for later functions
    Y_hat = np.dot(w, X.T) + b

    return Y_hat


def calculate_cost(X, Y, w, b):
    m = X.shape[0]
    Y_hat = forward_prop(X, w, b)
    cost = np.sum((Y_hat - Y) ** 2) / (2 * m)
    return cost


def calculate_grads(X, Y, w, b):
    m, n = X.shape
    Y_hat = forward_prop(X, w, b)
    db = np.mean(Y_hat - Y)
    dw = np.sum(((Y_hat - Y) * X.T), axis=1) / m
    return db, dw


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha

    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent

    Returns:
      w (ndarray (n,)) : Updated values of parameters
      b (scalar)       : Updated value of parameter
      """

    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in

    start = time.time()
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w, b)  ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw  ##None
        b = b - alpha * dj_db  ##None

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
    end = time.time()
    train_time = end - start

    return w, b, J_history, train_time  # return final w,b and J history for graphing


#######################################################################################################################
# testing funcs
#######################################################################################################################


def set_test_vars(X_df):
    m, n = X_df.shape
    X_test = X_df.drop(columns=['target']).values
    Y_test = X_df['target'].values

    return X_test, Y_test, m, n


def predict(X_in, w_arr, b_val):
    Y_hat = np.dot(w_arr, X_in.T) + b_val
    return Y_hat


def rmse(Y, Y_hat):
    return np.sqrt(np.mean((Y_hat - Y) ** 2))


def mape(Y, Y_hat):
    return np.mean((Y - Y_hat) / Y_hat)


##############################################################################################################
# def feature engineering funcs
##############################################################################################################


def scale_features(X):
    max_features = np.array([np.max(X.T[:][i]) for i in range(X.shape[1])])
    scaled_features = np.array([X.T[:][0] / max_features[i] for i in range(X.shape[1])]).T

    return max_features, scaled_features
