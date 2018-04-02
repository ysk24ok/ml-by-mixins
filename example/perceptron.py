import time

import numpy as np
from scipy.optimize import check_grad
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

from mlbymxn.glm import Perceptron as PerceptronMxn
from mlbymxn.glm import PerceptronScipy
from mlbymxn.svm import LinearCSVMbyScipy, LinearCSVMbySGD, LinearCSVMbyAdam
from mlbymxn.utils import add_bias

data = datasets.load_breast_cancer()
X, y = data.data, data.target
X_with_bias = add_bias(X)
np.place(y, y==0, -1)   # convert label 0 to 1

X_train, X_test, X_bias_train, X_bias_test, y_train, y_test = \
    train_test_split(X, X_with_bias, y)
initial_theta = np.random.randn(X_bias_train.shape[1])
# mlbymxn's Perceptron
s = time.time()
c = PerceptronMxn()
c.theta = np.copy(initial_theta)
c.fit(X_bias_train, y_train)
grad_approx = check_grad(c.loss_function, c.gradient, c.theta, X_bias_test, y_test)
score = accuracy_score(y_test, c.predict(c.theta, X_bias_test))
print("mlbymxn's Perceptron       time: {:.2f}s, score: {:.2%}, grad: {:e}".format(
    time.time() - s, score, grad_approx))
# mlbymxn's PerceptronScipy
s = time.time()
c = PerceptronScipy(max_iters=10000)
c.fit(X_train, y_train)
grad_approx = check_grad(c.loss_function, c.gradient, c.theta, X, y)
score = accuracy_score(y_test, c.predict(c.theta, X_test))
print("mlbymxn's PerceptronScipy time: {:.2f}s, score: {:.2%}, grad: {:e}".format(
    time.time() - s, score, grad_approx))
# sklearn's Perceptron
s = time.time()
c = Perceptron()
c.fit(X_train, y_train)
score = accuracy_score(y_test, c.predict(X_test))
print("sklearn's Perceptron       time: {:.2f}s, score: {:.2%}".format(
    time.time() - s, score))
# mlbymxn's LinearCSVMbyScipy
s = time.time()
c = LinearCSVMbyScipy(max_iters=10000)
c.theta = np.copy(initial_theta)
c.fit(X_bias_train, y_train)
grad_approx = check_grad(c.loss_function, c.gradient, c.theta, X_bias_test, y_test)
score = accuracy_score(y_test, c.predict(c.theta, X_bias_test))
print("mlbymxn's LinearCSVMbyScipy  time: {:.2f}s, score: {:.2%}, grad: {:e}".format(
    time.time() - s, score, grad_approx))
# mlbymxn's LinearCSVMbySGD
s = time.time()
c = LinearCSVMbySGD()
c.theta = np.copy(initial_theta)
c.fit(X_bias_train, y_train)
grad_approx = check_grad(c.loss_function, c.gradient, c.theta, X_bias_test, y_test)
score = accuracy_score(y_test, c.predict(c.theta, X_bias_test))
print("mlbymxn's LinearCSVMbySGD  time: {:.2f}s, score: {:.2%}, grad: {:e}".format(
    time.time() - s, score, grad_approx))
# mlbymxn's LinearCSVMbyAdam
s = time.time()
c = LinearCSVMbyAdam()
c.theta = np.copy(initial_theta)
c.fit(X_bias_train, y_train)
grad_approx = check_grad(c.loss_function, c.gradient, c.theta, X_bias_test, y_test)
score = accuracy_score(y_test, c.predict(c.theta, X_bias_test))
print("mlbymxn's LinearCSVMbyAdam time: {:.2f}s, score: {:.2%}, grad: {:e}".format(
    time.time() - s, score, grad_approx))
