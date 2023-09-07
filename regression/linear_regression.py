import numpy as np
import pandas as pd
import scipy.linalg as sla
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class MyLinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X_train, y):
        n, k = X_train.shape
        print(X_train, np.ones((n, 1)))
        if self.fit_intercept:
            X_train = np.hstack((X_train, np.ones((n, 1))))
        self.w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y
        return self

    def predict(self, X_test):
        n, k = X_test.shape
        if self.fit_intercept:
            X_test = np.hstack((X_test, np.ones((n, 1))))
        return X_test @ self.w

    def get_weights(self):
        return self.w


class MyGradientLinearRegression(MyLinearRegression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = None

    def fit(self, X, y, lr=0.01, max_iter=100):

        n, k = X.shape

        if self.w is None:
            self.w = np.random.randn(k + 1 if self.fit_intercept else k)

        X_train = np.hstack((X, np.ones((n, 1)))) if self.fit_intercept else X

        self.losses = []

        for iter_num in range(max_iter):
            y_pred = self.predict(X)
            self.losses.append(mean_squared_error(y_pred, y))

            grad = self._calc_gradient(X_train, y, y_pred)

            assert grad.shape == self.w.shape, f"gradient shape {grad.shape} is not equal weight shape {self.w.shape}"
            self.w -= lr * grad

        return self

    def _calc_gradient(self, X, y, y_pred):
        grad = 2 * (y_pred - y)[:, np.newaxis] * X
        grad = grad.mean(axis=0)
        return grad

    def get_losses(self):
        return self.losses


class MySGDLinearRegression(MyGradientLinearRegression):
    def __init__(self, n_sample=10, **kwargs):
        super().__init__(**kwargs)
        self.w = None
        self.n_sample = n_sample

    def _calc_gradient(self, X, y, y_pred):
        inds = np.random.choice(np.arange(X.shape[0]), size=self.n_sample, replace=False)

        grad = 2 * (y_pred[inds] - y[inds])[:, np.newaxis] * X[inds]
        grad = grad.mean(axis=0)

        return grad


def linear_expression(x):
    return 5 * x + 6


num_objects = 50
X = np.linspace(-5, 5, num_objects)
y = linear_expression(X) + np.random.randn(num_objects) * 5

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# plt.figure(figsize=(10, 7))
# plt.plot(X, linear_expression(X), label='real', c='g')
# plt.scatter(X_train, y_train, label='train', c='b')
# plt.scatter(X_test, y_test, label='test', c='orange')
#
# plt.title("Generated dataset")
# plt.grid(alpha=0.2)
# plt.legend()
# plt.show()
#
# regressor = MySGDLinearRegression()
# regressor.fit(X_train[:, np.newaxis], y_train)
# predictions = regressor.predict(X_test[:, np.newaxis])
#
# plt.figure(figsize=(20, 7))
#
# ax = None
#
# for i, types in enumerate([['train', 'test'], ['train'], ['test']]):
#     ax = plt.subplot(1, 3, i + 1, sharey=ax)
#     if 'train' in types:
#         plt.scatter(X_train, y_train, label='train', c='b')
#     if 'test' in types:
#         plt.scatter(X_test, y_test, label='test', c='orange')
#
#     plt.plot(X, linear_expression(X), label='real', c='g')
#     plt.plot(X, regressor.predict(X[:, np.newaxis]), label='predicted', c='r')
#
#     plt.ylabel('target')
#     plt.xlabel('feature')
#     plt.title(" ".join(types))
#     plt.grid(alpha=0.2)
#     plt.legend()
#
# plt.show()
