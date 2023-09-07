import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from matplotlib.colors import ListedColormap

from linear_regression import MySGDLinearRegression, linear_expression


class MySGDRidgeRegression(MySGDLinearRegression):
    def __init__(self, alpha=1, **kwargs):
        super().__init__(**kwargs)
        self.w = None
        self.alpha = alpha

    def _calc_gradient(self, X, y, y_pred):
        inds = np.random.choice(np.arange(X.shape[0]), size=self.n_sample, replace=False)

        lambaI = self.alpha * np.eye(self.w.shape[0])

        if self.fit_intercept:
            lambaI[-1][-1] = 0

        grad = 2 * (X[inds].T @ X[inds] / self.n_sample + lambaI) @ self.w
        grad -= 2 * X[inds].T @ y[inds] / self.n_sample

        return grad

objects_num = 50
X = np.linspace(-5, 5, objects_num)
y = linear_expression(X) + np.random.randn(objects_num) * 5

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)


regressor = MySGDRidgeRegression(alpha=1, n_sample=20).fit(X[:, np.newaxis], y, max_iter=1000, lr=0.01)
l = regressor.get_losses()
regressor.get_weights()
