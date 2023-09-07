import numpy as np
import pandas as pd
import scipy.linalg as sla
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def logits(x, w):
    return x @ w


def sigmoid(alpha):
    return 1 / (1 + np.exp(alpha))


class MyLogisticRegression:
    def __init__(self):
        self.w = None

    def fit(self, X_train, y, max_iter=100, lr=0.01):
        n, k = X_train.shape
        X_train = np.concatenate((np.ones((n, 1)), X_train), axis=1)

        if self.w is None:
            self.w = np.random.randn(k + 1)

        losses = []

        for i in range(max_iter):
            pred = sigmoid(logits(X_train, self.w))

            loss = self.__loss(y, pred)

            losses.append(loss)

            grad = X_train.T @ (pred - y)

            self.w -= lr * grad

        return losses

    def predict_proba(self, X):
        n, k = X.shape
        X = np.concatenate((np.ones((n, 1)), X), axis=1)
        return sigmoid(logits(X, self.w))

    def predict(self, X, treshold=0.5):
        return self.predict_proba(X) >= treshold

    def __loss(self, y, p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def get_weights(self):
        return self.w


clf = MyLogisticRegression()
X, y = make_blobs(n_samples=1000, centers=[[-2, 0.5], [2, -0.5]], cluster_std=1, random_state=42)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

all_losses = clf.fit(X_train, y_train)

w = clf.get_weights()



colors = ("red", "green")
colored_y = np.zeros(y.size, dtype=str)

for i, cl in enumerate([1, 0]):
    colored_y[y == cl] = str(colors[i])

plt.figure(figsize=(15,8))

eps = 0.1

xx, yy = np.meshgrid(np.linspace(np.min(X[:,0]) - eps, np.max(X[:,0]) + eps, 500),
                     np.linspace(np.min(X[:,1]) - eps, np.max(X[:,1]) + eps, 500))

print(xx.shape, yy.shape)

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

plt.scatter(X[:, 0], X[:, 1], c=colored_y)

plt.show()