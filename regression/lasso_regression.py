import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from matplotlib.colors import ListedColormap

from linear_regression import MySGDLinearRegression, linear_expression


def soft_sign(w, eps=1e-7):
    if abs(w) > eps:
        return np.sign(w)
    return w / eps


np_soft_sign = np.vectorize(soft_sign)


class MySGDLassoRegression(MySGDLinearRegression):
    def __init__(self, alpha=1, **kwargs):
        super().__init__(**kwargs)
        self.w = None
        self.alpha = alpha

    def _calc_gradient(self, X, y, y_pred):
        inds = np.random.choice(np.arange(X.shape[0]), size=self.n_sample, replace=False)

        sign_w = np_soft_sign(self.w)

        if self.fit_intercept:
            sign_w[-1] = 0

        grad = 1 / len(inds) * X.T @ (y_pred - y)
        grad += self.alpha * sign_w

        return grad


objects_num = 50
X = np.linspace(-5, 5, objects_num)
y = linear_expression(X) + np.random.randn(objects_num) * 5

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)

regressor = MySGDLassoRegression(alpha=1, n_sample=20).fit(X[:, np.newaxis], y, max_iter=1000, lr=0.01)
l = regressor.get_losses()
regressor.get_weights()

plt.figure(figsize=(10, 7))
plt.plot(X, linear_expression(X), label='real', c='g')

plt.scatter(X_train, y_train, label='train')
plt.scatter(X_test, y_test, label='test')
plt.plot(X, regressor.predict(X[:, np.newaxis]), label='predicted', c='r')

plt.grid(alpha=0.2)
plt.legend()
plt.show()

plt.figure(figsize=(10, 7))

plt.plot(l)

plt.title('Lasso learning with SGD')
plt.ylabel('loss')
plt.xlabel('iteration')
plt.grid(alpha=0.2)
plt.show()