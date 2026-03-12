import numpy as np

def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z))
    )

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)   # shape: (m,)

    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=float)        # shape: (n_features,)
    b = 0.0

    for _ in range(steps):
        z = X @ w + b                            # shape: (m,)
        pred = _sigmoid(z)                       # shape: (m,)

        dw = (X.T @ (pred - y)) / n_samples      # shape: (n_features,)
        db = np.sum(pred - y) / n_samples        # scalar

        w -= lr * dw
        b -= lr * db

    return w, b