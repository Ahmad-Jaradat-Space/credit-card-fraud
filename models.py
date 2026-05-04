"""From-scratch numpy implementations.

Two models:
- LogisticRegressionWeighted: binary logistic regression with mini-batch
  SGD, L2, and per-class loss weights (so the few positives don't get
  drowned out by the many negatives).
- GaussianAnomalyDetector: one Gaussian per feature, fit on the negative
  class only; score = sum of feature log-densities. Unsupervised baseline.
"""

import numpy as np


def _sigmoid(z):
    # split form keeps things stable for large |z|
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


class LogisticRegressionWeighted:
    """Binary logistic regression, mini-batch SGD, L2, per-class weights."""

    def __init__(self, lr=0.1, l2=1e-4, epochs=20, batch=512, seed=0,
                 class_weight="balanced"):
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        self.batch = batch
        self.rng = np.random.default_rng(seed)
        self.class_weight = class_weight
        self.w = None
        self.b = None
        self.history = []

    def _weights(self, y):
        if self.class_weight == "balanced":
            n = len(y)
            n_pos = max(int(y.sum()), 1)
            n_neg = n - n_pos
            return np.where(y == 1, n / (2 * n_pos), n / (2 * n_neg)).astype(np.float32)
        return np.ones_like(y, dtype=np.float32)

    def fit(self, X, y, X_val=None, y_val=None):
        n, d = X.shape
        self.w = self.rng.normal(0, 0.01, size=d).astype(np.float32)
        self.b = np.float32(0.0)
        w_per_sample = self._weights(y)

        for epoch in range(self.epochs):
            idx = self.rng.permutation(n)
            for start in range(0, n, self.batch):
                b = idx[start:start + self.batch]
                Xb, yb, wb = X[b], y[b].astype(np.float32), w_per_sample[b]
                p = _sigmoid(Xb @ self.w + self.b)
                err = (p - yb) * wb
                grad_w = Xb.T @ err / len(b) + self.l2 * self.w
                grad_b = err.mean()
                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b
            entry = {"epoch": epoch, "train_loss": self._loss(X, y, w_per_sample)}
            if X_val is not None:
                wv = self._weights(y_val)
                entry["val_loss"] = self._loss(X_val, y_val, wv)
            self.history.append(entry)
        return self

    def _loss(self, X, y, w):
        p = _sigmoid(X @ self.w + self.b).clip(1e-7, 1 - 1e-7)
        ll = y * np.log(p) + (1 - y) * np.log(1 - p)
        return float(-(w * ll).mean())

    def predict_proba(self, X):
        return _sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


class GaussianAnomalyDetector:
    """Fit a univariate Gaussian per feature on the negative class.

    Score: sum of log-densities (higher = more 'normal'). Anomalies have
    low scores. The threshold (epsilon) is picked on a labelled validation
    set by maximising F1.
    """

    def __init__(self, eps_var=1e-6):
        self.mu = None
        self.var = None
        self.eps_var = eps_var  # floor to keep variance from collapsing
        self.epsilon = None

    def fit(self, X_neg):
        self.mu = X_neg.mean(axis=0)
        self.var = X_neg.var(axis=0) + self.eps_var
        return self

    def score(self, X):
        # log N(x; mu, var) summed across features, ignoring the constant
        diff = X - self.mu
        return -0.5 * (np.log(2 * np.pi * self.var) + diff ** 2 / self.var).sum(axis=1)

    def select_epsilon(self, scores, y_true):
        """Sweep candidate thresholds, pick the one maximising F1."""
        order = np.argsort(scores)
        candidates = np.linspace(scores.min(), scores.max(), 200)
        best_f1, best_eps = -1.0, candidates[0]
        for eps in candidates:
            pred = (scores < eps).astype(int)
            tp = int(((pred == 1) & (y_true == 1)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            fn = int(((pred == 0) & (y_true == 1)).sum())
            if tp == 0:
                continue
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = 2 * prec * rec / (prec + rec)
            if f1 > best_f1:
                best_f1, best_eps = f1, eps
        self.epsilon = float(best_eps)
        return self.epsilon, best_f1

    def predict(self, X):
        if self.epsilon is None:
            raise ValueError("call select_epsilon on a labelled validation set first")
        return (self.score(X) < self.epsilon).astype(int)
