"""Feature transforms used by the Naive Bayes variants.

Both transforms are placed inside a Pipeline so they are fit on the
training fold only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def signed_log1p(x):
    """sign(x) * log(1 + |x|).

    Keeps the sign, is defined at zero, and compresses the heavy right tails
    of depth, SNR, period, insolation and the uncertainty columns. Gaussian
    Naive Bayes assumes each feature is normal within a class, and these
    columns are far from normal on the raw scale.
    """
    return np.sign(x) * np.log1p(np.abs(x))


class SpearmanPruner(BaseEstimator, TransformerMixin):
    """Drop one member of each feature pair with |Spearman r| above a threshold.

    The correlation matrix is computed on the training fold only. The walk is
    greedy and deterministic: columns are visited in their given order and a
    column is kept only if it is not too correlated with any column already
    kept. Naive Bayes assumes conditional independence, so near-duplicate
    columns get counted twice; this is one way to reduce that.
    """

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        # Pearson correlation of ranks equals Spearman correlation.
        corr = X.rank().corr(method="pearson").abs().fillna(0.0).to_numpy()
        keep: list[int] = []
        for j in range(corr.shape[0]):
            if all(corr[j, k] <= self.threshold for k in keep):
                keep.append(j)
        self.keep_idx_ = np.array(keep)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.iloc[:, self.keep_idx_].to_numpy()
