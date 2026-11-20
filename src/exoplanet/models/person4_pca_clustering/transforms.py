"""Person 4 (Nathan Pak) preprocessing helpers for PCA and K-Means.

Many KOI measurements (period, depth, insolation, centroid offsets and their
uncertainties) are heavy-tailed. Standardizing them directly lets a few extreme
rows dominate the principal components and produces K-Means clusters with only
one or two members. ``SignedLog1pSkewed`` compresses those columns with
sign(x) * log1p(|x|) before imputation and scaling.

Which columns are transformed is learned in ``fit`` from the training rows only,
so the transformer is safe to place inside a cross-validation pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from exoplanet.preprocess import make_preprocess_pipeline


class SignedLog1pSkewed(BaseEstimator, TransformerMixin):
    """Apply sign(x) * log1p(|x|) to columns whose |skewness| exceeds a threshold."""

    def __init__(self, skew_threshold: float = 3.0):
        self.skew_threshold = skew_threshold

    def fit(self, X, y=None):
        frame = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
        skew = frame.skew(skipna=True).abs()
        self.columns_ = [c for c in frame.columns if skew.get(c, 0.0) > self.skew_threshold]
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").copy()
        if self.columns_:
            values = frame[self.columns_]
            frame[self.columns_] = np.sign(values) * np.log1p(values.abs())
        return frame

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def make_pca_preprocess(*, log_skewed: bool = True, skew_threshold: float = 3.0) -> Pipeline:
    """Optional signed-log step, then the shared median imputer and StandardScaler."""
    shared = make_preprocess_pipeline(scale=True).steps
    if not log_skewed:
        return Pipeline(steps=shared)
    return Pipeline(steps=[("signed_log", SignedLog1pSkewed(skew_threshold)), *shared])
