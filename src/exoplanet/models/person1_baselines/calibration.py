"""Probability calibration nested inside each training fold.

``CalibratedClassifierCV`` is given inner folds that are grouped by host star
and built only from the outer training fold. The outer validation fold is
never used to fit either the classifier or the calibrator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedGroupKFold

from exoplanet.config import RANDOM_SEED as SEED

N_CALIBRATION_FOLDS = 4


def grouped_inner_cv(y_train, groups, *, n_splits: int = N_CALIBRATION_FOLDS, seed: int = SEED):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y_train)), np.asarray(y_train), np.asarray(groups)))


def make_calibrated(base_factory, method: str, groups, y_train):
    """Calibrated classifier whose inner folds come from the training rows only.

    ``base_factory`` is a zero-argument callable returning a fresh unfitted
    classifier Pipeline.

    With ensemble=True each inner split fits one classifier on three quarters
    of the training fold and one calibrator on the remaining quarter; the
    prediction is the mean of the four calibrated classifiers.
    """
    return CalibratedClassifierCV(
        base_factory(),
        method=method,
        cv=grouped_inner_cv(y_train, groups),
        ensemble=True,
    )


def reliability_bins(y_true_binary, p_pos, *, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width reliability table with counts per bin."""
    y = np.asarray(y_true_binary, dtype=int)
    p = np.asarray(p_pos, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        rows.append(
            {
                "bin": b,
                "bin_low": edges[b],
                "bin_high": edges[b + 1],
                "count": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()) if mask.any() else np.nan,
                "observed_fraction": float(y[mask].mean()) if mask.any() else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    # Cross-check against scikit-learn on the non-empty bins.
    prob_true, prob_pred = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
    nonempty = table.loc[table["count"] > 0]
    if not np.allclose(nonempty["observed_fraction"].to_numpy(), prob_true) or not np.allclose(
        nonempty["mean_predicted"].to_numpy(), prob_pred
    ):
        raise RuntimeError("Reliability bins disagree with sklearn.calibration_curve.")
    return table
