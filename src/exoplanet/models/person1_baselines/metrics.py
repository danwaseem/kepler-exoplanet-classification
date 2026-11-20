"""Label and probability metrics that the shared helper does not compute.

Conventions
-----------
* Macro averages are taken over the classes of the setup only (three for
  3class, two for binary).
* Binary ROC-AUC, PR-AUC (average precision) and Brier use P(CONFIRMED) with
  CONFIRMED as the positive class.
* 3-class ROC-AUC is one-vs-rest macro. 3-class PR-AUC is the macro mean of
  one-vs-rest average precision. 3-class Brier is the multiclass Brier score
  from scikit-learn (sum over classes, range 0 to 2).
* Log loss clips probabilities to [1e-15, 1 - 1e-15]. It is marked not
  applicable for dummy classifiers, whose probabilities are exactly 0 or 1.
* Expected calibration error (binary only) uses 10 equal-width bins on
  P(CONFIRMED).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

EPS = 1e-15


def _tag(name: str) -> str:
    return name.replace(" ", "_")


def label_metrics(y_true, y_pred, classes) -> dict[str, float]:
    classes = list(classes)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred, labels=classes),
    }
    for average in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=classes, average=average, zero_division=0
        )
        out[f"{average}_precision"], out[f"{average}_recall"], out[f"{average}_f1"] = p, r, f
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    for name, pi, ri, fi in zip(classes, p, r, f):
        out[f"precision_{_tag(name)}"] = pi
        out[f"recall_{_tag(name)}"] = ri
        out[f"f1_{_tag(name)}"] = fi
    return {k: float(v) for k, v in out.items()}


def expected_calibration_error(y_true_binary, p_pos, n_bins: int = 10) -> float:
    y = np.asarray(y_true_binary, dtype=float)
    p = np.asarray(p_pos, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def proba_metrics(y_true, proba: pd.DataFrame, classes, *, log_loss_applicable=True) -> dict:
    """Probability metrics. ``proba`` has one column per class name."""
    classes = list(classes)
    y_true = np.asarray(y_true)
    P = proba[classes].to_numpy(dtype=float)
    out: dict[str, float | str | None] = {}
    if len(classes) == 2:
        pos = classes[0]  # CONFIRMED
        y_bin = (y_true == pos).astype(int)
        p_pos = P[:, 0]
        out["roc_auc"] = roc_auc_score(y_bin, p_pos)
        out["pr_auc"] = average_precision_score(y_bin, p_pos)
        out["brier"] = brier_score_loss(y_bin, p_pos)
        out["ece_10bin"] = expected_calibration_error(y_bin, p_pos)
    else:
        # scikit-learn needs sorted labels for multiclass probability metrics.
        classes = sorted(classes)
        P = proba[classes].to_numpy(dtype=float)
        out["roc_auc_ovr_macro"] = roc_auc_score(
            y_true, P, multi_class="ovr", average="macro", labels=classes
        )
        aps = [average_precision_score((y_true == c).astype(int), P[:, i])
               for i, c in enumerate(classes)]
        out["pr_auc_ovr_macro"] = float(np.mean(aps))
        for c, ap in zip(classes, aps):
            out[f"pr_auc_{_tag(c)}"] = ap
        out["brier_multiclass"] = brier_score_loss(y_true, P, labels=classes)
    if log_loss_applicable:
        Pc = np.clip(P, EPS, 1 - EPS)
        Pc = Pc / Pc.sum(axis=1, keepdims=True)
        out["log_loss"] = log_loss(y_true, Pc, labels=classes)
    else:
        out["log_loss"] = None
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in out.items()}


def all_metrics(y_true, y_pred, proba, classes, *, log_loss_applicable=True) -> dict:
    out = label_metrics(y_true, y_pred, classes)
    out.update(proba_metrics(y_true, proba, classes, log_loss_applicable=log_loss_applicable))
    return out


def summarize_folds(fold_metrics: pd.DataFrame, prefix: str = "") -> dict:
    """Mean and SD (ddof=1) of every numeric fold column."""
    out = {}
    for column in fold_metrics.columns:
        if column == "fold":
            continue
        values = pd.to_numeric(fold_metrics[column], errors="coerce")
        if values.isna().all():
            out[f"{prefix}{column}"] = {"mean": None, "std": None}
            continue
        out[f"{prefix}{column}"] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    return out


LOWER_IS_BETTER = {"brier", "brier_multiclass", "log_loss", "ece_10bin"}


def corrected_paired_t(diffs, n_train: float, n_test: float, metric: str | None = None) -> dict:
    """Paired comparison of two models over the same folds.

    Differences are first minus second. Reports their mean and SD, how many
    folds favour the first model (difference above zero for metrics where
    higher is better, below zero for Brier, log loss and ECE), and the
    Nadeau-Bengio corrected resampled t statistic,
    which inflates the variance by n_test / n_train because training sets
    overlap across folds. With five folds this is a rough guide, not a
    formal test.
    """
    from scipy import stats

    d = np.asarray(diffs, dtype=float)
    k = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if k > 1 else 0.0
    lower = metric is not None and metric.replace("validation_", "") in LOWER_IS_BETTER
    out = {
        "better_when": "lower" if lower else "higher",
        "mean_diff": mean,
        "sd_diff": sd,
        "min_diff": float(d.min()),
        "max_diff": float(d.max()),
        "folds_favouring_first": int((d < 0).sum() if lower else (d > 0).sum()),
        "n_folds": k,
    }
    if sd > 0:
        t = mean / np.sqrt((1.0 / k + n_test / n_train) * sd**2)
        out["corrected_t"] = float(t)
        out["corrected_p_two_sided"] = float(2 * stats.t.sf(abs(t), df=k - 1))
    else:
        out["corrected_t"] = None
        out["corrected_p_two_sided"] = None
    return out
