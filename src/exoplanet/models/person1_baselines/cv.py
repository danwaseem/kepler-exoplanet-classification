"""Grouped cross-validation on the saved development folds, with probabilities.

The shared ``cross_validate_estimator`` reports label metrics for the three
class team setup. This module adds what Person 1 needs on top of it:

* out-of-fold class probabilities for every experiment,
* the binary vetting setup (CONFIRMED vs FALSE POSITIVE), whose macro
  scores must average over two classes, not three,
* nested cross-validation for choosing C, with inner folds grouped by host
  star and built from the outer training fold only,
* a summary dict in the same shape as the shared helper so results are
  written through ``write_experiment_results``.

Folds come from ``exoplanet.development.development_fold_masks``, so every
training and validation split is the team's saved split.
"""

from __future__ import annotations

import inspect
import json
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from exoplanet.config import CLASS_ORDER, GROUP_COLUMN, ID_COLUMN, TARGET_COLUMN
from exoplanet.development import (
    assert_no_test_rows,
    development_fold_masks,
    split_identity_record,
)
from exoplanet.evaluation import mean_std, write_experiment_results
from exoplanet.models.person1_baselines.metrics import (
    all_metrics,
    label_metrics,
    summarize_folds,
)

OWNER = "person1"
N_FOLDS = 5
N_INNER = 4


@dataclass
class CVResult:
    setup: str
    classes: tuple[str, ...]
    feature_columns: list[str]
    fold_rows: list[dict]          # shared-style per-fold label scores
    extra_fold_rows: list[dict]    # full validation metric set per fold
    oof: pd.DataFrame              # one row per development row of the setup
    fold_info: list[dict] = field(default_factory=list)  # iterations, coefs, kept features
    log_loss_applicable: bool = True

    def fold_scores(self, metric: str = "validation_macro_f1") -> np.ndarray:
        return np.array([row[metric] for row in self.fold_rows], dtype=float)

    def mean(self, metric: str = "validation_macro_f1") -> float:
        return float(self.fold_scores(metric).mean())


def _proba_frame(estimator, X, classes) -> pd.DataFrame:
    """Class probabilities with one column per class name, in ``classes`` order."""
    model = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
    if type(model).__name__ == "DummyClassifier" and getattr(model, "strategy", "") == "stratified":
        # predict_proba of a stratified dummy is a separate random draw; use the
        # one-hot of its predictions so labels and probabilities agree.
        pred = estimator.predict(X)
        return pd.DataFrame({c: (pred == c).astype(float) for c in classes}, index=X.index)
    P = estimator.predict_proba(X)
    frame = pd.DataFrame(P, columns=[str(c) for c in estimator.classes_], index=X.index)
    for c in classes:
        if c not in frame:
            frame[c] = 0.0
    return frame[list(classes)]


def _fit_one_fold(factory, X, y, groups, train, valid, classes, fold_id):
    """Fit and score one fold. Runs in a worker process."""
    kwargs = {}
    params = inspect.signature(factory).parameters
    if "groups" in params:
        kwargs["groups"] = groups[train]
    if "y_train" in params:
        kwargs["y_train"] = y[train]
    if "X_train" in params:
        kwargs["X_train"] = X.loc[train]
    estimator = factory(**kwargs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(X.loc[train], y.loc[train])
    n_conv = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    pred_train = estimator.predict(X.loc[train])
    pred_valid = estimator.predict(X.loc[valid])
    proba_valid = _proba_frame(estimator, X.loc[valid], classes)
    info = {"fold": fold_id, "convergence_warnings": int(n_conv)}
    final = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
    if hasattr(final, "coef_"):
        info["coef"] = np.asarray(final.coef_).tolist()
        info["model_classes"] = [str(c) for c in final.classes_]
        info["n_iter"] = np.asarray(final.n_iter_).ravel().tolist()
        info["nonzero_features"] = int((np.abs(final.coef_) > 1e-8).any(axis=0).sum())
    if hasattr(estimator, "named_steps") and "prune" in estimator.named_steps:
        info["n_features_used"] = int(len(estimator.named_steps["prune"].keep_idx_))
    else:
        info["n_features_used"] = int(X.shape[1])
    return {
        "fold": fold_id,
        "pred_train": pd.Series(pred_train, index=X.index[train]).astype(str),
        "pred_valid": pd.Series(pred_valid, index=X.index[valid]).astype(str),
        "proba_valid": proba_valid,
        "info": info,
    }


def run_cv(
    frame: pd.DataFrame,
    y: pd.Series,
    columns: list[str],
    factory,
    setup: str,
    classes,
    *,
    n_jobs: int = 1,
    log_loss_applicable: bool = True,
) -> CVResult:
    """Fit ``factory()`` on each saved development training fold."""
    assert_no_test_rows(frame)
    classes = tuple(classes)
    X = frame[columns]
    groups = frame[GROUP_COLUMN]
    masks = [development_fold_masks(frame, k) for k in range(N_FOLDS)]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one_fold)(factory, X, y, groups, tr, va, classes, k)
        for k, (tr, va) in enumerate(masks)
    )
    return assemble(frame, y, columns, setup, classes, results, masks,
                    log_loss_applicable=log_loss_applicable)


def run_cv_many(jobs: dict, *, n_jobs: int = -1) -> dict[str, CVResult]:
    """Run several cross-validations in one worker pool.

    ``jobs`` maps a key to a dict with frame, y, columns, factory, setup,
    classes and optionally log_loss_applicable. Every (job, fold) pair is a
    separate task so slow configurations do not block fast ones.
    """
    prepared, tasks = {}, []
    for key, job in jobs.items():
        frame = job["frame"]
        assert_no_test_rows(frame)
        X = frame[job["columns"]]
        groups = frame[GROUP_COLUMN]
        masks = [development_fold_masks(frame, k) for k in range(N_FOLDS)]
        prepared[key] = masks
        for k, (tr, va) in enumerate(masks):
            tasks.append((key, delayed(_fit_one_fold)(
                job["factory"], X, job["y"], groups, tr, va, tuple(job["classes"]), k)))
    outputs = Parallel(n_jobs=n_jobs)(task for _, task in tasks)
    by_key: dict[str, list] = {}
    for (key, _), out in zip(tasks, outputs):
        by_key.setdefault(key, []).append(out)
    results = {}
    for key, job in jobs.items():
        results[key] = assemble(
            job["frame"], job["y"], job["columns"], job["setup"], tuple(job["classes"]),
            by_key[key], prepared[key],
            log_loss_applicable=job.get("log_loss_applicable", True),
        )
    return results


def assemble(frame, y, columns, setup, classes, results, masks, *, log_loss_applicable=True):
    fold_rows, extra_rows, info = [], [], []
    oof_pred = pd.Series(index=frame.index, dtype=object)
    oof_proba = pd.DataFrame(index=frame.index, columns=list(classes), dtype=float)
    for res, (tr, va) in sorted(zip(results, masks), key=lambda t: t[0]["fold"]):
        k = res["fold"]
        tr_scores = label_metrics(y.loc[tr], res["pred_train"], classes)
        va_scores = label_metrics(y.loc[va], res["pred_valid"], classes)
        fold_rows.append(
            {
                "fold": k,
                "n_train": int(tr.sum()),
                "n_validation": int(va.sum()),
                "n_train_groups": int(frame.loc[tr, GROUP_COLUMN].nunique()),
                "n_validation_groups": int(frame.loc[va, GROUP_COLUMN].nunique()),
                "train_accuracy": tr_scores["accuracy"],
                "train_macro_precision": tr_scores["macro_precision"],
                "train_macro_recall": tr_scores["macro_recall"],
                "train_macro_f1": tr_scores["macro_f1"],
                "validation_accuracy": va_scores["accuracy"],
                "validation_macro_precision": va_scores["macro_precision"],
                "validation_macro_recall": va_scores["macro_recall"],
                "validation_macro_f1": va_scores["macro_f1"],
            }
        )
        extra = {"fold": k, "train_macro_f1": tr_scores["macro_f1"]}
        extra.update(
            all_metrics(y.loc[va], res["pred_valid"], res["proba_valid"], classes,
                        log_loss_applicable=log_loss_applicable)
        )
        extra["macro_f1_gap"] = extra["train_macro_f1"] - extra["macro_f1"]
        extra["convergence_warnings"] = res["info"]["convergence_warnings"]
        extra["n_features_used"] = res["info"]["n_features_used"]
        if "nonzero_features" in res["info"]:
            extra["nonzero_features"] = res["info"]["nonzero_features"]
        extra_rows.append(extra)
        info.append(res["info"])
        oof_pred.loc[res["pred_valid"].index] = res["pred_valid"]
        oof_proba.loc[res["proba_valid"].index, list(classes)] = res["proba_valid"].to_numpy()
    if oof_pred.isna().any():
        raise RuntimeError("An out-of-fold prediction is missing.")
    oof = pd.DataFrame(
        {
            ID_COLUMN: frame[ID_COLUMN].astype(str),
            GROUP_COLUMN: frame[GROUP_COLUMN],
            "dev_fold": frame["dev_fold"].astype(int),
            "y_true": y.astype(str),
            "y_pred": oof_pred.astype(str),
        },
        index=frame.index,
    )
    for c in classes:
        oof[f"p_{c.replace(' ', '_')}"] = oof_proba[c].astype(float)
    return CVResult(setup, tuple(classes), list(columns), fold_rows, extra_rows, oof, info,
                    log_loss_applicable)


# ----------------------------------------------------------------- summaries
def shared_style_summary(cv: CVResult) -> dict:
    """Summary dict in the shape written by ``write_experiment_results``.

    Macro scores average over the setup's own classes. For the binary setup
    the confusion matrix passed to the shared writer is the 3x3 CLASS_ORDER
    matrix, whose CANDIDATE row and column are zero because CANDIDATE rows are
    not in this setup; the true 2x2 matrix is stored as
    ``binary_confusion_matrix``.
    """
    y, pred = cv.oof["y_true"], cv.oof["y_pred"]
    summary: dict = {
        "setup": cv.setup,
        "setup_description": (
            "three classes: CONFIRMED, CANDIDATE, FALSE POSITIVE"
            if cv.setup == "3class"
            else "binary vetting setup: CONFIRMED vs FALSE POSITIVE, CANDIDATE rows dropped; "
            "macro scores average over these two classes only"
        ),
        "classes": list(cv.classes),
        "folds": cv.fold_rows,
        "oof_prediction": pred,
        "confusion_matrix": confusion_matrix(y, pred, labels=CLASS_ORDER).tolist(),
        "per_class": [],
        "split": split_identity_record(),
        "n_features": len(cv.feature_columns),
        "n_rows": int(len(cv.oof)),
        "test_scored": False,
    }
    if cv.setup == "binary":
        summary["binary_confusion_matrix"] = {
            "labels": list(cv.classes),
            "matrix": confusion_matrix(y, pred, labels=list(cv.classes)).tolist(),
        }
    pooled = label_metrics(y, pred, cv.classes)
    counts = y.value_counts()
    summary["per_class"] = [
        {
            "class": c,
            "precision": pooled[f"precision_{c.replace(' ', '_')}"],
            "recall": pooled[f"recall_{c.replace(' ', '_')}"],
            "f1": pooled[f"f1_{c.replace(' ', '_')}"],
            "support": int(counts.get(c, 0)),
        }
        for c in cv.classes
    ]
    for metric in (
        "train_accuracy", "validation_accuracy", "train_macro_f1", "validation_macro_f1",
        "train_macro_precision", "validation_macro_precision",
        "train_macro_recall", "validation_macro_recall",
    ):
        stats = mean_std([row[metric] for row in cv.fold_rows])
        summary[f"mean_{metric}"] = stats["mean"]
        summary[f"std_{metric}"] = stats["std"]
    summary["mean_macro_f1_gap"] = float(
        summary["mean_train_macro_f1"] - summary["mean_validation_macro_f1"]
    )
    return summary


def extra_metrics(cv: CVResult) -> dict:
    folds = pd.DataFrame(cv.extra_fold_rows)
    oof = cv.oof
    proba = oof[[f"p_{c.replace(' ', '_')}" for c in cv.classes]].copy()
    proba.columns = list(cv.classes)
    pooled = all_metrics(oof["y_true"], oof["y_pred"], proba, cv.classes,
                         log_loss_applicable=cv.log_loss_applicable)
    return {
        "setup": cv.setup,
        "classes": list(cv.classes),
        "conventions": (
            "fold_mean_sd: mean and SD (ddof=1) over the 5 saved development folds. "
            "pooled_oof: computed once on all out-of-fold predictions. Binary ROC-AUC, "
            "PR-AUC and Brier use P(CONFIRMED). 3-class ROC-AUC and PR-AUC are one-vs-rest "
            "macro; 3-class Brier is the multiclass Brier (sum over classes). Log loss clips "
            "at 1e-15 and is null where not applicable (dummy classifiers with 0/1 outputs)."
        ),
        "log_loss_applicable": cv.log_loss_applicable,
        "fold_mean_sd": summarize_folds(folds),
        "pooled_oof": pooled,
        "test_scored": False,
    }


def write_experiment(name: str, cv: CVResult, config: dict, *, summary: dict | None = None,
                     save_oof: bool = True) -> str:
    """Write the five shared files plus extra metrics and cached OOF predictions."""
    if summary is None:
        summary = shared_style_summary(cv)
    else:
        summary = dict(summary)
        summary.setdefault("setup", cv.setup)
        summary.setdefault("classes", list(cv.classes))
    config = {**config, "setup": cv.setup, "classes": list(cv.classes),
              "n_features": len(cv.feature_columns), "feature_columns": cv.feature_columns,
              "folds": "saved dev_fold 0-4 (host-star grouped)", "test_scored": False}
    root = write_experiment_results(OWNER, name, summary, config)
    extra = extra_metrics(cv)
    (root / "extra_metrics.json").write_text(json.dumps(extra, indent=2), encoding="utf-8")
    pd.DataFrame(cv.extra_fold_rows).to_csv(root / "extra_fold_metrics.csv", index=False)
    if save_oof:
        out = cv.oof.drop(columns=[GROUP_COLUMN]).round(6)
        out.to_csv(root / "oof_predictions.csv.gz", index=False, compression="gzip")
    return str(root)


# ----------------------------------------------------------------- nested CV
def inner_splits(frame_train: pd.DataFrame, y_train: pd.Series, seed: int, n_inner: int = N_INNER):
    """Inner folds grouped by host star, built from one outer training fold only."""
    splitter = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    return list(splitter.split(frame_train, y_train, groups=frame_train[GROUP_COLUMN]))


def _inner_task(factory, X, y, tr_idx, va_idx, classes, outer, inner, label):
    est = factory()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    pred = est.predict(X.iloc[va_idx]).astype(str)
    score = label_metrics(y.iloc[va_idx], pred, classes)["macro_f1"]
    return {"outer_fold": outer, "inner_fold": inner, "config": label, "inner_macro_f1": score}


def nested_inner_scores(frame, y, columns, factories: dict, classes, *, seed: int,
                        n_jobs: int = -1) -> pd.DataFrame:
    """Inner-fold macro F1 for every candidate configuration and outer fold.

    ``factories`` maps a configuration label to a zero-argument factory. The
    outer validation fold is never seen here. The outer score of the chosen
    configuration equals its score in the ordinary (non-nested) run on the
    same outer fold, because the refit on the full outer training fold is the
    same fit, so that score is reused rather than refit.
    """
    assert_no_test_rows(frame)
    X = frame[columns]
    tasks = []
    for outer in range(N_FOLDS):
        tr, _ = development_fold_masks(frame, outer)
        f_tr, y_tr = frame.loc[tr], y.loc[tr]
        X_tr = X.loc[tr]
        for inner, (i_tr, i_va) in enumerate(inner_splits(f_tr, y_tr, seed)):
            for label, factory in factories.items():
                tasks.append(delayed(_inner_task)(factory, X_tr, y_tr, i_tr, i_va, classes,
                                                  outer, inner, label))
    return pd.DataFrame(Parallel(n_jobs=n_jobs)(tasks))


def nested_selection(inner: pd.DataFrame, outer_scores: dict[str, np.ndarray],
                     candidates: list[str]) -> dict:
    """Choose a configuration per outer fold from the inner scores.

    Returns the nested estimate (mean of the outer scores of the configuration
    chosen in each fold) beside the non-nested estimate (best mean outer score
    over the candidates, which is what selecting on the reported folds gives).
    """
    sub = inner.loc[inner["config"].isin(candidates)]
    mean_inner = sub.groupby(["outer_fold", "config"])["inner_macro_f1"].mean().unstack()
    chosen, nested = [], []
    for outer in range(N_FOLDS):
        row = mean_inner.loc[outer, candidates]
        best = row.idxmax()  # first in candidate order wins ties
        chosen.append(best)
        nested.append(float(outer_scores[best][outer]))
    means = {c: float(np.mean(outer_scores[c])) for c in candidates}
    best_non_nested = max(candidates, key=lambda c: means[c])
    nn_scores = np.asarray(outer_scores[best_non_nested], dtype=float)
    return {
        "candidates": candidates,
        "chosen_per_outer_fold": chosen,
        "nested_outer_scores": nested,
        "nested_mean": float(np.mean(nested)),
        "nested_sd": float(np.std(nested, ddof=1)),
        "non_nested_choice": best_non_nested,
        "non_nested_outer_scores": nn_scores.tolist(),
        "non_nested_mean": float(nn_scores.mean()),
        "non_nested_sd": float(nn_scores.std(ddof=1)),
        "optimism": float(nn_scores.mean() - np.mean(nested)),
    }
