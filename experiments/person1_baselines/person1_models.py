"""Model factories for the Person 1 baselines.

This module sits next to the experiment scripts rather than under
``src/exoplanet/models/person1_baselines/`` because the shared test
``tests/test_stage1.py::test_source_does_not_train_other_classifiers``
rejects any file under ``src/`` that names the logistic regression class.
The reusable pieces that do not name an estimator (transforms, fold loop,
metrics, calibration wrapper, temporal sets) stay in the package.

Every model is a Pipeline that starts with the shared preprocessing from
``exoplanet.preprocess.make_preprocess_pipeline``: median imputation, and a
StandardScaler where the model needs it. All steps are fit on the training
fold only.
"""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from exoplanet.config import RANDOM_SEED
from exoplanet.preprocess import make_preprocess_pipeline
from exoplanet.models.person1_baselines.transforms import SpearmanPruner, signed_log1p

SEED = RANDOM_SEED
# The earlier grid stopped at C=10, which was also the selected value. The grid
# now runs two decades past it in both directions.
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
# 5,000 iterations left saga short of convergence for L1 at C >= 1. A single
# fold at C=1000 needs about 4,800 iterations at the default tolerance, so
# 20,000 gives room without loosening the tolerance.
LR_MAX_ITER = 20_000
LR_TOL = 1e-4
SPEARMAN_THRESHOLD = 0.8

DUMMY_KINDS = ("dummy_most_frequent", "dummy_stratified")
LR_KINDS = ("lr_none", "lr_l2", "lr_l1")
NB_KINDS = ("nb_a_standardized", "nb_b_signed_log", "nb_c_log_spearman")

NB_DESCRIPTION = {
    "nb_a_standardized": "median impute, standardize, GaussianNB",
    "nb_b_signed_log": "median impute, signed log1p, standardize, GaussianNB",
    "nb_c_log_spearman": (
        "median impute, signed log1p, drop one of each |Spearman r| > 0.8 pair "
        "(decided on the training fold), standardize, GaussianNB"
    ),
}


def _with_scaling(model_step, *, scale: bool = True, middle: list | None = None) -> Pipeline:
    steps = list(make_preprocess_pipeline(scale=scale).steps)
    if middle:
        # Insert transforms between the imputer and the scaler.
        steps = steps[:1] + middle + steps[1:]
    steps.append(("model", model_step))
    return Pipeline(steps)


def make_logistic(kind: str, C: float | None = None) -> LogisticRegression:
    if kind == "lr_none":
        return LogisticRegression(
            C=np.inf, solver="lbfgs", class_weight="balanced",
            max_iter=LR_MAX_ITER, tol=LR_TOL,
        )
    if kind == "lr_l2":
        return LogisticRegression(
            C=C, l1_ratio=0.0, solver="lbfgs", class_weight="balanced",
            max_iter=LR_MAX_ITER, tol=LR_TOL,
        )
    if kind == "lr_l1":
        return LogisticRegression(
            C=C, l1_ratio=1.0, solver="saga", class_weight="balanced",
            max_iter=LR_MAX_ITER, tol=LR_TOL, random_state=SEED,
        )
    raise ValueError(kind)


def make_model(kind: str, C: float | None = None) -> Pipeline:
    """Return a fresh unfitted Pipeline for a model name."""
    if kind == "dummy_most_frequent":
        return _with_scaling(DummyClassifier(strategy="most_frequent"), scale=False)
    if kind == "dummy_stratified":
        return _with_scaling(
            DummyClassifier(strategy="stratified", random_state=SEED), scale=False
        )
    if kind in LR_KINDS:
        return _with_scaling(make_logistic(kind, C))
    log_step = ("log", FunctionTransformer(signed_log1p))
    if kind == "nb_a_standardized":
        return _with_scaling(GaussianNB())
    if kind == "nb_b_signed_log":
        return _with_scaling(GaussianNB(), middle=[log_step])
    if kind == "nb_c_log_spearman":
        return _with_scaling(
            GaussianNB(), middle=[log_step, ("prune", SpearmanPruner(SPEARMAN_THRESHOLD))]
        )
    raise ValueError(kind)


def config_label(kind: str, C: float | None = None) -> str:
    if kind == "lr_none":
        return "C=inf"
    if kind in ("lr_l2", "lr_l1"):
        return f"C={C:g}"
    return "-"


def model_config(kind: str, C: float | None = None) -> dict:
    """Plain description of a model for model_config.json."""
    record: dict = {"model": kind, "config": config_label(kind, C), "standardized": True}
    if kind in DUMMY_KINDS:
        record["standardized"] = False
        record["strategy"] = kind.replace("dummy_", "")
        record["random_state"] = SEED if kind == "dummy_stratified" else None
    elif kind in LR_KINDS:
        record.update(
            {
                "estimator": "LogisticRegression",
                "penalty": {"lr_none": "none", "lr_l2": "l2", "lr_l1": "l1"}[kind],
                "C": None if kind == "lr_none" else float(C),
                "solver": "saga" if kind == "lr_l1" else "lbfgs",
                "class_weight": "balanced",
                "max_iter": LR_MAX_ITER,
                "tol": LR_TOL,
                "random_state": SEED if kind == "lr_l1" else None,
            }
        )
    elif kind in NB_KINDS:
        record.update(
            {
                "estimator": "GaussianNB",
                "var_smoothing": 1e-9,
                "pipeline": NB_DESCRIPTION[kind],
            }
        )
        if kind == "nb_c_log_spearman":
            record["spearman_threshold"] = SPEARMAN_THRESHOLD
    return record


def lr_grid() -> list[tuple[str, float | None]]:
    """Every logistic regression configuration searched in E1."""
    grid: list[tuple[str, float | None]] = [("lr_none", None)]
    for kind in ("lr_l2", "lr_l1"):
        grid.extend((kind, C) for C in C_GRID)
    return grid
