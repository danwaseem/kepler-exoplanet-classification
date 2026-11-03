"""Shared validation metrics and result-writing helpers.

Teammates should report grouped development validation scores with these
helpers so comparisons use the same class order and macro averaging. The
test partition is not scored here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from exoplanet.config import CLASS_ORDER, GROUP_COLUMN, ROOT, TARGET_COLUMN
from exoplanet.development import (
    DevelopmentBundle,
    assert_no_test_rows,
    development_fold_masks,
    feature_matrix,
    labels,
    split_identity_record,
)


def classification_scores(y_true, y_pred) -> dict[str, float]:
    """Accuracy and macro averages with a fixed three-class order."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", labels=CLASS_ORDER, zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", labels=CLASS_ORDER, zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=CLASS_ORDER, zero_division=0)
        ),
    }


def per_class_metrics(y_true, y_pred) -> pd.DataFrame:
    """Precision, recall, and F1 for each disposition."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_ORDER, zero_division=0
    )
    return pd.DataFrame(
        {
            "class": CLASS_ORDER,
            "precision": precision.astype(float),
            "recall": recall.astype(float),
            "f1": f1.astype(float),
            "support": support.astype(int),
        }
    )


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def cross_validate_estimator(
    bundle: DevelopmentBundle,
    factory,
) -> dict:
    """Fit a fresh estimator on each saved development training fold.

    ``factory`` must return a new unfitted estimator or Pipeline. Fit the
    imputer inside that pipeline on the training fold only. Predictions are
    collected out of fold. The test partition is not used.
    """
    frame = bundle.frame
    assert_no_test_rows(frame)
    columns = bundle.feature_columns
    x = feature_matrix(frame, columns)
    y = labels(frame)
    oof = pd.Series(index=frame.index, dtype=object)
    fold_rows = []
    for fold_id in range(5):
        train, valid = development_fold_masks(frame, fold_id)
        estimator = factory()
        estimator.fit(x.loc[train], y.loc[train])
        pred_train = estimator.predict(x.loc[train])
        pred_valid = estimator.predict(x.loc[valid])
        oof.loc[frame.index[valid]] = pred_valid
        train_scores = classification_scores(y.loc[train], pred_train)
        valid_scores = classification_scores(y.loc[valid], pred_valid)
        fold_rows.append(
            {
                "fold": fold_id,
                "n_train": int(train.sum()),
                "n_validation": int(valid.sum()),
                "n_train_groups": int(frame.loc[train, GROUP_COLUMN].nunique()),
                "n_validation_groups": int(frame.loc[valid, GROUP_COLUMN].nunique()),
                **{f"train_{name}": value for name, value in train_scores.items()},
                **{f"validation_{name}": value for name, value in valid_scores.items()},
            }
        )
    if oof.isna().any():
        raise RuntimeError("An out-of-fold prediction is missing.")
    summary = {
        "folds": fold_rows,
        "oof_prediction": oof,
        "confusion_matrix": confusion_matrix(y, oof, labels=CLASS_ORDER).tolist(),
        "per_class": per_class_metrics(y, oof).to_dict(orient="records"),
        "split": split_identity_record(),
        "n_features": len(columns),
        "n_rows": int(len(frame)),
        "test_scored": False,
    }
    for metric in (
        "train_accuracy",
        "validation_accuracy",
        "train_macro_f1",
        "validation_macro_f1",
        "train_macro_precision",
        "validation_macro_precision",
        "train_macro_recall",
        "validation_macro_recall",
    ):
        stats = mean_std([row[metric] for row in fold_rows])
        summary[f"mean_{metric}"] = stats["mean"]
        summary[f"std_{metric}"] = stats["std"]
    summary["mean_macro_f1_gap"] = float(
        summary["mean_train_macro_f1"] - summary["mean_validation_macro_f1"]
    )
    return summary


def write_experiment_results(
    owner: str,
    experiment_name: str,
    summary: dict,
    config: dict,
    *,
    results_root: Path | None = None,
) -> Path:
    """Write fold metrics, summaries, and config under results/<owner>/<name>/."""
    root = results_root or (ROOT / "results" / owner / experiment_name)
    root.mkdir(parents=True, exist_ok=True)
    folds = pd.DataFrame(summary["folds"])
    folds.to_csv(root / "fold_metrics.csv", index=False)
    per_class = pd.DataFrame(summary["per_class"])
    per_class.to_csv(root / "per_class_metrics.csv", index=False)
    pd.DataFrame(
        summary["confusion_matrix"], index=CLASS_ORDER, columns=CLASS_ORDER
    ).to_csv(root / "oof_confusion_matrix.csv")
    payload = {
        key: value
        for key, value in summary.items()
        if key not in {"folds", "oof_prediction", "per_class"}
    }
    (root / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    experiment_config = {
        **config,
        "owner": owner,
        "experiment_name": experiment_name,
        "test_scored": False,
        "split": summary.get("split", split_identity_record()),
    }
    (root / "model_config.json").write_text(
        json.dumps(experiment_config, indent=2), encoding="utf-8"
    )
    return root


def require_pipeline_imputer(estimator) -> None:
    """Remind teammates that preprocessing belongs inside a Pipeline."""
    if not isinstance(estimator, Pipeline):
        raise TypeError(
            "Fit a scikit-learn Pipeline that includes the median imputer "
            "(and StandardScaler when the model needs scaled features)."
        )
    if "impute" not in estimator.named_steps:
        raise TypeError("The Pipeline must include an 'impute' step.")
