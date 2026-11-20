"""Final test run, following docs/shared/FINAL_TEST_PROTOCOL.md.

Two modes:
  --verify  Runs the pre-test checks only. Test rows are never read.
  --run     Runs the same checks, then scores the test partition once.

Every model is built through an existing factory (protocol section 5). No new
model code is written here. Random Forest and the pruned Decision Tree are
scored from their saved pipelines.

Use the project environment:  .venv/bin/python scripts/run_final_test.py --verify
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "person1_baselines"))

from exoplanet.config import CLASS_ORDER, GROUP_COLUMN, ID_COLUMN, TARGET_COLUMN  # noqa: E402
from exoplanet.models.person2_knn_svm.models import (  # noqa: E402
    make_knn,
    make_linear_svm,
    make_rbf_svm,
)
from exoplanet.stage2 import make_tree_pipeline  # noqa: E402
from exoplanet.stage4 import make_forest_pipeline  # noqa: E402
from person1_models import make_model  # noqa: E402

DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
OUT = RESULTS / "final_test"

DATASET_PATH = ROOT / "data" / "raw" / "koi_cumulative_2026-10-06.csv"
DATASET_SHA256 = "a85348f20082d24fb14d4a05a9a44c644b801998fad7cec36f8980eb1f0a69ab"
SPLIT_PATH = DATA / "proposed_splits.csv"
SPLIT_SHA256 = "d0a2892a312ddfb47ae4fe37650563ec566ee254a98c781598485992442b021a"
INCOMPLETE_PATH = DATA / "incomplete_koi_split_assignments.csv"
FEATURE_PATH = DATA / "proposed_leakage_aware_features.csv"
MATRIX_PATH = DATA / "proposed_primary_matrix_unimputed.csv"
SEED = 245
N_FEATURES = 76
N_DEV = 7381
N_TEST = 1820
N_TEST_TOTAL = 1911
N_TEST_NO_DEPTH = 91
N_BOOT = 1000
SMOKE_TOL = 0.001

RF_PATH = RESULTS / "person3" / "random_forest" / "tuned_forest.joblib"
DT_PATH = RESULTS / "person3" / "decision_tree" / "stage3" / "selected_development_tree.joblib"

# Frozen settings (protocol section 5).
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 20,
    "max_features": "sqrt",
    "min_samples_leaf": 2,
    "class_weight": "balanced",
}
DT_CCP_ALPHA = 0.0011107571059714625

# Factories for a fresh fit. RF and DT are fit fresh only in the checks.
FACTORIES = {
    "Random Forest": lambda: make_forest_pipeline(RF_PARAMS),
    "RBF SVM": lambda: make_rbf_svm(C=10.0, gamma="scale", class_weight="balanced"),
    "Linear SVM": lambda: make_linear_svm(C=10.0, class_weight="balanced"),
    "Logistic Regression": lambda: make_model("lr_l2", C=1000.0),
    "Pruned Decision Tree": lambda: make_tree_pipeline(
        None, min_samples_split=2, min_samples_leaf=1, ccp_alpha=DT_CCP_ALPHA
    ),
    "KNN": lambda: make_knn(n_neighbors=7, metric="euclidean", weights="distance"),
    "Naive Bayes": lambda: make_model("nb_b_signed_log"),
    "Majority baseline": lambda: make_model("dummy_most_frequent"),
}
SAVED_MODELS = {"Random Forest": RF_PATH, "Pruned Decision Tree": DT_PATH}
MODEL_ORDER = list(FACTORIES)
SCORED_MODELS = [m for m in MODEL_ORDER if m != "Majority baseline"]  # 7 models

# Score type for ROC-AUC (protocol section 9). Checked against the pipeline.
SCORE_TYPE = {
    "Random Forest": "predict_proba",
    "Pruned Decision Tree": "predict_proba",
    "Logistic Regression": "predict_proba",
    "Naive Bayes": "predict_proba",
    "KNN": "predict_proba",
    "RBF SVM": "decision_function",
    "Linear SVM": "decision_function",
}

# Saved development summaries: (file, field path). Mean of the 5 grouped folds.
DEV_F1_SOURCE = {
    "Random Forest": ("person3/random_forest/model_config.json", "mean_validation_macro_f1"),
    "Pruned Decision Tree": (
        "person3/decision_tree/stage3/stage3_metrics.json",
        "selected.mean_validation_macro_f1",
    ),
    "RBF SVM": ("person2/rbf_svm_tuned/summary.json", "mean_validation_macro_f1"),
    "Linear SVM": ("person2/linear_svm_tuned/summary.json", "mean_validation_macro_f1"),
    "KNN": ("person2/knn_tuned/summary.json", "mean_validation_macro_f1"),
    "Logistic Regression": ("person1/e1_lr_l2_3class/summary.json", "mean_validation_macro_f1"),
    "Naive Bayes": ("person1/e2_nb_b_signed_log_3class/summary.json", "mean_validation_macro_f1"),
    "Majority baseline": (
        "person1/e0_dummy_most_frequent_3class/summary.json",
        "mean_validation_macro_f1",
    ),
}
# Saved fold-0 values for the smoke test: (file, filter column, filter value).
FOLD0_SOURCE = {
    "Random Forest": ("person3/random_forest/comparison_folds.csv", "Tuned forest"),
    "Pruned Decision Tree": ("person3/decision_tree/stage3/comparison_folds.csv", "selected"),
    "RBF SVM": ("person2/rbf_svm_tuned/fold_metrics.csv", None),
    "Linear SVM": ("person2/linear_svm_tuned/fold_metrics.csv", None),
    "KNN": ("person2/knn_tuned/fold_metrics.csv", None),
    "Logistic Regression": ("person1/e1_lr_l2_3class/fold_metrics.csv", None),
    "Naive Bayes": ("person1/e2_nb_b_signed_log_3class/fold_metrics.csv", None),
    "Majority baseline": ("person1/e0_dummy_most_frequent_3class/fold_metrics.csv", None),
}


class CheckFailed(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


# ---------------------------------------------------------------- data loading

def feature_list() -> list[str]:
    columns = pd.read_csv(FEATURE_PATH)["column"].astype(str).tolist()
    require(len(columns) == N_FEATURES and len(set(columns)) == N_FEATURES, "feature list is not 76 unique columns")
    return columns


def split_counts() -> dict:
    """Row counts from partition and primary_supervised_row columns only."""
    cols = ["partition", "primary_supervised_row"]
    splits = pd.read_csv(SPLIT_PATH, usecols=cols)
    incomplete = pd.read_csv(INCOMPLETE_PATH, usecols=cols)
    test = splits["partition"].eq("test")
    primary = splits["primary_supervised_row"].astype(str).eq("True")
    return {
        "dev_primary": int((splits["partition"].eq("development") & primary).sum()),
        "test_total": int(test.sum()),
        "test_primary": int((test & primary).sum()),
        "test_without_depth_in_splits": int((test & ~primary).sum()),
        "test_without_depth_in_incomplete_file": int(incomplete["partition"].eq("test").sum()),
    }


def matrix_partition_counts() -> dict:
    """Partition counts of the feature matrix, reading the partition column only."""
    part = pd.read_csv(MATRIX_PATH, usecols=["partition"])["partition"]
    return {k: int(v) for k, v in part.value_counts().items()}


def load_matrix(partition: str) -> pd.DataFrame:
    """Read only rows of one partition. Other rows are skipped at read time."""
    part = pd.read_csv(MATRIX_PATH, usecols=["partition"])["partition"]
    skip = [i + 1 for i in np.flatnonzero(part.to_numpy() != partition)]
    frame = pd.read_csv(MATRIX_PATH, skiprows=skip)
    require(frame["partition"].eq(partition).all(), f"loaded rows outside partition {partition}")
    return frame


# ------------------------------------------------------------- scoring helpers

def macro_f1(y_true, y_pred, sample_weight=None) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=CLASS_ORDER,
                          zero_division=0, sample_weight=sample_weight))


def get_scores(pipe, X):
    """predict_proba when the pipeline has it, else decision_function."""
    if hasattr(pipe, "predict_proba"):
        return "predict_proba", pipe.predict_proba(X)
    return "decision_function", pipe.decision_function(X)


def score_pipeline(pipe, X, y) -> dict:
    """The one scoring code path, used by the smoke test and the test run."""
    pred = pipe.predict(X)
    score_type, scores = get_scores(pipe, X)
    classes = list(pipe.classes_)
    out = {
        "pred": pred,
        "score_type": score_type,
        "macro_f1": macro_f1(y, pred),
        "accuracy": float(accuracy_score(y, pred)),
    }
    try:
        out["roc_auc_ovr_macro"] = float(
            roc_auc_score(y, scores, multi_class="ovr", average="macro", labels=classes)
        )
    except ValueError:
        out["roc_auc_ovr_macro"] = float("nan")
    return out


def full_metrics(y, pred) -> dict:
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=CLASS_ORDER, zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f.mean()),
        "mcc": float(matthews_corrcoef(y, pred)),
    }
    per_class = [
        {"class": c, "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, c in enumerate(CLASS_ORDER)
    ]
    cm = confusion_matrix(y, pred, labels=CLASS_ORDER)
    return {"overall": out, "per_class": per_class, "confusion": cm}


def get_field(path: Path, field: str):
    value = json.loads(path.read_text(encoding="utf-8"))
    for key in field.split("."):
        value = value[key]
    return float(value)


def saved_fold0(model: str) -> float:
    rel, model_name = FOLD0_SOURCE[model]
    rows = pd.read_csv(RESULTS / rel)
    if model_name is not None:
        rows = rows.loc[rows["model"].eq(model_name)]
    rows = rows.loc[rows["fold"].eq(0)]
    require(len(rows) == 1, f"expected one saved fold-0 row for {model}")
    return float(rows["validation_macro_f1"].iloc[0])


# --------------------------------------------------------------------- checks

def check_identity() -> dict:
    require(sha256(DATASET_PATH) == DATASET_SHA256, "dataset SHA-256 changed")
    require(sha256(SPLIT_PATH) == SPLIT_SHA256, "split SHA-256 changed")
    return {"dataset_sha256": DATASET_SHA256, "split_sha256": SPLIT_SHA256}


def check_versions() -> dict:
    pins = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "==" in line:
            name, version = line.split("==")
            pins[name.strip()] = version.strip()
    record = {}
    for name, pinned in pins.items():
        installed = md.version(name)
        record[name] = {"pinned": pinned, "installed": installed}
        require(installed == pinned, f"{name} installed {installed}, requirements.txt pins {pinned}")
    return record


def check_counts() -> dict:
    counts = split_counts()
    matrix = matrix_partition_counts()
    require(counts["dev_primary"] == N_DEV, f"development rows {counts['dev_primary']} != {N_DEV}")
    require(counts["test_total"] == N_TEST_TOTAL, f"test rows in total {counts['test_total']}")
    require(counts["test_primary"] == N_TEST, f"test rows with depth {counts['test_primary']} != {N_TEST}")
    require(counts["test_without_depth_in_splits"] == N_TEST_NO_DEPTH, "test rows without depth differ")
    require(counts["test_without_depth_in_incomplete_file"] == N_TEST_NO_DEPTH, "incomplete file test count differs")
    require(matrix.get("test") == N_TEST, f"matrix test rows {matrix.get('test')} != {N_TEST}")
    require(matrix.get("development") == N_DEV, f"matrix development rows {matrix.get('development')} != {N_DEV}")
    return {**counts, "matrix_partition_counts": matrix}


def load_saved(path: Path) -> dict:
    return joblib.load(path)


def check_decision_tree(columns: list[str]) -> dict:
    saved = load_saved(DT_PATH)
    pipe = saved["pipeline"]
    tree = pipe.named_steps["tree"]
    root = int(tree.tree_.n_node_samples[0])
    require(root == N_DEV, f"Decision Tree root sample count {root} != {N_DEV}")
    require(int(tree.n_features_in_) == N_FEATURES, f"Decision Tree n_features_in_ {tree.n_features_in_}")
    require(list(saved["feature_columns"]) == columns, "Decision Tree feature columns differ from the feature list")
    return {"root_n_node_samples": root, "n_features_in": int(tree.n_features_in_)}


def check_random_forest(dev: pd.DataFrame, columns: list[str]) -> dict:
    saved = load_saved(RF_PATH)
    pipe = saved["pipeline"]
    forest = pipe.named_steps["forest"]
    require(int(forest.n_features_in_) == N_FEATURES, f"Random Forest n_features_in_ {forest.n_features_in_}")
    require(list(saved["feature_columns"]) == columns, "Random Forest feature columns differ from the feature list")
    X, y = dev[columns], dev[TARGET_COLUMN]
    refit = FACTORIES["Random Forest"]()
    refit.fit(X, y)
    require(refit.named_steps["forest"].random_state == SEED, "refit forest random_state is not 245")
    same = np.array_equal(refit.predict(X), pipe.predict(X))
    require(same, "refit forest predictions do not match the saved forest on development rows")
    return {"n_features_in": int(forest.n_features_in_), "refit_predictions_match_saved": True,
            "n_rows_compared": int(len(dev))}


def smoke_test(dev: pd.DataFrame, columns: list[str]) -> list[dict]:
    """Fit on development folds 1-4, score fold 0, compare with the saved fold-0 macro-F1."""
    train = dev["dev_fold"].ne(0)
    valid = dev["dev_fold"].eq(0)
    require(not set(dev.loc[train, GROUP_COLUMN]) & set(dev.loc[valid, GROUP_COLUMN]), "fold 0 shares a host star")
    rows = []
    for name in MODEL_ORDER:
        pipe = FACTORIES[name]()
        pipe.fit(dev.loc[train, columns], dev.loc[train, TARGET_COLUMN])
        res = score_pipeline(pipe, dev.loc[valid, columns], dev.loc[valid, TARGET_COLUMN])
        saved = saved_fold0(name)
        diff = res["macro_f1"] - saved
        rows.append({"model": name, "smoke_fold0_macro_f1": res["macro_f1"], "saved_fold0_macro_f1": saved,
                     "difference": diff, "within_tolerance": abs(diff) <= SMOKE_TOL,
                     "score_type": res["score_type"]})
    return rows


def check_score_types(rows: list[dict]) -> None:
    for row in rows:
        want = SCORE_TYPE.get(row["model"])
        if want is not None:
            require(row["score_type"] == want, f"{row['model']} score type {row['score_type']} != {want}")


def run_checks() -> dict:
    record = {"identity": check_identity(), "versions": check_versions()}
    columns = feature_list()
    record["counts"] = check_counts()
    dev = load_matrix("development")
    require(len(dev) == N_DEV, f"loaded {len(dev)} development rows")
    require(set(columns) <= set(dev.columns), "development matrix is missing feature columns")
    record["decision_tree_check"] = check_decision_tree(columns)
    record["random_forest_check"] = check_random_forest(dev, columns)
    smoke = smoke_test(dev, columns)
    record["smoke_test"] = smoke
    check_score_types(smoke)
    bad = [r["model"] for r in smoke if not r["within_tolerance"]]
    require(not bad, f"smoke test outside {SMOKE_TOL} for: {bad}")
    return record, dev, columns


# ------------------------------------------------------------------ statistics

def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact (binomial) two-sided McNemar test on per-row correct vs incorrect."""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return {"a_right_b_wrong": b, "a_wrong_b_right": c, "n_discordant": 0, "p_value": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    p = min(1.0, 2 * tail / 2 ** n)
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "n_discordant": n, "p_value": float(p)}


def bootstrap(test: pd.DataFrame, preds: dict) -> dict:
    """Resample host stars (kepid), same resamples for every model."""
    rng = np.random.default_rng(SEED)
    groups, inverse = np.unique(test[GROUP_COLUMN].to_numpy(), return_inverse=True)
    n_groups = len(groups)
    y = test[TARGET_COLUMN].to_numpy()
    out = {m: {"macro_f1": [], "accuracy": []} for m in preds}
    for _ in range(N_BOOT):
        counts = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)
        w = counts[inverse]
        for m, p in preds.items():
            out[m]["macro_f1"].append(macro_f1(y, p, sample_weight=w))
            out[m]["accuracy"].append(float(accuracy_score(y, p, sample_weight=w)))
    return out


# ------------------------------------------------------------------- test run

def git_info() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    status = [ln for ln in git("status", "--porcelain").splitlines() if "results/final_test" not in ln]
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "tree_dirty_outside_results_final_test": bool(status)}


def score_test(dev: pd.DataFrame, columns: list[str]) -> dict:
    test = load_matrix("test")
    require(len(test) == N_TEST, f"loaded {len(test)} test rows, expected {N_TEST}")
    splits = pd.read_csv(SPLIT_PATH, usecols=[ID_COLUMN, "partition", "primary_supervised_row"])
    expected_ids = set(splits.loc[splits["partition"].eq("test")
                                  & splits["primary_supervised_row"].astype(str).eq("True"), ID_COLUMN])
    require(set(test[ID_COLUMN]) == expected_ids, "test row ids differ from the split file")
    require(not set(test[GROUP_COLUMN]) & set(dev[GROUP_COLUMN]), "a host star is in both partitions")
    X, y = test[columns], test[TARGET_COLUMN]
    X_dev, y_dev = dev[columns], dev[TARGET_COLUMN]

    results = {}
    for name in MODEL_ORDER:
        if name in SAVED_MODELS:
            pipe = load_saved(SAVED_MODELS[name])["pipeline"]
            source = f"saved pipeline {SAVED_MODELS[name].relative_to(ROOT)}"
        else:
            pipe = FACTORIES[name]()
            pipe.fit(X_dev, y_dev)
            source = "fit on 7,381 development rows"
        res = score_pipeline(pipe, X, y)
        res["source"] = source
        results[name] = res
        print(f"scored {name}", flush=True)
    return test, results


def write_outputs(test, results, checks, run_info) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    y = test[TARGET_COLUMN].to_numpy()
    preds = {m: r["pred"] for m, r in results.items()}
    boot = bootstrap(test, preds)

    summary, per_class, confusions, ci_rows = [], [], [], []
    for name in MODEL_ORDER:
        r = results[name]
        m = full_metrics(y, r["pred"])
        dev_file, dev_field = DEV_F1_SOURCE[name]
        dev_f1 = get_field(RESULTS / dev_file, dev_field)
        in_roc = name != "Majority baseline"
        summary.append({
            "model": name, **m["overall"],
            "roc_auc_ovr_macro": r["roc_auc_ovr_macro"] if in_roc else "excluded",
            "roc_auc_score_type": r["score_type"] if in_roc else "excluded",
            "dev_macro_f1_5fold_mean": dev_f1, "test_macro_f1": m["overall"]["macro_f1"],
            "test_minus_dev_macro_f1": m["overall"]["macro_f1"] - dev_f1,
            "dev_source": f"{dev_file}:{dev_field}", "model_source": r["source"],
        })
        for row in m["per_class"]:
            per_class.append({"model": name, **row})
        for i, true_class in enumerate(CLASS_ORDER):
            confusions.append({"model": name, "true_class": true_class,
                               **{f"pred_{c}": int(m["confusion"][i, j]) for j, c in enumerate(CLASS_ORDER)}})
        for metric in ("macro_f1", "accuracy"):
            v = np.asarray(boot[name][metric])
            ci_rows.append({"model": name, "metric": metric, "point_estimate": m["overall"][metric],
                            "ci95_low": float(np.percentile(v, 2.5)), "ci95_high": float(np.percentile(v, 97.5)),
                            "n_resamples": N_BOOT, "seed": SEED, "resampled_unit": GROUP_COLUMN})

    correct = {m: p == y for m, p in preds.items()}
    mcn = []
    for other in ("RBF SVM", "Logistic Regression"):
        mcn.append({"model_a": "Random Forest", "model_b": other,
                    **mcnemar_exact(correct["Random Forest"], correct[other]), "test": "exact binomial, two-sided"})

    pd.DataFrame(summary).to_csv(OUT / "metrics_summary.csv", index=False)
    pd.DataFrame(per_class).to_csv(OUT / "per_class_metrics.csv", index=False)
    pd.DataFrame(confusions).to_csv(OUT / "confusion_matrices.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(OUT / "bootstrap_ci.csv", index=False)
    pd.DataFrame(mcn).to_csv(OUT / "mcnemar.csv", index=False)
    pd.DataFrame(checks["smoke_test"]).to_csv(OUT / "pretest_smoke_test.csv", index=False)
    pred_frame = test[[ID_COLUMN, GROUP_COLUMN, TARGET_COLUMN]].copy()
    for name in MODEL_ORDER:
        pred_frame[name] = preds[name]
    pred_frame.to_csv(OUT / "test_predictions.csv", index=False)
    (OUT / "pretest_checks.json").write_text(json.dumps(checks, indent=2, default=str), encoding="utf-8")
    # The run record is written last. Its presence blocks a second run.
    (OUT / "run_record.json").write_text(json.dumps(run_info, indent=2, default=str), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", action="store_true", help="checks only, no test rows")
    mode.add_argument("--run", action="store_true", help="checks, then score the test set once")
    args = parser.parse_args()

    if args.run and (OUT / "run_record.json").exists():
        print("results/final_test/run_record.json exists. The test is run once. Stopping.")
        return 2

    try:
        checks, dev, columns = run_checks()
    except CheckFailed as error:
        print(f"CHECK FAILED: {error}")
        print("Do not score the test set.")
        return 1

    print(json.dumps({k: v for k, v in checks.items() if k != "smoke_test"}, indent=2, default=str))
    print(f"{'model':<22}{'smoke fold 0':>14}{'saved fold 0':>14}{'diff':>10}  ok")
    for r in checks["smoke_test"]:
        print(f"{r['model']:<22}{r['smoke_fold0_macro_f1']:>14.4f}{r['saved_fold0_macro_f1']:>14.4f}"
              f"{r['difference']:>10.5f}  {r['within_tolerance']}")
    if args.verify:
        print("ALL CHECKS PASSED. No test rows were read.")
        return 0

    started = dt.datetime.now(dt.timezone.utc)
    test, results = score_test(dev, columns)
    run_info = {
        "timestamp_utc": started.isoformat(timespec="seconds"),
        "git": git_info(),
        "dataset_sha256": DATASET_SHA256, "split_sha256": SPLIT_SHA256,
        "random_seed": SEED, "n_features": N_FEATURES,
        "n_train_rows": len(dev), "n_test_rows": len(test),
        "saved_pipeline_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in SAVED_MODELS.values()},
        "library_versions": checks["versions"],
        "python": platform.python_version(), "platform": platform.platform(),
        "command": "python scripts/run_final_test.py --run",
    }
    write_outputs(test, results, checks, run_info)
    print(f"Done. Results are in {OUT.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
