"""Demo: predict the disposition of Kepler Objects of Interest from the test partition.

Uses the saved pruned Decision Tree and, if present, the saved Random Forest.
Nothing is trained or changed. Only test-partition KOIs are allowed, because
the models were fit on the development KOIs.

Examples:
  python demo/predict_koi.py --kepoi K00752.01
  python demo/predict_koi.py --kepoi K00752.01 K00756.01
  python demo/predict_koi.py --sample 5
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "data" / "processed" / "proposed_primary_matrix_unimputed.csv"
SPLITS = ROOT / "data" / "processed" / "proposed_splits.csv"
FINAL = ROOT / "results" / "final_test" / "test_predictions.csv"
TREE = ROOT / "results" / "person3" / "decision_tree" / "stage3" / "selected_development_tree.joblib"
FOREST_SMALL = ROOT / "demo" / "tuned_forest_compressed.joblib"
FOREST_FULL = ROOT / "results" / "person3" / "random_forest" / "tuned_forest.joblib"
SEED = 245
ID, TARGET = "kepoi_name", "koi_disposition"


def fail(message):
    sys.exit(f"Error: {message}")


def load_models():
    if not TREE.exists():
        fail(f"missing {TREE.relative_to(ROOT)}")
    models = {"Pruned Decision Tree": joblib.load(TREE)}
    forest = next((p for p in (FOREST_SMALL, FOREST_FULL) if p.exists()), None)
    if forest is None:
        print("Random Forest file not found. Download from https://drive.google.com/file/d/1Clgn72n5EUQBIArY1r1FPTxTMRtEPA_I/view?usp=sharing into")
        print("results/person3/random_forest/ and run again. Running with the tree only.\n")
    else:
        models["Random Forest"] = joblib.load(forest)
    return models


def choose_kois(args):
    """Return the KOI names to predict. Refuses anything outside the final test set."""
    splits = pd.read_csv(SPLITS, usecols=[ID, "partition"]).set_index(ID)["partition"]
    final_ids = pd.read_csv(FINAL, usecols=[ID])[ID].tolist()
    if args.sample is not None:
        if not 1 <= args.sample <= len(final_ids):
            fail(f"--sample must be between 1 and {len(final_ids)}")
        rng = np.random.default_rng(SEED)
        return [final_ids[i] for i in rng.choice(len(final_ids), args.sample, replace=False)]
    for name in args.kepoi:
        if name not in splits.index:
            fail(f"{name} is not in the split file.")
        if splits[name] == "development":
            fail(f"{name} is a development KOI. The models were trained on it, so a "
                 "prediction would not be a fair test. Choose a test-partition KOI.")
        if name not in final_ids:
            fail(f"{name} is a test KOI without a measured transit depth. It was not scored.")
    return args.kepoi


def load_rows(names):
    """Read only the requested rows from the feature matrix (no other rows are loaded)."""
    ids = pd.read_csv(MATRIX, usecols=[ID])[ID].to_numpy()
    wanted = set(names)
    skip = [i + 1 for i, name in enumerate(ids) if name not in wanted]
    rows = pd.read_csv(MATRIX, skiprows=skip).set_index(ID)
    return rows.loc[names]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--kepoi", nargs="+", metavar="KOI", help="one or more KOI names, e.g. K00752.01")
    group.add_argument("--sample", type=int, metavar="N", help="pick N test KOIs with seed 245")
    args = parser.parse_args()

    names = choose_kois(args)
    models = load_models()
    rows = load_rows(names)
    final = pd.read_csv(FINAL).set_index(ID)
    truth = rows[TARGET]

    results = {}
    for model_name, saved in models.items():
        pipe = saved["pipeline"]
        X = rows[saved["feature_columns"]]
        pred = pipe.predict(X)
        if not (pred == final.loc[names, model_name].to_numpy()).all():
            fail(f"{model_name} predictions differ from results/final_test/test_predictions.csv.")
        results[model_name] = (pred, pipe.predict_proba(X), list(pipe.classes_))

    for i, name in enumerate(names):
        print(f"{name}   true disposition: {truth.iloc[i]}")
        for model_name, (pred, proba, classes) in results.items():
            mark = "correct" if pred[i] == truth.iloc[i] else "wrong"
            probs = "  ".join(f"{c} {p:.2f}" for c, p in zip(classes, proba[i]))
            print(f"  {model_name:<21} {pred[i]:<15} ({mark})   {probs}")
        print()

    for model_name, (pred, _, _) in results.items():
        right = int((pred == truth.to_numpy()).sum())
        print(f"{model_name}: {right} of {len(names)} correct")
    print("Predictions match the final test run.")


if __name__ == "__main__":
    main()
