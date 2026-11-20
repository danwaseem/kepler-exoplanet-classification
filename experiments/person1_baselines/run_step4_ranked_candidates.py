"""Step 4: ranked list of current CANDIDATE objects in the development partition.

The calibrated binary model selected in Step 2 (lowest pooled out-of-fold
Brier score) is fit on every development CONFIRMED and FALSE POSITIVE row,
with its calibrator's inner folds grouped by host star. It then scores the
development CANDIDATE rows, which have no resolved label. The test partition
is not loaded.

The probability is P(CONFIRMED | 76 features) under a model trained on
resolved KOIs. It is a follow-up priority, not a disposition.

Run from the repository root after Steps 1 and 2:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python experiments/person1_baselines/run_step4_ranked_candidates.py
"""

from __future__ import annotations

import json
from functools import partial

import numpy as np
import pandas as pd

from exoplanet.config import GROUP_COLUMN, ID_COLUMN, ROOT, TARGET_COLUMN
from exoplanet.development import load_development_bundle, split_identity_record
from exoplanet.models.person1_baselines.calibration import make_calibrated
from exoplanet.models.person1_baselines.data import setup_frame
from person1_models import make_model, model_config

OUT = ROOT / "results" / "person1"
CONTEXT_COLUMNS = ["koi_period", "koi_prad", "koi_depth", "koi_model_snr", "koi_impact",
                   "koi_dikco_msky"]


def main() -> None:
    choice = json.loads((OUT / "e4_calibration" / "selected_calibrated_model.json").read_text())
    bundle = load_development_bundle()
    F = list(bundle.feature_columns)
    train, y = setup_frame(bundle.frame, "binary")
    candidates = bundle.frame.loc[bundle.frame[TARGET_COLUMN].eq("CANDIDATE")]

    model = make_calibrated(partial(make_model, choice["model"], choice["C"]), choice["method"],
                            train[GROUP_COLUMN].to_numpy(), y.to_numpy())
    model.fit(train[F], y)
    pos = list(model.classes_).index("CONFIRMED")
    p_cal = model.predict_proba(candidates[F])[:, pos]

    raw = make_model(choice["model"], choice["C"]).fit(train[F], y)
    p_raw = raw.predict_proba(candidates[F])[:, list(raw.classes_).index("CONFIRMED")]

    ranked = pd.DataFrame({
        ID_COLUMN: candidates[ID_COLUMN].to_numpy(),
        GROUP_COLUMN: candidates[GROUP_COLUMN].to_numpy(),
        "dev_fold": candidates["dev_fold"].to_numpy(),
        "p_confirmed_calibrated": np.round(p_cal, 6),
        "p_confirmed_uncalibrated": np.round(p_raw, 6),
    })
    for c in CONTEXT_COLUMNS:
        ranked[c] = candidates[c].to_numpy()
    # A linear model extrapolates confidently. Count the features on which a
    # candidate lies outside the range seen in the resolved training rows, so
    # scores driven by out-of-range values can be checked by hand.
    lo, hi = train[F].min(), train[F].max()
    outside = candidates[F].lt(lo) | candidates[F].gt(hi)
    ranked["n_features_outside_training_range"] = outside.sum(axis=1).to_numpy()
    ranked["features_outside_training_range"] = [
        ";".join(outside.columns[row]) for row in outside.to_numpy()]
    # Isotonic calibration is a step function, so many candidates share a
    # calibrated value. Ties are broken by the uncalibrated probability of the
    # same model, then by name.
    ranked = ranked.sort_values(
        ["p_confirmed_calibrated", "p_confirmed_uncalibrated", ID_COLUMN],
        ascending=[False, False, True]).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    ranked["priority_band"] = pd.cut(
        ranked["p_confirmed_calibrated"], [-0.001, 0.1, 0.5, 0.9, 1.0],
        labels=["low (<0.1)", "uncertain (0.1-0.5)", "likely (0.5-0.9)", "high (>=0.9)"])
    ranked.to_csv(OUT / "ranked_candidates_development.csv", index=False)

    bands = ranked["priority_band"].value_counts().reindex(
        ["high (>=0.9)", "likely (0.5-0.9)", "uncertain (0.1-0.5)", "low (<0.1)"])
    summary = {
        "n_candidates_scored": int(len(ranked)),
        "model": choice,
        "fitted_on": f"{len(train)} development CONFIRMED and FALSE POSITIVE rows, 76 features",
        "band_counts": {k: int(v) for k, v in bands.items()},
        "expected_confirmed_if_calibrated": float(ranked["p_confirmed_calibrated"].sum()),
        "mean_p_calibrated": float(ranked["p_confirmed_calibrated"].mean()),
        "mean_p_uncalibrated": float(ranked["p_confirmed_uncalibrated"].mean()),
        "distinct_calibrated_values": int(ranked["p_confirmed_calibrated"].nunique()),
        "candidates_with_any_feature_outside_training_range": int(
            ranked["n_features_outside_training_range"].gt(0).sum()),
        "high_band_with_any_feature_outside_training_range": int(
            (ranked["n_features_outside_training_range"].gt(0)
             & ranked["priority_band"].eq("high (>=0.9)")).sum()),
        "tie_break": "uncalibrated probability of the same model, then kepoi_name",
        "spearman_calibrated_vs_uncalibrated": float(
            ranked[["p_confirmed_calibrated", "p_confirmed_uncalibrated"]]
            .corr(method="spearman").iloc[0, 1]),
        "note": "Scores are a follow-up priority under a model trained on resolved KOIs. "
                "They are not dispositions. Calibration was measured on resolved KOIs; "
                "CANDIDATE objects may differ from them, so the probabilities are a guide.",
        "split": split_identity_record(),
        "model_config": model_config(choice["model"], choice["C"]),
        "test_scored": False,
    }
    (OUT / "ranked_candidates_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("split", "model_config")},
                     indent=1, default=str))
    print(ranked.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
