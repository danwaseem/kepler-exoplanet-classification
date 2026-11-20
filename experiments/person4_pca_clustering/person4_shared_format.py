"""Person 4 results in the team's shared format (results/person4/<experiment>/).
PCA + classifier rows use Person 2's tuned RBF-SVM settings so they compare directly.
Dev partition, saved grouped folds, seed 245. Test partition not scored.
Usage: python person4_shared_format.py <repo_root> <out_results_person4_dir>"""
import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics as M

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SEED, CLASSES = 245, ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]
class SignedLog(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=3.0): self.threshold = threshold
    def fit(self, X, y=None):
        s = pd.DataFrame(X).skew().abs(); self.columns_ = list(s[s > self.threshold].index); return self
    def transform(self, X):
        X = pd.DataFrame(X).copy(); X[self.columns_] = np.sign(X[self.columns_]) * np.log1p(X[self.columns_].abs()); return X

split_file = ROOT / "data/processed/proposed_splits.csv"
sha = hashlib.sha256(split_file.read_bytes()).hexdigest()
assert sha == "d0a2892a312ddfb47ae4fe37650563ec566ee254a98c781598485992442b021a"
SPLIT = {"split_sha256": sha, "expected_split_sha256": sha, "n_split_rows": 9564, "n_primary_features": 76,
         "feature_list": "data/processed/proposed_leakage_aware_features.csv", "split_file": "data/processed/proposed_splits.csv",
         "id_column": "kepoi_name", "group_column": "kepid", "target_column": "koi_disposition",
         "class_order": CLASSES, "random_seed": SEED, "test_scored": False}
mat = pd.read_csv(ROOT / "data/processed/proposed_primary_matrix_unimputed.csv")
feats = pd.read_csv(ROOT / "data/processed/proposed_leakage_aware_features.csv")["column"].tolist()
dev = mat[mat.partition.eq("development")].reset_index(drop=True); assert len(dev) == 7381
y, fold = dev.koi_disposition, dev.dev_fold
feats56 = [c for c in feats if not c.startswith(("koi_fwm_", "koi_dicco_", "koi_dikco_"))]; assert len(feats56) == 56
PRE = "signed-log (|skew|>3 on training fold) -> median impute -> standardize, all fit on training fold"

def run(name, model_name, make_model, hyper, cols, n_pca, config, description):
    X = dev[cols]; rows = []; oof = pd.Series(index=X.index, dtype=object); scores = np.zeros((len(X), 3))
    for f in range(5):
        tr, va = fold.ne(f), fold.eq(f); assert not set(dev.kepid[tr]) & set(dev.kepid[va])
        pipe = Pipeline([("log", SignedLog()), ("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("pca", PCA(n_components=n_pca, random_state=SEED)), ("m", make_model())]).fit(X[tr], y[tr])
        r = {"fold": f, "n_train": int(tr.sum()), "n_validation": int(va.sum()),
             "n_train_groups": int(dev.kepid[tr].nunique()), "n_validation_groups": int(dev.kepid[va].nunique())}
        for part, idx in [("train", tr), ("validation", va)]:
            p = pipe.predict(X[idx])
            r[f"{part}_accuracy"] = M.accuracy_score(y[idx], p)
            r[f"{part}_macro_precision"] = M.precision_score(y[idx], p, average="macro")
            r[f"{part}_macro_recall"] = M.recall_score(y[idx], p, average="macro")
            r[f"{part}_macro_f1"] = M.f1_score(y[idx], p, average="macro")
        oof[va] = pipe.predict(X[va])
        assert list(pipe.classes_) == sorted(CLASSES)
        if hasattr(pipe.named_steps["m"], "predict_proba"): s = pipe.predict_proba(X[va])
        else: d = pipe.decision_function(X[va]); s = np.exp(d) / np.exp(d).sum(1, keepdims=True)
        scores[va.to_numpy()] = s; rows.append(r)
    fm = pd.DataFrame(rows); d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    fm.to_csv(d / "fold_metrics.csv", index=False)
    cm = M.confusion_matrix(y, oof, labels=CLASSES)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(d / "oof_confusion_matrix.csv")
    P, R, F, S = M.precision_recall_fscore_support(y, oof, labels=CLASSES)
    pd.DataFrame({"class": CLASSES, "precision": P, "recall": R, "f1": F, "support": S}).to_csv(d / "per_class_metrics.csv", index=False)
    summ = {"confusion_matrix": cm.tolist(), "split": SPLIT, "n_features": len(cols), "n_rows": len(X), "test_scored": False}
    for part in ["train", "validation"]:
        for m in ["accuracy", "macro_f1", "macro_precision", "macro_recall"]:
            summ[f"mean_{part}_{m}"] = float(fm[f"{part}_{m}"].mean()); summ[f"std_{part}_{m}"] = float(fm[f"{part}_{m}"].std(ddof=1))
    summ["mean_macro_f1_gap"] = summ["mean_train_macro_f1"] - summ["mean_validation_macro_f1"]
    summ["oof_roc_auc_ovr_macro"] = float(M.roc_auc_score(y, scores, multi_class="ovr", labels=sorted(CLASSES)))
    summ["n_pca_components"] = n_pca
    (d / "summary.json").write_text(json.dumps(summ, indent=2))
    cfg = {"model": model_name, "standardized": True, "hyperparameters": hyper, "configuration": config,
           "description": description, "n_features": len(cols), "n_pca_components": n_pca, "preprocessing": PRE,
           "imputation": "median, fit on training fold", "selection_metric": "none (fixed settings; PCA size = 90% explained variance)",
           "owner": "person4", "experiment_name": name, "test_scored": False, "split": SPLIT}
    if model_name.endswith("SVM"): cfg["roc_auc_note"] = "SVM ROC-AUC from softmax of one-vs-rest decision scores (no Platt scaling)"
    (d / "model_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"{name:34s} F1={summ['mean_validation_macro_f1']:.4f}±{summ['std_validation_macro_f1']:.4f} acc={summ['mean_validation_accuracy']:.4f} "
          f"AUC={summ['oof_roc_auc_ovr_macro']:.4f} recall={np.round(R,3).tolist()}", flush=True)
    return summ

svmA = {"C": 10.0, "gamma": "scale", "class_weight": "balanced"}; svmB = {"C": 30.0, "gamma": 0.01, "class_weight": "balanced"}
lr = {"max_iter": 3000, "class_weight": "balanced"}
run("pca25_rbf_svm", "PCA + RBF SVM", lambda: SVC(kernel="rbf", random_state=SEED, **svmA), svmA, feats, 25, "A",
    "PCA (25 components, 90% variance) + RBF SVM with person2's tuned configuration-A settings (76 features)")
run("pca17_rbf_svm_no_centroid", "PCA + RBF SVM", lambda: SVC(kernel="rbf", random_state=SEED, **svmB), svmB, feats56, 17, "B",
    "PCA (17 components, 90% variance) + RBF SVM with person2's retuned configuration-B settings (56 features, no centroid offsets)")
run("pca17_rbf_svm_no_centroid_fixed_params", "PCA + RBF SVM", lambda: SVC(kernel="rbf", random_state=SEED, **svmA), svmA, feats56, 17, "B",
    "PCA (17 components) + RBF SVM reusing the 76-feature configuration-A settings (56 features, no centroid offsets)")
run("pca25_logreg", "PCA + Logistic Regression", lambda: LogisticRegression(**lr), lr, feats, 25, "A",
    "PCA (25 components, 90% variance) + multinomial logistic regression (76 features)")
run("pca17_logreg_no_centroid", "PCA + Logistic Regression", lambda: LogisticRegression(**lr), lr, feats56, 17, "B",
    "PCA (17 components, 90% variance) + multinomial logistic regression (56 features, no centroid offsets)")

# Clustering: separate table, not a classifier ranking
d = OUT / "kmeans_clustering"; d.mkdir(parents=True, exist_ok=True); res = {}
for cfgname, cols in [("A_76_features", feats), ("B_56_features_no_centroid", feats56)]:
    Z = Pipeline([("log", SignedLog()), ("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]).fit_transform(dev[cols])
    for k in (3, 5):
        lab = KMeans(k, n_init=20, random_state=SEED).fit_predict(Z); ct = pd.crosstab(lab, y).reindex(columns=CLASSES, fill_value=0)
        res[f"{cfgname}_k{k}"] = {"ARI": M.adjusted_rand_score(y, lab), "NMI": M.normalized_mutual_info_score(y, lab),
            "purity": float(ct.max(axis=1).sum() / len(y)), "majority_baseline": float(y.value_counts(normalize=True).max()),
            "silhouette": float(M.silhouette_score(Z, lab, sample_size=3000, random_state=SEED)),
            "cluster_sizes": ct.sum(1).tolist(), "cluster_by_disposition_counts": ct.to_dict(orient="index")}
(d / "summary.json").write_text(json.dumps({"description": "K-Means on signed-log standardized features, development partition; "
    "unsupervised (labels used only to score agreement). Report separately from classifier ranking.",
    "n_rows": len(dev), "test_scored": False, "split": SPLIT, "results": res}, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else float(o)))
for k, v in res.items(): print(k, {a: round(v[a], 3) for a in ["ARI", "NMI", "purity", "silhouette"]}, v["cluster_sizes"])
