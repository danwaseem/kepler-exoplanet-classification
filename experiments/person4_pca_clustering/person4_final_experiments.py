"""Person 4 (Nathan Pak) final-round experiments: PCA + clustering.

Development partition only (7,381 KOIs, 5 saved grouped folds, seed 245).
The held-out test partition is never loaded for scoring.

Experiments
  E0  reproduce the Nov-3 Logistic Regression probe (sanity check)
  E1  original vs PCA-reduced inputs across four classifier families
  E2  56-feature set without centroid-offset columns (PCA + probe + K-Means)
  E3  clustering sensitivity: K-Means k=5, GMM (BIC), Ward hierarchical
  E4  clustering validity indices (silhouette, Davies-Bouldin, Calinski-Harabasz)
  E5  leakage sensitivity: add vetting flags back and re-cluster
Usage: python person4_final_experiments.py <repo_root> <out_dir>
"""
import sys, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics as M

warnings.filterwarnings("ignore")
ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
SEED = 245
CLASSES = ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]

class SignedLog(BaseEstimator, TransformerMixin):
    """sign(x)*log1p(|x|) on columns whose |skew| > threshold on the fit rows."""
    def __init__(self, threshold=3.0): self.threshold = threshold
    def fit(self, X, y=None):
        X = pd.DataFrame(X); s = X.skew().abs()
        self.columns_ = list(s[s > self.threshold].index); return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X[self.columns_] = np.sign(X[self.columns_]) * np.log1p(X[self.columns_].abs())
        return X

def prep(log=True):
    steps = ([("signed_log", SignedLog())] if log else []) + [
        ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    return Pipeline(steps)

# ---------- data ----------
mat = pd.read_csv(ROOT / "data/processed/proposed_primary_matrix_unimputed.csv")
feats = pd.read_csv(ROOT / "data/processed/proposed_leakage_aware_features.csv")["column"].tolist()
dev = mat[mat.partition.eq("development")].reset_index(drop=True)
assert not (dev.partition == "test").any()
X, y, fold = dev[feats], dev.koi_disposition, dev.dev_fold
print("dev rows", len(dev), "hosts", dev.kepid.nunique(), "features", len(feats))
print(y.value_counts().to_dict())
centroid = [c for c in feats if c.startswith(("koi_fwm_", "koi_dicco_", "koi_dikco_"))]
feats56 = [c for c in feats if c not in centroid]
print("centroid cols", len(centroid), "-> reduced set", len(feats56))

def cv(model_factory, Xin, n_pca=None, log=True):
    rows, oof_pred = [], pd.Series(index=Xin.index, dtype=object)
    oof_proba = np.zeros((len(Xin), 3))
    for f in range(5):
        tr, va = fold.ne(f), fold.eq(f)
        assert not set(dev.kepid[tr]) & set(dev.kepid[va])
        steps = list(prep(log).steps)
        if n_pca: steps.append(("pca", PCA(n_components=n_pca, random_state=SEED)))
        steps.append(("model", model_factory()))
        pipe = Pipeline(steps)
        t0 = time.time(); pipe.fit(Xin[tr], y[tr]); fit_s = time.time() - t0
        p_va, p_tr = pipe.predict(Xin[va]), pipe.predict(Xin[tr])
        SORTED = sorted(CLASSES); assert list(pipe.classes_) == SORTED
        if hasattr(pipe.named_steps["model"], "predict_proba") and not isinstance(pipe.named_steps["model"], SVC):
            proba = pipe.predict_proba(Xin[va]); ll = M.log_loss(y[va], proba, labels=SORTED)
        else:  # SVM: softmax of one-vs-rest decision scores, ranking metric only
            d = pipe.decision_function(Xin[va]); proba = np.exp(d) / np.exp(d).sum(1, keepdims=True); ll = np.nan
        oof_pred[va] = p_va
        rows.append(dict(fold=f, fit_seconds=fit_s,
            macro_f1=M.f1_score(y[va], p_va, average="macro"),
            train_macro_f1=M.f1_score(y[tr], p_tr, average="macro"),
            accuracy=M.accuracy_score(y[va], p_va),
            balanced_accuracy=M.balanced_accuracy_score(y[va], p_va),
            weighted_f1=M.f1_score(y[va], p_va, average="weighted"),
            mcc=M.matthews_corrcoef(y[va], p_va),
            kappa=M.cohen_kappa_score(y[va], p_va),
            roc_auc_ovr=M.roc_auc_score(y[va], proba, multi_class="ovr", labels=SORTED),
            log_loss=ll))
    r = pd.DataFrame(rows)
    out = r.drop(columns="fold").mean().to_dict()
    out["macro_f1_sd"] = r.macro_f1.std(ddof=1)
    rec = M.recall_score(y, oof_pred, labels=CLASSES, average=None)
    prc = M.precision_score(y, oof_pred, labels=CLASSES, average=None)
    for c, a, b in zip(CLASSES, rec, prc): out[f"recall_{c}"] = a; out[f"precision_{c}"] = b
    return out, oof_pred

MODELS = {
    "LogReg": lambda: LogisticRegression(max_iter=3000, class_weight="balanced"),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=25, weights="distance"),
    "RBF-SVM": lambda: SVC(C=3.0, gamma="scale", class_weight="balanced", random_state=SEED),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                        class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
}

# ---------- E0 + E1 ----------
rows, confusions = [], {}
for mname, mf in MODELS.items():
    for k in [None, 5, 10, 15, 25, 33]:
        t0 = time.time(); res, oof = cv(mf, X, k)
        rows.append(dict(model=mname, inputs="original 76" if k is None else f"PCA {k}",
                         n_inputs=76 if k is None else k, **res))
        if k in (None, 25):
            confusions[f"{mname}|{'orig' if k is None else 'pca25'}"] = M.confusion_matrix(y, oof, labels=CLASSES).tolist()
        print(f"{mname:12s} {rows[-1]['inputs']:12s} F1={res['macro_f1']:.4f}±{res['macro_f1_sd']:.4f} "
              f"AUC={res['roc_auc_ovr']:.4f} ({time.time()-t0:.0f}s)", flush=True)
e1 = pd.DataFrame(rows); e1.to_csv(OUT / "e1_multimodel_pca_vs_original.csv", index=False)
(OUT / "e1_oof_confusions.json").write_text(json.dumps(confusions, indent=1))

# ---------- E2: 56 features without centroid offsets ----------
z56 = prep().fit_transform(X[feats56]); p56 = PCA(random_state=SEED).fit(z56)
cum56 = np.cumsum(p56.explained_variance_ratio_)
e2 = dict(n_features=len(feats56), PC1=p56.explained_variance_ratio_[0], PC2=p56.explained_variance_ratio_[1],
          k80=int(np.argmax(cum56 >= .8) + 1), k90=int(np.argmax(cum56 >= .9) + 1), k95=int(np.argmax(cum56 >= .95) + 1))
load56 = pd.DataFrame(p56.components_[:3].T, index=feats56, columns=["PC1", "PC2", "PC3"])
load56.to_csv(OUT / "e2_loadings56.csv")
e2["top_PC1"] = load56.PC1.abs().sort_values(ascending=False).index[:6].tolist()
e2["top_PC2"] = load56.PC2.abs().sort_values(ascending=False).index[:6].tolist()
e2["top_PC3"] = load56.PC3.abs().sort_values(ascending=False).index[:6].tolist()
for name, k in [("orig56", None), ("pca_k90", e2["k90"])]:
    r, _ = cv(MODELS["LogReg"], X[feats56], k); e2[f"logreg_{name}_macro_f1"] = r["macro_f1"]; e2[f"logreg_{name}_sd"] = r["macro_f1_sd"]
r, _ = cv(MODELS["RandomForest"], X[feats56]); e2["rf_orig56_macro_f1"] = r["macro_f1"]
km = KMeans(3, n_init=20, random_state=SEED).fit(z56)
e2["kmeans3_ARI"] = M.adjusted_rand_score(y, km.labels_); e2["kmeans3_NMI"] = M.normalized_mutual_info_score(y, km.labels_)
e2["kmeans3_sizes"] = np.bincount(km.labels_).tolist()
(OUT / "e2_no_centroid.json").write_text(json.dumps(e2, indent=1, default=float)); print("E2", e2)

# ---------- E3/E4: clustering sensitivity on 76-feature signed-log space ----------
Z = prep().fit_transform(X)
def score_labels(lab, Zs=Z):
    ct = pd.crosstab(lab, y)
    return dict(ARI=M.adjusted_rand_score(y, lab), NMI=M.normalized_mutual_info_score(y, lab),
                homogeneity=M.homogeneity_score(y, lab), completeness=M.completeness_score(y, lab),
                v_measure=M.v_measure_score(y, lab), purity=ct.max(axis=1).sum() / len(y),
                silhouette=M.silhouette_score(Zs, lab, sample_size=3000, random_state=SEED),
                davies_bouldin=M.davies_bouldin_score(Zs, lab), calinski_harabasz=M.calinski_harabasz_score(Zs, lab),
                sizes=np.bincount(pd.factorize(lab)[0]).tolist())
crow = []
Z25 = PCA(25, random_state=SEED).fit_transform(Z)
for k in range(2, 11):
    km = KMeans(k, n_init=10, random_state=SEED).fit(Z)
    crow.append(dict(method="KMeans", space="76 std", k=k, **score_labels(km.labels_)))
    km25 = KMeans(k, n_init=10, random_state=SEED).fit(Z25)
    crow.append(dict(method="KMeans", space="PCA 25", k=k, **score_labels(km25.labels_, Z25)))
    g = GaussianMixture(k, covariance_type="diag", n_init=3, random_state=SEED).fit(Z25)
    crow.append(dict(method="GMM-diag", space="PCA 25", k=k, bic=g.bic(Z25), **score_labels(g.predict(Z25), Z25)))
for k in (3, 5):
    w = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(Z25)
    crow.append(dict(method="Ward", space="PCA 25", k=k, **score_labels(w.labels_, Z25)))
e3 = pd.DataFrame(crow); e3.to_csv(OUT / "e3_clustering_sensitivity.csv", index=False)
print(e3[["method", "space", "k", "ARI", "NMI", "silhouette", "davies_bouldin"]].round(3).to_string())

km5 = KMeans(5, n_init=20, random_state=SEED).fit(Z)
lab5 = pd.Series(km5.labels_)
ct5 = pd.crosstab(lab5, y).reindex(columns=CLASSES)
prof = dev[["koi_period", "koi_prad", "koi_model_snr", "koi_teq", "koi_depth", "koi_dikco_msky", "koi_count"]].groupby(lab5).median()
k5 = pd.concat([ct5, ct5.div(ct5.sum(1), axis=0).add_suffix("_pct"), prof], axis=1); k5.insert(0, "n", ct5.sum(1))
k5.to_csv(OUT / "e3_kmeans_k5_profile.csv"); print(k5.round(2).to_string())

# ---------- E5: leakage sensitivity ----------
raw = pd.read_csv(ROOT / "data/raw/koi_cumulative_2026-10-06.csv", comment="#",
                  usecols=["kepoi_name", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec", "koi_score"])
leak = dev[["kepoi_name"]].merge(raw, on="kepoi_name", how="left")
assert len(leak) == len(dev)
e5 = {}
for name, cols in [("flags", ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]),
                   ("flags+score", ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec", "koi_score"])]:
    XL = pd.concat([X, leak[cols].set_index(X.index)], axis=1)
    ZL = prep().fit_transform(XL)
    for k in (3, 5):
        km = KMeans(k, n_init=20, random_state=SEED).fit(ZL)
        e5[f"{name}_k{k}_ARI"] = M.adjusted_rand_score(y, km.labels_)
        e5[f"{name}_k{k}_NMI"] = M.normalized_mutual_info_score(y, km.labels_)
    # flags alone, unweighted
    ZF = prep(log=False).fit_transform(leak[cols])
    km = KMeans(3, n_init=20, random_state=SEED).fit(ZF)
    e5[f"{name}_only_k3_ARI"] = M.adjusted_rand_score(y, km.labels_)
    r, _ = cv(MODELS["LogReg"], XL); e5[f"logreg_76+{name}_macro_f1"] = r["macro_f1"]
(OUT / "e5_leakage_sensitivity.json").write_text(json.dumps(e5, indent=1)); print("E5", e5)
print("done")
